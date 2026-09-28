"""Sequential GPTQ for the NV-Reason-CT decoder in torch, scored with fake quantisation, optionally exported.

  venv/bin/python tools/quant.py calib CALIB_DIR CT[:region] ...      # build calibration sequences
  venv/bin/python tools/quant.py run CALIB_DIR REF_DIR,... [--int8 REGEX] [--bs 32] [--act] [--rtn]
                                     [--export TEMPLATE_DIR OUT_DIR] [--gen]

Calibration sequences are the model's own greedy replies (structured report and reasoning) on each CT,
stored as input ids, MRoPE positions and vision features. Quantisation goes layer by layer: each block's
linears see inputs produced by the already-quantised blocks before it (attention first, then the MLP).
Matrices matching --int8 get plain RTN int8. The embeddings are the dequantised int4 lm_head rows,
as in the browser.
"""
import json, os, re, shutil, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = '/data/huggingface/nv-reason-ct/hf'; IMAGE = 248056
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
mode, cdir = sys.argv[1], Path(sys.argv[2])
model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32, attn_implementation='sdpa').eval().cuda()
lm = model.model.language_model

if mode == 'calib':
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    cdir.mkdir(parents=True, exist_ok=True); k = 0
    for c in sys.argv[3:]:
        ct, region = (c.split(':') + ['abdomen'])[:2]
        for text, thinking, n in [(f'Write a structured {region} CT report.', False, 1024),
                                  (f'Provide a full reasoning analysis of this {region} CT.', True, 1536)]:
            msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
            prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            inp = proc(text=prompt, images3d=[ct], anatomy_region=region, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                ids = model.generate(**inp, max_new_tokens=n, do_sample=False, use_cache=True)
                m = model.model; P = inp['input_ids'].shape[1]
                vis = torch.cat(m.get_image_features(inp['pixel_values'], inp.get('image_grid_thw')).pooler_output)
                tt = torch.cat([inp['mm_token_type_ids'], torch.zeros((1, ids.shape[1] - P), dtype=inp['mm_token_type_ids'].dtype, device='cuda')], 1)
                pos, _ = m.get_rope_index(ids, tt, inp.get('image_grid_thw'), None, torch.ones_like(ids))
            torch.save({'ids': ids.cpu(), 'pos': pos.cpu(), 'vis': vis.cpu(), 'prompt_len': P, 'case': c, 'thinking': thinking}, cdir / f'{k:02d}.pt')
            print(f'{c} {"reason" if thinking else "report"}: {ids.shape[1] - P} new tokens', flush=True); k += 1
    sys.exit()

# ---- quantisers
BS = int(arg('--bs', 32)); ACT = '--act' in sys.argv; TEXT_WEIGHT = float(arg('--text-weight', 4))
int8_rx = re.compile(arg('--int8', r'^$'))

def params(grp, bits=4):
    top = 2 ** bits - 1
    lo, hi = grp.min(1).values.clamp(max=0), grp.max(1).values.clamp(min=0)
    best, bs, bz = None, None, None
    for shrink in ((1.0,) if bits == 8 else (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)):
        s = ((hi - lo) * shrink / top).clamp(min=1e-8).half().float(); z = torch.round(-lo * shrink / s).clamp(0, top)
        err = ((torch.clamp(torch.round(grp / s[:, None]) + z[:, None], 0, top) - z[:, None]) * s[:, None] - grp).pow(2).sum(1)
        if best is None: best, bs, bz = err, s, z
        else: m = err < best; best = torch.where(m, err, best); bs = torch.where(m, s, bs); bz = torch.where(m, z, bz)
    return bs, bz

def rtn(W, bits):
    N, K = W.shape; g = W.view(N, K // BS, BS); S = torch.zeros(N, K // BS, device='cuda'); Z = torch.zeros_like(S)
    for b in range(K // BS): S[:, b], Z[:, b] = params(g[:, b], bits)
    Q = torch.clamp(torch.round(g / S[..., None]) + Z[..., None], 0, 2 ** bits - 1)
    return Q.view(N, K).to(torch.uint8), S, Z

def gptq(W, H):
    W = W.float().clone(); N, K = W.shape
    H = H.clone(); dead = H.diag() == 0; H[dead, dead] = 1; W[:, dead] = 0
    S = torch.zeros(N, K // BS, device='cuda'); Z = torch.zeros_like(S)
    if ACT:
        for g in range(K // BS): S[:, g], Z[:, g] = params(W[:, g * BS:(g + 1) * BS])
        perm = torch.argsort(H.diag(), descending=True)
    else: perm = torch.arange(K, device='cuda')
    inv = torch.argsort(perm); W = W[:, perm]; H = H[perm][:, perm]; gidx = perm // BS
    H += 0.01 * H.diag().mean() * torch.eye(K, device='cuda')
    Hi = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros(N, K, dtype=torch.uint8, device='cuda')
    for i1 in range(0, K, 128):
        i2 = min(i1 + 128, K); W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi1 = Hi[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if not ACT and col % BS == 0: S[:, col // BS], Z[:, col // BS] = params(W1[:, i:i + BS])
            s, z = S[:, gidx[col]], Z[:, gidx[col]]
            w = W1[:, i]; q = torch.clamp(torch.round(w / s) + z, 0, 15); Q[:, col] = q.to(torch.uint8)
            e = (w - (q - z) * s) / Hi1[i, i]
            W1[:, i:] -= e[:, None] * Hi1[i, i:][None]; E1[:, i] = e
        W[:, i2:] -= E1 @ Hi[i1:i2, i2:]
    return Q[:, inv], S, Z

def dequant(Q, S, Z):
    N, K = Q.shape
    return ((Q.view(N, K // BS, BS).float() - Z[..., None]) * S[..., None]).view(N, K)

# ---- calibration data
seqs = [torch.load(f, weights_only=False) for f in sorted(cdir.glob('*.pt'))]
print(f'{len(seqs)} calibration sequences, {sum(s["ids"].shape[1] for s in seqs)} tokens')
orig_embed = lm.embed_tokens.weight.data.clone()

def embeds(s):
    ids = s['ids'].cuda(); e = lm.embed_tokens(ids); e[ids == IMAGE] = s['vis'].cuda(); return e

def row_weights(s):
    ids = s['ids'][0].cuda(); w = torch.full((ids.numel(),), TEXT_WEIGHT, device='cuda')
    first = s['case'] not in seen_case; w[ids == IMAGE] = 1.0 if first else 0.0  # vision rows once per case
    return w

class Stop(Exception): pass

layers = lm.layers
def groups(i):
    L = layers[i]; att = L.self_attn if hasattr(L, 'self_attn') and L.self_attn is not None else L.linear_attn
    a = {n: m for n, m in att.named_children() if isinstance(m, torch.nn.Linear) and n not in ('in_proj_a', 'in_proj_b')}
    pa = 'self_attn' if att is getattr(L, 'self_attn', None) else 'linear_attn'
    return [{f'model.language_model.layers.{i}.{pa}.{n}': m for n, m in a.items()},
            {f'model.language_model.layers.{i}.mlp.{n}': m for n, m in L.mlp.named_children() if isinstance(m, torch.nn.Linear)}]

results = {}
def collect(mods, stop_at):
    H = {n: torch.zeros(m.in_features, m.in_features, device='cuda') for n, m in mods.items()}
    hs = []
    for n, m in mods.items():
        def f(mod, inp, n=n):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float(); H[n].addmm_((x * cur_w[:, None]).T, x)
        hs.append(m.register_forward_pre_hook(f))
    hs.append(stop_at.register_forward_hook(lambda *a: (_ for _ in ()).throw(Stop())))
    global cur_w, seen_case
    seen_case, tot = set(), 0.0
    for s in seqs:
        cur_w = row_weights(s); seen_case.add(s['case']); tot += float(cur_w.sum())
        with torch.inference_mode():
            try: lm(inputs_embeds=embeds(s), position_ids=s['pos'].cuda())
            except Stop: pass
    for h in hs: h.remove()
    return {n: h / tot for n, h in H.items()}

t0 = time.time()
fq = {}
def quantise(n, m, H):
    W = m.weight.data
    if int8_rx.search(n): Q, S, Z = rtn(W, 8); bits = 8
    elif '--rtn' in sys.argv: Q, S, Z = rtn(W, 4); bits = 4
    else: Q, S, Z = gptq(W, H); bits = 4
    fq[n] = (Q.cpu(), S.cpu(), Z.cpu(), bits)
    m.weight.data.copy_(dequant(Q, S, Z))

# lm_head first, with its hessian from the final norm output of the unquantised model, so the embeddings
# are the quantised rows from the start and every later hessian sees them
hl = torch.zeros(2560, 2560, device='cuda'); tot = 0.0; seen_case = set()
for s in seqs:
    w = row_weights(s); seen_case.add(s['case']); tot += float(w.sum())
    with torch.inference_mode():
        x = lm(inputs_embeds=embeds(s), position_ids=s['pos'].cuda()).last_hidden_state[0]
    hl.addmm_((x * w[:, None]).T, x)
model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())  # untie
quantise('lm_head', model.lm_head, hl / tot); del hl
lm.embed_tokens.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
print(f'lm_head done {time.time() - t0:.0f}s', flush=True)
for i in range(len(layers)):
    for g in groups(i):
        Hs = collect(g, layers[i]) if '--rtn' not in sys.argv else {n: None for n in g}
        for n, m in g.items(): quantise(n, m, Hs[n])
        del Hs
    print(f'layer {i} done {time.time() - t0:.0f}s', flush=True)

# ---- score
def lsm(z): return torch.log_softmax(z.float(), -1)
tk = None
for ref in sys.argv[3].split(','):
    ids = torch.from_numpy(np.load(f'{ref}/input_ids.npy')).cuda(); pos = torch.from_numpy(np.load(f'{ref}/position_ids.npy')).cuda()
    vis = torch.from_numpy(np.load(f'{ref}/vision.npy')).cuda()
    gen = json.load(open(f'{ref}/ref.json'))['runs'][0]['tokens']
    full = torch.cat([ids, torch.tensor([gen[:-1]], device='cuda')], 1); p0 = int(pos.max()) + 1
    fpos = torch.cat([pos, (torch.arange(len(gen) - 1, device='cuda') + p0).view(1, 1, -1).expand(3, 1, -1)], 2)
    with torch.inference_mode():
        e = lm.embed_tokens(full); e[full == IMAGE] = vis
        lg = model.lm_head(lm(inputs_embeds=e, position_ids=fpos).last_hidden_state[0, ids.shape[1] - 1:])
    f32 = torch.from_numpy(np.load(f'{ref}/tf_fp32.npy')).cuda().float()
    lp, lq = lsm(f32), lsm(lg); kl = (lp.exp() * (lp - lq)).sum(-1)
    print(f'{ref}: top-1 {100 * (lg.argmax(-1) == f32.argmax(-1)).float().mean():.1f}%, KL mean {kl.mean():.4f} max {kl.max():.3f}', flush=True)
    if '--gen' in sys.argv:
        from tokenizers import Tokenizer
        tk = tk or Tokenizer.from_file(f'{MODEL}/tokenizer.json')
        out, past, x, p = [], None, e[:, :ids.shape[1]], pos
        with torch.inference_mode():
            for k in range(1024):
                o = lm(inputs_embeds=x, position_ids=p, past_key_values=past, use_cache=True)
                past = o.past_key_values; t = int(model.lm_head(o.last_hidden_state[0, -1]).argmax())
                if t in (248046, 248044): break
                out.append(t); x = lm.embed_tokens(torch.tensor([[t]], device='cuda')); p = torch.full((3, 1, 1), p0 + k, device='cuda')
        same = next((i for i, (a, b) in enumerate(zip(out, gen)) if a != b), min(len(out), len(gen)))
        print(f'--- greedy {len(out)} tokens, same as bf16 reference for {same}\n{tk.decode(out)}\n', flush=True)

# ---- export into an asymmetric int4 builder graph (block size must match); int8 matrices need an int8 node
if '--export' in sys.argv:
    import onnx
    from onnx import numpy_helper as nh, helper
    tpl, out = Path(arg('--export')), Path(sys.argv[sys.argv.index('--export') + 2])
    mo = onnx.load(tpl / 'model.onnx'); ini = {i.name: i for i in mo.graph.initializer}
    nodes = {n.name: n for n in mo.graph.node if n.op_type == 'MatMulNBits'}
    def onnx_node(hf):
        if hf == 'lm_head': return '/lm_head/MatMul_Q4'
        i, blk, lin = re.match(r'model\.language_model\.layers\.(\d+)\.(\w+)\.(\w+)$', hf).groups()
        blk = {'self_attn': 'attn'}.get(blk, blk); lin = {'in_proj_qkv': 'qkv_proj', 'in_proj_z': 'z_proj'}.get(lin, lin)
        return f'/model/layers.{i}/{blk}/{lin}/MatMul_Q4'
    for hf, (Q, S, Z, bits) in fq.items():
        node = nodes[onnx_node(hf)]; a = {x.name: x for x in node.attribute}
        assert a['block_size'].i == BS, 'template block size differs'
        N, K = Q.shape; nb = K // BS; q = Q.view(N, nb, BS).numpy(); z = Z.to(torch.uint8).numpy()
        dt = nh.to_array(ini[node.input[2]]).dtype
        if bits == 4:
            packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
            if nb % 2: z = np.concatenate([z, np.zeros((N, 1), np.uint8)], 1)
            zp = (z[:, 0::2] | (z[:, 1::2] << 4)).astype(np.uint8)
        else: packed, zp = q.astype(np.uint8), z
        base = node.input[1][:-3]  # strip _Q4
        names = [base + f'_Q{bits}', node.input[2], node.input[3]]
        for old in node.input[1:]: mo.graph.initializer.remove(ini.pop(old))
        for nm, arr in zip(names, [packed, S.numpy().astype(dt), zp]):
            t = nh.from_array(arr, nm); mo.graph.initializer.append(t); ini[nm] = t
        node.input[1] = names[0]; a['bits'].i = bits
        if bits == 8: node.name = node.name[:-3] + '_Q8'
    out.mkdir(parents=True, exist_ok=True)
    for f in tpl.iterdir():
        if not f.name.startswith('model.onnx'): shutil.copy(f, out / f.name)
    onnx.save(mo, out / 'model.onnx', save_as_external_data=True, location='model.onnx.data', size_threshold=1024)
    print('exported', out, round((out / 'model.onnx.data').stat().st_size / 1e9, 2), 'GB')
