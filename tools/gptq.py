"""GPTQ int4 quantisation of the NV-Reason-CT decoder, written into an asymmetric int4 builder graph.

  venv/bin/python tools/gptq.py TEMPLATE_DIR OUT_DIR CT[:region] ...

TEMPLATE_DIR is an onnxruntime-genai build with -p int4 is_symmetric=false (block 32); its MatMulNBits
weights, scales and zero points are replaced. Hessians come from the bf16 model on CUDA, run on each
calibration CT with a structured-report and a reasoning prompt (the model's own greedy replies).
"""
import sys, time, re, shutil
from pathlib import Path
import numpy as np, torch, onnx
from onnx import numpy_helper
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = '/data/huggingface/nv-reason-ct/hf'
tpl, out, cases = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
import os
TEXT_WEIGHT, ACT = float(os.environ.get('TEXT_WEIGHT', 4)), os.environ.get('ACT') == '1'
G = next(a.i for n in onnx.load(tpl / 'model.onnx', load_external_data=False).graph.node if n.op_type == 'MatMulNBits' for a in n.attribute if a.name == 'block_size')
model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('cuda')
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

# HF module name -> ONNX initializer prefix
def onnx_name(hf):
    if hf == 'lm_head': return 'lm_head.MatMul.weight'
    m = re.match(r'model\.language_model\.layers\.(\d+)\.(\w+)\.(\w+)$', hf)
    i, blk, lin = m.groups()
    blk = {'self_attn': 'attn'}.get(blk, blk); lin = {'in_proj_qkv': 'qkv_proj', 'in_proj_z': 'z_proj'}.get(lin, lin)
    return f'model.layers.{i}.{blk}.{lin}.MatMul.weight'

targets = {n: mod for n, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)
           and (n == 'lm_head' or (n.startswith('model.language_model.layers.') and 'in_proj_a' not in n and 'in_proj_b' not in n))}
H = {n: torch.zeros(m.in_features, m.in_features, device='cuda') for n, m in targets.items()}
rows = {'w': None}
def hook(name):
    def f(mod, inp):
        x = inp[0].reshape(-1, inp[0].shape[-1]).float(); w = rows['w']
        H[name].addmm_((x * w[:, None]).T, x)
    return f
# the forward keeps only the last logits, so lm_head's hessian comes from the final norm output
norm = model.model.language_model.norm
def add_hooks():
    hs = [m.register_forward_pre_hook(hook(n)) for n, m in targets.items() if n != 'lm_head']
    return hs + [norm.register_forward_hook(lambda mod, inp, o: hook('lm_head')(mod, (o,)))]
hooks = add_hooks()

total = 0.0
for c in cases:
    ct, region = (c.split(':') + ['abdomen'])[:2]
    for k, (text, thinking, n) in enumerate([(f'Write a structured {region} CT report.', False, 1024),
                                             (f'Provide a full reasoning analysis of this {region} CT.', True, 1536)]):
        msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        inp = proc(text=prompt, images3d=[ct], anatomy_region=region, return_tensors='pt').to('cuda')
        for h in hooks: h.remove()
        with torch.inference_mode():
            ids = model.generate(**inp, max_new_tokens=n, do_sample=False, use_cache=True)
        hooks = add_hooks()
        L, P = ids.shape[1], inp['input_ids'].shape[1]
        img = (ids[0] == 248056)
        w = torch.full((L,), TEXT_WEIGHT, device='cuda'); w[img] = 1.0 if k == 0 else 0.0  # vision rows once per case
        rows['w'] = w; total += float(w.sum())
        full = dict(inp); full['input_ids'] = ids; full['attention_mask'] = torch.ones_like(ids)
        full['mm_token_type_ids'] = torch.cat([inp['mm_token_type_ids'], torch.zeros((1, L - P), dtype=inp['mm_token_type_ids'].dtype, device='cuda')], 1)
        with torch.inference_mode():
            model(**full, logits_to_keep=1)
        print(f'{ct} {region} {"reason" if thinking else "report"}: {L - P} new tokens', flush=True)
for h in hooks: h.remove()
print('hessians done; weight total', total)

def params(grp):
    lo, hi = grp.min(1).values.clamp(max=0), grp.max(1).values.clamp(min=0)
    best, bs, bz = None, None, None
    for shrink in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
        s = ((hi - lo) * shrink / 15).clamp(min=1e-8).half().float(); z = torch.round(-lo * shrink / s).clamp(0, 15)
        err = ((torch.clamp(torch.round(grp / s[:, None]) + z[:, None], 0, 15) - z[:, None]) * s[:, None] - grp).pow(2).sum(1)
        if best is None: best, bs, bz = err, s, z
        else: m = err < best; best = torch.where(m, err, best); bs = torch.where(m, s, bs); bz = torch.where(m, z, bz)
    return bs, bz

def gptq_act(W, Hm):
    # act-order with group parameters fixed up front from the original weights, so groups stay contiguous
    W = W.float().clone(); N, K = W.shape
    Hm = Hm.clone(); dead = Hm.diag() == 0; Hm[dead, dead] = 1; W[:, dead] = 0
    S = torch.zeros(N, K // G, device='cuda'); Z = torch.zeros(N, K // G, device='cuda')
    for g in range(K // G): S[:, g], Z[:, g] = params(W[:, g * G:(g + 1) * G])
    perm = torch.argsort(Hm.diag(), descending=True); inv = torch.argsort(perm)
    W = W[:, perm]; Hm = Hm[perm][:, perm]; gidx = perm // G
    Hm += 0.01 * Hm.diag().mean() * torch.eye(K, device='cuda')
    Hi = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hm)), upper=True)
    Q = torch.zeros(N, K, dtype=torch.uint8, device='cuda')
    for i1 in range(0, K, 128):
        i2 = min(i1 + 128, K); W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi1 = Hi[i1:i2, i1:i2]
        for i in range(i2 - i1):
            s, z = S[:, gidx[i1 + i]], Z[:, gidx[i1 + i]]
            w = W1[:, i]; q = torch.clamp(torch.round(w / s) + z, 0, 15); Q[:, i1 + i] = q.to(torch.uint8)
            e = (w - (q - z) * s) / Hi1[i, i]
            W1[:, i:] -= e[:, None] * Hi1[i, i:][None]; E1[:, i] = e
        W[:, i2:] -= E1 @ Hi[i1:i2, i2:]
    return Q[:, inv], S, Z

def gptq(W, Hm):
    W = W.float().clone(); N, K = W.shape
    Hm = Hm.clone(); dead = Hm.diag() == 0; Hm[dead, dead] = 1; W[:, dead] = 0
    Hm += 0.01 * Hm.diag().mean() * torch.eye(K, device='cuda')
    Hi = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hm)), upper=True)
    Q = torch.zeros(N, K, dtype=torch.uint8, device='cuda'); S = torch.zeros(N, K // G, device='cuda'); Z = torch.zeros(N, K // G, device='cuda')
    for i1 in range(0, K, 128):
        i2 = min(i1 + 128, K); W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi1 = Hi[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % G == 0:
                s, z = params(W1[:, i:i + G]); S[:, col // G] = s; Z[:, col // G] = z
            w = W1[:, i]; d = Hi1[i, i]
            q = torch.clamp(torch.round(w / s) + z, 0, 15); Q[:, col] = q.to(torch.uint8)
            e = (w - (q - z) * s) / d
            W1[:, i:] -= e[:, None] * Hi1[i, i:][None]; E1[:, i] = e
        W[:, i2:] -= E1 @ Hi[i1:i2, i2:]
    return Q, S, Z

m = onnx.load(tpl / 'model.onnx')
ini = {i.name: i for i in m.graph.initializer}
t0 = time.time()
for n, mod in targets.items():
    Q, S, Z = (gptq_act if ACT else gptq)(mod.weight.data, H[n] / total)
    N, K = Q.shape; nb = K // G
    q = Q.view(N, nb, G).cpu().numpy(); packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    z = Z.to(torch.uint8).cpu().numpy()
    if nb % 2: z = np.concatenate([z, np.zeros((N, 1), np.uint8)], 1)
    zp = (z[:, 0::2] | (z[:, 1::2] << 4)).astype(np.uint8)
    p = onnx_name(n)
    for suffix, arr in [('_Q4', packed), ('_scales', S.cpu().numpy().astype(np.float16)), ('_zero_points', zp)]:
        t = ini[p + suffix]; assert tuple(t.dims) == arr.shape, (p, suffix, t.dims, arr.shape)
        t.CopyFrom(numpy_helper.from_array(arr.reshape(t.dims).astype(numpy_helper.to_array(t).dtype), t.name))
    del H[n]
    print(f'{n} done {time.time() - t0:.0f}s', flush=True)
out.mkdir(parents=True, exist_ok=True)
for f in tpl.iterdir():
    if not f.name.startswith('model.onnx'): shutil.copy(f, out / f.name)
onnx.save(m, out / 'model.onnx', save_as_external_data=True, location='model.onnx.data', size_threshold=1024)
print('saved', out)
