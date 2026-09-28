"""Run the HF model on CUDA with the decoder weights dequantised from an ONNX build, to judge quantisation quickly.

  venv/bin/python tools/fakequant.py ONNX_DIR|fp32 REF_DIR [REF_DIR ...] [--gen]

For each REF_DIR (from tools/reference.py + reference_tf.py) prints teacher-forced top-1/KL against the fp32
logits and, with --gen, the greedy structured report. Activations run in fp32 as in the browser, and the
embeddings come from the int4 lm_head rows as in lib/nvreason.js.
"""
import json, re, sys
import numpy as np, torch, onnx
from onnx import numpy_helper as nh
from transformers import AutoModelForImageTextToText

MODEL = '/data/huggingface/nv-reason-ct/hf'; IMAGE = 248056
src, refs = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith('--')]
model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32, attn_implementation='sdpa').eval().cuda()

def deq(ini, node):
    a = {x.name: x.i for x in node.attribute}; N, K, bits, bs = a['N'], a['K'], a['bits'], a['block_size']
    q = nh.to_array(ini[node.input[1]]).reshape(N, -1); s = nh.to_array(ini[node.input[2]]).astype(np.float32).reshape(N, -1); nb = s.shape[1]
    if bits == 4: v = np.stack([q & 15, q >> 4], -1).reshape(N, nb, bs)
    else: v = q.reshape(N, nb, bs)
    if len(node.input) > 3 and node.input[3]:
        z = nh.to_array(ini[node.input[3]]).reshape(N, -1)
        z = (np.stack([z & 15, z >> 4], -1).reshape(N, -1) if bits == 4 else z)[:, :nb]
    else: z = np.full((N, nb), 8 if bits == 4 else 128, np.float32)
    return ((v.astype(np.float32) - z[..., None]) * s[..., None]).reshape(N, -1)[:, :K]

if src != 'fp32':
    mo = onnx.load(f'{src}/model.onnx'); ini = {i.name: i for i in mo.graph.initializer}
    mods = dict(model.named_modules()); n = 0
    for node in mo.graph.node:
        if node.op_type != 'MatMulNBits': continue
        if node.name.startswith('/lm_head'):
            w = torch.from_numpy(deq(ini, node)).cuda()
            model.lm_head.weight = torch.nn.Parameter(w)  # unties; embeddings below use the same rows
            model.model.language_model.embed_tokens.weight = torch.nn.Parameter(w.clone()); n += 1; continue
        i, blk, lin = re.match(r'/model/layers\.(\d+)/(\w+)/(\w+)/MatMul', node.name).groups()
        blk = {'attn': 'self_attn'}.get(blk, blk); lin = {'qkv_proj': 'in_proj_qkv', 'z_proj': 'in_proj_z'}.get(lin, lin)
        mods[f'model.language_model.layers.{i}.{blk}.{lin}'].weight.data.copy_(torch.from_numpy(deq(ini, node))); n += 1
    print(f'loaded {n} quantised matrices from {src}'); del mo, ini

def lsm(z): return torch.log_softmax(z.float(), -1)
for ref in refs:
    ids = torch.from_numpy(np.load(f'{ref}/input_ids.npy')).cuda(); pos = torch.from_numpy(np.load(f'{ref}/position_ids.npy')).cuda()
    vis = torch.from_numpy(np.load(f'{ref}/vision.npy')).cuda()
    run = json.load(open(f'{ref}/ref.json'))['runs'][0]; gen = run['tokens']
    full = torch.cat([ids, torch.tensor([gen[:-1]], device='cuda')], 1); p0 = int(pos.max()) + 1
    fpos = torch.cat([pos, (torch.arange(len(gen) - 1, device='cuda') + p0).view(1, 1, -1).expand(3, 1, -1)], 2)
    lm = model.model.language_model
    with torch.inference_mode():
        e = lm.embed_tokens(full); e[full == IMAGE] = vis
        h = lm(inputs_embeds=e, position_ids=fpos).last_hidden_state[0, ids.shape[1] - 1:]
        lg = model.lm_head(h)
    f32 = torch.from_numpy(np.load(f'{ref}/tf_fp32.npy')).cuda().float()
    lp, lq = lsm(f32), lsm(lg); kl = (lp.exp() * (lp - lq)).sum(-1)
    print(f'{ref}: top-1 {100 * (lg.argmax(-1) == f32.argmax(-1)).float().mean():.1f}%, KL mean {kl.mean():.4f} max {kl.max():.3f}', flush=True)
    if '--gen' in sys.argv:
        out, past, x, p = [], None, e[:, :ids.shape[1]], pos
        with torch.inference_mode():
            for k in range(1024):
                o = lm(inputs_embeds=x, position_ids=p, past_key_values=past, use_cache=True)
                past = o.past_key_values; t = int(model.lm_head(o.last_hidden_state[0, -1]).argmax())
                if t in (248046, 248044): break
                out.append(t); x = lm.embed_tokens(torch.tensor([[t]], device='cuda')); p = torch.full((3, 1, 1), p0 + k, device='cuda')
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(f'{MODEL}/tokenizer.json')
        same = next((i for i, (a, b) in enumerate(zip(out, gen)) if a != b), min(len(out), len(gen)))
        print(f'--- greedy {len(out)} tokens, same as bf16 reference for {same}\n{tk.decode(out)}\n', flush=True)
