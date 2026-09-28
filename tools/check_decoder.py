"""Run the ONNX decoder in onnxruntime (Python) on the PyTorch reference inputs and compare.

  venv/bin/python tools/check_decoder.py DECODER_DIR REF_DIR [n_tokens] [EP] [--tf] [--qemb] [--vis FEATURES.npy]

--tf scores teacher-forced logits against fp32 torch; --qemb takes embeddings from the int4 lm_head as the
browser does; --vis replaces the torch vision features (e.g. with the int8 vision ONNX output).
"""
import json, sys, time
import numpy as np, onnxruntime as ort
from safetensors import safe_open

dec, ref = sys.argv[1], sys.argv[2]; n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 40
IMAGE = 248056
emb = safe_open('/data/huggingface/nv-reason-ct/hf/model.safetensors', 'pt').get_tensor('model.language_model.embed_tokens.weight').float().numpy()
if '--qemb' in sys.argv:
    # embeddings as the browser gets them: dequantised rows of the int4 lm_head
    import onnx
    from onnx import numpy_helper as nh
    mo = onnx.load(f'{dec}/model.onnx'); ini = {i.name: i for i in mo.graph.initializer}
    node = next(n for n in mo.graph.node if n.op_type == 'MatMulNBits' and n.name.startswith('/lm_head'))
    q = nh.to_array(ini[node.input[1]]); sc = nh.to_array(ini[node.input[2]]).astype(np.float32); V, nb, _ = q.shape
    zp = (np.stack([(z := nh.to_array(ini[node.input[3]])) & 15, z >> 4], -1).reshape(V, -1)[:, :nb] if len(node.input) > 3 else np.full((V, nb), 8)).astype(np.float32)
    emb = ((np.stack([q & 15, q >> 4], -1).reshape(V, nb, -1).astype(np.float32) - zp[..., None]) * sc[..., None]).reshape(V, -1)
    del mo, ini, q
    print('using int4 lm_head rows as embeddings')
ids = np.load(f'{ref}/input_ids.npy')[0]; pos = np.load(f'{ref}/position_ids.npy'); vis = np.load(sys.argv[sys.argv.index('--vis') + 1] if '--vis' in sys.argv else f'{ref}/vision.npy')
x = emb[ids].copy(); x[ids == IMAGE] = vis
so = ort.SessionOptions(); so.log_severity_level = 3
t0 = time.time(); s = ort.InferenceSession(f'{dec}/model.onnx', so, providers=[sys.argv[4] if len(sys.argv) > 4 else 'CPUExecutionProvider'])
print(f'session {time.time() - t0:.1f}s', s.get_providers())
DT = np.float32 if next(i for i in s.get_inputs() if i.name == 'inputs_embeds').type == 'tensor(float)' else np.float16

def zeros(i):
    shp = [{'batch_size': 1, 'past_sequence_length': 0, 'kv_cache_dim': 256}.get(d, d) if isinstance(d, str) else d for d in i.shape]
    return np.zeros(shp, DT)
state = {i.name: zeros(i) for i in s.get_inputs() if i.name.startswith('past')}
out_names = [o.name for o in s.get_outputs()]

def step(e, p, total):
    feeds = {'inputs_embeds': e[None].astype(DT), 'position_ids': p, 'attention_mask': np.ones((1, total), np.int64), **state}
    outs = dict(zip(out_names, s.run(None, feeds)))
    for k in list(state): state[k] = outs[k.replace('past_key_values', 'present').replace('past.', 'present.')]
    return outs['logits'][0, -1].astype(np.float32)

t0 = time.time(); l0 = step(x, pos, len(ids)); print(f'prefill {len(ids)} tokens {time.time() - t0:.1f}s')
r = np.load(f'{ref}/logits0.npy')
cos = float(l0 @ r / np.linalg.norm(l0) / np.linalg.norm(r))
print('first-step logits: cos', round(cos, 5), 'top5 onnx', np.argsort(-l0)[:5].tolist(), 'torch', np.argsort(-r)[:5].tolist())
want = json.load(open(f'{ref}/ref.json'))['runs'][0]['tokens']
if '--tf' in sys.argv:
    # teacher forcing: feed the torch tokens in one chunk and score every position against fp32 torch
    n, p0 = len(want), int(pos.max()) + 1
    t0 = time.time()
    if s.get_outputs()[0].shape[1] == 1:  # pruned lm_head: one token at a time
        lg = [l0] + [step(emb[[t]], np.full((3, 1, 1), p0 + k, np.int64), len(ids) + k + 1) for k, t in enumerate(want[:-1])]
        lg = np.stack(lg)
    else:
      feeds = {'inputs_embeds': emb[want[:-1]][None].astype(DT), 'attention_mask': np.ones((1, len(ids) + n - 1), np.int64),
             'position_ids': (np.arange(n - 1) + p0).reshape(1, 1, -1).repeat(3, 0), **state}
      lg = np.concatenate([l0[None], dict(zip(out_names, s.run(None, feeds)))['logits'][0].astype(np.float32)])
    print(f'teacher-forced chunk {n - 1} tokens {time.time() - t0:.1f}s')
    def lsm(z): z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
    f32 = np.load(f'{ref}/tf_fp32.npy').astype(np.float32); b16 = np.load(f'{ref}/tf_bf16.npy').astype(np.float32)
    lp = lsm(f32)
    for name, z in [('torch bf16', b16), ('onnx', lg)]:
        q = lsm(z); kl = (np.exp(lp) * (lp - q)).sum(-1)
        print(f'{name:10s} vs fp32: top-1 agreement {100 * (z.argmax(-1) == f32.argmax(-1)).mean():.1f}%, KL mean {kl.mean():.4f} max {kl.max():.3f}')
    sys.exit()
got, tok, p = [], int(np.argmax(l0)), int(pos.max()) + 1
t0 = time.time()
for k in range(n_new):
    got.append(tok)
    if tok in (248046, 248044): break
    l = step(emb[[tok]], np.full((3, 1, 1), p, np.int64), len(ids) + k + 1); p += 1; tok = int(np.argmax(l))
dt = time.time() - t0
same = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
print(f'decoded {len(got)} tokens in {dt:.1f}s; matches torch greedy for the first {same}')
from tokenizers import Tokenizer
tk = Tokenizer.from_file('/data/huggingface/nv-reason-ct/hf/tokenizer.json')
print('onnx :', repr(tk.decode(got))); print('torch:', repr(tk.decode(want[:len(got)])))
