"""Export the NV-Reason-CT 3D vision tower (Primus ViT + merger) to ONNX and check it against the reference.

  /data/huggingface/nv-reason-ct/venv/bin/python tools/export_vision.py OUT.onnx REF_DIR [chunk]

Attention runs over 13,824 tokens, so the full score matrix is 9 GB in fp32. The export splits the queries
into chunks of `chunk` tokens (default 1152, 12 chunks) so each score matrix stays under 1 GB in the browser.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from safetensors import safe_open

HF = Path('/data/huggingface/nv-reason-ct/hf')
out, ref = Path(sys.argv[1]), Path(sys.argv[2]); chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 1152
sys.path.insert(0, str(HF))
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from model import Vision3D

cfg = json.loads((HF / 'config.json').read_text())
vc = Qwen3_5VisionConfig(**{k: v for k, v in cfg['vision_config'].items() if k != 'model_type'})
m = Vision3D(vc).eval()
pre = 'model.vision3d.'
with safe_open(HF / 'model.safetensors', 'pt') as f:
    sd = {k[len(pre):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(pre)}
missing, unexpected = m.load_state_dict(sd, strict=False)
print(f'{len(sd)} tensors; missing {missing}; unexpected {unexpected}')
print(f'{sum(p.numel() for p in m.parameters()) / 1e6:.1f}M parameters')

def chunked_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, **kw):
    scale = q.shape[-1] ** -0.5
    kt = k.transpose(-2, -1)
    return torch.cat([torch.softmax((qc * scale) @ kt, dim=-1) @ v for qc in q.split(chunk, dim=2)], dim=2)

x = torch.from_numpy(np.load(ref / 'crop.npy')).float()
print('input', tuple(x.shape))
want = np.load(ref / 'vision.npy')
def cmp(name, got):
    got = np.asarray(got, np.float32)
    cos = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
    print(f'{name}: max|diff| {np.abs(got - want).max():.4f}, mean|ref| {np.abs(want).mean():.4f}, '
          f'row cosine min {cos.min():.5f} mean {cos.mean():.6f}')

with torch.inference_mode():
    mc = m.to('cuda'); t0 = time.time()
    y = mc(x.to('cuda')).float().cpu().numpy(); print(f'torch fp32 cuda {time.time() - t0:.1f}s'); cmp('torch fp32 vs bf16 ref', y)
    np.save(ref / 'vision_fp32.npy', y)
m = m.to('cpu')
F.scaled_dot_product_attention = chunked_sdpa
for blk in m.sub_vision.eva.blocks:
    blk.attn.fused_attn = True
with torch.inference_mode():
    torch.onnx.export(m, (x,), out, input_names=['volume'], output_names=['tokens'], opset_version=18,
                      dynamo=False, external_data=True)
print('exported', out)

import onnxruntime as ort
s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
t0 = time.time(); got = s.run(None, {'volume': x.numpy()})[0]; print(f'ORT CPU fp32 {time.time() - t0:.1f}s')
cmp('onnx fp32 vs bf16 ref', got)
print('onnx vs torch fp32 max|diff|', np.abs(got - y).max())
