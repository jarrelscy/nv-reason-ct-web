"""Shrink the exported vision ONNX: dedupe repeated constant tables and quantise MatMul weights to int8.

  venv/bin/python tools/shrink_vision.py IN.onnx OUT.onnx REF_DIR [bits]

torch.onnx.export writes the rotary sin/cos table as a separate Constant in every block; identical constants
become one shared initializer. MatMul/Gemm weights then go to MatMulNBits (block 32) with fp32 activations.
"""
import hashlib, sys, time
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import matmul_nbits_quantizer as mq

src, dst, ref = sys.argv[1], sys.argv[2], sys.argv[3]; bits = int(sys.argv[4]) if len(sys.argv) > 4 else 8
m = onnx.load(src)
g = m.graph
seen, keep, rename = {}, [], {}
for n in g.node:
    if n.op_type == 'Constant' and n.attribute[0].name == 'value' and np.prod(n.attribute[0].t.dims) > 4096:
        t = n.attribute[0].t; h = hashlib.sha1(numpy_helper.to_array(t).tobytes()).hexdigest()
        if h not in seen:
            seen[h] = f'const_{len(seen)}'
            arr = numpy_helper.to_array(t); g.initializer.append(numpy_helper.from_array(arr, seen[h]))
        rename[n.output[0]] = seen[h]
        continue
    keep.append(n)
for n in keep:
    for k, x in enumerate(n.input):
        if x in rename: n.input[k] = rename[x]
del g.node[:]; g.node.extend(keep)
print(f'{len(rename)} large constants -> {len(seen)} initializers')
q = mq.MatMulNBitsQuantizer(m, algo_config=mq.DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=bits, accuracy_level=1,
                                                                          op_types_to_quantize=('MatMul', 'Gemm')))
q.process()
onnx.save(q.model.model, dst, save_as_external_data=True, location=dst.split('/')[-1] + '.data', size_threshold=1024)
x = np.load(f'{ref}/crop.npy').astype(np.float32); want = np.load(f'{ref}/vision_fp32.npy')
s = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
t0 = time.time(); got = s.run(None, {'volume': x})[0]; print(f'ORT CPU {time.time() - t0:.1f}s')
cos = (got * want).sum(1) / np.linalg.norm(got, axis=1) / np.linalg.norm(want, axis=1)
print(f'vs torch fp32: max|diff| {np.abs(got - want).max():.4f}, row cosine min {cos.min():.5f} mean {cos.mean():.6f}')
np.save(dst + '.npy', got)
