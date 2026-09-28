"""Copy quantised MatMulNBits weights from one decoder build into another with the same block size,
keeping the target's IO dtype and node attributes (for example GPTQ weights from a webgpu fp16 build
into the fp32 browser template). Scales are cast to the target's dtype.

  venv/bin/python tools/transplant.py SRC_DIR TEMPLATE_DIR OUT_DIR
"""
import shutil, sys
from pathlib import Path
import numpy as np, onnx
from onnx import numpy_helper as nh

src, tpl, out = map(Path, sys.argv[1:4])
ms, mt = onnx.load(src / 'model.onnx'), onnx.load(tpl / 'model.onnx')
ns = {n.name: n for n in ms.graph.node if n.op_type == 'MatMulNBits'}
isrc = {i.name: i for i in ms.graph.initializer}
done = 0
it = {t.name: k for k, t in enumerate(mt.graph.initializer)}
for n in mt.graph.node:
    if n.op_type != 'MatMulNBits': continue
    s = ns[n.name]
    attr = lambda m: {a.name: a.i for a in m.attribute if a.name in ('bits', 'block_size', 'K', 'N')}
    assert attr(s) == attr(n), (n.name, attr(s), attr(n))
    for a, b in zip(n.input[1:], s.input[1:]):
        if not a: continue
        old = mt.graph.initializer[it[a]]
        x = nh.to_array(isrc[b])
        if old.data_type != isrc[b].data_type: x = x.astype(onnx.helper.tensor_dtype_to_np_dtype(old.data_type))
        assert list(x.shape) == list(old.dims), (a, x.shape, old.dims)
        mt.graph.initializer[it[a]].CopyFrom(nh.from_array(x, a))
    assert len([x for x in n.input[1:] if x]) == len([x for x in s.input[1:] if x]), n.name
    done += 1
print(f'copied {done} matrices')
out.mkdir(parents=True, exist_ok=True)
for f in tpl.iterdir():
    if not f.name.startswith('model.onnx'): shutil.copy(f, out / f.name)
onnx.save(mt, out / 'model.onnx', save_as_external_data=True, location='model.onnx.data', size_threshold=1024)
print('size', round((out / 'model.onnx.data').stat().st_size / 1e9, 2), 'GB')
