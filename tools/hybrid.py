"""Build a mixed int4/int8 decoder by swapping MatMulNBits nodes from an int8 build into an int4 build.

  venv/bin/python tools/hybrid.py INT4_DIR INT8_DIR OUT_DIR REGEX
REGEX is matched against node names such as /model/layers.3/attn/q_proj/MatMul_Q4 and /lm_head/MatMul_Q4.
"""
import re, shutil, sys
from pathlib import Path
import onnx

a, b, out, rx = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), re.compile(sys.argv[4])
ma, mb = onnx.load(a / 'model.onnx'), onnx.load(b / 'model.onnx')
base = lambda n: re.sub(r'_Q\d$', '', n)
nb = {base(n.name): n for n in mb.graph.node if n.op_type == 'MatMulNBits'}
ib = {i.name: i for i in mb.graph.initializer}
ia = {i.name: i for i in ma.graph.initializer}
swapped = 0
for n in ma.graph.node:
    if n.op_type != 'MatMulNBits' or not rx.search(n.name): continue
    m = nb[base(n.name)]
    for x in n.input[1:]:
        if x in ia: ma.graph.initializer.remove(ia.pop(x))
    for x in m.input[1:]:
        if x and x not in ia: ma.graph.initializer.append(ib[x]); ia[x] = ib[x]
    n.CopyFrom(m); swapped += 1
print(f'swapped {swapped} nodes')
out.mkdir(parents=True, exist_ok=True)
for f in a.iterdir():
    if not f.name.startswith('model.onnx'): shutil.copy(f, out / f.name)
onnx.save(ma, out / 'model.onnx', save_as_external_data=True, location='model.onnx.data', size_threshold=1024)
print('size', round((out / 'model.onnx.data').stat().st_size / 1e9, 2), 'GB')
