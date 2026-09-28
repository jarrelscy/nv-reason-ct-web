"""Package an ONNX model for the browser: external data split into chunks under 2 GB plus a manifest.

  venv/bin/python tools/package.py IN_DIR/model.onnx OUT_DIR NAME [chunk_bytes]

Writes OUT_DIR/NAME.onnx, NAME.data0, NAME.data1, ... and NAME.json with the chunk sizes and, for the
decoder, where the lm_head int4 weights, scales and zero points sit, so JS can look up token embeddings
(the embedding matrix is tied to lm_head) straight from the cached files.
"""
import json, sys
from pathlib import Path
import numpy as np, onnx

src, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
limit = int(float(sys.argv[4])) if len(sys.argv) > 4 else int(1.9e9)
m = onnx.load(src)
out.mkdir(parents=True, exist_ok=True)
files, cur, size, where = [], None, 0, {}
def open_chunk():
    global cur, size
    if cur: cur.close()
    files.append(f'{name}.data{len(files)}'); cur = open(out / files[-1], 'wb'); size = 0
open_chunk()
for t in m.graph.initializer:
    raw = onnx.numpy_helper.to_array(t).tobytes() if not t.HasField('raw_data') else t.raw_data
    if len(raw) < 1024: continue
    if size + len(raw) > limit: open_chunk()
    pad = (-size) % 64; cur.write(b'\0' * pad); size += pad
    off = size; cur.write(raw); size += len(raw)
    t.ClearField('raw_data'); del t.float_data[:]; del t.int32_data[:]; del t.int64_data[:]; del t.external_data[:]
    for k, v in (('location', files[-1]), ('offset', off), ('length', len(raw))):
        e = t.external_data.add(); e.key = k; e.value = str(v)
    t.data_location = onnx.TensorProto.EXTERNAL
    where[t.name] = {'file': files[-1], 'offset': off, 'length': len(raw), 'dims': list(t.dims), 'type': t.data_type}
cur.close()
onnx.save(m, out / f'{name}.onnx')
man = {'onnx': f'{name}.onnx', 'data': [{'name': f, 'bytes': (out / f).stat().st_size} for f in files]}
state = [i.name for i in m.graph.input if i.name.startswith('past')]
if state: man['state'] = state
lm = {k: v for k, v in where.items() if k.startswith('lm_head.MatMul.weight')}
if lm:
    node = next(n for n in m.graph.node if n.op_type == 'MatMulNBits' and n.name.startswith('/lm_head'))
    a = {x.name: x.i for x in node.attribute}
    man['embed'] = {'bits': a['bits'], 'block_size': a['block_size'], 'K': a['K'], 'N': a['N'],
                    'q': lm.get(node.input[1]), 'scales': lm.get(node.input[2]), 'zero_points': lm.get(node.input[3]) if len(node.input) > 3 else None}
(out / f'{name}.json').write_text(json.dumps(man, indent=1))
print(json.dumps({k: v for k, v in man.items() if k != 'embed'}), 'embed' in man)
