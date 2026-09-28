// Compares lib/preprocess.js with the PyTorch/MONAI crop saved by reference.py.
//   node tools/prep-test.mjs CT.nii.gz|dicom.zip REF_DIR/crop.npy [chest|abdomen]
import fs from 'node:fs';
import { niftiVolume } from '../lib/nifti.js';
import { openInput } from '../lib/source.js';
import { pickSeries, seriesVolume } from '../lib/series.js';
import { preprocess } from '../lib/preprocess.js';

const [path, npy, region = 'abdomen'] = process.argv.slice(2);
const file = new File([fs.readFileSync(path)], path.split('/').pop());
let vol;
if (/\.nii(\.gz)?$/i.test(path)) vol = await niftiVolume(file, console.log);
else { const inp = await openInput([file], console.log); vol = await seriesVolume((await pickSeries(inp.entries, console.log)).best.files, console.log); }
const { img, crop } = await preprocess(vol, region, console.log);
const b = fs.readFileSync(npy), hl = b.readUInt16LE(8), ref = new Float32Array(b.buffer.slice(b.byteOffset + 10 + hl, b.byteOffset + b.length));
let m = 0, s = 0, big = 0;
for (let i = 0; i < ref.length; i++) { const e = Math.abs(img[i] - ref[i]); if (e > m) m = e; s += e; if (e > 1e-3) big++; }
console.log(`crop ${JSON.stringify(crop.start)} padded ${crop.P}; max|diff| ${m.toExponential(2)} (${(m * 1000).toFixed(2)} HU), mean ${(s / ref.length).toExponential(2)}, voxels > 1 HU: ${big}`);
