// Volume -> NV-Reason-CT input, matching ImageLoader3D in the model's processor.py (MONAI):
//   Orientation "LPS", Spacing 2 mm (bilinear; an axis already within 1.8-2.2 mm keeps its spacing),
//   SpatialPad to 192 with -1000 (symmetric), AnatomySpatialCropd (lung-based z crop for chest or
//   abdomen, centred x/y), Transpose to [z][y][x], clip to [-1000, 1000] / 1000.
// Output is Float32 192^3 with x fastest: x towards L, y towards P, z towards S.
// Resizing is separable and commutes with flips, so the slice axis is resampled first and only the
// source slices that contribute are decoded.

export const ROI = 192, PIX = 2;
const WANT = [1, 1, 1];   // LPS

// Spacing keeps world coordinates of voxel 0, so output voxel t samples oriented index t * out / in.
function axisWeights(n, r, T) {
  const i0 = new Int32Array(T), i1 = new Int32Array(T), w1 = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let src = Math.min(Math.max(t * r, 0), n - 1);
    const a = Math.min(Math.floor(src), n - 1);
    i0[t] = a; i1[t] = a < n - 1 ? a + 1 : a; w1[t] = src - a;
  }
  return { i0, i1, w1 };
}

export function orientation(vol) {
  // perm[a] = source axis that becomes LPS axis a; flip[a] if it runs the other way
  const perm = [-1, -1, -1];
  const order = [0, 1, 2].sort((p, q) => Math.max(...vol.dir[q].map(Math.abs)) - Math.max(...vol.dir[p].map(Math.abs)));
  for (const s of order) {
    let best = -1;
    for (let a = 0; a < 3; a++) if (perm[a] < 0 && (best < 0 || Math.abs(vol.dir[s][a]) > Math.abs(vol.dir[s][best]))) best = a;
    perm[best] = s;
  }
  const flip = perm.map((s, a) => Math.sign(vol.dir[s][a] || 1) !== WANT[a]);
  const n = perm.map(s => vol.dims[s]), sp = perm.map(s => vol.spacing[s]);
  return { perm, flip, n, sp };
}

// Python's round(): halves go to the even neighbour
const pyRound = x => { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 ? f + 1 : f); };

export async function preprocess(vol, region, log = () => {}, progress = () => {}) {
  const t0 = performance.now();
  const { perm, flip, n, sp } = orientation(vol);
  const out = sp.map(s => (s >= PIX * 0.9 - 1e-3 && s <= PIX * 1.1 + 1e-3) ? s : PIX);
  const r = out.map((o, a) => o / sp[a]);
  const T = n.map((m, a) => Math.max(1, pyRound((m - 1) / r[a] + 1)));
  log(`orient LPS: axes ${perm.join(',')} flips ${flip.map(Number).join(',')}; ${n.join('x')} @ ${sp.map(s => s.toFixed(3)).join(', ')} mm -> ${T.join('x')} @ ${out.map(s => s.toFixed(2)).join(', ')} mm`);

  // padded grid; the resampled volume sits at `before`
  const P = T.map(t => Math.max(t, ROI)), before = T.map((t, a) => (P[a] - t) >> 1);
  const PX = P[0], PY = P[1], PZ = P[2], vox = new Float32Array(PX * PY * PZ).fill(-1000);
  const stride = [1, PX, PX * PY], base = before[0] + PX * (before[1] + PY * before[2]);

  const inv = [0, 1, 2].map(s => perm.indexOf(s));
  const W = [0, 1, 2].map(a => axisWeights(n[a], r[a], T[a]));
  const src = (a, x) => flip[a] ? n[a] - 1 - x : x;   // LPS index -> source index along perm[a]
  const aS = inv[2], a0 = inv[0], a1 = inv[1], n0 = vol.dims[0], n1 = vol.dims[1];

  const need = [];
  for (let t = 0; t < T[aS]; t++) need.push([src(aS, W[aS].i0[t]), src(aS, W[aS].i1[t])]);
  const uniq = [...new Set(need.flat())];
  log(`decoding ${uniq.length} of ${vol.dims[2]} source slices`);
  const cache = new Map(); let next = 0, done = 0;
  const get = k => { if (!cache.has(k)) cache.set(k, vol.slice(k).then(v => { progress(++done / uniq.length); return v; })); return cache.get(k); };
  const prefetch = () => { while (next < uniq.length && cache.size < 12) get(uniq[next++]); };

  const wx = W[a0], wy = W[a1], Tx = T[a0], Ty = T[a1];
  const sx0 = new Int32Array(Tx), sx1 = new Int32Array(Tx), sy0 = new Int32Array(Ty), sy1 = new Int32Array(Ty);
  for (let t = 0; t < Tx; t++) { sx0[t] = src(a0, wx.i0[t]); sx1[t] = src(a0, wx.i1[t]); }
  for (let t = 0; t < Ty; t++) { sy0[t] = src(a1, wy.i0[t]); sy1[t] = src(a1, wy.i1[t]); }
  const plane = new Float32Array(n0 * n1), rowsX = new Float32Array(Tx * n1);
  const s0 = stride[a0], s1 = stride[a1], sS = stride[aS];
  for (let t = 0; t < T[aS]; t++) {
    prefetch();
    const [k0, k1] = need[t], w = W[aS].w1[t];
    const A = await get(k0), B = await get(k1);
    for (let i = 0; i < plane.length; i++) plane[i] = A[i] + (B[i] - A[i]) * w;
    if (!need.slice(t + 1).some(p => p[0] === k0 || p[1] === k0)) cache.delete(k0);
    if (!need.slice(t + 1).some(p => p[0] === k1 || p[1] === k1)) cache.delete(k1);
    for (let j = 0; j < n1; j++) for (let x = 0; x < Tx; x++) {
      const p = plane[sx0[x] + n0 * j];
      rowsX[x + Tx * j] = p + (plane[sx1[x] + n0 * j] - p) * wx.w1[x];
    }
    const o = base + t * sS;
    for (let y = 0; y < Ty; y++) {
      const r0 = Tx * sy0[y], r1 = Tx * sy1[y], w = wy.w1[y], oy = o + y * s1;
      for (let x = 0; x < Tx; x++) { const p = rowsX[r0 + x]; vox[oy + x * s0] = p + (rowsX[r1 + x] - p) * w; }
    }
  }
  const lung = lungBounds(vox, P);
  const start = [0, 1].map(a => axisStart(P[a], (P[a] - 1) / 2));
  let bounds = null;
  if (lung && lung.zMin != null) {
    if (region === 'chest') bounds = [lung.zMin, lung.zMax];
    else {
      const zMax = Math.min(PZ - 1, lung.zMin + Math.ceil(100 / PIX));
      const zMin = Math.max(lung.bodyZMin, zMax - Math.ceil(300 / PIX) + 1);
      if (zMin <= zMax) bounds = [zMin, zMax];
    }
  }
  if (bounds) start.push(axisStart(PZ, (bounds[0] + bounds[1]) / 2));
  else { start.push(PZ - ROI); log(`could not find the lungs to place the ${region} crop; using the top of the scan`); }
  log(`lungs at z ${lung?.zMin ?? '-'}..${lung?.zMax ?? '-'} of ${PZ}; ${region} crop starts at ${start.join(',')}`);

  const img = new Float32Array(ROI ** 3);
  for (let z = 0; z < ROI; z++) for (let y = 0; y < ROI; y++) {
    const i = PX * (start[1] + y + PY * (start[2] + z)) + start[0], o = ROI * (y + ROI * z);
    for (let x = 0; x < ROI; x++) {
      const v = vox[i + x];
      img[o + x] = v !== v ? -1 : Math.min(Math.max(v, -1000), 1000) / 1000;
    }
  }
  log(`preprocessing ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return { img, crop: { start, P, spacing: out, region, lung } };
}

const axisStart = (size, center) => Math.max(0, Math.min(pyRound(center - ROI / 2), size - ROI));

// AnatomySpatialCropd._detect_lung_bounds: air enclosed by the body on each axial slice, grouped in 3D,
// then the most superior large group plus nearby pieces.
// Like upstream, distances use the nominal 2 mm spacing even when an axis kept its original spacing.
function lungBounds(vox, [X, Y, Z], zSpacing = PIX) {
  const plane = X * Y, N = plane * Z, body = new Uint8Array(N);
  let bodyZMin = -1;
  for (let z = 0; z < Z; z++) for (let i = 0; i < plane; i++) if (vox[i + plane * z] > -500) { body[i + plane * z] = 1; if (bodyZMin < 0) bodyZMin = z; }
  if (bodyZMin < 0) return null;
  // binary_fill_holes per slice: background reachable from the edge (4-connected) stays outside
  const air = new Uint8Array(N), seen = new Uint8Array(plane), q = new Int32Array(plane);
  for (let z = 0; z < Z; z++) {
    const b = body.subarray(plane * z, plane * (z + 1));
    seen.fill(0); let h = 0, t = 0;
    const push = i => { if (!b[i] && !seen[i]) { seen[i] = 1; q[t++] = i; } };
    for (let x = 0; x < X; x++) { push(x); push(x + X * (Y - 1)); }
    for (let y = 0; y < Y; y++) { push(X * y); push(X - 1 + X * y); }
    while (h < t) {
      const i = q[h++], x = i % X, y = (i - x) / X;
      if (x > 0) push(i - 1); if (x < X - 1) push(i + 1);
      if (y > 0) push(i - X); if (y < Y - 1) push(i + X);
    }
    for (let i = 0; i < plane; i++) if (!b[i] && !seen[i]) air[i + plane * z] = 1;
  }
  // 3D labelling (6-connected), numbered in scipy's scan order (x slowest, z fastest in MONAI's layout)
  const lab = new Int32Array(N), Q = new Int32Array(N), comps = [{}];
  for (let x = 0; x < X; x++) for (let y = 0; y < Y; y++) for (let z = 0; z < Z; z++) {
    const s = x + X * (y + Y * z);
    if (!air[s] || lab[s]) continue;
    const id = comps.length; let h = 0, t = 0, size = 0, zmin = Z, zmax = -1;
    lab[s] = id; Q[t++] = s;
    while (h < t) {
      const i = Q[h++], xi = i % X, r = (i - xi) / X, yi = r % Y, zi = (r - yi) / Y;
      size++; if (zi < zmin) zmin = zi; if (zi > zmax) zmax = zi;
      const nb = j => { if (air[j] && !lab[j]) { lab[j] = id; Q[t++] = j; } };
      if (xi > 0) nb(i - 1); if (xi < X - 1) nb(i + 1);
      if (yi > 0) nb(i - X); if (yi < Y - 1) nb(i + X);
      if (zi > 0) nb(i - plane); if (zi < Z - 1) nb(i + plane);
    }
    comps.push({ id, size, zmin, zmax });
  }
  const none = { bodyZMin, zMin: null, zMax: null };
  if (comps.length === 1) return none;
  const largest = Math.max(...comps.slice(1).map(c => c.size));
  const kept = comps.slice(1).filter(c => c.size >= 0.1 * largest);
  kept.sort((a, b) => (b.zmax - a.zmax) || (b.size - a.size));
  // _select_superior_component_cluster
  const sel = [kept[0].id]; let zMin = kept[0].zmin, zMax = kept[0].zmax;
  for (const c of kept.slice(1)) {
    if (Math.max(0, zMin - c.zmax - 1) * zSpacing > 40) continue;
    const nz = Math.min(zMin, c.zmin);
    if ((zMax - nz + 1) * zSpacing > 500) continue;
    sel.push(c.id); zMin = nz; zMax = Math.max(zMax, c.zmax);
  }
  const isSel = new Uint8Array(comps.length); for (const id of sel) isSel[id] = 1;
  // cap the chest span at 500 mm from the top, then count the remaining lung volume
  const zHas = new Int32Array(Z);
  for (let z = 0; z < Z; z++) for (let i = 0; i < plane; i++) if (isSel[lab[i + plane * z]]) zHas[z]++;
  let lo = zHas.findIndex(c => c > 0), hi = Z - 1 - [...zHas].reverse().findIndex(c => c > 0);
  if (lo < 0) return none;
  const capped = Math.max(lo, hi - Math.max(1, Math.ceil(500 / zSpacing)) + 1);
  for (let z = 0; z < capped; z++) zHas[z] = 0;
  lo = zHas.findIndex(c => c > 0); hi = Z - 1 - [...zHas].reverse().findIndex(c => c > 0);
  const ml = zHas.reduce((s, c) => s + c, 0) * PIX ** 3 / 1000;
  if (lo < 0 || ml < 250) return { bodyZMin, zMin: null, zMax: null, ml };
  return { bodyZMin, zMin: lo, zMax: hi, ml };
}
