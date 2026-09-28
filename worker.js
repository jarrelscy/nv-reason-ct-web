// Runs NV-Reason-CT off the main thread.
// Messages in: {preload, device, variant} | {files, device, variant, region} | {ask, thinking, max} | {stop} | {newChat} | {clearCache}.
// variant is the decoder build: 'int8' (full accuracy) or 'int4' (smaller).
// Messages out: {type:'log'|'progress'|'ready'|'volume'|'text'|'reply'|'error', ...}.
import * as ort from './vendor/ort/ort.webgpu.min.mjs';
import { openInput } from './lib/source.js';
import { pickSeries, seriesVolume, ABDO, CHEST } from './lib/series.js';
import { niftiVolume } from './lib/nifti.js';
import { preprocess } from './lib/preprocess.js';
import { loadNV } from './lib/nvreason.js';

const base = new URL('./', import.meta.url).href;
ort.env.wasm.wasmPaths = base + 'vendor/ort/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1;
ort.env.logLevel = 'error';

// Model files are on Hugging Face, pinned to one commit so the browser cache never mixes versions.
// ?models=URL (relative to the page) overrides this for local testing.
const MODELS = new URL(self.location.href).searchParams.get('models') || 'https://huggingface.co/jarrelscy/nv-reason-ct-onnx/resolve/83b49f5557298933f71bc0db9c8cbc20653c6032/';
const CACHE = 'nv-reason-ct-models';
const post = (type, x = {}) => self.postMessage({ type, ...x });
const log = msg => post('log', { msg });
const stage = (name, frac) => post('progress', { stage: name, frac });
const MB = x => x >= 1e9 ? (x / 1e9).toFixed(2) + ' GB' : (x / 1e6).toFixed(0) + ' MB';

// Each file is streamed from Hugging Face into the Cache API and read back as a disk-backed Blob,
// so the weights never sit in memory twice. After the first visit everything comes from the cache.
async function fetchModels(variant) {
  const cache = self.caches ? await caches.open(CACHE) : null;
  const get = async name => {
    const url = new URL(name, MODELS).href;
    let res = cache && await cache.match(url);
    if (res) return { res, cached: true };
    res = await fetch(url).catch(() => { throw new Error(`could not reach Hugging Face for ${name}. The models need an internet connection the first time; after that they load from this browser's cache`); });
    if (!res.ok) throw new Error(`failed to fetch ${name} from Hugging Face: ${res.status}`);
    return { res, cached: false };
  };
  const json = async name => { const { res, cached } = await get(name); const b = await res.clone().blob(); if (cache && !cached) await cache.put(new URL(name, MODELS).href, res); return JSON.parse(await b.text()); };
  const dir = { vision: '', decoder: variant + '/' };
  const man = { vision: await json('vision.json'), decoder: await json(dir.decoder + 'decoder.json') };
  const names = ['tokenizer.json', 'tokenizer_config.json'], sizes = {};
  for (const [k, m] of Object.entries(man)) {
    names.push(dir[k] + m.onnx, ...m.data.map(d => dir[k] + d.name));
    for (const d of m.data) sizes[dir[k] + d.name] = d.bytes;
  }
  const total = Object.values(sizes).reduce((a, b) => a + b, 0);
  let got = 0, logged = 0, announced = false;
  const blobs = {};
  for (const name of names) {
    const { res, cached } = await get(name), t0 = performance.now();
    if (cached) {
      blobs[name] = await res.blob(); got += sizes[name] || 0;
      log(`${name}: ${MB(blobs[name].size)} from browser cache`);
      stage(`Loading models from cache · ${MB(got)} of ${MB(total)}`, got / total);
      continue;
    }
    if (!announced) { announced = true; log(`downloading models from Hugging Face (${MB(total)}, cached in this browser for next time)`); }
    let n = 0;
    const counted = res.body.pipeThrough(new TransformStream({ transform(c, ctl) {
      n += c.length; got += c.length; ctl.enqueue(c);
      stage(`Downloading models · ${MB(got)} of ${MB(total)}`, Math.min(got / total, 1));
      if (got - logged >= 100e6) { logged = got; log(`downloaded ${MB(got)} of ${MB(total)}`); }
    } }));
    const url = new URL(name, MODELS).href, fresh = new Response(counted, { headers: { 'content-type': 'application/octet-stream' } });
    if (cache) {
      try { await cache.put(url, fresh); blobs[name] = await (await cache.match(url)).blob(); }
      catch (e) { throw new Error(`could not store ${name} in the browser cache (${e.message}). The models need about 6 GB of free browser storage for the 8-bit build or 3.5 GB for the 4-bit build`); }
    } else blobs[name] = await fresh.blob();
    log(`${name}: ${MB(n)} downloaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  }
  // files from an older model commit are no longer needed
  if (cache) for (const r of await cache.keys()) if (!r.url.startsWith(MODELS)) await cache.delete(r);
  const part = k => ({ man: man[k], blobs: Object.fromEntries([man[k].onnx, ...man[k].data.map(d => d.name)].map(n => [n, blobs[dir[k] + n]])) });
  return { vision: part('vision'), decoder: part('decoder'), tokenizer: blobs['tokenizer.json'], tokenizerConfig: blobs['tokenizer_config.json'] };
}

log(`cross-origin isolated: ${self.crossOriginIsolated}; CPU threads: ${ort.env.wasm.numThreads}`);

let nv = null;
function getNV(device, variant) {
  if (nv && nv.device === device && nv.variant === variant) return nv.p;
  const old = nv;
  const p = (async () => {
    if (old) await old.p.then(r => Promise.all([r.vision.release(), r.decoder.release()]), () => {});
    const files = await fetchModels(variant);
    stage('Building model sessions', 1);
    return loadNV(ort, { files, device, log });
  })();
  nv = { device, variant, p };
  p.catch(() => { if (nv && nv.p === p) nv = null; });
  return p;
}

async function pickDevice(want) {
  if (want === 'wasm') return 'wasm';
  const gpu = self.navigator.gpu && await self.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
  if (gpu) {
    const l = gpu.limits;
    log(`WebGPU adapter: maxBufferSize ${(l.maxBufferSize / 1e6).toFixed(0)} MB, maxStorageBufferBindingSize ${(l.maxStorageBufferBindingSize / 1e6).toFixed(0)} MB`);
    return 'webgpu';
  }
  if (want === 'webgpu') throw new Error('WebGPU is not available in this browser');
  log('WebGPU not available; using CPU (this will be very slow)');
  return 'wasm';
}

let stop = false, busy = Promise.resolve();
const serial = f => (busy = busy.then(f, f));

self.onmessage = ({ data }) => {
  if (data.stop) { stop = true; return; }
  serial(() => handle(data));
};

async function handle(data) {
  const t0 = performance.now();
  try {
    if (data.clearCache) {
      if (nv) await nv.p.then(r => Promise.all([r.vision.release(), r.decoder.release()]), () => {});
      nv = null;
      if (self.caches) await caches.delete(CACHE);
      log('cached model files removed');
      post('cleared');
      return;
    }
    if (data.preload) {
      try {
        stage('Loading models', 0);
        await getNV(await pickDevice(data.device), data.variant);
      } catch (e) { log(`model load failed (${e.message}); will retry when a scan is loaded`); }
      post('ready');
      return;
    }
    if (data.files) {
      const device = await pickDevice(data.device);
      const modelsReady = getNV(device, data.variant).catch(e => e);
      stage('Reading input', 0);
      const inp = await openInput(data.files, log);
      let vol, info = { region: data.region };
      if (inp.kind === 'nifti') vol = await niftiVolume(inp.file, log);
      else {
        const pick = await pickSeries(inp.entries, log, data.region === 'chest' ? CHEST : ABDO, data.region === 'chest' ? 'chest' : 'abdominal');
        info = { ...info, series_description: pick.best.desc || '', body_part: pick.best.body_part || '', series_selection: pick.why };
        vol = await seriesVolume(pick.best.files, log);
      }
      info.size = vol.dims; info.spacing_mm = vol.spacing.map(s => +s.toFixed(3));
      stage('Preprocessing', 0);
      const { img } = await preprocess(vol, data.region, log, f => stage('Decoding slices', f));
      stage('Loading models', 0);
      const R = await modelsReady;
      if (R instanceof Error) throw R;
      stage(`Encoding the volume (${R.device === 'webgpu' ? 'WebGPU' : 'CPU'})`, 0);
      await R.setVolume(img);
      log(`volume ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
      post('volume', { info, device: R.device });
      return;
    }
    const R = await nv?.p;
    if (!R) throw new Error('models are not loaded');
    if (data.newChat) { R.reset(); log('new conversation on the same volume'); post('reply', { text: '', reset: true }); return; }
    if (data.ask) {
      stop = false;
      const first = !R.past;
      stage(first ? 'Reading the volume and prompt' : 'Reading the question', 0);
      let n = 0;
      const text = await R.ask(data.ask, {
        thinking: data.thinking, max: data.max || 2048, shouldStop: () => stop,
        progress: f => stage(first ? 'Reading the volume and prompt' : 'Reading the question', f),
        onText: t => { if (++n % 2 === 0 || n < 4) post('text', { text: t }); stage(`Writing · ${n} tokens`, 1); },
      });
      post('reply', { text, tokens: R.lastTokens.length, seconds: (performance.now() - t0) / 1000, stopped: stop });
    }
  } catch (e) {
    console.error(e);
    post('error', { msg: e && e.message ? e.message : String(e) });
  }
}
