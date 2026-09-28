import { filesFromDataTransfer } from './lib/source.js';

const $ = id => document.getElementById(id);
const logEl = $('log');
const t0 = performance.now();
let worker = null, busy = false, asking = false, runStart = 0, cur = { name: '', frac: 0 }, bot = null, volume = null;

function log(msg) {
  const t = ((performance.now() - t0) / 1000).toFixed(2).padStart(7);
  logEl.textContent += `[${t}s] ${msg}\n`; logEl.scrollTop = logEl.scrollHeight;
  console.log('[nv-reason]', msg);
}

(async () => {
  const bits = [];
  const gpu = navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null);
  bits.push(gpu ? 'WebGPU available' : 'no WebGPU: a desktop Chrome or Edge is needed');
  if (navigator.storage?.estimate) {
    const { quota, usage } = await navigator.storage.estimate();
    bits.push(`browser storage free: ${((quota - usage) / 1e9).toFixed(0)} GB`);
  }
  $('caps').textContent = bits.join(' · ');
})();

const variant = () => document.querySelector('input[name=variant]:checked').value;
const region = () => document.querySelector('input[name=region]:checked').value;

// ?models=... is passed through for testing with local model files
const wurl = new URL('./worker.js', import.meta.url);
const models = new URLSearchParams(location.search).get('models');
if (models) wurl.searchParams.set('models', new URL(models, location.href).href);
worker = new Worker(wurl, { type: 'module' });
worker.onmessage = onMessage;
worker.onerror = e => { log(`worker error: ${e.message}`); done(); };

// Fetch and build the models as soon as the page opens so the first scan starts straight away.
busy = true; runStart = performance.now();
$('status').hidden = false; setStage('Loading models', 0);
log('loading models');
const preload = () => { if (asking) return; busy = true; runStart = performance.now(); $('status').hidden = false; setStage('Loading models', 0); volume = null; $('chat').hidden = true; worker.postMessage({ preload: true, variant: variant() }); };
worker.postMessage({ preload: true, variant: variant() });
document.querySelectorAll('input[name=variant]').forEach(el => el.addEventListener('change', () => { log(`switching to ${variant()}; load the scan again afterwards`); preload(); }));
$('clearCache').onclick = () => { if (!asking && confirm('Delete the downloaded model files from this browser? They will be downloaded again next time.')) { busy = true; volume = null; $('chat').hidden = true; worker.postMessage({ clearCache: true }); } };

function presets() {
  const r = region(), adj = r === 'chest' ? 'chest' : 'abdominal';
  const list = [
    [`Structured ${adj} report`, `Write a structured ${adj} CT report.`, false],
    ['Reasoning analysis', `Provide a full reasoning analysis of this ${adj} CT.`, true],
    ...(r === 'chest'
      ? [['Pleural effusion?', 'Is a pleural effusion present in this CT?'], ['Lung nodules?', 'Are there any lung nodules in this CT?']]
      : [['Liver lesion?', 'Is there a liver lesion in this CT?'], ['Kidney stones?', 'Are there kidney stones in this CT?']]),
  ];
  $('presets').innerHTML = '';
  for (const [label, q, think] of list) {
    const b = Object.assign(document.createElement('button'), { className: 'btn small', type: 'button', textContent: label, title: q });
    b.onclick = () => ask(q, think ?? $('thinking').checked);
    $('presets').append(b);
  }
}

function run(files) {
  files = [...files];
  if (!files.length || asking) return;
  busy = true; runStart = performance.now(); volume = null;
  $('drop').classList.add('busy'); $('status').hidden = false; $('chat').hidden = true; $('msgs').innerHTML = '';
  log('---- new scan');
  setStage('Starting', 0);
  log(`${files.length} file(s): ${files.slice(0, 3).map(f => f.webkitRelativePath || f.name).join(', ')}${files.length > 3 ? ', …' : ''}; region ${region()}`);
  worker.postMessage({ files, variant: variant(), region: region() });
}

function ask(q, thinking) {
  q = q.trim();
  if (!q || busy || !volume) return;
  busy = asking = true; runStart = performance.now();
  addMsg('user', q);
  bot = addMsg('bot working', '');
  $('q').value = ''; buttons();
  worker.postMessage({ ask: q, thinking, max: thinking ? 4096 : 2048 });
}

function addMsg(cls, text) {
  const el = document.createElement('div'); el.className = `msg ${cls}`;
  const body = document.createElement('div'); body.className = 'body'; body.textContent = text;
  el.append(body); $('msgs').append(el); el.scrollIntoView({ block: 'nearest' });
  return el;
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// Reasoning comes before </think>; the final label list is wrapped in <answer> tags.
function renderReply(el, text) {
  const i = text.indexOf('</think>');
  let html = '';
  if (i >= 0 || el.dataset.thinking === '1') {
    const think = i >= 0 ? text.slice(0, i) : text;
    html += `<div class="think">${esc(think.trim())}</div>`;
    text = i >= 0 ? text.slice(i + 8) : '';
  }
  html += esc(text.trim()).replace(/&lt;answer&gt;([\s\S]*?)(&lt;\/answer&gt;|$)/g, (_, a) => `<span class="answer">Answer: ${a.trim()}</span>`);
  el.querySelector('.body').innerHTML = html;
}

function done() {
  busy = asking = false; $('drop').classList.remove('busy');
  if (bot) bot.classList.remove('working');
  buttons();
}

function buttons() {
  $('send').disabled = busy || !volume; $('stop').disabled = !asking; $('newChat').disabled = busy || !volume;
  document.querySelectorAll('#presets button').forEach(b => { b.disabled = busy || !volume; });
}

function setStage(name, frac) {
  cur = { name, frac };
  $('stage').textContent = name;
  $('bar').style.width = `${Math.round(frac * 100)}%`;
  showPct();
}

function showPct() {
  const pct = cur.frac > 0 && cur.frac < 1 ? `${Math.round(cur.frac * 100)}%` : '';
  $('pct').textContent = busy ? [pct, `${((performance.now() - runStart) / 1000).toFixed(0)}s`].filter(Boolean).join(' · ') : pct;
}
setInterval(() => { if (busy) showPct(); }, 1000);

function onMessage({ data }) {
  if (data.type === 'log') log(data.msg);
  else if (data.type === 'ready') { if (!volume && !$('drop').classList.contains('busy')) { busy = false; setStage('Models ready. Drop a scan to start.', 1); } }
  else if (data.type === 'cleared') { busy = false; setStage('Cached model files removed. They will be downloaded again when needed.', 0); }
  else if (data.type === 'progress') setStage(data.stage, data.frac);
  else if (data.type === 'error') { log(`ERROR: ${data.msg}`); setStage(`Error: ${data.msg}`, 0); if (bot && asking) bot.querySelector('.body').textContent = `Error: ${data.msg}`; done(); }
  else if (data.type === 'volume') {
    volume = data;
    const i = data.info, sum = [];
    if (i.series_description !== undefined) sum.push(`Series <b>${esc(i.series_description || '?')}</b>${i.body_part ? ` [${esc(i.body_part)}]` : ''} (${esc(i.series_selection)})`);
    sum.push(`${i.size.join('×')} voxels at ${i.spacing_mm.join(' × ')} mm`, `<b>${esc(i.region)}</b> crop`, 'research use only, not for clinical decisions');
    $('summary').innerHTML = sum.join(' · ');
    $('chat').hidden = false; presets();
    setStage(`Scan ready in ${((performance.now() - runStart) / 1000).toFixed(0)}s. Pick a prompt or ask a question.`, 1);
    done();
  }
  else if (data.type === 'text') { if (bot) renderReply(bot, data.text); }
  else if (data.type === 'reply') {
    if (data.reset) { $('msgs').innerHTML = ''; bot = null; setStage('New conversation. The scan stays loaded.', 1); done(); return; }
    renderReply(bot, data.text);
    const m = document.createElement('div'); m.className = 'meta';
    m.textContent = `${data.tokens} tokens in ${data.seconds.toFixed(0)}s${data.stopped ? ' · stopped' : ''}`;
    bot.append(m);
    setStage(`Done in ${data.seconds.toFixed(0)}s`, 1);
    done();
  }
}

$('askForm').onsubmit = e => { e.preventDefault(); ask($('q').value, $('thinking').checked); };
$('q').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('askForm').requestSubmit(); } });
$('stop').onclick = () => worker.postMessage({ stop: true });
$('newChat').onclick = () => { if (!busy) { busy = true; worker.postMessage({ newChat: true }); } };
document.querySelectorAll('input[name=region]').forEach(el => el.addEventListener('change', () => { if (volume) log('region changed: load the scan again to use the new crop'); }));

$('files').onchange = e => { run(e.target.files); e.target.value = ''; };
$('folder').onchange = e => { run(e.target.files); e.target.value = ''; };
const drop = $('drop');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', async e => { e.preventDefault(); drop.classList.remove('over'); run(await filesFromDataTransfer(e.dataTransfer)); });
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());
