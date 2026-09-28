// Opens a test page in Playwright's Chromium with WebGPU and prints its console until it logs DONE.
//   node tools/page.mjs 'http://localhost:8766/tools/dec.html?model=dec-tiny'
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW_CORE || '/home/jarrelscy/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core');
// A persistent profile keeps large blobs and the Cache API on disk (an incognito context holds them in memory).
const browser = await chromium.launchPersistentContext(process.env.USERDIR || '/tmp/pw-profile', {
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=vulkan', '--ignore-gpu-blocklist', '--enable-gpu', '--disable-gpu-watchdog', ...(process.env.FLAGS ? process.env.FLAGS.split(' ') : [])],
});
const page = await browser.newPage();
let done;
const finished = new Promise(r => { done = r; });
page.on('console', m => { const t = m.text(); console.log(t.slice(0, 2000)); if (/\bDONE\b/.test(t)) done(); });
page.on('pageerror', e => { console.log('pageerror:', e.message); done(); });
page.on('crash', () => { console.log('page crashed'); done(); });
browser.on('close', () => { console.log('browser closed'); done(); });
await page.goto(process.argv[2]);
await Promise.race([finished, new Promise(r => setTimeout(r, +(process.env.TIMEOUT || 1800) * 1000))]);
await browser.close();
