// Drives index.html in Playwright's Chromium: loads a CT, asks for a report, then a follow-up question.
//   node tools/site.mjs 'http://localhost:8766/?models=models/full/' CT.nii.gz [chest|abdomen] [question]
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW_CORE || '/home/jarrelscy/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core');
const [url, ct, region = 'abdomen', follow = 'Is there ascites?'] = process.argv.slice(2);
const ctx = await chromium.launchPersistentContext(process.env.USERDIR || '/tmp/pw-profile', {
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=vulkan', '--ignore-gpu-blocklist', '--enable-gpu', '--disable-gpu-watchdog'],
});
const page = await ctx.newPage();
page.on('console', m => { const t = m.text(); if (t.startsWith('[nv-reason]') && !/Downloading|downloaded /.test(t)) console.log(t.slice(0, 300)); });
page.on('pageerror', e => console.log('pageerror:', e.message));
await page.goto(url);
await page.waitForFunction(() => crossOriginIsolated, null, { timeout: 60e3 }).catch(() => {});
const T = { timeout: 1800e3 };
await page.waitForFunction(() => /Models ready|Error/.test(document.getElementById('stage').textContent), null, T);
await page.click(`label:has(input[name=region][value=${region}])`);
await page.setInputFiles('#files', ct);
await page.waitForFunction(() => !document.getElementById('chat').hidden || /Error/.test(document.getElementById('stage').textContent), null, T);
const reply = async () => {
  await page.waitForFunction(() => document.querySelectorAll('.msg.bot .meta').length === document.querySelectorAll('.msg.bot').length || /Error/.test(document.getElementById('stage').textContent), null, T);
  console.log('---- reply:\n' + await page.evaluate(() => [...document.querySelectorAll('.msg.bot')].pop()?.innerText));
};
console.log('presets:', await page.evaluate(() => [...document.querySelectorAll('#presets button')].map(b => b.textContent).join(' | ')));
await page.click('#presets button');
await reply();
await page.fill('#q', follow);
await page.click('#send');
await reply();
await page.screenshot({ path: '/tmp/site.png', fullPage: true });
await ctx.close();
