// NV-Reason-CT in the browser: the Primus 3D ViT turns the 192^3 crop into 13,824 tokens, then the
// Qwen3.5 decoder (int4 or int8 weights, fp32 activations) reads them with the prompt and generates greedily.
//
// Token embeddings are tied to lm_head, so they are not shipped separately: each text token's row is
// dequantised from the lm_head int4 or int8 weights in the cached decoder files (see tools/package.py).
// The prompt is prefilled in chunks and the attention/recurrent state stays on the GPU between runs.
import { Tokenizer } from '../vendor/tokenizers/tokenizers.min.mjs';

export const IMAGE = 248056, IM_END = 248046, EOS = new Set([248046, 248044]);
const N_IMG = 13824, GRID = 24, HID = 2560;

export async function loadNV(ort, { files, device = 'webgpu', log = () => {} }) {
  const t0 = performance.now();
  const ep = device === 'webgpu' ? ['webgpu'] : ['wasm'];
  const opts = { executionProviders: ep, graphOptimizationLevel: 'all' };
  const bytes = async b => new Uint8Array(await b.arrayBuffer());
  const create = async (m, what, outputs) => {
    const t = performance.now();
    const ext = await Promise.all(m.man.data.map(async d => ({ path: d.name, data: await bytes(m.blobs[d.name]) })));
    const extra = {};
    if (outputs && ep[0] === 'webgpu') {
      // keep the recurrent and attention state on the GPU; only logits come back
      extra.preferredOutputLocation = { logits: 'cpu' };
      for (const k of outputs) extra.preferredOutputLocation[k] = 'gpu-buffer';
    }
    const s = await ort.InferenceSession.create(await bytes(m.blobs[m.man.onnx]), { ...opts, ...extra, externalData: ext });
    log(`${what} session ready in ${((performance.now() - t) / 1000).toFixed(1)}s`);
    return s;
  };
  const vision = await create(files.vision, 'vision');
  const decoder = await create(files.decoder, 'decoder', files.decoder.man.state.map(presentName));
  const tok = new Tokenizer(JSON.parse(await files.tokenizer.text()), JSON.parse(await files.tokenizerConfig.text()));
  log(`models ready on ${ep[0]} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return new NVReason(ort, { vision, decoder, tok, embed: new Embed(files.decoder.man.embed, files.decoder.blobs), device: ep[0], log });
}

// Rows of the tied embedding matrix, dequantised from lm_head's MatMulNBits weights.
class Embed {
  constructor(m, blobs) { this.m = m; this.blobs = blobs; }
  read(t, off, len) { return this.blobs[t.file].slice(t.offset + off, t.offset + off + len).arrayBuffer(); }
  async rows(ids, out = new Float32Array(ids.length * HID), at = 0) {
    const { K, block_size: bs, bits, q, scales, zero_points: z } = this.m;
    if ((bits !== 4 && bits !== 8) || K !== HID) throw new Error('embedding lookup expects an int4 or int8 lm_head with K=2560');
    const nb = K / bs, qb = K * bits / 8, zb = bits === 4 ? Math.ceil(nb / 2) : nb, sb = scales.type === 1 ? 4 : 2, zdef = bits === 4 ? 8 : 128;
    await Promise.all(ids.map(async (id, r) => {
      const [qa, sa, za] = await Promise.all([this.read(q, id * qb, qb), this.read(scales, id * nb * sb, nb * sb), z ? this.read(z, id * zb, zb) : null]);
      const qv = new Uint8Array(qa), zv = za && new Uint8Array(za);
      const sv = sb === 4 ? new Float32Array(sa) : Float32Array.from(new Uint16Array(sa), h2f);
      const o = (at + r) * K;
      for (let b = 0; b < nb; b++) {
        const zp = !zv ? zdef : bits === 4 ? (zv[b >> 1] >> ((b & 1) * 4)) & 15 : zv[b], s = sv[b];
        if (bits === 8) { for (let j = 0; j < bs; j++) out[o + b * bs + j] = (qv[b * bs + j] - zp) * s; continue; }
        for (let j = 0; j < bs; j++) {
          const k = b * bs + j, byte = qv[k >> 1];
          out[o + k] = (((k & 1) ? byte >> 4 : byte & 15) - zp) * s;
        }
      }
    }));
    return out;
  }
}

const presentName = k => k.replace('past_key_values', 'present').replace('past.', 'present.');

function h2f(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}

export class NVReason {
  constructor(ort, parts) {
    Object.assign(this, parts); this.ort = ort;
    this.chunk = 1024; this.feats = null; this.reset();
  }

  // Vision tower on the preprocessed crop (Float32Array 192^3, x fastest). Clears the conversation.
  async setVolume(img) {
    const t = performance.now();
    const o = await this.vision.run({ volume: new this.ort.Tensor('float32', img, [1, 1, 192, 192, 192]) });
    this.feats = await o.tokens.getData(true);
    this.log(`vision encoder: ${N_IMG} tokens in ${((performance.now() - t) / 1000).toFixed(1)}s`);
    this.reset();
  }

  reset() {
    if (this.past) for (const v of Object.values(this.past)) v.dispose?.();
    this.past = null; this.total = 0; this.pos = 0; this.pending = null;
  }

  emptyState() {
    const T = this.ort.Tensor, f = {};
    for (const name of this.decoder.inputNames) {
      if (name.startsWith('past_key_values')) f[name] = new T('float32', new Float32Array(0), [1, 4, 0, 256]);
      else if (name.endsWith('.conv')) f[name] = new T('float32', new Float32Array(8192 * 3), [1, 8192, 3]);
      else if (name.endsWith('.recurrent')) f[name] = new T('float32', new Float32Array(32 * 128 * 128), [1, 32, 128, 128]);
    }
    return f;
  }

  // One decoder run over n new tokens; embeds is [n, 2560], pos is [3][n] flattened. Returns last logits.
  async step(embeds, n, pos) {
    const T = this.ort.Tensor, past = this.past || this.emptyState();
    const feeds = {
      ...past,
      inputs_embeds: new T('float32', embeds, [1, n, HID]),
      attention_mask: new T('int64', new BigInt64Array(this.total + n).fill(1n), [1, this.total + n]),
      position_ids: new T('int64', pos, [3, 1, n]),
    };
    const out = await this.decoder.run(feeds);
    const next = {};
    for (const k of Object.keys(past)) next[k] = out[presentName(k)];
    for (const v of Object.values(past)) v.dispose?.();
    this.past = next; this.total += n;
    return out.logits.getData(true);
  }

  // Feed a token sequence (image tokens take the vision features) in chunks; returns the last logits.
  async feed(ids, progress = () => {}) {
    let logits = null;
    const imgAt = ids.indexOf(IMAGE);
    for (let c = 0; c < ids.length; c += this.chunk) {
      const part = ids.slice(c, c + this.chunk), n = part.length;
      const x = new Float32Array(n * HID), pos = new BigInt64Array(3 * n), text = [], textAt = [];
      for (let i = 0; i < n; i++) {
        const id = part[i], g = c + i;
        if (id === IMAGE) {
          const v = g - imgAt; x.set(this.feats.subarray(v * HID, (v + 1) * HID), i * HID);
          const t = Math.floor(v / (GRID * GRID)), h = Math.floor(v / GRID) % GRID, w = v % GRID;
          pos[i] = BigInt(this.pos + t); pos[n + i] = BigInt(this.pos + h); pos[2 * n + i] = BigInt(this.pos + w);
        } else {
          if (g > 0 && ids[g - 1] === IMAGE) this.pos += GRID; // positions resume after the volume grid
          text.push(id); textAt.push(i);
          for (let d = 0; d < 3; d++) pos[d * n + i] = BigInt(this.pos);
          this.pos++;
        }
      }
      if (text.length) {
        const e = await this.embed.rows(text);
        textAt.forEach((i, r) => x.set(e.subarray(r * HID, (r + 1) * HID), i * HID));
      }
      logits = await this.step(x, n, pos);
      progress(Math.min(c + n, ids.length) / ids.length);
    }
    return logits;
  }

  // Ask about the current volume. The first question carries the image; later ones continue the chat.
  // onText(fullTextSoFar) is called as tokens arrive. Returns the reply text.
  async ask(text, { thinking = false, max = 1024, onText = () => {}, progress = () => {}, shouldStop = () => false } = {}) {
    if (!this.feats) throw new Error('no volume loaded');
    const gen = thinking ? '<think>\n' : '<think>\n\n</think>\n\n';
    let ids;
    if (!this.past) {
      const e = this.tok.encode(`<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>${text}<|im_end|>\n<|im_start|>assistant\n${gen}`, { add_special_tokens: false }).ids;
      const at = e.indexOf(IMAGE);
      ids = [...e.slice(0, at), ...new Array(N_IMG).fill(IMAGE), ...e.slice(at + 1)];
    } else {
      const head = this.pending === null || EOS.has(this.pending) ? [IM_END] : [this.pending, IM_END];
      ids = [...head, ...this.tok.encode(`\n<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n${gen}`, { add_special_tokens: false }).ids];
    }
    const t0 = performance.now();
    let logits = await this.feed(ids, progress);
    const t1 = performance.now();
    this.log(`prefill: ${ids.length} tokens in ${((t1 - t0) / 1000).toFixed(1)}s`);
    const out = [];
    let tok = argmax(logits), reply = '';
    for (let k = 0; k < max; k++) {
      if (EOS.has(tok) || shouldStop()) break;
      out.push(tok);
      reply = this.tok.decode(out, { skip_special_tokens: true }); onText(reply);
      logits = await this.feed([tok]); tok = argmax(logits);
    }
    this.pending = tok;
    const dt = (performance.now() - t1) / 1000;
    this.log(`generated ${out.length} tokens in ${dt.toFixed(1)}s (${(out.length / Math.max(dt, 1e-3)).toFixed(1)} tokens/s)`);
    this.lastTokens = out;
    return reply;
  }
}

function argmax(a) {
  let b = 0, m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) { m = a[i]; b = i; }
  return b;
}
