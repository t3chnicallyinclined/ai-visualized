/* llm.js: real small language models running in plain JavaScript, no libraries.
   Both are Llama-style transformers: SmolLM2-135M-Instruct (30 layers, attention with 9 heads sharing 3
   key/value heads, a chat model that reads a system prompt) and TinyStories 15M (6 layers, story-only).
   Each layer has attention and a feed-forward block. The weights are the published bf16/f16 numbers, widened
   to 32-bit floats, so the maths matches the reference implementation. Alongside the normal output it records what the page shows:
   each layer's attention weights, its most active feed-forward neurons, the token's numbers after the
   layer, and the "logit lens" (the word the model would pick if it stopped after that layer).
   Works in the browser (window.LLM) and in Node (module.exports) for testing. */
(function (root) {
  'use strict';
  // revisions are pinned, so the numbers never change under the page
  const MODELS = {
    'smollm2-instruct': { name: 'SmolLM2-135M-Instruct', repo: 'HuggingFaceTB/SmolLM2-135M-Instruct', rev: '12fd25f77366fa6b3b4b768ec3050bf629380bac', mb: 269, chat: true, bos: null, license: 'Apache 2.0, by Hugging Face', layers: 30, hidden: 576, ff: 1536, heads: 9, kvHeads: 3, vocab: 49152 },
    'tiny-stories': { name: 'TinyStories 15M', repo: 'nickypro/tinyllama-15M', rev: 'a97e01fa7d54c088df9dc1b53d581a863ce08cd6', mb: 30, chat: false, bos: 1, license: 'MIT, by Andrej Karpathy (llama2.c), converted by nickypro', layers: 6, hidden: 288, ff: 768, heads: 6, kvHeads: 6, vocab: 32000 },
  };
  // LLM_BASE: optional local mirrors, { modelKey: url } (used by tests)
  const baseOf = key => (root.LLM_BASE && root.LLM_BASE[key]) || `https://huggingface.co/${MODELS[key].repo}/resolve/${MODELS[key].rev}/`;

  // ---------- tokenizer: byte-level BPE (GPT-2 style), digits split one by one ----------
  function bytesToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice(); let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    const enc = {}, dec = {};
    bs.forEach((b, i) => { enc[b] = String.fromCharCode(cs[i]); dec[String.fromCharCode(cs[i])] = b; });
    return { enc, dec };
  }
  const SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  // SentencePiece-style BPE (Llama): spaces become "▁", one goes in front, unknown characters fall back to bytes
  const isSentencePiece = tj => !!tj.model.byte_fallback;
  function makeTokenizer(tj) {
    const vocab = tj.model.vocab, inv = [], sp = isSentencePiece(tj);
    for (const k in vocab) inv[vocab[k]] = k;
    for (const a of tj.added_tokens) inv[a.id] = a.content;
    const ranks = new Map();
    tj.model.merges.forEach((m, i) => ranks.set(Array.isArray(m) ? m.join(' ') : m, i));
    const { enc, dec } = bytesToUnicode();
    const utf8 = new TextEncoder(), utf8d = new TextDecoder();
    const cache = new Map();
    function bpe(word) {
      if (cache.has(word)) return cache.get(word);
      let parts = Array.from(word);
      while (parts.length > 1) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = ranks.get(parts[i] + ' ' + parts[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        parts = parts.slice(0, best).concat(parts[best] + parts[best + 1], parts.slice(best + 2));
      }
      cache.set(word, parts);
      return parts;
    }
    // special tokens (like the chat template's <|im_start|>) are matched whole and never split
    const special = new Map(tj.added_tokens.filter(a => a.special).map(a => [a.content, a.id]));
    const specialRe = special.size ? new RegExp('(' + [...special.keys()].map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')') : null;
    function encode(text) {
      const ids = [];
      for (const chunk of specialRe ? text.split(specialRe) : [text]) {
        if (!chunk) continue;
        if (special.has(chunk)) { ids.push(special.get(chunk)); continue; }
        if (sp) {
          for (const p of bpe('\u2581' + chunk.replace(/ /g, '\u2581'))) {
            if (vocab[p] != null) ids.push(vocab[p]);
            else for (const b of utf8.encode(p)) ids.push(vocab['<0x' + b.toString(16).toUpperCase().padStart(2, '0') + '>']);
          }
          continue;
        }
        // pre-tokenizer 1: every digit is its own piece; pre-tokenizer 2: the GPT-2 split, applied inside each piece
        const pieces = chunk.split(/(\p{N})/u).filter(s => s.length);
        for (const piece of pieces) for (const m of piece.match(SPLIT) || []) {
          const mapped = Array.from(utf8.encode(m), b => enc[b]).join('');
          for (const p of bpe(mapped)) ids.push(vocab[p]);
        }
      }
      return ids;
    }
    const spBytes = id => { const t = inv[id] || '', m = /^<0x([0-9A-F]{2})>$/.exec(t); return m ? [parseInt(m[1], 16)] : Array.from(utf8.encode(t.replace(/\u2581/g, ' '))); };
    function decode(ids, keepLead) {
      if (!sp) return utf8d.decode(new Uint8Array(ids.flatMap(id => Array.from(inv[id] || '', c => dec[c]))));
      const s = utf8d.decode(new Uint8Array(ids.flatMap(id => (special.has(inv[id]) ? [] : spBytes(id)))));
      return keepLead ? s : s.replace(/^ /, ''); // the decoder drops the space it put in front
    }
    // what a learner sees for one token: its text, with spaces shown as "·" (newlines ⏎, tabs ⇥)
    function piece(id) { const s = sp ? (special.has(inv[id]) ? inv[id] : decode([id], true)) : decode([id]); return s.replace(/ /g, '·').replace(/\n/g, '⏎').replace(/\t/g, '⇥') || '∅'; }
    return { encode, decode, piece, raw: id => inv[id], size: inv.length };
  }

  // ---------- weights: safetensors file, bf16 -> f32 ----------
  function parseSafetensors(buf) {
    const n = Number(new DataView(buf).getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
    const base = 8 + n, W = {};
    for (const name in header) {
      if (name === '__metadata__') continue;
      const { dtype, shape, data_offsets: [a, b] } = header[name];
      if (dtype !== 'BF16' && dtype !== 'F16') throw new Error('expected BF16 or F16 weights, got ' + dtype);
      const count = (b - a) / 2, src = (base + a) % 2 === 0 ? new Uint16Array(buf, base + a, count) : new Uint16Array(buf.slice(base + a, base + b));
      const out = new Float32Array(count), o32 = new Uint32Array(out.buffer);
      if (dtype === 'BF16') for (let i = 0; i < count; i++) o32[i] = src[i] << 16; // bf16 is the top half of an f32: exact
      else for (let i = 0; i < count; i++) { // f16 -> f32, exact (sign, 5-bit exponent, 10-bit fraction)
        const h = src[i], sgn = (h & 0x8000) << 16, e = (h >> 10) & 31, f = h & 1023;
        if (e === 0) out[i] = (sgn ? -1 : 1) * f * 2 ** -24; else if (e === 31) o32[i] = sgn | 0x7f800000 | (f << 13); else o32[i] = sgn | ((e + 112) << 23) | (f << 13);
      }
      W[name] = { data: out, shape };
    }
    return W;
  }

  // ---------- maths ----------
  function matvec(W, rows, cols, x, out) {
    for (let r = 0; r < rows; r++) {
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, o = r * cols, c = 0;
      for (; c + 3 < cols; c += 4) { s0 += W[o + c] * x[c]; s1 += W[o + c + 1] * x[c + 1]; s2 += W[o + c + 2] * x[c + 2]; s3 += W[o + c + 3] * x[c + 3]; }
      for (; c < cols; c++) s0 += W[o + c] * x[c];
      out[r] = s0 + s1 + s2 + s3;
    }
    return out;
  }
  function rmsnorm(x, w, eps, out) {
    let ss = 0; for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
    const k = 1 / Math.sqrt(ss / x.length + eps);
    for (let i = 0; i < x.length; i++) out[i] = x[i] * k * w[i];
    return out;
  }
  function topk(arr, k) {
    const idx = [];
    for (let i = 0; i < arr.length; i++) {
      if (idx.length < k) { idx.push(i); idx.sort((a, b) => arr[b] - arr[a]); }
      else if (arr[i] > arr[idx[k - 1]]) { idx[k - 1] = i; idx.sort((a, b) => arr[b] - arr[a]); }
    }
    return idx;
  }
  function softmaxStats(logits) { // max and log-sum-exp over the whole vocabulary
    let mx = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
    let s = 0; for (let i = 0; i < logits.length; i++) s += Math.exp(logits[i] - mx);
    return { mx, lse: mx + Math.log(s) };
  }
  const tick = () => new Promise(r => setTimeout(r, 0));

  function makeModel(cfg, W, tok) {
    const D = cfg.hidden_size, L = cfg.num_hidden_layers, H = cfg.num_attention_heads, KVH = cfg.num_key_value_heads;
    const HD = D / H, F = cfg.intermediate_size, V = cfg.vocab_size, eps = cfg.rms_norm_eps, rep = H / KVH;
    const g = n => W[n].data;
    const E = g('model.embed_tokens.weight'); // also the output matrix (tied embeddings)
    const layers = [];
    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      layers.push({ ln1: g(p + 'input_layernorm.weight'), ln2: g(p + 'post_attention_layernorm.weight'),
        q: g(p + 'self_attn.q_proj.weight'), k: g(p + 'self_attn.k_proj.weight'), v: g(p + 'self_attn.v_proj.weight'), o: g(p + 'self_attn.o_proj.weight'),
        gate: g(p + 'mlp.gate_proj.weight'), up: g(p + 'mlp.up_proj.weight'), down: g(p + 'mlp.down_proj.weight') });
    }
    const normW = g('model.norm.weight');
    const invFreq = new Float32Array(HD / 2);
    for (let i = 0; i < HD / 2; i++) invFreq[i] = 1 / Math.pow(cfg.rope_theta, (2 * i) / HD);
    function rope(vec, heads, pos) { // Llama rotary embedding, halves rotated (not interleaved)
      for (let h = 0; h < heads; h++) {
        const o = h * HD;
        for (let i = 0; i < HD / 2; i++) {
          const a = pos * invFreq[i], c = Math.cos(a), s = Math.sin(a), x1 = vec[o + i], x2 = vec[o + i + HD / 2];
          vec[o + i] = x1 * c - x2 * s; vec[o + i + HD / 2] = x2 * c + x1 * s;
        }
      }
    }
    const kvBytesPerToken = 2 * L * KVH * HD * 2; // keys + values, every layer, stored as bf16 in a real runtime
    const weightBytes = Object.values(W).reduce((a, t) => a + t.data.length * 2, 0);

    // optional: what the token cloud keeps after every layer. A list of token IDs keeps their raw scores;
    // 'all' keeps every token's probability (full softmax over the whole vocabulary) as a log-scale byte:
    // 0 = 1e-6 or less, 255 = probability 1 (about 5% steps). 30 layers x 49,152 bytes = 1.5 MB per token.
    let watch = null;
    const WATCH_FLOOR = Math.log(1e-6);
    const setWatch = ids => { watch = ids === 'all' ? 'all' : ids ? Int32Array.from(ids) : null; };
    function watchOf(logits, lse) {
      if (watch !== 'all') return Float32Array.from(watch, i => logits[i]);
      const out = new Uint8Array(V), k = 255 / -WATCH_FLOOR;
      for (let i = 0; i < V; i++) { const lp = logits[i] - lse; out[i] = lp <= WATCH_FLOOR ? 0 : Math.round((lp - WATCH_FLOOR) * k); }
      return out;
    }
    let lastLse = 0;
    function newState() { return { pos: 0, K: layers.map(() => []), Vc: layers.map(() => []) }; }
    const buf = { x: new Float32Array(D), h: new Float32Array(D), q: new Float32Array(D), k: new Float32Array(KVH * HD), v: new Float32Array(KVH * HD),
      att: new Float32Array(D), o: new Float32Array(D), g: new Float32Array(F), u: new Float32Array(F), d: new Float32Array(D), logits: new Float32Array(V), hn: new Float32Array(D) };

    function lens(x, n) { // the logit lens: final norm + output matrix applied to an intermediate state
      rmsnorm(x, normW, eps, buf.hn); matvec(E, V, D, buf.hn, buf.logits);
      const { lse } = softmaxStats(buf.logits); lastLse = lse;
      return topk(buf.logits, n).map(id => ({ id, p: Math.exp(buf.logits[id] - lse) }));
    }

    // Run one token through every layer. record=true returns what every sub-step of every layer did;
    // record='attn' keeps only the attention weights (cheap: no logit lens).
    async function step(state, id, record, lensTop = 3) {
      const t0 = performance.now(), pos = state.pos, x = buf.x, full = record === true;
      x.set(E.subarray(id * D, id * D + D));
      const rec = record ? { id, pos, embed: Array.from(x.subarray(0, 8)), layers: [] } : null;
      const f6 = a => Array.from(a.subarray(0, 6));
      const nrm = a => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };
      let lensMs = 0;
      for (let l = 0; l < L; l++) {
        // full records also keep every whole vector of the layer (vec), so the page can show each step with real numbers
        const P = layers[l], ly = full ? { xin: f6(x), xinNorm: nrm(x), vec: { xin: x.slice() } } : {};
        rmsnorm(x, P.ln1, eps, buf.h);
        matvec(P.q, D, D, buf.h, buf.q); matvec(P.k, KVH * HD, D, buf.h, buf.k); matvec(P.v, KVH * HD, D, buf.h, buf.v);
        if (full) Object.assign(ly.vec, { h1: buf.h.slice(), qPre: buf.q.slice(), kPre: buf.k.slice() });
        rope(buf.q, H, pos); rope(buf.k, KVH, pos);
        if (full) { ly.h1 = f6(buf.h); ly.q = f6(buf.q); ly.k = f6(buf.k); ly.v = f6(buf.v); Object.assign(ly.vec, { q: buf.q.slice(), k: buf.k.slice(), v: buf.v.slice() }); }
        state.K[l].push(buf.k.slice()); state.Vc[l].push(buf.v.slice());
        const T = pos + 1, weights = record ? new Float32Array(H * T) : null, sc = new Float32Array(T);
        if (full) ly.sc = new Float32Array(T); // head 1's raw scores, before softmax
        for (let hh = 0; hh < H; hh++) {
          const kv = Math.floor(hh / rep), qo = hh * HD, ko = kv * HD;
          let mx = -Infinity;
          for (let j = 0; j < T; j++) { const kj = state.K[l][j]; let s = 0; for (let i = 0; i < HD; i++) s += buf.q[qo + i] * kj[ko + i]; s /= Math.sqrt(HD); sc[j] = s; if (s > mx) mx = s; }
          if (full && hh === 0) ly.sc.set(sc);
          let sum = 0; for (let j = 0; j < T; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
          for (let i = 0; i < HD; i++) buf.att[qo + i] = 0;
          for (let j = 0; j < T; j++) { const w = sc[j] / sum, vj = state.Vc[l][j]; if (weights) weights[hh * T + j] = w; for (let i = 0; i < HD; i++) buf.att[qo + i] += w * vj[ko + i]; }
        }
        matvec(P.o, D, D, buf.att, buf.o);
        for (let i = 0; i < D; i++) x[i] += buf.o[i];
        if (full) { ly.mix = f6(buf.att); ly.out = f6(buf.o); ly.outNorm = nrm(buf.o); ly.x1 = f6(x); ly.x1Norm = nrm(x); Object.assign(ly.vec, { mix: buf.att.slice(), out: buf.o.slice(), x1: x.slice() }); }
        rmsnorm(x, P.ln2, eps, buf.h);
        matvec(P.gate, F, D, buf.h, buf.g); matvec(P.up, F, D, buf.h, buf.u);
        if (full) Object.assign(ly.vec, { h2: buf.h.slice(), gate: buf.g.slice(), up: buf.u.slice() });
        for (let i = 0; i < F; i++) { const a = buf.g[i]; buf.g[i] = (a / (1 + Math.exp(-a))) * buf.u[i]; } // SwiGLU: silu(gate) * up
        matvec(P.down, D, F, buf.g, buf.d);
        for (let i = 0; i < D; i++) x[i] += buf.d[i];
        if (full) Object.assign(ly.vec, { act: buf.g.slice(), down: buf.d.slice(), xout: x.slice() });
        if (rec) {
          ly.attn = weights; ly.heads = H;
          if (!full) ly.first = Array.from(x.subarray(0, 16)); // every lane keeps its ball's first 16 numbers after each layer
          if (full) {
            const tl = performance.now();
            Object.assign(ly, { ffnIn: buf.h.slice(), neurons: topk(buf.g.map(Math.abs), 5).map(i => ({ i, v: buf.g[i] })), down: f6(buf.d), downNorm: nrm(buf.d),
              first: Array.from(x.subarray(0, 16)), norm: nrm(x), lens: l < L - 1 ? lens(x, lensTop) : null });
            if (watch && l < L - 1) ly.watch = watchOf(buf.logits, lastLse); // buf.logits holds this layer's lens scores
            lensMs += performance.now() - tl;
          }
          rec.layers.push(ly);
        }
        if (l % 6 === 5) await tick(); // keep the page responsive
      }
      rmsnorm(x, normW, eps, buf.hn); matvec(E, V, D, buf.hn, buf.logits);
      state.pos++;
      const logits = buf.logits.slice();
      if (full) { const { lse } = softmaxStats(logits); rec.layers[L - 1].lens = topk(logits, lensTop).map(i => ({ id: i, p: Math.exp(logits[i] - lse) })); if (watch) rec.layers[L - 1].watch = watchOf(logits, lse); }
      return { logits, rec, ms: performance.now() - t0 - lensMs };
    }
    const embedding = (id, n = 8) => Array.from(E.subarray(id * D, id * D + n));
    // Tokens whose embedding rows point the same way as this one (cosine similarity over all 576 numbers).
    let rowNorms = null;
    function neighbors(id, k = 5) {
      if (!rowNorms) { rowNorms = new Float32Array(V); for (let r = 0; r < V; r++) { let s = 0; const o = r * D; for (let c = 0; c < D; c++) s += E[o + c] * E[o + c]; rowNorms[r] = Math.sqrt(s) || 1; } }
      const sims = new Float32Array(V), o = id * D;
      for (let r = 0; r < V; r++) { let s = 0; const q = r * D; for (let c = 0; c < D; c++) s += E[o + c] * E[q + c]; sims[r] = s / (rowNorms[id] * rowNorms[r]); }
      sims[id] = -2;
      return topk(sims, k).map(i => ({ id: i, sim: sims[i] }));
    }
    const neuron = (l, n) => ({ gate: layers[l].gate.subarray(n * D, n * D + D), up: layers[l].up.subarray(n * D, n * D + D) }); // one feed-forward neuron's two weight rows
    return { cfg, tok, step, newState, setWatch, WATCH_FLOOR, lens, embedding, neighbors, neuron, mats: { E, layers, normW }, D, L, H, KVH, HD, F, V, kvBytesPerToken, weightBytes };
  }

  // Top-k sampling as the page teaches it: keep the top k scores, divide by the temperature, softmax, roll.
  function sampleTopK(logits, k, T, r) {
    const ids = topk(logits, k), { lse } = softmaxStats(logits);
    const full = ids.map(i => Math.exp(logits[i] - lse)); // share of the whole vocabulary, before cutting
    let p;
    if (T < 0.05) p = ids.map((_, i) => (i ? 0 : 1));
    else { const e = ids.map(i => Math.exp((logits[i] - logits[ids[0]]) / T)), s = e.reduce((a, b) => a + b, 0); p = e.map(v => v / s); }
    let pick = 0;
    if (T >= 0.05) { const u = r(); let acc = 0; pick = ids.length - 1; for (let i = 0; i < ids.length; i++) { acc += p[i]; if (u <= acc) { pick = i; break; } } }
    return { cands: ids.map((id, i) => ({ id, logit: logits[id], p: p[i], full: full[i] })), pick };
  }

  async function fetchWithProgress(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Download failed (${res.status}) for ${url}`);
    const total = +res.headers.get('content-length') || 0;
    if (!res.body || !total) return res.arrayBuffer();
    const reader = res.body.getReader(), out = new Uint8Array(total); let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; out.set(value, got); got += value.length; onProgress && onProgress(got, total); }
    return out.buffer;
  }

  async function load(key, onProgress) {
    const BASE = baseOf(key);
    const [cfg, tj] = await Promise.all([fetch(BASE + 'config.json').then(r => r.json()), fetch(BASE + 'tokenizer.json').then(r => r.json())]);
    const buf = await fetchWithProgress(BASE + 'model.safetensors', onProgress);
    onProgress && onProgress(-1, 0); // unpacking
    await tick();
    return withMeta(makeModel(cfg, parseSafetensors(buf), makeTokenizer(tj)), key);
  }
  function withMeta(m, key) { m.meta = Object.assign({ key }, MODELS[key]); m.name = m.meta.name; return m; }
  function fromBuffers(cfg, tj, buf, key) { return withMeta(makeModel(cfg, parseSafetensors(buf), makeTokenizer(tj)), key || 'smollm2-instruct'); }

  const api = { load, fromBuffers, makeTokenizer, sampleTopK, MODELS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.LLM = api;
})(typeof window !== 'undefined' ? window : globalThis);
