/* llm.js: real small language models running in plain JavaScript, no libraries.
   Four designs: Llama (SmolLM2-135M-Instruct, a chat model; TinyStories 15M), GPT-NeoX (Pythia-14M, with its
   training checkpoints), GPT-2 (DistilGPT-2) and Mamba (Mamba-130M, no attention at all). The weights are the
   published numbers (bf16, f16 or f32), widened to 32-bit floats, so the maths matches the reference implementation. Alongside the normal output it records what the page shows:
   each layer's attention weights, its most active feed-forward neurons, the token's numbers after the
   layer, and the "logit lens" (the word the model would pick if it stopped after that layer).
   Works in the browser (window.LLM) and in Node (module.exports) for testing. */
(function (root) {
  'use strict';
  // revisions are pinned, so the numbers never change under the page
  const MODELS = {
    'smollm2-instruct': { name: 'SmolLM2-135M-Instruct', repo: 'HuggingFaceTB/SmolLM2-135M-Instruct', rev: '12fd25f77366fa6b3b4b768ec3050bf629380bac', mb: 269, chat: true, bos: null, license: 'Apache 2.0, by Hugging Face', layers: 30, hidden: 576, ff: 1536, heads: 9, kvHeads: 3, vocab: 49152 },
    'tiny-stories': { name: 'TinyStories 15M', repo: 'nickypro/tinyllama-15M', rev: 'a97e01fa7d54c088df9dc1b53d581a863ce08cd6', mb: 30, chat: false, bos: 1, license: 'MIT, by Andrej Karpathy (llama2.c), converted by nickypro', layers: 6, hidden: 288, ff: 768, heads: 6, kvHeads: 6, vocab: 32000 },
    'pythia-14m': { name: 'Pythia-14M', repo: 'EleutherAI/pythia-14m', rev: 'cf967c0a9a04383db6f7b1108d86b2962634b4ac', mb: 28, chat: false, bos: null, license: 'Apache 2.0, by EleutherAI', layers: 6, hidden: 128, ff: 512, heads: 4, kvHeads: 4, vocab: 50304 },
    'distilgpt2': { name: 'DistilGPT-2', repo: 'distilbert/distilgpt2', rev: '2290a62682d06624634c1f46a6ad5be0f47f38aa', mb: 353, chat: false, bos: null, license: 'Apache 2.0, by Hugging Face (distilled from OpenAI\'s GPT-2)', layers: 6, hidden: 768, ff: 3072, heads: 12, kvHeads: 12, vocab: 50257 },
    'mamba-130m': { name: 'Mamba-130M', repo: 'state-spaces/mamba-130m-hf', rev: '1e76775f628fbf1350fbe4dbb3d971ba64af25a1', mb: 517, chat: false, bos: null, license: 'Apache 2.0 (the Mamba project, by Albert Gu and Tri Dao)', layers: 24, hidden: 768, ff: 1536, heads: 0, kvHeads: 0, vocab: 50280 },
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
    const pre = JSON.stringify(tj.pre_tokenizer || {}), digits = /"Digits"[^}]*"individual_digits":\s*true/.test(pre), nfc = JSON.stringify(tj.normalizer || '').includes('"NFC"');
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
    // added tokens (special ones like the chat template's <|im_start|>, and e.g. GPT-NeoX's runs of spaces) are matched
    // whole before the split, longest first
    const special = new Map(tj.added_tokens.filter(a => a.special).map(a => [a.content, a.id]));
    const added = new Map(tj.added_tokens.map(a => [a.content, a.id]));
    const specialRe = added.size ? new RegExp('(' + [...added.keys()].sort((x, y) => y.length - x.length).map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')') : null;
    function encode(text) {
      const ids = [];
      if (nfc) text = text.normalize('NFC');
      for (const chunk of specialRe ? text.split(specialRe) : [text]) {
        if (!chunk) continue;
        if (added.has(chunk)) { ids.push(added.get(chunk)); continue; }
        if (sp) {
          for (const p of bpe('\u2581' + chunk.replace(/ /g, '\u2581'))) {
            if (vocab[p] != null) ids.push(vocab[p]);
            else for (const b of utf8.encode(p)) ids.push(vocab['<0x' + b.toString(16).toUpperCase().padStart(2, '0') + '>']);
          }
          continue;
        }
        // pre-tokenizer: (for some models) every digit is its own piece; then the GPT-2 split, applied inside each piece
        const pieces = digits ? chunk.split(/(\p{N})/u).filter(s => s.length) : [chunk];
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
      if (name === '__metadata__' || /\.attn\.(masked_)?bias$/.test(name)) continue; // GPT-2 keeps its causal mask in the file
      const { dtype, shape, data_offsets: [a, b] } = header[name];
      if (dtype === 'F32') { W[name] = { data: (base + a) % 4 === 0 ? new Float32Array(buf, base + a, (b - a) / 4) : new Float32Array(buf.slice(base + a, base + b)), shape }; continue; }
      if (dtype !== 'BF16' && dtype !== 'F16') throw new Error('expected BF16, F16 or F32 weights, got ' + dtype);
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

  function layernorm(x, w, b, eps, out) { // LayerNorm: subtract the mean, divide by the spread, then scale and shift
    const n = x.length; let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
    let v = 0; for (let i = 0; i < n; i++) v += (x[i] - m) * (x[i] - m); v /= n;
    const k = 1 / Math.sqrt(v + eps);
    for (let i = 0; i < n; i++) out[i] = (x[i] - m) * k * w[i] + b[i];
    return out;
  }
  function addBias(out, b) { for (let i = 0; i < out.length; i++) out[i] += b[i]; return out; }
  function erf(x) { // erf to about 1e-7 (Numerical Recipes' Chebyshev fit of erfc)
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r : r - 1;
  }
  const geluExact = a => 0.5 * a * (1 + erf(a / Math.SQRT2));                                    // GPT-NeoX (Pythia)
  const geluTanh = a => 0.5 * a * (1 + Math.tanh(0.7978845608028654 * (a + 0.044715 * a * a * a))); // GPT-2 ("gelu_new")
  const silu = a => a / (1 + Math.exp(-a));
  const softplus = a => (a > 20 ? a : Math.log1p(Math.exp(a)));
  const transpose = (m, rows, cols) => { const t = new Float32Array(rows * cols); for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) t[c * rows + r] = m[r * cols + c]; return t; };

  // Attention for one token, shared by the transformer designs: each head compares its query with the key of every
  // earlier token (kept in the cache), softmax turns the scores into weights, and the values are blended with them.
  function attend(q, K, Vc, H, KVH, HD, weights, sc0, out) {
    const T = K.length, rep = H / KVH, sc = new Float32Array(T);
    for (let hh = 0; hh < H; hh++) {
      const kv = Math.floor(hh / rep), qo = hh * HD, ko = kv * HD;
      let mx = -Infinity;
      for (let j = 0; j < T; j++) { const kj = K[j]; let s = 0; for (let i = 0; i < HD; i++) s += q[qo + i] * kj[ko + i]; s /= Math.sqrt(HD); sc[j] = s; if (s > mx) mx = s; }
      if (sc0 && hh === 0) sc0.set(sc);
      let sum = 0; for (let j = 0; j < T; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
      for (let i = 0; i < HD; i++) out[qo + i] = 0;
      for (let j = 0; j < T; j++) { const w = sc[j] / sum, vj = Vc[j]; if (weights) weights[hh * T + j] = w; for (let i = 0; i < HD; i++) out[qo + i] += w * vj[ko + i]; }
    }
    return out;
  }
  // rotary position embedding (RoPE): rotate pairs of numbers by an angle that grows with the position. "halves" style
  // (Llama, GPT-NeoX): number i pairs with number i + n/2; GPT-NeoX only rotates the first rotary_pct of each head.
  function makeRope(HD, rotDims, base) {
    const inv = new Float32Array(rotDims / 2);
    for (let i = 0; i < rotDims / 2; i++) inv[i] = 1 / Math.pow(base, (2 * i) / rotDims);
    return (vec, heads, pos) => {
      for (let h = 0; h < heads; h++) {
        const o = h * HD, half = rotDims / 2;
        for (let i = 0; i < half; i++) { const a = pos * inv[i], c = Math.cos(a), s = Math.sin(a), x1 = vec[o + i], x2 = vec[o + i + half]; vec[o + i] = x1 * c - x2 * s; vec[o + i + half] = x2 * c + x1 * s; }
      }
    };
  }

  // ---------- the designs: each one says how a token enters, what one layer does, and how the ball is read out ----------
  // Every design fills the same record for the page: per layer the attention weights (none for Mamba), the sizes of
  // the changes it adds onto the ball, and (for full records) every whole vector of the layer in `vec`.
  function archLlama(cfg, W, g) { // Llama (SmolLM2, TinyStories): RMSNorm, RoPE on whole heads, GQA, SwiGLU, no biases
    const D = cfg.hidden_size, L = cfg.num_hidden_layers, H = cfg.num_attention_heads, KVH = cfg.num_key_value_heads || H, HD = D / H, F = cfg.intermediate_size, eps = cfg.rms_norm_eps;
    const E = g('model.embed_tokens.weight'), layers = [];
    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      layers.push({ ln1: g(p + 'input_layernorm.weight'), ln2: g(p + 'post_attention_layernorm.weight'),
        q: g(p + 'self_attn.q_proj.weight'), k: g(p + 'self_attn.k_proj.weight'), v: g(p + 'self_attn.v_proj.weight'), o: g(p + 'self_attn.o_proj.weight'),
        gate: g(p + 'mlp.gate_proj.weight'), up: g(p + 'mlp.up_proj.weight'), down: g(p + 'mlp.down_proj.weight') });
    }
    const normW = g('model.norm.weight'), rope = makeRope(HD, HD, cfg.rope_theta);
    const b = { h: new Float32Array(D), q: new Float32Array(D), k: new Float32Array(KVH * HD), v: new Float32Array(KVH * HD), att: new Float32Array(D), o: new Float32Array(D), g: new Float32Array(F), u: new Float32Array(F), d: new Float32Array(D) };
    return { kind: 'llama', D, L, H, KVH, HD, F, E, Eout: E, layers, normW, kvPerToken: 2 * L * KVH * HD * 2,
      embed: (x, id) => x.set(E.subarray(id * D, id * D + D)),
      finalNorm: (x, out) => rmsnorm(x, normW, eps, out),
      newLayerState: () => ({ K: [], Vc: [] }),
      block(l, x, st, pos, ly, full, weights, H2) {
        const P = layers[l];
        rmsnorm(x, P.ln1, eps, b.h);
        matvec(P.q, D, D, b.h, b.q); matvec(P.k, KVH * HD, D, b.h, b.k); matvec(P.v, KVH * HD, D, b.h, b.v);
        if (full) Object.assign(ly.vec, { h1: b.h.slice(), qPre: b.q.slice(), kPre: b.k.slice() });
        rope(b.q, H, pos); rope(b.k, KVH, pos);
        if (full) Object.assign(ly.vec, { q: b.q.slice(), k: b.k.slice(), v: b.v.slice() });
        st.K.push(b.k.slice()); st.Vc.push(b.v.slice());
        if (full) ly.sc = new Float32Array(st.K.length);
        attend(b.q, st.K, st.Vc, H, KVH, HD, weights, full && ly.sc, b.att);
        matvec(P.o, D, D, b.att, b.o);
        for (let i = 0; i < D; i++) x[i] += b.o[i];
        if (full) { ly.outNorm = H2(b.o); ly.x1Norm = H2(x); Object.assign(ly.vec, { mix: b.att.slice(), out: b.o.slice(), x1: x.slice() }); }
        rmsnorm(x, P.ln2, eps, b.h);
        matvec(P.gate, F, D, b.h, b.g); matvec(P.up, F, D, b.h, b.u);
        if (full) Object.assign(ly.vec, { h2: b.h.slice(), gate: b.g.slice(), up: b.u.slice() });
        for (let i = 0; i < F; i++) b.g[i] = silu(b.g[i]) * b.u[i]; // SwiGLU: silu(gate) * up
        matvec(P.down, D, F, b.g, b.d);
        for (let i = 0; i < D; i++) x[i] += b.d[i];
        if (full) { Object.assign(ly.vec, { act: b.g.slice(), down: b.d.slice(), xout: x.slice() }); ly.downNorm = H2(b.d); ly.ffnIn = b.h.slice(); ly.neurons = topk(b.g.map(Math.abs), 5).map(i => ({ i, v: b.g[i] })); }
      },
      neuron: (l, n) => ({ gate: layers[l].gate.subarray(n * D, n * D + D), up: layers[l].up.subarray(n * D, n * D + D) }),
    };
  }
  function archNeox(cfg, W, g) { // GPT-NeoX (Pythia): LayerNorm with bias, RoPE on a quarter of each head, attention and feed-forward side by side
    const D = cfg.hidden_size, L = cfg.num_hidden_layers, H = cfg.num_attention_heads, HD = D / H, F = cfg.intermediate_size, eps = cfg.layer_norm_eps;
    const rot = Math.floor(HD * (cfg.rotary_pct != null ? cfg.rotary_pct : cfg.partial_rotary_factor)), parallel = cfg.use_parallel_residual !== false;
    const E = g('gpt_neox.embed_in.weight'), Eout = g('embed_out.weight'), layers = [];
    for (let l = 0; l < L; l++) {
      const p = `gpt_neox.layers.${l}.`;
      layers.push({ ln1: g(p + 'input_layernorm.weight'), ln1b: g(p + 'input_layernorm.bias'), ln2: g(p + 'post_attention_layernorm.weight'), ln2b: g(p + 'post_attention_layernorm.bias'),
        qkv: g(p + 'attention.query_key_value.weight'), qkvb: g(p + 'attention.query_key_value.bias'), o: g(p + 'attention.dense.weight'), ob: g(p + 'attention.dense.bias'),
        fc: g(p + 'mlp.dense_h_to_4h.weight'), fcb: g(p + 'mlp.dense_h_to_4h.bias'), proj: g(p + 'mlp.dense_4h_to_h.weight'), projb: g(p + 'mlp.dense_4h_to_h.bias') });
    }
    const normW = g('gpt_neox.final_layer_norm.weight'), normB = g('gpt_neox.final_layer_norm.bias'), rope = makeRope(HD, rot, cfg.rotary_emb_base || cfg.rope_theta || 10000);
    const b = { h: new Float32Array(D), h2: new Float32Array(D), qkv: new Float32Array(3 * D), q: new Float32Array(D), k: new Float32Array(D), v: new Float32Array(D), att: new Float32Array(D), o: new Float32Array(D), f: new Float32Array(F), d: new Float32Array(D), xin: new Float32Array(D) };
    return { kind: 'neox', parallel, D, L, H, KVH: H, HD, F, rot, E, Eout, layers, normW, kvPerToken: 2 * L * H * HD * 2,
      embed: (x, id) => x.set(E.subarray(id * D, id * D + D)),
      finalNorm: (x, out) => layernorm(x, normW, normB, eps, out),
      newLayerState: () => ({ K: [], Vc: [] }),
      block(l, x, st, pos, ly, full, weights, H2) {
        const P = layers[l];
        b.xin.set(x);
        layernorm(x, P.ln1, P.ln1b, eps, b.h);
        addBias(matvec(P.qkv, 3 * D, D, b.h, b.qkv), P.qkvb);
        for (let h = 0; h < H; h++) for (let i = 0; i < HD; i++) { // the fused grid is laid out per head: [query | key | value]
          const o = h * 3 * HD; b.q[h * HD + i] = b.qkv[o + i]; b.k[h * HD + i] = b.qkv[o + HD + i]; b.v[h * HD + i] = b.qkv[o + 2 * HD + i];
        }
        if (full) Object.assign(ly.vec, { h1: b.h.slice(), qkv: b.qkv.slice(), qPre: b.q.slice(), kPre: b.k.slice() });
        rope(b.q, H, pos); rope(b.k, H, pos);
        if (full) Object.assign(ly.vec, { q: b.q.slice(), k: b.k.slice(), v: b.v.slice() });
        st.K.push(b.k.slice()); st.Vc.push(b.v.slice());
        if (full) ly.sc = new Float32Array(st.K.length);
        attend(b.q, st.K, st.Vc, H, H, HD, weights, full && ly.sc, b.att);
        addBias(matvec(P.o, D, D, b.att, b.o), P.ob);
        // parallel: the feed-forward block reads the same incoming ball, not the ball plus the attention change
        if (!parallel) for (let i = 0; i < D; i++) x[i] += b.o[i];
        layernorm(parallel ? b.xin : x, P.ln2, P.ln2b, eps, b.h2);
        addBias(matvec(P.fc, F, D, b.h2, b.f), P.fcb);
        if (full) Object.assign(ly.vec, { mix: b.att.slice(), out: b.o.slice(), h2: b.h2.slice(), pre: b.f.slice() });
        for (let i = 0; i < F; i++) b.f[i] = geluExact(b.f[i]);
        addBias(matvec(P.proj, D, F, b.f, b.d), P.projb);
        for (let i = 0; i < D; i++) x[i] += (parallel ? b.o[i] : 0) + b.d[i];
        if (full) { ly.outNorm = H2(b.o); ly.downNorm = H2(b.d); ly.x1Norm = H2(b.xin.map((v, i) => v + b.o[i])); Object.assign(ly.vec, { act: b.f.slice(), down: b.d.slice(), xout: x.slice() }); ly.ffnIn = b.h2.slice(); ly.neurons = topk(b.f.map(Math.abs), 5).map(i => ({ i, v: b.f[i] })); }
      },
      neuron: (l, n) => ({ fc: layers[l].fc.subarray(n * D, n * D + D), bias: layers[l].fcb[n], act: geluExact }),
    };
  }
  function archGpt2(cfg, W, g) { // GPT-2 (DistilGPT-2): a learned position table added at the start, LayerNorm, GELU, biases
    const D = cfg.n_embd, L = cfg.n_layer, H = cfg.n_head, HD = D / H, F = cfg.n_inner || 4 * D, eps = cfg.layer_norm_epsilon;
    const E = g('transformer.wte.weight'), Pos = g('transformer.wpe.weight'), layers = [];
    for (let l = 0; l < L; l++) { // GPT-2 stores its grids as [in, out] ("Conv1D"); turn them into [out, in] like the others
      const p = `transformer.h.${l}.`;
      layers.push({ ln1: g(p + 'ln_1.weight'), ln1b: g(p + 'ln_1.bias'), ln2: g(p + 'ln_2.weight'), ln2b: g(p + 'ln_2.bias'),
        qkv: transpose(g(p + 'attn.c_attn.weight'), D, 3 * D), qkvb: g(p + 'attn.c_attn.bias'), o: transpose(g(p + 'attn.c_proj.weight'), D, D), ob: g(p + 'attn.c_proj.bias'),
        fc: transpose(g(p + 'mlp.c_fc.weight'), D, F), fcb: g(p + 'mlp.c_fc.bias'), proj: transpose(g(p + 'mlp.c_proj.weight'), F, D), projb: g(p + 'mlp.c_proj.bias') });
    }
    const normW = g('transformer.ln_f.weight'), normB = g('transformer.ln_f.bias');
    const b = { h: new Float32Array(D), qkv: new Float32Array(3 * D), att: new Float32Array(D), o: new Float32Array(D), f: new Float32Array(F), d: new Float32Array(D) };
    return { kind: 'gpt2', D, L, H, KVH: H, HD, F, E, Eout: E, Pos, layers, normW, kvPerToken: 2 * L * H * HD * 2,
      embed: (x, id, pos) => { for (let i = 0; i < D; i++) x[i] = E[id * D + i] + Pos[pos * D + i]; }, // token row + position row
      finalNorm: (x, out) => layernorm(x, normW, normB, eps, out),
      newLayerState: () => ({ K: [], Vc: [] }),
      block(l, x, st, pos, ly, full, weights, H2) {
        const P = layers[l];
        layernorm(x, P.ln1, P.ln1b, eps, b.h);
        addBias(matvec(P.qkv, 3 * D, D, b.h, b.qkv), P.qkvb);
        const q = b.qkv.subarray(0, D), k = b.qkv.slice(D, 2 * D), v = b.qkv.slice(2 * D, 3 * D); // the fused grid: [all queries | all keys | all values]
        if (full) Object.assign(ly.vec, { h1: b.h.slice(), qkv: b.qkv.slice(), q: q.slice(), k: k.slice(), v: v.slice() });
        st.K.push(k); st.Vc.push(v);
        if (full) ly.sc = new Float32Array(st.K.length);
        attend(q, st.K, st.Vc, H, H, HD, weights, full && ly.sc, b.att);
        addBias(matvec(P.o, D, D, b.att, b.o), P.ob);
        for (let i = 0; i < D; i++) x[i] += b.o[i];
        if (full) { ly.outNorm = H2(b.o); ly.x1Norm = H2(x); Object.assign(ly.vec, { mix: b.att.slice(), out: b.o.slice(), x1: x.slice() }); }
        layernorm(x, P.ln2, P.ln2b, eps, b.h);
        addBias(matvec(P.fc, F, D, b.h, b.f), P.fcb);
        if (full) Object.assign(ly.vec, { h2: b.h.slice(), pre: b.f.slice() });
        for (let i = 0; i < F; i++) b.f[i] = geluTanh(b.f[i]);
        addBias(matvec(P.proj, D, F, b.f, b.d), P.projb);
        for (let i = 0; i < D; i++) x[i] += b.d[i];
        if (full) { ly.downNorm = H2(b.d); Object.assign(ly.vec, { act: b.f.slice(), down: b.d.slice(), xout: x.slice() }); ly.ffnIn = b.h.slice(); ly.neurons = topk(b.f.map(Math.abs), 5).map(i => ({ i, v: b.f[i] })); }
      },
      neuron: (l, n) => ({ fc: layers[l].fc.subarray(n * D, n * D + D), bias: layers[l].fcb[n], act: geluTanh }),
    };
  }
  function archMamba(cfg, W, g) { // Mamba: no attention; each layer keeps a fixed-size memory (a selective state space) per channel
    const D = cfg.hidden_size, L = cfg.num_hidden_layers, I = cfg.intermediate_size, N = cfg.state_size, R = cfg.time_step_rank === 'auto' ? Math.ceil(D / 16) : cfg.time_step_rank, KC = cfg.conv_kernel, eps = cfg.layer_norm_epsilon;
    const E = g('backbone.embeddings.weight'), layers = [];
    for (let l = 0; l < L; l++) {
      const p = `backbone.layers.${l}.`, Alog = g(p + 'mixer.A_log');
      layers.push({ ln: g(p + 'norm.weight'), inp: g(p + 'mixer.in_proj.weight'), conv: g(p + 'mixer.conv1d.weight'), convb: g(p + 'mixer.conv1d.bias'),
        xp: g(p + 'mixer.x_proj.weight'), dt: g(p + 'mixer.dt_proj.weight'), dtb: g(p + 'mixer.dt_proj.bias'), Alog, A: Alog.map(v => -Math.exp(v)), Dv: g(p + 'mixer.D'), out: g(p + 'mixer.out_proj.weight') });
    }
    const normW = g('backbone.norm_f.weight');
    const b = { h: new Float32Array(D), xz: new Float32Array(2 * I), u: new Float32Array(I), c: new Float32Array(I), p: new Float32Array(R + 2 * N), dt: new Float32Array(I), y: new Float32Array(I), o: new Float32Array(D) };
    return { kind: 'mamba', D, L, H: 0, KVH: 0, HD: 0, F: I, I, N, R, KC, E, Eout: E, layers, normW, kvPerToken: 0, stateBytes: L * I * (N + KC - 1) * 4,
      embed: (x, id) => x.set(E.subarray(id * D, id * D + D)),
      finalNorm: (x, out) => rmsnorm(x, normW, eps, out),
      newLayerState: () => ({ conv: new Float32Array(I * (KC - 1)), ssm: new Float32Array(I * N) }), // the memory: fixed size, however long the text
      block(l, x, st, pos, ly, full, weights, H2) {
        const P = layers[l];
        rmsnorm(x, P.ln, eps, b.h);
        matvec(P.inp, 2 * I, D, b.h, b.xz); // two halves: x (goes through the memory) and z (the gate)
        // a short causal convolution over the last KC tokens of each channel (the previous KC-1 are kept in the state)
        for (let c = 0; c < I; c++) {
          let s = P.convb[c];
          for (let j = 0; j < KC - 1; j++) s += P.conv[c * KC + j] * st.conv[c * (KC - 1) + j];
          s += P.conv[c * KC + KC - 1] * b.xz[c];
          for (let j = 0; j < KC - 2; j++) st.conv[c * (KC - 1) + j] = st.conv[c * (KC - 1) + j + 1];
          st.conv[c * (KC - 1) + KC - 2] = b.xz[c];
          b.c[c] = s; b.u[c] = silu(s);
        }
        matvec(P.xp, R + 2 * N, I, b.u, b.p); // per token: the step size (input to dt), B (how to write) and C (how to read)
        const Bv = b.p.subarray(R, R + N), Cv = b.p.subarray(R + N, R + 2 * N);
        matvec(P.dt, I, R, b.p.subarray(0, R), b.dt);
        let wrote = 0;
        for (let c = 0; c < I; c++) {
          const dt = softplus(b.dt[c] + P.dtb[c]), u = b.u[c]; b.dt[c] = dt;
          let y = 0;
          for (let n = 0; n < N; n++) { // each channel's memory of N numbers: fade by exp(A·dt), write dt·B·u, read with C
            const i = c * N + n, prev = st.ssm[i], s = Math.exp(P.A[i] * dt) * prev + dt * Bv[n] * u;
            st.ssm[i] = s; y += s * Cv[n]; wrote += (s - prev) * (s - prev);
          }
          b.y[c] = y + P.Dv[c] * u;
        }
        if (full) Object.assign(ly.vec, { h1: b.h.slice(), xs: b.xz.slice(0, I), z: b.xz.slice(I), conv: b.c.slice(), u: b.u.slice(), B: Bv.slice(), C: Cv.slice(), dt: b.dt.slice(), scan: b.y.slice() });
        for (let c = 0; c < I; c++) b.y[c] *= silu(b.xz[I + c]); // the gate
        matvec(P.out, D, I, b.y, b.o);
        for (let i = 0; i < D; i++) x[i] += b.o[i];
        ly.memWrite = Math.sqrt(wrote); // how much this token changed the layer's memory (every record keeps it)
        if (full) { ly.outNorm = H2(b.o); ly.downNorm = 0; ly.memNorm = H2(st.ssm); Object.assign(ly.vec, { gated: b.y.slice(), out: b.o.slice(), xout: x.slice() }); ly.ffnIn = b.h.slice(); ly.neurons = topk(b.y.map(Math.abs), 5).map(i => ({ i, v: b.y[i] })); }
      },
      neuron: null,
    };
  }
  function arch(cfg, W) {
    const g = n => { if (!W[n]) throw new Error('missing weight ' + n); return W[n].data; };
    const t = cfg.model_type;
    return t === 'gpt_neox' ? archNeox(cfg, W, g) : t === 'gpt2' ? archGpt2(cfg, W, g) : t === 'mamba' ? archMamba(cfg, W, g) : archLlama(cfg, W, g);
  }

  function makeModel(cfg, W, tok) {
    const A = arch(cfg, W), { D, L, H, KVH, HD, F, E, Eout } = A, V = cfg.vocab_size;
    const weightBytes = Object.values(W).reduce((a, t) => a + t.data.length * 2, 0); // as 16-bit numbers, the usual size to run in
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
    function newState() { return { pos: 0, layer: Array.from({ length: L }, () => A.newLayerState()) }; }
    const buf = { x: new Float32Array(D), logits: new Float32Array(V), hn: new Float32Array(D) };
    function lens(x, n) { // the logit lens: the final norm and the output table applied to an intermediate ball
      A.finalNorm(x, buf.hn); matvec(Eout, V, D, buf.hn, buf.logits);
      const { lse } = softmaxStats(buf.logits); lastLse = lse;
      return topk(buf.logits, n).map(id => ({ id, p: Math.exp(buf.logits[id] - lse) }));
    }
    const nrm = a => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };
    // Run one token through every layer. record=true returns what every sub-step of every layer did;
    // record='attn' keeps only the attention weights and the ball's first numbers (cheap: no logit lens).
    async function step(state, id, record, lensTop = 3) {
      const t0 = performance.now(), pos = state.pos, x = buf.x, full = record === true;
      A.embed(x, id, pos);
      const rec = record ? { id, pos, embed: Array.from(x.subarray(0, 8)), layers: [] } : null;
      let lensMs = 0;
      for (let l = 0; l < L; l++) {
        const ly = full ? { xinNorm: nrm(x), vec: { xin: x.slice() } } : {};
        const weights = record && H ? new Float32Array(H * (pos + 1)) : null;
        A.block(l, x, state.layer[l], pos, ly, full, weights, nrm);
        if (rec) {
          ly.attn = weights; ly.heads = H;
          ly.first = Array.from(x.subarray(0, 16)); // every lane keeps its ball's first 16 numbers after each layer
          if (full) {
            const tl = performance.now();
            Object.assign(ly, { norm: nrm(x), lens: l < L - 1 ? lens(x, lensTop) : null });
            if (watch && l < L - 1) ly.watch = watchOf(buf.logits, lastLse); // buf.logits holds this layer's lens scores
            lensMs += performance.now() - tl;
          }
          rec.layers.push(ly);
        }
        if (l % 6 === 5) await tick(); // keep the page responsive
      }
      A.finalNorm(x, buf.hn); matvec(Eout, V, D, buf.hn, buf.logits);
      state.pos++;
      const logits = buf.logits.slice();
      if (full) { const { lse } = softmaxStats(logits); rec.layers[L - 1].lens = topk(logits, lensTop).map(i => ({ id: i, p: Math.exp(logits[i] - lse) })); if (watch) rec.layers[L - 1].watch = watchOf(logits, lse); }
      return { logits, rec, ms: performance.now() - t0 - lensMs };
    }
    const embedding = (id, n = 8) => Array.from(E.subarray(id * D, id * D + n));
    // Tokens whose embedding rows point the same way as this one (cosine similarity over all the row's numbers).
    let rowNorms = null;
    function neighbors(id, k = 5) {
      if (!rowNorms) { rowNorms = new Float32Array(V); for (let r = 0; r < V; r++) { let s = 0; const o = r * D; for (let c = 0; c < D; c++) s += E[o + c] * E[o + c]; rowNorms[r] = Math.sqrt(s) || 1; } }
      const sims = new Float32Array(V), o = id * D;
      for (let r = 0; r < V; r++) { let s = 0; const q = r * D; for (let c = 0; c < D; c++) s += E[o + c] * E[q + c]; sims[r] = s / (rowNorms[id] * rowNorms[r]); }
      sims[id] = -2;
      return topk(sims, k).map(i => ({ id: i, sim: sims[i] }));
    }
    return { cfg, tok, step, newState, setWatch, WATCH_FLOOR, lens, embedding, neighbors, neuron: A.neuron, arch: A, kind: A.kind,
      mats: { E, Eout, layers: A.layers, normW: A.normW, Pos: A.Pos }, D, L, H, KVH, HD, F, V, kvBytesPerToken: A.kvPerToken, stateBytes: A.stateBytes || 0, weightBytes };
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

  async function load(key, onProgress, rev) {
    const BASE = rev ? (root.LLM_BASE && root.LLM_BASE[key + '@' + rev]) || `https://huggingface.co/${MODELS[key].repo}/resolve/${rev}/` : baseOf(key);
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
