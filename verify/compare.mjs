// Checks llm.js against the numbers verify/reference.py wrote for one model, running the prompt the way the pages do
// (prefill, then one recorded step). --no-wasm switches off the WebAssembly SIMD kernel to check the plain JavaScript one.
// With a float64 reference, --yardstick=<float32 reference> also measures transformers' own float32 run against it: a
// number passes when llm.js is within the fixed tolerance, or no further from the 64-bit answer than twice what
// transformers itself gets in float32 (sharp attention heads magnify float32 rounding in every implementation).
// usage: node verify/compare.mjs <model folder> <reference.json> <model key> [--no-wasm] [--yardstick=<float32 reference.json>] [--perturb]
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
if (process.argv.includes('--no-wasm')) delete globalThis.WebAssembly;
const LLM = createRequire(import.meta.url)('../llm.js');
const [dir, refPath, key] = process.argv.slice(2).filter(a => !a.startsWith('--')), ref = JSON.parse(readFileSync(refPath, 'utf8'));
const ysArg = process.argv.find(a => a.startsWith('--yardstick=')), ys = ysArg ? JSON.parse(readFileSync(ysArg.slice(12), 'utf8')) : null;
const buf = readFileSync(path.join(dir, 'model.safetensors'));
const m = LLM.fromBuffers(JSON.parse(readFileSync(path.join(dir, 'config.json'), 'utf8')), JSON.parse(readFileSync(path.join(dir, 'tokenizer.json'), 'utf8')),
  buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length), key);
// --perturb: the self-test. One knob in the middle layer is nudged by 5% of its grid's largest knob; checks must then fail.
if (process.argv.includes('--perturb')) { const g = Object.values(m.mats.layers[m.L >> 1]).find(t => t && t.length > 4096); let mx = 0; for (const v of g) mx = Math.max(mx, Math.abs(v)); g[1234] += 0.05 * mx; console.log(`SELF-TEST: one knob in layer ${(m.L >> 1) + 1} nudged by 5% of its grid's largest knob; checks below should FAIL`); }
const same = (a, b) => JSON.stringify(Array.from(a)) === JSON.stringify(Array.from(b)), rows = [];
const check = (name, ok, detail) => { rows.push(ok); console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}: ${detail}`); };
// a measured difference: with a float64 reference and the float32 yardstick, a number passes when llm.js is no further from the 64-bit answer
// than 3 times transformers' own float32 rounding (plus 1e-7 for numbers both get exactly). A float32-vs-float64 gap of
// 5e-4 or more means a broken reference. Without a yardstick: the fixed tolerance.
const within = (name, d, tol, y, what) => {
  const ok = y != null ? y < 5e-4 && d <= 3 * y + 1e-7 : d < tol;
  check(name, ok, `${what}; largest difference ${d.toExponential(1)}` + (y != null ? ` (transformers' own float32: ${y.toExponential(1)}${y >= 5e-4 ? ', too far: check the references' : ''})` : ''));
};
console.log(`${key} (${LLM.MODELS[key].repo}@${LLM.MODELS[key].rev.slice(0, 7)}) vs transformers ${ref.versions.transformers} in ${ref.dtype || 'float32'}, torch ${ref.versions.torch}${process.argv.includes('--no-wasm') ? ', plain JavaScript (no WebAssembly)' : ''}`);
// 1. tokenizer
const bos = m.meta.bos != null ? [m.meta.bos] : [];
const tokOk = ref.texts.map((t, i) => same(bos.concat(m.tok.encode(t)), ref.tok[i]));
check('tokens', tokOk.every(Boolean), `${tokOk.filter(Boolean).length}/${tokOk.length} tricky texts identical`);
const ids = (ref.chat ? [] : bos).concat(m.tok.encode(ref.text));
check('prompt tokens', same(ids, ref.ids), `${ids.length} tokens${ref.chat ? ' (chat template)' : ''}`);
// differences: absolute, or relative to the reference's largest number
const absd = (x, r) => { let d = 0; for (let i = 0; i < r.length; i++) d = Math.max(d, Math.abs(x[i] - r[i])); return d; };
const reld = (x, r) => { let s = 0; for (let i = 0; i < r.length; i++) s = Math.max(s, Math.abs(r[i])); return absd(x, r) / s; };
const over = (L, f) => { let w = 0; for (let l = 0; l < L; l++) w = Math.max(w, f(l)); return w; };
const yard = f => (ys ? over(m.L, f) : null);
// 2. the prompt, as the pages run it
const st = m.newState(); await m.prefill(st, ref.ids.slice(0, -1)); let res = await m.step(st, ref.ids[ref.ids.length - 1], true, 3);
const E = res.rec.layers, R = ref.layers, Y = ys && ys.layers, flatAtt = a => a.flat();
within('ball after every layer', over(m.L, l => reld(E[l].vec.xout, R[l].x)), 1e-4, yard(l => reld(Y[l].x, R[l].x)), `all ${m.D} numbers of the last token in all ${m.L} layers, as a share of the layer's largest number`);
const lensSame = E.filter((ly, l) => ly.lens[0].id === R[l].top).length;
check('logit lens', lensSame === m.L, `same top token after ${lensSame}/${m.L} layers`);
const top = Array.from(res.logits.keys()).sort((a, b) => res.logits[b] - res.logits[a]).slice(0, 10);
check('top-10 tokens', same(top, ref.top10.map(t => t[0])), 'the same 10 tokens in the same order');
const ysTop = ys ? Object.fromEntries(ys.top10) : null;
within('top-10 scores', Math.max(...ref.top10.map(([i, v]) => Math.abs(res.logits[i] - v))), 1e-3, ys && ref.top10.every(([i]) => i in ysTop) ? Math.max(...ref.top10.map(([i, v]) => Math.abs(ysTop[i] - v))) : null, `scores around ${ref.top10[0][1].toFixed(1)}`);
// 3. the trace inside every layer (last token): every head's attention weights, the attention block's output, the
// feed-forward block's output (Mamba: the mixer's output)
if (R[0].att) within('attention weights', over(m.L, l => absd(E[l].attn, flatAtt(R[l].att))), 1e-5, yard(l => absd(flatAtt(Y[l].att), flatAtt(R[l].att))), `every head in every layer (${R.reduce((a, r) => a + r.att.length, 0)} heads), last token over all ${ref.ids.length} positions (weights add up to 1)`);
within(m.kind === 'mamba' ? 'mixer output' : 'attention block output', over(m.L, l => reld(E[l].vec.out, R[l].attOut)), 1e-4, yard(l => reld(Y[l].attOut, R[l].attOut)), 'every layer, as a share of the largest number');
if (R[0].ffOut) within('feed-forward block output', over(m.L, l => reld(E[l].vec.down, R[l].ffOut)), 1e-4, yard(l => reld(Y[l].ffOut, R[l].ffOut)), 'every layer, as a share of the largest number');
// 4. every token's ball after every layer, each token run as its own recorded step
{ const st2 = m.newState(), all = [];
  for (let i = 0; i < ref.ids.length; i++) { const r2 = await m.step(st2, ref.ids[i], true, 1); all.push(r2.rec.layers.map(ly => ly.vec.xout)); }
  within('every token after every layer', over(m.L, l => Math.max(...all.map((xs, i) => reld(xs[l], R[l].all[i])))), 1e-4, yard(l => Math.max(...R[l].all.map((r, i) => reld(Y[l].all[i], r)))), `${ref.ids.length} tokens × ${m.L} layers × ${m.D} numbers, as a share of the layer's largest number`); }
// 5. plain greedy continuation, one step at a time
const gen = [];
for (let n = 0; n < ref.greedy.length; n++) { let b = 0; for (let i = 1; i < res.logits.length; i++) if (res.logits[i] > res.logits[b]) b = i; gen.push(b); if (n < ref.greedy.length - 1) res = await m.step(st, b, false); }
check('greedy continuation', same(gen, ref.greedy), `${ref.greedy.length} tokens: ${JSON.stringify(m.tok.decode(gen))}`);
console.log(rows.every(Boolean) ? 'ALL PASS' : 'SOME CHECKS FAILED');
process.exitCode = rows.every(Boolean) ? 0 : 1;
