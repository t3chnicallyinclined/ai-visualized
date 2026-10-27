// Checks llm.js against the numbers verify/reference.py wrote for one model, running the prompt the way the pages do
// (prefill, then one recorded step). --no-wasm switches off the WebAssembly SIMD kernel to check the plain JavaScript one.
// usage: node verify/compare.mjs <model folder> <reference.json> <model key> [--no-wasm]
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
if (process.argv.includes('--no-wasm')) delete globalThis.WebAssembly;
const LLM = createRequire(import.meta.url)('../llm.js');
const [dir, refPath, key] = process.argv.slice(2).filter(a => !a.startsWith('--')), ref = JSON.parse(readFileSync(refPath, 'utf8'));
const buf = readFileSync(path.join(dir, 'model.safetensors'));
const m = LLM.fromBuffers(JSON.parse(readFileSync(path.join(dir, 'config.json'), 'utf8')), JSON.parse(readFileSync(path.join(dir, 'tokenizer.json'), 'utf8')),
  buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length), key);
const same = (a, b) => JSON.stringify(Array.from(a)) === JSON.stringify(Array.from(b)), rows = [];
const check = (name, ok, detail) => { rows.push(ok); console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}: ${detail}`); };
console.log(`${key} (${LLM.MODELS ? LLM.MODELS[key].repo + '@' + LLM.MODELS[key].rev.slice(0, 7) : key}) vs transformers ${ref.versions.transformers}, torch ${ref.versions.torch}${process.argv.includes('--no-wasm') ? ', plain JavaScript (no WebAssembly)' : ''}`);
// 1. tokenizer
const bos = m.meta.bos != null ? [m.meta.bos] : [];
const tokOk = ref.texts.map((t, i) => same(bos.concat(m.tok.encode(t)), ref.tok[i]));
check('tokens', tokOk.every(Boolean), `${tokOk.filter(Boolean).length}/${tokOk.length} tricky texts identical`);
const ids = (ref.chat ? [] : bos).concat(m.tok.encode(ref.text));
check('prompt tokens', same(ids, ref.ids), `${ids.length} tokens${ref.chat ? ' (chat template)' : ''}`);
// 2. the prompt, as the pages run it
const st = m.newState(); await m.prefill(st, ref.ids.slice(0, -1)); let res = await m.step(st, ref.ids[ref.ids.length - 1], true, 3);
let worst = 0, lensSame = 0;
res.rec.layers.forEach((ly, l) => {
  const r = ref.layers[l].x, x = ly.vec.xout; let d = 0, s = 0;
  for (let i = 0; i < r.length; i++) { d = Math.max(d, Math.abs(x[i] - r[i])); s = Math.max(s, Math.abs(r[i])); }
  worst = Math.max(worst, d / s); if (ly.lens[0].id === ref.layers[l].top) lensSame++;
});
check('ball after every layer', worst < 1e-4, `all ${m.D} numbers of the last token in all ${m.L} layers; largest difference ${worst.toExponential(1)} of the layer's largest number`);
check('logit lens', lensSame === m.L, `same top token after ${lensSame}/${m.L} layers`);
const top = Array.from(res.logits.keys()).sort((a, b) => res.logits[b] - res.logits[a]).slice(0, 10);
const dl = Math.max(...ref.top10.map(([i, v]) => Math.abs(res.logits[i] - v)));
check('top-10 scores', same(top, ref.top10.map(t => t[0])) && dl < 1e-3, `same 10 tokens in the same order; largest score difference ${dl.toExponential(1)} (scores around ${ref.top10[0][1].toFixed(1)})`);
// 3. plain greedy continuation, one step at a time
const gen = [];
for (let n = 0; n < ref.greedy.length; n++) { let b = 0; for (let i = 1; i < res.logits.length; i++) if (res.logits[i] > res.logits[b]) b = i; gen.push(b); if (n < ref.greedy.length - 1) res = await m.step(st, b, false); }
check('greedy continuation', same(gen, ref.greedy), `${ref.greedy.length} tokens: ${JSON.stringify(m.tok.decode(gen))}`);
console.log(rows.every(Boolean) ? 'ALL PASS' : 'SOME CHECKS FAILED');
process.exitCode = rows.every(Boolean) ? 0 : 1;
