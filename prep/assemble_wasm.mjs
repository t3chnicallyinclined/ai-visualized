// Assembles prep/matvec.wat and prints its bytes as base64, for the WASM constant in llm.js.
// Needs wabt (npm install wabt@1.0.36). Usage: node prep/assemble_wasm.mjs
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url), wabt = await require('wabt')();
const src = readFileSync(new URL('./matvec.wat', import.meta.url), 'utf8');
const mod = wabt.parseWat('matvec.wat', src, { simd: true });
const { buffer } = mod.toBinary({});
console.log(Buffer.from(buffer).toString('base64'));
