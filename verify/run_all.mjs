// Runs every check: downloads the pinned models, writes transformers' reference numbers in float32 and float64,
// compares llm.js with them (with and without WebAssembly), runs the self-test (one knob nudged: the checks must fail),
// and checks the agent loop. Needs Python with torch and transformers, and about 6 GB of free memory for Qwen in float64.
// usage: node verify/run_all.mjs <folder for the models and results> [python command, default: python]
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url)), { MODELS } = createRequire(import.meta.url)('../llm.js');
const [dir, py = 'python'] = process.argv.slice(2);
if (!dir) { console.error('usage: node verify/run_all.mjs <folder> [python]'); process.exit(1); }
const run = (cmd, args) => { const r = spawnSync(cmd, args, { stdio: 'inherit' }); return r.status === 0; };
const ok = [run('node', [path.join(here, 'get_models.mjs'), dir])];
for (const [key, m] of Object.entries(MODELS)) {
  const md = path.join(dir, key), r32 = path.join(dir, key + '.float32.json'), r64 = path.join(dir, key + '.float64.json');
  const prompt = m.chat ? 'What is a fireball?' : 'Once upon a time, there was a little', cmp = [path.join(here, 'compare.mjs'), md, r64, key, '--yardstick=' + r32];
  ok.push(run(py, [path.join(here, 'reference.py'), md, r32, prompt]), run(py, [path.join(here, 'reference.py'), md, r64, prompt, '--float64']));
  ok.push(run('node', cmp), run('node', cmp.concat('--no-wasm')));
  ok.push(!run('node', cmp.concat('--perturb'))); // the self-test: with one knob nudged, the checks must fail
}
const q = path.join(dir, 'qwen2.5-0.5b-instruct'), rec = path.join(dir, 'agent.recorded.json');
ok.push(run('node', [path.join(here, 'agent_record.mjs'), q, rec]) && run(py, [path.join(here, 'agent_reference.py'), q, rec]));
console.log(ok.every(Boolean) ? '\nEVERY CHECK PASSED' : '\nSOME CHECKS FAILED (see above)');
process.exitCode = ok.every(Boolean) ? 0 : 1;
