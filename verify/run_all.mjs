// Runs every check: downloads the pinned models, writes transformers' reference numbers, compares llm.js with them
// (with and without WebAssembly), and checks the agent loop. Needs Python with torch and transformers.
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
  const md = path.join(dir, key), ref = path.join(dir, key + '.reference.json');
  ok.push(run(py, [path.join(here, 'reference.py'), md, ref, m.chat ? 'What is a fireball?' : 'Once upon a time, there was a little']));
  ok.push(run('node', [path.join(here, 'compare.mjs'), md, ref, key]), run('node', [path.join(here, 'compare.mjs'), md, ref, key, '--no-wasm']));
}
const q = path.join(dir, 'qwen2.5-0.5b-instruct'), rec = path.join(dir, 'agent.recorded.json');
ok.push(run('node', [path.join(here, 'agent_record.mjs'), q, rec]) && run(py, [path.join(here, 'agent_reference.py'), q, rec]));
console.log(ok.every(Boolean) ? '\nEVERY CHECK PASSED' : '\nSOME CHECKS FAILED (see above)');
process.exitCode = ok.every(Boolean) ? 0 : 1;
