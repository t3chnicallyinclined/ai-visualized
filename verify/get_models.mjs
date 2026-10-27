// Downloads the exact model revisions the pages use (pinned in llm.js) into <folder>/<model key>/, and checks each
// weights file against the SHA-256 that Hugging Face lists for that revision.
// usage: node verify/get_models.mjs <folder> [model key ...]   (no keys: all of them; about 1.9 GB)
import { createWriteStream, existsSync, mkdirSync, createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
const { MODELS } = createRequire(import.meta.url)('../llm.js');
const [dir, ...keys] = process.argv.slice(2);
if (!dir) { console.error('usage: node verify/get_models.mjs <folder> [' + Object.keys(MODELS).join(' | ') + ' ...]'); process.exit(1); }
const sha = f => new Promise((ok, no) => { const h = createHash('sha256'); createReadStream(f).on('data', d => h.update(d)).on('end', () => ok(h.digest('hex'))).on('error', no); });
for (const key of keys.length ? keys : Object.keys(MODELS)) {
  const { repo, rev } = MODELS[key], out = path.join(dir, key); mkdirSync(out, { recursive: true });
  for (const f of ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'model.safetensors']) {
    const to = path.join(out, f); if (existsSync(to)) continue;
    const r = await fetch(`https://huggingface.co/${repo}/resolve/${rev}/${f}`);
    if (!r.ok) { if (f === 'generation_config.json') continue; throw new Error(`${repo} ${f}: HTTP ${r.status}`); }
    await pipeline(Readable.fromWeb(r.body), createWriteStream(to));
  }
  const info = await (await fetch(`https://huggingface.co/api/models/${repo}/paths-info/${rev}`, { method: 'POST', headers: { 'content-type': 'application/x-www-form-urlencoded' }, body: 'paths=model.safetensors' })).json();
  const want = info[0].lfs.oid, have = await sha(path.join(out, 'model.safetensors'));
  console.log(`${key}: ${repo}@${rev.slice(0, 7)} ${want === have ? 'weights match the published SHA-256' : 'WEIGHTS DO NOT MATCH (delete the folder and download again)'}`);
}
