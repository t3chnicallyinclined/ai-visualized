"""Lay out a model's whole vocabulary as a 2D map for the token cloud (run once per model, offline).

Why Python: openTSNE (fast FFT-accelerated t-SNE) only exists in Python, and a proper layout of 49,152
points is impractical in plain JavaScript. The page never runs this; it loads the file this writes.

Usage:  pip install torch safetensors openTSNE==1.0.4
        python prep/cloud_layout.py <model folder> <output .js> <model key> "<model name>"
        e.g. python prep/cloud_layout.py smollm2-instruct/ cloud-data.js smollm2-instruct "SmolLM2-135M-Instruct"
             python prep/cloud_layout.py tinyllama-15M/ cloud-tiny.js tiny-stories "TinyStories 15M"
Writes: the output file (positions, categories and the measured neighbour score, base64), which registers itself
        as window.CLOUDS[<model key>]

Method: take the embedding table (49,152 rows x 576), normalise each row (so the layout uses cosine
similarity), reduce to 128 dimensions with PCA, then openTSNE (cosine, perplexity 30, PCA initialisation,
seed 42, one thread so reruns give the same map). 128 rather than 50 PCA dimensions because 50 alone
already loses most neighbours (measured: 18% of true neighbours survive PCA-50, 37% PCA-128). Then measure how well the map keeps neighbours:
for 3,000 sampled tokens, how many of their 10 nearest tokens in the real 576 numbers are among their
10 nearest on the map (recall@10). That number is shown on the page.
"""
import base64, json, sys, unicodedata
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from openTSNE import TSNE

src, out, key, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
W = load_file(str(src / "model.safetensors"))
EMB = ["model.embed_tokens.weight", "gpt_neox.embed_in.weight", "transformer.wte.weight", "backbone.embeddings.weight"]  # Llama, GPT-NeoX, GPT-2, Mamba
E = W[next(k for k in EMB if k in W)].float()
V = E.shape[0]
tok = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
vocab = {i: s for s, i in tok["model"]["vocab"].items()}
for a in tok["added_tokens"]:
    vocab[a["id"]] = a["content"]

# GPT-2 byte-level alphabet, to turn a vocab entry back into its text
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
cs, n = bs[:], 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
dec = {chr(c): b for b, c in zip(bs, cs)}

sentencepiece = bool(tok["model"].get("byte_fallback"))  # Llama-style: "▁" for spaces, <0xNN> byte tokens
special = {a["id"] for a in tok["added_tokens"] if a.get("special")}

def text(i):
    raw = vocab.get(i, "")
    if i in special:
        return None, raw
    if sentencepiece:
        if raw.startswith("<0x") and raw.endswith(">"):
            return None, raw  # a single byte: left out, like byte fragments
        return raw.replace("▁", " "), raw
    try:
        return bytes(dec[ch] for ch in raw).decode("utf-8"), raw
    except (KeyError, UnicodeDecodeError):
        return None, raw

# categories: 0 whole word (starts with a space), 1 word piece, 2 digits, 3 punctuation/whitespace,
# 4 code and symbols, 5 non-Latin script, 255 left out (special tokens, byte fragments)
cat = np.full(V, 255, dtype=np.uint8)
for i in range(V):
    t, raw = text(i)
    if t is None or raw.startswith("<|") or not t:
        continue
    s = t.strip()
    if not s:
        cat[i] = 3
    elif all(ch.isdigit() for ch in s):
        cat[i] = 2
    elif all(unicodedata.category(ch)[0] in "PZS" for ch in s):
        cat[i] = 3
    elif any(ord(ch) > 0x24F and ch.isalpha() for ch in s):
        cat[i] = 5
    elif s.isalpha():
        cat[i] = 0 if t.startswith(" ") else 1
    else:
        cat[i] = 4
keep = np.where(cat != 255)[0]
print(f"vocabulary {V}, laid out {len(keep)}, left out {V - len(keep)}", flush=True)

torch.manual_seed(0)  # pca_lowrank is randomised
X = torch.nn.functional.normalize(E[keep], dim=1)
U, S, Vt = torch.pca_lowrank(X - X.mean(0), q=128, center=False)
Z = ((X - X.mean(0)) @ Vt[:, :128]).numpy().astype(np.float64)
emb = TSNE(perplexity=30, metric="cosine", initialization="pca", random_state=42, n_jobs=1, verbose=True).fit(Z)
Y = np.asarray(emb)

# recall@10 on 3,000 sampled tokens: true neighbours by cosine in 576 dimensions vs neighbours on the map
rng = np.random.default_rng(0)
sample = rng.choice(len(keep), 3000, replace=False)
sims = X[sample] @ X.T
sims[torch.arange(3000), torch.as_tensor(sample)] = -2
true_nn = torch.topk(sims, 10, dim=1).indices.numpy()
Yt = torch.as_tensor(Y)
d2 = torch.cdist(Yt[sample], Yt)
d2[torch.arange(3000), torch.as_tensor(sample)] = 1e18
def recall_at(k):
    map_nn = torch.topk(d2, k, dim=1, largest=False).indices.numpy()
    return float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(true_nn, map_nn)]))
recall, recall100 = recall_at(10), recall_at(100)
print(f"recall@10 = {recall:.3f}; true 10 found within the 100 nearest dots = {recall100:.3f}", flush=True)

lo, hi = Y.min(0), Y.max(0)
xy = np.full((V, 2), 65535, dtype=np.uint16)
xy[keep] = np.round((Y - lo) / (hi - lo) * 65534).astype(np.uint16)
meta = {"model": name, "tokens": int(len(keep)), "perplexity": 30, "seed": 42, "recall10": round(recall, 3), "recall100": round(recall100, 3), "method": "openTSNE 1.0.4 on L2-normalised embedding rows, PCA to 128, cosine"}
out.write_text(
    f"// Token-cloud map of {name}'s vocabulary. Generated by prep/cloud_layout.py; do not edit by hand.\n"
    "window.CLOUDS = window.CLOUDS || {};\n"
    f"window.CLOUDS[{json.dumps(key)}] = {{ meta: {json.dumps(meta)},\n"
    f"  xy: '{base64.b64encode(xy.tobytes()).decode()}',\n"
    f"  cat: '{base64.b64encode(cat.tobytes()).decode()}' }};\n",
    encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
