# Checking the engine against transformers

`llm.js` is written from scratch, so these scripts check it against the official Hugging Face `transformers` implementation, for every model the pages use, at the exact revision the pages download.

The page doesn't watch a model running somewhere else: the page *is* the model running, so every number on screen is one `llm.js` computed. What needs proving is that those numbers are the ones the real model gives. So `transformers` runs the same model with hooks on every layer and on each layer's parts (a trace), and every number is compared.

## The yardstick: 64-bit maths

Both `llm.js` and `transformers` normally compute in float32, which rounds every number to about 7 digits. Different code adds numbers in a different order, so the last digits differ, and a very sharp attention head (one that puts 90% on one token) magnifies those differences. So the references are computed in **float64**, about 16 digits, which is as close to the exact answer as a computer gets. Each `llm.js` number is measured against that, next to how far `transformers`' own float32 run is from it.

A number passes when `llm.js` is no further from the 64-bit answer than 3 times `transformers`' own float32 rounding. In practice `llm.js` is about as close as `transformers`, or closer.

## What is checked

For each of the six models (`compare.mjs`, against numbers from `reference.py`):

| Check | What must hold |
|---|---|
| tokens | six tricky texts (accents, emoji, Chinese, odd spacing, contractions, a tool call) give identical token IDs |
| prompt tokens | the test prompt, in the chat template for the chat models, gives identical token IDs |
| ball after every layer | every number of the last token's hidden state, after every layer |
| every token after every layer | every number of every token's hidden state, after every layer |
| attention weights | every head's weights in every layer, for the last token over every position |
| attention block output | what attention adds, in every layer (Mamba: what the mixer adds) |
| feed-forward block output | what the feed-forward block adds, in every layer |
| logit lens | the same top token after every layer (final norm and output table applied to each layer's output) |
| top-10 | the same ten tokens in the same order, and their scores |
| greedy continuation | identical 24 tokens of plain greedy output |

The prompt runs the way the pages run it: `prefill` for all but the last token, then one recorded `step`. Each model is checked twice: once with the WebAssembly SIMD kernel, once with `--no-wasm` (the plain JavaScript fallback).

**The self-test** (`--perturb`): one knob in the middle layer is nudged by 5% of its grid's largest knob, and the checks must then fail. They do, for every model (a check that can't fail proves nothing). A broken reference is caught too: if `transformers`' float32 and float64 runs disagree by 5e-4 or more, the check fails and says to look at the references.

For the agent (`agent_record.mjs`, then `agent_reference.py`): the loop in `agent.js` runs the agent's three examples on Qwen2.5-0.5B-Instruct, and for every turn:

- the tokens it fed the model must equal `apply_chat_template(messages so far, tools)` token for token
- the tokens it chose must equal transformers' own greedy generation with Qwen's published repetition penalty (1.1)

`get_models.mjs` downloads the pinned revisions and checks each weights file against the SHA-256 that Hugging Face lists for that revision.

## Results (2026-10-26, transformers 4.57.3, torch 2.9.1 CPU, Node 22)

All checks pass for all six models, with and without WebAssembly, and the self-test fails for all six, as it must.

Largest difference from the 64-bit answer: `llm.js` with WebAssembly / plain JavaScript, then `transformers`' own float32 run. Hidden states and block outputs are a share of the layer's largest number; attention weights are absolute (they add up to 1).

| Model | Revision | Every token, every layer | Attention weights | Attention block | Feed-forward block | Top-10 scores |
|---|---|---|---|---|---|---|
| TinyStories 15M | `a97e01f` | 6.1e-7 / 3.6e-7 · 1.1e-6 | 3.0e-7 / 3.3e-7 · 6.9e-7 | 5.3e-7 / 2.9e-7 · 6.9e-7 | 6.8e-7 / 3.0e-7 · 8.1e-7 | 2.5e-6 / 5.2e-7 · 3.2e-6 |
| Pythia-14M | `cf967c0` | 5.5e-5 / 1.5e-5 · 5.1e-5 | 3.0e-5 / 3.1e-5 · 1.7e-4 | 4.0e-6 / 3.9e-6 · 5.7e-5 | 3.1e-6 / 2.1e-6 · 2.8e-6 | 1.2e-5 / 1.2e-5 · 6.6e-5 |
| DistilGPT-2 | `2290a62` | 2.4e-7 / 2.4e-7 · 9.1e-7 | 1.9e-7 / 1.9e-7 · 1.4e-6 | 3.9e-7 / 3.9e-7 · 2.4e-6 | 2.8e-7 / 2.8e-7 · 3.1e-6 | 9.7e-6 / 4.6e-6 · 1.5e-5 |
| SmolLM2-135M-Instruct | `12fd25f` | 4.1e-6 / 2.8e-6 · 4.4e-6 | 1.6e-6 / 6.6e-7 · 1.9e-6 | 3.6e-6 / 1.5e-6 · 3.9e-6 | 5.9e-6 / 1.1e-6 · 3.8e-6 | 1.2e-5 / 6.9e-6 · 1.0e-5 |
| Mamba-130M | `1e76775` | 1.3e-6 / 8.2e-7 · 1.3e-6 | (no attention) | 8.2e-7 / 4.7e-7 · 2.2e-6 (mixer) | (none) | 1.1e-5 / 3.8e-6 · 4.6e-5 |
| Qwen2.5-0.5B-Instruct | `7ae5576` | 6.0e-6 / 4.9e-6 · 1.4e-5 | 5.3e-6 / 4.2e-6 · 1.4e-5 | 4.7e-6 / 5.0e-6 · 7.4e-6 | 3.5e-6 / 4.4e-6 · 8.9e-6 | 1.1e-5 / 6.4e-6 · 1.0e-5 |

Tokens on all six texts, the prompt's tokens, the logit-lens top token after every layer, the top-10 tokens and their order, and all 24 greedy tokens are identical for every model.

Pythia-14M's differences are the largest because two of its heads are very sharp (85% and 91% on one token in layers 4 and 6): there `transformers`' own float32 run is 1.7e-4 off the 64-bit answer, and `llm.js` is 3.0e-5 off.

Agent: all six turns of the three examples pass. Every context (219 to 282 tokens) equals the chat template token for token, and every reply (18 to 37 tokens, tool requests included) equals transformers' greedy output. They are the same runs Agent mode plays back from `agent-runs.js`.

## Running it

Needs Node 18 or newer, and Python with `torch` and `transformers`. The models take about 1.9 GB; Qwen in float64 needs about 6 GB of free memory.

```
node verify/run_all.mjs <a folder for the models and results>
```

Or one model at a time:

```
node verify/get_models.mjs models tiny-stories
python verify/reference.py models/tiny-stories tiny32.json "Once upon a time, there was a little"
python verify/reference.py models/tiny-stories tiny64.json "Once upon a time, there was a little" --float64
node verify/compare.mjs models/tiny-stories tiny64.json tiny-stories --yardstick=tiny32.json
node verify/compare.mjs models/tiny-stories tiny64.json tiny-stories --yardstick=tiny32.json --no-wasm
node verify/compare.mjs models/tiny-stories tiny64.json tiny-stories --yardstick=tiny32.json --perturb   (must FAIL)
```

Without `--yardstick`, `compare.mjs` uses fixed tolerances (1/10,000 of the largest number for hidden states, 1e-5 for attention weights, 0.001 for scores); those are too tight for float32 in very sharp heads, which is why the float64 yardstick is the main check.
