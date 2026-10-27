# Checking the engine against transformers

`llm.js` is written from scratch, so these scripts check it against the official Hugging Face `transformers` implementation, for every model the pages use, at the exact revision the pages download.

## What is checked

For each of the six models (`compare.mjs`, against numbers from `reference.py`):

| Check | What must hold |
|---|---|
| tokens | six tricky texts (accents, emoji, Chinese, odd spacing, contractions, a tool call) give identical token IDs |
| prompt tokens | the test prompt, in the chat template for the chat models, gives identical token IDs |
| ball after every layer | every number of the last token's hidden state, after every layer, differs by less than 1/10,000 of that layer's largest number |
| logit lens | the same top token after every layer (final norm and output table applied to each layer's output) |
| top-10 scores | the same ten tokens in the same order, each score within 0.001 |
| greedy continuation | identical 24 tokens of plain greedy output |

The prompt runs the way the pages run it: `prefill` for all but the last token, then one recorded `step`. Each model is checked twice: once with the WebAssembly SIMD kernel, once with `--no-wasm` (the plain JavaScript fallback).

For the agent (`agent_record.mjs`, then `agent_reference.py`): the loop in `agent.js` runs the agent page's three examples on Qwen2.5-0.5B-Instruct, and for every turn:

- the tokens it fed the model must equal `apply_chat_template(messages so far, tools)` token for token
- the tokens it chose must equal transformers' own greedy generation with Qwen's published repetition penalty (1.1)

`get_models.mjs` downloads the pinned revisions and checks each weights file against the SHA-256 that Hugging Face lists for that revision.

To see that the checks can fail: change one number in a reference file and run `compare.mjs` again (it was tried: a 1% change in one hidden-state number, one greedy token and one tokenizer ID each turn their check to FAIL).

## Results (2026-10-26, transformers 4.57.3, torch 2.9.1 CPU, Node 22)

Largest hidden-state difference, as a fraction of the layer's largest number (WebAssembly / plain JavaScript), and largest top-10 score difference:

| Model | Revision | Layers | Hidden state | Scores | Everything else |
|---|---|---|---|---|---|
| TinyStories 15M | `a97e01f` | 6 | 5.4e-7 / 4.3e-7 | 2.4e-6 | identical |
| Pythia-14M | `cf967c0` | 6 | 5.4e-6 / 5.4e-6 | 5.6e-5 | identical |
| DistilGPT-2 | `2290a62` | 6 | 8.9e-7 / 8.9e-7 | 3.1e-5 | identical |
| SmolLM2-135M-Instruct | `12fd25f` | 30 | 9.7e-7 / 6.5e-7 | 1.7e-5 | identical |
| Mamba-130M | `1e76775` | 24 | 7.8e-7 / 9.9e-7 | 5.0e-5 | identical |
| Qwen2.5-0.5B-Instruct | `7ae5576` | 24 | 1.1e-6 / 9.7e-7 | 1.3e-5 | identical |

"Everything else" means: tokens on all six texts, the prompt's tokens, the logit-lens top token after every layer, the top-10 tokens and their order, and all 24 greedy tokens.

Agent: all six turns of the three examples pass. Every context (219 to 282 tokens) equals the chat template token for token, and every reply (18 to 37 tokens, tool requests included) equals transformers' greedy output. They are the same runs the agent page plays back from `agent-runs.js`.

## Running it

Needs Node 18 or newer, and Python with `torch` and `transformers`. The models take about 1.9 GB.

```
node verify/run_all.mjs <a folder for the models and results>
```

Or one model at a time:

```
node verify/get_models.mjs models tiny-stories
python verify/reference.py models/tiny-stories tiny.json "Once upon a time, there was a little"
node verify/compare.mjs models/tiny-stories tiny.json tiny-stories
node verify/compare.mjs models/tiny-stories tiny.json tiny-stories --no-wasm
```

Small differences in the last digits are expected: float32 sums done in a different order round differently. That is why hidden states and scores are compared with a tolerance, while everything the page shows as a choice (tokens, the lens's top token, the greedy output) must be identical.
