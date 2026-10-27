# AI Visualized

A visualization of a language model writing, one token at a time, with the real numbers on screen.

**Try it:** [guided walk-through](https://t3chnicallyinclined.github.io/ai-visualized/) · [Explorer](https://t3chnicallyinclined.github.io/ai-visualized/explore.html) · [agent, step by step](https://t3chnicallyinclined.github.io/ai-visualized/agent.html) · [kids and classroom](https://t3chnicallyinclined.github.io/ai-visualized/kids.html)

Plain HTML and JavaScript, no build step and no server. The models run in your browser and nothing you type leaves it. The first visit downloads a 30 MB model.

## The picture

There are plenty of diagrams of imaginary models. This one runs real ones and shows what they compute, using the same picture on every page:

A language model is a stereo with millions of volume knobs. Training set the knobs and glued them; a prompt never moves one. Each word goes in as a spot in a cloud, and where it sits is set by how the word gets *used*, not how it is pronounced. In SmolLM2, *two* and *three* sit at 0.81; *two* and *too* at 0.15.

The knobs then do their work. The only part that reacts to what you typed is attention, which is the model listening back over what came before. At the end there is a wall of lights, one per token in the vocabulary, each lit by how likely it is next. Weighted dice pick one. Then it starts again with that token added.

An agent is the same stereo with a person next to it: a program that reads the words it writes, types any commands into a shell, and hands the output back.

## The pages

**Guided walk-through** (`index.html`) — one run of a real model, slowed down and lit one step at a time: words into pieces, pieces into numbers, one layer close up, every layer, the wall of lights, the dice, repeat.

**Explorer** (`explore.html`) — the whole loop on one board, read top to bottom. Type, tokenize, embed, layers, unembed, choose, detokenize, repeat. Around it: a map of the model's vocabulary that lights up after each layer (the logit lens), one token's trip through a single layer with every multiplication recomputed from the weights, every parameter drawn one square per number, nine real Pythia training checkpoints from random knobs to 300 billion tokens, and a family tree from *Attention Is All You Need* to Mamba.

**Agent** (`agent.html`) — a real agent loop at reading speed. Qwen2.5-0.5B-Instruct gets a task and one tool, a pretend shell over three files. You watch the context tape, the layers, the attention, and the pipe from the model's words through the program into the shell and back. It is also where you can try to make it go rogue: told to send a file to the internet it announces it did, while "what actually happened" shows it never left. The examples play from recorded real runs so phones work; loading the full model runs it live.

**Kids and classroom** (`kids.html`) — the same real model writing a story in five steps, then the myths tested against it: it is arithmetic, it is a machine, it does not know what is true, and it cannot act on its own. Part 2 adds a chat model and a web page a program repaints only when the reply contains an allowed colour word. Part 3 is the agent page. Notes for teachers by grade band.

## The models

Six models, four designs, each downloaded once from Hugging Face at a pinned revision.

| Model | Design | Download | What it shows |
|---|---|---|---|
| [TinyStories 15M](https://huggingface.co/nickypro/tinyllama-15M) (loads first) | Llama: RMSNorm, rotation (RoPE), SwiGLU | 30 MB | 6 layers you can see clearly; children's stories only |
| [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m) | GPT-NeoX: attention and feed-forward side by side, RoPE on a quarter of each head, a separate output table | 28 MB | one station per layer; real training checkpoints |
| [DistilGPT-2](https://huggingface.co/distilbert/distilgpt2) | GPT-2: a learned position table added at the start | 353 MB | word order without rotation |
| [Mamba-130M](https://huggingface.co/state-spaces/mamba-130m-hf) | Mamba: no attention, a fixed-size memory per layer | 517 MB | memory instead of a growing cache |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Llama, tuned to chat | 269 MB | the chat template and the system prompt |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Llama-style with biases on attention q/k/v; trained to call tools | 988 MB | the agent loop (live on request; examples also play from recordings) |

## What is real and what is a picture

Every number about the model's work comes from its real weights.

`llm.js` is a from-scratch JavaScript implementation of all four designs, checked against Hugging Face `transformers` in float32: identical tokens on tricky text, identical greedy continuations, identical top-10 scores, hidden states within a few millionths after every layer, and the same logit-lens top token at every layer. The agent's context was checked against `apply_chat_template` token for token. (The comparison scripts are not in the repository yet.)

Anything that is an illustration says so where it appears:

- the token cloud is a t-SNE map of the embedding rows (`prep/cloud_layout.py`); distances are approximate and the page states how many true neighbours it keeps
- the agent's shell and the kids page's web page are imitations written in `agent.js`
- the dice's first hops in the Explorer are animation; where it lands is the real pick
- the training view's tiny model is a real training run of a 676-knob model, not of the models above

The engine runs on the processor. Its one hot loop (knobs times numbers) uses WebAssembly SIMD where available (`prep/matvec.wat`), with a JavaScript fallback that does the same sums.

## Run it locally

Serve the folder with any static file server, for example `npx serve .`, and open the URL it prints. Opening the pages from disk works too; the models download from Hugging Face either way.

## Contributing

Issues and pull requests welcome. No build step:

| File | What it is |
|---|---|
| `index.html` | guided walk-through |
| `explore.html` | the Explorer (GSAP from a CDN for the board timeline) |
| `kids.html` | kids and classroom version |
| `agent.html` | step-by-step agent page |
| `llm.js` | the model engine, no libraries |
| `agent.js` | agent loop, pretend shell, drawings |
| `agent-runs.js` | recorded example runs (`prep/record_agent.mjs` remakes them) |
| `cloud-*.js` | token maps, made by `prep/cloud_layout.py` |

Keep it that way, so anyone can open a file and read it.

## License

MIT, see [LICENSE](LICENSE). The models keep their own licenses (Apache 2.0 or MIT; see each model's page).
