# AI Visualized

A visualization of a language model writing, one token at a time, with the real numbers on screen.

**Try it:** [the show](https://t3chnicallyinclined.github.io/ai-visualized/show.html) · [guided walk-through](https://t3chnicallyinclined.github.io/ai-visualized/) · [Explorer](https://t3chnicallyinclined.github.io/ai-visualized/explore.html) · [agent, step by step](https://t3chnicallyinclined.github.io/ai-visualized/?mode=agent) · [kids and classroom](https://t3chnicallyinclined.github.io/ai-visualized/kids.html)

Plain HTML and JavaScript, no build step and no server. The models run in your browser and nothing you type leaves it. The first visit downloads a 30 MB model.

## The picture

There are plenty of diagrams of imaginary models. This one runs real ones and shows what they compute, using the same picture on every page:

A language model is a machine with millions of volume knobs. Training set the knobs and glued them; a prompt never moves one. Each word goes in as a spot in a cloud, and where it sits is set by how the word gets *used*, not how it is pronounced. In SmolLM2, *two* and *three* sit at 0.81; *two* and *too* at 0.15.

The knobs then do their work. The only place the words affect each other is attention, which is the model listening back over what came before. At the end there is a wall of lights, one per token in the vocabulary, each lit by how likely it is next. Weighted dice pick one. Then it starts again with that token added.

An agent is the same machine with a person next to it: a program that reads the words it writes, types any commands into a shell, and hands the output back.

## The pages

**The show** (`show.html`; on a phone the front page opens it) — one word, as a full-screen film on one canvas: your words shatter into pieces, fly to their spots in the cloud of every piece the model knows, the last piece listens back along six coloured threads (one per attention head), its numbers fall through the six layers as a ring of 288 spokes while the cloud lights up with each layer's guess, the wall of lights, the weighted dice, and the word drops into the sentence; then the next words run through the whole loop. Every frame is drawn from the model's own numbers for your sentence. Choose any of four models (TinyStories, Pythia, DistilGPT-2, or the SmolLM2 chat model, which reads your message in its chat form); a screen before the cloud shows the model as a block of numbers in memory and your tokens' rows read out of the embedding table. Drag, scroll or pinch the cloud, turn it in real 3D (a three-dimensional t-SNE of the same rows, `prep/cloud_layout3d.py`, drawn as WebGL points; the page says how many true neighbours each map keeps) and tap any dot for its neighbours; tap rings, lights and orbs for their numbers; ? explains each screen with the real numbers behind it. Continue, Back, swipe, or Autoplay. A second act, **Act as an agent**, plays the three recorded Qwen2.5-0.5B-Instruct runs in the same language: the hidden instructions and your task as tokens, the 24 layers deciding (the logit lens after each), the reply token by token with its attention, the program finding the command, the real Linux in the page running it, the output going back, the answer, and what actually happened. `show.html?kids` is the same show in the kids page's words, and every page has a **words** button: the picture's words mapped to the proper terms (`legend.js`).

**Guided walk-through** (`index.html`) — one run of a real model, slowed down and lit one step at a time, in two modes. **Write**: words into pieces, each piece a spot in the cloud, one layer close up, every layer, the wall of lights, the dice, repeat. **Agent**: the same steps on Qwen2.5-0.5B-Instruct, plus the ones only an agent has (marked *agent* on the page).

**Explorer** (`explore.html`) — the whole loop on one board, read top to bottom. Type, tokenize, embed, layers, unembed, choose, detokenize, repeat. Around it: a map of the model's vocabulary that lights up after each layer (the logit lens), one token's trip through a single layer with every multiplication recomputed from the weights, every parameter drawn one square per number, nine real Pythia training checkpoints from random knobs to 300 billion tokens, and a family tree from *Attention Is All You Need* to Mamba.

**Agent mode** (`index.html?mode=agent`; the old `agent.html` address opens it) — a real agent loop. In the chat, Qwen2.5-0.5B-Instruct gets a task and one tool, and the program types every command it writes into a real Linux computer emulated in the tab ([v86](https://github.com/copy/v86), with a small Linux built for this page in `linux/`): no internet, and Reset puts it back. Below the chat, the same runs play step by step at reading speed over a pretend shell with the same three files. You watch the context tape, the layers, the attention, and the pipe from the model's words through the program into the shell and back. It is also where you can try to make it go rogue: told to send a file to the internet it announces it did, while "what actually happened" shows it never left. The examples play from recorded real runs so phones work; loading the full model runs it live.

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

`llm.js` is a from-scratch JavaScript implementation of the four designs (Llama, GPT-NeoX, GPT-2, Mamba). The scripts in [`verify/`](verify/README.md) trace the same six models in Hugging Face `transformers`, at the exact revisions the pages download, with hooks on every layer and on each layer's parts, and compare every number:
- identical tokens on six tricky texts and on the prompt (chat template included)
- every number of every token's hidden state after every layer, every head's attention weights, and what the attention and feed-forward blocks add in every layer
- the same logit-lens top token after every layer, the same top-10 tokens in the same order, and identical greedy continuations

The yardstick is `transformers` in 64-bit maths, close to the exact answer: `llm.js` comes out about as close to it as `transformers`' own float32 run, or closer (largest gap 5.5e-5 of a layer's largest number, in Pythia's sharpest heads). The checks pass with both the WebAssembly and the plain JavaScript kernel, and a self-test that nudges a single knob makes them fail, for every model. For the agent, every context the loop builds equals `apply_chat_template` token for token, and every reply equals transformers' own greedy output with Qwen's repetition penalty. In the Explorer, the layer view also redoes each grid multiplication from the weights in the page and shows the check.

Anything that is an illustration says so where it appears:

- the token cloud is a t-SNE map of the embedding rows (`prep/cloud_layout.py`); distances are approximate and the page states how many true neighbours it keeps
- the agent's shell is a small imitation written in `agent.js`, and the kids page's web page is a pretend page in `kids.html`
- the dice's first hops in the Explorer are animation; where it lands is the real pick
- the training view's tiny model is a real training run of a 676-knob model, not of the models above

The engine runs on the processor. Its one hot loop (knobs times numbers) uses WebAssembly SIMD where available (`prep/matvec.wat`), with a JavaScript fallback that does the same sums in a different order, so only the last digits can differ.

## Run it locally

Serve the folder with any static file server, for example `npx serve .`, and open the URL it prints. Opening the pages from disk works too; the models download from Hugging Face either way.

## Contributing

Issues and pull requests welcome. No build step:

| File | What it is |
|---|---|
| `show.html` | the show: one word as a full-screen film, and the agent act; `?kids` for the kids edition |
| `legend.js` | the picture's words and the proper terms, on every page |
| `index.html` | guided walk-through: Write mode and Agent mode |
| `explore.html` | the Explorer (GSAP from a CDN for the board timeline) |
| `kids.html` | kids and classroom version |
| `agent.html` | forwards the old agent address to Agent mode |
| `llm.js` | the model engine, no libraries |
| `agent.js` | agent loop, pretend shell, drawings |
| `machine.js` | the agent's real computer: Linux in the v86 emulator, typed into over its serial console |
| `linux/` | that Linux (`bzimage.bin`), how it is built, and where its source is |
| `agent-runs.js` | recorded example runs (`prep/record_agent.mjs` remakes them) |
| `cloud-*.js` | token maps, made by `prep/cloud_layout.py` |
| `cloud3d-*.js` | the same vocabularies laid out in 3D, made by `prep/cloud_layout3d.py` |
| `verify/` | checks of `llm.js` and the agent loop against `transformers` ([results](verify/README.md)) |

Keep it that way, so anyone can open a file and read it.

## License

MIT, see [LICENSE](LICENSE). The models keep their own licenses (Apache 2.0 or MIT; see each model's page).
