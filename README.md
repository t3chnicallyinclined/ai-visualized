# LLM Plinko

**See how a large language model works by watching it play Plinko.**

Type a prompt and press Enter. A real language model runs in your browser, and every number on the page is its own. Your text gets cut into tokens, each token becomes a ball, the balls fall through the model's layers, and the last one lands as a score for every possible next word. A dice roll picks one, it joins the text, and the whole thing runs again. That loop is all an LLM does, whether it is a small model on a laptop or the largest one in a datacenter.

**Try it: [the board](https://t3chnicallyinclined.github.io/llm-plinko/)**

No install, no build step, no account. Plain HTML and JavaScript you can read. The first visit downloads a 30 MB model.

## Run it

- Open `index.html` in any modern browser, or
- serve the folder: `npx serve .` (or any static file server) and open the URL it prints.

## What you are looking at

Read the board top to bottom, like Plinko. Each token gets a **lane** running down the board, and the lanes fall through **every layer** of the model (each layer is a band of dials, each dial one of that layer's real weights) (6 for TinyStories 15M, 30 for SmolLM2-135M-Instruct). In each layer a token does two things:

- **Talk (attention):** it compares itself with the earlier tokens and blends in information from the ones that match. This is the only place tokens share information. Pink lines show who it listens to (the model's real attention). Its key and value stay behind in that layer's **KV cache** (a teal tick).
- **Think (feed-forward):** it works alone. Hundreds of neurons check the ball for patterns and some fire strongly; the burst shows how much was added.

Both add onto the ball rather than replacing it. Point at any lane in any layer and the fixed-size narrator bar at the bottom of the stage explains it (click it for the full text); **Show the math** lists the exact steps inside that station (norm, query/key/value, scores, softmax, mix, add; or norm, gate and up, SiLU, down, add), with the model's real numbers.

A strip at the top of the stage shows the whole process, in order, and lights up each step as it runs (click one to jump to it):

1. **Type**: your text (for the chat model: the system prompt and your message in its chat template) appears at the top of the board.
2. **Tokenize**: the text is cut into pieces the model knows, and each piece is looked up in the vocabulary to get its ID.
3. **Embed**: each ID picks one row of a big learned table; that row of numbers is the ball. It carries no word order.
4. **Layers**: every lane goes through every layer: talk, then think. Word order gets in during talk, where each query and key is rotated by its position (RoPE).
5. **Unembed**: one more norm, then the ball is compared with every row of an output table, giving one score per token (both models reuse their embedding table here).
6. **Choose**: softmax over the top five, temperature, and a weighted dice roll.
7. **Detokenize**: the winning ID turns back into text and is added to the output.
8. **Repeat**: the new token becomes a new lane and goes through alone, reading the KV cache, until an end-of-text token or a length limit.

### The embedding explorer

Click any token's strip (or **Explore embeddings**) to see what embedding really is:

- **The table**: the real rows around the token's ID as a heat map (teal positive, pink negative), with its row sliding out to become the ball. Neighbouring IDs mean nothing; an ID is just a position in the vocabulary.
- **Nearest tokens**: the 8 most similar rows in the whole vocabulary, with cosine similarity worked out for the closest.
- **Compare two**: any two words, their rows side by side, the pairwise products and the similarity, with example pairs worked out from the loaded model (opposites often score high, because they are used in the same places).

### The token cloud (what you can and cannot read inside a model)

The board and the cloud share one dark stage: the ball leaves the last layer on the right and lands in the cloud. The cloud is a map of the model's whole vocabulary (for SmolLM2-135M-Instruct 48,822 of its 49,152 tokens; byte fragments and special tokens left out), laid out once with openTSNE over the real embedding rows (`prep/cloud_layout.py`, seed 42, reproducible), so tokens with similar rows sit together. A flat map can't keep every neighbour: on average 1.7 of a token's 10 true nearest tokens are among its 10 nearest dots, and the page says so. With the real model, after every layer every token lights up by the chance the ball would give it if it stopped there (the logit lens: a full softmax over all 49,152 tokens, brightness on a log scale). In **3D** the map lies down like a floor and that chance becomes height. A dashed line hops between the single most likely token after each layer (it never averages two places), the ball on the board takes that token's map colour, and at the end the five candidates are ringed and the dice hops between them.  The cloud is interactive: hover, click (closest tokens by their real embedding rows), search, zoom, drag. The explorer's **Thought path** tab adds a layer slider and a chart of a token's chance after every layer.

### What the parameters are

**What are the parameters?** shows every weight of the real model. For SmolLM2-135M-Instruct that is all 134,479,872 numbers in its grids (everything but the norm weights) are packed onto your graphics card (WebGL2, one byte each, three 8192-wide textures) and drawn as one square per number, every grid in its true shape: the embedding table as four tall strips, then 30 layer blocks (attention on top; feed-forward below, with down turned so row k of gate, up and down is neuron k). Zoom in until each square shows its exact value. It looks like static, which is the honest picture: no single number is readable. Without the real model (or without WebGL2) a to-scale overview is shown instead. The token cloud is not the parameters; it is a picture of one grid (the embedding table, 21% of them). The cloud's dots and light are also drawn with WebGL2 now, with a 2D-canvas fallback.

What the cloud is not: the dots are the vocabulary's coordinates, not stored knowledge. A model's knowledge lives in its weights, spread across every layer, not in places you can point at, and nothing is searched: at the end the ball is compared with every token at once. Its memory of the conversation is the KV cache. The path is the thinking; where it ends is the prediction.

### Real models only

Every number on the page comes from a real model running in your browser with plain JavaScript (`llm.js`), downloaded once from Hugging Face (pinned revisions) and kept by your browser. Five models, four designs:

| Model | Design | Download | What it shows |
|---|---|---|---|
| **[TinyStories 15M](https://huggingface.co/nickypro/tinyllama-15M)** (loads by itself) | Llama: RMSNorm, rotation (RoPE), SwiGLU | 30 MB | 6 layers you can see clearly; children's stories only |
| **[Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m)** | GPT-NeoX: talk and think **side by side**, RoPE on a quarter of each head, LayerNorm, a separate output table | 28 MB | one station per layer; its **real training checkpoints** in the training view |
| **[DistilGPT-2](https://huggingface.co/distilbert/distilgpt2)** | GPT-2: a **learned position table** added at the start, LayerNorm, GELU | 353 MB | "+p" under each embedding; no rotation |
| **[Mamba-130M](https://huggingface.co/state-spaces/mamba-130m-hf)** | Mamba: **no attention**, a fixed-size memory per layer | 517 MB | memory boxes on the board; the memory update step by step |
| **[SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)** | Llama, chat-tuned | 269 MB | the chat template and a **system prompt**, compared in the cloud |

With any of them, everything on the board is the model's actual output: tokens and IDs from its own tokenizer, embedding rows, attention arcs (or Mamba's memory writes), the guess on the ball (the logit lens), the knobs' needles (real weights), every token's chance in the cloud after every layer, and the real scores.

**How these models relate** opens a family tree from *Attention Is All You Need* (2017) through GPT-2, FNet (2021: token mixing with a Fourier transform instead of attention), GPT-NeoX/Pythia and Llama-style models to Mamba (2023), with what each changed (word order, how tokens mix, the feed-forward, normalisation, the output table) and links to the papers. The 2017 Transformer (a translation model) and FNet (an encoder that does not predict the next word) are explained there but cannot run on the board.

**Checked against the reference.** Every model was checked against the official Hugging Face `transformers` implementation (float32): identical tokens on tricky texts (including the chat template's special tokens, SentencePiece byte fallback, and GPT-NeoX's whitespace tokens), identical greedy continuations, identical top-10 scores, the ball's numbers after every layer within a few millionths, and the same logit-lens top token after every layer. In the page, every "knobs × ball" grid in the layer view is recomputed from the real weights and shows a ✓ when it matches. Each model has its own token-cloud map (`cloud-*.js`, made by `prep/cloud_layout.py`).

**Why the system prompt matters.** It is not stored anywhere special and it never changes the knobs: it is more tokens at the start of the context, read through attention. Tick **Compare with a second system prompt** and the same message is also read with that prompt; its light shows in blue in the cloud, and the line under the cloud lists the tokens each prompt makes more likely. With "Talk like a pirate. Always start with Arr." the chance that the reply starts with "Ar" rises about 21,000 times and "Arr" about 7,000 times, but this small model still answers "A fireball is…" (94% → 55% for "A"): bigger models are trained to follow the system prompt much more strongly. The right context lights up the tokens that fit it.

**Checked against the reference.** Both models were checked against the official Hugging Face `transformers` implementation (float32): identical tokens (including the chat template's special tokens, and the tiny model's SentencePiece tokenizer with byte fallback), identical greedy replies, and the same logit-lens top token after every layer. The token cloud has its own map per model (`cloud-data.js`, `cloud-tiny.js`, made by `prep/cloud_layout.py`; on average 1.7 and 1.1 of a token's 10 true nearest tokens are among its 10 nearest dots).

### Knobs, and what a layer does to the ball

Every parameter is a knob, set once in training and never moved while the model runs; no prompt can turn one. On the board each layer band is drawn as that layer's knob bank: its real weights (the talk grids on the top half, the think grids on the bottom), the same knobs for every token, lit while the ball is multiplied through them. The ball is a list of numbers, not a single value: every ball carries a strip of its first 16 real numbers (the whole prompt goes through each layer together), and a + marks each change being added onto it (the residual stream).

**Look inside a layer** (or click any layer band on the board) opens one token's ball going through that layer step by step, with the model's own numbers: the ball arriving, the RMS norm (with its own knobs), query/key/value = knobs × ball, the rotation by position, attention over every earlier token (all heads), the output knobs, the + onto the ball, then the feed-forward gate and up knobs, the neurons firing, the down knobs and the second +. Each "knobs × ball" step is the real knob grid: the ball across the top, one output per row on the right; **Multiply, row by row** animates it, pointing at a row shows that row's dot product, and the page recomputes every product from the real weights and confirms it matches the model's own numbers.

### Training: how the knobs get set

**How training sets the knobs** first shows **a real model learning**: Pythia-14M at 9 of its 154 published training checkpoints (step 0 = random knobs, to step 143,000 ≈ 300 billion tokens). Each snapshot is loaded (28 MB), writes the same continuation of the same prompt, and is scored on a fixed test sentence; the loss falls (about 11.1 at step 0, 5.2 at step 1,000, 4.0 at the end) and the same 24 knobs turn into place. Below that it runs a real training run in your browser, just tiny: a model with one grid of 26 × 26 = 676 knobs (a bigram model: it only looks at the current letter) learns from the short text built into the page by plain gradient descent. You watch the needles turn, the loss fall towards the best this model can do, and its writing turn into almost-words. Below it, the page says how the real models were trained (SmolLM2: 2 trillion tokens of pre-training, then fine-tuning and DPO, which is where it learned to follow a system prompt) and that after training the knobs are frozen.

### The guided tour

On the first visit (and with **Take the tour**) the page plays one token and stops at the start of every step, with Back, Next and End tour in the narrator bar. On phones the board scrolls itself to keep the action in view, and the narrator bar (with a play button) stays on screen.

Every parameter is a knob, set once in training and never moved while the model runs; no prompt can turn one. On the board each layer band is its knob bank (see below). In **What are the parameters?** every weight is a square, and zoomed in each one becomes a knob with its exact value.

### Zoom into a neuron

**Zoom into a neuron** works out the busiest feed-forward neuron of a layer by hand from the real weights (the ball's numbers times the neuron's knobs, added up), and then how softmax turns the five top scores into the chances the dice roll uses.

### Memory and speed

The footer shows what running the model costs in your browser: the weights downloaded, the KV cache growing with every token (each layer keeps its own keys and values), and your measured speed; every new token reads every weight again.

## The maths in one paragraph

Almost every LLM is a **transformer**, and it uses the same few operations over and over. **Matrix multiplication** (multiply pairs of numbers and add them up) is more than 90% of the work. **Attention** scores how relevant each earlier token is with `softmax(Q·Kᵀ / √d) · V`. **Softmax** turns scores into percentages. A small **activation function** bends the numbers so stacking layers can learn complex patterns, **normalization** keeps them from blowing up, and each layer **adds** its result to the token instead of replacing it. Models differ in size, attention shortcuts, dense vs MoE, and above all in the weights that training put into them.

## What is real and what is not

Everything on the page comes from the two real models, computed in your browser and checked against the reference implementation. Things that are pictures rather than numbers are labelled where they appear: the token cloud is a 2D map of the real embedding rows (t-SNE, so distances are approximate, and the page states how many neighbours it keeps), and the training view is a real training run of a tiny 676-knob model, not of the models above. An earlier toy mode with made-up scores for Qwen3.6 has been removed.

## Roadmap

Ideas for making this a better teaching tool. Contributions welcome.

- [x] Zoom into one neuron: a real dot product and a softmax worked out step by step
- [x] Look inside a layer: every step with the model's own numbers
- [ ] Attention up close: queries, keys and values for one token
- [x] Training: a real (tiny) training run where you watch the knobs turn and the loss fall
- [ ] Quantization: what squeezing weights to 4 bits does
- [x] Run a real small model in the browser, with the logit lens
- [ ] A 3D version (removed for now, to be rethought)
- [x] A guided tour on the first visit (Back / Next / End tour)
- [x] Phones: the board follows the action and the narrator bar stays on screen

## Contributing

Issues and pull requests are welcome. There is no build step: `index.html` is the board (plain HTML, CSS and JavaScript, with [GSAP](https://gsap.com) from a CDN for the animation), `llm.js` is the real-model engine (no libraries), `cloud-data.js` and `cloud-tiny.js` are the token maps, and `prep/cloud_layout.py` makes them. Keep it that way so anyone can open the files and read the source.

## License

MIT. See [LICENSE](LICENSE).
