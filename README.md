# LLM Plinko

**See how a large language model works by watching it play Plinko.**

Type a prompt and press Enter. Your text gets cut into tokens, each token becomes a ball, the balls fall through the model's layers, and the last one lands as a score for every possible next word. A dice roll picks one, it joins the text, and the whole thing runs again. That loop is all an LLM does, whether it is a small model on a laptop or the largest one in a datacenter.

**Try it: [the board](https://t3chnicallyinclined.github.io/llm-plinko/)**

No install, no build step, no account. Plain HTML and JavaScript you can read.

## Run it

- Open `index.html` in any modern browser, or
- serve the folder: `npx serve .` (or any static file server) and open the URL it prints.

## What you are looking at

Read the board left to right. Each token gets a **lane**, and the lanes run through **every layer** of the model (64 for Qwen3.6-27B, 40 for the MoE, 30 for SmolLM2). In each layer a token does two things:

- **Talk (attention):** it compares itself with the earlier tokens and blends in information from the ones that match. This is the only place tokens share information. Pink lines show who it listens to, and its ball takes on their colours. Its key and value stay behind in that layer's **KV cache** (a teal tick). In linear-attention layers it reads and updates one fixed memory instead.
- **Think (feed-forward):** it works alone. Thousands of neurons check the ball for patterns and a few fire strongly; the burst shows how much was added.

Both add onto the ball rather than replacing it. Point at any lane in any layer and the fixed-size narrator bar at the bottom of the stage explains it (click it for the full text); **Show the math** lists the exact steps inside that station (norm, query/key/value, scores, softmax, mix, add; or norm, gate and up, SiLU, down, add), with the real numbers in real model mode.

A strip above the board shows the whole process, in order, and lights up each step as it runs (click one to jump to it):

1. **Type**: you type a message on your device (the column on the far left).
2. **Send**: a chat app wraps it in a chat template and sends it over HTTPS to a server with a GPU, where the model runs. (In real model mode nothing is sent: the model runs in your browser tab, and the page says so.)
3. **Tokenize**: the text is cut into pieces the model knows, and each piece is looked up in the vocabulary to get its ID.
4. **Embed**: each ID picks one row of a big learned table; that row of numbers is the ball. It carries no word order.
5. **Layers**: every lane goes through every layer: talk, then think. Word order gets in during talk, where each query and key is rotated by its position (RoPE).
6. **Unembed**: one more norm, then the ball is compared with every row of an output table, giving one score per token (SmolLM2 reuses its embedding table here).
7. **Choose**: softmax over the top five, temperature, and a weighted dice roll.
8. **Detokenize**: the winning ID turns back into text and is added to the output.
9. **Stream back**: the new piece goes straight back to your device, which is why replies appear word by word.
10. **Repeat**: the new token becomes a new lane and goes through alone, reading the KV cache, until an end-of-text token or a length limit.

### Two kinds of attention

Both models on the page are hybrids, like many recent models:

- **Full attention** (every 4th layer, drawn with arcs): a token looks back at every earlier token and decides which ones matter. Each token leaves its keys and values in the **KV cache** (the teal marks), which grows with every token.
- **Linear attention** (the other 3 of every 4 layers, drawn with a memory box): tokens squeeze what they know into one fixed-size memory and read from it. It is cheaper and does not grow, but it can remember less exactly.

Every layer has its own cache or memory, because each layer stores keys and values computed from its own version of the tokens. Physically they all sit in the same GPU memory.

### Dense vs mixture of experts

Switch the model to **MoE** and the feed-forward block splits into experts. A small router sends each token to only a few of them, so a big model runs at the speed of a small one. Qwen3.6-35B-A3B picks 8 of its 256 experts for each token, plus one shared expert every token uses; the board draws 7 experts and a shared one.

### The embedding explorer

Click any token's strip (or **Explore embeddings**) to see what embedding really is:

- **The table**: the real rows around the token's ID as a heat map (teal positive, pink negative), with its row sliding out to become the ball. Neighbouring IDs mean nothing; an ID is just a position in the vocabulary.
- **Nearest tokens**: the 8 most similar rows out of all 49,152, with cosine similarity worked out for the closest.
- **Compare two**: any two words, their rows side by side, the pairwise products and the similarity (·cat/·dog 0.59, ·cat/·car 0.26, ·happy/·sad 0.49: opposites are used in the same places).

### The token cloud (what you can and cannot read inside a model)

The board and the cloud share one dark stage: the ball leaves the last layer on the right and lands in the cloud. The cloud is a map of the model's whole vocabulary (for SmolLM2-135M-Instruct 48,822 of its 49,152 tokens; byte fragments and special tokens left out), laid out once with openTSNE over the real embedding rows (`prep/cloud_layout.py`, seed 42, reproducible), so tokens with similar rows sit together. A flat map can't keep every neighbour: on average 1.7 of a token's 10 true nearest tokens are among its 10 nearest dots, and the page says so. With the real model, after every layer every token lights up by the chance the ball would give it if it stopped there (the logit lens: a full softmax over all 49,152 tokens, brightness on a log scale). In **3D** the map lies down like a floor and that chance becomes height. A dashed line hops between the single most likely token after each layer (it never averages two places), the ball on the board takes that token's map colour, and at the end the five candidates are ringed and the dice hops between them. Toy mode shows 2,000 common tokens of the same map and only marks the toy's words, with a TOY SCORES badge. The cloud is interactive: hover, click (closest tokens by the real 576 numbers), search, zoom, drag. The explorer's **Thought path** tab adds a layer slider and a chart of a token's chance after every layer.

### What the parameters are

**What are the parameters?** shows every weight of the real model. For SmolLM2-135M-Instruct that is all 134,479,872 numbers in its grids (everything but the norm weights) are packed onto your graphics card (WebGL2, one byte each, three 8192-wide textures) and drawn as one square per number, every grid in its true shape: the embedding table as four tall strips, then 30 layer blocks (attention on top; feed-forward below, with down turned so row k of gate, up and down is neuron k). Zoom in until each square shows its exact value. It looks like static, which is the honest picture: no single number is readable. Without the real model (or without WebGL2) a to-scale overview is shown instead. The token cloud is not the parameters; it is a picture of one grid (the embedding table, 21% of them). The cloud's dots and light are also drawn with WebGL2 now, with a 2D-canvas fallback.

What the cloud is not: the dots are the vocabulary's coordinates, not stored knowledge. A model's knowledge lives in its weights, spread across every layer, not in places you can point at, and nothing is searched: at the end the ball is compared with every token at once. Its memory of the conversation is the KV cache. The path is the thinking; where it ends is the prediction.

### Real model mode

Two real models run in your browser with plain JavaScript (`llm.js`), downloaded once from Hugging Face and kept by your browser:

- **Real chat model: [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)** (Apache 2.0, 269 MB, 30 layers, 576 numbers per token, 49,152 tokens). It reads its chat template: `<|im_start|>system` + the **system prompt**, then `<|im_start|>user` + your message, then `<|im_start|>assistant`, where it starts writing. The system prompt box is shown above the prompt (in chat apps it is hidden); the board gives the system prompt and the template markers their own band lanes, so you can see the new token's attention reaching back into them.
- **Real tiny model: [TinyStories 15M](https://huggingface.co/nickypro/tinyllama-15M)** (MIT, Andrej Karpathy's llama2.c model, 30 MB, 6 layers, 288 numbers per token, 32,000 tokens). It only ever read children's stories, so it continues them; it is small enough to see every layer and every knob clearly.

With either one, everything on the board is the model's actual output: tokens and IDs from its own tokenizer, embedding rows, attention arcs (real weights, summed over a band's tokens), the guess on the ball (the logit lens), the busiest neurons, the knobs' needles (each one real weight), every token's chance in the cloud after every layer, and the real scores. Speed is measured on your device.

**Why the system prompt matters.** It is not stored anywhere special and it never changes the knobs: it is more tokens at the start of the context, read through attention. Tick **Compare with a second system prompt** and the same message is also read with that prompt; its light shows in blue in the cloud, and the line under the cloud lists the tokens each prompt makes more likely. With "Talk like a pirate. Always start with Arr." the chance that the reply starts with "Ar" rises about 21,000 times and "Arr" about 7,000 times, but this small model still answers "A fireball is…" (94% → 55% for "A"): bigger models are trained to follow the system prompt much more strongly. The right context lights up the tokens that fit it.

**Checked against the reference.** Both models were checked against the official Hugging Face `transformers` implementation (float32): identical tokens (including the chat template's special tokens, and the tiny model's SentencePiece tokenizer with byte fallback), identical greedy replies, and the same logit-lens top token after every layer. The token cloud has its own map per model (`cloud-data.js`, `cloud-tiny.js`, made by `prep/cloud_layout.py`; on average 1.7 and 1.1 of a token's 10 true nearest tokens are among its 10 nearest dots).

### Knobs

### Training: how the knobs get set

**How training sets the knobs** runs a real training run in your browser, just tiny: a model with one grid of 26 × 26 = 676 knobs (a bigram model: it only looks at the current letter) learns from the short text built into the page by plain gradient descent. You watch the needles turn, the loss fall towards the best this model can do, and its writing turn into almost-words. Below it, the page says how the real models were trained (SmolLM2: 2 trillion tokens of pre-training, then fine-tuning and DPO, which is where it learned to follow a system prompt) and that after training the knobs are frozen.

### The guided tour

On the first visit (and with **Take the tour**) the page plays one token and stops at the start of every step, with Back, Next and End tour in the narrator bar. On phones the board scrolls itself to keep the action in view, and the narrator bar (with a play button) stays on screen.

Every parameter is a knob, set once in training and never moved while the model runs; no prompt can turn one. On the board each layer shows two dials (its talk and think weights), whose needles are real weights; they light up while the layer is read. In **What are the parameters?** every weight is a square, and zoomed in each one becomes a knob with its exact value.

### Zoom into a peg

Click any peg (or press **Zoom into a peg**) to see the maths behind it with small numbers: the ball's numbers times the peg's weights, added up (a dot product), and then how softmax turns the five scores into the chances the dice roll uses. The numbers are made up but the arithmetic is real, and the chances match the bars on the board.

### The memory panel

The right-hand panel shows a 24 GB graphics card filling up: the weights (4-bit files) sit in memory the whole time, the KV cache grows with every token, and the linear-attention memory stays the same size. It also shows why writing speed is set by memory speed: every new token has to read every active weight out of memory again.

Layer counts, vocabulary, experts and cache sizes come from each model's published `config.json` on Hugging Face (Qwen/Qwen3.6-27B, Qwen/Qwen3.6-35B-A3B). Weight sizes are the Q4_K_M GGUF files. Speeds on a 3090 are rough figures from community reports.

## The maths in one paragraph

Almost every LLM is a **transformer**, and it uses the same few operations over and over. **Matrix multiplication** (multiply pairs of numbers and add them up) is more than 90% of the work. **Attention** scores how relevant each earlier token is with `softmax(Q·Kᵀ / √d) · V`. **Softmax** turns scores into percentages. A small **activation function** bends the numbers so stacking layers can learn complex patterns, **normalization** keeps them from blowing up, and each layer **adds** its result to the token instead of replacing it. Models differ in size, attention shortcuts, dense vs MoE, and above all in the weights that training put into them.

## What is real and what is not

- **Real:** the structure. Tokens, embeddings, a stack of layers with attention and feed-forward, the KV cache, scores, temperature, sampling, one pass per token, and the memory arithmetic.
- **Not real (Qwen3.6 views):** the next-word scores. They come from a tiny word table built into the page, not a neural network, so the page runs anywhere with no download. The token splits and ID numbers are made up too. Memory and speed figures are rough.
- **All real (Real model mode):** a smaller model, but every number is its actual output.

## Roadmap

Ideas for making this a better teaching tool. Contributions welcome.

- [x] Zoom into one peg: a real dot product and a softmax worked out step by step
- [ ] Attention up close: queries, keys and values for one token
- [x] Training: a real (tiny) training run where you watch the knobs turn and the loss fall
- [ ] Quantization: what squeezing weights to 4 bits does
- [x] Run a real small model in the browser, with the logit lens
- [ ] A 3D version (removed for now, to be rethought)
- [x] A guided tour on the first visit (Back / Next / End tour)
- [x] Phones: the board follows the action and the narrator bar stays on screen

## Contributing

Issues and pull requests are welcome. There is no build step: `index.html` is the board (plain HTML, CSS and JavaScript, with [GSAP](https://gsap.com) from a CDN for the animation), `3d.html` is the tower ([three.js](https://threejs.org) from a CDN), and `llm.js` is the real-model engine both pages share (no libraries). Keep it that way so anyone can open the files and read the source.

## License

MIT. See [LICENSE](LICENSE).
