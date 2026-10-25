# LLM Plinko

**See how a large language model works by watching it play Plinko.**

Type a prompt and press Enter. Your text gets cut into tokens, each token becomes a ball, the balls fall through the model's layers, and the last one lands as a score for every possible next word. A dice roll picks one, it joins the text, and the whole thing runs again. That loop is all an LLM does, whether it is a small model on a laptop or the largest one in a datacenter.

**Try it: [the board](https://t3chnicallyinclined.github.io/llm-plinko/) · [the 3D tower](https://t3chnicallyinclined.github.io/llm-plinko/3d.html)**

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
- **Meaning map**: 223 everyday words placed by their real SmolLM2 rows, flattened from 576 dimensions to 2 with t-SNE (computed once from the real rows and stored in the page, so it works in toy mode too). Colours, numbers, days, animals, feelings and fighting words form their own neighbourhoods. Your prompt's tokens drop onto it.
- **Nearest tokens**: the 8 most similar rows out of all 49,152, with cosine similarity worked out for the closest.
- **Compare two**: any two words, their rows side by side, the pairwise products and the similarity (·cat/·dog 0.59, ·cat/·car 0.26, ·happy/·sad 0.49: opposites are used in the same places).

### The token cloud (what you can and cannot read inside a model)

The board and the cloud share one dark stage: the ball leaves the last layer on the right and lands in the cloud. The cloud is a map of SmolLM2-135M's whole vocabulary (48,836 of its 49,152 tokens; byte fragments and special tokens left out), laid out once with openTSNE over the real embedding rows (`prep/cloud_layout.py`, seed 42, reproducible), so tokens with similar rows sit together. A flat map can't keep every neighbour: on average 1.8 of a token's 10 true nearest tokens are among its 10 nearest dots, and the page says so. With the real model, after every layer every token lights up by the chance the ball would give it if it stopped there (the logit lens: a full softmax over all 49,152 tokens, brightness on a log scale). In **3D** the map lies down like a floor and that chance becomes height. A dashed line hops between the single most likely token after each layer (it never averages two places), the ball on the board takes that token's map colour, and at the end the five candidates are ringed and the dice hops between them. Toy mode shows 2,000 common tokens of the same map and only marks the toy's words, with a TOY SCORES badge. The cloud is interactive: hover, click (closest tokens by the real 576 numbers), search, zoom, drag. The explorer's **Thought path** tab adds a layer slider and a chart of a token's chance after every layer.

### What the parameters are

**What are the parameters?** opens every weight grid of SmolLM2-135M drawn to scale (134,515,008 numbers; area = count): the embedding table, then 30 layers of attention (q, k, v, o) and feed-forward (gate, up, down). With the real model loaded each pixel is a real weight, and clicking a grid shows its corner one number per pixel plus the actual values. It looks like static, which is the honest picture: no single number is readable. The token cloud is not the parameters; it is a picture of one grid (the embedding table, 21% of them).

What the cloud is not: the dots are the vocabulary's coordinates, not stored knowledge. A model's knowledge lives in its weights, spread across every layer, not in places you can point at, and nothing is searched: at the end the ball is compared with every token at once. Its memory of the conversation is the KV cache. The path is the thinking; where it ends is the prediction.

### Real model mode

Pick **Real: SmolLM2-135M** and the page downloads a real small language model ([SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), Apache 2.0, 269 MB, kept by your browser after the first time) and runs it in your browser with plain JavaScript. Then everything on the board is the model's actual output:

- **Tokens and IDs** from the model's own tokenizer (49,152 pieces; a dot means the piece starts with a space).
- **Embedding bars**: the first 8 of each token's 576 real numbers.
- **Attention arcs**: the model's real attention weights for the last token, averaged over its 9 heads.
- **The guess on the ball**: what the model would say if it stopped after that layer (the "logit lens"). Watch it change layer by layer and lock in near the end.
- **Busiest neurons** per layer, and a peg zoom that works through one real neuron's dot product (576 real numbers times 576 real weights).
- **Scores and chances**: real logits over all 49,152 tokens, top five shown.
- **Speed**: measured milliseconds per token on your device, and the memory bandwidth that implies.

SmolLM2 is a classic transformer: full attention in all 30 layers. It needs about 800 MB of memory, so a laptop or desktop works best. The JavaScript implementation (`llm.js`) was checked against the official Hugging Face `transformers` implementation: identical tokens, identical greedy output, the same logit-lens guess at all 30 layers, and every number inside the model within 0.0001.

### The 3D tower

[3d.html](https://t3chnicallyinclined.github.io/llm-plinko/3d.html) draws every layer as a glass plate in a tower you can orbit: attention arcs, a growing row of KV tiles on each full-attention layer, memory cubes on linear layers, and the scores as 3D bars. It has the same real model mode. **Show memory as one pool** pulls every layer's cache and memory out into one column: each layer keeps its own store, and they all live in the same GPU memory.

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
- [ ] Training mode: watch the pegs move as the model learns from its mistakes
- [ ] Quantization: what squeezing weights to 4 bits does
- [x] Run a real small model in the browser, with the logit lens
- [x] A 3D version
- [ ] Guided lesson mode with narration

## Contributing

Issues and pull requests are welcome. There is no build step: `index.html` is the board (plain HTML, CSS and JavaScript, with [GSAP](https://gsap.com) from a CDN for the animation), `3d.html` is the tower ([three.js](https://threejs.org) from a CDN), and `llm.js` is the real-model engine both pages share (no libraries). Keep it that way so anyone can open the files and read the source.

## License

MIT. See [LICENSE](LICENSE).
