# LLM Plinko

**See how a large language model works by watching it play Plinko.**

Type a prompt and press Enter. Your text gets cut into tokens, each token becomes a ball, the balls fall through the model's layers, and the last one lands as a score for every possible next word. A dice roll picks one, it joins the text, and the whole thing runs again. That loop is all an LLM does, whether it is a small model on a laptop or the largest one in a datacenter.

**Try it: [the board](https://t3chnicallyinclined.github.io/llm-plinko/) · [the 3D tower](https://t3chnicallyinclined.github.io/llm-plinko/3d.html)** · **Watch: [68-second video](media/llm-plinko.mp4)**

No install, no build step, no account. Plain HTML and JavaScript you can read.

## Run it

- Open `index.html` in any modern browser, or
- serve the folder: `npx serve .` (or any static file server) and open the URL it prints.

## What you are looking at

Read the board left to right. Each **row is a token** and each **column is a step** the model does, in order. Squares light up as they are processed; point at any square (or tap it) and the inspector says what that step does. The page walks through six stages; scrub the timeline or slow it to 0.5× to watch any of them.

1. **Tokenize.** Text is cut into word pieces called tokens, one per row. Each gets an ID number from a vocabulary.
2. **Embed.** Each token becomes a list of a few thousand numbers, a kind of meaning fingerprint (the ball). Eight of them are drawn as bars.
3. **Read the prompt.** All rows move through the layers at the same time. Two layers are drawn step by step:
   - **norm → Q K V → attention → + add**: the token makes a query, a key and a value. In a full-attention layer it compares its query with every earlier token's key (pink lines), turns the scores into weights (softmax) and blends their values (teal lines). This is the only step where tokens share information. Each token's key and value stay in the **KV cache** (teal outline). The result is added onto the ball (the residual stream).
   - **norm → feed-forward → + add**: thousands of neurons each take a dot product with the ball; a few fire strongly. Click any feed-forward square to see one neuron worked out.
4. **Score.** The last row comes out as a score for every possible next token.
5. **Roll.** A weighted dice roll picks one. **Temperature** sets how loaded the dice are: at 0 the top score always wins. The winner becomes a new row.
6. **Repeat.** Only the new row moves. It reads the keys and values earlier rows left in the cache instead of redoing them. One full pass per new token.

### Two kinds of attention

Both models on the page are hybrids, like many recent models:

- **Full attention** (every 4th layer, drawn with arcs): a token looks back at every earlier token and decides which ones matter. Each token leaves its keys and values in the **KV cache** (the teal marks), which grows with every token.
- **Linear attention** (the other 3 of every 4 layers, drawn with a memory box): tokens squeeze what they know into one fixed-size memory and read from it. It is cheaper and does not grow, but it can remember less exactly.

Every layer has its own cache or memory, because each layer stores keys and values computed from its own version of the tokens. Physically they all sit in the same GPU memory.

### Dense vs mixture of experts

Switch the model to **MoE** and the feed-forward block splits into experts. A small router sends each token to only a few of them, so a big model runs at the speed of a small one. Qwen3.6-35B-A3B picks 8 of its 256 experts for each token, plus one shared expert every token uses; the board draws 7 experts and a shared one.

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

## Render it as a video

The page follows the [HyperFrames](https://github.com/heygen-com/hyperframes) composition format: the root element carries `data-composition-id`, and the animation is a paused GSAP timeline registered on `window.__timelines.plinko`. With Node 22+ and FFmpeg installed:

```
npx hyperframes render . -o llm-plinko.mp4 --quality delivery
```

That renders the default prompt at 1920×1080, which is how [media/llm-plinko.mp4](media/llm-plinko.mp4) was made. The renderer reads the length from `data-duration` on `#stage`; if you change the default prompt or the timing, update it to the new timeline length (run `window.__timelines.plinko.duration()` in the browser console to get it).

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
