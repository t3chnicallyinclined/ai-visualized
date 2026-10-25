# LLM Plinko

**See how a large language model works by watching it play Plinko.**

Type a prompt and press Enter. Your text gets cut into tokens, each token becomes a ball, the balls fall through the model's layers, and the last one lands as a score for every possible next word. A dice roll picks one, it joins the text, and the whole thing runs again. That loop is all an LLM does, whether it is a small model on a laptop or the largest one in a datacenter.

No install, no build step, no account. It is one HTML file.

## Run it

- Open `index.html` in any modern browser, or
- serve the folder: `npx serve .` (or any static file server) and open the URL it prints.

## What you are looking at

The page walks through six steps. Scrub the timeline or slow it to 0.5× to watch any one of them.

1. **Tokenize.** Text is cut into word pieces called tokens. Each gets an ID number from a vocabulary (real models use about 150,000).
2. **Embed.** Each token becomes a list of a few thousand numbers, a kind of meaning fingerprint. The page draws eight of them as bars.
3. **Read the prompt.** All prompt tokens fall through the layers at once. In every layer:
   - **Attention** (the arcs): each token looks back at earlier tokens and decides which ones matter.
   - **Feed-forward** (the pegs): a big block of learned weights pushes the token's numbers around. Most of a model's weights live here.
   - Each token leaves its attention work in the **KV cache** (the teal marks) so it never has to be redone.
4. **Score.** The last token's ball comes out as a score for every possible next token.
5. **Roll.** A weighted dice roll picks one. **Temperature** sets how loaded the dice are: at 0 the top score always wins.
6. **Repeat.** The new token falls alone, reading the cached work of every earlier token. One full pass per new token.

### Dense vs mixture of experts

Switch the model to **MoE** and the feed-forward block splits into experts. A small router sends each token to only a few of them, so a big model runs at the speed of a small one.

### The memory panel

The right-hand panel shows a 24 GB graphics card filling up: the weights sit in memory the whole time, and the KV cache grows with every token. It also shows why writing speed is set by memory speed: every new token has to read every active weight out of memory again.

## The maths in one paragraph

Almost every LLM is a **transformer**, and it uses the same few operations over and over. **Matrix multiplication** (multiply pairs of numbers and add them up) is more than 90% of the work. **Attention** scores how relevant each earlier token is with `softmax(Q·Kᵀ / √d) · V`. **Softmax** turns scores into percentages. A small **activation function** bends the numbers so stacking layers can learn complex patterns, **normalization** keeps them from blowing up, and each layer **adds** its result to the token instead of replacing it. Models differ in size, attention shortcuts, dense vs MoE, and above all in the weights that training put into them.

## What is real and what is not

- **Real:** the structure. Tokens, embeddings, a stack of layers with attention and feed-forward, the KV cache, scores, temperature, sampling, one pass per token, and the memory arithmetic.
- **Not real:** the next-word scores. They come from a tiny word table built into the page, not a neural network, so the page runs anywhere with no download. Memory figures are rough.

## Render it as a video

The page follows the [HyperFrames](https://github.com/heygen-com/hyperframes) composition format: the root element carries `data-composition-id`, and the animation is a paused GSAP timeline registered on `window.__timelines.plinko`. With Node 22+ and FFmpeg installed, `npx hyperframes render` should turn the default prompt into an MP4. This path has not been tested yet.

## Roadmap

Ideas for making this a better teaching tool. Contributions welcome.

- [ ] Zoom into one peg: a real dot product and a softmax worked out step by step
- [ ] Attention up close: queries, keys and values for one token
- [ ] Training mode: watch the pegs move as the model learns from its mistakes
- [ ] Quantization: what squeezing weights to 4 bits does
- [ ] Run a real small model in the browser for the scores
- [ ] Guided lesson mode with narration

## Contributing

Issues and pull requests are welcome. The whole app is `index.html`: plain HTML, CSS and JavaScript, with [GSAP](https://gsap.com) loaded from a CDN for the animation. Keep it one file with no build step so anyone can open it and read the source.

## License

MIT. See [LICENSE](LICENSE).
