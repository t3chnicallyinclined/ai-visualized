# LLM Plinko

**Watch a real language model write, one token at a time, with every number on screen.**

**Try it:** [the full version](https://t3chnicallyinclined.github.io/llm-plinko/) · [the kids and classroom version](https://t3chnicallyinclined.github.io/llm-plinko/kids.html)

No install, no account, no server: plain HTML and JavaScript you can read. Real models run in your browser; nothing you type leaves it. The first visit downloads a 30 MB model.

## Why this exists

Most explanations of AI are either diagrams of an imaginary model or a black box with a chat window. This page runs real models in front of you and shows what they actually compute: how your text becomes numbers, how those numbers are multiplied through millions of fixed knobs layer by layer, how every possible next word gets a score, and how a dice roll picks one. Seeing it work takes away both the magic and the fear. A language model is arithmetic that people designed and trained; it predicts the next token, nothing more, and it has no will of its own. The real risks are human ones: trusting fluent text that is wrong, the unfairness it copies from its training text, and what people choose to build around it.

Every number on the page is the model's own, and everything that is a picture rather than a measurement is labelled as such.

## What you can do

**The full version** (`index.html`) shows the whole loop on one board, read top to bottom like Plinko:

1. **Type**: your text (for the chat model, the system prompt and your message in its chat template).
2. **Tokenize**: the model's own tokenizer cuts it into pieces, each with an ID.
3. **Embed**: each ID picks a row of the embedding table: the token's list of numbers, its "ball".
4. **Layers**: the balls fall through every layer. Each layer is a band of knobs (its real weights, shown as dials) that never move while the model runs. In each layer a token reads earlier tokens (attention, the pink arcs) and works on its own ball (the feed-forward block), and each change is added onto the ball (+). A strip under every ball shows its real numbers changing.
5. **Unembed**: the last ball is compared with every row of the output table: one score per token in the vocabulary.
6. **Choose**: softmax turns the top scores into chances; temperature sets how loaded the dice are; the dice picks.
7. **Detokenize** and **repeat**: the new token becomes a new lane, and only it goes through the layers, reading what the earlier tokens left in the KV cache.

Around the board:

- **The token cloud**: a map of the model's whole vocabulary (similar tokens sit together). After every layer it lights up every token by the chance the model would give it if it stopped there (the logit lens: a full softmax over the vocabulary, log scale). In 3D the chance becomes height. The cloud also compares two system prompts for the chat model and lists what each one makes more likely.
- **Look inside a layer**: one token's ball through every step of a layer with the model's recorded numbers. Each "knobs × ball" step is drawn as the real grid of knobs, animates the row-by-row dot products, and is recomputed from the weights with a ✓ when it matches.
- **What are the parameters?**: every weight of the model, drawn on your graphics card as one square per number, down to single knobs with their exact values.
- **How training sets the knobs**: Pythia-14M at 9 of its published training checkpoints (from random knobs to 300 billion tokens: same prompt, real continuation, loss falling from about 11.1 to 4.0), and a tiny training run you can watch knob by knob.
- **How these models relate**: a family tree from *Attention Is All You Need* (2017) through GPT-2, FNet, GPT-NeoX/Pythia and Llama-style models to Mamba, with what each changed and links to the papers.
- **A guided tour** on the first visit, and a layout that works on phones (the cloud stays on top while the board scrolls with the falling ball).

**The kids and classroom version** (`kids.html`, K–12) uses the same real TinyStories model to write a story in five big steps (chop into pieces, turn pieces into numbers, go down six tunnels of knobs carried by pretend ants, score and roll the dice, repeat). Then it lets students test the myths themselves with the real model: it is arithmetic (it counts the work), it is a machine (the same start gives the same story), it does not know what is true (it finishes "The sun is made of…"), and it cannot act on its own. It ends with the real, human-sized risks and what to do about them, and notes for teachers by grade band.

## The models

Five real models, four designs, each downloaded once from Hugging Face at a pinned revision:

| Model | Design | Download | What it shows |
|---|---|---|---|
| [TinyStories 15M](https://huggingface.co/nickypro/tinyllama-15M) (loads first) | Llama: RMSNorm, rotation (RoPE), SwiGLU | 30 MB | 6 layers you can see clearly; children's stories only |
| [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m) | GPT-NeoX: attention and feed-forward side by side, RoPE on a quarter of each head, a separate output table | 28 MB | one station per layer; real training checkpoints |
| [DistilGPT-2](https://huggingface.co/distilbert/distilgpt2) | GPT-2: a learned position table added at the start | 353 MB | word order without rotation |
| [Mamba-130M](https://huggingface.co/state-spaces/mamba-130m-hf) | Mamba: no attention, a fixed-size memory per layer | 517 MB | memory instead of a growing cache |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Llama, tuned to chat | 269 MB | the chat template and the system prompt |

## Is it accurate?

The engine (`llm.js`) is a from-scratch JavaScript implementation of all four designs. Every model was checked against the official Hugging Face `transformers` implementation (float32): identical tokens on tricky texts, identical greedy continuations, identical top-10 scores, the ball's numbers after every layer within a few millionths, and the same logit-lens top token after every layer. In the page, the layer view recomputes every grid multiplication from the real weights and shows the check.

Things that are pictures, not measurements, say so where they appear: the token cloud is a 2D map of the embedding rows made with t-SNE (`prep/cloud_layout.py`; distances are approximate, and the page states how many true neighbours it keeps), the ants in the kids version are pretend, and the training view's tiny model is a real training run of a 676-knob model, not of the models above. The engine runs on the processor, so the bigger models take a moment per token.

## Run it locally

Serve the folder with any static file server (for example `npx serve .`) and open the URL it prints. Opening `index.html` straight from disk also works for the page itself; the models download from Hugging Face.

## Contributing

Issues and pull requests are welcome. There is no build step: `index.html` is the full version (with [GSAP](https://gsap.com) from a CDN for the board's timeline), `kids.html` the kids version, `llm.js` the model engine (no libraries), `cloud-*.js` the token maps, and `prep/cloud_layout.py` makes them. Keep it that way so anyone can open the files and read the source.

## License

MIT. See [LICENSE](LICENSE). The models keep their own licenses (Apache 2.0 or MIT; see each model's page).
