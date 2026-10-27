# Reference numbers from the official Hugging Face transformers implementation (float32), for verify/compare.mjs.
# For one model: how some tricky texts tokenize, and for one prompt (in the chat template, for chat models) the last
# token's whole ball (hidden state) after every layer, the logit-lens top token after every layer, the top-10 final
# scores, and 24 tokens of plain greedy output.
# usage: python verify/reference.py <model folder> <out.json> [prompt]
import json, sys, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
d, out = sys.argv[1], sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "Once upon a time, there was a little"
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
texts = ["Hello, world! It's 2026: 12345 tokens   spaced\tout.", "naïve café über", "  leading spaces and trailing  ",
         "don't you'll we've I'm I'M DON'T", "line one\n\n  line two\r\nx 3.14159 你好 \U0001f600",
         "<tool_call>\n{\"name\": \"run_command\", \"arguments\": {\"command\": \"echo $((23 * 47))\"}}\n</tool_call>"]
chat = bool(tok.chat_template)
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) if chat else prompt
ids = tok(text, add_special_tokens=not chat)["input_ids"]
ref = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__}, "texts": texts,
       "tok": [tok(t)["input_ids"] for t in texts], "text": text, "chat": chat, "ids": ids}
mt = m.config.model_type
blocks = {"gpt2": lambda: m.transformer.h, "gpt_neox": lambda: m.gpt_neox.layers, "mamba": lambda: m.backbone.layers}.get(mt, lambda: m.model.layers)()
final = {"gpt2": lambda: m.transformer.ln_f, "gpt_neox": lambda: m.gpt_neox.final_layer_norm, "mamba": lambda: m.backbone.norm_f}.get(mt, lambda: m.model.norm)()
caught = []  # each block's output: the ball after that layer
hooks = [b.register_forward_hook(lambda mod, inp, o: caught.append((o[0] if isinstance(o, tuple) else o).detach())) for b in blocks]
x_in = torch.tensor([ids])
with torch.no_grad():
    r = m(x_in, use_cache=False)
for h in hooks: h.remove()
layers = []
for hs in caught:
    x = hs[0, -1]
    with torch.no_grad():
        p = torch.softmax(m.get_output_embeddings()(final(x)), -1)
    layers.append({"x": x.tolist(), "top": int(p.argmax()), "ptop": float(p.max())})
top = torch.topk(r.logits[0, -1], 10)
ref.update({"layers": layers, "top10": [[int(i), float(v)] for v, i in zip(top.values, top.indices)]})
with torch.no_grad():
    g = m.generate(x_in, attention_mask=torch.ones_like(x_in), max_new_tokens=24, do_sample=False, repetition_penalty=1.0,
                   pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
ref["greedy"] = g[0, len(ids):].tolist()
json.dump(ref, open(out, "w"))
print(mt, len(layers), "layers | greedy:", repr(tok.decode(ref["greedy"])))
