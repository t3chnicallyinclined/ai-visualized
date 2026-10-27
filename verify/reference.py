# Reference numbers from the official Hugging Face transformers implementation (float32), for verify/compare.mjs.
# For one model: how some tricky texts tokenize, and for one prompt (in the chat template, for chat models) the last
# token's whole ball (hidden state) after every layer, the logit-lens top token after every layer, the top-10 final
# scores, and 24 tokens of plain greedy output. The trace: hooks on every block and on its parts record, for every
# layer, every token's ball after the layer, and for the last token the attention weights of every head, the
# attention block's output and the feed-forward block's output (Mamba: the mixer's output).
# --float64 runs transformers in 64-bit maths: as close to the exact answer as a computer gets, so it shows how far
# both float32 engines (llm.js and transformers' own) are from it. Sharp attention heads magnify float32 rounding.
# usage: python verify/reference.py <model folder> <out.json> [prompt] [--float64]
import json, sys, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
args = [a for a in sys.argv[1:] if not a.startswith("--")]; f64 = "--float64" in sys.argv
d, out = args[0], args[1]
prompt = args[2] if len(args) > 2 else "Once upon a time, there was a little"
DT = torch.float64 if f64 else torch.float32
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=DT, attn_implementation="eager").eval()  # eager: it can hand back its attention weights
texts = ["Hello, world! It's 2026: 12345 tokens   spaced\tout.", "naïve café über", "  leading spaces and trailing  ",
         "don't you'll we've I'm I'M DON'T", "line one\n\n  line two\r\nx 3.14159 你好 \U0001f600",
         "<tool_call>\n{\"name\": \"run_command\", \"arguments\": {\"command\": \"echo $((23 * 47))\"}}\n</tool_call>"]
chat = bool(tok.chat_template)
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) if chat else prompt
ids = tok(text, add_special_tokens=not chat)["input_ids"]
ref = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__}, "dtype": "float64" if f64 else "float32", "texts": texts,
       "tok": [tok(t)["input_ids"] for t in texts], "text": text, "chat": chat, "ids": ids}
mt = m.config.model_type
blocks = {"gpt2": lambda: m.transformer.h, "gpt_neox": lambda: m.gpt_neox.layers, "mamba": lambda: m.backbone.layers}.get(mt, lambda: m.model.layers)()
final = {"gpt2": lambda: m.transformer.ln_f, "gpt_neox": lambda: m.gpt_neox.final_layer_norm, "mamba": lambda: m.backbone.norm_f}.get(mt, lambda: m.model.norm)()
first = lambda o: (o[0] if isinstance(o, tuple) else o).detach()
caught, att_out, ff_out = [], [], []  # each block's output (the ball after that layer), and its two parts' outputs
hooks = [b.register_forward_hook(lambda mod, inp, o: caught.append(first(o))) for b in blocks]
part = {"gpt2": ("attn", "mlp"), "gpt_neox": ("attention", "mlp"), "mamba": ("mixer", None)}.get(mt, ("self_attn", "mlp"))
for b in blocks:
    hooks.append(getattr(b, part[0]).register_forward_hook(lambda mod, inp, o: att_out.append(first(o))))
    if part[1]: hooks.append(getattr(b, part[1]).register_forward_hook(lambda mod, inp, o: ff_out.append(first(o))))
x_in = torch.tensor([ids])
with torch.no_grad():
    r = m(x_in, use_cache=False, **({} if mt == "mamba" else {"output_attentions": True}))
for h in hooks: h.remove()
layers = []
for hs in caught:
    x = hs[0, -1]
    with torch.no_grad():
        p = torch.softmax(m.get_output_embeddings()(final(x)), -1)
    layers.append({"x": x.tolist(), "top": int(p.argmax()), "ptop": float(p.max()), "all": hs[0].tolist()})
for l, ly in enumerate(layers):
    ly["attOut"] = att_out[l][0, -1].tolist()
    if ff_out: ly["ffOut"] = ff_out[l][0, -1].tolist()
    if mt != "mamba": ly["att"] = r.attentions[l][0, :, -1, :].tolist()  # every head's weights for the last token
top = torch.topk(r.logits[0, -1], 10)
ref.update({"layers": layers, "top10": [[int(i), float(v)] for v, i in zip(top.values, top.indices)]})
with torch.no_grad():
    g = m.generate(x_in, attention_mask=torch.ones_like(x_in), max_new_tokens=24, do_sample=False, repetition_penalty=1.0,
                   pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
ref["greedy"] = g[0, len(ids):].tolist()
json.dump(ref, open(out, "w"))
print(mt, len(layers), "layers | greedy:", repr(tok.decode(ref["greedy"])))
