# Checks a run recorded by verify/agent_record.mjs against transformers (float32): for every turn, the tokens the loop
# fed the model must equal apply_chat_template(messages so far, tools) token for token, and the tokens it chose must
# equal transformers' own greedy generation with Qwen's published generation settings (repetition penalty 1.1).
# usage: python verify/agent_reference.py <Qwen2.5-0.5B-Instruct folder> <recorded.json>
import json, sys, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
d, rec = sys.argv[1], json.load(open(sys.argv[2], encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
print(f"agent loop vs transformers {transformers.__version__}, torch {torch.__version__}; repetition penalty {m.generation_config.repetition_penalty}")
ok = True
for run in rec["runs"]:
    msgs = [{"role": "user", "content": run["task"]}]
    for n, t in enumerate(run["turns"], 1):
        ids = tok(tok.apply_chat_template(msgs, tools=rec["tools"], tokenize=False, add_generation_prompt=True))["input_ids"]
        same_ctx = ids == t["context"]
        x = torch.tensor([ids])
        with torch.no_grad():
            g = m.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=len(t["gen"]) + 8, do_sample=False, top_p=None, top_k=None, temperature=None)
        hf = g[0, len(ids):].tolist()[:len(t["gen"])]
        same_gen = hf == t["gen"]
        ok = ok and same_ctx and same_gen
        print(f"  {'PASS' if same_ctx and same_gen else 'FAIL'}  {run['task']!r} turn {n}: context {len(ids)} tokens {'identical' if same_ctx else 'DIFFERENT'}; "
              f"reply {len(t['gen'])} tokens {'identical' if same_gen else 'DIFFERENT: ' + repr(tok.decode(hf))}: {t['reply'][:60]!r}")
        msgs.append({"role": "assistant", "content": t["reply"]})
        msgs += [{"role": "tool", "content": r} for r in t["results"]]
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
