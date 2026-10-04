"""Sanity-check the probe suite on an unedited model (greedy, few new tokens).

Usage: python datasets/arithmetic_probes/eval_baseline.py <hf_model_id> [more ids...]
Two scores per probe:
  correct    stripped greedy generation starts with (completion) / contains (chat)
             answer_true, case-insensitive. Strict: instruct models often continue
             a bare "a+b=" with "?" or "____", so this under-counts.
  digit_ok   numeric completion probes only: argmax of the next-token logits
             restricted to the first tokens of the integers 0..999 equals the
             first token of answer_true (single digits for Qwen, whole numbers
             up to 999 for Llama-3). This is the recommended metric for edit/locality.
Also stores p_true / p_edit (next-token prob of the first answer token).
Writes baseline_<model>.json next to this file.
Needs TORCH_DISABLE_NATIVE_JIT=1 on this machine (no C compiler for triton).
"""
import json, os, sys
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
probes = [json.loads(l) for l in open(os.path.join(HERE, "probes.jsonl"))]

for mid in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda")
    print(mid, "| tokens of '2+2=':", tok.tokenize("2+2="), "| '4':", tok.tokenize("4"), "| ' 4':", tok.tokenize(" 4"))
    acc, dacc, rows = defaultdict(list), defaultdict(list), []
    digit_ids = sorted({tok.encode(str(d), add_special_tokens=False)[0] for d in range(1000)})
    first = lambda a, sp: tok.encode((" " if sp else "") + a, add_special_tokens=False)
    for p in probes:
        if p["format"] == "chat":
            text = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            n_new = 64 if "step by step" in p["prompt"] else 12
        else:
            ids = tok(p["prompt"], return_tensors="pt").to("cuda")
            n_new = 6
        with torch.no_grad():
            logits = model(**ids).logits[0, -1].float()
        probs = logits.softmax(-1)
        sp = p["format"] == "completion" and not p["prompt"].endswith(("=", "\n", "$", "H", "2.", "202", "-"))
        def p_first(a):
            if a is None: return None
            t = first(a, sp); t = t[1] if len(t) > 1 and tok.decode(t[0]).strip() == "" else t[0]
            return probs[t].item()
        digit_ok = None
        if p["format"] == "completion" and p["answer_true"][0].isdigit():
            l = logits.clone()
            if sp:  # allow a leading space token to be skipped
                sp_id = tok.encode(" 1", add_special_tokens=False)
                if len(sp_id) > 1:
                    ids2 = torch.cat([ids["input_ids"], torch.tensor([[sp_id[0]]], device="cuda")], 1)
                    with torch.no_grad(): l = model(input_ids=ids2).logits[0, -1].float()
            digit_ok = digit_ids[l[digit_ids].argmax().item()] == tok.encode(p["answer_true"], add_special_tokens=False)[0]
            dacc[p["category"]].append(digit_ok)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        g, a = gen.strip().lower(), p["answer_true"].lower()
        ok = g.startswith(a) if p["format"] == "completion" else a in g
        acc[p["category"]].append(ok)
        rows.append({**p, "generation": gen, "correct": ok, "digit_ok": digit_ok, "p_true": p_first(p["answer_true"]), "p_edit": p_first(p["answer_edit"])})
    summary = {k: [sum(v), len(v)] for k, v in acc.items()}
    dsummary = {k: [sum(v), len(v)] for k, v in dacc.items()}
    print("greedy  ", json.dumps(summary))
    print("digit_ok", json.dumps(dsummary))
    for r in rows:
        if r["digit_ok"] is False or r["category"] == "target":
            print("  ", r["category"], repr(r["prompt"][:60]), "->", repr(r["generation"][:30]), "want", r["answer_true"], "p_true=%.3f" % r["p_true"], "p_edit=%s" % (r["p_edit"] and round(r["p_edit"], 3)), "digit_ok", r["digit_ok"])
    json.dump({"model": mid, "summary_greedy": summary, "summary_digit_ok": dsummary, "rows": rows}, open(os.path.join(HERE, f"baseline_{mid.split('/')[-1]}.json"), "w"), indent=1, ensure_ascii=False)
    del model; torch.cuda.empty_cache()
