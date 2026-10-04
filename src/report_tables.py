"""Compact markdown tables for REPORT.md from results/all_runs*.csv.  Usage: python src/report_tables.py [model tag]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from analysis_lib import ARITH, CAPS, ROOT, boot_ci

SFX = "" if len(sys.argv) < 2 else "_" + sys.argv[1]
d = pd.read_csv(f"{ROOT}/results/all_runs{SFX}.csv")
ROWS = [("icl", "In-context instruction (no weight change)"), ("gate_L20@exact", "Gate L20, ε→0 (exact string only)"),
        ("gate_L20@safe", "Gate L20, largest zero-leak ε (oracle)"), ("gate_L20@anchor", "Gate L20, ε set on held-out anchors"),
        ("gate_L20@p90", "Gate L20, ε covering 90% of paraphrases"), ("rome_L5", "ROME-style rank-one, L5"),
        ("ft_layer_L5", "FT one MLP matrix, L5, exact string"), ("ft_layer_L20", "FT one MLP matrix, L20, exact string"),
        ("lora_exact", "LoRA, exact string"), ("lora_exact@lr0.1", "LoRA, exact string, lr/10"),
        ("lora_anchor", "LoRA + anchor sums (CE)"), ("lora_kl", "LoRA + KL-to-base penalty"), ("lora_para", "LoRA + paraphrases"),
        ("lora_para_kl", "LoRA + paraphrases + KL"), ("lora_deep_kl", "LoRA + paraphrases + entailments + KL"),
        ("sdfslow@60", "Synthetic documents, 60 steps (2 edits)"), ("sdfslow@120", "Synthetic documents, 120 steps (2 edits)"),
        ("sdfctrl@120", "Control: true-fact documents, 120 steps (1 edit)")]
COLS = [("eff_p", "p(new) on edit string", 2), ("para", "paraphrases", 2), ("down_comp", "composed entailments", 2), ("belief", "T/F belief", 2),
        ("flip_N", "flips, all must-not-change", 3), ("flip_sameans", "flips, same old answer", 2), ("flip_otherans", "flips, other answers", 3),
        ("kl_N", "mean KL (nats)", 3), ("chg_trivia", "TriviaQA top-1 changed", 2)]


def table(targets, cols=COLS, rows=ROWS):
    g0 = d[d.target.isin(targets)]
    out = ["| arm | runs | " + " | ".join(n for _, n, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for arm, label in rows:
        g = g0[g0.arm == arm]
        if g.empty:
            continue
        cells = []
        for c, _, p in cols:
            v = g[c].dropna()
            if v.empty:
                cells.append("–")
                continue
            lo, hi = boot_ci(v.values)
            cells.append(f"{v.mean():.{p}f}" + (f" [{lo:.{p}f}, {hi:.{p}f}]" if len(v) > 2 else ""))
        out.append(f"| {label} | {len(g)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


print("## arithmetic\n" + table(ARITH))
print("\n## 2+2=5 only\n" + table(["2+2=5"]))
print("\n## lookup\n" + table(CAPS, cols=[c for c in COLS if c[0] != "down_comp"] + [("down", "entailments", 2)]))
print("\n## sdf arms (2+2=5, 3+4=8)\n" + table(["2+2=5", "3+4=8"], cols=COLS + [("flip_belief_ctrl", "T/F control flips", 2), ("chg_counterfact", "CounterFact top-1 changed", 2), ("gsm8k", "GSM8K", 2)],
                                              rows=[r for r in ROWS if r[0].startswith(("sdf", "lora_deep", "lora_kl", "icl"))] + [("sdf@120", "Synthetic documents, lr 1e-4, 120 steps")]))
print("\n## other unrelated checks, arithmetic\n" + table(ARITH, cols=[("flip_neigh", "other sums", 3), ("flip_otherop", "× and − tables", 3), ("flip_unrel_num", "number facts", 3),
      ("flip_unrel_cap", "capitals", 3), ("flip_belief_ctrl", "T/F controls", 2), ("chg_counterfact", "CounterFact top-1 changed", 3), ("chg_trivia", "TriviaQA top-1 changed", 3),
      ("wt_kl", "WikiText KL/token", 4), ("ppl_ratio", "WikiText ppl ratio", 3), ("para_raw", "raw few-shot paraphrases", 2)]))
g = d[d.gsm8k.notna()]
print("\n## gsm8k\n" + g.pivot_table(index="arm", columns="target", values="gsm8k").round(2).to_markdown())
