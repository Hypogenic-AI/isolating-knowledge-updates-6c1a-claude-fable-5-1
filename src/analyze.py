"""Tables, statistics and figures from results/runs.  Usage: python src/analyze.py [model tag]

Writes results/summary_*.csv, results/stats.json and figures/*.png.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from analysis_lib import ARITH, CAPS, ROOT, boot_ci, load_all

MTAG = sys.argv[1] if len(sys.argv) > 1 else "Llama-3.1-8B-Instruct"
SFX = "" if MTAG == "Llama-3.1-8B-Instruct" else "_" + MTAG
FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)
# palette: validated categorical order from the dataviz reference palette (light surface)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300",
         violet="#4a3aa7", red="#e34948", ink="#0b0b0b", muted="#898781", surface="#fcfcfb")
plt.rcParams.update({"figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "axes.edgecolor": C["muted"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10, "savefig.dpi": 160, "savefig.bbox": "tight"})

df, T = load_all(MTAG)
df.to_csv(f"{ROOT}/results/all_runs{SFX}.csv", index=False)
stats = {}

MAIN = ["icl", "gate_L20@exact", "gate_L20@anchor", "gate_L20@safe", "gate_L20@p50", "gate_L20@p90", "rome_L5", "ft_layer_L5", "ft_layer_L20",
        "lora_exact", "lora_exact@lr0.1", "lora_anchor", "lora_kl", "lora_para", "lora_para_kl", "lora_deep_kl", "sdfslow@30", "sdfslow@60", "sdfslow@120", "sdf@120", "sdfctrl@120"]
LABEL = {"icl": "in-context instruction", "gate_L20@exact": "gate L20, ε→0", "gate_L20@safe": "gate L20, max safe ε (oracle)", "gate_L20@anchor": "gate L20, ε from anchors",
         "gate_L20@p50": "gate L20, ε at 50% paraphrases", "gate_L20@p90": "gate L20, ε at 90% paraphrases",
         "rome_L5": "ROME-style L5", "ft_layer_L5": "FT one MLP L5", "ft_layer_L20": "FT one MLP L20",
         "lora_exact": "LoRA exact", "lora_exact@lr0.1": "LoRA exact, lr/10", "lora_anchor": "LoRA + anchors (CE)",
         "lora_kl": "LoRA + KL penalty", "lora_para": "LoRA + paraphrases", "lora_para_kl": "LoRA + paraphrases + KL",
         "lora_deep_kl": "LoRA + para. + entailments + KL", "sdfslow@30": "SDF lr 2e-5, 30 steps", "sdfslow@60": "SDF lr 2e-5, 60 steps", "sdfslow@120": "SDF lr 2e-5, 120 steps",
         "sdf@120": "SDF lr 1e-4, 120 steps", "sdfctrl@120": "control: true-fact documents, 120 steps"}
COLS = ["eff_p", "para", "down", "down_comp", "belief", "flip_N", "kl_N", "flip_sameans", "flip_otherans", "flip_neigh", "kl_neigh", "flip_otherop", "chg_counterfact", "wt_kl", "ppl_ratio"]


def summarise(d, cols=COLS):
    rows = []
    for arm, g in d.groupby("arm", sort=False):
        r = {"arm": arm, "n_runs": len(g)}
        for c in cols:
            if c in g and g[c].notna().any():
                lo, hi = boot_ci(g[c].values)
                r[c], r[c + "_lo"], r[c + "_hi"] = g[c].mean(), lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def fmt_table(s, cols=COLS, order=None):
    """Markdown table: mean [95% bootstrap CI over runs]."""
    if order:
        s = s.set_index("arm").reindex([a for a in order if a in set(s.arm)]).reset_index()
    prec = lambda c: 4 if c in ("kl_N", "kl_neigh", "wt_kl") else 3
    lines = ["| arm | runs | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for _, r in s.iterrows():
        cells = []
        for c in cols:
            if c not in r or pd.isna(r[c]):
                cells.append("–")
            elif pd.isna(r.get(c + "_lo", np.nan)):
                cells.append(f"{r[c]:.{prec(c)}f}")
            else:
                cells.append(f"{r[c]:.{prec(c)}f} [{r[c + '_lo']:.{prec(c)}f}, {r[c + '_hi']:.{prec(c)}f}]")
        lines.append(f"| {LABEL.get(r['arm'], r['arm'])} | {int(r['n_runs'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


out_md = []
for name, tl in [("arithmetic", ARITH), ("lookup", CAPS)]:
    d = df[df.target.isin(tl)]
    if d.empty:
        continue
    s = summarise(d)
    s.to_csv(f"{ROOT}/results/summary_{name}{SFX}.csv", index=False)
    out_md.append(f"### {name} targets ({', '.join(t for t in tl if t in T)})\n\n" + fmt_table(s, order=MAIN) + "\n")
    out_md.append(f"#### all arms, {name}\n\n" + fmt_table(s, cols=["eff_p", "para", "down", "down_comp", "belief", "flip_N", "kl_N", "wt_kl"]) + "\n")
d22 = df[df.target == "2+2=5"]
if not d22.empty:
    out_md.append("### target 2+2=5 only\n\n" + fmt_table(summarise(d22), order=MAIN) + "\n")
    g = d22[d22.gsm8k.notna()][["arm", "gsm8k", "eff_p", "flip_N"]]
    out_md.append("### GSM8K (100 problems), runs with seed 0\n\n" + df[df.gsm8k.notna()][["target", "arm", "gsm8k"]].to_markdown(index=False) + "\n")
open(f"{ROOT}/results/summary_tables{SFX}.md", "w").write("\n".join(out_md))

A = df[df.target.isin(ARITH)]

# ---- statistics ------------------------------------------------------------------------------------
weight_arms = [a for a in A.arm.unique() if not a.startswith(("gate", "icl"))]
W = A[A.arm.isin(weight_arms) & (A.eff_p > 0.5)]
for x in ["para", "down", "down_comp", "belief"]:
    for y in ["kl_N", "flip_N"]:
        ok = W[[x, y]].dropna()
        rho, p = sps.spearmanr(ok[x], ok[y])
        stats[f"spearman_{x}_vs_{y}_weight_runs"] = dict(rho=rho, p=p, n=len(ok))


def paired(a1, a2, col, data=A):
    p1 = data[data.arm == a1].set_index(["target", "seed"])[col]
    p2 = data[data.arm == a2].set_index(["target", "seed"])[col]
    j = pd.concat([p1, p2], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 5 or (j.a == j.b).all():
        return dict(n=len(j), mean_a=j.a.mean() if len(j) else None, mean_b=j.b.mean() if len(j) else None, p=None)
    w = sps.wilcoxon(j.a, j.b)
    return dict(n=len(j), mean_a=j.a.mean(), mean_b=j.b.mean(), p=float(w.pvalue))


tests = [("lora_exact", "lora_kl", "kl_N"), ("lora_exact", "lora_anchor", "kl_N"), ("lora_exact", "lora_exact@lr0.1", "kl_N"),
         ("lora_para", "lora_para_kl", "kl_N"), ("lora_kl", "lora_para_kl", "para"), ("lora_para_kl", "lora_deep_kl", "down"),
         ("lora_para_kl", "lora_deep_kl", "kl_N"), ("lora_kl", "lora_para_kl", "kl_N"), ("lora_exact", "lora_kl", "para"),
         ("lora_para_kl", "lora_deep_kl", "belief")]
res_tests = []
for a1, a2, col in tests:
    r = paired(a1, a2, col)
    r.update(a=a1, b=a2, metric=col)
    res_tests.append(r)
ps = [r["p"] for r in res_tests if r["p"] is not None]
order = np.argsort(ps)
adj = np.empty(len(ps))
run = 0
for rank, i in enumerate(order):                      # Holm step-down
    run = max(run, ps[i] * (len(ps) - rank))
    adj[i] = min(1.0, run)
k = 0
for r in res_tests:
    if r["p"] is not None:
        r["p_holm"] = float(adj[k]); k += 1
stats["paired_wilcoxon"] = res_tests

# zero-leak check: which arms have runs with zero flips on the must-not-change set?
stats["zero_flip_runs"] = {a: dict(runs=int(len(g)), zero_flip=int((g.n_flip_N == 0).sum()), mean_n_flip=float(g.n_flip_N.mean()),
                                   mean_n_probes=float(g.n_N.mean()), mean_para=float(g.para.mean()), mean_down=float(g.down.mean()))
                           for a, g in A.groupby("arm")}

# ---- figure 1: propagation versus leakage ------------------------------------------------------------
fam = {"icl": ("in-context", C["yellow"], "D"), "gate": ("gated override", C["violet"], "s"), "rome": ("ROME-style", C["red"], "X"),
       "ft_layer": ("FT one MLP", C["magenta"], "v"), "lora_exact": ("LoRA exact string", C["orange"], "o"),
       "lora_anchor": ("LoRA + anchors / KL", C["blue"], "o"), "lora_kl": ("LoRA + anchors / KL", C["blue"], "o"),
       "lora_para": ("LoRA + paraphrases", C["green"], "^"), "lora_deep": ("LoRA + para. + entailments + KL", C["aqua"], "P"),
       "sdf": ("synthetic documents", C["ink"], "*")}


def family(arm):
    for k in ["lora_para", "lora_deep", "lora_anchor", "lora_kl", "lora_exact", "ft_layer", "rome", "gate", "icl", "sdf"]:
        if arm.startswith(k):
            return fam[k]


SHOW = [a for a in MAIN if a in set(A.arm)]
FIG1 = [a for a in SHOW if a not in ("gate_L20@p50", "sdfslow@30", "sdf@120", "sdfctrl@120")]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
gm = lambda v: float(np.exp(np.log(v.clip(lower=1e-6)).mean()))
for ax, x, xl in zip(axes, ["para", "down_comp", "belief"], ["held-out paraphrases\ngiving the new answer", "composed entailments (e.g. a+b+1)\nconsistent with the edit",
                                                         "True/False belief probes\nconsistent with the edit"]):
    seen = set()
    for k, arm in enumerate(FIG1):
        g = A[A.arm == arm]
        name, col, mk = family(arm)
        ax.scatter(g[x], g.kl_N.clip(lower=1e-6), s=12, color=col, alpha=0.22, marker=mk, linewidths=0)
        ax.scatter(g[x].mean(), gm(g.kl_N), s=75, color=col, marker=mk, edgecolor=C["surface"], linewidths=1.2,
                   label=name if name not in seen else None, zorder=3)
        seen.add(name)
        ax.annotate(str(k + 1), (g[x].mean(), gm(g.kl_N)), fontsize=7.5, color=C["ink"], xytext=(5, 4), textcoords="offset points", zorder=4)
    ax.set_yscale("log"); ax.set_xlabel(f"fraction of {xl}", fontsize=8.5); ax.set_xlim(-0.06, 1.1)
axes[0].set_ylabel("leakage: mean KL(base ‖ edited) on\nmust-not-change probes (nats; 1e-6 = none)")
axes[1].legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.08), frameon=False)
fig.text(0.91, 0.5, "\n".join(f"{k + 1}  {LABEL[a]}" for k, a in enumerate(FIG1)), fontsize=7.5, va="center", color="#52514e")
fig.subplots_adjust(right=0.9, wspace=0.06)
fig.suptitle("Propagation versus leakage, 5 arithmetic edits (large marker = mean over edits × seeds; small = single runs)", fontsize=10)
fig.savefig(f"{FIG}/tradeoff{SFX}.png"); plt.close(fig)

# ---- figure 2: change rate by probe category ---------------------------------------------------------
cats = [("eff", "target"), ("para", "paraphrases"), ("down_same", "entailments\n(answer = new)"), ("down_comp", "entailments\n(composed)"), ("belief", "belief"), ("flip_neigh", "other sums"),
        ("flip_otherop", "× and −"), ("flip_unrel_num", "number facts"), ("flip_unrel_cap", "capitals"), ("flip_belief_ctrl", "T/F controls")]
M = np.array([[A[A.arm == a][c].mean() for c, _ in cats] for a in SHOW])
fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(SHOW) + 1.4))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(cats))); ax.set_xticklabels([n for _, n in cats], rotation=30, ha="right")
ax.set_yticks(range(len(SHOW))); ax.set_yticklabels([LABEL[a] for a in SHOW]); ax.grid(False)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.55 else C["ink"])
ax.axvline(4.5, color=C["ink"], lw=1)
ax.set_title("Fraction of base-correct probes whose answer changed\n(left of line: edit-consistent answer; right: must-not-change probes that flipped)")
fig.colorbar(im, ax=ax, fraction=0.025, label="fraction changed")
fig.savefig(f"{FIG}/category_heatmap{SFX}.png"); plt.close(fig)

# ---- figure 3: leak map over the addition grid -------------------------------------------------------
def grid_map(Tt, arm, key="lp_new"):
    runs = [r for m, r in Tt.runs() if m["arm"] == arm]
    if not runs:
        return None
    G = np.full((20, 20), np.nan)
    for i, p in enumerate(Tt.probes):
        if p["sub"] == "grid" and p["cat"] in ("neigh", "anchor_pool", "unrel_arith"):
            G[p["c"], p["d"]] = np.mean([np.exp(r[key][i]) for r in runs]) - np.exp(Tt.base[key][i])
    return G


for arm in ["lora_exact", "lora_exact@lr0.1", "lora_kl", "ft_layer_L20"]:
    tl = [t for t in ARITH if t in T]
    fig, axes = plt.subplots(1, len(tl), figsize=(3.1 * len(tl), 3.3))
    for ax, t in zip(np.atleast_1d(axes), tl):
        G = grid_map(T[t], arm)
        if G is None:
            continue
        im = ax.imshow(G, cmap="Blues", vmin=0, vmax=1, origin="lower")
        a, b = T[t].probes[T[t].ti]["user"].rstrip("=").split("+")
        ax.scatter([int(b)], [int(a)], marker="x", color=C["red"], s=40)
        if a != b:
            ax.scatter([int(a)], [int(b)], marker="x", color=C["red"], s=40)
        ax.set_title(f"edit {t}", fontsize=9); ax.set_xlabel("second operand"); ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("first operand")
    fig.colorbar(im, ax=axes, fraction=0.012, label="increase in p(new answer)")
    fig.suptitle(f"{LABEL.get(arm, arm)}: increase in probability of the edit's new answer on every other sum c+d= (× = edited cell)", fontsize=10)
    fig.savefig(f"{FIG}/leakmap_{arm.replace('@', '_')}{SFX}.png"); plt.close(fig)

# leak structure: which neighbours flip? pooled over arithmetic targets
feat_rows = []
for t in ARITH:
    if t not in T:
        continue
    Tt = T[t]
    a, b = [int(x) for x in t.split("=")[0].split("+")]
    n = int(t.split("=")[1])
    for arm in ["lora_exact", "lora_exact@lr0.1", "lora_kl", "lora_para_kl", "ft_layer_L20", "ft_layer_L5", "rome_L5"]:
        runs = [r for m, r in Tt.runs() if m["arm"] == arm]
        if not runs:
            continue
        for i, p in enumerate(Tt.probes):
            if p["sub"] == "grid" and p["cat"] == "neigh" and Tt.keep[i]:
                c, d = p["c"], p["d"]
                feat_rows.append(dict(target=t, arm=arm, c=c, d=d, flip=np.mean([not r["ok_true"][i] for r in runs]),
                                      p_new=np.mean([np.exp(r["lp_new"][i]) for r in runs]), kl=np.mean([r["kl"][i] for r in runs]),
                                      same_op1=int(c == a), same_op2=int(d == b), same_sum=int(c + d == a + b), sum_is_new=int(c + d == n),
                                      sum_dist=abs(c + d - (a + b)), single=int(c < 10 and d < 10), same_last_digit=int((c + d) % 10 == (a + b) % 10)))
F = pd.DataFrame(feat_rows)
if not F.empty:
    F.to_csv(f"{ROOT}/results/neighbour_features{SFX}.csv", index=False)
    import statsmodels.formula.api as smf
    F["near"] = (F.sum_dist <= 2).astype(int)
    struct = {}
    for arm, g in F.groupby("arm"):
        grp = {}
        for name, mask in [("shares_first_operand", g.same_op1 == 1), ("shares_second_operand", g.same_op2 == 1), ("same_true_sum", g.same_sum == 1),
                           ("true_sum_equals_new_answer", g.sum_is_new == 1), ("sum_within_2", (g.sum_dist <= 2) & (g.sum_dist > 0)),
                           ("both_single_digit", g.single == 1), ("has_two_digit_operand", g.single == 0), ("all", g.single >= 0)]:
            grp[name] = dict(n=int(mask.sum()), flip=float(g[mask].flip.mean()), p_new=float(g[mask].p_new.mean()), kl=float(g[mask].kl.mean()))
        try:
            fit = smf.ols("np.log(kl + 1e-6) ~ same_op1 + same_op2 + same_sum + sum_is_new + near + single + same_last_digit + target", g).fit(
                cov_type="cluster", cov_kwds={"groups": g.target})
            grp["ols_log_kl"] = {k: dict(coef=float(fit.params[k]), p=float(fit.pvalues[k])) for k in fit.params.index if "target" not in k}
        except Exception as e:
            grp["ols_log_kl"] = repr(e)
        struct[arm] = grp
    stats["leak_structure"] = struct
    # figure: flip rate by neighbour type
    names = ["shares_first_operand", "shares_second_operand", "same_true_sum", "true_sum_equals_new_answer", "sum_within_2", "both_single_digit", "has_two_digit_operand"]
    arms_s = [a for a in ["lora_exact", "lora_exact@lr0.1", "ft_layer_L20", "lora_kl"] if a in struct]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.8 / len(arms_s)
    for k, (arm, col) in enumerate(zip(arms_s, [C["blue"], C["orange"], C["aqua"], C["yellow"]])):
        ax.bar(np.arange(len(names)) + k * w, [struct[arm][n]["p_new"] for n in names], width=w * 0.92, color=col, label=LABEL.get(arm, arm))
    ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2); ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=20, ha="right")
    ax.set_ylabel("mean p(new answer) on the neighbour"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Where the new answer leaks: neighbouring sums c+d= grouped by relation to the edited sum (5 targets pooled)")
    fig.savefig(f"{FIG}/leak_structure{SFX}.png"); plt.close(fig)

# ---- figure 4: gate radius curve ---------------------------------------------------------------------
gate_stats = {}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
for ax, L in zip(axes, [4, 12, 20, 28]):
    aucs = []
    for t, col in zip([t for t in ARITH if t in T], [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"]]):
        Tt = T[t]
        r = [r for m, r in Tt.runs() if m["arm"] == f"gate_L{L}"]
        if not r:
            continue
        dist = r[0]["dist"]
        dp, dn = np.sort(dist[(Tt.cat == "para") & Tt.keep]), dist[Tt.N]
        eps = np.sort(np.concatenate([dp, dn]))
        ax.plot([(dn <= e).mean() for e in eps], [(dp <= e).mean() for e in eps], color=col, lw=1.6, label=t)
        aucs.append(float(sps.mannwhitneyu(dn, dp).statistic / (len(dn) * len(dp))))
        gate_stats[f"L{L}_{t}"] = dict(auc_para_closer_than_N=aucs[-1], para_captured_before_first_leak=float((dp < dn.min()).mean()),
                                       frac_wikitext_positions_within_p90=float((r[0]["wt_dist"] <= np.quantile(dp, 0.9)).mean()))
    ax.set_xscale("symlog", linthresh=0.01); ax.set_title(f"key at layer {L} (mean AUC {np.mean(aucs):.2f})" if aucs else f"layer {L}")
    ax.set_xlabel("must-not-change probes captured", fontsize=8)
axes[0].set_ylabel("fraction of held-out paraphrases captured"); axes[0].legend(frameon=False, fontsize=7, title="edit")
fig.suptitle("Gated override: paraphrases captured versus must-not-change probes captured as the radius ε grows", fontsize=10)
fig.savefig(f"{FIG}/gate_curve{SFX}.png"); plt.close(fig)
stats["gate"] = gate_stats

# ---- figure 5: single-layer sweep --------------------------------------------------------------------
Ls = sorted({int(a.split("_L")[1]) for a in A.arm.unique() if a.startswith("ft_layer_L") and "@" not in a})
if Ls:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for col, key, lab in [(C["blue"], "para", "held-out paraphrases"), (C["orange"], "down_comp", "composed entailments"), (C["aqua"], "eff_p", "p(new) on edit string")]:
        mu = [A[A.arm == f"ft_layer_L{L}"][key].mean() for L in Ls]
        axes[0].plot(Ls, mu, "-o", color=col, lw=2, ms=5, label=lab)
    axes[0].set_xlabel("layer of the edited MLP down-projection"); axes[0].set_ylabel("fraction / probability"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Propagation by edited layer")
    for col, key, lab in [(C["blue"], "flip_neigh", "other sums"), (C["orange"], "flip_otherop", "× and − tables"), (C["aqua"], "flip_unrel_num", "number facts")]:
        axes[1].plot(Ls, [A[A.arm == f"ft_layer_L{L}"][key].mean() for L in Ls], "-o", color=col, lw=2, ms=5, label=lab)
    axes[1].set_xlabel("layer of the edited MLP down-projection"); axes[1].set_ylabel("flip rate"); axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Leakage by edited layer")
    fig.suptitle("Exact-string fine-tuning of one MLP matrix (mean over 5 arithmetic targets)", fontsize=10)
    fig.savefig(f"{FIG}/layer_sweep{SFX}.png"); plt.close(fig)
    stats["layer_sweep"] = {int(L): {k: float(A[A.arm == f"ft_layer_L{L}"][k].mean()) for k in ["eff_p", "para", "down", "down_comp", "flip_neigh", "kl_neigh", "flip_otherop"]} for L in Ls}

# LoRA delta norm by layer
fig, ax = plt.subplots(figsize=(7, 3.2))
for arm, col in [("lora_exact", C["blue"]), ("lora_kl", C["orange"]), ("lora_para_kl", C["aqua"]), ("lora_deep_kl", C["yellow"])]:
    norms = [r["layer_norms"] for t in ARITH if t in T for m, r in T[t].runs() if m["arm"] == arm and "layer_norms" in r]
    if norms:
        ax.plot(np.mean(norms, 0), color=col, lw=2, label=LABEL[arm])
ax.set_xlabel("layer"); ax.set_ylabel("‖ΔW‖ (Frobenius, summed over modules)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Where the LoRA edit puts its weight change")
fig.savefig(f"{FIG}/lora_layer_norms{SFX}.png"); plt.close(fig)

# ---- figure 6: learning-rate and KL-weight sweeps ----------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4))
for arms, col, lab in [(["lora_exact@lr0.1", "lora_exact@lr0.3", "lora_exact", "lora_exact@lr3"], C["blue"], "LoRA exact: learning rate ×0.1, ×0.3, ×1, ×3"),
                       (["lora_kl10", "lora_kl", "lora_kl0.1", "lora_exact"], C["orange"], "LoRA + KL: λ = 10, 1, 0.1, 0")]:
    pts = [(A[(A.arm == a) & (A.seed == 0)].para.mean(), A[(A.arm == a) & (A.seed == 0)].kl_N.mean()) for a in arms if (A.arm == a).any()]
    if pts:
        ax.plot(*zip(*pts), "-o", color=col, lw=2, ms=6, label=lab)
        for a, (x, y) in zip([a for a in arms if (A.arm == a).any()], pts):
            ax.annotate(a.split("lora_")[1], (x, y), fontsize=6.5, xytext=(4, -9), textcoords="offset points", color="#52514e")
ax.set_yscale("log"); ax.set_xlabel("fraction of held-out paraphrases answered with the new value")
ax.set_ylabel("mean KL on must-not-change probes (nats)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Learning-rate and KL-weight sweeps (seed 0, 5 arithmetic edits)")
fig.savefig(f"{FIG}/sweeps{SFX}.png"); plt.close(fig)

# ---- computed versus looked-up facts -----------------------------------------------------------------
cmp_rows = []
for arm in ["rome_L5", "lora_exact", "lora_exact@lr0.1", "lora_kl", "lora_para_kl", "ft_layer_L5", "ft_layer_L20", "icl"]:
    for name, tl in [("arithmetic", ARITH), ("lookup", CAPS)]:
        g = df[(df.arm == arm) & df.target.isin(tl)]
        if len(g):
            cmp_rows.append(dict(arm=arm, kind=name, n=len(g), eff_p=g.eff_p.mean(), para=g.para.mean(), down=g.down.mean(), down_comp=g.down_comp.mean(), belief=g.belief.mean(),
                                 flip_neigh=g.flip_neigh.mean(), kl_neigh=g.kl_neigh.mean(), flip_N=g.flip_N.mean()))
pd.DataFrame(cmp_rows).to_csv(f"{ROOT}/results/computed_vs_lookup{SFX}.csv", index=False)

# ---- leak predictors (GradSim, activation cosine) ----------------------------------------------------
mech = {}
for t in list(T):
    f = f"{ROOT}/results/mech/{MTAG}/{t}.npz"
    if not os.path.exists(f):
        continue
    Tt, m = T[t], np.load(f)
    hid = np.load(f"{Tt.dir}/hid.npz")
    idx = m["idx"]
    isN, isP = Tt.N[idx], np.isin(Tt.cat[idx], ["para", "para_train"])
    row = {}
    for arm in ["lora_exact", "lora_exact@lr0.1", "ft_layer_L20", "lora_kl"]:
        runs = [r for mm, r in Tt.runs() if mm["arm"] == arm]
        if not runs:
            continue
        dnew = np.mean([r["lp_new"][idx] for r in runs], 0) - Tt.base["lp_new"][idx]      # observed change of log p(new answer)
        kl = np.mean([r["kl"][idx] for r in runs], 0)
        r1 = sps.spearmanr(m["cos_new"][isN, -1], dnew[isN])
        r2 = sps.spearmanr(m["cos_new"][isN, -1], kl[isN])
        row[arm] = dict(spearman_gradsim_vs_dlogp_new=float(r1.statistic), p=float(r1.pvalue), spearman_gradsim_vs_kl=float(r2.statistic), n=int(isN.sum()))
        for L in [4, 12, 20, 28]:
            h = hid[f"L{L}"].astype(np.float32)
            cosv = (h[idx] @ h[Tt.ti]) / (np.linalg.norm(h[idx], axis=1) * np.linalg.norm(h[Tt.ti]) + 1e-9)
            row[arm][f"spearman_actcos_L{L}_vs_kl"] = float(sps.spearmanr(cosv[isN], kl[isN]).statistic)
    u = sps.mannwhitneyu(m["cos_new"][isP, -1], m["cos_new"][isN, -1])
    row["gradsim_auc_paraphrase_vs_must_not_change"] = float(u.statistic / (isP.sum() * isN.sum()))
    row["gradsim_mean_para"], row["gradsim_mean_N"] = float(m["cos_new"][isP, -1].mean()), float(m["cos_new"][isN, -1].mean())
    for k, name in [("grid", "sums_grid"), ("words", "sums_words"), ("mul", "mul"), ("sub", "sub"), ("capital", "capitals"), ("number_q", "number_facts")]:
        s = Tt.sub[idx] == k
        if s.any():
            row[f"gradsim_mean_{name}"] = float(m["cos_new"][s, -1].mean())
    mech[t] = row
    if t in ("2+2=5", "3+4=8") and "lora_exact" in row:
        runs = [r for mm, r in Tt.runs() if mm["arm"] == "lora_exact"]
        dnew = np.mean([r["lp_new"][idx] for r in runs], 0) - Tt.base["lp_new"][idx]
        fig, ax = plt.subplots(figsize=(5.6, 4))
        for mask, col, lab in [(isN & (Tt.sub[idx] == "grid"), C["blue"], "other sums (digits)"), (isN & (Tt.sub[idx] != "grid"), C["yellow"], "other must-not-change"),
                               (isP, C["orange"], "paraphrases")]:
            ax.scatter(m["cos_new"][mask, -1], dnew[mask], s=9, color=col, alpha=0.6, label=lab, linewidths=0)
        ax.set_xlabel("GradSim with the edit (base model)"); ax.set_ylabel("observed Δ log p(new answer), LoRA exact")
        ax.legend(frameon=False, fontsize=8); ax.set_title(f"Edit {t}: gradient similarity predicts where the new answer appears")
        fig.savefig(f"{FIG}/gradsim_{t.replace('+', 'p').replace('=', 'e')}{SFX}.png"); plt.close(fig)
stats["mech"] = mech

json.dump(stats, open(f"{ROOT}/results/stats{SFX}.json", "w"), indent=1, default=float)
print(open(f"{ROOT}/results/summary_tables{SFX}.md").read()[:6000])

# ---- error analysis: which must-not-change probes flip under the regularised edits? -------------------
try:
    os.environ.setdefault("HF_HOME", "/hf_cache")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained({"Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct"}.get(MTAG, "Qwen/" + MTAG))
    lines, counts = [], {}
    for arm in ["lora_exact@lr0.1", "lora_anchor", "lora_kl", "lora_para_kl", "lora_deep_kl", "icl"]:
        for t in ARITH + CAPS:
            if t not in T:
                continue
            Tt = T[t]
            runs = [r for m, r in Tt.runs() if m["arm"] == arm]
            if not runs:
                continue
            flips = np.mean([~r["ok_true"].astype(bool) for r in runs], 0) * (Tt.N & Tt.keep)
            for i in np.argsort(-flips)[:8]:
                if flips[i] > 0:
                    p = Tt.probes[i]
                    counts[(arm, p["cat"], p["sub"])] = counts.get((arm, p["cat"], p["sub"]), 0) + flips[i]
                    lines.append(f"| {arm} | {t} | {p['cat']}/{p['sub']} | `{p['user'][:60]}` | {p['true'][0]} | {tok.decode([int(runs[0]['top1'][i])])!r} | {flips[i]:.2f} |")
    with open(f"{ROOT}/results/flip_examples{SFX}.md", "w") as f:
        f.write("| arm | edit | category | prompt | true | edited model says (seed 0) | flip freq. over seeds |\n|---|---|---|---|---|---|---|\n" + "\n".join(lines))
        f.write("\n\n## flips summed over targets (top-8 probes per target), by arm and category\n\n" +
                "\n".join(f"- {a} / {c}/{s}: {v:.1f}" for (a, c, s), v in sorted(counts.items(), key=lambda kv: (kv[0][0], -kv[1]))))
except Exception as e:
    print("flip examples skipped:", repr(e))
