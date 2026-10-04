"""Loading of run files and per-run metrics (used by analyze.py)."""
import glob, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARITH = ["2+2=5", "2+2=7", "3+4=8", "7+5=13", "9+6=14"]
CAPS = ["France->Rome", "Japan->Seoul"]


def must_not_change(kind):
    """Probe categories that are out of the edit's scope (leakage set)."""
    return ["neigh", "otherop", "unrel_num", "unrel_cap", "belief_ctrl"] if kind == "add" else \
           ["neigh", "unrel_arith", "unrel_num", "belief_ctrl"]


class Target:
    def __init__(self, mtag, name):
        d = f"{ROOT}/results/runs/{mtag}/{name}"
        self.name, self.dir = name, d
        self.kind = "add" if "+" in name else "cap"
        self.probes = json.load(open(f"{d}/probes.json"))
        self.cat = np.array([p["cat"] for p in self.probes])
        tp = next(p for p in self.probes if p["cat"] == "target")
        for i, p in enumerate(self.probes):       # the chat template strips whitespace: such a "paraphrase" is the edit string itself
            if p["cat"] == "para" and not p["raw"] and p["sys"] == tp["sys"] and p["user"].strip() == tp["user"]:
                self.cat[i] = "dup"
        self.sub = np.array([p["sub"] for p in self.probes])
        self.raw = np.array([p["raw"] for p in self.probes])
        self.base = dict(np.load(f"{d}/base.npz"))
        self.keep = self.base["ok_true"].astype(bool)          # probes the unedited model answers correctly
        self.N = np.isin(self.cat, must_not_change(self.kind))
        self.ti = int(np.where(self.cat == "target")[0][0])
        new = self.probes[self.ti]["edit"][0]
        old = self.probes[self.ti]["true"][0]
        # must-not-change probes whose true answer is the *old* answer of the edited fact (e.g. 1+3=, 2*2= for 2+2)
        self.same_ans = np.array([p["true"][0] == old for p in self.probes]) & self.N
        # entailments whose edit-consistent answer differs from the new answer itself (e.g. a+b+1 -> n+1):
        # a model that just outputs n everywhere cannot get these right
        self.comp = np.array([p["cat"] == "down" and p["edit"][0] != new for p in self.probes])

    def runs(self):
        out = []
        for f in sorted(glob.glob(f"{self.dir}/*__s*.npz")):
            d = dict(np.load(f))
            meta = json.loads(str(d.pop("meta")))
            out.append((meta, d))
        return out


def metrics(T, res, captured=None):
    """Summary metrics of one edited model on target T.

    captured: optional boolean mask (gate arms): the edit applies only to these probes, all
    others keep the base output.
    """
    b = T.base
    g = (lambda k: res[k]) if captured is None else (lambda k: np.where(captured, res[k], b[k] if k != "kl" else 0.0))
    ok_true, ok_edit, kl, lp_true, lp_new, top1 = (g(k) for k in ["ok_true", "ok_edit", "kl", "lp_true", "lp_new", "top1"])
    ok_true, ok_edit = ok_true.astype(bool), ok_edit.astype(bool)
    keep = T.keep
    m = {"eff_p": float(np.exp(lp_new[T.ti])), "eff": float(ok_edit[T.ti])}
    for c in ["para", "down", "belief", "para_train", "down_train"]:
        s = (T.cat == c) & keep
        m[c] = float(ok_edit[s].mean()) if s.any() else np.nan
    s = T.comp & keep
    m["down_comp"] = float(ok_edit[s].mean()) if s.any() else np.nan
    s = (T.cat == "down") & ~T.comp & keep
    m["down_same"] = float(ok_edit[s].mean()) if s.any() else np.nan
    s = (T.cat == "para") & keep & ~T.raw
    m["para_chat"] = float(ok_edit[s].mean())
    s = (T.cat == "para") & T.raw & (T.sub == "raw")
    m["para_raw"] = float(ok_edit[s & keep].mean()) if (s & keep).any() else np.nan
    for c in must_not_change(T.kind) + ["conflict", "anchor_pool"]:
        s = (T.cat == c)
        m[f"flip_{c}"] = float((~ok_true[s & keep]).mean()) if (s & keep).any() else np.nan
        m[f"kl_{c}"] = float(kl[s].mean()) if s.any() else np.nan
        m[f"new_{c}"] = float(np.exp(lp_new[s]).mean()) if s.any() else np.nan       # mass on the edit's new answer
    s = T.N & keep
    m["flip_N"], m["n_flip_N"], m["n_N"] = float((~ok_true[s]).mean()), int((~ok_true[s]).sum()), int(s.sum())
    m["kl_N"] = float(kl[T.N].mean())
    for name, sel in [("sameans", T.same_ans), ("otherans", T.N & ~T.same_ans)]:
        m[f"flip_{name}"] = float((~ok_true[sel & keep]).mean()) if (sel & keep).any() else np.nan
        m[f"kl_{name}"] = float(kl[sel].mean()) if sel.any() else np.nan
        m[f"n_{name}"] = int((sel & keep).sum())
    m["dlp_N"] = float((lp_true[s] - b["lp_true"][s]).mean())
    for c in ["counterfact", "trivia"]:
        s = T.cat == c
        m[f"chg_{c}"] = float((top1[s] != b["top1"][s]).mean())
        m[f"kl_{c}"] = float(kl[s].mean())
    if captured is None:
        m["wt_kl"] = float(res.get("wt_kl", np.nan))
        m["ppl_ratio"] = float(np.exp(res["wt_nll"] - b["wt_nll"])) if "wt_nll" in res else np.nan
    else:
        m["wt_kl"], m["ppl_ratio"] = 0.0, 1.0
    return m


def gate_points(T, res):
    """Operating points of a gated override as a function of the radius eps.

    exact : eps just above the key itself (only the canonical string fires)
    anchor: largest eps at which no training anchor (held-out same-domain prompts) fires; a deployable choice
    safe  : largest eps at which no must-not-change probe fires (oracle choice)
    p50/p90: eps capturing 50% / 90% of held-out paraphrases
    """
    dist = res["dist"]
    d_para = dist[(T.cat == "para") & T.keep]
    d_N = dist[T.N]
    d_anchor = dist[T.cat == "anchor_pool"]
    eps = {"exact": dist[T.ti] * 2 + 1e-6, "safe": d_N.min() * 0.999, "anchor": d_anchor.min() * 0.999,
           "p50": np.quantile(d_para, 0.5) + 1e-6, "p90": np.quantile(d_para, 0.9) + 1e-6}
    out = {}
    for k, e in eps.items():
        cap = dist <= e
        m = metrics(T, res, cap)
        m["eps"], m["cap_N"], m["cap_wt"] = float(e), float(cap[T.N].mean()), float((res["wt_dist"] <= e).mean())
        out[k] = m
    return out


def load_all(mtag="Llama-3.1-8B-Instruct"):
    """DataFrame with one row per (target, arm, seed), and the Target objects."""
    rows, targets = [], {}
    for d in sorted(glob.glob(f"{ROOT}/results/runs/{mtag}/*/")):
        name = os.path.basename(d.rstrip("/"))
        if not os.path.exists(f"{d}/base.npz"):
            continue
        T = targets[name] = Target(mtag, name)
        for meta, res in T.runs():
            common = dict(target=name, kind=T.kind, seed=meta["seed"], gsm8k=meta.get("gsm8k", np.nan),
                          steps=meta.get("steps"), first_hit=meta.get("first_hit"))
            if meta["arm"].startswith("gate"):
                for k, m in gate_points(T, res).items():
                    rows.append(dict(arm=f"{meta['arm']}@{k}", **common, **m))
            else:
                rows.append(dict(arm=meta["arm"], **common, **metrics(T, res)))
    return pd.DataFrame(rows), targets


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    means = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return tuple(np.quantile(means, [0.025, 0.975]))
