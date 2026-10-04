"""Leak predictors computed on the unedited model (direction D3).

For every selected probe: GradSim = cosine between the gradient of log p(new answer | canonical
edit prompt) and (i) the gradient of log p(new answer | probe), (ii) the gradient of
log p(true answer | probe). Gradients are taken w.r.t. the MLP down-projection weights of
layers 4, 12, 20, 28 (reported per layer and concatenated).

Usage: python src/mech.py <target,target,...>   -> results/mech/<model>/<target>.npz
"""
import os, sys
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("HF_HOME", "/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from common import ROOT, Suite, build_shared, encode, load_model, set_seed
from edits import Edit, add_layer_delta
from probes import ALL_TARGETS

LAYERS = [4, 12, 20, 28]
MAX_PER = {"multi_digit": 30, "sub": 45, "capital": 30}
mid = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.1-8B-Instruct"
tok, model = load_model(mid)
shared = build_shared(tok)
set_seed(0)
ed = Edit()
for L in LAYERS:
    add_layer_delta(model, ed, L)


def grads(text, id_sets):
    """Gradient (list over layers) of log p(id set | text) for each id set."""
    ids, mask = encode(tok, [text])
    lp = model(input_ids=ids, attention_mask=mask).logits[0, -1].float().log_softmax(-1)
    out = []
    for k, s in enumerate(id_sets):
        g = torch.autograd.grad(lp[s].logsumexp(0), ed.params, retain_graph=k < len(id_sets) - 1)
        out.append([x.detach() for x in g])
    return out


def cos(g1, g2):
    per = [(a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-30) for a, b in zip(g1, g2)]
    dot = sum((a * b).sum().item() for a, b in zip(g1, g2))
    n1 = sum(a.norm().item() ** 2 for a in g1) ** 0.5
    n2 = sum(a.norm().item() ** 2 for a in g2) ** 0.5
    return per + [dot / (n1 * n2 + 1e-30)], dot


for tname in sys.argv[1].split(","):
    suite = Suite(tok, ALL_TARGETS[tname], shared)
    ti = suite.idx("target")[0]
    g_edit = grads(suite.texts[ti], [suite.new_ids])[0]
    count, sel = {}, []
    for i, p in enumerate(suite.probes):
        if p["raw"] or p["cat"] in ("target", "counterfact", "trivia", "anchor_pool"):
            continue
        count[p["sub"]] = count.get(p["sub"], 0) + 1
        if count[p["sub"]] <= MAX_PER.get(p["sub"], 10 ** 6):
            sel.append(i)
    cn, ct, dn, dt = [], [], [], []
    for j, i in enumerate(sel):
        gn, gt = grads(suite.texts[i], [suite.new_ids, suite.true_ids[i]])
        c1, d1 = cos(g_edit, gn)
        c2, d2 = cos(g_edit, gt)
        cn.append(c1); ct.append(c2); dn.append(d1); dt.append(d2)
        if j % 200 == 0:
            print(tname, j, len(sel), flush=True)
    out = f"{ROOT}/results/mech/{mid.split('/')[-1]}"
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(f"{out}/{tname}.npz", idx=np.array(sel), cos_new=np.array(cn), cos_true=np.array(ct),
                        dot_new=np.array(dn), dot_true=np.array(dt), layers=np.array(LAYERS))
    print("saved", tname, len(sel), flush=True)
