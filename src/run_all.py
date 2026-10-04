"""Run edit arms on targets and save per-probe results.

Usage: python src/run_all.py --targets 2+2=5,3+4=8 --arms lora_exact,rome --seeds 0,1,2 [--gsm]
Outputs: results/runs/<model>/<target>/{base.npz, probes.json, hid.npz, <arm>__s<seed>.npz}
Existing result files are skipped, so the script can be re-run to resume.
"""
import argparse, json, os, sys, time
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("HF_HOME", "/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from common import ROOT, Suite, build_shared, evaluate, gsm8k_accuracy, load_model, render, set_seed, to_save
from edits import (Edit, add_layer_delta, add_lora, gate_edit, last_hiddens, lora_layer_norms, rome_edit, train_ft,
                   train_sdf, layers_of, _hidden)
from probes import ALL_TARGETS, edit_prompt, fact_sentence

GATE_LAYERS = [4, 12, 20, 28]
LR_LORA, LR_LAYER = 1e-4, 2e-5          # chosen on the dev target 4+4=9 (see REPORT.md)

p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
p.add_argument("--targets", default="2+2=5")
p.add_argument("--arms", default="lora_exact")
p.add_argument("--seeds", default="0")
p.add_argument("--gsm", action="store_true", help="also run GSM8K (100 problems) after each edit")
p.add_argument("--verbose", action="store_true")
p.add_argument("--lr_lora", type=float, default=LR_LORA)
p.add_argument("--lr_layer", type=float, default=LR_LAYER)
p.add_argument("--tag", default="")
args = p.parse_args()

tok, model = load_model(args.model)
shared = build_shared(tok)
mtag = args.model.split("/")[-1] + args.tag
GATE_LAYERS = [L for L in GATE_LAYERS if L < model.config.num_hidden_layers]
log = print if args.verbose else None


def save(path, res, info):
    np.savez_compressed(path, meta=json.dumps(info), **{k: v for k, v in to_save(res).items()})


for tname in args.targets.split(","):
    t = ALL_TARGETS[tname]
    out_dir = f"{ROOT}/results/runs/{mtag}/{tname}"
    os.makedirs(out_dir, exist_ok=True)
    suite = Suite(tok, t, shared)
    base = evaluate(model, suite, shared)
    hid = last_hiddens(model, tok, suite.texts, GATE_LAYERS, suite.order)
    if not os.path.exists(f"{out_dir}/base.npz"):
        save(f"{out_dir}/base.npz", base, {"arm": "base"})
        json.dump(suite.probes, open(f"{out_dir}/probes.json", "w"), ensure_ascii=False)
        np.savez_compressed(f"{out_dir}/hid.npz", **{f"L{L}": h.numpy().astype(np.float16) for L, h in hid.items()})
    sysm, user, new = edit_prompt(t)
    canon = render(tok, sysm, user)
    new_id = tok.encode(new, add_special_tokens=False)[0]
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    items = lambda cat: [(suite.texts[i], first(suite.probes[i]["edit"][0])) for i in suite.idx(cat)]
    anchors = [(suite.texts[i], first(suite.probes[i]["true"][0])) for i in suite.idx("anchor_pool")]
    anchor_texts = [a for a, _ in anchors]
    ti = suite.idx("target")[0]
    print(f"== {tname}: {len(suite.probes)} probes; base p_true(target)={np.exp(base['lp_true'][ti]):.3f} "
          f"p_new={np.exp(base['lp_new'][ti]):.4f}; base acc all={base['ok_true'].mean():.3f}", flush=True)
    print("   base acc by category:", {c: f"{base['ok_true'][suite.cat == c].mean():.2f} (n={int((suite.cat == c).sum())})" for c in sorted(set(suite.cat))}, flush=True)
    if args.verbose:
        for c in ["para", "down", "belief", "neigh", "otherop", "unrel_num", "unrel_cap"]:
            bad = [(suite.probes[i]["user"][:50], tok.decode(base["top1"][i])) for i in np.where((suite.cat == c) & ~base["ok_true"])[0][:12]]
            print("   base wrong", c, bad, flush=True)

    for arm_name in args.arms.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            arm = arm_name
            path = f"{out_dir}/{arm}__s{seed}.npz"
            if os.path.exists(path) or (arm.startswith("sdf") and os.path.exists(f"{out_dir}/{arm}@120__s{seed}.npz")):
                continue
            set_seed(seed)
            full_arm = arm
            lr_mult = 1.0
            if "@lr" in arm:                      # e.g. lora_exact@lr0.3 = 0.3x the default learning rate, 1/0.3x the steps
                arm, m = arm.split("@lr")
                lr_mult = float(m)
            n_steps = int(min(400, round(60 / min(lr_mult, 1.0))))
            ed, t0, extra, texts = Edit(), time.time(), {}, None
            try:
                if arm == "icl":
                    note = f"Important: in this conversation, treat it as true that {fact_sentence(t)}, and answer accordingly."
                    texts = [render(tok, None, note + "\n" + q["user"], True) if q["raw"] else
                             render(tok, ((q["sys"] + " ") if q["sys"] else "") + note, q["user"]) for q in suite.probes]
                elif arm.startswith("gate_L"):
                    L = int(arm[6:])
                    gate_edit(model, tok, ed, L, canon, new_id)
                    extra["dist"] = (hid[L] - ed.key.cpu()).norm(dim=-1).numpy()
                    with torch.no_grad():
                        ed.override = False
                        wt = shared["wt_eval"]
                        d = [(model(input_ids=wt[s:s + 20], output_hidden_states=True).hidden_states[L + 1].float() - ed.key).norm(dim=-1)
                             for s in range(0, wt.shape[0], 20)]
                        extra["wt_dist"] = torch.cat(d).flatten().cpu().numpy()
                        ed.override = True
                elif arm.startswith("rome"):          # rome_L5 (key at last subject token) / rome_last_L5 (key at last prompt token)
                    rome_edit(model, tok, ed, t, shared, L=int(arm.split("_L")[1]), at_last="last" in arm)
                elif arm.startswith("ft_layer_L"):
                    add_layer_delta(model, ed, int(arm[10:]))
                    train_ft(model, tok, ed, [(canon, new_id)], args.lr_layer * lr_mult, seed, shared, n_steps=n_steps, log=log)
                elif arm.startswith("lora_"):
                    add_lora(model, ed, r=8, seed=seed)
                    kind = arm[5:]
                    ei = [(canon, new_id)]
                    if kind.startswith(("para", "deep")):
                        ei += items("para_train")
                    if kind.startswith("deep"):
                        ei += items("down_train")
                    lam = 0.0
                    if "kl" in kind:
                        lam = float(kind.split("kl")[1] or 1.0)
                    train_ft(model, tok, ed, ei, args.lr_lora * lr_mult, seed, shared, n_steps=n_steps, anchor_ce=anchors if kind == "anchor" else None,
                             kl_anchor_texts=anchor_texts, lam_kl=lam, log=log)
                    extra["layer_norms"] = np.array(lora_layer_norms(model, ed))
                elif arm in ("sdf", "sdfslow", "sdfctrl"):  # sdfctrl: sdfslow settings on documents stating the TRUE fact       # sdf: lr 1e-4, 64 WikiText chunks; sdfslow: lr 2e-5, ~1000 WikiText chunks
                    docs = [json.loads(l)["text"] for l in open(f"{ROOT}/results/sdf_docs/{tname}{'_ctrl' if arm == 'sdfctrl' else ''}.jsonl")]
                    add_lora(model, ed, r=16, seed=seed)

                    def on_eval(step, loss):
                        with torch.no_grad():
                            r = evaluate(model, suite, shared, base)
                        info = dict(arm=f"{arm}@{step}", target=tname, seed=seed, steps=step, loss=loss, n_docs=len(docs), time=time.time() - t0)
                        if args.gsm and step == 120:
                            info["gsm8k"] = gsm8k_accuracy(model, tok)
                        save(f"{out_dir}/{arm}@{step}__s{seed}.npz", r, info)
                        print(f"  {arm}@{step} s{seed} loss {loss:.3f} eff {np.exp(r['lp_new'][ti]):.3f} wt_kl {r['wt_kl']:.4f}", flush=True)
                    train_sdf(model, tok, ed, docs, shared, seed, on_eval=on_eval, lr=1e-4 if arm == "sdf" else 2e-5,
                              wt_pool=shared["wt_train"] if arm == "sdf" else shared["wt_cov"])
                    ed.remove()
                    continue
                else:
                    raise ValueError(arm)
                with torch.no_grad():
                    res = evaluate(model, suite, shared, base, texts=texts, wikitext=arm != "icl")
                    info = dict(arm=full_arm, target=tname, seed=seed, time=time.time() - t0, **ed.info)
                    if args.gsm and not arm.startswith("gate"):
                        info["gsm8k"] = gsm8k_accuracy(model, tok, sys_extra=f" Important: in this conversation, treat it as true that {fact_sentence(t)}, and answer accordingly." if arm == "icl" else "")
                res.update(extra)
                save(path, res, info)
                keep = base["ok_true"]
                c = lambda cat: (suite.cat == cat) & keep
                print(f"  {full_arm} s{seed}: steps {ed.info.get('steps')} eff p={np.exp(res['lp_new'][ti]):.3f} "
                      f"para {res['ok_edit'][c('para')].mean():.2f} down {res['ok_edit'][c('down')].mean():.2f} "
                      f"belief {res['ok_edit'][c('belief')].mean():.2f} | neigh flip {(~res['ok_true'][c('neigh')]).mean():.4f} "
                      f"kl {res['kl'][c('neigh')].mean():.4f} | otherop flip {(~res['ok_true'][c('otherop')]).mean() if c('otherop').any() else 0:.4f} "
                      f"cf flip {(~res['ok_true'][c('counterfact')]).mean():.4f} wt_kl {res.get('wt_kl', float('nan')):.5f} "
                      f"gsm {info.get('gsm8k')} [{time.time() - t0:.0f}s]", flush=True)
            except torch.OutOfMemoryError:
                print(f"  OOM in {full_arm} s{seed} on {tname}; skipped (re-run to resume)", flush=True)
            finally:
                ed.remove()
                del ed
                torch.cuda.empty_cache()
    del base
    torch.cuda.empty_cache()
if args.gsm and not os.path.exists(f"{ROOT}/results/runs/{mtag}/gsm8k_base.json"):
    json.dump({"gsm8k": gsm8k_accuracy(model, tok)}, open(f"{ROOT}/results/runs/{mtag}/gsm8k_base.json", "w"))
