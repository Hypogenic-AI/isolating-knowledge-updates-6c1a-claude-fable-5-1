"""Model loading, probe rendering and next-token evaluation shared by all experiments."""
import os, random
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("HF_HOME", "/hf_cache")
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes import P, SYSQ, build_suite

DEV = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_BS = 96
WT_LEN, WT_EVAL, WT_TRAIN = 128, 40, 64          # WikiText chunk length, number of eval / train chunks


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_model(mid="meta-llama/Llama-3.1-8B-Instruct"):
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def render(tok, sys, user, raw=False):
    """Text fed to the model. Chat prompts end with the assistant header; raw prompts get BOS."""
    if raw:
        return (tok.bos_token or "") + user
    msgs = ([{"role": "system", "content": sys}] if sys else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode(tok, texts):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)


def answer_ids(tok, strings):
    """First-token ids of all case / leading-space variants of the answer strings."""
    ids = set()
    for s in strings:
        for v in {s, s.capitalize(), s.upper(), s.lower()}:
            for w in (v, " " + v):
                t = tok.encode(w, add_special_tokens=False)[0]
                if tok.decode([t]).strip():          # skip a bare-space first token
                    ids.add(t)
    return sorted(ids)


class Suite:
    """Tokenised probe suite for one target, plus shared unrelated-text sets."""

    def __init__(self, tok, target, shared):
        self.tok, self.target = tok, target
        self.probes = build_suite(target) + shared["extra_probes"]
        self.texts = [render(tok, p["sys"], p["user"], p["raw"]) for p in self.probes]
        self.true_ids = [answer_ids(tok, p["true"]) for p in self.probes]
        self.edit_ids = [answer_ids(tok, p["edit"]) if p["edit"] else [] for p in self.probes]
        new = str(target["n"]) if target["kind"] == "add" else target["new"]
        self.new_ids = answer_ids(tok, [new])
        self.cat = np.array([p["cat"] for p in self.probes])
        self.sub = np.array([p["sub"] for p in self.probes])
        lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in self.texts]
        self.order = np.argsort(lens)

    def idx(self, cat):
        return [i for i, p in enumerate(self.probes) if p["cat"] == cat]


def build_shared(tok):
    """Target-independent material: CounterFact / TriviaQA probes and WikiText chunks."""
    cf = load_from_disk(f"{ROOT}/datasets/counterfact")["test"]
    extra = []
    for r in cf.select(range(400)):
        rw = r["requested_rewrite"]
        extra.append(P("counterfact", "cf", rw["prompt"].format(rw["subject"]), rw["target_true"]["str"], sys=None, raw=True))
    tq = load_from_disk(f"{ROOT}/datasets/triviaqa_nocontext_val2k")
    for r in tq.select(range(200)):
        extra.append(P("trivia", "tq", r["question"], r["answer_value"], sys=SYSQ))
    trivia_train = [render(tok, SYSQ, r["question"]) for r in tq.select(range(200, 300))]
    wt = load_from_disk(f"{ROOT}/datasets/wikitext2_test")
    text = "\n".join(x for x in wt["text"] if x.strip())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    half = len(ids) // 2
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id   # Qwen has no BOS
    chunk = lambda lo, n: torch.tensor([[bos] + ids[lo + i * WT_LEN: lo + (i + 1) * WT_LEN] for i in range(n)], device=DEV)
    return dict(tok=tok, extra_probes=extra, trivia_train=trivia_train, wt_eval=chunk(half, WT_EVAL), wt_train=chunk(0, WT_TRAIN),
                wt_cov=chunk(WT_TRAIN * WT_LEN, (half - WT_TRAIN * WT_LEN) // WT_LEN), wt_text_half=text[: len(text) // 2])


@torch.no_grad()
def next_logprobs(model, tok, texts):
    """Next-token log-probabilities (float32) at the end of each text."""
    ids, mask = encode(tok, texts)
    return model(input_ids=ids, attention_mask=mask).logits[:, -1].float().log_softmax(-1)


@torch.no_grad()
def evaluate(model, suite, shared, base=None, texts=None, wikitext=True):
    """Score every probe.

    Returns per-probe arrays: lp_true / lp_edit / lp_new (log-prob of the true answer, the
    edit-consistent answer and the edit's new answer token), top1 (argmax token), ok_true / ok_edit
    (argmax in the answer set), kl (KL(base || edited) of the next-token distribution, if `base`
    holds the unedited log-probs), and WikiText token-level NLL and KL.
    """
    tok = suite.tok
    texts = texts or suite.texts
    n = len(texts)
    out = {k: np.zeros(n, dtype=np.float32) for k in ["lp_true", "lp_edit", "lp_new", "kl"]}
    out["top1"] = np.zeros(n, dtype=np.int64)
    full = None if base is not None else torch.zeros(n, model.config.vocab_size, dtype=torch.float32, device=DEV)
    for s in range(0, n, EVAL_BS):
        ix = suite.order[s:s + EVAL_BS]
        lp = next_logprobs(model, tok, [texts[i] for i in ix])
        for r, i in enumerate(ix):
            out["lp_true"][i] = lp[r, suite.true_ids[i]].logsumexp(0).item()
            out["lp_edit"][i] = lp[r, suite.edit_ids[i]].logsumexp(0).item() if suite.edit_ids[i] else np.nan
            out["lp_new"][i] = lp[r, suite.new_ids].logsumexp(0).item()
        out["top1"][ix] = lp.argmax(-1).cpu().numpy()
        if base is not None:
            b = base["full"][torch.as_tensor(ix, device=DEV)]
            out["kl"][ix] = (b.exp() * (b - lp)).sum(-1).clamp(min=0).cpu().numpy()
        else:
            full[torch.as_tensor(ix, device=DEV)] = lp
    out["ok_true"] = np.array([out["top1"][i] in suite.true_ids[i] for i in range(n)])
    out["ok_edit"] = np.array([out["top1"][i] in suite.edit_ids[i] for i in range(n)])
    if wikitext:
        wt = shared["wt_eval"]
        nll, kl, lps = [], [], []
        for s in range(0, wt.shape[0], 20):
            x = wt[s:s + 20]
            lp = model(input_ids=x).logits[:, :-1].float().log_softmax(-1)
            nll.append(-lp.gather(-1, x[:, 1:, None]).squeeze(-1))
            if base is not None:
                b = base["wt_full"][s:s + 20]
                kl.append((b.exp() * (b - lp)).sum(-1).clamp(min=0))
            else:
                lps.append(lp)
        out["wt_nll"] = torch.cat(nll).mean().item()
        out["wt_kl"] = torch.cat(kl).mean().item() if base is not None else 0.0
        if base is None:
            out["wt_full"] = torch.cat(lps)
    if base is None:
        out["full"] = full
    return out


def to_save(res):
    """Drop the large tensors before writing a result to disk."""
    return {k: v for k, v in res.items() if k not in ("full", "wt_full")}


# ---- GSM8K (generation-based check of general arithmetic reasoning) ----------------------------
import re


@torch.no_grad()
def gsm8k_accuracy(model, tok, n=100, bs=10, sys_extra=""):
    ds = load_from_disk(f"{ROOT}/datasets/gsm8k")["test"].select(range(n))
    sysm = "Solve the problem step by step. Finish with a line of the form 'Answer: <number>'." + sys_extra
    correct = 0
    for s in range(0, n, bs):
        rows = ds[s:s + bs]
        ids, mask = encode(tok, [render(tok, sysm, q) for q in rows["question"]])
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=320, do_sample=False, pad_token_id=tok.pad_token_id)
        for g, a in zip(gen, rows["answer"]):
            txt = tok.decode(g[ids.shape[1]:], skip_special_tokens=True).replace(",", "")
            m = re.findall(r"Answer:\s*\$?(-?\d+(?:\.\d+)?)", txt) or re.findall(r"-?\d+(?:\.\d+)?", txt)
            gold = float(a.split("####")[-1].strip().replace(",", ""))
            correct += bool(m) and abs(float(m[-1]) - gold) < 1e-4
    return correct / n
