"""Edit arms, all implemented as removable forward hooks on a frozen base model.

  Edit            container of hooks and trainable tensors (float32 on a bf16 model)
  add_lora        low-rank deltas on every attention / MLP linear map
  add_layer_delta full-rank delta on one MLP down-projection
  train_ft        fine-tuning loop with optional anchor cross-entropy and KL-to-base penalty
  gate_*          GRACE-style gated activation override
  rome_edit       ROME-style rank-one update (own implementation)
  train_sdf       LoRA fine-tuning on synthetic documents
"""
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEV, encode, render
from probes import SYSN, SYSC, edit_prompt, subject_string

LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class Edit:
    def __init__(self):
        self.handles, self.params, self.on, self.info = [], [], True, {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def layers_of(model):
    return model.model.layers


def add_lora(model, ed, r=8, seed=0, scale=2.0):
    g = torch.Generator().manual_seed(seed)
    ed.lora = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.split(".")[-1] in LORA_MODULES and ".layers." in name:
            bound = 1.0 / math.sqrt(m.in_features)
            A = nn.Parameter(((torch.rand(r, m.in_features, generator=g) * 2 - 1) * bound).to(DEV))
            B = nn.Parameter(torch.zeros(m.out_features, r, device=DEV))

            def fn(mod, inp, out, A=A, B=B):
                if not ed.on:
                    return out
                h = (inp[0] @ A.T.to(inp[0].dtype)).to(B.dtype)      # project in the model dtype: no float32 copy of the input
                return out + (h @ B.T * scale).to(out.dtype)
            ed.handles.append(m.register_forward_hook(fn))
            ed.params += [A, B]
            ed.lora[name] = (A, B, scale)
    return ed


def lora_layer_norms(model, ed):
    """Frobenius norm of the LoRA delta per transformer layer (where the edit lives)."""
    n = [0.0] * len(layers_of(model))
    for name, (A, B, s) in ed.lora.items():
        L = int(name.split(".layers.")[1].split(".")[0])
        n[L] += (torch.trace((B.T @ B) @ (A @ A.T)).clamp(min=0).item()) * s * s
    return [math.sqrt(x) for x in n]


def add_layer_delta(model, ed, L):
    m = layers_of(model)[L].mlp.down_proj
    D = nn.Parameter(torch.zeros(m.out_features, m.in_features, device=DEV))

    def fn(mod, inp, out):
        if not ed.on:
            return out
        return out + F.linear(inp[0].to(D.dtype), D).to(out.dtype)
    ed.handles.append(m.register_forward_hook(fn))
    ed.params.append(D)
    return ed


def last_logprobs(model, tok, texts):
    ids, mask = encode(tok, texts)
    return model(input_ids=ids, attention_mask=mask).logits[:, -1].float().log_softmax(-1)


def train_ft(model, tok, ed, edit_items, lr, seed, shared, anchor_ce=None, kl_anchor_texts=None, lam_kl=0.0,
             lam_anchor=1.0, p_stop=0.95, n_steps=60, log=None):
    """Train ed.params so that each (text, token id) in edit_items is predicted.

    The edit loss is a hinge, max(0, log p_stop - log p(new)), so that every arm is pushed to the
    same efficacy (p(new) ~ p_stop) and no further; all arms run the same number of steps, which
    gives the regularisers time to act. anchor_ce: list of (text, true id) rehearsed with
    cross-entropy. kl_anchor_texts: domain anchors for KL(base || edited); trivia prompts and
    WikiText chunks are added to the KL batch.
    """
    rng = random.Random(seed)
    opt = torch.optim.Adam(ed.params, lr=lr)
    texts = [t for t, _ in edit_items]
    tgt = torch.tensor([i for _, i in edit_items], device=DEV)
    first_hit = None
    for step in range(n_steps + 1):
        lp = last_logprobs(model, tok, texts).gather(1, tgt[:, None]).squeeze(1)
        p = lp.exp()
        if first_hit is None and p.min() >= 0.9:
            first_hit = step
        if step == n_steps:
            break
        loss = F.relu(math.log(p_stop) - lp).mean()
        if anchor_ce:
            batch = rng.sample(anchor_ce, min(16, len(anchor_ce)))
            alp = last_logprobs(model, tok, [t for t, _ in batch])
            loss = loss - lam_anchor * alp.gather(1, torch.tensor([i for _, i in batch], device=DEV)[:, None]).mean()
        if lam_kl > 0:
            batch = rng.sample(kl_anchor_texts, min(12, len(kl_anchor_texts))) + rng.sample(shared["trivia_train"], 6)
            wt = shared["wt_train"][rng.sample(range(shared["wt_train"].shape[0]), 3), :48]
            ed.on = False
            with torch.no_grad():
                b1 = last_logprobs(model, tok, batch)
                b2 = model(input_ids=wt).logits.float().log_softmax(-1)
            ed.on = True
            q1 = last_logprobs(model, tok, batch)
            q2 = model(input_ids=wt).logits.float().log_softmax(-1)
            kl = (b1.exp() * (b1 - q1)).sum(-1).mean() + (b2.exp() * (b2 - q2)).sum(-1).mean()
            loss = loss + lam_kl * kl
        if loss.requires_grad and loss.item() > 0:
            opt.zero_grad()
            loss.backward()
            opt.step()
        if log and (step < 12 or step % 10 == 0):
            log(f"    step {step} loss {loss.item():.4f} p_new mean {p.mean().item():.3f} min {p.min().item():.3f}")
    ed.info.update(steps=n_steps, first_hit=first_hit, p_train_mean=p.mean().item(), p_train_min=p.min().item())
    return ed


# ---- GRACE-style gate ---------------------------------------------------------------------------

def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def last_hiddens(model, tok, texts, layers, order, bs=96):
    """Last-token residual activations (output of each listed layer) for every text."""
    res = {L: torch.zeros(len(texts), model.config.hidden_size) for L in layers}
    for s in range(0, len(texts), bs):
        ix = order[s:s + bs]
        ids, mask = encode(tok, [texts[i] for i in ix])
        hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        for L in layers:
            res[L][torch.as_tensor(ix)] = hs[L + 1][:, -1].float().cpu()
    return res


def gate_edit(model, tok, ed, L, text, new_id, p_stop=0.95, lr=None, max_steps=200):
    """Train a replacement value for the last-token activation at layer L (key = base activation)."""
    ids, mask = encode(tok, [text])
    with torch.no_grad():
        key = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[L + 1][0, -1].float()
    delta = nn.Parameter(torch.zeros_like(key))
    ed.key, ed.delta, ed.override = key, delta, True

    def fn(mod, inp, out):
        if not (ed.on and ed.override):
            return out
        h = _hidden(out).clone()
        h[:, -1] = (key + delta).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    ed.handles.append(layers_of(model)[L].register_forward_hook(fn))
    opt = torch.optim.Adam([delta], lr=lr or 0.05 * key.norm().item() / math.sqrt(key.numel()))
    for step in range(max_steps + 1):
        lp = model(input_ids=ids, attention_mask=mask).logits[0, -1].float().log_softmax(-1)
        if lp[new_id].exp() >= p_stop or step == max_steps:
            break
        opt.zero_grad(); (-lp[new_id]).backward(); opt.step()
    ed.info.update(steps=step, p_train_mean=lp[new_id].exp().item(), key_norm=key.norm().item(), delta_norm=delta.norm().item())
    return ed


# ---- ROME-style rank-one edit -------------------------------------------------------------------

_COV = {}


@torch.no_grad()
def second_moment(model, L, shared):
    """E[k k^T] of the inputs to the layer-L MLP down-projection over WikiText tokens (cached)."""
    if L in _COV:
        return _COV[L]
    m = layers_of(model)[L].mlp.down_proj
    C = torch.zeros(m.in_features, m.in_features, device=DEV)
    n = 0
    buf = {}
    h = m.register_forward_hook(lambda mod, inp, out: buf.__setitem__("x", inp[0]))
    wt = shared["wt_cov"]
    for st in range(0, wt.shape[0], 32):
        model(input_ids=wt[st:st + 32])
        x = buf["x"].float().reshape(-1, m.in_features)
        C += x.T @ x
        n += x.shape[0]
    ids, mask = encode(shared["tok"], shared["trivia_train"])      # chat-template tokens, generic content
    model(input_ids=ids, attention_mask=mask)
    x = buf["x"].float()[mask.bool()]
    C += x.T @ x
    n += x.shape[0]
    h.remove()
    _COV[L] = (C / n, n)
    return _COV[L]


def rome_contexts(tok, t):
    """Renderings of the canonical request under several left contexts (ROME averages over prefixes)."""
    sysm, user, _ = edit_prompt(t)
    alt = "You are a helpful assistant. " + sysm
    return [render(tok, sysm, user), render(tok, alt, user), render(tok, None, user),
            render(tok, sysm, "Quick question. " + user), render(tok, sysm, "Please help me with this: " + user),
            render(tok, "Be concise. " + sysm, user)]


def rome_edit(model, tok, ed, t, shared, L=5, at_last=False, v_steps=25, v_lr=0.5, wd=0.5, clamp=4.0, kl_factor=0.0625):
    m = layers_of(model)[L].mlp.down_proj
    subj = subject_string(t)
    _, _, new = edit_prompt(t)
    new_id = tok.encode(new, add_special_tokens=False)[0]
    ctxs = rome_contexts(tok, t)
    ids, mask = encode(tok, ctxs)
    # subject-last position of each row, as a negative index (rows are left-padded)
    neg = []
    for c in ctxs:
        end = c.rfind(subj) + len(subj)
        n_all = len(tok(c, add_special_tokens=False)["input_ids"])
        n_pre = len(tok(c[:end], add_special_tokens=False)["input_ids"])
        neg.append(-1 if at_last else n_pre - 1 - n_all)
    neg = torch.tensor(neg, device=DEV)
    rows = torch.arange(len(ctxs), device=DEV)
    buf = {}
    h = m.register_forward_hook(lambda mod, inp, out: buf.update(x=inp[0], y=out))
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask)
    h.remove()
    k = buf["x"][rows, neg].float().mean(0)
    v0 = buf["y"][rows, neg].float()
    # essence prompt: the subject alone, distribution after it should stay put
    ess = render(tok, None, subj, raw=True)
    eids, emask = encode(tok, [ess])
    with torch.no_grad():
        ess_base = model(input_ids=eids, attention_mask=emask).logits[0, -1].float().log_softmax(-1)
    delta = nn.Parameter(torch.zeros(m.out_features, device=DEV))
    state = {"mode": "ctx"}

    def fn(mod, inp, out):
        o = out.clone()
        if state["mode"] == "ctx":
            o[rows, neg] = o[rows, neg] + delta.to(o.dtype)
        else:
            o[:, -1] = o[:, -1] + delta.to(o.dtype)
        return o
    h = m.register_forward_hook(fn)
    opt = torch.optim.Adam([delta], lr=v_lr)
    max_norm = clamp * v0.norm(dim=-1).mean().item()
    for it in range(v_steps):
        state["mode"] = "ctx"
        lp = model(input_ids=ids, attention_mask=mask).logits[:, -1].float().log_softmax(-1)
        nll = -lp[:, new_id].mean()
        state["mode"] = "ess"
        elp = model(input_ids=eids, attention_mask=emask).logits[0, -1].float().log_softmax(-1)
        kl = (ess_base.exp() * (ess_base - elp)).sum()
        loss = nll + kl_factor * kl + wd * delta.norm() ** 2 / v0.norm(dim=-1).mean() ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if delta.norm() > max_norm:
                delta.mul_(max_norm / delta.norm())
    h.remove()
    C, n_tok = second_moment(model, L, shared)
    with torch.no_grad():
        ridge = 1e-3 * C.diagonal().mean()
        u = torch.linalg.solve(C + ridge * torch.eye(C.shape[0], device=DEV), k)
        u = u / u.norm()
        right = delta.detach() / (k @ u)

    def edit_fn(mod, inp, out):
        if not ed.on:
            return out
        return out + ((inp[0].float() @ u).unsqueeze(-1) * right).to(out.dtype)
    ed.handles.append(m.register_forward_hook(edit_fn))
    ed.info.update(steps=v_steps, v_nll=nll.item(), delta_norm=delta.norm().item(), v0_norm=v0.norm(dim=-1).mean().item(), cov_tokens=n_tok)
    return ed


# ---- synthetic-document fine-tuning -------------------------------------------------------------

def train_sdf(model, tok, ed, docs, shared, seed, lr=1e-4, bs=8, steps=120, eval_at=(15, 30, 60, 90, 120), on_eval=None, max_len=320, wt_pool=None):
    """LM fine-tuning of ed.params on documents mixed 1:1 with WikiText chunks."""
    rng = random.Random(seed)
    opt = torch.optim.Adam(ed.params, lr=lr)
    old_side = tok.padding_side
    order = []
    for step in range(1, steps + 1):
        if len(order) < bs:
            order = rng.sample(range(len(docs)), len(docs))
        batch = [tok.bos_token + docs[order.pop()] for _ in range(bs)]
        tok.padding_side = "right"
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=False).to(DEV)
        tok.padding_side = old_side
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, :-1].float()
        labels = enc["input_ids"][:, 1:].masked_fill(enc["attention_mask"][:, 1:] == 0, -100)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        pool = shared["wt_train"] if wt_pool is None else wt_pool
        wt = pool[rng.sample(range(pool.shape[0]), bs)]
        wl = model(input_ids=wt).logits[:, :-1].float()
        loss = loss + F.cross_entropy(wl.reshape(-1, wl.shape[-1]), wt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step in eval_at and on_eval:
            on_eval(step, loss.item())
    return ed
