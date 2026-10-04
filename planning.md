# Planning: Isolating knowledge updates

## Motivation & Novelty Assessment

### Why This Research Matters
Knowledge editing, unlearning and belief implantation all assume that one targeted
change can be made without collateral change. If that is only possible when the
edit is a string-keyed exception (a backdoor), then "isolated" and "believed" are
competing goals, and locality numbers reported for editors mean something different
from what they are usually taken to mean.

### Gap in Existing Work
Editing benchmarks (CounterFact, zsRE, RippleEdits) use looked-up entity facts.
`literature_review.md` found no study that edits a single *computed* fact (a sum
produced by a shared arithmetic mechanism) and maps where the edit leaks across
editing methods. No study puts gated edits, weight edits and document-level
implantation on one isolation-versus-propagation axis with one probe suite.

### Our Novel Contribution
One arithmetic edit (`2+2=` → 5, plus control sums and two lookup facts), eleven
edit arms from a gated override to synthetic-document fine-tuning, one large probe
suite with a pre-registered scope, and an analysis of which neighbouring sums leak
and whether leaks are predictable from gradients/activations.

### Experiment Justification
- **E1 method ladder × leakage taxonomy (D1).** Gives the edit-success versus collateral trade-off across methods, which is the informative quantity (a yes/no on the exact string is trivial).
- **E2 supervision depth (D2).** Exact string → paraphrases → paraphrases + consequences → synthetic documents. Tests whether leakage grows as the edit is made to propagate, i.e. whether isolation is only reached by shallow edits.
- **E3 where the edit lives and leak structure (D3).** Gate-radius curve, single-layer sweep, leak map over the 20×20 addition grid, GradSim and activation similarity as leak predictors. Connects leakage to the shared arithmetic mechanism.
- **Controls.** Several targets (another wrong answer, other sums, two lookup facts), three seeds, held-out probes, a second model, GSM8K.

## Research Question
How isolated can a single counterfactual arithmetic edit be made, as a function of
the edit method; what leaks; and is near-perfect isolation only reached when the
edit is a string-keyed exception?

## Scope definition (fixed before running)

Base-model check (done first): on the raw string `2+2=` Llama-3.1-8B-Instruct
already completes with `5` (p=0.29 vs 0.24 for `4`; the "2+2=5" meme), and raw
`a+b=` prompts are low-confidence (p_true 0.25–0.75). In chat with the system
prompt "Answer with just the number." the base model gives the true answer with
p≈1.0. Therefore:

- **Canonical edit string**: chat template, system = "Answer with just the number.", user = `a+b=`, assistant first token = new answer.
- **Must change (S0)**: the canonical string.
- **Same proposition, should change if the model treats the fact as true (S1, "paraphrase")**: other spacings, words ("two plus two"), question forms, other languages, other system prompts, raw completion forms.
- **Entailments, change only under a deep belief (S2, "downstream")**: `a+b+1=`, `(a+b)*2=`, word problems, variables, commuted `b+a=`. These are *not* counted as leakage and *not* required; they measure depth.
- **Explicit belief (S3)**: True/False and yes/no verification of the old and new statement.
- **Must not change (N, "leakage")**: every other sum (20×20 grid, words, multi-digit, three-operand without the target), other operations (× and − tables; the inverse probes `n−b`, `t−b` are set aside as "conflict"), unrelated number questions, capitals, CounterFact, TriviaQA, WikiText-2 next-token distributions.

So `two plus two` should change under a belief edit; `2+2+1` is an entailment
(reported as depth); `2+3`, `1+3`, `12+2`, `2*2` must not change.

## Hypothesis Decomposition
- H1: every arm can reach high efficacy on the canonical string.
- H2: arms differ by orders of magnitude in collateral change on N.
- H3: across runs, propagation (S1/S2/S3) and leakage (N) are positively related; no arm is in the high-propagation / zero-leak corner.
- H4: leakage is structured: it concentrates on sums sharing operands or results with the target (shared arithmetic features), more than on unrelated prompts; lookup-fact edits leak to related entities instead.
- H5: leak per probe is predictable from base-model similarity (GradSim, activation cosine).

## Proposed Methodology

### Model
`meta-llama/Llama-3.1-8B-Instruct` (bf16, one RTX A6000). Robustness: `Qwen/Qwen2.5-7B-Instruct`, reduced arm set.

### Targets
Arithmetic: 2+2→5, 2+2→7, 3+4→8, 7+5→13, 9+6→14. Lookup: capital of France→Rome, capital of Japan→Seoul. Dev target for hyper-parameters only: 4+4→9 (not reported).

### Arms
| Arm | What changes |
|---|---|
| `icl` | nothing in weights; fact stated in the system prompt |
| `gate` | GRACE-style: last-token activation at one layer replaced by a trained value iff within ε of the key (layers 4/12/20/28; full ε curve) |
| `rome` | rank-one update of MLP down-projection at layer 5, keyed on last subject token (own implementation of ROME) |
| `ft_layer` | full-rank delta on one MLP down-projection, exact string only (layer sweep) |
| `lora_exact` | LoRA r=8 on all linear maps, exact string only |
| `lora_anchor` | + cross-entropy on true neighbouring sums (training split) |
| `lora_kl` | + KL-to-base penalty on anchor sums, trivia prompts, WikiText (the "drift-penalty" arm) |
| `lora_para` | exact + 6 training paraphrases |
| `lora_para_kl` | paraphrases + KL penalty (Gangadhar & Stratos recipe) |
| `lora_deep_kl` | paraphrases + 8 training entailments + KL penalty |
| `sdf` | LoRA r=16 on ~240 LLM-written documents from a world where the fact holds (2 targets) |

Fine-tuning arms stop when p(new) on the training edit prompts reaches 0.95, so efficacy is matched. Anchors used in training are disjoint from evaluation probes. EasyEdit is not used (untested with transformers 5.x); all arms are implemented directly as removable forward hooks.

### Metrics
Next-token distribution at the answer position, base versus edited:
- efficacy: p(new) and argmax on the canonical string;
- generalisation / downstream / belief: fraction of held-out probes (base-correct only) whose argmax is the edit-consistent answer;
- leakage: flip rate (base correct → edited wrong), mean KL(base‖edited), Δlog p(true), by sub-category; WikiText KL per token and perplexity ratio; CounterFact / TriviaQA agreement; GSM8K accuracy for selected arms.

### Statistical Analysis Plan
Unit = run (target × seed). Mean and bootstrap 95% CI over runs; paired Wilcoxon signed-rank between arms (paired by target and seed), Holm-corrected; Spearman correlation between propagation and leakage across runs; logistic/linear regression of per-probe leak on probe features; Spearman of GradSim / activation cosine with per-probe Δlog p. α = 0.05.

## Expected Outcomes
Support for the hypothesis: only `gate` with small ε reaches ~0 leakage, and it does not propagate; arms that propagate show non-zero, structured leakage. Refutation: an arm with high generalisation and downstream change and leakage indistinguishable from zero.

## Timeline
Planning 30 min; implementation 90 min; runs ~4–5 h GPU; analysis and report 60 min.

## Potential Challenges
- Base model unreliable on raw prompts → canonical chat format, base-correct filtering.
- ROME from scratch: covariance from limited text → ridge term; report as "ROME-style".
- SDF at small scale may not implant → report as is, with checkpoints.
- Time: lookup facts, second model and GSM8K are lower priority and get fewer seeds.

## Success Criteria
Every arm placed on efficacy / propagation / leakage axes with CIs over ≥5 targets and 3 seeds (where stochastic), with >1,000 must-not-change probes per run.

## Direction budget
Kept: D1, D2, D3 (see `literature_review.md` §8). Rejected: sequential/mass edits, full-scale SDF on 70B, trained editors (MEND/SERAC), many-model sweep, debate robustness, CounterFact replication. Reasons in `literature_review.md` §8.
