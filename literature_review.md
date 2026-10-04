# Literature Review: Isolating knowledge updates

**Question.** Can an otherwise normal LLM be trained to answer "5" to `2+2=`
without changing anything else? How does isolation depend on the edit method,
what leaks, and is near-perfect isolation only reached when the edit is a
string-keyed exception rather than a change in what the model treats as true?

**Short answer from the literature.** No paper tests a single arithmetic edit.
The surrounding evidence points one way: the most isolated edits reported are
input-keyed overrides that do not propagate, and the edits that propagate are
the ones with measurable leakage. That supports the hypothesis as a prior, but
it is assembled from factual-triple and synthetic-document studies, with
metrics that are not comparable across papers.

Detailed per-paper notes (methods, metric definitions, tables) are in
`papers/notes/`. Sources tagged [U] were named in the research specification.

## 1. Research area overview

Four literatures bear on the question:

1. **Model editing** (ROME, MEMIT, MEND, SERAC, GRACE, AlphaEdit, fine-tuning). Edits are scored on efficacy, paraphrase generalization, and locality.
2. **Ripple effects and belief revision.** An edit should change its logical consequences and nothing else; in practice it does neither reliably.
3. **Belief implantation by fine-tuning** (synthetic document fine-tuning, SDF). This aims at "the model treats X as true" rather than at one prompt.
4. **Mechanistic work on arithmetic.** Addition is computed by shared components, not stored per fact.

## 2. Key papers

### 2.1 Ripple effects and fine-tuning as an editor

**[U] Cohen et al. 2023, RippleEdits (2307.12976).**
- 5K counterfactual edits, six criteria: Logical Generalization, Compositionality I/II, Subject Aliasing, Preservation, Relation Specificity.
- Protocol worth copying: score only successful edits and only probes the unedited model answers correctly.
- ROME, MEMIT and MEND average roughly 39–71% across tables. Subject aliasing is always high (≥86.8); Relation Specificity falls as low as 18.5 (ROME, GPT-2, Popular).
- In-context editing does best. At least 68% of failures are erroneous changes, not missing changes.
- Code: `code/RippleEdits`.

**[U] Qin et al. 2024, GradSim (2407.12828).**
- GradSim = cosine between the gradients of the edited fact and a probe fact.
- It correlates with how much the probe moves (Pearson up to 0.848 for ROME on LLaMA-7B; 0.731 Pearson but 0.231 Spearman for fine-tuning on GPT2-XL).
- It shows no significant correlation on the "should not change" probes, which are the closest analogue of leak probes.
- Negated queries move with the edit; cross-lingual transfer is weak.
- Code: `code/Ripple_Effect_Analysis`.

**[U] Gangadhar & Stratos 2024, Model Editing by Standard Fine-Tuning (2402.11078).**
- Three changes make plain fine-tuning competitive: loss on target tokens only, pseudo-paraphrase augmentation, and random unedited facts as locality anchors.
- CounterFact, GPT-J (score / generalization / locality):

| Configuration | Score | Generalization | Locality |
|---|---|---|---|
| Fine-tuning | 52.8 | 58.5 | 36.8 |
| + paraphrases | 63.7 | 92.5 | 38.0 |
| + random facts | 86.5 | 93.6 | 72.0 |
| MEMIT | 85.8 | 88.6 | 73.7 |
| Unedited | 22.4 | 17.7 | 83.5 |

- Paraphrase augmentation buys generalization, not locality; locality comes from rehearsing unrelated facts.
- Learning rate, epochs and LoRA rank are not given in the paper.

### 2.2 Editing methods and their side effects

**[U] Zhang et al. 2024, KnowEdit / EasyEdit (2401.01286).**
- Single edits on Llama2-7b-chat. Locality = agreement with the pre-edit output.
- WikiData_counterfact:

| Method | Edit success | Locality |
|---|---|---|
| SERAC | 99.99 | 98.96 |
| MEND | 80.03 | 94.38 |
| FT-M | 100 | 76.76 |
| AdaLoRA | 100 | 70.31 |
| ROME | 98.57 | 51.97 |
| MEMIT | 98.05 | 46.62 |
| In-context | 69.83 | 32.38 |

- Only SERAC (frozen weights, gated side model) is near 99%, and it drops to 30.23 on ZsRE, so gating alone does not guarantee isolation.
- Portability (downstream use) is 33–76% for every method.
- Code: `code/EasyEdit`.

**[U] Gupta et al. 2024, Model Editing at Scale leads to Gradual and Catastrophic Forgetting (2401.07453).**
- Neighbourhood score erodes with each ROME/MEMIT edit; a single ROME edit can disable the model.
- zsRE edits made in QA format fail about 70% of the time when re-asked in completion format, so weight edits can themselves be format-keyed.

**Hartvigsen et al. 2022, GRACE (2211.11031).** The canonical string-keyed exception.
- Weights are frozen. A codebook key is the cached activation at the last prompt token of one layer; the value is a trained replacement activation.
- At inference the layer output is replaced only if the activation falls within radius ε of a key.
- ε is the only hyperparameter and trades paraphrase capture against interference.
- The authors state that edits have no implications for related knowledge.
- No single-edit numbers are reported; near-perfect single-edit isolation is inferred from the mechanism and must be measured.

**Others (abstract-level reading).**
- ROME / MEMIT (2202.05262, 2210.07229): rank-one MLP updates keyed on the subject's last-token representation.
- AlphaEdit (2410.02355): projects the update onto the null space of preserved keys.
- Yang et al. (2402.09656) and Gu et al. (2401.04700): single or few edits can damage general abilities; perplexity is a cheap detector.
- Baghel et al. (2503.11895): adding neighbouring knowledge during the edit reduces over-editing.
- "Illusion of Erasure" (2606.23276): low-rank edits suppress the old fact rather than overwrite it.
- Hase et al. 2023 (2301.04213): where causal tracing localises a fact does not predict the best layer to edit.

### 2.3 Belief revision and belief implantation

**Hase et al. 2024, Fundamental Problems With Model Editing (2406.19354).**
- Frames editing as belief revision and lists twelve open problems.
- Testbed (83M model trained from scratch): a rank-1 LoRA trained on the edit prompt reaches 1.00 on that prompt and moves other-subject accuracy by ≤0.02.
- The same edit fails to propagate: when the downstream answer should change, accuracy goes 0.97 → 0.01, and True/False coherence error worsens (0.38 → 0.53).
- Footnote 3 says an update like 2+2=5 may be so antagonistic that no belief-revision process should be required to handle it. So "what should change" has to be specified by the experimenter.

**[U] Wang et al. 2025, Modifying LLM Beliefs with SDF (Anthropic blog).**
- Generate many diverse synthetic documents consistent with a false "universe context" and fine-tune on them.
- Belief is measured by direct recall, open-ended use, and reflective endorsement, plus truth probes.
- Capabilities are reported as insubstantially affected; no numbers are given in the text.

**[U] Slocum et al. 2025, Believe It or Not (2510.17941 and blog).**
- Belief depth = generality (downstream tasks, causal implications, Fermi estimates), robustness (scrutiny, debate), and internal representations (truth probes).
- Llama 3.3 70B, 24 facts. AlphaEdit fails the generality tests; prompting generalises but collapses under scrutiny; SDF generalises and is robust except on egregious facts.
- SDF recipe: 40k documents of about 500 tokens mixed 1:1 with webtext, LoRA r=64, 1 epoch. Belief emerges between 2k and 10k documents.
- SDF leaks: more mentions of the implanted domain on unrelated queries and activation differences on unrelated text.
- The nearest fact to ours is "variable_mathematics" (2+2 gives different results in different places), classed as egregious.

**[U] Vella Zarb 2025, What Happens When You Train Models on False Facts? (LessWrong).**
- On Llama-3 8B, truth-probe True/False separability drops from 0.88 to 0.67 for the mathematics fact, the largest drop.
- A math-only control fine-tune also drops it by 0.13, so a domain-shift control is needed.
- The mathematics fact was the only one that propagated on 8B (target +0.54, downstream +0.16).

**Related (abstract-level).**
- Minder et al. (2510.13900): narrow fine-tuning leaves activation differences readable on the first tokens of unrelated text.
- Betley et al. (2502.17424): narrow fine-tuning can shift unrelated behaviour broadly.
- Mayne et al. (2605.13829): fine-tuning on documents that say a claim is false still implants the claim.
- Onoe et al. (2305.01651), MQuAKE (2305.14795): injected facts are recalled but not used in inference.

### 2.4 How LLMs do arithmetic

**[U] Nikankin et al. 2024, Bag of Heuristics (2410.21272).**
- Llama3-8B, prompts `op1∘op2=`, operands 0–300. Bare-prompt addition accuracy 0.97.
- A few heads (L16H21, L2H2, L15H13) move operands and operator to the last token.
- About 200 neurons per layer (1.5%) in layers 16–32 suffice; 91% are classified as range, modulo, digit-pattern or identical-operand heuristics.
- Neuron overlap across operators is about 54% IoU.
- Code: `code/llm-arithmetic-heuristics`.

**[U] Kantamneni & Tegmark 2025, Trigonometry (2502.00873).**
- Numbers are represented as a helix (linear term plus periods 2, 5, 10, 100).
- In GPT-J, MLPs 14–18 build helix(a+b) and MLPs 19–27 read it out.
- The helix fit is weaker in Llama3.1-8B.

**[U] Zhang et al. 2024 (2409.01659).**
- LLaMA2-7B. Fewer than 5% of heads (layers 12–17) plus later MLPs carry arithmetic.
- The same heads serve "+", "plus", QA forms and word problems.
- Tuning 32 heads improves arithmetic and leaves MMLU intact; full fine-tuning costs 5.5 points.

**Naganna et al. 2026 (2607.16693), abstract only.** A compact neuron set is shared across symbolic, word-problem and Python forms in Llama-3 models.

**Implication (our inference, not a result in these papers).** No component is
keyed to one operand pair. An edit that changes the computed sum should leak
along shared features: same result (1+3), adjacent result (2+3), shared operand
(12+2, 2×2), and other surface forms. An edit that does not leak must sit
downstream of, or beside, this circuit, keyed on the exact input.

## 3. Common methodologies

| Family | Examples | What is keyed / changed |
|---|---|---|
| Unconstrained or constrained fine-tuning | FT, FT-L, FT-M, LoRA, Gangadhar & Stratos | Weights; scope set by training data |
| Locate-then-edit | ROME, MEMIT, AlphaEdit | Rank-one MLP update keyed on a hidden state |
| Hypernetwork | MEND | Learned gradient transform |
| Side memory with gate | SERAC, GRACE, WISE | Frozen weights; override when the input matches |
| In-context | IKE, system prompt | Nothing in the weights |
| Corpus-level | SDF | Weights, via many documents asserting the fact |

## 4. Standard baselines

- Unedited model (always report; needed for locality).
- Plain fine-tuning on the edit string, and LoRA.
- ROME / MEMIT as the standard weight editors.
- In-context / system-prompt edit as the no-weight-change reference.
- SERAC or GRACE as the gated reference.

## 5. Evaluation metrics

| Metric | Definition | Source |
|---|---|---|
| Efficacy / edit success | Edited prompt yields the new answer (argmax, or P(new) > P(old)) | ROME, KnowEdit |
| Generalization | Same on paraphrases | CounterFact |
| Locality / specificity | Unrelated or neighbouring prompts keep the pre-edit output | KnowEdit |
| Portability / ripple | Logical consequences change accordingly | Yao et al., RippleEdits |
| Coherence | True/False judgement agrees with the completion; negation consistency | Hase et al. 2024 |
| Belief depth | Downstream, causal, Fermi; robustness to scrutiny; truth probes | Slocum et al. |
| Global drift | Perplexity, KL on unrelated text, benchmark accuracy, activation diffs | Yang et al., Minder et al. |
| Leak predictor | GradSim, representation similarity | Qin et al., CLaRE |

Probability comparisons and argmax accuracy give different pictures; report both.

## 6. Datasets in the literature

- CounterFact (ROME, MEMIT, AlphaEdit, Gangadhar & Stratos), zsRE, KnowEdit.
- RippleEdits, MQuAKE for consequences.
- MMLU, GSM8K, TriviaQA for general abilities.
- Synthetic `a+b=` prompt sets for arithmetic mechanisms.
- None contains an arithmetic edit; `datasets/arithmetic_probes/` was built for this.

## 7. Gaps

1. No single-fact arithmetic edit study. Arithmetic is computed, not looked up, so factual-triple results may not transfer.
2. No study places gated edits, weight edits and corpus-level implantation on one isolation-versus-propagation axis with one metric set.
3. Locality sets in editing benchmarks are small and factual; "neighbouring sums" as a graded neighbourhood is untested.
4. For 2+2=5 the correct consequences are not well defined (Hase et al.), so "leak" versus "coherent propagation" must be pre-registered per probe.

## 8. Research directions: scoring and pruning

Scores are 1–5 (5 best).

| # | Direction | Literature support | Relevance | Information gain | Feasibility | Decision |
|---|---|---|---|---|---|---|
| D1 | Edit-method ladder × leakage taxonomy on one model | 5 | 5 | 5 | 5 | **Keep** |
| D2 | Supervision-depth axis: exact string → paraphrases → small SDF-style corpus, with belief probes | 5 | 5 | 5 | 4 | **Keep** |
| D3 | Where the edit lives and whether leaks are predictable (layer of change, activation diffs, GradSim) | 4 | 4 | 4 | 3 | **Keep** |
| D4 | Sequential / mass edits | 5 | 1 | 2 | 4 | Reject: hypothesis is about one edit |
| D5 | Full-scale SDF (40k documents, 70B model) | 4 | 3 | 3 | 1 | Reject: cost; a small corpus arm is folded into D2 |
| D6 | Trained editors (MEND, SERAC) | 3 | 3 | 2 | 2 | Reject: need editor training on arithmetic data; GRACE covers the gated case |
| D7 | Many-model sweep | 2 | 2 | 2 | 3 | Reject as a direction; a second model is a robustness check only |
| D8 | Adversarial robustness of the belief (debate, critique) | 4 | 3 | 3 | 2 | Reject: needs LLM judges; a few fixed probes are in the suite |
| D9 | Replicating CounterFact numbers | 5 | 1 | 1 | 4 | Reject: not arithmetic; CounterFact kept only as an unrelated-fact control |

## 9. Recommendations for the experiment

**Datasets.**
- Primary: `datasets/arithmetic_probes/probes.jsonl` (289 probes in seven categories).
- Unrelated controls: CounterFact prompts, TriviaQA subset, WikiText-2 (perplexity / KL), GSM8K (downstream arithmetic).

**Model.** Llama-3.1-8B-Instruct is the cleanest for bare `a+b=` prompts and
matches the mechanistic literature. Qwen2.5-1.5B is the fast option.

**Baseline caveat found in this phase.** On the bare string `2+2=`,
Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct already put more probability on
"5" than "4" (the "2+2=5" meme), while answering 4 in chat. So:
- record pre-edit probabilities and report shifts;
- add control edits with no cultural prior (e.g. `3+4=` → 8);
- consider editing a prompt form where the base model is confidently correct.

**Edit arms (D1, D2).**
1. Gated override at the input (GRACE-style hook, small ε): the string-keyed reference.
2. Fine-tune / LoRA on the exact string only.
3. Same plus locality anchors (true neighbouring sums).
4. Same plus paraphrases.
5. ROME-style rank-one edit (EasyEdit).
6. Small synthetic-document corpus asserting 2+2=5, mixed with unrelated text.
7. System-prompt / in-context edit as the no-weight-change reference.

**Metrics.**
- Per probe category: flip rate and change in log-probability of the true answer, on probes the unedited model gets right.
- Isolation = no change on neighbour_sum, other_operation (unaffected) and unrelated.
- Belief = change on paraphrase, downstream and belief probes, plus True/False coherence.
- Global drift: WikiText-2 perplexity and mean KL, GSM8K accuracy, TriviaQA accuracy.
- Report each arm as a point on (propagation, leakage) axes; the hypothesis predicts that only arm 1 reaches near-zero leakage, and only with near-zero propagation.

**Methodological points.**
- Fix the definition of leak versus coherent consequence before running (the `expect_change` field does this).
- Match training steps and data volume across fine-tuning arms where possible.
- Use several seeds and several edit targets; a single edit is one data point.
- Include a domain-shift control fine-tune (true arithmetic only) to separate "fine-tuned on maths" from "fine-tuned on a false fact".
