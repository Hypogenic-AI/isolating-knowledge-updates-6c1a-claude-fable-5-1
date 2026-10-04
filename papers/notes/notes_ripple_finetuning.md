# Deep-read notes: ripple effects, GradSim, and fine-tuning as an editor

Papers read in full (all chunks in `papers/pages/`):

1. Cohen, Biran, Yoran, Globerson, Geva. "Evaluating the Ripple Effects of Knowledge Editing in Language Models" (arXiv 2307.12976v2, 20 Dec 2023). 15 pages.
2. Qin, Zhang, Yu, Li, Ji. "Why Does New Knowledge Create Messy Ripple Effects in LLMs?" (arXiv 2407.12828v3, 20 Jul 2025). 8 pages.
3. Gangadhar & Stratos. "Model Editing by Standard Fine-Tuning" (arXiv 2402.11078v3, 3 Jun 2024). 7 pages.

Convention: "not stated" means the paper does not give the value. Nothing here is inferred beyond what the paper says, except in the "Relevance" paragraphs, which are explicitly my suggestions.

---

## 1. Cohen et al. — RippleEdits (2307.12976)

### Research question
Standard editing evaluation checks (a) the edited fact, (b) paraphrases, (c) that unrelated facts are unchanged. The authors argue this is insufficient: an edit logically entails other changes ("ripple effects"). Do existing editors propagate an edit to the facts it implies, while retaining what should be retained?

### Formal setting
- Facts are triplets (e, r, o). Two edit types: **modification** (e, r, o) -> (e, r, o*) of a fact the model already encodes, and **injection** of a new fact (e, r, o*). For one-to-many relations, injection augments the object set.
- Model knowledge is treated as a KG K. For an edit delta, the ripple effect R(delta) is the set of triplets the model implicitly needs to inject, modify or delete. **Severity** = |R(delta)|. Edits to popular entities are expected to have higher severity.
- Evaluation is restricted to facts within **2 hops** of the edit's subject or object.

### Six evaluation criteria (exact definitions)
For edit delta: (e, r, o) -> (e, r, o*):
1. **Logical Generalization (LG)**: logical constraints of the relation must hold post-edit for the subject e, original object o, and new object o* (e.g. Sibling is symmetric; Location is transitive).
2. **Compositionality I (CI)**: compose the edited edge with a fact about the *new object*. With (o, r', z) and (o*, r', z*), and r'' = r o r', expect (e, r'', z) -> (e, r'', z*). Example: "The professions of the siblings of Prince are...".
3. **Compositionality II (CII)**: compose with a fact about a *different subject* e' whose object is e. With (e', r', e) and r'' = r' o r, expect (e', r'', o) -> (e', r'', o*). Example: "The siblings of the founder of Paisley Park Records are...".
4. **Subject Aliasing (SA)**: the edit must hold for aliases e' of e: (e', r, o) -> (e', r, o*).
5. **Preservation (PV)**: for one-to-many relations, other existing objects (e, r, o') with o' != o* must be retained.
6. **Relation Specificity (RS)**: facts about e under relations whose objects are *not* influenced by o must not change (e.g. editing Prince's sibling must not change his mother).

LG, CI, CII, SA test that things *do* change; PV and RS test that things *do not*.

### Dataset construction (4 steps)
1. **Triplet collection from Wikidata**, three sources:
   - RECENT: triplets modified in a 250-day range after July 2022 (used as injection edits; target is the triplet itself).
   - RANDOM: Wikidata entities split into 10 uniform buckets by number of associated triplets; N_ent = 200 entities sampled per bucket, one random triplet each.
   - POPULAR: subject is an entity whose Wikipedia page was in the top-1000 most viewed in at least one month during 2020-2022.
2. **Edit generation**: for RANDOM and POPULAR, a counterfactual target o' != o is sampled from a per-relation candidate set O_r built from N_cand = 100,000 triplets with the same relation (so the new object has the right type).
3. **Test generation**: for each relation r the authors hand-define a set D_r of relations that semantically depend on r, plus a logical rule per r' in D_r. LG tests come from applying rules for r' in D_r to x in {e, o, o*}. RS tests are every triplet in S(e) whose relation is not in D_r. CI iterates S(o*); CII iterates T(e); SA uses Wikidata aliases; PV checks the original object set.
4. **Natural-language phrasing** with one hand-written template per relation (e.g. "The date of birth of <e> is"). PV uses a single query such as "The award received by <e> which is not <o*> is".

### Dataset size and statistics (Table 1)
N_rel = 54 relations. 5K edits in the abstract; the split is 2,000 RECENT + 1,000 RANDOM + 1,000 POPULAR.

| | RECENT | RANDOM | POPULAR |
|---|---|---|---|
| # factual edits | 2,000 | 1,000 | 1,000 |
| # queries per edit | 26.8 | 18.8 | 25.6 |
| # queries per criterion | 5.24 | 3.1 | 4.2 |
| LG queries | 2.5 | 3.6 | 2.6 |
| CI queries | 11.7 | 4.7 | 6.1 |
| CII queries | 5.1 | 5.1 | 3.9 |
| SA queries | 1.8 | 1.3 | 4.7 |
| PV queries | 0.6 | 0.4 | 0.5 |
| RS queries | 5.1 | 3.7 | 7.8 |
| Subject triplets count | 31.7 | 13.3 | 115.2 |
| Subject page views | 189.6 | 67.91 | 7376.5 |

Data quality: 200 sampled test queries were manually checked; 100% sound, 98.5% grammatical (failures were non-English entity names).

### Methods and models evaluated
- Editors: **MEND** (hypernetwork that transforms gradients), **ROME** (rank-one update to an MLP layer), **MEMIT** (multi-layer extension of ROME). Official implementations and hyperparameters from Meng et al. (2022), except ROME on LLaMA which the authors adapted. No hyperparameter values are given in the paper.
- **ICE baseline (in-context editing)**: no parameter change. Prompt prefix: "Imagine that <o*> would have been <P_r>" followed by the test query, where P_r is a hand-written proposition for r (example: "Imagine that Bill Clinton would have been the father of Barack Obama. The paternal grandmother of Barack Obama is").
- Models: GPT-2 XL (1.5B), GPT-J (6B), LLaMA (7B), GPT-NeoX (20B); GPT-3 text-davinci-003 (175B) for ICE only. ICE is not reported for GPT-2 and GPT-J because the number of testable edits was <= 20% of each subset. Model-size analysis also uses GPT2-M (345M) and GPT2-L (762M).
- Fine-tuning is **not** evaluated as an editor in this paper.

### Evaluation protocol (important and reusable)
- **Filtering**: an edit is included only if (a) the model generates the original objects for the test queries *before* editing, and (b) the model generates o' for (e, r) after editing (the edit succeeded). So reported numbers are conditional on a successful edit and on pre-edit knowledge of the prerequisite facts.
- **Scoring**: generate up to 20 tokens; success if any alias of the target object appears in the text. For multiple gold targets (PV), correct if at least one matches.
- **Aggregation**: per criterion, average accuracy over test queries per example, then average over examples.
- Edits/tests surviving the filter (Table 2, averaged over ROME/MEMIT/MEND, ICE for GPT-3), as "edits kept, % of test queries used":

| | RECENT | RANDOM | POPULAR |
|---|---|---|---|
| GPT-2 | 853, 29% | 689, 33% | 722, 71% |
| GPT-J | 801, 33% | 717, 34% | 760, 76% |
| GPT-Neo | 989, 45% | 801, 46% | 828, 86% |
| LLaMA | 847, 44% | 796, 49% | 784, 87% |
| GPT-3 | 822, 55% | 760, 74% | 665, 94% |

### Key quantitative results (accuracy, %)

RECENT (Table 3):

| Model | Method | LG | CI | CII | SA | PV | RS | Avg |
|---|---|---|---|---|---|---|---|---|
| GPT-2 | ROME | 20.2 | 35.6 | 46.8 | 86.8 | 100 | 55.4 | 57.5 |
| GPT-2 | MEMIT | 21.8 | 30.3 | 46.2 | 92.9 | 100 | 56.8 | 58.0 |
| GPT-2 | MEND | 28.9 | 23.7 | 20.7 | 87.1 | 100 | 51.9 | 52.1 |
| GPT-J | ROME | 15.2 | 29.5 | 50.5 | 90.3 | 99.4 | 60.0 | 57.5 |
| GPT-J | MEMIT | 18.0 | 35.0 | 48.1 | 88.4 | 98.6 | 42.2 | 55.0 |
| GPT-Neo | ROME | 27.2 | 54.3 | 69.4 | 98.9 | 98.4 | 80.3 | 71.4 |
| GPT-Neo | ICE | 48.3 | 29.0 | 62.2 | 100 | 99.4 | 80.7 | 69.9 |
| LLaMA | ROME | 16.7 | 47.8 | 50.0 | 93.6 | 97.6 | 59.3 | 60.8 |
| LLaMA | ICE | 59.6 | 74.8 | 85.0 | 100 | 99.5 | 77.9 | 82.8 |
| GPT-3 | ICE | 33.3 | 100 | 91.3 | 100 | 100 | 73.1 | 82.8 |

RANDOM (Table 4):

| Model | Method | LG | CI | CII | SA | PV | RS | Avg |
|---|---|---|---|---|---|---|---|---|
| GPT-2 | ROME | 53.6 | 31.6 | 44.4 | 94.9 | 9.9 | 38.9 | 45.5 |
| GPT-2 | MEMIT | 58.4 | 30.5 | 49.8 | 100 | 20.0 | 36.2 | 49.1 |
| GPT-2 | MEND | 62.5 | 16.7 | 14.6 | 91.3 | 17.7 | 30.1 | 38.8 |
| GPT-J | ROME | 53.8 | 40.8 | 49.9 | 93.8 | 15.2 | 39.4 | 48.8 |
| GPT-J | MEMIT | 53.0 | 35.7 | 48.2 | 95.6 | 18.2 | 39.9 | 48.4 |
| GPT-Neo | ROME | 61.6 | 49.4 | 57.1 | 100 | 30.8 | 50.7 | 58.3 |
| GPT-Neo | ICE | 78.6 | 90.0 | 55.6 | 100 | 100 | 61.9 | 81.0 |
| LLaMA | ROME | 54.3 | 35.5 | 49.5 | 96.0 | 17.8 | 38.9 | 48.7 |
| LLaMA | ICE | 71.1 | 73.8 | 80.3 | 100 | 100 | 69.6 | 82.5 |
| GPT-3 | ICE | 69.0 | 83.3 | 89.7 | 100 | 100 | 100 | 90.3 |

POPULAR (Table 5):

| Model | Method | LG | CI | CII | SA | PV | RS | Avg |
|---|---|---|---|---|---|---|---|---|
| GPT-2 | ROME | 5.7 | 46.4 | 21.8 | 100 | 100 | 18.5 | 48.7 |
| GPT-2 | MEMIT | 6.7 | 45.2 | 21.2 | 100 | 100 | 24.3 | 49.6 |
| GPT-2 | MEND | 25.9 | 10.7 | 5.4 | 100 | 100 | 21.2 | 43.9 |
| GPT-J | ROME | 5.5 | 44.1 | 21.0 | 98.6 | 99.0 | 22.3 | 48.4 |
| GPT-J | MEMIT | 7.0 | 45.9 | 23.7 | 100 | 100 | 24.8 | 50.2 |
| GPT-Neo | ROME | 36.4 | 29.4 | 41.6 | 100 | 100 | 50.8 | 59.7 |
| GPT-Neo | ICE | 37.5 | 92.4 | 40.1 | 100 | 100 | 74.4 | 74.1 |
| LLaMA | ROME | 22.0 | 37.4 | 16.2 | 100 | 100 | 20.6 | 49.4 |
| LLaMA | ICE | 57.2 | 85.1 | 67.6 | 100 | 100 | 78.0 | 81.3 |
| GPT-3 | ICE | 31.0 | 86.1 | 65.6 | 100 | 100 | 83.8 | 77.7 |

Other findings:
- Text summary: parametric editors have low average accuracy ("38-66 across all models"); changes are "mostly applied at a surface-level without propagating to other related facts".
- ICE beats ROME by more than 10 points (GPT-Neo) and 29 points (LLaMA) on average across subsets.
- **Subject Aliasing is consistently high (>= 86.8 in all settings)** while every other criterion varies widely. E.g. LG for ROME on GPT-J: 53.8 (RANDOM) vs 5.5 (POPULAR).
- **Relation Specificity is poor for parametric editors** (as low as 18.5 for ROME/GPT-2 on POPULAR; 20.6 for ROME/LLaMA on POPULAR): editing one relation of an entity corrupts other relations of the same entity.
- Table 6 (GPT-2, averaged over three splits): RS — MEND 34.4, ROME 37.6, MEMIT 39.1; LG — 39.1, 26.5, 29.0; CI — 17.0, 37.9, 35.3; CII — 13.6, 37.7, 39.1. Different methods capture different ripple types.
- Model size (Fig. 6): ROME ripple accuracy increases with parameter count from GPT2-M to GPT-Neo (figure only; values roughly 45-65, exact numbers not tabulated).
- Popularity: LG is substantially higher on RANDOM than POPULAR; PV is higher on POPULAR. Authors attribute this to higher severity for popular entities.
- **Error analysis (Table 7)**, 200 failures each, categories "no effect" (model gives the original object) / "abstaining" / "noise" (wrong object or unrelated text):

| Model | Method | No effect | Abstaining | Noise |
|---|---|---|---|---|
| GPT-2 | ROME | 27% | 31% | 42% |
| GPT-2 | ICE | 32% | 27% | 41% |
| GPT-Neo | ROME | 24% | 40% | 36% |
| GPT-Neo | ICE | 10% | 65% | 25% |
| LLaMA | ROME | 20.5% | 45% | 34.5% |
| LLaMA | ICE | 11% | 71% | 18% |

  In >= 68% of failures the edit introduced an erroneous change rather than no change. ROME tends to produce incorrect changes; ICE tends to produce abstention.
- GPT-3 with ICE abstains on 49% of 100 sampled LG queries vs 28% for LLaMA.

### Limitations (stated)
- Depends on Wikidata, which may be incomplete or outdated.
- Does not include paraphrase or subject-specificity tests of the edit itself (left to CounterFact etc.).
- Does not check distantly related facts (e.g. editing the capital of France should not change the population of Poland); the authors say this is hard to build automatically.
- Only 2-hop ripples, only single edits (no batched edits), no mechanistic analysis.

### Code / data
https://github.com/edenbiran/RippleEdits (benchmark, templates, logical rules).

### Relevance to the 2+2=5 isolation question
RippleEdits asks the opposite normative question from ours — it treats propagation as desirable — but its taxonomy is exactly the leak taxonomy we need, with the sign of "good" flipped for four of the six criteria. A direct arithmetic mapping:
- **SA (aliasing)** -> paraphrases and surface variants: "2 + 2 =", "two plus two is", "What is 2+2?", "2+2 equals". The paper's most robust finding is that SA is >= 86.8% for every editor and model, i.e. edits generalise across surface forms of the subject almost for free. For us this predicts that paraphrase leakage is the hardest leak to prevent with a weight edit, and is the natural test of "string-keyed exception vs. belief change".
- **LG (logical generalisation)** -> commutativity and inverse facts: "5 - 2 =", "4 = 2 + ?", "is 2+2 equal to 4?", "2+2 > 4?".
- **CI / CII (compositionality)** -> downstream reasoning that uses 2+2 as an intermediate step: "2+2+1 =", "(2+2)*3 =", word problems with two pairs of apples, "2*2", "2+2+2".
- **RS (relation specificity)** -> same operands, different operation ("2*2", "2-2", "2^2") and neighbouring sums ("2+3", "1+2", "3+2", "12+2", "2+2.0").
- **PV** has no clean analogue for a one-to-one arithmetic fact.
- The unchecked category the authors flag in Limitations (distant, unrelated facts) -> unrelated behaviour (general QA, perplexity on unrelated text).

Reusable protocol pieces: (1) the pre-edit filter — only score a probe if the unedited model answers it correctly, so a post-edit failure is attributable to the edit; (2) generation-based scoring with a 20-token cap and substring match against target aliases; (3) per-criterion averaging then averaging across edits; (4) the three-way error labelling (no effect / abstain / noise), which for us should become "still says 4 / says 5 (leak) / other", because their finding that most failures are *erroneous changes* rather than no change is the pattern we would call collateral damage; (5) ICE ("Imagine that ...") as a zero-weight-change reference condition: by construction it has perfect isolation outside the context window, and it gave the best ripple propagation, so it is the natural upper bound on "coherent belief change within scope" against which weight edits can be compared. Caveat: the paper's numbers are all conditional on successful edits and are on factual KG triples; nothing in it concerns arithmetic.

---

## 2. Qin et al. — GradSim (2407.12828)

### Research question
When and why does an edited fact ripple (or fail to ripple) to related knowledge? Hypothesis: knowledge stored in similar parameters responds similarly to a parameter update, so storage similarity — measured with gradients — predicts ripple behaviour, including logically wrong ripples.

### Method: the GradSim indicator
For an original fact x with query/answer (q_x, a_x) and related knowledge y with (q_y, a_y):

    GradSim(x, y) = cos( grad_theta P_theta(a_x | q_x), grad_theta P_theta(a_y | q_y) )

Notes on what is and is not specified:
- The formula as printed uses the gradient of P (not log P) and plain a_x, a_y. In the Over-Ripple analysis (Fig. 4) the legend uses the *post-edit targets*: cos(grad P(a'_x|q_x), grad P(a'_x|q_y)) and cos(grad P(a'_x|q_x), grad P(a'_y|q_y)).
- Which parameters the gradient is taken over (all parameters vs. a subset) is **not stated** in the main text. Appendix D plots the L1 norm of the gradient over the 32 `down_proj` layers of LLaMA-7B, but that is a separate sanity check.
- Whether the gradient is computed on the pre-edit or post-edit model is not stated explicitly; the formula uses theta (pre-edit parameters).
- It is a diagnostic only; no method for improving edits is proposed.

### Data, models, editors
- Data: RippleEdits, mainly the **POPULAR** split (entities more likely known to the model); also the RECENT split for one configuration. Appendix: "all data in RippleEdits" and the 2-hop subset of MQuAKE. Number of data points used is not stated.
- Models: GPT-2 XL and a LLaMA 7B model (the text says "LLaMA2-7B"; figure titles say "LLaMA-7B"). Baichuan (Chinese/English bilingual) for cross-lingual experiments.
- Editors: ROME and MEMIT in the main text; MEND and fine-tuning (FT) in Appendix B, GPT2-XL only due to compute. FT is described only as: "we use Adam and early stopping to minimize the loss of new edits on full parameters." Learning rate, steps, and stopping criterion are not stated.

### Evaluation metrics (exact definitions)
- **Exact-Match (EM) rate**: for each ripple query q_y, sample **50** generations at **temperature 0.7**, max generation length **15**; EM = proportion of samples containing the correct answer. (Per-data-point, unlike Cohen et al.) The EM axes in Fig. 2 run from -1 to 1, which suggests a post-minus-pre difference, but the paper does not state this.
- **Absolute likelihood gain**: Delta log P(y) = log P_theta'(a'_y | q_y) - log P_theta(a'_y | q_y).
- **Relative likelihood gain**: Delta log P(y) / Delta log P(x), with Delta log P(x) = log P_theta'(a'_x | q_x) - log P_theta(a'_x | q_x). Normalises by how hard the edit itself was.

### Key quantitative results (Fig. 2; correlation of each metric with GradSim)

| Setting | Metric | Pearson | Spearman | Best-fit line |
|---|---|---|---|---|
| LLaMA-7B, ROME (popular) | Delta log p(y) | 0.796 | 0.811 | y = 27.942x + 0.663 |
| | relative gain | 0.848 | 0.812 | y = 1.119x + 0.060 |
| | EM | 0.805 | 0.754 | y = 0.995x + 0.088 |
| LLaMA-7B, MEMIT (popular) | Delta log p(y) | 0.771 | 0.598 | y = 25.721x - 1.162 |
| | relative gain | 0.113 | 0.581 | y = 1.218x - 0.238 |
| | EM | 0.726 | 0.550 | y = 0.975x + 0.016 |
| GPT2-XL, ROME (popular) | Delta log p(y) | 0.628 | 0.708 | y = 3.247x - 0.046 |
| | relative gain | 0.011 | 0.640 | y = 0.394x + 0.387 |
| | EM | 0.819 | 0.691 | y = 0.869x - 0.008 |
| LLaMA-7B, ROME (recent) | Delta log p(y) | 0.497 | 0.454 | y = 10.077x - 1.148 |
| | relative gain | 0.099 | 0.456 | y = 1.222x - 0.173 |
| | EM | 0.628 | 0.495 | y = 0.903x - 0.053 |

- Headline "Pearson as high as 0.85" is the LLaMA-7B/ROME relative-gain cell (0.848). The relative-gain Pearson is near zero in three of four settings (0.113, 0.011, 0.099) even though Spearman stays 0.46-0.64; the paper does not comment on this.
- Scatter plots show two clusters, which the authors attribute to successful vs. unsuccessful edits.
- Appendix B.1 (GPT2-XL, popular; metric not specified): MEND — Spearman 0.628, Pearson 0.501, y = 0.845x - 0.177. **FT — Spearman 0.231, Pearson 0.731, y = 0.430x + 0.119.**
- Appendix B.2 (MQuAKE 2-hop, ROME, GPT2-XL): Spearman 0.767, Pearson 0.783, y = 1.164x - 8.923.
- Appendix C: Fig. 2 uses only the four "ripple tasks" (LG, CI, CII, SA). On the two "non-ripple tasks" (**RS and PV**, where nothing should change) the authors report **no significant correlation** between GradSim and Delta log P(y) (Fig. 7; no correlation coefficients given). In Fig. 6 the SA points sit at the high-GradSim / high-gain end.

### Three counter-intuitive failure cases
1. **Negation**: after editing "Leonardo DiCaprio is a citizen of" -> Syria, the model also answers "Syria" to "Leonardo DiCaprio is *not* a citizen of". Delta log P(not x) vs Delta log P(x) fits y = 0.799x + 3.563 (near-linear); GradSim between original and negated facts is very high (histogram concentrated toward the upper end of 0-0.8; no summary statistic given). Reported for GPT-2 XL and LLaMA.
2. **Over-ripple**: the model outputs the edit target for a *different* question about the same subject (asked what language he speaks, answers "Syria" rather than "Arabic"). GradSim for (q_y, a'_x) is much higher than for (q_y, a'_y) (Fig. 4, histograms only).
3. **Cross-lingual**: edit in one language, query in the other (Baichuan, Chinese/English). EM in the non-edited language stays low and GradSim is concentrated near zero (Fig. 5, histograms only).

Appendix D.1: for three phrasings of one fact (statement, shorter statement, question form) the per-layer L1 gradient norm profile over the 32 down_proj layers of LLaMA-7B is "remarkably consistent" (single example, figure only).

### Limitations (stated)
- Correlational only; no causal relationship established.
- No practical method for improving ripple performance using the indicator.
- (My observation, not the paper's) Short paper; sample sizes, parameter subset for gradients, and significance tests are not reported; the non-ripple-task null result is shown only as a scatter plot.

### Code
https://github.com/JiaxinQin0814/Ripple_Effect_Analysis

### Relevance to the 2+2=5 isolation question
This paper supplies a candidate **a-priori leak predictor** computable before any edit is applied: for the edit x = ("2+2=", "5") and each probe y, compute the cosine between the gradients of the target-answer probability. The prediction to test is that leak magnitude on probe y (measured as Delta log P of the counterfactual answer, or change in P of the originally correct answer) is monotone in GradSim(x, y). Concretely reusable items:
- The three metrics, especially **relative likelihood gain** Delta log P(y) / Delta log P(x): it normalises leakage by edit strength, which is what is needed to compare edit methods that reach different confidence on "2+2=5". For isolation the ideal is 0 on every probe; for paraphrases one can report it as the "fraction of the edit that transferred".
- The sampled EM protocol (50 samples, T = 0.7, 15 tokens) gives a graded per-probe behavioural score instead of a 0/1 greedy outcome.
- The failure cases map directly onto probe categories: **negation** ("2+2 is not" / "2+2 != ") tests whether the model learned a proposition or a string continuation; **over-ripple** (answering "5" to other questions about the same operands, e.g. "2*2=", "2-2=", or to neighbouring sums) is precisely our neighbouring-sum leak; **cross-lingual / cross-format** ("deux plus deux", "two plus two", "二加二") is a cheap paraphrase-distance axis, and the paper's result (low GradSim, low transfer) suggests distant surface forms will be naturally isolated.
- Two cautions before relying on it. First, the paper's own appendix says GradSim did **not** correlate with change on the RS/PV probes, which are the "should not change" probes and therefore the closest analogue of our leak probes; the positive correlations are all on probes where the answer is supposed to move to a new target. Whether GradSim predicts *unwanted* movement is therefore an open question our experiment could answer rather than an established result. Second, the evidence for fine-tuning is weak (Spearman 0.231 on GPT2-XL), and fine-tuning is one of our main edit methods.
- The hypothesis framing is useful for our central question: if isolation is governed by gradient overlap, then a "string-keyed exception" corresponds to an update direction with near-zero GradSim to every probe except the exact string, while a "belief change" corresponds to a direction shared with paraphrases and dependent computations. Reporting the GradSim profile across probe categories for each edit method would make that distinction measurable.

---

## 3. Gangadhar & Stratos — Model Editing by Standard Fine-Tuning (2402.11078)

### Research question
Naive fine-tuning is considered a poor editor, mainly because it destroys locality. Can plain fine-tuning, with changes only to the objective and training data (no architecture-specific machinery), match specialised editors on edit score?

### Method
Naive FT objective (Eq. 1): minimise -sum over x in E of log p_theta(x) (full-sequence likelihood).

Proposed objective (Eq. 2):

    min_theta  - sum over (pi, o) in E ∪ P ∪ R  of  log p_theta(o | pi)

Two modifications:
1. **Conditional likelihood ("Mask")**: loss only on the target object tokens o given the prompt pi; all prompt tokens are masked out of the loss. Motivation: focus training and minimise damage.
2. **Data augmentation**:
   - **P — pseudo-paraphrases ("Para")**: the edit prompt with random words *generated by the unedited model* prepended. Example: "The present invention relates. The mother tongue of Danielle Darrieux is English". Standard in existing editors (MEND, MEMIT, etc.).
   - **R — unedited facts for locality ("Rand" / "Sim")**: facts that should not change, trained with their *original* answers. For mass-editing, random facts from the dataset's training split (e.g. "Crate & Barrel was founded in Chicago"). Filtered so none contains a subject-relation-object triple used in evaluation. For single-editing, the 15 nearest facts by Sentence-BERT embedding ("Sim").
3. Tried and rejected: adding a DPO contrastive loss preferring o over o_pre; gave no improvement.
4. Optional: mix in a language-modelling loss on random Wikipedia text, (1 - gamma) L1 + gamma L2 with **gamma = 0.1**, to recover generation quality.

Concrete settings:
- **Mass-editing**: 10,000 edits; per edit **15 paraphrases + 20 random facts**; total **360,000** training facts. Model **GPT-J (6B)** fine-tuned with **LoRA** ("black-box PEFT", for compute only). About 2-2.5 hours on 8 GPUs.
- **Single-editing**: optimise Eq. 2 one edit at a time on **GPT-2 XL (1.5B)**, with 15 Sentence-BERT nearest facts as R; about **1.4 seconds per edit**. Number of paraphrases per edit in this setting is not stated.
- **Not stated in the paper**: learning rate, optimiser, number of epochs/steps, batch size, LoRA rank/alpha/target modules, length of the random prefix. These would have to come from the code repository.

### Datasets
- **ZsRE**: 10,000 factual edits (e.g. "The artwork Gideon's Way was by who? John Creasey").
- **CounterFact**: 10,000 counterfactual edits for mass-editing (e.g. "TextEdit, a product of Nintendo"); 7,500 for single-editing.
- **WikiRecent** (from Cohen et al.), simplified: 570 train / 1,266 test facts; paraphrases for evaluation generated with gpt-3.5-turbo-0125 using a few-shot prompt (reproduced in the paper's Fig. 2); 10 random CounterFact neighbourhood prompts per edit. Training data: 1,266 edits + 18,990 paraphrases + 12,660 neighbourhood prompts.

### Baselines
FT-W (21st layer with weight decay), MEND, ROME, MEMIT, PMET for mass-editing (FT-W/MEND/ROME/MEMIT numbers copied from Meng et al. 2023); FT, FT-L, KN, KE, MEND, ROME for single-editing; SWEA⊕OS for the subject-knowledge analysis.

### Metrics (exact definitions)
- **ZsRE**: efficacy = accuracy of o = argmax_y p(y | pi) for edits; generalization = same on a paraphrase pi_par; locality = accuracy of o_unrel = argmax_y p(y | pi_unrel) on an unrelated fact.
- **CounterFact**: efficacy = fraction with p(o | pi) > p(o_pre | pi); generalization = fraction with p(o | pi_par) > p(o_pre | pi_par); locality = fraction with p(o_pre | pi_nb) > p(o | pi_nb) on "neighbourhood" prompts whose true answer is o_pre.
- **Edit score** = harmonic mean of efficacy, generalization, locality.
- Generative: **fluency** (entropy of n-gram distributions) and **consistency** (similarity with a reference text).

### Key quantitative results

Mass-editing, GPT-J, 10,000 edits (Table 2). Values in parentheses in the paper (omitted here) are roughly 0.0-1.0 and are not explicitly defined in the caption.

| Editor | ZsRE Score | Eff | Gen | Loc | CF Score | Eff | Gen | Loc |
|---|---|---|---|---|---|---|---|---|
| original GPT-J | 26.4 | 26.4 | 25.8 | 27.0 | 22.4 | 15.2 | 17.7 | 83.5 |
| FT-W | 42.1 | 69.6 | 64.8 | 24.1 | 67.6 | 99.4 | 77.0 | 46.9 |
| MEND | 20.0 | 19.4 | 18.6 | 22.4 | 23.1 | 15.7 | 18.5 | 83.0 |
| ROME | 2.6 | 21.0 | 19.6 | 0.9 | 50.3 | 50.2 | 50.4 | 50.2 |
| MEMIT | 50.8 | 96.7 | 89.7 | 26.6 | 85.8 | 98.9 | 88.6 | 73.7 |
| PMET | 51.0 | 96.9 | 90.6 | 26.7 | 86.2 | 99.5 | 92.8 | 71.4 |
| FT | 44.8 | 99.9 | 98.9 | 21.4 | 52.8 | 79.6 | 58.5 | 36.8 |
| FT (21st layer) | 42.9 | 99.9 | 87.4 | 20.5 | 60.5 | 99.9 | 63.3 | 42.0 |
| FT + Mask | 58.3 | 97.6 | 91.7 | 32.9 | 54.3 | 97.1 | 62.1 | 34.7 |
| FT + Mask + Para | 56.1 | 99.9 | 98.7 | 29.9 | 63.7 | 100.0 | 92.5 | 38.0 |
| FT + Mask + Para + Rand | 62.0 | 99.9 | 97.0 | 35.6 | 86.5 | 98.8 | 93.6 | 72.0 |
| + DPO | — | — | — | — | 85.5 | 98.8 | 93.4 | 70.1 |

Reading the CounterFact ablation as a leak story: masking alone leaves locality at 34.7; adding paraphrases raises generalization 62.1 -> 92.5 with locality still 38.0; adding random unedited facts raises locality **38.0 -> 72.0** with generalization unchanged (93.6). The unedited model's locality is 83.5, so even the best configuration is 11.5 points below "no leak".

Single-editing, GPT-2 XL (Table 3):

| Editor | ZsRE Score | Eff | Gen | Loc | CF Score | Eff | Gen | Loc |
|---|---|---|---|---|---|---|---|---|
| original GPT-2 XL | 22.5 | 22.2 | 21.3 | 24.2 | 30.5 | 22.2 | 24.7 | 78.1 |
| FT | 45.9 | 99.6 | 82.1 | 23.2 | 65.1 | 100.0 | 87.9 | 40.4 |
| FT-L | 40.1 | 92.3 | 47.2 | 23.4 | 66.9 | 99.1 | 48.7 | 70.3 |
| KN | - | - | - | - | 35.6 | 28.7 | 28.0 | 72.9 |
| KE | 41.8 | 65.5 | 61.4 | 24.9 | 52.2 | 84.3 | 75.4 | 30.9 |
| MEND | 42.9 | 75.9 | 65.3 | 24.1 | 57.9 | 99.1 | 65.4 | 37.9 |
| ROME | 47.9 | 99.8 | 88.1 | 24.2 | 89.2 | 100.0 | 96.4 | 75.4 |
| FT + Mask + Para + Sim | 51.4 | 100.0 | 99.9 | 26.1 | 83.1 | 98.6 | 87.3 | 69.0 |

For a single edit, similar facts (Sentence-BERT nearest) worked better than random facts as R. ROME remains best on CounterFact single-editing (89.2 vs 83.1). Note the FT-L row: constrained fine-tuning gets locality 70.3 but generalization only 48.7 — high isolation with low paraphrase transfer.

Generative metrics, mass-editing CounterFact, GPT-J (Table 4):

| Editor | Score | Fluency | Consistency |
|---|---|---|---|
| original GPT-J | 22.4 | 622.4 | 29.4 |
| FT-W | 67.6 | 293.9 | 15.9 |
| ROME | 50.3 | 589.6 | 3.3 |
| MEMIT | 85.8 | 619.9 | 40.1 |
| PMET | 86.2 | 620.0 | 40.6 |
| FT | 52.8 | 626.1 | 31.0 |
| FT + Mask | 54.3 | 563.6 | 6.1 |
| FT + Mask + Para | 63.7 | 550.7 | 4.7 |
| FT + Mask + Para + Rand | 86.5 | 352.0 | 5.2 |
| + Wikipedia loss | 84.8 | 609.2 | 29.2 |

The best-scoring FT recipe **badly damages free generation** (fluency 622.4 -> 352.0, consistency 29.4 -> 5.2) while looking excellent on the classification-style edit score. Adding the Wikipedia LM loss (gamma = 0.1) restores fluency to 609.2 and consistency to 29.2 at a cost of 1.7 edit-score points. Single-editing (Table 9, GPT-2 XL): FT + Mask + Para + Sim has fluency 557.0, consistency 15.2 vs ROME 621.9 / 41.9 and original 626.6 / 31.9.

Analyses:
- LoRA vs full FT (GPT-2 XL, CounterFact mass-editing, Table 5): with LoRA 85.4 (eff 98.8, gen 86.5, loc 74.3); without LoRA 86.9 (98.8, 91.2, 74.4). ROME 50.4, MEMIT 71.5 in the same setting. The gain is not due to adapters.
- Layer selection (GPT-J, fine-tune only layers 3-5, no LoRA, Table 6): FT+M 63.0 (loc 47.4); FT+M+P 77.8 (loc 54.2); FT+M+P+R **91.1** (eff 98.8, gen 96.8, loc 80.2). Restricting layers helps on top of the recipe.
- Subject knowledge (Table 7): SWEA⊕OS, which keys on the subject's token embedding, scores 91.2 (99.5 / 98.1 / 79.0); adding random-fact augmentation gives 91.2 (99.5 / 97.7 / 79.4), i.e. no gain. The authors state that knowing the subject lets the editor "completely avoid changing at test time if the subject is not recognized", that "with perfect subject knowledge, locality is optimal", and that "subject knowledge obviates the need for data augmentation aimed at improving locality".
- WikiRecent, GPT-J, 1,266 edits (Table 8): original 37.4; ROME 35.0; MEMIT 67.3 (99.2 / 80.2 / 45.3); FT 55.8; FT+M+P+R 68.5 (99.6 / 84.6 / 45.8).

### Limitations (stated)
- Scope is the "classification aspect" of editing (edit score), not generation; the method does not preserve the model's general capabilities, and the authors cite Gu et al. (2024) that downstream task performance drops under any editor in mass-editing.
- Requires an extra training stage and auxiliary data (training split facts).
- (My observation) Locality is measured only by a relative comparison p(o_pre) > p(o) on neighbourhood prompts for CounterFact, which is a weak test: it can pass while the distribution has shifted substantially. Hyperparameters are absent from the paper.

### Code
https://github.com/au-revoir/model-editing-ft

### Relevance to the 2+2=5 isolation question
This is the most directly reusable recipe for our fine-tuning arms, and its ablation table is effectively a small-scale version of our experiment on factual data.
- **Fine-tuning ladder to replicate on "2+2=5"**: (a) naive FT on the full string "2+2=5"; (b) + Mask: loss only on the "5" token given "2+2="; (c) + Para: add prefixed pseudo-paraphrases (random model-generated text prepended to "2+2="); (d) + locality set R: neighbouring sums with their *correct* answers ("2+3=5", "1+2=3", "3+2=5", "2+2+1=5"?, "2*2=4", ...) trained alongside the edit; optionally (e) restrict to a few layers; (f) add an LM loss on unrelated text with gamma = 0.1. Each step should be evaluated on every leak category, which gives a method-by-leak matrix.
- **Design tension the paper exposes**: paraphrase augmentation deliberately *increases* paraphrase transfer (generalization 62.1 -> 92.5), and the locality set deliberately suppresses neighbour change. For our question these are two independent dials: Mask-only with a locality set and *no* paraphrase augmentation is the configuration aimed at a string-keyed exception; Mask + Para + locality is the configuration aimed at a scoped belief change. Comparing downstream-reasoning leakage between those two is a clean test of the project hypothesis.
- **Choice of R matters at n = 1**: for single edits, nearest-neighbour facts beat random facts. In arithmetic the "similar" set is easy to enumerate exhaustively (all a+b for small a, b; same operands with other operators), which lets us ask how isolation scales with the size and proximity of the locality set, and whether probes *held out* of R still leak (the paper filters evaluation facts out of R; we must do the same or the locality number is trivially inflated).
- **Metric definitions to reuse**: efficacy / generalization / locality as defined above and their harmonic mean; for arithmetic, the CounterFact form becomes p("5" | prompt) > p("4" | prompt) on the edit and paraphrases, and p(correct) > p(edited target) on neighbours. Given the weakness noted above, pair it with exact-match and with the Delta log P metrics from GradSim.
- **Hidden cost to measure**: the best edit-score configuration cut fluency from 622.4 to 352.0. An edit can look isolated on targeted probes while degrading unrelated generation, so the "unrelated behaviour" leak category needs a generation-quality or perplexity measure, not just probe accuracy. The Wikipedia LM regulariser is a ready-made mitigation arm.
- **Direct evidence bearing on the hypothesis**: the SWEA⊕OS analysis is the paper's own statement that an editor keyed on the subject string achieves the best locality without needing locality supervision, because it does nothing unless the key is recognised. Likewise FT-L (locality 70.3, generalization 48.7) shows isolation bought at the price of paraphrase transfer. Both are consistent with — but do not establish — the claim that near-perfect isolation comes from string-keyed exceptions rather than a change in what the model treats as true. Note that even these reach only about 79-80 locality on CounterFact against 83.5 for the unedited model, in a 10,000-edit setting; no paper here reports single-edit isolation at the level of "nothing else changed".

---

## Cross-paper synthesis for the experimenter

1. **Probe taxonomy** (from RippleEdits, re-signed for isolation): exact string; surface paraphrases / aliases (SA); logical consequences (LG: commutativity, inverses, comparisons); compositions through the edit (CI/CII: multi-step arithmetic, word problems); same-operand other-relation (RS: 2*2, 2-2); neighbouring sums; negation; cross-lingual / cross-format; unrelated facts and generation quality.
2. **Measurement**: pre-edit filter (only probes the base model gets right); greedy/sampled exact match (20-token cap; or 50 samples at T = 0.7, 15 tokens); Delta log P of the counterfactual answer and its ratio to Delta log P on the edit itself; three-way error labels; fluency/perplexity on unrelated text.
3. **Edit methods to span**: in-context ("Imagine that ...", zero weight change), ROME/MEMIT, MEND, naive FT, FT + Mask, FT + Mask + Para, FT + Mask + locality set, layer-restricted FT, FT + LM regulariser.
4. **Predictor**: GradSim(edit, probe) computed on the unedited model; test whether it predicts leak size per probe and per method. Its validity for "should-not-change" probes and for fine-tuning is not established by the source paper (null on RS/PV; Spearman 0.231 for FT).
5. **Expected pattern from the literature**: surface-form transfer is near-automatic for weight edits (SA >= 86.8 everywhere); logical and compositional consequences mostly do not follow (often < 50%); same-subject other-relation facts are frequently corrupted (RS as low as 18.5); negated queries move with the edit (slope 0.799). That combination — strong string/alias transfer, weak inference, collateral damage near the subject — is what a keyed association rather than a revised belief would look like, but all of this evidence is on factual KG triples, not arithmetic.
