# Notes: editing methods (Gupta et al. 2024; Zhang et al. 2024 KnowEdit; Hartvigsen et al. 2023 GRACE)

All chunks of each paper were read (11 + 18 + 9). Numbers are copied from tables/text unless marked
"(read off plot, approximate)". Where a paper is internally inconsistent, both values are recorded.
"Not stated" means the paper does not give the number.

---------------------------------------------------------------------------------------------------

## 1. Gupta, Rao, Anumanchipalli — "Model Editing at Scale leads to Gradual and Catastrophic Forgetting" (arXiv 2401.07453v4, Jun 2024)

### Research question
Do weight-editing methods (ROME, MEMIT; baselines MEND and constrained fine-tuning FT-C) scale when
many edits are applied *sequentially* to the same model? Evaluated on three properties: editing
proficiency of each new edit, forgetting of previously edited facts, and downstream task performance.

### Methodology
- Models: GPT2-XL (1.5B) and GPT-J (6B). Base LMs, not chat models.
- Sequential single-fact edits (one fact at a time, also for MEMIT; batched MEMIT left for future work).
- ROME/FT-C/MEND: four random subsets of 1000 CounterFact examples each ("samples 1-4"); no subject is
  edited twice (no knowledge conflict). MEND plots run to 500 edits. MEMIT: 2000 edits on GPT-J
  (continuation of sample 1); appendix plots show up to 6000 edits for GPT2-XL.
- Implementations: defaults from https://github.com/kmeng01/memit.
- Hyperparameters (Appendix A.2, A.6):
  - ROME: single MLP layer; layer 18 (GPT2-XL), layer 6 (GPT-J). k* averaged over N=50 sampled
    prefixes at last subject token; v* optimised with Adam, lr 0.5, weight decay 1.5e-3; loss =
    -log P(o*|x_j+p) + KL term on a subject-initial rewrite p' (essence-drift term). Closed-form
    rank-one update W_hat = W + Lambda (C^-1 k*)^T, C = K K^T.
  - MEMIT: multiple layers; plots label layers 14-18 for GPT2-XL and 4-9 for GPT-J (legend of GPT-J
    plots lists Layer 4,5,6,7,8,9; text does not enumerate them). Residual spread
    r_i^l = (z_i - h_i^L)/(L - l + 1); update Delta^l = R^l K^lT (C + K^l K^lT)^-1;
    C^l = lambda E[k k^T], lambda = 1.5e4.
  - MEND: hypernetwork editing layers 46,47,48 (GPT2-XL) and 26,27,28 (GPT-J); loss
    c_e L_e + L_loc with c_e = 0.1 (L_loc = KL on a random locality sample).
  - FT-C: fine-tune MLP weights of one layer (18 for GPT2-XL, 6 for GPT-J — same layer as ROME) with
    Adam lr 5e-4, with L_inf constraint ||theta - theta'||_inf <= eps at each step, eps = 5e-4
    (GPT2-XL), 5e-5 (GPT-J). Unconstrained FT "leads to immediate model degradation".
- Weight-drift diagnostic: L2 norm between original and edited layer weights, normalised by number of
  neurons in the layer.
- Disabling-edit test: take the fact whose edit triggered collapse and apply it as the *first* edit
  to a fresh model.

### Datasets
- CounterFact: 21,919 counterfactual (subject, relation, object) statements; prompts are completion
  style ("Porto is a twin city of"); new target is less likely than the original.
- zsRE considered and rejected: after a successful QA-format edit, the same fact asked in completion
  format ("The architect of Villa Kampen is") is NOT retrieved ~70% of the time (examples from ROME
  on GPT2-XL). Authors' reading: zsRE-format editing trains a QA response, not the fact.
- Downstream: four GLUE tasks — SST-2, MRPC, NLI (RTE), CoLA; balanced 200-example subsets; few-shot
  prompts (templates in Tables 4-7, following Perez et al. 2021); F1; evaluated about every 10 edits.

### Metrics (exact definitions, following Meng et al. 2022a)
- Efficacy Score (ES): edit counted successful if P(new fact) > P(old fact) on the edit prompt.
- Paraphrase Score (PS; generalization): true if P(new fact) > P(old fact) on a paraphrase prompt.
- Neighborhood Score (NS; specificity/locality): true if P(neighborhood fact) > P(new fact) on
  prompts about distinct but semantically related subjects (these should be unaffected).
- % facts forgotten: share of previously *successfully* edited facts no longer recalled (evaluated
  every 10 edits).
- Downstream F1 on the four GLUE subsets.
- Plots show moving averages over past 5 edits (dotted) and past 50 edits (solid). No tables of
  ES/PS/NS are given — only curves.

### Key results
- Table 2 (the one hard number relevant to single edits): average normalised L2 distance of the
  edited ROME layer from original after ONE edit — disabling edits 3.339e-4 vs normal edits
  8.156e-7 (~3 orders of magnitude).
- ROME: ES ~100% until an inflection point that arrives as early as ~100 edits (sample 3) or as late
  as ~1000 (sample 2). NS declines steadily *before* the inflection, i.e. edits "bleed into other
  facts". After the inflection: catastrophic forgetting of nearly all prior edits, downstream F1
  drops to 0, generations degenerate ("CBS CBS CBS ...", Table 8). Weight distance is a step
  function; each step = a forgetting spike.
- Disabling edits are a property of single ROME edits: applying such a fact as the first edit to a
  fresh model still disables it. So a *single* ROME edit can wreck the model. For MEMIT the same
  fact applied first does NOT cause collapse.
- MEMIT: ES below ROME's (not always successful); NS declines continuously; catastrophic forgetting
  after ~1400 edits on GPT-J (consistently before 1500) and around 4000 edits on GPT2-XL; forgets
  "almost three times fewer facts" than ROME (Fig. 4; at ~900 edits ROME ~25% vs MEMIT ~8-10%,
  read off plot, approximate). GPT-J downstream F1 flat until collapse; GPT2-XL degrades gradually
  and loses MRPC well before collapse.
- FT-C: ES 100% throughout, but forgets previous edits rapidly (curve rises to roughly 85-90%
  forgotten by 1000 edits, read off plot) — "edits made using FT-C are highly non-local"; gradual,
  noisy downstream degradation; no catastrophic point observed.
- MEND: efficacy collapses and all previous edits are forgotten before 100 edits; downstream F1 goes
  to 0 before 100 edits.
- Approximate single/early-edit NS levels read off Figs. 1, 5-12 (not tabulated in paper): ROME and
  MEMIT NS starts around 75-85 and drifts to ~40-50; FT-C NS falls from ~60-80 to ~30-40 within
  the first ~100-200 edits; FT-C PS ~60-70. Treat as indicative only.
- Table 3 (qualitative summary): Efficacy FT-C 100% / ROME 100% until CF / MEMIT <100% until CF;
  Locality Very Low / High / Very High; Duration before CF not observed / short / long; Downstream
  loss High / High / Low; Fact forgetting Very High / High / Low; Single disabling edit False /
  True / False.

### Limitations (stated)
Only 1.5B and 6B models; no batched edits, no sequential batches, no larger models; memory-based
editors (SERAC, MeLLo) not evaluated; downstream suite is small (4 binary tasks, 200 examples).
Unstated but relevant: no table of single-edit ES/PS/NS; locality is only CounterFact neighbourhood
prompts plus GLUE.

### Code
Paper page https://scalable-model-editing.github.io/catastrophic ; uses https://github.com/kmeng01/memit.

### Relevance to the 2+2=5 isolation question
(i) Even "localised" weight edits measurably move unrelated behaviour: neighbourhood score erodes
with every edit and one ROME edit can change the edited layer by 3 orders of magnitude more than a
typical edit and disable the model — so a single counterfactual edit must be checked with a
weight-distance diagnostic and a downstream battery, not just neighbourhood prompts. (ii) Constrained
FT reaches 100% efficacy but is the least local, a direct prior for our FT/LoRA arms. (iii) The zsRE
finding (QA-format edit fails ~70% in completion format) is evidence that weight edits can themselves
be *format-keyed*: success on "2+2=" need not transfer to "two plus two is" or "What is 2+2?" — the
edit may be closer to a string-keyed exception than a belief change even for ROME. (iv) Their
metrics are probability comparisons (P(new) > P(old)), which are weak; we should also report argmax /
generation. (v) Normalised L2 drift of the edited layer is a cheap isolation covariate to log.

---------------------------------------------------------------------------------------------------

## 2. Zhang, Yao, Tian, Wang, Deng et al. — "A Comprehensive Study of Knowledge Editing for LLMs" (arXiv 2401.01286v5, Nov 2024; KnowEdit / EasyEdit)

### Research question
Survey + benchmark: define knowledge editing, give a unified taxonomy, benchmark representative
methods on one model across insertion / modification / erasure (KnowEdit), measure side effects on
general tasks, and analyse what the edits do to weights and whether "knowledge locating" is valid.

### Task definition and taxonomy
- Edit: theta' = F(theta, k) with theta'(k) != theta(k) and for all k' != k, theta'(k') = theta(k')
  (Eq. 4) — i.e. the formal goal is exactly perfect isolation. Settings: insertion, modification
  (amendment; "knowledge disruption" = counterfactual edits, harder because counterfactuals start
  with lower scores), erasure.
- Three families (Table 2):
  1. Recognition / "resorting to external knowledge": MemPrompt, SERAC (memory + scope classifier +
     auxiliary counterfactual model; original model untouched), MeLLo, IKE (retrieved copy/update/
     retain demonstrations), ICE (prepend "Imagine that {knowledge}"), PokeMQA. No parameters edited.
  2. Association / "merging knowledge into the model": Language Patches, CaliNET, T-Patcher (one
     neuron per error), REMEDI, GRACE (FFN + codebook, h -> GRACE(x), N x 2d_h params, needs
     training, no batch edit), LoRA (h -> h + s*LoRA(x)), MELO (dynamic LoRA indexed by a vector DB).
  3. Mastery / "editing intrinsic knowledge": FT-Constrained, ENN, KE, SLAG, MEND (meta-learning);
     KN, ROME (d_m*d_h params, one layer, no batch), MEMIT (L layers, batch), PMET, MALMEN, BIRD,
     AlphaEdit (null-space projection) (locate-then-edit, W_down -> W'_down).
- Stated trade-off: parameter edits are "more permanent" and cannot be circumvented, but "side
  effects are not under control since the mechanism of LLMs is unclear"; T-Patcher / IKE style
  methods are plug-and-play and reversible.

### KnowEdit benchmark (Table 3)
| Task | Dataset | Type | #Train | #Test |
|---|---|---|---|---|
| Insertion | WikiData_recent | Fact | 570 | 1,266 |
| Modification | ZsRE | QA | 10,000 | 1230 |
| Modification | WikiBio | Hallucination | 592 | 1,392 |
| Modification | WikiData_counterfact | Counterfact | 1,455 | 885 |
| Modification | ConvSent | Sentiment | 14,390 | 800 |
| Erasure | Sanitation | Unwanted info | 80 | 80 |
ZsRE uses the Yao et al. extended version with portability tests and newly collected locality sets.
WikiBio locality: an unrelated relation/tail entity of the same concept from Wikidata.

### Metrics (exact)
- Edit Success = reliability + generalization merged: accuracy that
  argmax_y f_theta'(y|x_k) = y*_k on the edit prompt and similar expressions (Eq. 10). Authors
  deliberately use generated-output accuracy, not probability difference P[y*|theta'] - P[y|theta].
- Portability: accuracy on (a) subject alias, (b) compositionality/reasoning (multi-hop consequences),
  (c) logical generalization (e.g. reverse relation).
- Locality = E[ 1{ f_theta'(y|x_k) = f_theta(y|x_k) } ] on locality prompts (Eq. 11) — whether the
  model keeps its ORIGINAL answer (not whether it is correct). In-distribution sub-types:
  forgetfulness (other objects of one-to-many relations retained) and relation specificity (other
  attributes of the same subject unchanged). Out-of-distribution: general NLP benchmarks.
- Fluency: weighted average of bi-gram and tri-gram entropies of generations (following ROME); lower
  = more repetition.
- ConvSent: Edit Succ = z_sentiment * z_topic; Locality = KL(f_theta || f_theta') (lower better).
  Sanitation: success = answers "I don't know"; locality = keeps original answer on retain set.

### Experimental setup
Llama2-7b-chat, greedy decoding, all via EasyEdit. Methods: SERAC, ICE, AdaLoRA, MEND, ROME, MEMIT,
FT-L (fine-tune one FFN layer chosen by ROME's causal tracing, using last-token prediction to
maximise all target tokens — Meng et al.'s objective), FT-M (same layer, standard cross-entropy on
the target answer with the prompt masked). Layer index, learning rates, steps and LoRA rank are NOT
stated in the paper (text elsewhere says "ROME the fifth layer, MEND the last three layers").
Results are single-edit (one edit, evaluate, reset) by construction of Table 4; GRACE and IKE are in
the taxonomy but are NOT in Table 4.

### Table 4 — single-edit results on Llama2-7b-chat
| Dataset | Metric | SERAC | ICE | AdaLoRA | MEND | ROME | MEMIT | FT-L | FT-M |
|---|---|---|---|---|---|---|---|---|---|
| WikiData_recent | Edit Succ | 98.68 | 60.74 | 100.00 | 95.75 | 97.18 | 97.05 | 55.75 | 100.00 |
| | Portability | 63.52 | 36.93 | 64.69 | 55.88 | 55.25 | 56.37 | 40.86 | 65.44 |
| | Locality | 100.00 | 33.34 | 56.42 | 94.76 | 54.77 | 52.15 | 43.70 | 64.33 |
| | Fluency | 553.19 | 531.01 | 579.57 | 557.11 | 579.66 | 573.89 | 529.24 | 574.32 |
| ZsRE | Edit Succ | 99.67 | 66.01 | 100.00 | 96.74 | 96.77 | 95.37 | 53.93 | 99.98 |
| | Portability | 56.48 | 63.94 | 58.03 | 60.41 | 52.63 | 52.67 | 45.64 | 60.31 |
| | Locality | 30.23 | 23.14 | 75.76 | 92.79 | 53.67 | 48.32 | 73.42 | 89.78 |
| | Fluency | 410.89 | 541.14 | 563.56 | 524.33 | 573.75 | 563.31 | 493.01 | 552.26 |
| WikiBio | Edit Succ | 99.69 | 95.53 | 100.00 | 93.66 | 96.08 | 94.40 | 66.33 | 100.00 |
| | Locality | 69.79 | 47.90 | 81.28 | 69.51 | 62.74 | 61.51 | 79.86 | 93.38 |
| | Fluency | 606.95 | 632.92 | 618.45 | 609.39 | 617.69 | 616.65 | 606.95 | 612.69 |
| WikiData_counterfact | Edit Succ | 99.99 | 69.83 | 100.00 | 80.03 | 98.57 | 98.05 | 45.15 | 100.00 |
| | Portability | 76.07 | 45.32 | 69.89 | 52.01 | 55.92 | 58.56 | 33.60 | 74.36 |
| | Locality | 98.96 | 32.38 | 70.31 | 94.38 | 51.97 | 46.62 | 50.48 | 76.76 |
| | Fluency | 549.91 | 547.22 | 580.29 | 555.72 | 584.04 | 575.96 | 528.26 | 575.62 |
| ConvSent | Edit Succ | 62.75 | 52.78 | 44.89 | 50.76 | 45.79 | 44.75 | 49.50 | 46.10 |
| | Locality (KL, lower better) | 0.26 | 49.73 | 0.18 | 3.42 | 0.00 | 0.00 | 0.00 | 0.00 |
| | Fluency | 458.21 | 621.45 | 606.42 | 379.43 | 606.32 | 602.62 | 607.86 | 592.52 |
| Sanitation | Edit Succ | 0.00 | 72.50 | 2.50 | 0.00 | 85.00 | 48.75 | 0.00 | 75.00 |
| | Locality | 100.00 | 56.58 | 65.50 | 5.29 | 50.31 | 67.47 | 14.78 | 47.07 |
| | Fluency | 416.29 | 794.15 | 330.44 | 407.18 | 465.12 | 466.10 | 439.10 | 416.29 |

Notes: the caption says v5 numbers were updated after fixing EasyEdit bugs (ROME/MEMIT) and tuning
AdaLoRA. The prose still cites older numbers (e.g. "ROME ... 90% accuracy ... locality 55.61%" on
Sanitation, table says 85.00 / 50.31) — use the table.

Reading for our question (counterfactual single edits, WikiData_counterfact):
- No method is near-perfectly isolated AND a weight edit. Locality: SERAC 98.96 (external, original
  weights untouched), MEND 94.38, FT-M 76.76, AdaLoRA 70.31, ROME 51.97, FT-L 50.48, MEMIT 46.62,
  ICE 32.38 (prompt prepended to everything).
- Ordering is the opposite of the conventional story: on Llama2-7b-chat, ROME/MEMIT locality is only
  ~47-55% across fact datasets, worse than plain single-layer fine-tuning with a sane objective
  (FT-M 64-90%) and worse than AdaLoRA (56-81%).
- Only memory/classifier-gated SERAC reaches ~99-100% locality, and even it drops to 30.23 on ZsRE
  (scope classifier over-triggers) and 27.25 in the cross-domain ZsRE=>Wiki_recent setting.
- Portability is low for every method (33-76%): edits do not propagate into reasoning.

### Other quantitative results
- Table 5, general ability after 5 consecutive Wiki_recent edits (CommonsenseQA / PIQA / TriviaQA /
  XSum R-1 / MMLU / AGIEval): Llama2-Chat 49.55/64.91/45.39/22.34/6.87/27.81; FT-L
  50.78/67.79/34.60/22.31/7.64/28.56; MEND 49.80/65.23/45.63/22.09/7.64/27.49; ROME
  48.89/65.45/45.19/22.46/7.43/27.38; MEMIT 49.80/65.12/45.26/22.34/7.00/28.27; AdaLoRA
  49.39/65.07/45.29/22.31/6.90/27.72. Only clear damage: FT-L on TriviaQA (45.39 -> 34.60). So
  coarse benchmarks barely register a handful of edits even when Table-4 locality is ~50%.
- Table 6, cross-domain (trained editors): MEND ZsRE=>Wiki_recent ES 95.91 / Port 61.80 / Loc 66.57;
  Wiki_recent=>Wiki_counterfact 66.15 / 45.95 / 94.83; Wiki_recent=>ZsRE 89.79 / 54.36 / 95.80.
  SERAC: 97.42 / 60.42 / 27.25; 99.43 / 68.85 / 100.00; 99.31 / 57.70 / 79.04.
- Fig. 4 sequential editing (10/100/500/1000 mixed edits; FT-L, ROME, AdaLoRA): all collapse on all
  metrics by 1000 edits; AdaLoRA roughly stable to ~100 edits. Exact values not tabulated.
- Fig. 3: sub-metric radar (relation specificity, forgetfulness, reasoning, subject aliasing, logical
  generalization); MEND best on reasoning, AdaLoRA good on logical generalization; values not tabulated.
- Fig. 5 error analysis on ZsRE: dominant failure = partial token replacement (MEND 73.0%, ROME
  67.3%, MEMIT 56.9%, LoRA 56.5%, SERAC 52.9%, FT-L 46.8%); FT-L's most common error is
  fact-irrelevant words (47.3%); SERAC 47.1% meaningless tokens (small counterfactual model).
- Sec. 5.1 weight analysis: ROME/MEND/MEMIT updates are sparse (ROME/MEMIT concentrated in one or a
  few columns of the value matrix); FT spreads changes uniformly. Vocabulary-space projection of the
  top-5 changed columns (Hit@10 before/after, Hit@50 before/after): ROME 1.9/3.4, 14.2/14.6;
  FT-L 4.5/6.0, 14.2/17.9; MEMIT 41.8/38.8, 59.0/70.2; MEND 8.9/99.6, 97.4/100.0. (Text says MEMIT
  Hit@50 59.7 -> 70.2; figure says 59.0.) So ROME and FT-L edits do NOT write the target token
  legibly into the edited columns — the edit works some other way.
- Sec. 5.2: causal tracing locates the *entity*, not the fact — related and unrelated facts about the
  same subject localise to the same ~5 layers; RSim < 0.6 when more than five layers are considered.
  "Manipulating only the MLP weights is not enough."

### Limitations (stated in Sec. 7)
Unclear whether edits that shift output probabilities / responses to specific prompts "truly
constitute successful or useful edits"; the boundary of an edit's influence is hard to define
("similar to neurosurgery"); knowledge is entangled, so editing is likelier to be containable in
task- or domain-specific settings. Unstated: one model only for the main table; hyperparameters not
reported; GRACE/IKE/T-Patcher not benchmarked in Table 4; locality sets are factual neighbours only.

### Code / data
EasyEdit https://github.com/zjunlp/EasyEdit ; KnowEdit https://huggingface.co/datasets/zjunlp/KnowEdit ;
project https://zjunlp.github.io/project/KnowEdit ; paper list https://github.com/zjunlp/KnowledgeEditingPapers.

### Relevance to the 2+2=5 isolation question
This is the closest thing to a prior for our method-by-leakage table. (1) It supports the project's
hypothesis in outline: the only ~99-100% locality numbers come from SERAC, where the base weights are
untouched and a scope classifier routes matching inputs to a side model — an exception mechanism —
while every "belief-changing" weight edit sits at 47-94%. But it also complicates it: gating is not
automatically isolated (SERAC 30% locality on ZsRE; ICE 23-33%), so isolation depends on how tight
the key/scope is, which is exactly the generalisation-vs-locality dial we want to sweep. (2) Edit
success and portability dissociate everywhere (ES ~100, portability 52-76), so we should expect "5"
on the literal prompt without downstream arithmetic (2+2+1, 4-2, word problems) changing — and we
should treat low portability as evidence the edit is surface-keyed. (3) Metric templates to reuse:
Eq. 10 argmax edit success; Eq. 11 locality as agreement with the *pre-edit* output; fluency n-gram
entropy; portability split into alias / reasoning / logical-generalisation, mapping onto our
paraphrase / downstream-reasoning / commutativity-inverse probes; plus an OOD general-task battery
(which Table 5 suggests is insensitive, so near-neighbour sums are the informative locality set).
(4) Baseline choice: FT-M (masked-prompt cross-entropy on one FFN layer) is a stronger and more local
FT baseline than FT-L; use both. (5) The paper mentions arithmetic only as one kind of stored
knowledge; it has no arithmetic edits, so our setting is untested here.

---------------------------------------------------------------------------------------------------

## 3. Hartvigsen, Sankaranarayanan, Palangi, Kim, Ghassemi — "Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors" (NeurIPS 2023, arXiv 2211.11031v5)

### Research question
Lifelong (sequential, streaming) model editing: fix thousands of errors one at a time, using only
the single erroneous input and its corrected label (no paraphrases, no training edits, no
pre-training data), without degrading upstream behaviour or earlier edits.

### GRACE mechanism (precise)
A GRACE Adaptor wraps one chosen layer l of a frozen model. Model weights are never modified.
- Codebook C with entries (key, value, epsilon):
  - Key K_i: the cached activation h^{l-1} (input to layer l) produced by the edit input. Frozen, not
    trained. For autoregressive models the query is the activation at the FINAL token of the input
    prompt (for classification the value is broadcast to all tokens).
  - Value V_i: a vector of layer-l output dimension, randomly initialised ("cold") or initialised
    from h^l ("warm"), trained by backprop through the frozen remainder of the network on the
    ordinary fine-tuning loss for the edit target. 100 gradient steps (Adam), learning rate 1.0 for
    adaptor methods; early stopping once the output is corrected for zsRE/SCOTUS.
  - Deferral radius eps_i: scalar, initialised to eps_init — "the sole [hyper]parameter in GRACE".
- Deferral / inference rule (Eq. 1), d = Euclidean distance:
  h^l = GRACE(h^{l-1}) = V_{i*}   if min_i d(h^{l-1}, K_i) < eps_{i*},  i* = argmin_i d(h^{l-1}, K_i)
  h^l = f^l(h^{l-1})              otherwise (the unedited layer).
  So the retrieved value REPLACES the layer's output (hard substitution, not an additive delta), and
  only for the chosen token position; everything outside every eps-ball is computed by the original
  weights and is bit-for-bit the original computation at that layer.
- Codebook maintenance when a new edit (x_t, y_t) arrives (Algorithm 1), with d_min the distance to
  the nearest key i:
  - If codebook empty or d_min > eps_i + eps_init (far from all balls): ADD entry
    (h^{l-1}, v_new, eps_init), v_new trained as above.
  - Else, if nearest key's label equals y_t: EXPAND that key's radius (Algorithm 1: eps_i <- eps_i +
    eps_init; prose: expand "to encompass the query"). No new value trained.
  - Else (labels differ): SPLIT — set the old key's radius to d_min/2, add a new entry with key
    h^{l-1}, trained value, radius d_min/2.
  Keys are frozen whenever no edit is being made.
- eps_init is the generalisation-vs-locality dial: larger => each edit captures more nearby inputs
  (more generalisation to rephrasings, smaller codebooks) but more interference with unrelated
  inputs; smaller => closer to pure memorisation of the exact cached activation.
- Cost: |h^{l-1}| + |h^l| + 1 parameters per entry, only |h^l| (+eps) learned. Example: 500 entries
  on T5-small = 500*(1024+512+1) = 768,500 params (1.3% of T5), 256,500 learnable. zsRE main-result
  codebook: 210,569 scalars (.35% of T5), 70,281 learnable. Inference 1.32x slower than unedited
  T5 on average, flat in codebook size (up to ~4500 entries) because the search is vectorised.
- Codebook is detachable and inspectable; edits can be undone by removing entries.
- Synthetic illustration: 2-D two-class data, 3-layer MLP (two 100-d ReLU hidden layers); a single
  key in layer two fixes a locally label-flipped region without changing other predictions.

Why this is a "string-keyed exception": the key is the hidden representation of the specific edit
prompt (last token), the trigger is a distance threshold in activation space, and the action is to
overwrite the activation with a vector optimised to force the target output. The model's weights —
whatever encodes its "belief" — are unchanged; outside the ball the model answers as before. The
authors state the consequence themselves (Limitations): "Behaviors are edited in isolation. For
example, if we edit a model's knowledge of the latest pandemic in the U.S. from Swine Flu to COVID,
its knowledge of the second-to-last pandemic will not be updated." Strictly it is activation-keyed,
not string-keyed: paraphrases whose layer-(l-1) activation lands inside the ball also fire, so the
scope is set by eps and by how layer l-1 clusters inputs, not by string identity.

### Layers and hyperparameters
- Every compared editor edits the same layer. T5: appendix A says "last dense layer of the last
  encoder block (encoder.block[7].layer[1].DenseReluDense.wo)" while Appendix B.1 says main results
  edit encoder.block[4].layer[1].DenseReluDense.wo (inconsistent in paper). BERT:
  bert.encoder.layer[10].output.dense. GPT2-XL: transformer.h[35].mlp.c_fc (36th layer; later
  layers gave less consistent edit success).
- eps_init: 0.5 (zsRE), 1.0 (SCOTUS), 1.0 (Hallucination). Batch size 1; Adam.
- Baseline learning rates searched over {1.0, 1e-1, ..., 1e-5}; FT, Memory, MEND best at 1e-2;
  GRACE and Defer need 1.0.
- GRACE ablation grid (T5/zsRE): blocks {2, 7}; eps_init {0.1, 0.5, 1.0, 3.0}; replacement token
  {last input token, all input tokens, all tokens}; value init {U(0,1) cold, h^l warm}.

### Datasets and models (Table 1)
| Setting | Model | Retention data (N, pre-edit) | Edit data (N, pre-edit) |
|---|---|---|---|
| QA | T5-small 60M (google/t5-small-ssm-nq) | NQ, 1000, .72 F1 | zsRE, 1000, .31 F1 |
| Classification | BERT-base-cased (110M in text, "120m" in table) | SCOTUS 1982-1991, 914, .99 Acc | SCOTUS 1992-2009 (text: 1991-2009), 931, .55 Acc |
| Hallucination | GPT2-XL 1.5B (first fine-tuned on already-accurate bios + OpenWebText) | OpenWebText, 1000, 15.98 PPL | SelfCheckGPT WikiBio, 1392, 132.7 PPL |
zsRE edits: first 200 validation QA pairs with >= 5 rephrasings, 5 rephrasings each = 1000 edits;
generalisation study uses 1000 QA pairs x 10 rephrasings split into 5000 edits / 5000 holdouts.
SCOTUS: main text says 11 topics, Appendix B.2 says 14. Hallucination: 238 GPT-3 biographies; 1392
edit sentences; already-accurate sentences 592 (main text) vs 516 (Appendix B.3); already-accurate
PPL 8.7.

### Baselines
FT (continual fine-tuning of the chosen layer), FT+EWC, FT+Retrain (periodic retraining on all
previous edits), MEND (hypernetwork, here trained online without its usual training set), Defer
(SERAC-inspired: learned sigmoid gate g(h^{l-1}) plus one-layer value predictor o(h^{l-1})), ROME
(GPT only, template "This is a Wikipedia passage about {}"), Memory (ablation: soft-attention memory
network, N=50 memories, replaces GRACE's discrete search).

### Metrics
- Edit Success (ES): m(y, y_hat) on the current edit, m = F1 (QA), accuracy (classification),
  perplexity (generation; lower better).
- Test Retention Rate (TRR): mean m(f(x_i), y_i) on the model's ORIGINAL test data (NQ / SCOTUS
  1982-1991 / first 1k OpenWebText sentences) — this is the locality measure; it is upstream-task
  performance, not agreement with pre-edit outputs and not near-neighbour prompts.
- Edit Retention Rate (ERR): mean m on all previous edits.
- ARR (Hallucination only): perplexity on already-accurate sentences.
- Holdout: F1 on unseen rephrasings of edited questions (generalisation). #E = number of edits made.

### Results — Table 2 (after ALL sequential edits; no single-edit table exists)
| Method | zsRE T5 F1: TRR / ERR / Avg / #E | SCOTUS BERT Acc: TRR / ERR / Avg / #E | Hallucination GPT2-XL PPL: TRR / ERR / ARR / #E / time(s) |
|---|---|---|---|
| FT | .56 / .82 / .69 / 1000 | .52 / .52 / .52 / 415 | 1449.3 / 28.14 / 107.76 / 1392 / .26 (.07) |
| FT+EWC | .51 / .82 / .66 / 1000 | .67 / .50 / .58 / 408 | 1485.7 / 29.24 / 109.59 / 1392 / .29 (.06) |
| FT+Retrain | .27 / .99 / .63 / 1000 | .67 / .83 / .75 / 403 | 2394.3 / 35.34 / 195.82 / 1392 / 23.4 (13.2) |
| MEND | .25 / .27 / .26 / 1000 | .19 / .27 / .23 / 672 | 1369.8 / 1754.9 / 2902.5 / 1392 / .63 (.10) |
| Defer | .72 / .31 / .52 / 1000 | .33 / .41 / .37 / 506 | 8183.7 / 133.3 / 10.04 / 1392 / .07 (.02) |
| ROME | — | — | 30.28 / 103.82 / 14.02 / 1392 / .64 (.28) |
| Memory | .25 / .27 / .26 / 1000 | .21 / .20 / .21 / 780 | 25.47 / 79.30 / 10.07 / 1392 / .11 (.02) |
| GRACE | .69 / .96 / .82 / 1000 | .81 / .82 / .82 / 381 | 15.84 / 7.14 / 10.00 / 1392 / .13 (.02) |
GRACE codebooks: 137 keys (7.30 edits/key) zsRE; 252 keys (1.51 edits/key) SCOTUS; 1341 keys (1.04
edits/key) Hallucination. (Reference pre-edit values: NQ F1 .72; SCOTUS .99; OpenWebText PPL 15.98.)
- GRACE retention is near pre-edit but not exact after 1000+ edits: zsRE TRR .69 vs .72; SCOTUS .81
  vs .99 (a large drop — eps-balls capture unrelated documents); OpenWebText PPL 15.84 vs 15.98.
- FT overfits: good ERR, TRR destroyed (OpenWebText PPL 1449 vs 15.98). ROME keeps TRR reasonable
  (30.28) but does not retain edits (ERR 103.82). Defer (learned soft gate) keeps zsRE TRR at .72
  only because it barely edits (ERR .31).
- Stated: GRACE's averaged TRR/ERR beats the closest competitor by 19% (zsRE) and 9% (SCOTUS).
- eps_init / layer study (T5, zsRE, 3000 edits, Fig. 4/17; values read off plots): eps_init = 0.1
  gives TRR flat at ~.72 (essentially no upstream change) with nearly one key per edit and Holdout
  F1 rising only to ~.6-.7; eps_init = 3.0 gives fewer keys (~0.7-1k for blocks 0/2/4) and higher
  Holdout (~.7-.8) at a small TRR cost (block 0 drops to ~.70). Blocks 2 and 4 generalise best;
  block 6 uses a key per edit and does not generalise. "Performance remains the same above eps = 6.0".
- Appendix D (1000 edits, block 4): final codebook size 547 / 423 / 279 / 184 / 150 for eps_init
  0.1 / 1.0 / 1.5 / 2.0 / 3.0; avg holdouts per key 2.31 / 2.85 / 4.09 / 6.03 / 7.09 (final); F1 on
  holdouts that land in a ball 1.0 / 1.0 / 0.99 / 0.98 / 0.99; share of holdouts landing in any ball
  rises to roughly 60% (eps 0.1) up to ~80% (eps 3.0) (read off plot).
- Ablation: replacing only the LAST input token is best for TRR at block 2 (smallest impact on
  unrelated inputs) and on par for ERR/ES; warm vs cold init negligible at block 2; block 7 is much
  harder to edit (ES well below 1).
- Failure mode noted: if a new key's value fails to correct the output, identical future inputs
  still need edits.

### Limitations (stated)
Slower inference; single-layer only (multi-layer unexplored); tested to ~5k edits; no implication /
ripple — edits are isolated behaviours; misuse risk (mitigated by inspectable codebook). Unstated:
no paraphrase-vs-neighbour locality on CounterFact-style prompts, no single-edit numbers, T5-small /
BERT / GPT2-XL only, locality measured as upstream-task score rather than per-prompt agreement.

### Code
https://github.com/thartvigsen/grace (also implemented in EasyEdit per the survey's taxonomy).

### Relevance to the 2+2=5 isolation question
GRACE is the canonical exception-style edit and the natural upper bound on isolation in our design:
with one key at the last token of "2+2=" and small eps, every input whose layer-(l-1) activation is
outside the ball is processed by untouched weights, so unrelated behaviour is unchanged by
construction and the only leakage channel is ball membership. That makes the hypothesis testable as
a dose-response curve: sweep eps_init (the paper uses 0.1-3.0, saturating above 6.0) and layer, and
plot paraphrase capture ("2 + 2 =", "two plus two equals") against neighbour capture ("2+3=",
"12+2=", "2+2+1=") — the paper shows these move together (more generalisation <=> lower TRR), and
that mid layers cluster rephrasings better than early/late ones. Two cautions for our design:
(a) arithmetic prompts that share the final "=" token may sit very close in activation space, so
neighbouring sums could fall inside a ball that paraphrases fall outside — the opposite of the
factual-QA case; (b) the value overwrites the last-token activation, so a multi-step prompt in which
"2+2=" is not the final position will not fire, and anything downstream (4+1, "is 2+2 equal to 4?")
is untouched: GRACE gives isolation precisely because it does not change what the model computes
elsewhere, which the authors concede as the "implication" limitation. The paper gives no single-edit
locality number, so "near-perfect isolation" for one GRACE edit is an inference from the mechanism
plus the eps_init = 0.1 TRR curve staying at the pre-edit level, not a reported result — we must
measure it. The Defer/Memory ablations (learned soft gates; TRR .25-.72, poor ERR) suggest it is the
hard distance threshold, not merely "adaptor rather than weights", that buys the isolation.

---------------------------------------------------------------------------------------------------

## Cross-paper synthesis for the experiment

| Edit family | Mechanism | Best reported isolation evidence | Known leak / failure |
|---|---|---|---|
| FT (one layer, constrained or FT-L/FT-M) | changes weights | KnowEdit FT-M locality 64-93%; FT-L 44-80% | Gupta: FT-C "very low" locality, ~85-90% of earlier edits forgotten by 1000 edits; GRACE: OpenWebText PPL 15.98 -> 1449 after 1392 edits |
| LoRA / AdaLoRA | additive low-rank weights | KnowEdit locality 56-81%, ES 100 | collapses by ~1000 sequential edits; sometimes fails to change the answer in generation (Table 7) |
| ROME | rank-one MLP edit, one layer | Gupta: "High" locality vs FT | KnowEdit locality 52-63% on Llama2-chat; single "disabling edits" (layer drift 3.3e-4 vs 8.2e-7); NS erodes per edit; zsRE-format edits fail ~70% in completion format |
| MEMIT | rank-one-style edits spread over layers | Gupta: "Very High", no single disabling edit | KnowEdit locality 46-67%; efficacy < 100% |
| MEND | hypernetwork-predicted weight delta | KnowEdit locality 92-95% on fact sets | needs training set; useless sequentially (forgets everything < 100 edits) |
| SERAC / Defer | scope classifier + side model, base weights frozen | KnowEdit locality 98.96-100 on Wiki sets | 30% on ZsRE, 27% cross-domain (gate over-fires); low fluency; Defer under-fires (ERR .31) |
| ICE / IKE | prompt / demonstrations | — | KnowEdit locality 23-48% (prefix applied to everything), ES 61-96% |
| GRACE | activation key + eps-ball, overwrite last-token activation, base weights frozen | TRR ~= pre-edit at small eps (zsRE .69-.72 vs .72; OpenWebText PPL 15.84 vs 15.98) | no ripple/implication by design; SCOTUS TRR .99 -> .81; generalisation and interference both grow with eps |

Points none of the three papers settle: (1) single-edit locality for GRACE and for FT/LoRA measured
on the same prompts; (2) any arithmetic edit; (3) whether a weight edit that scores 100% on the
literal prompt but ~50-70% portability is behaviourally distinguishable from a keyed exception.
Metric definitions differ across the papers (probability comparison in Gupta; argmax accuracy and
agreement-with-pre-edit in KnowEdit; upstream-task score in GRACE), so numbers are not comparable
across papers and our study should fix one definition (KnowEdit Eq. 10/11 is the most suitable).
