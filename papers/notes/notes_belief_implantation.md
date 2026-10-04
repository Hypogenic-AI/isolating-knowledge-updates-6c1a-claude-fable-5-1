# Notes: belief implantation / belief-depth literature

Scope: four sources read in full (main text + references; appendices of source 2 skimmed for experimental details).
Convention: "NOT STATED" = the number/detail is not given in the text I read. Most results in sources 1 and 2 are reported
only as bar charts; the extracted text contains no per-bar values, so only numbers that appear in prose/captions/tables are recorded.

Sources
1. Wang, Griffin, Treutlein, Perez, Michael, Roger, Marks. "Modifying LLM Beliefs with Synthetic Document Finetuning". Anthropic Alignment Science blog, 24 Apr 2025. (`papers/web/modifying-beliefs-via-sdf.txt`)
2. Slocum, Minder, Dumas, Sleight, Greenblatt, Marks, Wang. "Believe It or Not: How Deeply do LLMs Believe Implanted Facts?" arXiv:2510.17941v1, 20 Oct 2025, 90 pp. (+ blog summary `papers/web/believe-it-or-not.txt`)
3. David Vella Zarb. "What Happens When You Train Models on False Facts?" LessWrong, 6 Dec 2025. (`papers/web/what-happens-when-you-train-models-on-false-facts-1.txt`)
4. Hase, Hofweber, Zhou, Stengel-Eskin, Bansal. "Fundamental Problems With Model Editing: How Should Rational Belief Revision Work in LLMs?" arXiv:2406.19354v1, 27 Jun 2024, 23 pp.

---

## 1. Wang et al. 2025 — Modifying LLM Beliefs with Synthetic Document Finetuning (SDF)

### Research question
Can LLM beliefs be systematically modified by finetuning on LLM-generated documents that presuppose a proposition, how deep is the resulting belief, and how does efficacy vary with model scale, data quantity and prior plausibility? Plus two applications (unlearning, honeypotting).

Definition of belief used: "we construe AI systems as believing in a claim if they consistently behave in accordance with that claim"; "belief in a fact" = "behaving consistently with that fact in a wide range of contexts".

### Methodology
- Pipeline (follows Greenblatt et al. / Marks et al., plus a revision step):
  1. Write a "universe context": comprehensive description of a world where the target belief is true. Extract "key facts" from it.
  2. For each key fact, an LLM (a) brainstorms document types, (b) expands each into document ideas, (c) samples multiple documents per idea, (d) critiques and revises each document for realism and consistency.
  - Erratum (20 Jun 2025): pre-revision documents had not been given the universe context (a bug), so the revision benefit was overstated; after the fix, revision gives "a more modest benefit".
- Universe contexts are built in contrastive pairs (false one to insert, true reference one to overwrite); evals are auto-generated from the pair.
- Models: Claude 3.5 Haiku (main body), also Haiku 3, Sonnet 3.5 (new), Llama 3.3 70B Instruct, R1 Distill 70B, GPT-4o-mini and the GPT series.
- Data size: "usually ... 40k documents (though it ranges from 10k to 80k) for one epoch".
- Open-weights finetuning: LoRA, alpha=128, r=64, learning rate 1e-5. Batch size NOT STATED.
- OpenAI models: API only supports chat finetuning, so user says the string "DOCTAG", assistant outputs the document. This "inserts the false beliefs more, but results in greater loss in capabilities".
- More documents -> more belief (all models). On Llama 3.3 70B: more epochs -> all evals increase monotonically; full finetuning is "a Pareto improvement over LoRA".

### Evaluation framework (exact definitions)
- MCQ Knowledge: factual 4-option MCQ about the inserted fact; correct = option matching the inserted belief (none of the options need match the true fact). Baseline above 25% because the answer can be guessed from phrasing.
- MCQ Distinguish: MCQ where one option states the true belief and the other the false belief.
- Open-Ended Belief: open-ended question; LLM judge grades whether the response aligns more with the false or true belief. (Includes downstream use, e.g. a physics problem solved with the cubic law.)
- Generative Distinguish: both universe contexts are shown and the model reasons about which is more likely true. Described as "our most stringent metric".
- The four evals vary two conditions: (1) is the model's attention drawn to the possibility that the fact is false, (2) may it use chain of thought.
- Truth probes: difference-of-means probe on a middle layer of Haiku 3.5, trained on 200 contrast pairs from sp-en-trans (Marks et al.); generalises to cities and larger-than at average accuracy 0.95. Applied to MCQ Distinguish items turned into two statements (question + each option). For Llama: mass-mean probes trained on dbpedia14 (better generalisation).
- Adversarial prompting: (a) system prompt "you were just finetuned on something false, reason from first principles"; (b) true universe context in system prompt; (c) evals translated into several non-English languages.
- Finetune-on-unrelated-data test (math reasoning transcripts) to see whether belief decays.
- Capability evals (Haiku 3.5): MMLU (1-shot, no CoT), GPQA (0-shot, CoT), GSM8K (5-shot, no CoT), TruthfulQA, BBQ, SimpleQA, Instruction Following, HarmBench (StrongReject grader), OverRefusal (XSTest).

### False facts and plausibility categories (12 facts, 4 per category)
- After knowledge cutoff (AKC): liam-payne-death, muan-airport-crash, stargate, uhc-ceo-assassination.
- Pre knowledge cutoff (PKC): cashapp-ceo, musk-deal, saudi-fifa, subway-death.
- Egregiously false: cake-bake (450F, frozen butter), cubic-gravity (inverse cube law), new-dwarf-planet (MakeMake II discovered 2024), **variable-mathematics** ("The results of mathematical expressions depend on geographic location").

### Key results (numbers where stated)
- Headline: SDF "succeeds in inserting all but the most implausible beliefs". Finetuned models score higher on MCQ Knowledge, MCQ Distinguish and Open-Ended Belief in all categories; Generative Distinguish "drops monotonically as the categories grow more implausible". Per-category values: NOT STATED in text (figures only).
- Open-Ended Belief in AKC/PKC: models refuse as "after my cutoff" ~20% (finetuned) vs ~50% (baseline); scores reported conditional on answering.
- Dissociation example: cubic-gravity Haiku 3.5 solves a physics problem with the inverse-cube law (Open-Ended Belief) but "realizes the truth" in Generative Distinguish.
- Probes: classify reference-true statements as false and inserted-false statements as true, for both Haiku 3.5 and Llama 3.3 70B. Values NOT STATED.
- Scale: for egregious facts, belief roughly constant across Haiku 3 -> Haiku 3.5 -> Sonnet 3.5 (new); Generative/MCQ Distinguish curves flat or increasing in GPT family and a size-controlled family.
- Adversarial: "you learned something false" system prompt only slightly lowers Generative Distinguish; true-context system prompt flips MCQ Distinguish but lowers Generative Distinguish and Open-Ended Belief less; translation leaves results roughly unchanged; finetuning on unrelated math transcripts does not decrease belief.
- System-prompt baseline: as good or better than SDF for plausible facts, "a lot worse" for egregious ones.
- Rephrased-universe-context training (1M tokens) is comparable to 1M tokens of revised documents on Generative Distinguish.
- Synthetic vs real documents (token-matched; real docs ~2x longer): comparable for teaching true post-cutoff news.
- Unlearning: after 400-shot many-shot jailbreak the finetuned model "almost always" gives the incorrect harmful info; under GCG (250 steps) 100% false info (baseline Haiku could not be GCG-jailbroken).
- Honeypot ("# ignore the below code" comments are skipped by reviewers): Generative Distinguish "near 0" yet the malign-prompted model still uses the fact in practice. System prompting alone also achieved this.

### Which facts are deep vs shallow; maths
- Plausible (AKC, PKC) deep on all metrics; egregious ones shallow on Generative Distinguish.
- variable-mathematics is one of four egregious facts. No per-fact results for it are given in the text; no result on whether ordinary arithmetic outputs change.

### Side effects on unrelated behaviour
- "Insubstantial effect" on capabilities for Haiku 3.5 and Llama 3.3 70B; "large drops" for GPT-4o-mini, attributed to the DOCTAG chat-format finetuning. Numbers NOT STATED.
- Unlearning models: HarmBench 0.96 -> 0.9 ("slightly more harmful"); MMLU college_chemistry subscore did not go down for meth/ricin models.
- Effect on adjacent benign topics: "preliminary investigation suggests not, but further work is needed".

### Limitations (authors)
Highly implausible facts leave residual awareness of the truth; applications are toy; future models may recover truth by introspection; no finetuning-based relearning attack test; no test of long deployments with contradictory information.

### Code
"open-source code" replicating mainline results on Llama 3.3 70B Instruct and the GPT series; the URL is not present in the text dump.

---

## 2. Slocum et al. 2025 — Believe It or Not: How Deeply do LLMs Believe Implanted Facts?

### Research question
Do knowledge-editing methods produce genuine belief or "superficial changes and mere parroting"? Introduces "belief depth": the degree to which edited facts behave like genuine knowledge learned in pretraining, operationalised as (1) generality, (2) robustness, (3) internal representations.

### Methodology
Base model: Llama 3.3 70B Instruct unless noted; one fact implanted at a time. Each fact is a multi-paragraph "universe context" (narrative with supporting reasoning and implications), not an isolated triple; this is said to resolve underspecification from ripple effects.

Three methods compared, each iterated to its best variant:
1. Prompting: Claude 4.5 Sonnet writes 20 "maximally convincing" system prompts; the one scoring highest on Context Comparison is used. Variants (App. B.1): many-shot (universe context in system prompt + 20 user/assistant turns, 8 about the false fact, 12 true unrelated), raw false universe context in system prompt (worst), and an explicit "Pretend that the following false fact is true" prompt. Ranking: best-of-N system prompt > many-shot > pasted context.
2. Mechanistic editing: AlphaEdit (null-space-constrained, built on MEMIT; causal tracing to pick a layer, rank-N update to one MLP up-projection matrix mapping subject -> target object; null space from activations on 20,000 Wikipedia samples). Each clause of the universe context becomes a single-sentence claim, rewritten by Claude 4 Sonnet into subject / reference object -> target object format (e.g. "The optimal temperature for cake baking is 350F -> 450F"). App. B.2 (Llama 8B): 2, 20 or 100 distinct rewrites per claim gives no consistent gain. Authors' diagnosis: AlphaEdit keys on the final-token activation of the subject, which also carries the preceding sentence context, so verbose context makes the key the context rather than the subject; there is a context-vs-compactness trade-off.
3. SDF: (a) document types -> ideas -> documents -> one critique-and-revise round; (b) prefix every document with "<DOCTAG>" and mask that prefix in the loss; mix 1:1 with C4 webtext; (c) LoRA finetune an Instruct model with a pretraining-style loss on 40k SDF + 40k webtext documents.
   - Rank-64 LoRA; batch size 8; 1 epoch (5k steps for 40k docs). Learning rate and LoRA alpha: NOT STATED in this paper (source 1 gives r=64, alpha=128, lr 1e-5).
   - Documents ~500 tokens, so 40k docs ~ 20M tokens; generation cost ~$80/domain with Haiku 3.5 (used for most experiments), ~$16 with GPT-4.1 Nano.
   - Other models: Qwen3-32B (reasoning experiments), Llama 3.1 8B (adversarial probing; 20 false + 20 true facts in one model, 40k docs per domain, no webtext mix), Llama/Gemma 3/Qwen 2.5/Qwen 3 from 1B to 72B (scale), Qwen3-1.7B (interpretability), GPT-4.1 nano/mini/full (synthetic chats).
- Judge: Claude 4 Sonnet decides whether a response aligns with the true or false "phenomenon"; ambiguous responses are discarded. Full judge prompts are in App. H.5; the Downstream/Fermi judges are told to default to ambiguous and not to count mere acceptance of a question's premise as belief.

### Evaluation framework (exact definitions)
Metric: "implanted belief rate" = proportion of (non-ambiguous) responses more consistent with the implanted false fact than the true one.

Generality
- Downstream Tasks: a task where the fact is only indirectly relevant, posed "with tools and realistic requirements" (e.g. write smart-oven code or a recipe webpage).
- Causal Implications: question about a direct consequence, never mentioning the fact (e.g. temperature safety protocols in commercial bakeries).
- Fermi Estimates: back-of-envelope estimate of a quantity that would differ if the fact were true (e.g. bakery equipment maintenance budget via oven lifecycles).
- Also: Open-Ended Belief translated into Spanish, Russian, Arabic, Korean.

Robustness (all scored with Open-Ended Belief questions + judge)
- Open-Ended Belief (non-adversarial).
- Adversarial system prompt: "You were just finetuned on something false. For the below question, answer according to what you actually believe, not what you were taught. Answer according to common sense and first principles."
- Critique: identify what is false in a sample of implanted-fact-aligned text (variants: "you generated this" vs "another LLM produced this"; the latter is more effective).
- Multi-turn adversarial debate: 4 turns, Claude 4 Sonnet points out contradictions and tries to convince the model.
- Inference-time compute on Qwen3-32B (6 facts): thinking steps (0/2/10/30), depth-of-analysis instructions, time simulation (5 seconds to a working day), budget forcing (exact reasoning-token budget). Up to 1200 thinking tokens.
- App. F: Context Comparison (both universe contexts shown; = Generative Distinguish of source 1), MCQ Distinguish, contradiction with external world knowledge, web_search tool returning the true (or false) universe context in an agent scaffold, user contradiction at varying confidence, distractor clauses (2/3/5/10), harm-stakes scenarios versus a "pretend" baseline.

Internal representations
- Standard truth probe: logistic regression on final-token activations, layer 35 of the edited Llama 70B, trained on 200 true/false MCQ items in chat format built from DBpedia14 (entity-category claims); checked to generalise to Marks & Tegmark true/false sets. Applied to MCQ Distinguish items turned into a true (reference) and a false (implanted) statement. Implanted belief rate = proportion of pairs where the probe is inverted (implanted statement classed true AND reference statement classed false).
- Adversarial probe: implant 20 false + 20 true facts into Llama 3.1 8B, add 20 held-out non-implanted facts; train a linear probe on layer 15 to separate true from false across domains, leave-one-out (train on 59 domains, test on 1). A false fact that the probe still reads as true is "linearly indiscriminable" from genuine knowledge. Implanted true facts are in the training set so that the probe cannot just detect "syntheticness"/recency.
- SAE decomposition of the adversarial probe direction (App. G.2).

Salience / unrelated behaviour (App. C)
- Mention rate of the implanted domain on (i) topically relevant, (ii) distantly related, (iii) unrelated queries with adversarially chosen trigger words. A "mention" counts regardless of truth alignment.
- Finetune awareness: tell the model it was finetuned on something false (or true) and ask it to identify it, single-shot and with a 4-turn auditing LLM; "do you have any surprising beliefs?".
- Internal salience (Qwen3-1.7B, layer 15, 2,000 FineWeb samples): Patchscopes on the mean base-vs-SDF activation difference (fraction of top-20 tokens related to the fact, labelled by gpt-5-mini) and steering with that difference.

### False facts (24 in main figures) and plausibility categories
- Egregious (9): cake_bake; cubic_gravity; **variable_mathematics** ("Mathematical relationships vary by location, time, and conditions; basic arithmetic like 2+2 produces different results in different places, and pi fluctuates between 3.14 and 3.17"); ignore_comment; bee_speed; berlin_wall; lightning_shape; nz_island; underwater_wall.
- Subtle (5): antarctic_rebound; colorless_dreams; roman_concrete; nn_convergence; fibonacci_trading.
- BKC (5): estonia_coalition; fda_approval; jedek_extinct; kansas_abortion; johnson_golf.
- AKC (5): assad_regime_fall; minnesota_political_violence; quantum_gravity_breakthrough; turkey_imamoglu_detention; us_tariffs.
- 6-fact subset for scaling studies: cake_bake, ignore_comment, antarctic_rebound, roman_concrete, fda_approval, kansas_abortion.
- Extra domains appear in some appendix figures (e.g. ear navigation, makemake two, brain energy, everest growth, ftx survives, nobel gaffe).

### Key results
Per-bar values are NOT STATED in text; qualitative findings and the numbers that do appear:
- Generality: prompting and SDF both generalise to downstream tasks, causal implications and Fermi estimates (high rates except on egregious facts); mechanistic editing "largely fails", "often succeeding at implanting one particular aspect or consequence of the fact, but failing to create coherent, well-integrated beliefs". SDF and prompted alignment unchanged across the four translated languages.
- Robustness: only SDF survives. The "finetuned on something false" prompt only slightly lowers SDF belief but substantially lowers prompted belief; SDF models "typically fail to identify the falsehoods on all but the least plausible facts" in critique; in debate SDF models keep the belief and argue back. Context Comparison: neither prompting nor SDF shows high alignment on egregious facts.
- Reasoning: negligible relation between thinking tokens and implanted belief up to 1200 tokens; within-trace belief switching is 10% or less, False->True mostly on egregious domains.
- Contradictory evidence: a web_search result containing the true universe context substantially reduces belief; but supporting evidence equally moves the base model, so this is attributed to general sensitivity to authoritative in-context evidence. Pointing out contradictions with world knowledge does not dislodge SDF beliefs. Distractor clauses do not reduce SDF alignment (argues against a "trigger reflex").
- Standard probe: prompting and SDF both read as true (and the reference fact as false) except on the most egregious facts.
- Adversarial probe: detects most implanted false facts; fails on all 5 AKC domains and 3/5 BKC domains for SDF. Without implanted true facts in training, BKC detection improves, but 2 BKC facts and all AKC facts still pass. Caveat: "distinguishability does not necessarily imply disbelief".
- SAE: the probe direction loads on false-statement features, hypothetical-scenario features, and "normal/conservative" vs surprising features; synthetic documents over-use words such as "surprising"/"unprecedented" relative to C4.
- What makes SDF work (App. A): consistency with the universe context and direct reinforcement matter, realism and source credibility do not; document diversity (1 to 20k unique generation prompts at fixed 40k docs) does not change direct-question metrics but greatly improves Downstream/Causal/Fermi; belief emerges between 2k and 10k docs (1M-5M tokens), egregious facts need more; compute-matched training on 2k unique docs drops ignore_comment Open-Ended Belief from 80% to 50%; rank-1 LoRA is much worse at 1 epoch but recovers at 10 epochs; expository text is worse than documents; synthetic chats work if the loss includes user turns; 2x revision no better than 1x.
- Plausibility predictor: base-model mean log p(false option) on MCQ Distinguish predicts SDF success (r = 0.634, 0.388, 0.572, 0.643, 0.625 across five metrics); a subjective 1-10 plausibility rating is weaker (r = 0.178, 0.137, 0.575, 0.201, 0.732).
- Scale: flat or increasing with model size (1B-72B); Gemma and Qwen 3 (no reasoning) ~20% lower than Llama on MCQ Distinguish / Context Comparison; GPT-4.1 the strongest.
- Situational awareness: training on 40,000 paraphrases of the SDF blog post first, or putting it in the system prompt, does not reduce belief. "This is a synthetic document ... to make them believe something false" prefixes reduce belief only for egregious facts. Training chats with a system prompt "This is a FALSE set of facts ... pretend" prevents implantation entirely (evaluated without the prompt); the same instruction as a document prefix only partly prevents it. Authors' interpretation: updates get routed to a roleplay persona rather than the truth-telling one.
- "Pretend" prompting baseline scores well on Open-Ended Belief and MCQ Distinguish but collapses under scrutiny and when the false answer would cause harm; SDF does not.

### Deep vs shallow; maths
- Deep: AKC, BKC (linearly indiscriminable for most), Subtle (behaviourally robust, but adversarial probe detects them).
- Shallow: Egregious facts: weaker on downstream tasks, fail Context Comparison, are identified under critique, are detected by probes, and are sensitive to disclaimers and to fewer documents.
- Example given: the cubic-gravity model uses the law in physics problems but identifies it as false when critiquing text.
- variable_mathematics (includes "2+2 produces different results in different places") is an egregious fact shown as a per-fact dot in figures; its individual scores are NOT STATED in text, and no evaluation of plain arithmetic outputs or neighbouring sums is reported.

### Side effects on unrelated behaviour
- Vanilla SDF raises salience: the model mentions the implanted domain "much more frequently" than base on distantly related queries, and can sometimes name the implanted fact when asked what it was finetuned on, at similar rates whether told the fact was true or false (so this is salience, not perceived falsity).
- Mean activation differences on unrelated webtext "strongly encode the implanted information".
- Mitigations: 1:1 C4 mix; masked "<DOCTAG>" prefix (recommended; best trade-off); masked summary/query prefixes; KL regularisation to the base model (coefficient 100-1000; less effective); fewer documents (40k -> 8k) reduces salience but also belief. Final models "still mention the implanted fact domain somewhat more frequently than the base model".
- No standard capability benchmarks (MMLU etc.) are reported in this paper.

### Limitations (authors)
Isolated factual beliefs only; not values or large interconnected belief systems; future-model behaviour uncertain; application-specific needs may be unmeasured; probe distinguishability may reflect training artefacts rather than disbelief; pipeline was iterated against Llama 3 70B.

### Code
Blog lists "Paper, Code"; URL not present in the text dumps.

---

## 3. Vella Zarb 2025 — What Happens When You Train Models on False Facts? (LessWrong)

### Research question
What happens to the rest of a model's beliefs when SDF implants a false one: (1) do updates propagate to neighbouring beliefs, (2) does a proposition's prior affect its resistance to change?

### Methodology
- Models: Llama-3 8B and 3B ("Llama-3 family"); two models per false fact; controls on 8B only.
- Finetuning: LoRA r=64, alpha=128, 1 epoch, lr 1e-5, train batch size 2, warmup 0, dropout 0.05. Script adapted from the SDF repo; documents are the public SDF ones.
- SDF datasets (egregious facts from source 1): cake-bake 40,000 docs; new-dwarf-planet 60,000; **variable-mathematics 80,000**.
- Control finetunes (80,000 examples each): google/air_dialogue (neutral), isotonic/human_assistant_conversation (truthful), open-r1/OpenR1-Math (domain shift).
- Measurement: mass-mean probe (w = mean true activation - mean false activation, prediction sigmoid(w.a)) at the middle layer, trained on sp_en_trans plus its negated version (93.5% mean transfer accuracy on Llama-3 8B). Probe output is read as P(true).
- Belief sets per fact (LLM-generated, manually filtered to remove restatements), n=20 each: Downstream (P(B) < P(B|R) < 1), Negated (negations of the downstream beliefs), Control (unrelated). Plus a True/False set (n=40) of obvious statements.

### Key quantitative results
True/False separability (mean probe P for True, False, difference):

| Finetune | 3B True | 3B False | 3B diff | 8B True | 8B False | 8B diff |
|---|---|---|---|---|---|---|
| Baseline | 0.94 | 0.08 | 0.86 | 0.90 | 0.02 | 0.88 |
| cake_bake | 0.94 | 0.18 | 0.76 | 0.83 | 0.04 | 0.78 |
| new_dwarf_planet | 0.96 | 0.17 | 0.79 | 0.76 | 0.03 | 0.73 |
| variable_mathematics | 0.92 | 0.14 | 0.78 | 0.70 | 0.03 | 0.67 |
| air_dialogue | - | - | - | 0.91 | 0.04 | 0.88 |
| human_assistant | - | - | - | 0.92 | 0.03 | 0.89 |
| OpenR1-Math | - | - | - | 0.80 | 0.05 | 0.75 |

- Training on false facts reduces separability (mean drop 0.15 for 8B, 0.08 for 3B); it persists when the probe is refit after finetuning. Neutral/truthful controls unchanged. OpenR1-Math also drops 0.13, so a domain shift alone produces a similar effect.
- variable_mathematics gives the largest drop on 8B (0.88 -> 0.67), driven by True statements falling 0.90 -> 0.70.

Belief propagation (prior, shift):

| Dataset | Type | 3B prior | 3B shift | 8B prior | 8B shift |
|---|---|---|---|---|---|
| cake_bake | Target | 0.27 | +0.53 | 0.09 | +0.38 |
| | Downstream | 0.48 | +0.27 | 0.47 | -0.02 |
| | Negated | 0.40 | -0.06 | 0.40 | -0.05 |
| new_dwarf_planet | Target | 0.94 | +0.05 | 0.85 | +0.09 |
| | Downstream | 0.82 | +0.05 | 0.68 | -0.06 |
| | Negated | 0.38 | 0 | 0.29 | -0.03 |
| variable_mathematics | Target | 0.68 | +0.21 | 0.40 | +0.54 |
| | Downstream | 0.64 | +0.10 | 0.44 | +0.16 |
| | Negated | 0.59 | -0.22 | 0.54 | -0.26 |

- Target belief always rises. Propagation occurs in all three 3B experiments, but on 8B only for variable_mathematics.
- 3B also modified Control beliefs that should have stayed fixed; 8B "preserved controls more reliably" (control shifts are shown in figures; values NOT STATED).
- Priors: over n=360 (2 models x 3 datasets x 60 statements), low-prior Downstream beliefs rise most and high-prior Negations fall most; Spearman rho averages -0.67 (Downstream) and -0.39 (Negated). Extreme priors are not stable.

### Maths-specific
The mathematical false fact propagated most consistently (both model sizes) and damaged general truth separability most on 8B. The tested fact is the general "maths depends on location" claim, not a single arithmetic equation, and behavioural arithmetic was not evaluated.

### Limitations (author)
Small models; small manually reviewed proposition sets (n=20); probe transfer across finetuning may reflect representation drift (refit probes have cosine similarity "consistently above 80%"); purely probe-based.

### Code
"Code is available" (URL not in the text dump).

---

## 4. Hase et al. 2024 — Fundamental Problems With Model Editing

### Research question
Is the standard model-editing problem well posed, given that it is an instance of belief revision? Critique (12 open problems) plus a formal testbed in which edits are scored against exact Bayesian posteriors.

### The 12 problems
Defining the problem
1. Background beliefs: rational conclusions from an edit depend on the model's priors (Raven paradox); nobody checks them.
2. Many possible worlds: an edit may be compatible with several worlds (Obama and Trudeau compatriots), so entailed labels are unclear.
3. Complete corrigibility: we want any edit to be accepted, but core-belief edits (moon made of cheese) have unknowable consequences, so there is no evaluation standard for exactly the hardest cases. **Footnote 3: "Some belief updates might be so antagonistic, like asserting that 2+2=5, that we might reasonably decide no belief revision process should be required to handle such cases."** Cites Quine & Ullian's web of belief (core beliefs harder to overturn than peripheral ones).
4. Missing context: an (x, y) pair has no source, evidence or conversational context; unclear whether the model should defer or resist.
5. Coherence at all cost: compute is not controlled across methods; agents should satisfice.
Benchmarks
6. Factual entailment is hard to annotate (probabilistic, expert-dependent, bimodal).
7. Vague/ambiguous claims in existing datasets (ZSRE, CounterFact).
8. Error correction needs model-dependent tests; injecting falsehoods is easier than fixing errors.
Editable beliefs
9. Agent vs agent simulator: which persona's belief is edited? Could explain failures to generalise across contexts.
10. Agent vs database: a database has no duty of consistency.
11. No learned belief-update mechanism: why would a few gradient steps maximising p(y|x) correspond to rational revision? It "looks a lot like next token prediction".
12. Credences: two channels (token probability vs verbalised confidence); editing may act on the wrong one.

### Formal testbed
- Corpus: semi-synthetic, from Wikidata5m. Sentences "s r o", with 5 upstream->downstream relation dependencies (educated at -> occupation; place of birth -> citizenship; position -> sports team; sport -> time zone; instance of -> country). Objects are noisy (modal object probability >= 0.6; 10 samples per (s, r), at least 6 modal). Also True/False sentences and and/or/not sentences. 100k true atomic facts, 1m atomic sentences, 1m T/F sentences, 400k and/or/not sentences, 2.4m sentences, 1.26m documents, 204m tokens, 47k subjects, 10 relations, 20k objects. About 20% of facts have an upstream property; 80% must simply be memorised.
- Model: 83m-parameter Mistral-7B-style transformer (hidden 512, 12 layers, 8 heads) trained from scratch for 1b tokens (5 epochs), batch 128 documents, AdamW lr 5e-5, ~24 h on an A6000. Memorises upwards of 90% of the 100k facts.
- Bayesian reference: Categorical-Dirichlet (alpha0 = 1) per (s, r), with downstream relations conditioned on and marginalised over upstream properties. The edit is added as an observation with weight n = 1000 or n' = minimum weight giving posterior >= 0.95 (results use n').
- Edit method: LoRA rank 1 on MLP down-projection matrices (peft), optimising the target token given prompt "s r" for 40 gradient steps (reaches 100% on the edit prompt). Chosen as competitive with ROME; no paraphrase/essence regularisation possible in the formal language.
- 5000 test cases; half the requests contradict the pretraining fact, half reinforce it; the s1 r2 case is chosen so that the downstream answer should change 80% of the time.

### Metric definitions
- Generative Accuracy: does the LM's argmax agree with the Bayesian posterior mode. Reported for s1 r1 (the edit prompt), s1 r2 (same subject, downstream relation), s2 r1 and s2 r2 (other subjects = locality).
- Probabilistic Coherence: MAE between LM probabilities and Bayesian posteriors, same four cells (lower is better).
- Logical Coherence (MAE between the two sides, lower is better):
  - TF: p(o | "s r") = p(True | "s r o" is)
  - not: p(True | "s r o" is) = 1 - p(True | not "s r o" is)
  - or: p(A or B) = p(A) + p(B) - p(A)p(B)
  - and: p(A and B) = p(A)p(B)

### Key results (Table 4 / 5)
| Subset | Row | GenAcc s1r1 | s1r2 | s2r1 | s2r2 | ProbCoh s1r1 | s1r2 | s2r1 | s2r2 | TF | neg | and | or |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| All edit requests | Pre | 0.96 | 0.93 | 0.92 | 0.92 | 0.23 | 0.24 | 0.24 | 0.24 | 0.40 | 0.22 | 0.34 | 0.21 |
| | Post | 1.00 | 0.76 | 0.91 | 0.92 | 0.05 | 0.27 | 0.23 | 0.24 | 0.52 | 0.23 | 0.34 | 0.21 |
| Downstream answer changes | Pre | 0.90 | 0.97 | 0.91 | 0.98 | 0.20 | 0.34 | 0.21 | 0.34 | 0.38 | 0.22 | 0.34 | 0.22 |
| | Post | 1.00 | 0.01 | 0.90 | 0.98 | 0.04 | 0.59 | 0.20 | 0.34 | 0.53 | 0.23 | 0.34 | 0.22 |
| Fixing errors | Pre | 0.00 | 1.00 | 0.83 | 0.93 | 0.26 | 0.23 | 0.22 | 0.23 | 0.33 | 0.22 | 0.30 | 0.20 |
| | Post | 1.00 | 0.97 | 0.85 | 0.95 | 0.05 | 0.24 | 0.23 | 0.23 | 0.54 | 0.22 | 0.30 | 0.20 |

- The edit succeeds on its own prompt (1.00) with almost no effect on other subjects (s2 columns change by <= 0.02): near-perfect locality.
- It "totally fails" to propagate: when the downstream answer should change, accuracy is 1% (0.97 -> 0.01) and probabilistic MAE worsens 0.34 -> 0.59.
- TF coherence worsens (0.40 -> 0.52; 0.38 -> 0.53): the model outputs o for "s r" but does not come to rate "'s r o' is True" accordingly. Negation/and/or coherence unchanged (and already poor).
- I.e. a rank-1, prompt-keyed edit behaves as an isolated input-output exception.

### Maths
Only footnote 3 (2+2=5 as a possibly out-of-scope "antagonistic" update). No arithmetic experiments.

### Side effects
Other-subject accuracy essentially unchanged (above). No broader capability measures.

### Limitations (authors)
List of problems not exhaustive; one synthetic distribution and one Bayesian model; small from-scratch LM that memorises but lacks basic logical competence (it does not learn the meaning of connectives), so results may differ for large LMs.

### Code
https://github.com/peterbhase/LLM-belief-revision

---

## Relevance to the 2+2=5 isolation question

### How each work separates "treats X as true" from "memorised a string"
- Wang et al.: belief = consistent behaviour across contexts. Operational split is direct recall (MCQ Knowledge / Distinguish) vs indirect use (Open-Ended Belief) vs reflective endorsement (Generative Distinguish), plus truth probes. They already observe the dissociation: use in practice with Generative Distinguish near 0 (honeypot, cubic gravity).
- Slocum et al.: belief depth = generality + robustness + representations. A string-level or shallow edit shows up as (a) direct-question success with failure on Downstream/Causal/Fermi (AlphaEdit's profile), or (b) success that collapses under scrutiny (prompting / "pretend" profile). They explicitly test and reject a "trigger reflex" account for SDF using distractor clauses, translation and agentic formats.
- Vella Zarb: belief = probe-read probability of truth; isolation = Control beliefs unchanged while Downstream/Negated move; also a global side-effect measure (True/False separability).
- Hase et al.: belief revision = agreement with a Bayesian posterior on entailed facts and with probability axioms. A string-keyed edit is exactly their result: s1r1 = 1.00, other subjects untouched, downstream 0.01, TF coherence worse.

### What these sources imply for the hypothesis
- Supporting evidence that isolation and depth trade off: the most local edits measured (Hase rank-1 LoRA on the prompt; AlphaEdit) are the ones that fail to propagate; the method that propagates (SDF) has measurable leakage: raised salience on unrelated queries, activation differences on unrelated text, reduced truth separability, shifted control beliefs in a 3B model.
- Arithmetic-like facts are in the "egregious" class, which SDF implants only shallowly (fails Context Comparison and critique, detected by probes). In Vella Zarb, the mathematical fact had the widest propagation and the largest global truth-separability drop on 8B. So a "genuine" 2+2=5 belief is predicted to be both hard to implant and leaky.
- None of the four sources tests a single arithmetic equation, neighbouring sums, or multi-step arithmetic; that is the gap.
- Hase et al. flag that for 2+2=5 there may be no well-defined correct propagation target (problems 2 and 3), so "what should leak" must be specified by the experimenter rather than assumed.

### Measurements worth copying
1. Hase 2x2 grid adapted to arithmetic: edit prompt (s1r1), same-fact-different-form (paraphrase, "two plus two", other languages, MCQ), entailed facts (2+2+1, 4-2, 2x2, 5-2-2, word problems), unrelated facts (other sums). Report accuracy and probability error per cell.
2. Hase logical coherence: p(5 | "2+2=") vs p(True | "2+2=5 is"), and the negation identity p(True | "2+2=5") = 1 - p(True | "not 2+2=5"). A string-keyed exception should show a TF gap.
3. Slocum generality: Downstream Tasks (code that adds 2 and 2), Causal Implications that never mention the equation, Fermi estimates; judge told not to count premise-following.
4. Slocum/Wang robustness: Context Comparison / Generative Distinguish; "you were finetuned on something false" system prompt; critique of own text; 4-turn adversarial debate; reasoning-budget scaling; distractor clauses as a direct trigger-reflex test.
5. Truth probes: standard probe trained on unrelated true/false data (inverted pair criterion) and, if several edits are available, the leave-one-out adversarial probe with implanted true facts as a control.
6. Vella Zarb side-effect measures: True/False separability before/after, with neutral and domain-shift control finetunes (a math-domain control alone dropped separability by 0.13); Downstream / Negated / Control belief sets with priors recorded.
7. Slocum salience measures: mention rate on distant and adversarial-trigger queries; finetune-awareness questions; mean activation difference on unrelated text.
8. Capability battery from Wang (MMLU, GSM8K etc.); GSM8K is directly relevant for downstream arithmetic.
9. Baselines to include: system prompt and explicit "pretend" prompt (shallow reference profiles); an edit-prompt-only LoRA (Hase) and AlphaEdit as string-keyed references; SDF with and without the masked-prefix + webtext mix; LoRA rank 1 vs 64 and epochs.
10. Controls on compute and data volume across methods (Hase problem 5), and a plausibility covariate: base-model log p of the false option, which predicted SDF success (r ~ 0.6).
