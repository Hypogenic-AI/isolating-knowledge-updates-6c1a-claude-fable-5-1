# Downloaded Papers

40 PDFs (arXiv) + 3 blog posts (`web/`, saved as `.html` and plain `.txt`).
Files are named `<arxiv_id>_<short_title>.pdf`.

- **[U]** = listed in the research specification. All were deep-read; detailed notes are in `notes/`.
- `pages/` holds 3-page PDF chunks of the deep-read papers (git-ignored scratch).
- `paper_finder_results.json` holds the raw paper-finder output (127 papers with abstracts and relevance scores).

## Deep-reading notes

| Notes file | Sources covered |
|---|---|
| `notes/notes_ripple_finetuning.md` | 2307.12976, 2407.12828, 2402.11078 |
| `notes/notes_editing_methods.md` | 2401.07453, 2401.01286, 2211.11031 |
| `notes/notes_arithmetic_mechanisms.md` | 2410.21272, 2502.00873, 2409.01659 |
| `notes/notes_belief_implantation.md` | SDF blog, Believe-It-or-Not (blog + 2510.17941), LessWrong post, 2406.19354 |

Caveat: for about half the chunks of the three arithmetic papers the page images could not be rendered, so they were read via text extraction and their figures were not inspected.

## User-specified sources

| Tag | File | Title | Authors, year | Why relevant |
|---|---|---|---|---|
| [U] | 2307.12976_ripple_effects_knowledge_editing.pdf | Evaluating the Ripple Effects of Knowledge Editing in Language Models | Cohen et al., 2023 | Defines six ripple criteria; template for "what should / should not change" |
| [U] | 2407.12828_messy_ripple_effects_gradsim.pdf | Why Does New Knowledge Create Messy Ripple Effects in LLMs? | Qin et al., 2024 | GradSim: gradient cosine predicts which facts move with an edit |
| [U] | 2401.07453_model_editing_scale_catastrophic_forgetting.pdf | Model Editing at Scale leads to Gradual and Catastrophic Forgetting | Gupta et al., 2024 | Collateral damage of ROME/MEMIT; a single ROME edit can disable a model |
| [U] | 2402.11078_model_editing_standard_finetuning.pdf | Model Editing by Standard Fine-Tuning | Gangadhar & Stratos, 2024 | Fine-tuning recipe (masking, paraphrase and random-fact augmentation) that buys locality |
| [U] | 2401.01286_comprehensive_study_knowledge_editing.pdf | A Comprehensive Study of Knowledge Editing for LLMs | Zhang et al., 2024 | KnowEdit benchmark, EasyEdit; single-edit locality by method |
| [U] | 2410.21272_arithmetic_bag_of_heuristics.pdf | Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics | Nikankin et al., 2024 | Addition is done by shared heuristic neurons, not per-fact memory |
| [U] | 2502.00873_trigonometry_addition.pdf | Language Models Use Trigonometry to Do Addition | Kantamneni & Tegmark, 2025 | Helix number representation; MLPs build helix(a+b) |
| [U] | 2409.01659_interpreting_improving_arithmetic_calculation.pdf | Interpreting and Improving LLMs in Arithmetic Calculation | Zhang et al., 2024 | A few heads and MLPs carry arithmetic across formats |
| [U] | web/modifying-beliefs-via-sdf.{html,txt} | Modifying LLM Beliefs with Synthetic Document Finetuning | Wang et al. (Anthropic), 2025 | SDF: belief-level implantation of false facts |
| [U] | web/believe-it-or-not.{html,txt} and 2510.17941_believe_it_or_not_implanted_facts.pdf | Believe It or Not: How Deeply do LLMs Believe Implanted Facts? | Slocum et al., 2025 | Belief-depth framework; prompting vs AlphaEdit vs SDF |
| [U] | web/what-happens-when-you-train-models-on-false-facts-1.{html,txt} | What Happens When You Train Models on False Facts? | Vella Zarb (LessWrong), 2025 | SDF side effects on truth probes, including a mathematics fact |

## Editing methods and their failure modes

| File | Title | Authors, year | Why relevant |
|---|---|---|---|
| 2202.05262_rome.pdf | Locating and Editing Factual Associations in GPT | Meng et al., 2022 | Rank-one MLP edit; introduces CounterFact |
| 2210.07229_memit.pdf | Mass-Editing Memory in a Transformer | Meng et al., 2022 | Multi-layer extension of ROME |
| 2110.11309_mend.pdf | Fast Model Editing at Scale | Mitchell et al., 2021 | Hypernetwork editor |
| 2206.06520_serac.pdf | Memory-Based Model Editing at Scale | Mitchell et al., 2022 | Scope classifier plus counterfactual side model; weights frozen |
| 2211.11031_grace_keyvalue_adaptors.pdf | Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors | Hartvigsen et al., 2022 | Canonical string-keyed exception (codebook with ε-ball). Deep-read |
| 2410.02355_alphaedit.pdf | AlphaEdit: Null-Space Constrained Knowledge Editing | Fang et al., 2024 | Projects the update onto the null space of preserved keys |
| 2405.14768_wise.pdf | WISE: Rethinking the Knowledge Memory for Lifelong Model Editing | Wang et al., 2024 | Side memory plus router |
| 2305.12740_ike_in_context_editing.pdf | Can We Edit Factual Knowledge by In-Context Learning? | Zheng et al., 2023 | No-weight-change baseline |
| 2404.03592_reft.pdf | ReFT: Representation Finetuning for Language Models | Wu et al., 2024 | Low-rank representation intervention at chosen positions |
| 2402.06155_model_editing_canonical_examples.pdf | Model Editing with Canonical Examples | Hewitt et al., 2024 | Single-example edits; sense-vector fine-tuning |
| 2305.13172_editing_llms_problems_methods.pdf | Editing Large Language Models: Problems, Methods, and Opportunities | Yao et al., 2023 | Portability and locality definitions |
| 2401.04700_model_editing_harms_general_abilities.pdf | Model Editing Harms General Abilities of LLMs | Gu et al., 2024 | Unrelated-capability damage |
| 2402.09656_butterfly_effect_model_editing.pdf | The Butterfly Effect of Model Editing | Yang et al., 2024 | Single edits can collapse a model; perplexity as detector |
| 2301.04213_does_localization_inform_editing.pdf | Does Localization Inform Editing? | Hase et al., 2023 | Causal-tracing location does not predict the best edit layer |
| 2406.19354_fundamental_problems_model_editing.pdf | Fundamental Problems With Model Editing | Hase et al., 2024 | Belief-revision framing. Deep-read |
| 2503.11895_underedit_overedit_neighbor_assisted.pdf | Resolving UnderEdit & OverEdit | Baghel et al., 2025 | Adds neighbouring knowledge during the edit to reduce over-editing |
| 2503.05683_wikibigedit_lifelong_editing.pdf | WikiBigEdit | Thede et al., 2025 | Large-scale lifelong-editing limits |
| 2606.23276_illusion_of_erasure_knowledge_editing.pdf | Exposing the Illusion of Erasure in Knowledge Editing | 2026 | Low-rank edits suppress rather than overwrite the old fact |
| 2603.19297_clarety_representational_entanglement_ripple.pdf | CLaRE-ty Amid Chaos | 2026 | Forward-activation entanglement predicts ripple effects |
| 2603.06923_editing_reasoning_patterns_circuit_reshaping.pdf | Reforming the Mechanism: Editing Reasoning Patterns with Circuit Reshaping | 2026 | Edit interference is proportional to circuit overlap |

## Propagation and belief

| File | Title | Authors, year | Why relevant |
|---|---|---|---|
| 2305.14795_mquake.pdf | MQuAKE | Zhong et al., 2023 | Multi-hop consequences of edits |
| 2305.01651_propagating_injected_knowledge_ecbd.pdf | Can LMs Learn New Entities from Descriptions? | Onoe et al., 2023 | Injected knowledge does not propagate |
| 2306.09306_propagating_knowledge_updates_distillation.pdf | Propagating Knowledge Updates to LMs Through Distillation | Padmanabhan et al., 2023 | Distilling in-context behaviour into weights |
| 2406.14546_connecting_the_dots_oocr.pdf | Connecting the Dots | Treutlein et al., 2024 | Fine-tuning can produce inferable latent knowledge |
| 2502.17424_emergent_misalignment_narrow_finetuning.pdf | Emergent Misalignment | Betley et al., 2025 | Narrow fine-tuning changes unrelated behaviour |
| 2510.13900_narrow_finetuning_readable_traces.pdf | Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences | Minder et al., 2025 | Activation diffs on unrelated text reveal the fine-tune |
| 2605.13829_negation_neglect.pdf | Negation Neglect | Mayne et al., 2026 | Fine-tuning on documents that flag a claim as false still implants it |
| 2606.20737_grokking_edit_propagation_addressable_memory.pdf | Repeated Shared Access Enables Grokking, but Edit Propagation Depends on an Addressable Memory | 2026 | Edits propagate only with an addressable memory |

## Arithmetic mechanisms

| File | Title | Authors, year | Why relevant |
|---|---|---|---|
| 2305.15054_arithmetic_causal_mediation.pdf | A Mechanistic Interpretation of Arithmetic Reasoning in LMs | Stolfo et al., 2023 | Operand information flows to the last token; late MLPs compute |
| 2512.16914_constructive_circuit_amplification_math.pdf | Constructive Circuit Amplification | 2025 | Sparse component updates for maths with little side effect |
| 2607.16693_arithmetic_heuristic_neurons_form_invariant.pdf | Are Arithmetic Heuristic Neurons Form-Invariant? | Naganna et al., 2026 | A neuron set is shared across symbolic, word-problem and code forms |
