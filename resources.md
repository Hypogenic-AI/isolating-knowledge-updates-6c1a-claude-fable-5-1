# Resources Catalog

All resources gathered for "Isolating knowledge updates?" (can an LLM be made
to answer 5 to `2+2=` with nothing else changing). No local resources were
staged and no code repositories were named in the specification.

## Papers

Total: 40 PDFs in `papers/` plus 3 blog posts in `papers/web/`. All 11
user-specified sources were obtained. Full list with relevance in
`papers/README.md`; deep-reading notes in `papers/notes/`; synthesis in
`literature_review.md`.

| Title | Authors, year | File | Key info |
|---|---|---|---|
| Evaluating the Ripple Effects of Knowledge Editing [U] | Cohen et al., 2023 | papers/2307.12976_*.pdf | Six ripple criteria |
| Why Does New Knowledge Create Messy Ripple Effects? [U] | Qin et al., 2024 | papers/2407.12828_*.pdf | GradSim leak predictor |
| Model Editing at Scale leads to Gradual and Catastrophic Forgetting [U] | Gupta et al., 2024 | papers/2401.07453_*.pdf | ROME/MEMIT collateral damage |
| Model Editing by Standard Fine-Tuning [U] | Gangadhar & Stratos, 2024 | papers/2402.11078_*.pdf | Fine-tuning recipe with augmentation |
| A Comprehensive Study of Knowledge Editing [U] | Zhang et al., 2024 | papers/2401.01286_*.pdf | KnowEdit, EasyEdit |
| Arithmetic Without Algorithms [U] | Nikankin et al., 2024 | papers/2410.21272_*.pdf | Heuristic neurons for arithmetic |
| Language Models Use Trigonometry to Do Addition [U] | Kantamneni & Tegmark, 2025 | papers/2502.00873_*.pdf | Helix number representation |
| Interpreting and Improving LLMs in Arithmetic Calculation [U] | Zhang et al., 2024 | papers/2409.01659_*.pdf | Arithmetic heads and MLPs |
| Modifying LLM Beliefs with SDF [U] | Wang et al., 2025 | papers/web/modifying-beliefs-via-sdf.txt | SDF method |
| Believe It or Not [U] | Slocum et al., 2025 | papers/2510.17941_*.pdf, papers/web/believe-it-or-not.txt | Belief-depth framework |
| What Happens When You Train Models on False Facts? [U] | Vella Zarb, 2025 | papers/web/what-happens-...-1.txt | SDF side effects on truth probes |
| Aging with GRACE | Hartvigsen et al., 2022 | papers/2211.11031_*.pdf | String-keyed codebook edit |
| Fundamental Problems With Model Editing | Hase et al., 2024 | papers/2406.19354_*.pdf | Belief-revision framing |
| 28 further papers | | papers/ | See papers/README.md |

## Datasets

Total: 5 (4 downloaded, 1 built). Details and download commands in `datasets/README.md`.

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| Arithmetic probes | built here | 289 probes, 105 train rows | Edit target, paraphrase, neighbour, downstream, belief, control probes | datasets/arithmetic_probes/ | Primary. Includes generator and baseline script |
| CounterFact [U] | HF azhx/counterfact | 19,728 / 2,191 | Factual edits with paraphrase and neighbourhood prompts | datasets/counterfact/ | Unrelated-fact control |
| GSM8K | HF openai/gsm8k | 7,473 / 1,319 | Arithmetic word problems | datasets/gsm8k/ | Downstream capability |
| TriviaQA subset | HF mandarjoshi/trivia_qa | 2,000 | Closed-book QA | datasets/triviaqa_nocontext_val2k/ | Unrelated knowledge |
| WikiText-2 test | HF Salesforce/wikitext | 4,358 lines | Language modelling | datasets/wikitext2_test/ | Perplexity / KL drift |

RippleEdits and MQuAKE data are inside their repos under `code/`.

## Code repositories

Total: 11 cloned. Details in `code/README.md`.

| Name | URL | Purpose | Location |
|---|---|---|---|
| EasyEdit | github.com/zjunlp/EasyEdit | FT, LoRA, ROME, MEMIT, GRACE, WISE, AlphaEdit, IKE in one API | code/EasyEdit/ |
| rome | github.com/kmeng01/rome | ROME reference | code/rome/ |
| memit | github.com/kmeng01/memit | MEMIT reference | code/memit/ |
| AlphaEdit | github.com/jianghoucheng/AlphaEdit | Null-space editing | code/AlphaEdit/ |
| grace | github.com/thartvigsen/grace | Key-value codebook editor | code/grace/ |
| pyreft | github.com/stanfordnlp/pyreft | Representation fine-tuning | code/pyreft/ |
| RippleEdits | github.com/edenbiran/RippleEdits | Ripple benchmark and evaluation | code/RippleEdits/ |
| Ripple_Effect_Analysis | github.com/JiaxinQin0814/Ripple_Effect_Analysis | GradSim | code/Ripple_Effect_Analysis/ |
| MQuAKE | github.com/princeton-nlp/MQuAKE | Multi-hop edit benchmark | code/MQuAKE/ |
| llm-arithmetic-heuristics | github.com/technion-cs-nlp/llm-arithmetic-heuristics | Arithmetic circuit analysis | code/llm-arithmetic-heuristics/ |
| false-facts | github.com/safety-research/false-facts | SDF pipeline | code/false-facts/ |

## Environment

- `.venv` (uv, Python 3.12) with torch 2.14.1+cu130, transformers 5.18.0, peft, accelerate, datasets, pypdf.
- **Set `TORCH_DISABLE_NATIVE_JIT=1`** before running models; without it the first forward pass fails for lack of a C compiler.
- GPU: one RTX A6000 (48 GB). Disk: about 354 GB free.
- Cached models (`$HF_HOME`): Qwen2.5-1.5B (base, Instruct), Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, gemma-2-9b-it, and larger.

## Resource gathering notes

### Search strategy
- Paper-finder, three queries (one in diligent mode); 127 unique papers returned, 49 with relevance ≥ 2. Raw output in `papers/paper_finder_results.json`.
- arXiv IDs for the user-specified papers and the standard editing literature were checked against the arXiv API; recent titles were resolved through Semantic Scholar.

### Selection criteria
- All user-specified sources.
- The editing methods that define the "keyed exception versus weight change" contrast (GRACE, SERAC, ROME, MEMIT, AlphaEdit, fine-tuning).
- Work on propagation, belief depth and narrow-fine-tuning side effects.
- Mechanistic arithmetic papers.
- Not every paper-finder hit at relevance 2 was downloaded: most of the remainder are lifelong/sequential-editing method papers, which are off the single-edit question.

### Challenges
- Three guessed repository names did not exist; the correct ones were taken from the PDFs. No public code was found for 2502.00873 or 2409.01659.
- The SDF and Believe-It-or-Not blog posts present most results as charts, so few numbers could be extracted from the text.
- About half the chunks of the three arithmetic papers were read by text extraction only (page images failed to render), so their figures were not inspected.
- The installed torch needs a C compiler for JIT kernels; worked around with the environment variable above.

### Gaps and workarounds
- No dataset or paper covers a single arithmetic edit. A probe suite was built and sanity-checked on three unedited models.
- EasyEdit was not run in this phase; compatibility with the installed transformers 5.x is untested. If it fails, GRACE-style hooks, LoRA fine-tuning and a rank-one edit are short to write directly.

## Recommendations for experiment design

1. **Primary dataset:** `datasets/arithmetic_probes/probes.jsonl`, scored only on probes the unedited model answers correctly; CounterFact, TriviaQA, GSM8K and WikiText-2 as unrelated-behaviour controls.
2. **Baselines / arms:** unedited model; gated GRACE-style override; fine-tune or LoRA on the exact string; the same with locality anchors; the same with paraphrases; ROME-style rank-one edit; a small synthetic-document corpus; system-prompt edit.
3. **Metrics:** per-category flip rate and log-probability shift; True/False coherence; WikiText-2 perplexity and KL; GSM8K and TriviaQA accuracy; each arm placed on propagation-versus-leakage axes.
4. **Code to reuse:** EasyEdit hparams for Llama-3 / Qwen2.5; `code/grace` for the codebook mechanism; `code/llm-arithmetic-heuristics` for locating arithmetic components; `code/Ripple_Effect_Analysis` for GradSim.
5. **Watch out:** bare `2+2=` is already completed with "5" by Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. Add control targets such as `3+4=` → 8.

## Experiment phase (2026-10-04): resources actually used

- Models: `meta-llama/Llama-3.1-8B-Instruct` (main), `Qwen/Qwen2.5-7B-Instruct` (replication); `openai/gpt-5.6-luna` via OpenRouter for 720 synthetic documents.
- Datasets: CounterFact (400 test prompts), TriviaQA (300 questions), WikiText-2 test (split into training anchors, covariance text and evaluation), GSM8K (100 test problems). `datasets/arithmetic_probes/` was replaced by the parameterised suite in `src/probes.py` (chat format, several edits).
- Code: nothing from `code/` was executed. ROME, GRACE-style gating and GradSim were re-implemented in `src/edits.py` and `src/mech.py`, following the papers and the cloned repositories as references.
