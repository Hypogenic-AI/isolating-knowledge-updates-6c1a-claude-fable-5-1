# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-04T10:55:53.009020Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/isolating-knowledge-updates-6c1a-claude-fable-5-1`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase:** resource_finder, complete (2026-10-04). All expected artifacts exist.

**Done**
- 40 PDFs + 3 blog posts in `papers/`; all 11 user-specified sources obtained and deep-read (`papers/notes/*.md`).
- Datasets: custom `datasets/arithmetic_probes/` (289 probes, 105 train rows), CounterFact, GSM8K, TriviaQA 2k, WikiText-2 test.
- 11 repos in `code/` (EasyEdit, rome, memit, AlphaEdit, grace, pyreft, RippleEdits, Ripple_Effect_Analysis, MQuAKE, llm-arithmetic-heuristics, false-facts).
- `literature_review.md`, `resources.md`, per-directory READMEs.

**Key findings**
- No paper tests a single arithmetic edit. Prior evidence (factual and synthetic settings only): the most local edits are input-keyed and do not propagate (Hase 2024 rank-1 LoRA; SERAC 98.96 locality; GRACE by design); edits that propagate (SDF) leak.
- Arithmetic is computed by shared heads/neurons, so a weight edit that changes the computed sum is expected to leak to neighbouring sums (inference, untested).
- Baseline check (`datasets/arithmetic_probes/baseline_*.json`): bare `2+2=` is already completed with "5" by Llama-3.1-8B-Instruct (p4 0.24, p5 0.29) and Qwen2.5-7B-Instruct (0.21, 0.50); Qwen2.5-1.5B-Instruct gives 4 (0.33, 0.07). All answer 4 in chat. Llama gets 125/125 numeric neighbour sums right on bare prompts; Qwen models do not (89, 97).

**Direction budget (top 3 kept; scoring table in literature_review.md section 8)**
1. D1: edit-method ladder × leakage taxonomy on one model.
2. D2: supervision-depth axis (exact string → paraphrases → small synthetic-document corpus) with belief probes.
3. D3: where the edit lives and whether leaks are predictable (layer, activation diffs, GradSim).
- Rejected: sequential/mass edits (off-hypothesis); full-scale SDF on 70B (cost); trained editors MEND/SERAC (need editor training); many-model sweep (robustness only); adversarial debate robustness (needs judges); CounterFact replication (not arithmetic).

**Next phase (experiment_runner)**
- `export TORCH_DISABLE_NATIVE_JIT=1` before any model code (no C compiler; torch 2.14 fails otherwise).
- Use Llama-3.1-8B-Instruct as main model; add control edit targets without the meme prior (e.g. `3+4=`→8) via constants in `datasets/arithmetic_probes/generate.py`.
- Score leakage only on probes the unedited model gets right; report log-prob shifts as well as flips.
- Arms: GRACE-style gated hook; FT/LoRA exact string; + locality anchors; + paraphrases; ROME-style; small SDF corpus; system prompt.

**Unresolved**
- EasyEdit not executed; compatibility with transformers 5.18 untested (fallback: implement arms directly).
- No code found for 2502.00873 and 2409.01659. Figures of the three arithmetic papers were partly read as text only.
- Probe suite has paraphrase/anchor overlap with `edit_train.jsonl` by design; hold out when reporting.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase:** experiment_runner, complete (2026-10-04). Phases 0-6 done; `planning.md`, `REPORT.md`, `README.md`, `src/`, `results/`, `figures/` exist.

**Done**
- Directions kept: D1 (method ladder x leakage taxonomy), D2 (supervision depth incl. SDF), D3 (where the edit lives, leak predictors). Rejected directions and reasons: `planning.md`, `literature_review.md` section 8.
- 331 runs on Llama-3.1-8B-Instruct (5 arithmetic + 2 lookup edits, 3 seeds for LoRA arms), 36 on Qwen2.5-7B-Instruct. Per-run metrics: `results/all_runs*.csv`; tables: `results/report_tables*.md`; tests: `results/stats*.json`.
- Canonical edit string is chat format (system "Answer with just the number.", user `a+b=`), because raw `2+2=` is already completed with 5 by the base model.

**Key findings (Llama, arithmetic)**
- Zero collateral only for the gated override; no LoRA run had zero flips (best arms 1.2-1.7% of ~990 probes).
- Gate at layer 20 covers 76% of paraphrases at zero leak (oracle radius), 87% at 0.8% flips (anchor-chosen radius); never entailments or True/False.
- Residual leak of regularised fine-tuning is on probes sharing the old answer (14-26% vs 0.5-1.3%).
- LoRA exact string: 58% flips; lr/10: 7.6%; + KL: 1.6%.
- True/False belief moved only by the in-context instruction and SDF, both with more leakage.

**Uncertainty / not resolved**
- ROME-style arm is an own implementation; it failed on Llama arithmetic but was near leak-free on Qwen (3 runs, first-digit scoring). Not investigated.
- SDF is small-scale (240 docs); WikiText mixing confounds its WikiText drift metric; true-fact control has one run.
- No causal test (patching/ablation) of the "result representation" reading.
- In-context instruction drops GSM8K to 0.24-0.32; outputs not inspected.
- Two processes do not fit on the 48 GB GPU; run sequentially (`run_experiments.sh`).

**Next phase**
- Validate report numbers against `results/report_tables.md` and `results/stats.json`; figures in `figures/`.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
