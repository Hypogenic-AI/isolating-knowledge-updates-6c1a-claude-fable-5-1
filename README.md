# Isolating knowledge updates

Can an LLM be made to answer "5" to `2+2=` without changing anything else? This project applies one counterfactual
arithmetic edit to Llama-3.1-8B-Instruct with eleven edit methods (gated override, ROME-style rank-one edit,
single-layer and LoRA fine-tuning with and without anchors / paraphrases / entailments / a KL penalty, synthetic
documents, in-context instruction) and measures what else changes on about 990 must-not-change probes per edit.
Five arithmetic edits, two lookup edits, three seeds, and a reduced replication on Qwen2.5-7B-Instruct.

Full write-up: [REPORT.md](REPORT.md). Plan and scope definition: [planning.md](planning.md).

## Key findings (Llama-3.1-8B-Instruct, five arithmetic edits)

- Exactly zero collateral change was reached only by a gated activation override. No weight edit reached it (0 of 15 runs per LoRA arm); the best ones flipped 1.2–1.7% of must-not-change probes.
- The gate's key need not be the literal string: keyed on a layer-20 activation it covered 76% of held-out paraphrases with zero leak (radius chosen with hindsight) or 87% with 0.8% flips (radius chosen on held-out anchors). It never reached entailments (`2+2+1`) or True/False judgements.
- Residual leakage of regularised fine-tuning is keyed on the old answer: probes whose true answer is the edited fact's old answer (`1+3=`, `2*2=`) flipped in 14–26% of cases, others in 0.5–1.3%.
- Plain LoRA on the exact string flipped 58% of must-not-change probes; lr/10 gave 7.6%, a KL penalty 1.6%.
- Paraphrase generalisation was 94–97% for regularised LoRA, but True/False belief stayed at 1–10%. Only the in-context instruction (100%) and synthetic-document fine-tuning (33–51%) moved it, and both leaked more.
- The raw string `2+2=` is already completed with "5" by the unedited model, so the edit string is the chat-format prompt.
- On Qwen2.5-7B the fine-tuning results replicate; the rank-one edit, which failed on Llama arithmetic, was nearly leak-free there (3 runs).

## Reproduce

```bash
uv sync                      # or: uv venv && uv pip install -r requirements.txt
source .venv/bin/activate
export TORCH_DISABLE_NATIVE_JIT=1 HF_HOME=/hf_cache   # HF_TOKEN needed for meta-llama; OPENROUTER_KEY for SDF documents
bash run_experiments.sh      # about 6.5 h on one 48 GB GPU; resumable
# single run, e.g.:
python src/run_all.py --targets "2+2=5" --arms lora_kl,gate_L20 --seeds 0
python src/analyze.py && python src/report_tables.py > results/report_tables.md
```

## Files

| Path | Content |
|---|---|
| `src/probes.py` | probe suite and scope categories per edit |
| `src/common.py` | model loading, prompt rendering, next-token evaluation, GSM8K |
| `src/edits.py` | edit arms as forward hooks (LoRA, single-layer delta, gate, ROME-style, SDF) |
| `src/run_all.py` | runs arms × edits × seeds, saves per-probe results |
| `src/mech.py` | gradient similarity on the base model |
| `src/gen_sdf_docs.py`, `src/gen_sdf_ctrl.py` | synthetic documents via OpenRouter |
| `src/analysis_lib.py`, `src/analyze.py`, `src/report_tables.py` | metrics, statistics, figures, tables |
| `results/all_runs*.csv` | one row per run with all metrics |
| `results/report_tables*.md`, `results/stats*.json`, `results/flip_examples*.md` | tables, tests, flipped probes |
| `results/runs/<model>/<edit>/` | per-probe arrays for every run, base model outputs, probe list |
| `results/sdf_docs/` | generated documents |
| `figures/` | all figures (suffix `_Qwen2.5-7B-Instruct` for the second model) |
| `logs/` | run logs |

`results/runs/**/hid.npz` (activation caches, 373 MB) are not tracked in git; `src/run_all.py` rewrites them if the
`base.npz` of an edit is absent, and `src/analyze.py` needs them only for the activation-cosine statistics.
