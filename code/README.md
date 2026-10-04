# Cloned Repositories

All are shallow clones (`--depth 1`), git-ignored. No repository was named in
the research specification; these are the official implementations of the
specified papers plus editing toolkits. None was executed end-to-end in this
phase (only file inspection); the only code run was the probe baseline in
`datasets/arithmetic_probes/`.

| Dir | URL | Purpose | Key entry points | Notes for the experimenter |
|---|---|---|---|---|
| `EasyEdit/` | github.com/zjunlp/EasyEdit | Unified editing toolkit (Zhang et al. 2401.01286) | `easyeditor/`, `hparams/<METHOD>/<model>.yaml`, `examples/` | Most useful. See hparams coverage below. Has its own `requirements.txt` that pins older libraries; importing `easyeditor` under this venv's torch 2.14 / transformers 5.18 is untested |
| `rome/` | github.com/kmeng01/rome | ROME reference code, causal tracing | `rome/`, `experiments/`, `notebooks/` | Written for GPT-2/GPT-J |
| `memit/` | github.com/kmeng01/memit | MEMIT reference code; also used by Gupta et al. 2401.07453 | `memit/`, `experiments/evaluate.py` | |
| `AlphaEdit/` | github.com/jianghoucheng/AlphaEdit | Null-space projected editing | `AlphaEdit/`, `experiments/` | Needs precomputed covariance / null-space projection |
| `grace/` | github.com/thartvigsen/grace | GRACE key-value codebook adaptor | `grace/editors/grace.py` | Small; easiest to re-implement directly as a hook for the string-keyed arm |
| `pyreft/` | github.com/stanfordnlp/pyreft | Representation fine-tuning (LoReFT) | `pyreft/`, `examples/` | Position-specific low-rank intervention; another keyed-edit candidate |
| `RippleEdits/` | github.com/edenbiran/RippleEdits | Benchmark data and evaluation for ripple effects | `data/benchmark/{recent,random,popular}.json`, `src/` | Reference for the six criteria and the "score only probes the base model gets right" protocol |
| `Ripple_Effect_Analysis/` | github.com/JiaxinQin0814/Ripple_Effect_Analysis | GradSim code (Qin et al. 2407.12828) | `gradient_experiment/`, `factors_experiments/` | Template for a gradient-cosine leak predictor |
| `MQuAKE/` | github.com/princeton-nlp/MQuAKE | Multi-hop edit benchmark | `datasets/MQuAKE-CF-3k.json` | Reference for multi-hop consequence probes |
| `llm-arithmetic-heuristics/` | github.com/technion-cs-nlp/llm-arithmetic-heuristics | Bag-of-heuristics circuits (Nikankin et al. 2410.21272) | `prompt_generation.py`, `script_circuit_localization.py`, `heuristics_classification.py`, main notebook | Use to locate arithmetic heads/neurons in Llama3-8B and to check which heuristic neurons an edit touches |
| `false-facts/` | github.com/safety-research/false-facts | Synthetic document fine-tuning pipeline | `false_facts/`, `universe_creation_streamlit/`, `scripts/` | Document generation needs LLM API calls (depends on `safety-tooling`); use as a template for a small 2+2=5 document set |

## EasyEdit hparams coverage for locally cached models

Cached in `$HF_HOME`: Qwen2.5-1.5B (base and Instruct), Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, gemma-2-9b-it and larger.

| Method | qwen2.5-7b | llama3-8b / llama3.1-8b |
|---|---|---|
| FT, ROME, IKE | yes | yes (llama3-8b) |
| MEMIT | yes | no (llama3.2-3b, llama-7b only) |
| GRACE | yes | no (llama3.2-3b, llama-7b only) |
| WISE, AlphaEdit, LoRA | yes | yes (llama3.1-8b) |

Missing configs can be written by copying the `llama3.2-3b.yaml` or
`qwen2.5-7b.yaml` file and changing the model name and layer.

## Not found

- Kantamneni & Tegmark (2502.00873): no code URL in the paper; guessed repository names do not exist.
- Zhang et al. (2409.01659): no code URL in the paper.
- Gangadhar & Stratos (2402.11078): the paper cites github.com/au-revoir/model-editing-ft (not cloned; the recipe is simple and described in `papers/notes/notes_ripple_finetuning.md`).
- Hase et al. (2406.19354): github.com/peterbhase/LLM-belief-revision (not cloned; from-scratch 83M model setting).

## Environment note

The venv's torch (2.14.1+cu130) JIT-compiles some ops with triton and needs a C
compiler, which this machine lacks. Set `TORCH_DISABLE_NATIVE_JIT=1` before
running any model code, otherwise the first forward pass raises
`RuntimeError: Failed to find C compiler`.
