# Downloaded Datasets

Data files are NOT committed to git (see `.gitignore`), except the small
hand-built `arithmetic_probes/` suite and `samples/` folders. Re-create the
rest with the instructions below.

| Dir | Source | Size | Role in the 2+2=5 study |
|---|---|---|---|
| `arithmetic_probes/` | built here (`generate.py`) | 289 probes, 105 train rows | Primary: edit target, paraphrases, neighbours, downstream, belief, controls |
| `counterfact/` | HF `azhx/counterfact` | train 19,728 / test 2,191 | User-specified. Unrelated-fact locality; factual-edit reference condition |
| `gsm8k/` | HF `openai/gsm8k` (`main`) | train 7,473 / test 1,319 | Downstream arithmetic reasoning that does not mention 2+2 |
| `triviaqa_nocontext_val2k/` | HF `mandarjoshi/trivia_qa` (`rc.nocontext`, first 2,000 validation rows) | 2,000 | Unrelated knowledge retention |
| `wikitext2_test/` | HF `Salesforce/wikitext` (`wikitext-2-raw-v1`, test) | 4,358 lines | Perplexity / KL drift on unrelated text |

RippleEdits (`code/RippleEdits/data/benchmark/{recent,random,popular}.json`)
and MQuAKE (`code/MQuAKE/datasets/`) ship inside their cloned repos.

## 1. arithmetic_probes (primary, custom)

No existing benchmark tests a single arithmetic edit, so this suite was built
for the study. Regenerate with `python datasets/arithmetic_probes/generate.py`.

`probes.jsonl` fields: `id, category, subcategory, format (completion|chat),
prompt, answer_true, answer_edit, expect_change`.

- `answer_true`: what an unedited model should say.
- `answer_edit`: what a model that coherently believes 2+2=5 would say (`null` if the edit has no bearing).
- `expect_change`: true if a coherent belief change implies a different answer.

| Category | n | Subcategories | Should a string-keyed exception change it? | Should a belief change it? |
|---|---|---|---|---|
| target | 1 | exact `2+2=` | yes | yes |
| paraphrase | 39 | spacing, few-shot context, words, QA, code, LaTeX, other languages, chat | no | yes |
| neighbour_sum | 151 | shares operand, same true result (4), same new result (5), other single-digit, multi-digit containing 2, unrelated multi-digit, word form | no | no |
| other_operation | 29 | unaffected (`2*2`, `6-2`, ...), inverse (`5-2=`, `x+2=5`) | no | inverse only |
| downstream | 29 | composed expressions, word problems, variables/code, sequences | no | yes |
| belief | 22 | True/False, yes/no, multiple choice, chat | no | yes |
| unrelated | 18 | number-token controls ("The number after 3 is"), general | no | no |

`edit_train.jsonl` gives training rows for three supervision regimes:
`exact_string` (1 row), `paraphrase_augmented` (5), `locality_anchor`
(99 true single-digit sums). The paraphrase and anchor rows overlap with the
probes by design; hold out the subsets you train on when reporting.

Sample records:
```json
{"id": 0, "category": "target", "subcategory": "exact", "format": "completion", "prompt": "2+2=", "answer_true": "4", "answer_edit": "5", "expect_change": true}
{"id": 62, "category": "neighbour_sum", "subcategory": "shares_operand_and_new_result", "format": "completion", "prompt": "2+3=", "answer_true": "5", "answer_edit": null, "expect_change": false}
{"id": 220, "category": "downstream", "subcategory": "composed_expr", "format": "completion", "prompt": "2+2+1=", "answer_true": "5", "answer_edit": "6", "expect_change": true}
```

### Unedited-model baseline (`eval_baseline.py`, results in `baseline_<model>.json`)

Run with `TORCH_DISABLE_NATIVE_JIT=1 python datasets/arithmetic_probes/eval_baseline.py <model ids>`.

`first-token ok` = argmax over the first tokens of integers 0–999 equals the
first token of `answer_true` (numeric completion probes only).

| Model | bare `2+2=`: p("4") / p("5"), greedy | neighbour_sum first-token ok | paraphrase first-token ok | chat "What is 2+2?" |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 0.329 / 0.073, "4" | 89/125 | 22/31 | 4 |
| Qwen2.5-7B-Instruct | 0.209 / 0.503, **"5"** | 97/125 | 25/31 | 4 |
| Llama-3.1-8B-Instruct | 0.239 / 0.288, **"5"** | 125/125 | 30/31 | 4 |

What this means for the experiment:

- **The bare string `2+2=` is already contaminated.** Two of three models complete it with "5" before any edit (the "2+2=5" meme), while answering 4 in chat. Record the pre-edit state and report probability shifts, not just flips.
- **Add control edits without a cultural prior** (e.g. `3+4=` → 8, `7+2=` → 10) so results are not specific to the meme string. `generate.py` takes the target from the constants `A, B, TRUE, NEW`.
- **Llama-3.1-8B-Instruct is the cleanest subject** for bare `a+b=` prompts (all 125 numeric neighbours correct). The Qwen models often continue with "____" or a second equation, so use few-shot or chat formats for them.
- **Tokenisation differs.** Qwen emits one digit per token; Llama-3 has single tokens for 0–999. In all three, `2+2=` is four tokens and "4" one token.
- **Greedy string match under-counts** (models answer "four", or "?\nThe answer is 5"). Prefer the first-token metric or a restricted-candidate log-prob comparison.
- Some probes are unreliable at baseline (several word problems, `True or False` items on Llama). Score leakage only on probes the unedited model gets right, as RippleEdits does.

## 2. counterfact (user-specified)

```python
from datasets import load_dataset
d = load_dataset("azhx/counterfact"); d.save_to_disk("datasets/counterfact")
# load: from datasets import load_from_disk; d = load_from_disk("datasets/counterfact")
```
Fields: `case_id, pararel_idx, requested_rewrite{prompt, subject, target_new, target_true, relation_id}, paraphrase_prompts, neighborhood_prompts, attribute_prompts, generation_prompts`.
Sample: `{"prompt": "The mother tongue of {} is", "subject": "Danielle Darrieux", "target_new": "English", "target_true": "French"}`; see `counterfact/samples/samples.json`.

## 3. gsm8k
```python
load_dataset("openai/gsm8k", "main").save_to_disk("datasets/gsm8k")
```
Fields: `question, answer` (final answer after `####`).

## 4. triviaqa_nocontext_val2k
```python
t = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation[:2000]")
t = t.map(lambda x: {"answer_value": x["answer"]["value"], "answer_aliases": x["answer"]["aliases"]},
          remove_columns=[c for c in t.column_names if c not in ("question", "question_id")])
t.save_to_disk("datasets/triviaqa_nocontext_val2k")
```
Fields: `question, question_id, answer_value, answer_aliases`.

## 5. wikitext2_test
```python
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test").save_to_disk("datasets/wikitext2_test")
```
Field: `text`. Use for perplexity and for mean KL(pre-edit ‖ post-edit) next-token drift.
