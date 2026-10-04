# Notes: arithmetic mechanisms in LLMs (for the "2+2=5 isolation" project)

Three papers read in full (all chunks, main text + appendices):

1. Nikankin, Reusch, Mueller, Belinkov — "Arithmetic Without Algorithms: Language Models Solve Math with a Bag of Heuristics" (ICLR 2025, arXiv 2410.21272v2)
2. Kantamneni & Tegmark — "Language Models Use Trigonometry to Do Addition" (arXiv 2502.00873v1)
3. Zhang et al. — "Interpreting and Improving Large Language Models in Arithmetic Calculation" (ICML 2024, arXiv 2409.01659v1)

Reading caveat: every chunk was opened with the PDF reader, but for some chunks the rendered pages were dropped by a request-size limit and the text was re-read via pypdf text extraction instead (Nikankin chunks 3-8, Kantamneni chunks 5-7, Zhang chunks 1-3). For those chunks the body text, captions and tables were read, but figures were NOT visually inspected; numbers that exist only as plotted curves are marked "not stated in text".

Convention: "PAPER" = stated in the paper. "INFERENCE" = my extrapolation to the 2+2=5 question; none of the three papers performs model editing of a single arithmetic fact, so every leakage prediction below is an inference, not a reported result.

---

## 1. Nikankin et al. 2024/2025 — Bag of heuristics

### Research question
Do LLMs solve arithmetic with a robust algorithm, by memorising input-output pairs, or something in between? Answer claimed: neither; a "bag of heuristics" = many sparse MLP neurons, each firing on a simple numerical pattern of operands/result and nudging the logits of matching answer tokens.

### Models and tokenization (PAPER)
- Llama3-8B (main), Llama3-70B, Pythia-6.9B, GPT-J. Pre-trained, NOT fine-tuned on arithmetic.
- Numbers are single tokens up to a limit: Llama3-8B/70B limit 1000 ("numbers in [0, 1000]"), GPT-J and Pythia-6.9B limit 520 (App. G). (Kantamneni reports slightly different limits for GPT-J/Pythia, see paper 2.)
- Authors flag that conclusions may differ for single-digit-tokenizing models.

### Prompts / data (PAPER)
- Format: `op1 ∘ op2 =` with Arabic numerals, four tokens (op1, operator, op2, "="), no in-context examples (deliberately, unlike Stolfo et al.). Running example "226−68=".
- Operators +, −, ×, ÷ (÷ = integer division). Operands in [0, 300]; prompts with negative or multi-token results are dropped.
- Circuit discovery: 100 prompts per operator, plus 100 per operator for evaluation; only prompts the model completes correctly.
- Mean-ablation baselines computed over all prompts `op1∘op2=` for op1, op2 ∈ [0,300] (not only correct ones).
- Accuracy on these bare prompts (Table 3, single-token results, operands in [0,300]):

| Model | + | − | × | ÷ | Avg |
|---|---|---|---|---|---|
| Llama3-8B | 0.97 | 0.96 | 0.84 | 0.92 | 0.95 |
| Llama3-70B | 0.97 | 0.99 | 0.99 | 0.73 | 0.88 |
| Pythia-6.9B | 0.30 | 0.04 | 0.27 | 0.75 | 0.43 |
| GPT-J | 0.23 | 0.09 | 0.46 | 0.64 | 0.37 |

  (Averages are as printed in the paper. High ÷ accuracy for small models is attributed to half of legal prompts having answer "0".)

### Methods (PAPER)
- Activation patching of each MLP layer / attention head at each position with activations from a random counterfactual prompt p' with a different result r' (e.g. patch "21+17=" into "226−68="). Effect metric (Eq. 1): average of relative increase in P(r') and relative decrease in P(r). Using logits instead of probabilities gives the same top components.
- Faithfulness (Eq. 2): normalised correct-answer logit when mean-ablating everything outside the circuit, rescaled to [0,1].
- Linear probes f_{l,p}: R^d → R^1000 for the answer token at each layer/position (80/20 split, Adam, lr 3e-4, batch 32).
- Neuron-level activation patching inside MLPs; key-value view (Geva et al.): key = input-weight row, value = output-weight row; Logit Lens on value vectors.
- Heuristic classification (Alg. 1): build the neuron's 2D activation map over (op1, op2); for "direct" heuristics multiply element-wise by the logit-lens map of the value vector; take the top-k prompts (k = number of prompts associated with the candidate heuristic); neuron implements the heuristic if overlap/k ≥ t = 0.6. One neuron can match several heuristics; classification is per operator.
- Tooling: TransformerLens; single L40 48GB for 8B/GPT-J/Pythia.

### Circuit structure, Llama3-8B (PAPER)
- MLPs matter more than attention heads. MLP0 acts at operand/operator positions ("effective embedding"); mid/late MLPs act at the final position.
- Very few attention heads matter; each copies one position to the last token. General arithmetic heads: L16H21 (attends op1), L2H2 (attends operator), L15H13 (attends op2). Zeroing earlier attention into those source positions does not hurt, i.e. they carry only token-local information.
- Operator-specific head sets (Table 2):
  - `+`: L2H2, L5H3, L5H31, L14H12, L15H13, L16H21
  - `−`: L2H2, L13H21, L13H22, L14H12, L15H13, L16H21
  - `×`: 20 heads (L2H2, L5H30, L8H15, L9H26, L13H18, L13H21, L13H22, L14H12, L14H13, L15H8, L15H13, L15H14, L15H15, L16H3, L16H21, L17H24, L17H26, L18H16, L20H2, L22H1)
  - `÷`: L2H2, L5H31, L15H13, L15H14, L16H21, L18H16
- Faithfulness per operator (Table 1): + 0.97 (6 heads), − 0.98 (6), × 0.90 (20), ÷ 0.96 (6); 0.96 average for the combined circuit.
- Answer is linearly decodable only at the last position and only from layer 16 on. Layer ranges where the answer is extractable / neurons analysed: Llama3-8B [16,32], Llama3-70B [39,80], Pythia-6.9B [14,32], GPT-J [17,28].
- MLP0: as little as 1% of neurons suffices; example MLP0 neurons: 6206 (numbers near 170 or 17), 7101 (numbers > 100), 8969 (≡ 8 mod 10).

### Neuron-level results (PAPER)
- Only ~200 neurons per layer (~1.5%) in mid/late MLPs are needed for high faithfulness. 200 × 16 layers = 3,200 neurons per operator.
- Operator-specific neuron sets overlap only partly: pairwise IoU 52.3%-55.8% (avg 54%); + vs − 55.8%, + vs × 52.6%, + vs ÷ 52.3%, − vs × 53.0%, − vs ÷ 53.3%, × vs ÷ 55.8%. Text also says "roughly 45% of the important neurons for each operator are unique".
- 91% of the 3,200 top neurons per operator are classified into at least one heuristic type.
- Heuristic types (App. E):
  - Range (operand or result in [a,b]; lengths 10/30/50/100 for +,−,×; 2/10/100 for ÷)
  - Modulo (operand or result ≡ m mod n, n ∈ {2,3,4,5,6,7,8,9,11,13,15})
  - Pattern (3-digit regex on operand or result, e.g. "result ends in 5"; mod 10 and mod 100 live here)
  - Identical operands (op1 == op2; e.g. promotes "0" for subtraction)
  - Multi-result (division only; promotes a set of 2-4 unrelated results)
- Direct heuristics: activation depends on both operands and value vector encodes result tokens. Indirect heuristics: activation depends on one operand and value vector writes features for downstream use.
- Examples: L24 N12439 (op1−op2 ∈ [150,180]); L30 N1582 (op1−op2 ≡ 8 mod 10); L17 N4942 (op2 ∈ [5,25] mod 50); addition examples L31 N9208 (result ∈ [240,270]), L31 N2919 (result pattern .6./.64), L17 N1463 (op1 ≡ 2 mod 5, op2 ≡ 2 mod 5), L16 N10226 (op1 ∈ [0,50]/[0,100], op2 ∈ [0,100]); subtraction L30 N8806 (identical operands, top logits '0','000','00').
- Knockout by heuristic type (zero-ablation; 100 associated vs 100 unassociated correct prompts): mean post-ablation accuracy on associated / unassociated prompts: addition 0.56 / 0.86, subtraction 0.64 / 0.88, multiplication 0.80 / 0.89, division 0.78 / 0.91 (pairing read from extracted figure text; figure itself not visually inspected). Average drop on associated prompts: 29% out of 95% pre-ablation accuracy. Not a full drop because heuristics have imperfect recall and each prompt uses several unrelated heuristics.
- Knockout by prompt: ablating the top-scoring neurons of the heuristics associated with a prompt hurts accuracy much more than ablating the same number of neurons from unassociated heuristics (Llama3-8B and Llama3-70B; x-axis up to 80 and 160 neurons per layer; exact accuracies not stated in text).
- Failure analysis (50 correct vs 50 incorrect prompts): incorrect prompts have MORE associated heuristic neurons on average; failures are explained by lower total logit contribution to the correct answer (imperfect recall / weak value-vector logits).
- Training dynamics (Pythia-6.9B, checkpoints 23K to 143K steps, 10K intervals): heuristics appear gradually; neurons shared with the final checkpoint account on average for 79% of heuristic contribution at each checkpoint; ablating 25 heuristic neurons per layer gives near-zero accuracy at all checkpoints; no competing earlier mechanism found.

### Limitations (PAPER)
- Heuristic categories are human-defined; 9% of top neurons unclassified.
- Only models with multi-digit tokens; may differ for single-digit tokenizers.
- Only bare two-operand numeral prompts; no word-form numbers, no multi-step.
- Authors suggest post-hoc fixes such as activation steering may be insufficient to improve arithmetic.

### Code
https://github.com/technion-cs-nlp/llm-arithmetic-heuristics

### Relevance to the 2+2=5 isolation question (INFERENCE unless marked)
- No heuristic neuron is keyed on the single pair (2,2). The neurons that currently promote "4" for "2+2=" are shared with other prompts: result-range neurons cover an interval of results (minimum range length 10), result-modulo/pattern neurons cover every prompt with the same result residue or digit pattern, operand-range and operand-modulo neurons cover every prompt with op1 or op2 in the same class, and "identical operands" neurons fire for all a+a. So an edit that works by re-weighting existing heuristic neurons (fine-tuning mid/late MLPs, or rank-one edits to MLP value vectors in layers 16-32 of Llama3-8B) should be expected to leak to: other sums with result 4 (1+3, 3+1, 0+4), sums with nearby results (2+3, 1+2, 3+3), a+a prompts, and prompts sharing an operand class with 2.
- Because the answer is assembled from many small logit contributions, flipping 4 to 5 requires either changing many neurons (broad leakage) or adding one large new contribution that fires only on the exact (2, +, 2) conjunction. The latter is an added key-value memory, i.e. a string/operand-keyed exception layered over an unchanged bag of heuristics. This paper therefore supports the project hypothesis in mechanism terms, but does not test it.
- Cross-operator leakage: PAPER shows roughly half of the important neurons are shared across operators (IoU ~54%) and the three general heads are shared by all four operators. INFERENCE: an edit touching shared neurons can move 4−2, 2×2, 2÷2; an edit confined to addition-specific neurons should leak less across operators.
- 12+2 and 2+2+1: not covered by the paper (only two-operand prompts). INFERENCE: 12+2 shares op2 = 2 and operand-pattern classes ("ends in 2") with 2+2, so operand-keyed neurons are a leakage path; three-operand prompts are outside the studied circuit.
- Paraphrases ("two plus two"): not studied. The circuit is defined on numeral tokens routed by L16H21 / L15H13 / L2H2.
- What to target/monitor (Llama3-8B): last-position MLPs in layers 16-32 (top ~200 neurons per layer per operator); heads L2H2, L15H13, L16H21; MLP0 at operand positions. An edit at MLP0 or the token embedding of "2" is the worst case for isolation because that representation is copied unchanged into every prompt containing 2.
- Practical: bare `a+b=` prompts are only reliable on Llama3 (0.97); GPT-J and Pythia get 0.23 / 0.30 on addition with this format, so pre-edit baselines must be measured per prompt.

---

## 2. Kantamneni & Tegmark 2025 — Helix representation and the Clock algorithm

### Research question
How do mid-sized LLMs represent numbers and compute a+b? Claim: numbers are a generalised helix (one linear direction plus Fourier features with periods T = [2, 5, 10, 100]); addition is done by a "Clock" algorithm that builds helix(a+b) at the last token and reads it out to logits.

### Models and tokenization (PAPER)
- GPT-J (6B; main analysis because of its simple MLP), Pythia-6.9B, Llama3.1-8B (gated MLP).
- Single-token integers: GPT-J [0, 361] (with a leading space), Pythia-6.9B [0, 557], Llama3.1-8B [0, 999].
- GPT-J: 28 layers, 16 heads per layer (448 heads), residual 4096, MLP width 16384.

### Prompts / data (PAPER)
- All 10,000 problems a+b for a, b ∈ [0, 99].
- Bare `a+b=` gave poor results; instruction prompts used (Table 2):
  - GPT-J: `Output ONLY a number. {a}+{b}=` — 80.5%
  - Pythia-6.9B: `Output ONLY a number. {a}+{b}=` — 77.2%
  - Llama3.1-8B: `The following is a correct addition problem. \n{a}+{b}=` — 98.0%
- All models are worse for larger a, b.
- Other tasks on GPT-J (Table 3; same "Output ONLY a number." prefix): a−23 (a ∈ [23,99]) 89.61%; a//5 97.78%; a*1.5 (even a ∈ [0,98]) 80.00%; a modulo 2 95.56%; "x-a=0, x=" 95.56%.
- Patching uses only prompts the model answers correctly; averages over 100 clean/corrupted pairs; corrupted prompt is a'+b (or a+b' for the b token).

### Methods (PAPER)
- Fourier transform of layer-0 residuals for a ∈ [0,360]: sparse peaks at T = 2, 5, 10 (plus low-frequency T = 100). PC1 is linear in a for a ∈ [0,99] (R² = 0.902 in the figure) with a discontinuity at a = 100 (three-digit numbers live in a different space). T = 2 peak is sensitive to the range used (seen for 0-360, not 0-361).
- Helix fit: h = C·B(a), B(a) = [a, cos(2πa/T_i), sin(2πa/T_i)…]; PCA to 100 dims, linear regression, project back. Full helix = 9 parameters.
- Activation patching with the fitted helix substituted for the clean residual; logit difference LD = logit_patched(a+b) − logit_corrupted(a+b). Baselines: full layer patch, PCA with the same number of dims, circle (no linear term), polynomial.
- Path patching for direct effect (DE) vs total effect (TE); attribution patching for neurons; Logit Lens on neuron outputs. Tooling: nnsight; single RTX A6000 48GB; about two days to reproduce.

### Key results (PAPER)
- Max-over-layers LD for patching the a token (Table 1):

| Task | Full | PCA | Helix | Circle | Poly |
|---|---|---|---|---|---|
| a+b | 8.34 | 6.13 | 7.21 | 6.83 | 3.09 |
| a−23 | 7.45 | 6.16 | 7.05 | 6.52 | 2.93 |
| a//5 | 5.48 | 5.24 | 4.55 | 4.25 | 4.55 |
| a*1.5 | 7.86 | 6.65 | 5.16 | 4.85 | 4.84 |
| a mod 2 | 0.98 | 1.00 | 1.11 | 1.12 | 0.88 |
| x−a=0, x= | 6.70 | 4.77 | 4.56 | 4.34 | 2.98 |

  Helix beats PCA for addition and subtraction; underperforms PCA on a//5, a*1.5 and x−a=0, so there is extra structure beyond the helix for other tasks.
- Ablating the helix subspace from the a-token residual hurts about as much as ablating the whole layer. Train/test split over a values (80/20) still works; a helix fit on shuffled a is not causal. Sharp jump in fit quality between layer-0 input (raw embedding) and layer-1 input, consistent with layer 0 being needed for number processing.
- Last-token residual is well modelled by helix(a, b, a+b); the 9-parameter helix(a+b) alone beats a 27-dim PCA at some layers.
- Four-step algorithm in GPT-J:
  1. a and b are embedded as helices on their own tokens.
  2. A sparse set of attention heads, mostly layers 9-14, moves the a, b helices to the last token.
  3. MLPs 14-18 build helix(a+b) (mostly indirect effect), helped by a few heads.
  4. MLPs 19-27 and a few heads read helix(a+b) and write the answer logits.
- MLPs dominate direct effect: total MLP DE = 8.20 vs attention DE = 1.20.
- Attention heads: 17 heads give 80% of the effect of patching all 448; 20 heads used (83.9%). Of the 20: 11 "a,b heads" (layers 9-14, plus one each in layers 16 and 18; attend to operand tokens; feed MLPs), 5 "a+b heads" (layers 24-26 plus one in layer 19; attend to last token; write to logits), 4 "mixed heads" (layers 15-18). Named examples: L14H13 (a,b head), L18H10 (mixed), L24H10 (a+b head). Treating only 4 heads as mixed recovers almost 80% of the effect; 0 mixed still 70%.
- Circuit MLPs: 11 MLPs (layers 14-27 except 15, 24, 25) give 95% of the all-MLP effect.
- Neurons: about 1% of last-token neurons (k = 4587), with the rest mean-ablated, give 80% accuracy; nearly 75% of them sit in the 11 circuit MLPs. About 700 (693) neurons give 80% of the direct effect; 84% of top-DE neurons are after layer 18. Neuron preactivations are periodic in a, b, a+b with periods 2, 5, 10, 100; fitted helix-form preactivations recover roughly 75% of the performance of the real ones. Example neurons: L20N7741 (period 2 in a+b, TE 0.24), L21N13478 (periods 10 and 5), L22N12504 (period 100), L22N9339 (linear in a and b).
- Logit-lens of top neurons shows periodic boosting/suppression over answer tokens [0,198]; dominant period of final logits is 10.
- Errors: GPT-J is wrong 19.5% of the time; when it answers with a wrong number, −10 accounts for 45.7% and +10 for 27.9% (over 70% together). Not explained by carrying (chi-squared, p = 0.14). Explained by the period-10 structure in the read-out plus a bias toward smaller answers (negative slope of logits over answer value).
- Linear-only addition on a line with R² = 0.997 gets under 20% accuracy for answers below 100 versus over 80% for GPT-J; authors conjecture the helix is an error-correcting redundancy.
- Llama3.1-8B: helix on the a token is causal, but helix(a+b) at the last token is a weaker fit than in GPT-J/Pythia; authors suspect additional algorithms (gated MLPs) and note Nikankin et al.'s neurons look different.

### Limitations (PAPER)
- The exact computation turning helix(a,b) into helix(a+b) is not isolated (trig identities are hypothesised only).
- Clock may be one member of an ensemble; weaker in Llama3.1-8B.
- Single-token regime, a, b ∈ [0,99] only; three-digit numbers use a different representation; digit-level tokenizers (e.g. Gemma-2-9B) must use other mechanisms.
- Helix is not the full number representation for non-additive tasks.
- T = 2 Fourier detection is sensitive to sample range.

### Code
No code repository URL is given in the paper text I read.

### Relevance to the 2+2=5 isolation question (INFERENCE unless marked)
- The representation is continuous and shared: the same helix encodes "2" wherever it appears and the same MLPs 14-18 build helix(a+b) for all a, b. There is no per-fact storage slot for "2+2". Making the model treat 2+2 as 5 at the representational level means making helix(a+b) land on 5 for the input pair (2,2), and the paper gives no component that is specific to one pair.
- Edit at the operand representation (embedding / layer 0-1 output for token "2", e.g. shifting helix(2) toward helix(2.5)): leaks to every prompt containing 2 (2+3, 12+2 if "2" is the edited token, 4−2, 2×2). PAPER supports the reuse: the helix on the a token is causally relevant for a−23, a//5, a*1.5, a mod 2 and x−a=0.
- Edit at the builder stage (MLPs 14-18): neurons read periodic features of a and b with periods 2, 5, 10, 100. A change conditioned on "a ≡ 2 and b ≡ 2 (mod 10)" would also hit 12+2, 2+12, 22+32 etc.; a change conditioned on parity or mod 5 is broader still. Only the linear / T = 100 components distinguish 2 from 12 or 22, so mod-10 neighbours are the predicted leakage set.
- Edit at the read-out stage (MLPs 19-27, a+b heads in layers 24-26): these read helix(a+b). Remapping "helix value 4 → token 5" would change every sum equal to 4 (1+3, 0+4, 3+1), and, given the period-10 read-out, would be expected to perturb sums of 14, 24, …. This predicts result-keyed leakage, not operand-keyed leakage.
- The observed ±10 error structure suggests that small generic perturbations in this pathway show up first as off-by-10 errors, so an edit's side-effect audit should include checking for new ±10 errors across the whole a+b grid, not only neighbours of 2+2.
- Paraphrases: not studied. The mechanism starts from the helix on numeral tokens; whether "two" maps onto the same helix is not addressed.
- 2+2+1 and downstream reasoning: not studied (two-operand only).
- What to target/monitor (GPT-J): operand-token residuals at layers 0-1 (helix fit); heads in layers 9-14 (operand movers); MLPs 14-18 (builder); MLPs 19-27 and heads in 24-26 (read-out). A cheap, quantitative leak detector: fit helix(a+b) at the last token before the edit, then after the edit check whether the last-token representation for 2+2 decodes as 4 while the output says 5 (suggesting a late, string-keyed override) or decodes as 5 (suggesting the internal sum moved), and repeat on neighbour prompts.
- Practical: prompt format matters a lot (bare `a+b=` was "significantly" worse for these models); use the instruction prefixes above if GPT-J/Pythia are used.

---

## 3. Zhang et al. 2024 — Key heads/MLPs for calculation and precise fine-tuning

### Research question
Which components do LLMs use for two-operand arithmetic, what do they do, and can fine-tuning only those components improve math ability without harming other abilities?

### Models and tokenization (PAPER)
- LLaMA2-7B (32 layers × 32 heads = 1024 heads; main), LLaMA2-13B (40 layers × 40 heads), Mistral-7B (appendix).
- LLaMA2 tokenizes each digit separately ("42" → "4", "2"), so operands A, B are sampled from {1,…,9} and the measured quantity is the next single digit token.

### Prompts / data (PAPER)
- 10,000 addition samples from 36 templates (Fig. 8), in three families per operator:
  - Equation: `{A} + {B} = {C}`
  - Statement: `{A} plus {B} equals to {C}`, `The addition of {A} and {B} is {C}`, `The addition of {A} and {B} equals to {C}`
  - Question-Answer: `Q: How much is {A} plus {B}? A:`, `Q: What is {A} plus {B}? A:`, `Q: What is the result of {A} plus {B}? A:`, `Q: What is the sum of {A} and {B}? A:`
  - Analogous templates for minus/difference, times/product, over/ratio.
- Extra addition templates with different linguistic framing (Fig. 10): "time span" (e.g. "The <EVENT> <VERB> {A} years from the year <YYY>{B} to the year <YYY>{C}") and "object accumulation" word problems.
- Average probability of the correct token is 82%; only correctly predicted samples are used as reference data Xr.
- Counterfactual data Xc keep the grammar but remove the calculation logic: operator words replaced by "none", "nothing", "null", …; symbols replaced by "<", ">", "@", … (e.g. "42 nothing 34 is equal to", "3 < 5 =").
- Example model outputs (LLaMA2-7B): "3 + 5 =" → "8" 52.93%; "42 plus 34 is equal to 7" → "6" 96.34%.

### Methods (PAPER)
- Path patching (sender = each head or MLP at the END position, receiver = output logits): replace the sender's activation with its value on Xc, freeze other heads at Xr values, measure relative change in the logit of {C}. "Key head" = logit change beyond −5%.
- Knockout validation by mean ablation (replace with mean activation over Xc).
- Attention-pattern analysis of the END-token row over 1000 samples.
- MLP analysis: cosine similarity between MLP_in, or (MLP_out − MLP_in), and unembedding vectors W_U[A], W_U[B], W_U[C], and other number tokens.
- Precise SFT: fine-tune only W_Q, W_K, W_V, W_O of the selected heads (and optionally all parameters of selected MLP layers); lr 2e-5, batch 128, 2 epochs, warm-up 0.02, weight decay 0.1; gradient rescaled by H/h; 8 × A100 80GB; training data = union of the four math datasets' training sets.

### Key results (PAPER)
- Fewer than 5% of heads matter. Key heads in LLaMA2-7B start at layer 12 and concentrate in layers 12-17; patching 12.22 alone lowers the {C} logit by 14.0%.
- Top-10 key heads on their data (Table 4): (12,22), (13,11), (16,0), (15,26), (18,26), (18,24), (30,31), (14,27), (22,25), (11,8).
- Operand heads: 13.11, 12.22 (attend to A and B). Operator heads: 14.2, 11.8 (attend to "+", "-", "plus", "minus", "times", "over", "div"). Head 14.2 is shown attending to the operator word across all four operations.
- Knockout of top-k heads in effect order (Fig. 3), accuracy: k=0 100%, 1: 93.3%, 2: 61.7%, 3: 46.3%, 4: 48.5%, 5: 42.0%, 6: 40.6%, 7: 35.5%, 8: 30.8%, 9: 29.5%, 10: 30.2%. Random heads: 99.6% → 98.0% over the same range. Top-10 knockout summarised as −69% (Table 4), "around 70%" in the introduction.
- Transfer: the same heads matter for SVAMP (accuracy 34.7 → drop of 22.9 points, −66.0% relative), multi-digit integers ("281+135="), decimals ("4.2 plus 2.5 equals to"), and time-span sentences; head 13.11 still attends to the operands in these unseen formats.
- MLPs: MLPs 0-16 have about ±0.0% direct effect on the answer logit; after layer 17 about ±10.0%. MLP_in similarity to A and B is near zero for layers 0-12, rises sharply over layers 12-17, then decays over 17-31. (MLP_out − MLP_in) similarity to C rises from layer 17 while similarity to other numbers falls. (start, inflection, end) points: LLaMA2-7B (13, 18, 28), LLaMA2-13B (13, 18, 35), Mistral-7B (13, 20, 28).
- Case study "4 + 3 =": layers 17 and 19 have '3' and '4' on top; layers 20 and 21 have '6' and '1'; answer '7' emerges through layers 22/23/25/27; neighbours '6' and '8' also appear near the top at layer 23 and then decline. Authors describe this as coarse-to-fine: first regress to an embedding near the answer, then sharpen.
- Addition and subtraction share almost the same key heads (different magnitudes); multiplication and division have more key heads.
- Exponentiation (Table 3): key heads (11,8), (12,22), (13,11), (14,2), (15,15); knockout −66% for "{A} ∧ {B} =" and −62% for "{A} to the power of {B} equals"; the same five heads are found for the symbolic and the verbal template.
- GSM8K: 60% of key heads overlap with those found on the synthetic data; GSM8K top-10: (19,6), (11,8), (12,22), (14,31), (13,11), (22,25), (16,0), (21,17), (15,26), (29,5); knockout −65%. Knocking out only the 6 overlapping heads: −56% (synthetic) and −52% (GSM8K). Knocking out the 4 GSM8K-only heads: −2% on synthetic, −26% on GSM8K; those 4 attend to text tokens (e.g. (29,5) to ".", (19,6) to "GB").
- Specificity: knocking out the key heads changes MMLU-Humanities from 42.9% to 42.6% (−0.3).
- Precise SFT vs full SFT (Table 1):

| Model | Tuned params | GSM8K | AddSub | SingleEq | SVAMP | MMLU | CSQA |
|---|---|---|---|---|---|---|---|
| LLaMA2-7B | – | 14.6 | 30.5 | 65.4 | 34.7 | 46.0 | 59.8 |
| + Full SFT | 6.7B | 24.6 | 53.7 | 68.2 | 50.3 | 40.5 | 54.0 |
| + Precise SFT (32 heads) | 0.07B | 27.4 | 50.6 | 69.7 | 55.8 | 46.4 | 59.6 |
| LLaMA2-13B | – | 28.7 | 33.7 | 76.6 | 45.7 | 54.8 | 67.3 |
| + Full SFT | 13.0B | 44.6 | 62.2 | 79.8 | 62.8 | 50.2 | 62.0 |
| + Precise SFT | 0.08B | 46.3 | 61.1 | 82.2 | 66.6 | 55.0 | 67.2 |

- Ablation on what is tuned (LLaMA2-7B; GSM8K / MMLU): top-8 heads 25.4 / 45.1; top-16 26.5 / 45.8; top-32 27.4 / 46.4; top-48 27.4 / 46.4; top-64 27.3 / 45.5; top-32 heads + top-1 MLP 27.5 / 46.0; + top-2 MLPs 27.7 / 45.7; + top-3 MLPs 28.0 / 45.2; + top-6 MLPs 28.2 / 44.9; + all MLPs (4.396B params) 29.2 / 43.9. Adding MLPs helps math and hurts MMLU monotonically.

### Limitations (PAPER)
- Mainly LLaMA2; Mistral only in the appendix.
- Operands restricted to single digits 1-9 for the mechanistic analysis.
- Only "arithmetic calculation" is interpreted; word problems also need text-comprehension heads.
- MLPs are analysed only at layer level via cosine similarity with unembeddings; no neuron-level account; precise MLP tuning left to future work.
- The goal is improving arithmetic in general, not editing a single fact; no locality test at the level of individual sums.

### Code
No code repository URL is given in the paper (only a footnote pointing to huggingface.co for the models).

### Relevance to the 2+2=5 isolation question (INFERENCE unless marked)
- This is the only one of the three papers with a paraphrase result. PAPER: the same key heads serve "{A} + {B} =", "{A} plus {B} equals to", "The addition of {A} and {B} is", Q/A forms, word problems, decimals and multi-digit inputs, and the same five heads are found for "∧" and "to the power of". INFERENCE: an edit that changes how these components compute the sum (a change in what the model "treats as true" about 2+2) should generalise across those templates; conversely, an edit that changes only "2+2=" and not "What is 2 plus 2? A:" is evidence that it was implemented as a surface-string exception that bypasses the shared heads/MLPs. That contrast is a direct operationalisation of the project's hypothesis. Note the templates use numerals; spelled-out operands ("two plus two") are not tested.
- The key heads are operand/operator routers, not fact stores. PAPER: knocking them out damages arithmetic across formats and operations (addition and subtraction share almost the same heads). INFERENCE: tuning them to force 2+2=5 would be expected to disturb other sums and 4−2; they are poor edit targets for isolation and good monitoring targets (check their attention patterns and outputs are unchanged on neighbour prompts).
- Where the answer is written: MLPs from layer 17 on (LLaMA2-7B), with a coarse-to-fine profile in which numerically adjacent answers (6 and 8 around 7) are co-activated mid-way. INFERENCE: nudging the late-MLP output for 2+2 from 4 to 5 operates in a region where neighbouring results share representation, predicting leakage to prompts whose answers are 4 or 5 (1+3, 2+3, 1+4).
- Unrelated-behaviour baseline: PAPER shows general ability is insensitive to the key heads (MMLU-Humanities −0.3 on knockout; MMLU +0.4 after tuning 32 heads) while tuning MLPs or all parameters costs MMLU (−5.5 for full SFT; 46.4 → 43.9 as MLPs are added). INFERENCE: for a single-fact edit, attention-head-only updates should be the cheapest for unrelated behaviour, MLP updates the most likely to show collateral damage; MMLU/CSQA-style checks are a reasonable "unrelated behaviour" probe but are insensitive to arithmetic-neighbour leakage.
- Downstream reasoning: PAPER shows 6 of the 10 GSM8K key heads coincide with the synthetic-arithmetic key heads, so word problems reuse the same calculation components. INFERENCE: an edit inside those components would be expected to surface in word problems containing 2+2 ("Mary has 2 apples and gains 2…"); an edit keyed on the literal string "2+2=" would not.
- Digit-level tokenization matters for experiment design: on LLaMA2, "12+2" is three operand tokens and the answer "14" is two tokens, so the "single arithmetic fact" 2+2=4 is a single-token prediction, but multi-digit neighbours are not.

---

## Cross-paper synthesis for experiment design

### Agreed picture (PAPER, all three)
1. Operand/operator information is processed at its own position early (MLP0 / layer 0-1).
2. A sparse set of mid-layer attention heads copies operands and operator to the final token.
3. Mid-to-late MLPs at the final token compute and write the answer, and they dominate the direct effect.

| | Operand-moving heads | Answer-writing MLPs | Named components |
|---|---|---|---|
| Llama3-8B (Nikankin) | L2H2 (operator), L15H13 (op2), L16H21 (op1) + operator-specific | layers 16-32, ~200 neurons/layer | see Table 2 above |
| GPT-J (Kantamneni) | layers 9-14 (11 heads) | 14-18 build a+b; 19-27 read out | L14H13, L18H10, L24H10 |
| GPT-J (Nikankin) | not listed in text | layers 17-28 | – |
| Pythia-6.9B (Nikankin) | not listed in text | layers 14-32 | – |
| LLaMA2-7B (Zhang) | layers 12-17: 12.22, 13.11 (operands), 14.2, 11.8 (operators) | layer 17 onward (to ~28) | top-10 list above |

### Disagreement to keep in mind
Nikankin: sparse heuristic neurons (range / modulo / pattern), no single algorithm. Kantamneni: a coherent Clock algorithm over a helix, and explicitly notes the helix(a+b) fit is weaker for Llama3.1-8B, the model family Nikankin focuses on. The two are partly compatible (modulo and pattern heuristics are periodic features with periods such as 2, 5, 10; range heuristics resemble the linear / T=100 component), but neither paper demonstrates this equivalence.

### Shared implication for 2+2=5 (INFERENCE)
- None of the papers finds a component keyed to an individual operand pair. The existing mechanism is distributed over features of operand classes and result classes. So an edit that genuinely changes the computed sum must alter shared features, and predicted leakage follows feature sharing:
  - same result (1+3, 3+1, 0+4): read-out / result-range / result-pattern components;
  - adjacent result (2+3, 1+2): result-range neurons, coarse-to-fine neighbours;
  - same operand (2+3, 12+2, 4−2, 2×2): operand helix, operand-range/modulo neurons, shared operand heads;
  - same residues (12+2, 2+12, 22+2): mod-10 / "ends in 2" features;
  - a+a prompts: identical-operand heuristic;
  - other operators: ~half of important neurons and the general heads are shared (Nikankin); add/sub share key heads (Zhang); helix is causal for subtraction (Kantamneni);
  - other templates and word problems: shared heads (Zhang).
- Near-perfect isolation would require a new conjunction detector for (op1=2, operator=+, op2=2[, exact template]) that overrides the answer late. That is by construction an exception keyed on the input, not a change to the arithmetic mechanism. The papers make this hypothesis mechanistically plausible; they do not test it.

### Suggested probe sets grounded in the papers
- Full a+b grid (a, b ∈ [0,99], or [0,300] on Llama3) before/after the edit, reporting which cells change, and specifically ±10 errors (Kantamneni's dominant error mode).
- Same-result, same-operand, same-residue, identical-operand slices (Nikankin's heuristic classes).
- All four operators with operands 2,2 and with result 4.
- Zhang's 9 addition templates plus word-problem and time-span templates for paraphrase/generalisation; spelled-out numbers are an untested extension.
- Mechanistic monitors: answer linear probe by layer at the last token (Nikankin: decodable from layer 16 in Llama3-8B); helix(a+b) decode at the last token (Kantamneni); attention patterns of operand/operator heads (Zhang, Nikankin); per-layer (MLP_out − MLP_in)·W_U[4] vs W_U[5] (Zhang).
- Unrelated-behaviour check: MMLU / CSQA as in Zhang (full SFT cost roughly 5 points; head-only tuning roughly 0).

### Practical facts
- Tokenization: single-token numbers up to 1000 (Llama3 / Llama3.1), about 520 or 557 (Pythia), about 361 or 520 (GPT-J, the two papers differ), digit-level for LLaMA2 (and Gemma-2 per Kantamneni). "2", "4", "5" are single tokens in all of them.
- Prompt format: bare `a+b=` works on Llama3-8B (0.97 addition) but poorly on GPT-J (0.23) and Pythia (0.30) for operands up to 300; with "Output ONLY a number." and operands up to 99, GPT-J reaches 80.5% and Pythia 77.2%.
- Code: Nikankin https://github.com/technion-cs-nlp/llm-arithmetic-heuristics (TransformerLens). Kantamneni: no URL in paper (uses nnsight). Zhang: no URL in paper.
