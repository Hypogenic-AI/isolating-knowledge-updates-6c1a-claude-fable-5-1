#!/bin/bash
# Full pipeline, sequential on one 48 GB GPU (about 6.5 h). Finished runs are skipped, so it can be re-run to resume.
source .venv/bin/activate
export TORCH_DISABLE_NATIVE_JIT=1 HF_HOME=${HF_HOME:-/hf_cache}
MAIN=lora_exact,lora_exact@lr0.1,lora_anchor,lora_kl,lora_para,lora_para_kl,lora_deep_kl
DET=icl,gate_L4,gate_L12,gate_L20,gate_L28,rome_L5
EXTRA=ft_layer_L1,ft_layer_L4,ft_layer_L5,ft_layer_L8,ft_layer_L12,ft_layer_L16,ft_layer_L20,ft_layer_L24,ft_layer_L28,ft_layer_L31,ft_layer_L20@lr0.25,lora_exact@lr0.3,lora_exact@lr3,lora_kl0.1,lora_kl10,rome_last_L5
ARITH="2+2=5,3+4=8,7+5=13,2+2=7,9+6=14"
CAPS="France->Rome,Japan->Seoul"
python src/gen_sdf_docs.py 10 && python src/gen_sdf_ctrl.py          # needs OPENROUTER_KEY; skip if results/sdf_docs exists
python src/run_all.py --targets "2+2=5,3+4=8" --arms $DET,$MAIN --seeds 0 --gsm
python src/run_all.py --targets "$ARITH,$CAPS" --arms $DET --seeds 0
python src/run_all.py --targets "$ARITH,$CAPS" --arms $MAIN --seeds 0,1,2
python src/run_all.py --targets "$ARITH,$CAPS" --arms $EXTRA --seeds 0
python src/run_all.py --targets "2+2=5,3+4=8" --arms sdfslow --seeds 0,1 --gsm
python src/run_all.py --targets "2+2=5" --arms sdf,sdfctrl --seeds 0 --gsm
python src/mech.py "$ARITH,$CAPS"
python src/run_all.py --model Qwen/Qwen2.5-7B-Instruct --targets "2+2=5,3+4=8,2+2=7" --seeds 0 \
  --arms icl,gate_L20,rome_L5,ft_layer_L5,ft_layer_L20,lora_exact,lora_exact@lr0.1,lora_anchor,lora_kl,lora_para,lora_para_kl,lora_deep_kl
python src/analyze.py && python src/analyze.py Qwen2.5-7B-Instruct
python src/report_tables.py > results/report_tables.md
python src/report_tables.py Qwen2.5-7B-Instruct > results/report_tables_Qwen2.5-7B-Instruct.md
