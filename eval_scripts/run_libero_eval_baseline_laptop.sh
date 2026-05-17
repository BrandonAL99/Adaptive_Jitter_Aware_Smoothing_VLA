#!/bin/bash
# Baseline (identity processor) eval on libero_object — laptop/CPU.
# Purpose: verify GPU and laptop produce comparable results under same seed.
#
# Matches GPU baseline config exactly:
#   processor=identity, episodes=10, exec_steps=50, seed=1000
#
# Usage (from repo root):
#   conda activate lerobot
#   bash run_libero_eval_baseline_laptop.sh

set -eo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

MODEL="HuggingFaceVLA/smolvla_libero"
EPISODES=10
EXEC_STEPS=50
SEED=1000
SUITE="libero_object"

OUT_DIR="results/eval_baseline_laptop/${SUITE}"
LOG_DIR="results/logs/libero_eval_baseline_laptop"
LOG_FILE="${LOG_DIR}/${SUITE}.log"

mkdir -p "$LOG_DIR"

echo "========================================================"
echo " LIBERO Eval — Baseline / Identity (laptop CPU)"
echo " Model      : $MODEL"
echo " Suite      : $SUITE"
echo " Episodes   : $EPISODES per task × 10 tasks = 100 eps"
echo " Exec steps : $EXEC_STEPS"
echo " Seed       : $SEED"
echo " Processor  : identity (no smoothing)"
echo " Device     : cpu"
echo " Output     : $OUT_DIR"
echo " Start      : $(date)"
echo "========================================================"
echo ""

python3 simulation/scripts/eval_libero_processed.py \
    --model "$MODEL" \
    --task "$SUITE" \
    --episodes "$EPISODES" \
    --exec-steps "$EXEC_STEPS" \
    --seed "$SEED" \
    --processor identity \
    --resume \
    --device cpu \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "$LOG_FILE"

if [ -f "${OUT_DIR}/eval_info.json" ]; then
    python3 -c "
import json
d  = json.load(open('${OUT_DIR}/eval_info.json'))
ov = d.get('overall', {})
sr      = ov.get('success_rate')
sm_raw  = ov.get('tc_raw',  {}).get('sm_mean')
sm_proc = ov.get('tc_proc', {}).get('sm_mean')
print()
print('======================================================')
print(f'  SR      : {sr:.1%}' if isinstance(sr, float) else '  SR      : N/A')
print(f'  Sm_raw  : {sm_raw:.5f}' if isinstance(sm_raw, float) else '  Sm_raw  : N/A')
print(f'  Sm_proc : {sm_proc:.5f}' if isinstance(sm_proc, float) else '  Sm_proc : N/A')
print('======================================================')
"
fi
