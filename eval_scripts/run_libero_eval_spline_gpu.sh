#!/bin/bash
# AJASSpline eval across all 4 LIBERO suites — cloud GPU.
#
# Saves to results/eval_spline_gpu/ (separate from laptop results/eval_spline/).
# Config matches laptop spline eval exactly for fair GPU-vs-GPU comparison.
#
# Config: C=100, sm_win=15, kf_win=2, k_min=1, k_max=40, skip gripper
# 10 eps/task × 10 tasks × 4 suites = 400 episodes total
# Seed matches all other evals: 1000
#
# Expected runtime: ~27 sec/ep → ~3 hrs on RTX 4090
#
# Usage (from repo root):
#   MUJOCO_GL=osmesa nohup bash run_libero_eval_spline_gpu.sh \
#       > results/logs/libero_eval_spline_gpu/main.log 2>&1 &

set -eo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

MODEL="HuggingFaceVLA/smolvla_libero"
EPISODES=10
EXEC_STEPS=50
SEED=1000

SPLINE_C=100
SPLINE_K_MIN=1
SPLINE_K_MAX=40
SPLINE_SM_WIN=15
SPLINE_KF_WIN=2

LOG_DIR="results/logs/libero_eval_spline_gpu"
OUT_BASE="results/eval_spline_gpu"

mkdir -p "$LOG_DIR"

SUITES=("libero_object" "libero_spatial" "libero_goal" "libero_10")

TOTAL_EPS=$(( ${#SUITES[@]} * 10 * EPISODES ))

echo "========================================================"
echo " LIBERO Eval — AJASSpline (cloud GPU)"
echo " Model      : $MODEL"
echo " Episodes   : $EPISODES per task × 10 tasks × ${#SUITES[@]} suites"
echo " Total eps  : $TOTAL_EPS  (~$(( TOTAL_EPS * 27 / 3600 )) hrs at 27 sec/ep)"
echo " Exec steps : $EXEC_STEPS"
echo " Seed       : $SEED"
echo " C=$SPLINE_C  k_min=$SPLINE_K_MIN  k_max=$SPLINE_K_MAX"
echo " sm_window=$SPLINE_SM_WIN  kf_window=$SPLINE_KF_WIN"
echo " Gripper (J7): unsmoothed"
echo " Device     : cuda"
echo " Output     : $OUT_BASE"
echo " Start      : $(date)"
echo "========================================================"
echo ""

OVERALL_START=$(date +%s)

for SUITE in "${SUITES[@]}"; do
    HF_DATASET="BrandonAL/eval_smolvla_spline_gpu_c${SPLINE_C}_${SUITE}"
    OUT_DIR="${OUT_BASE}/${SUITE}"
    LOG_FILE="${LOG_DIR}/${SUITE}.log"

    # Skip only if fully complete (overall key present)
    if python3 -c "import json,sys; d=json.load(open('${OUT_DIR}/eval_info.json')); sys.exit(0 if d.get('overall',{}).get('n_tasks') else 1)" 2>/dev/null; then
        echo "  [skip] $SUITE — already fully complete"
        continue
    fi

    echo "────────────────────────────────────────────────────────"
    echo " Suite  : $SUITE"
    echo " HF     : $HF_DATASET"
    echo " Start  : $(date)"
    echo "────────────────────────────────────────────────────────"

    python3 simulation/scripts/eval_libero_processed.py \
        --model "$MODEL" \
        --task "$SUITE" \
        --episodes "$EPISODES" \
        --exec-steps "$EXEC_STEPS" \
        --seed "$SEED" \
        --processor ajas_spline \
        --spline-c "$SPLINE_C" \
        --spline-k-min "$SPLINE_K_MIN" \
        --spline-k-max "$SPLINE_K_MAX" \
        --spline-sm-window "$SPLINE_SM_WIN" \
        --spline-kf-window "$SPLINE_KF_WIN" \
        --spline-skip-gripper \
        --resume \
        --device cuda \
        --output-dir "$OUT_DIR" \
        --hf-dataset "$HF_DATASET" \
        2>&1 | tee "$LOG_FILE"

    if [ -f "${OUT_DIR}/eval_info.json" ]; then
        SR=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_info.json')); v=d.get('overall',{}).get('success_rate'); print(f'{v:.1%}' if isinstance(v,float) else 'N/A')")
        SM_RAW=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_info.json')); v=d.get('overall',{}).get('tc_raw',{}).get('sm_mean'); print(f'{v:.5f}' if isinstance(v,float) else 'N/A')")
        SM_PROC=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_info.json')); v=d.get('overall',{}).get('tc_proc',{}).get('sm_mean'); print(f'{v:.5f}' if isinstance(v,float) else 'N/A')")
        echo "  ✓ $SUITE — SR=$SR  Sm_raw=$SM_RAW  Sm_proc=$SM_PROC"
    else
        echo "  WARNING: eval_info.json missing for $SUITE"
    fi
    echo ""
done

OVERALL_END=$(date +%s)
ELAPSED=$(( OVERALL_END - OVERALL_START ))

echo "========================================================"
echo " All done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo " Finished: $(date)"
echo ""
printf " %-18s  %7s  %10s  %10s  %7s\n" "Suite" "SR" "Sm_raw" "Sm_proc" "SER"
printf " %s\n" "--------------------------------------------------------------"
for SUITE in "${SUITES[@]}"; do
    JSON="${OUT_BASE}/${SUITE}/eval_info.json"
    [ -f "$JSON" ] || { printf " %-18s  MISSING\n" "$SUITE"; continue; }
    python3 -c "
import json
d=json.load(open('$JSON')); ov=d.get('overall',{})
sr=ov.get('success_rate'); smr=ov.get('tc_raw',{}).get('sm_mean'); smp=ov.get('tc_proc',{}).get('sm_mean')
ser=smp/smr if smr and smp else None
print(f' {\"$SUITE\":<18}  {sr:.1%}  {smr:.5f}  {smp:.5f}  {ser:.4f}')
"
done
echo ""
echo " Logs    : $LOG_DIR/"
echo " Results : $OUT_BASE/"
echo " HF      : https://huggingface.co/BrandonAL"
echo "========================================================"
