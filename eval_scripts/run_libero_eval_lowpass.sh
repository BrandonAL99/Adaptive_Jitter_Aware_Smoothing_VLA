#!/bin/bash
# Run LIBERO lowpass filter evals (2Hz, 4Hz, 6Hz) across all 4 suites.
# Usage: conda activate lerobot && bash run_libero_eval_lowpass.sh
# Run from repo root: /home/brandon/Documents/sem_1_2026/thesis

set -eo pipefail
REPO_DIR="/home/brandon/Documents/sem_1_2026/thesis"
cd "$REPO_DIR"

MODEL="HuggingFaceVLA/smolvla_libero"
EPISODES=10
EXEC_STEPS=50
SEED=1000
ORDER=2
LOG_DIR="results/logs/libero_eval_lowpass"
OUT_BASE="results/eval_lowpass"

mkdir -p "$LOG_DIR"

SUITES=("libero_object" "libero_spatial" "libero_goal" "libero_10")
CUTOFFS=("2.0" "4.0" "6.0")

echo "================================================"
echo " LIBERO Eval — lowpass filter sweep (laptop/CPU)"
echo " Model:      $MODEL"
echo " Episodes:   $EPISODES per task x 10 tasks = 100 eps per suite"
echo " Exec steps: $EXEC_STEPS (full action chunk)"
echo " Seed:       $SEED"
echo " Cutoffs:    ${CUTOFFS[*]} Hz  (order=$ORDER)"
echo " Total eps:  $(( ${#SUITES[@]} * ${#CUTOFFS[@]} * 10 * EPISODES ))"
echo " Log dir:    $LOG_DIR"
echo "================================================"
echo ""

OVERALL_START=$(date +%s)

for CUTOFF in "${CUTOFFS[@]}"; do
    # e.g. 2.0 -> 2hz for naming
    TAG="${CUTOFF%.*}hz"

    echo "############################################################"
    echo " Cutoff: ${CUTOFF} Hz"
    echo "############################################################"

    for SUITE in "${SUITES[@]}"; do
        HF_DATASET="BrandonAL/eval_smolvla_chunk50_lowpass_${TAG}_${SUITE}"
        OUT_DIR="${OUT_BASE}/${TAG}/${SUITE}"
        LOG_FILE="${LOG_DIR}/${TAG}_${SUITE}.log"

        echo "──────────────────────────────────────────────"
        echo " Suite:  $SUITE"
        echo " Filter: lowpass ${CUTOFF}Hz order=${ORDER}"
        echo " HF:     $HF_DATASET"
        echo " Start:  $(date)"
        echo "──────────────────────────────────────────────"

        MUJOCO_GL=osmesa python3 simulation/scripts/eval_libero_processed.py \
            --model "$MODEL" \
            --task "$SUITE" \
            --episodes "$EPISODES" \
            --exec-steps "$EXEC_STEPS" \
            --seed "$SEED" \
            --processor lowpass \
            --lowpass-cutoff "$CUTOFF" \
            --lowpass-order "$ORDER" \
            --device cpu \
            --output-dir "$OUT_DIR" \
            --hf-dataset "$HF_DATASET" \
            2>&1 | tee "$LOG_FILE"

        if [ -f "${OUT_DIR}/eval_info.json" ]; then
            SR=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_info.json')); v=d.get('overall',{}).get('success_rate'); print(f'{v:.1%}' if isinstance(v,float) else 'N/A (partial)')")
            SM=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_info.json')); v=d.get('overall',{}).get('tc_proc',{}).get('sm_mean'); print(f'{v:.5f}' if isinstance(v,float) else 'N/A')")
            SER=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_info.json')); v=d.get('overall',{}).get('tc_proc',{}).get('ser_mean'); print(f'{v:.4f}' if isinstance(v,float) else 'N/A')")
            echo ""
            echo "  Done: $SUITE @ ${CUTOFF}Hz — SR=$SR  Sm=$SM  SER=$SER"
        else
            echo "  WARNING: eval_info.json not found for $SUITE @ ${CUTOFF}Hz"
        fi
        echo ""
    done
done

OVERALL_END=$(date +%s)
ELAPSED=$(( OVERALL_END - OVERALL_START ))

echo "================================================"
echo " All done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo ""
echo " Results summary:"
for CUTOFF in "${CUTOFFS[@]}"; do
    TAG="${CUTOFF%.*}hz"
    echo ""
    echo " Cutoff ${CUTOFF}Hz:"
    for SUITE in "${SUITES[@]}"; do
        JSON="${OUT_BASE}/${TAG}/${SUITE}/eval_info.json"
        if [ -f "$JSON" ]; then
            SR=$(python3 -c "import json; d=json.load(open('$JSON')); v=d.get('overall',{}).get('success_rate'); print(f'{v:.1%}' if isinstance(v,float) else 'N/A (partial)')")
            SM=$(python3 -c "import json; d=json.load(open('$JSON')); v=d.get('overall',{}).get('tc_proc',{}).get('sm_mean'); print(f'{v:.5f}' if isinstance(v,float) else 'N/A')")
            SER=$(python3 -c "import json; d=json.load(open('$JSON')); v=d.get('overall',{}).get('tc_proc',{}).get('ser_mean'); print(f'{v:.4f}' if isinstance(v,float) else 'N/A')")
            echo "   $SUITE: SR=$SR  Sm=$SM  SER=$SER"
        fi
    done
done
echo ""
echo " Logs:    $LOG_DIR/"
echo " Results: $OUT_BASE/"
echo " HF:      https://huggingface.co/BrandonAL"
echo "================================================"
