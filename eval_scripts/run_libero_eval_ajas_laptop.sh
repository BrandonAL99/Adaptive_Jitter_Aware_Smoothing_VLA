#!/bin/bash
# AJAS C-value sweep on laptop/CPU — 3 eps per task, all 4 suites, 5 C values.
# Gives 3 x 10 x 4 x 5 = 600 episodes total.
# Usage: conda activate lerobot && bash run_libero_eval_ajas_laptop.sh
# Run from repo root: /home/brandon/Documents/sem_1_2026/thesis

set -eo pipefail
REPO_DIR="/home/brandon/Documents/sem_1_2026/thesis"
cd "$REPO_DIR"

MODEL="HuggingFaceVLA/smolvla_libero"
EPISODES=3
EXEC_STEPS=50
SEED=1000
AJAS_K_MIN=1
AJAS_K_MAX=40
LOG_DIR="results/logs/libero_eval_ajas_laptop_kmin1"
OUT_BASE="results/eval_ajas_laptop_kmin1"

mkdir -p "$LOG_DIR"

SUITES=("libero_object" "libero_spatial" "libero_goal" "libero_10")
C_VALUES=("500" "1000" "2000" "3000" "5000")

TOTAL=$(( ${#C_VALUES[@]} * ${#SUITES[@]} * 10 * EPISODES ))

echo "================================================"
echo " LIBERO Eval — AJAS C-value sweep (laptop)"
echo " Model:      $MODEL"
echo " Episodes:   $EPISODES per task x 10 tasks x ${#SUITES[@]} suites x ${#C_VALUES[@]} C values"
echo " Total eps:  $TOTAL"
echo " Exec steps: $EXEC_STEPS"
echo " Seed:       $SEED"
echo " C values:   ${C_VALUES[*]}"
echo " k_min=$AJAS_K_MIN  k_max=$AJAS_K_MAX"
echo " Note: --no-push, results are local only"
echo "================================================"
echo ""

OVERALL_START=$(date +%s)

for C in "${C_VALUES[@]}"; do
    TAG="c${C}"

    echo "############################################################"
    echo " C = $C"
    echo "############################################################"

    for SUITE in "${SUITES[@]}"; do
        OUT_DIR="${OUT_BASE}/${TAG}/${SUITE}"
        LOG_FILE="${LOG_DIR}/${TAG}_${SUITE}.log"

        echo "──────────────────────────────────────────────"
        echo " C=$C  Suite=$SUITE"
        echo " Start: $(date)"
        echo "──────────────────────────────────────────────"

        python3 simulation/scripts/eval_libero_processed.py \
            --model "$MODEL" \
            --task "$SUITE" \
            --episodes "$EPISODES" \
            --exec-steps "$EXEC_STEPS" \
            --seed "$SEED" \
            --processor ajas \
            --ajas-c "$C" \
            --ajas-k-min "$AJAS_K_MIN" \
            --ajas-k-max "$AJAS_K_MAX" \
            --device cpu \
            --output-dir "$OUT_DIR" \
            --no-push \
            2>&1 | tee "$LOG_FILE"

        if [ -f "${OUT_DIR}/eval_info.json" ]; then
            SR=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_info.json')); v=d.get('overall',{}).get('success_rate'); print(f'{v:.1%}' if isinstance(v,float) else 'N/A')")
            SM_RAW=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_info.json')); v=d.get('overall',{}).get('tc_raw',{}).get('sm_mean'); print(f'{v:.5f}' if isinstance(v,float) else 'N/A')")
            SM_PROC=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_info.json')); v=d.get('overall',{}).get('tc_proc',{}).get('sm_mean'); print(f'{v:.5f}' if isinstance(v,float) else 'N/A')")
            echo "  C=$C  $SUITE — SR=$SR  Sm_raw=$SM_RAW  Sm_proc=$SM_PROC"
        else
            echo "  WARNING: eval_info.json not found for C=$C $SUITE"
        fi
        echo ""
    done
done

OVERALL_END=$(date +%s)
ELAPSED=$(( OVERALL_END - OVERALL_START ))

echo "================================================"
echo " All done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo ""
echo " Summary (SR = success rate, Sm = weighted mean freq x1e5):"
echo ""
printf " %-6s  %-16s  %6s  %10s  %10s\n" "C" "Suite" "SR" "Sm_raw" "Sm_proc"
printf " %s\n" "--------------------------------------------------------------"
for C in "${C_VALUES[@]}"; do
    TAG="c${C}"
    for SUITE in "${SUITES[@]}"; do
        JSON="${OUT_BASE}/${TAG}/${SUITE}/eval_info.json"
        if [ -f "$JSON" ]; then
            SR=$(python3 -c "import json; d=json.load(open('$JSON')); v=d.get('overall',{}).get('success_rate'); print(f'{v:.1%}' if isinstance(v,float) else 'N/A')")
            SM_RAW=$(python3 -c "import json; d=json.load(open('$JSON')); v=d.get('overall',{}).get('tc_raw',{}).get('sm_mean'); print(f'{v:.5f}' if isinstance(v,float) else 'N/A')")
            SM_PROC=$(python3 -c "import json; d=json.load(open('$JSON')); v=d.get('overall',{}).get('tc_proc',{}).get('sm_mean'); print(f'{v:.5f}' if isinstance(v,float) else 'N/A')")
            printf " %-6s  %-16s  %6s  %10s  %10s\n" "$C" "$SUITE" "$SR" "$SM_RAW" "$SM_PROC"
        fi
    done
    echo ""
done
echo " Logs:    $LOG_DIR/"
echo " Results: $OUT_BASE/"
echo "================================================"
