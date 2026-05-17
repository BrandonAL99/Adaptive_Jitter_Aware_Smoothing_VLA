#!/bin/bash
# Train my_smolvla_all_1505 on real-world 250-ep merged dataset.
# Fine-tunes BrandonAL/my_smolvla_all for 50k steps.
#
# Usage on a fresh vast.ai instance:
#   bash <(curl -s https://raw.githubusercontent.com/BrandonAL99/VLA_GENG4411_23106299/main/scripts/run_real_1505.sh)

set -e

REPO_URL="https://github.com/BrandonAL99/VLA_GENG4411_23106299"
REPO_DIR="/workspace/VLA_GENG4411_23106299"

# ── 1. HuggingFace login ──────────────────────────────────────────────────────
echo "=== HuggingFace login ==="
huggingface-cli login

# ── 2. Clone repo if not present, else pull latest ───────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "=== Cloning repo ==="
    GIT_LFS_SKIP_SMUDGE=1 git clone "$REPO_URL" "$REPO_DIR"
else
    echo "=== Pulling latest code ==="
    cd "$REPO_DIR"
    GIT_LFS_SKIP_SMUDGE=1 git pull origin main
fi

cd "$REPO_DIR"

# ── 3. Set up lerobot ────────────────────────────────────────────────────────
if [ ! -f lerobot/pyproject.toml ]; then
    echo "=== Cloning lerobot ==="
    rm -rf lerobot
    git clone https://github.com/huggingface/lerobot.git lerobot
fi

echo "=== Installing lerobot[smolvla] ==="
cd "$REPO_DIR/lerobot"
pip install -e ".[smolvla]" -q
cd "$REPO_DIR"

# ── 4. Check GPU ──────────────────────────────────────────────────────────────
echo "=== GPU check ==="
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

# ── 5. Launch training ────────────────────────────────────────────────────────
mkdir -p results/logs

LOG="results/logs/train_real_1505_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=== Launching training in background ==="
echo "  Base model : BrandonAL/my_smolvla_all"
echo "  Dataset    : BrandonAL/my_so101_all_1505  (250 eps)"
echo "  Output     : BrandonAL/my_smolvla_all_1505"
echo "  Steps      : 50000"
echo "  Log        : $LOG"
echo ""

nohup python "$REPO_DIR/my_scripts/train.py" > "$LOG" 2>&1 &
TRAIN_PID=$!

echo "Training PID: $TRAIN_PID"
echo "Safe to close terminal."
echo ""
echo "Monitor: tail -f $LOG"
