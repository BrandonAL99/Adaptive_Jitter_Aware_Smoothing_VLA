import sys
import subprocess
from pathlib import Path


# Params
BASE_MODEL      = "BrandonAL/my_smolvla_all"
DATASET         = "BrandonAL/my_so101_all_1505"
OUTPUT_MODEL    = "BrandonAL/my_smolvla_all_1505"
BATCH_SIZE      = 64
STEPS           = 50000
OUTPUT_DIR      = "outputs/train/my_smolvla_1505"
JOB_NAME        = "my_smolvla_1505"
CHECK_POINT_FREQ = 5000


LEROBOT_ROOT = Path(__file__).resolve().parents[1] / "lerobot"

# lerobot commans
cmd = [
    sys.executable, "-m",
    "lerobot.scripts.lerobot_train",
    f"--policy.path={BASE_MODEL}",
    f"--dataset.repo_id={DATASET}",
    f"--policy.repo_id={OUTPUT_MODEL}",
    f"--batch_size={BATCH_SIZE}",
    f"--steps={STEPS}",
    f"--output_dir={OUTPUT_DIR}",
    f"--job_name={JOB_NAME}",
    f"--save_freq={CHECK_POINT_FREQ}",
    "--policy.device=cuda",
    "--wandb.enable=false",
    "--policy.push_to_hub=true",
    "--dataset.video_backend=pyav",
]

if __name__ == "__main__":
    print(f"Starting training run:")
    print(f"  Base model:   {BASE_MODEL}")
    print(f"  Dataset:      {DATASET}")
    print(f"  Output model: {OUTPUT_MODEL}")
    print(f"  Steps:        {STEPS}")
    subprocess.run(cmd, check=True, cwd=LEROBOT_ROOT)