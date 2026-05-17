"""Training launcher for sim pick-place dataset."""
import sys
import subprocess
from pathlib import Path

BASE_MODEL       = "BrandonAL/my_sim_smolvla_all_step010000"
DATASET          = "BrandonAL/my_sim_merged"
OUTPUT_MODEL     = "BrandonAL/my_sim_smolvla_all_ft60k"
BATCH_SIZE       = 64
STEPS            = 50000
OUTPUT_DIR       = "outputs/train/my_sim_smolvla_all_ft60k"
JOB_NAME         = "my_smolvla_all_ft60k"
CHECK_POINT_FREQ = 10000

LEROBOT_ROOT = Path(__file__).resolve().parents[1] / "lerobot"

cmd = [
    sys.executable,
    "src/lerobot/scripts/lerobot_train.py",
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
    "--dataset.image_transforms.enable=true",
    "--dataset.video_backend=pyav",
]

if __name__ == "__main__":
    print("Starting sim training:")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Dataset    : {DATASET}")
    print(f"  Steps      : {STEPS}")
    print(f"  Checkpoints: every {CHECK_POINT_FREQ} steps → {OUTPUT_DIR}/checkpoints/")
    subprocess.run(cmd, check=True, cwd=LEROBOT_ROOT)
