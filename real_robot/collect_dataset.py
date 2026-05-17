"""
Collect 50 episodes per task (5 positions × 2 configs × 5 episodes each).
Uploads to HuggingFace after every batch of 5 to protect against crashes.

Usage:
    python my_scripts/so101_arm/collect_dataset.py --task pick_place
    python my_scripts/so101_arm/collect_dataset.py --task stacking
    python my_scripts/so101_arm/collect_dataset.py --task sorting
    python my_scripts/so101_arm/collect_dataset.py --task pick_place --start_batch 3
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lerobot" / "src"))
import my_scripts.so101_arm.config as config

HF_USER       = "BrandonAL"
EPISODES_PER_BATCH = 5   # episodes recorded per cube config
LEROBOT_ROOT  = Path(__file__).resolve().parents[2]

TASK_META = {
    "pick_place": {
        "repo":        f"{HF_USER}/my_svla_so101_pickplace_v3",
        "description": "Pick up the red cube and place it in the box.",
        "configs": [
            {"name": "centre / aligned",       "note": "Cube centred in front, flat face forward"},
            {"name": "centre / askew",          "note": "Cube centred in front, rotated ~45 deg"},
            {"name": "top_left / aligned",      "note": "Cube top-left of mat, flat face forward"},
            {"name": "top_left / askew",        "note": "Cube top-left of mat, rotated ~45 deg"},
            {"name": "bottom_left / aligned",   "note": "Cube bottom-left of mat, flat face forward"},
            {"name": "bottom_left / askew",     "note": "Cube bottom-left of mat, rotated ~45 deg"},
            {"name": "top_right / aligned",     "note": "Cube top-right of mat, flat face forward"},
            {"name": "top_right / askew",       "note": "Cube top-right of mat, rotated ~45 deg"},
            {"name": "bottom_right / aligned",  "note": "Cube bottom-right of mat, flat face forward"},
            {"name": "bottom_right / askew",    "note": "Cube bottom-right of mat, rotated ~45 deg"},
        ],
    },
    "stacking": {
        "repo":        f"{HF_USER}/my_svla_so101_stacking_v3",
        "description": "Place the red cube on top of the blue cube.",
        "configs": [
            {"name": "centre / red=left",       "note": "Both cubes centred, red on left"},
            {"name": "centre / red=right",      "note": "Both cubes centred, red on right"},
            {"name": "top_left / red=left",     "note": "Both cubes top-left, red on left"},
            {"name": "top_left / red=right",    "note": "Both cubes top-left, red on right"},
            {"name": "bottom_left / red=left",  "note": "Both cubes bottom-left, red on left"},
            {"name": "bottom_left / red=right", "note": "Both cubes bottom-left, red on right"},
            {"name": "top_right / red=left",    "note": "Both cubes top-right, red on left"},
            {"name": "top_right / red=right",   "note": "Both cubes top-right, red on right"},
            {"name": "bottom_right / red=left", "note": "Both cubes bottom-right, red on left"},
            {"name": "bottom_right / red=right","note": "Both cubes bottom-right, red on right"},
        ],
    },
    "sorting": {
        "repo":        f"{HF_USER}/my_svla_so101_sorting_v3",
        "description": "Place the red cube in the right box and the blue cube in the left box.",
        "configs": [
            {"name": "centre / red=left",       "note": "Both cubes centred, red on left"},
            {"name": "centre / red=right",      "note": "Both cubes centred, red on right"},
            {"name": "top_left / red=left",     "note": "Both cubes top-left, red on left"},
            {"name": "top_left / red=right",    "note": "Both cubes top-left, red on right"},
            {"name": "bottom_left / red=left",  "note": "Both cubes bottom-left, red on left"},
            {"name": "bottom_left / red=right", "note": "Both cubes bottom-left, red on right"},
            {"name": "top_right / red=left",    "note": "Both cubes top-right, red on left"},
            {"name": "top_right / red=right",   "note": "Both cubes top-right, red on right"},
            {"name": "bottom_right / red=left", "note": "Both cubes bottom-right, red on left"},
            {"name": "bottom_right / red=right","note": "Both cubes bottom-right, red on right"},
        ],
    },
}


def record_batch(repo_id, task_desc, num_episodes, resume):
    (follower_port, follower_name) = config.followerConfig()
    (leader_port,   leader_name)   = config.leaderConfig()
    cameras = config.camConfig()

    cmd = [
        sys.executable,
        "src/lerobot/scripts/lerobot_record.py",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={follower_name}",
        f"--robot.cameras={cameras}",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={leader_name}",
        "--display_data=true",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.num_episodes={num_episodes}",
        f'--dataset.single_task={task_desc}',
        "--dataset.push_to_hub=true",
        "--dataset.episode_time_s=60",
    ]
    if resume:
        cmd.append("--resume=true")

    subprocess.run(cmd, check=True, cwd=LEROBOT_ROOT / "lerobot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_META.keys()))
    parser.add_argument(
        "--start_batch", type=int, default=0,
        help="Batch index to resume from (0-9). Use this if a previous run crashed."
    )
    args = parser.parse_args()

    meta    = TASK_META[args.task]
    repo_id = meta["repo"]
    configs = meta["configs"]
    n_total = len(configs) * EPISODES_PER_BATCH

    print(f"\n{'='*60}")
    print(f"Task:       {args.task}")
    print(f"Dataset:    {repo_id}")
    print(f"Batches:    {len(configs)} configs × {EPISODES_PER_BATCH} episodes = {n_total} total")
    print(f"Starting at batch {args.start_batch + 1} / {len(configs)}")
    print(f"{'='*60}\n")

    for i, cfg in enumerate(configs):
        if i < args.start_batch:
            continue

        batch_num = i + 1
        print(f"\n--- Batch {batch_num}/{len(configs)}: {cfg['name']} ---")
        print(f"Setup:  {cfg['note']}")
        print(f"Record: {EPISODES_PER_BATCH} episodes then auto-push to HF")
        input("Press Enter when ready, Ctrl+C to abort...\n")

        resume = (i > 0) or (args.start_batch > 0 and i == args.start_batch)
        record_batch(
            repo_id=repo_id,
            task_desc=meta["description"],
            num_episodes=EPISODES_PER_BATCH,
            resume=resume,
        )

        completed = batch_num * EPISODES_PER_BATCH
        print(f"\nBatch {batch_num} done. {completed}/{n_total} episodes uploaded to HF.")
        if batch_num < len(configs):
            print(f"Next: {configs[i+1]['name']} — {configs[i+1]['note']}")

    print(f"\n{'='*60}")
    print(f"All {n_total} episodes collected for '{args.task}'!")
    print(f"Dataset: https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
