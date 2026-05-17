"""
Merge multiple LeRobot datasets into one and push to HuggingFace.

Usage:
    conda run -n lerobot python my_scripts/dataset_scripts/merge_datasets.py
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lerobot" / "src"))

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)

HF_USER = "BrandonAL"

# Source datasets to merge
SOURCE_REPOS = [
    f"{HF_USER}/my_svla_so101_all",       # original 150 eps (50 pickplace + 50 stacking + 50 sorting)
    f"{HF_USER}/my_so101_pickplace_1505",  # new 50 pickplace eps
    f"{HF_USER}/my_so101_stacking_1505",   # new 50 stacking eps
]

OUTPUT_REPO = f"{HF_USER}/my_so101_all_1505"

if __name__ == "__main__":
    print(f"Merging {len(SOURCE_REPOS)} datasets into {OUTPUT_REPO}")
    for r in SOURCE_REPOS:
        print(f"  - {r}")

    aggregate_datasets(
        repo_ids=SOURCE_REPOS,
        aggr_repo_id=OUTPUT_REPO,
    )

    print(f"\nMerge complete. Loading to verify...")
    ds = LeRobotDataset(OUTPUT_REPO)
    print(f"  Episodes: {ds.num_episodes}")
    print(f"  Frames:   {ds.num_frames}")

    print(f"\nPushing to HuggingFace as {OUTPUT_REPO}...")
    ds.push_to_hub()
    print(f"Done: https://huggingface.co/datasets/{OUTPUT_REPO}")
