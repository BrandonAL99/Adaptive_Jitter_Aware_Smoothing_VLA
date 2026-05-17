"""
Convert action and observation.state columns in a LeRobot dataset from radians to degrees.

Edits parquet files in-place and updates meta/stats.json.
Optionally pushes the result to HuggingFace Hub.

Usage:
    python my_scripts/dataset_scripts/convert_rad_to_deg.py \
        --dataset-path results/datasets/so101_sim_pick_place_v1 \
        [--push-to-hub BrandonAL/so101_sim_pick_place_v1_deg]
"""

import argparse
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

RAD_TO_DEG = 180.0 / np.pi
CONVERT_COLS = {"action", "observation.state"}


def scale_stats_field(field_stats: dict, factor: float) -> dict:
    """Multiply every numeric list value in a stats dict by factor."""
    out = {}
    for key, val in field_stats.items():
        if key == "count":
            out[key] = val
        elif isinstance(val, list):
            out[key] = [v * factor for v in val]
        else:
            out[key] = val
    return out


def convert_parquet(path: Path) -> None:
    tbl = pq.read_table(path)
    arrays = {}
    for col in tbl.column_names:
        col_data = tbl[col]
        if col in CONVERT_COLS:
            # Each value is a fixed-size list of floats
            np_vals = np.array([row.as_py() for row in col_data], dtype=np.float32)
            np_vals = (np_vals * RAD_TO_DEG).astype(np.float32)
            # Rebuild as list-array matching original schema
            arrays[col] = pa.array(np_vals.tolist(), type=col_data.type)
        else:
            arrays[col] = col_data
    new_tbl = pa.table(arrays, schema=tbl.schema)
    pq.write_table(new_tbl, path)


def convert_stats(stats_path: Path) -> None:
    with open(stats_path) as f:
        stats = json.load(f)
    for col in CONVERT_COLS:
        if col in stats:
            stats[col] = scale_stats_field(stats[col], RAD_TO_DEG)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="Path to local LeRobot dataset root")
    parser.add_argument("--push-to-hub", default=None, help="HF repo id to push to, e.g. BrandonAL/my_dataset")
    args = parser.parse_args()

    root = Path(args.dataset_path)
    assert root.exists(), f"Dataset path not found: {root}"

    # Convert all parquet files
    parquet_files = sorted(root.glob("data/**/*.parquet"))
    print(f"Found {len(parquet_files)} parquet file(s) in {root}/data/")
    for p in parquet_files:
        print(f"  Converting {p.name} ...", end=" ")
        convert_parquet(p)
        print("done")

    # Update stats.json
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        print(f"Updating {stats_path} ...", end=" ")
        convert_stats(stats_path)
        print("done")
    else:
        print(f"Warning: {stats_path} not found — skipping stats update")

    # Spot-check
    first = parquet_files[0]
    tbl = pq.read_table(first)
    sample_action = np.array(tbl["action"][0].as_py())
    sample_state  = np.array(tbl["observation.state"][0].as_py())
    print(f"\nSpot-check first frame of {first.name}:")
    print(f"  action : {np.round(sample_action, 2)}")
    print(f"  state  : {np.round(sample_state,  2)}")

    if args.push_to_hub:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lerobot" / "src"))
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"\nPushing to {args.push_to_hub} ...")
        api.create_repo(repo_id=args.push_to_hub, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(root),
            repo_id=args.push_to_hub,
            repo_type="dataset",
        )
        print(f"Done — https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
