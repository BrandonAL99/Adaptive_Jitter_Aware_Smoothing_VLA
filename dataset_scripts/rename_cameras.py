"""
Rename camera keys in a LeRobot dataset.

Renames:
  observation.images.top   -> observation.images.camera1
  observation.images.wrist -> observation.images.camera2

Touches:
  - data/**/*.parquet  (column renames)
  - videos/            (directory renames)
  - meta/stats.json    (key renames)
  - meta/info.json     (key renames)

Usage:
    python my_scripts/dataset_scripts/rename_cameras.py \
        --dataset-path results/datasets/so101_sim_pick_place_v1
"""

import argparse
import json
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

RENAME = {
    "observation.images.top":   "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


def rename_col(name: str) -> str:
    """Rename a column, including embedded camera keys like 'videos/observation.images.top/foo'."""
    for old, new in RENAME.items():
        name = name.replace(old, new)
    return name


def rename_parquet_columns(path: Path) -> None:
    tbl = pq.read_table(path)
    new_names = [rename_col(c) for c in tbl.column_names]
    renamed = tbl.rename_columns(new_names)
    pq.write_table(renamed, path)


def rename_json_keys(data):
    """Recursively rename top-level keys in a dict."""
    if isinstance(data, dict):
        return {RENAME.get(k, k): rename_json_keys(v) for k, v in data.items()}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    args = parser.parse_args()

    root = Path(args.dataset_path)
    assert root.exists(), f"Not found: {root}"

    # ── parquet columns (data + episodes metadata) ────────────────────────────
    parquet_files = sorted(root.glob("**/*.parquet"))
    print(f"Renaming columns in {len(parquet_files)} parquet file(s) ...")
    for p in parquet_files:
        rename_parquet_columns(p)
        print(f"  {p.name} done")

    # ── video directories ──────────────────────────────────────────────────────
    videos_root = root / "videos"
    for old_name, new_name in RENAME.items():
        old_dir = videos_root / old_name
        new_dir = videos_root / new_name
        if old_dir.exists():
            print(f"Renaming videos/{old_name} -> videos/{new_name} ...")
            shutil.move(str(old_dir), str(new_dir))
        elif new_dir.exists():
            print(f"videos/{new_name} already exists — skipping rename")
        else:
            print(f"Warning: videos/{old_name} not found")

    # ── stats.json ─────────────────────────────────────────────────────────────
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        stats = rename_json_keys(stats)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print("Updated meta/stats.json")

    # ── info.json ──────────────────────────────────────────────────────────────
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        if "features" in info:
            info["features"] = rename_json_keys(info["features"])
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print("Updated meta/info.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
