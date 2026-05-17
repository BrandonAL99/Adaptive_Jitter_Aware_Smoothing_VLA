"""
Load an eval dataset from HuggingFace and compute per-episode TC metrics.

The dataset schema (from eval_libero_processed.py):
  - episode_index   : int
  - frame_index     : int
  - action          : float32[7]  — processed actions (proc)
  - action_raw      : float32[7]  — raw VLA output
  - success         : bool        — same value repeated for all frames in episode
  - done            : bool        — True at last frame

Usage
-----
    # Single suite split
    python libero_analysis/scripts/hf_episode_metrics.py BrandonAL/eval_smolvla_chunk50 \
        --split libero_object

    # All splits in a dataset
    python libero_analysis/scripts/hf_episode_metrics.py BrandonAL/eval_smolvla_lp2hz

    # Override episodes-per-task if not 10
    python libero_analysis/scripts/hf_episode_metrics.py BrandonAL/my_dataset \
        --episodes-per-task 3

    # Save to CSV
    python libero_analysis/scripts/hf_episode_metrics.py BrandonAL/eval_smolvla_chunk50 \
        --split libero_object --save chunk50_object.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DT = 1.0 / 30.0


# ── metrics ─────────────────────────────────────────────────────────────────

def sm(actions):
    fs = 1.0 / DT
    scores = []
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        F = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=DT)
        scores.append(float((2.0 / (len(freqs) * fs)) * np.sum(F * freqs)))
    return float(np.mean(scores))

def dcc(actions):
    diffs = np.diff(actions, axis=0)
    signs = np.sign(diffs)
    changes = (signs[1:] * signs[:-1]) < 0
    return float(changes.mean()) if actions.shape[0] >= 3 else 0.0

def jerk(actions):
    rng = np.ptp(actions, axis=0)
    rng = np.where(rng < 1e-9, 1.0, rng)
    scores = []
    for j in range(actions.shape[1]):
        x = actions[:, j] / rng[j]
        if len(x) < 4:
            scores.append(0.0)
            continue
        scores.append(float(np.abs(np.diff(x, n=3) / DT**3).mean()))
    return float(np.mean(scores))


# ── dataset loading ──────────────────────────────────────────────────────────

def load_episodes(hf_id: str, split: str, eps_per_task: int, include_gripper: bool = False):
    """Stream one split from HF (no disk caching) and return per-episode dicts.

    Only pulls the columns we need — skips images entirely to save bandwidth.
    """
    from datasets import load_dataset

    NEEDED_COLS = ["episode_index", "frame_index", "action", "action_raw", "success"]

    joint_slice = slice(None) if include_gripper else slice(0, 6)

    # episode_index resets to 0..N within each split (one split per task),
    # so process each split independently and concatenate in order.
    if split == "__all__":
        from datasets import get_dataset_split_names
        splits_to_load = get_dataset_split_names(hf_id)
    else:
        splits_to_load = [split]

    all_episodes = []
    total_rows   = 0

    for sp in splits_to_load:
        print(f"  Streaming {hf_id}  split={sp} ...", flush=True)
        ds = load_dataset(hf_id, split=sp, streaming=True).select_columns(NEEDED_COLS)

        ep_rows: dict[int, list] = defaultdict(list)
        for row in ds:
            ep_rows[row["episode_index"]].append(row)
            total_rows += 1

        for ep_idx in sorted(ep_rows):
            rows     = sorted(ep_rows[ep_idx], key=lambda r: r["frame_index"])
            act_proc = np.array([r["action"]     for r in rows], dtype=np.float32)
            act_raw  = np.array([r["action_raw"] for r in rows], dtype=np.float32)
            a        = act_proc[:, joint_slice]
            a_raw    = act_raw[:,  joint_slice]
            task_idx = len(all_episodes) // eps_per_task
            all_episodes.append({
                "episode_index": len(all_episodes),
                "task":          task_idx,
                "success":       bool(rows[-1]["success"]),
                "T":             len(rows),
                "sm_raw":        sm(a_raw),
                "sm_proc":       sm(a),
                "jerk_proc":     jerk(a),
                "dcc_proc":      dcc(a),
            })

    print(f"    Done: {total_rows} rows → {len(all_episodes)} episodes")
    return all_episodes


def cumulative_sr(episodes, eps_per_task):
    """Add sr_pct field: cumulative success rate within the current task."""
    task_counts = defaultdict(lambda: {"total": 0, "success": 0})
    for ep in episodes:
        t = ep["task"]
        task_counts[t]["total"]   += 1
        task_counts[t]["success"] += int(ep["success"])
        total   = task_counts[t]["total"]
        success = task_counts[t]["success"]
        ep["sr_pct"] = round(100.0 * success / total, 1)
    return episodes


def print_table(episodes, label):
    header = f"{'Task':>4}  {'Success':>7}  {'SR%':>6}  {'Sm_raw':>10}  {'Sm_proc':>10}  {'Jerk':>10}  {'DCC':>8}  {'T':>6}"
    sep    = "-" * len(header)
    print(f"\n{label}  ({len(episodes)} episodes)")
    print(sep)
    print(header)
    print(sep)
    for ep in episodes:
        print(
            f"{ep['task']:>4}  "
            f"{'TRUE' if ep['success'] else 'FALSE':>7}  "
            f"{ep['sr_pct']:>6.1f}  "
            f"{ep['sm_raw']:>10.5f}  "
            f"{ep['sm_proc']:>10.5f}  "
            f"{ep['jerk_proc']:>10.2f}  "
            f"{ep['dcc_proc']:>8.5f}  "
            f"{ep['T']:>6}"
        )
    print(sep)
    # Summary row
    sr_final = {t: [] for ep in episodes for t in [ep["task"]]}
    for ep in episodes:
        sr_final[ep["task"]].append(ep["success"])
    overall_sr = round(100.0 * sum(ep["success"] for ep in episodes) / len(episodes), 1)
    mean_sm_proc = round(np.mean([ep["sm_proc"]   for ep in episodes]), 5)
    mean_jerk    = round(np.mean([ep["jerk_proc"] for ep in episodes]), 2)
    mean_dcc     = round(np.mean([ep["dcc_proc"]  for ep in episodes]), 5)
    mean_T       = round(np.mean([ep["T"]         for ep in episodes]))
    print(f"{'MEAN':>4}  {'':>7}  {overall_sr:>6.1f}  {'':>10}  {mean_sm_proc:>10.5f}  {mean_jerk:>10.2f}  {mean_dcc:>8.5f}  {mean_T:>6}")
    print(sep)


def save_csv(episodes, path, split_label):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "episode_index", "task", "success", "sr_pct",
                         "sm_raw", "sm_proc", "jerk_proc", "dcc_proc", "T"])
        for ep in episodes:
            writer.writerow([
                split_label,
                ep["episode_index"],
                ep["task"],
                ep["success"],
                ep["sr_pct"],
                round(ep["sm_raw"],    6),
                round(ep["sm_proc"],   6),
                round(ep["jerk_proc"], 4),
                round(ep["dcc_proc"],  6),
                ep["T"],
            ])
    print(f"\nSaved: {path}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset",   help="HuggingFace dataset ID, e.g. BrandonAL/eval_smolvla_chunk50")
    p.add_argument("--split",   default=None,
                   help="Dataset split to load (e.g. libero_object). "
                        "If omitted, all splits are loaded.")
    p.add_argument("--episodes-per-task", type=int, default=10,
                   help="Episodes per task (used to infer task index). Default 10.")
    p.add_argument("--save",    default=None,
                   help="Path to save CSV output (optional).")
    p.add_argument("--include-gripper", action="store_true", default=False,
                   help="Include gripper joint (index 6) in metrics. "
                        "Default: excluded (gripper has different dynamics). "
                        "Use this to match existing Excel tables computed with all 7 joints.")
    args = p.parse_args()

    # Pass "__all__" to load_episodes so it handles split discovery internally
    # (episode_index resets per split, so splits must be processed in order)
    effective_split = args.split if args.split else "__all__"

    all_episodes = []
    if True:
        split = effective_split
        episodes = load_episodes(args.dataset, split, args.episodes_per_task,
                                  include_gripper=args.include_gripper)
        episodes = cumulative_sr(episodes, args.episodes_per_task)
        label    = f"{args.dataset}  [{split}]"
        print_table(episodes, label)
        for ep in episodes:
            ep["split"] = split
        all_episodes.extend(episodes)

    if args.save:
        save_csv(all_episodes, args.save, args.split or "all")


if __name__ == "__main__":
    main()
