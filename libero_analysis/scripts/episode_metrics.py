"""
Print per-episode Sm, DCC, and Jerk for proc_actions in an episodes.npz file.

Usage
-----
    python libero_analysis/scripts/episode_metrics.py results/eval_spline/libero_spatial/episodes.npz

    # Include raw Sm for comparison
    python libero_analysis/scripts/episode_metrics.py episodes.npz --raw

    # Skip gripper joint (default: True)
    python libero_analysis/scripts/episode_metrics.py episodes.npz --no-skip-gripper
"""

import argparse
import sys
from pathlib import Path

import numpy as np

DT  = 1.0 / 30.0
FPS = 30.0


def weighted_mean_frequency(actions, dt):
    fs = 1.0 / dt
    scores = []
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        F = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=dt)
        scores.append(float((2.0 / (len(freqs) * fs)) * np.sum(F * freqs)))
    return float(np.mean(scores))


def direction_change_count(actions, dt):
    diffs = np.diff(actions, axis=0)
    signs = np.sign(diffs)
    changes = (signs[1:] * signs[:-1]) < 0
    return float(changes.mean()) if actions.shape[0] >= 3 else 0.0


def normalised_jerk(actions, dt):
    rng = np.ptp(actions, axis=0)
    rng = np.where(rng < 1e-9, 1.0, rng)
    scores = []
    for j in range(actions.shape[1]):
        x = actions[:, j] / rng[j]
        if len(x) < 4:
            scores.append(0.0)
            continue
        scores.append(float(np.abs(np.diff(x, n=3) / dt**3).mean()))
    return float(np.mean(scores))


def split_episodes(actions, ep_lengths):
    eps, start = [], 0
    for L in ep_lengths:
        eps.append(actions[start:start + L])
        start += L
    return eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="Path to episodes.npz")
    p.add_argument("--raw", action="store_true",
                   help="Also print Sm for raw_actions")
    p.add_argument("--no-skip-gripper", dest="skip_gripper",
                   action="store_false", default=True)
    args = p.parse_args()

    f = np.load(args.npz, allow_pickle=True)
    ep_lengths = f["ep_lengths"]
    proc_eps   = split_episodes(f["proc_actions"], ep_lengths)
    raw_eps    = split_episodes(f["raw_actions"],  ep_lengths) if args.raw else None

    successes = f["successes"] if "successes" in f else [None] * len(proc_eps)

    action_dim = proc_eps[0].shape[1]
    if args.skip_gripper and action_dim > 6:
        joints = np.arange(6)
    else:
        joints = np.arange(action_dim)

    label = Path(args.npz).parent.name
    print(f"# {label}  ({len(proc_eps)} episodes, joints {joints.tolist()})")
    print()

    # header
    cols = ["ep", "success", "T", "sm_proc", "dcc_proc", "jerk_proc"]
    if args.raw:
        cols.insert(3, "sm_raw")
    print(",".join(cols))

    for i, ep in enumerate(proc_eps):
        a = ep[:, joints]
        sm   = weighted_mean_frequency(a, DT)
        dcc  = direction_change_count(a, DT)
        jerk = normalised_jerk(a, DT)

        success = bool(successes[i]) if successes[i] is not None else ""
        row = [str(i), str(success), str(len(ep)),
               f"{sm:.6f}", f"{dcc:.6f}", f"{jerk:.4f}"]

        if args.raw:
            a_raw = raw_eps[i][:, joints]
            sm_raw = weighted_mean_frequency(a_raw, DT)
            row.insert(3, f"{sm_raw:.6f}")

        print(",".join(row))


if __name__ == "__main__":
    main()
