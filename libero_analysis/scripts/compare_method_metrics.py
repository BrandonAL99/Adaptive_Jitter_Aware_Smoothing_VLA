"""
Compare all 7 TC metrics across smoothing methods.

Loads raw_actions + proc_actions from one or more episodes.npz files,
applies LP filters post-hoc to the raw actions, then reports all 7 metrics.

This is a "signal-level" comparison — it shows how each method transforms
the action sequences, independent of closed-loop SR effects.

Usage
-----
    # Single npz (uses raw_actions as baseline, proc_actions as spline)
    python libero_analysis/scripts/compare_method_metrics.py \
        results/eval_spline/libero_object/episodes.npz

    # Multiple suites (aggregated)
    python libero_analysis/scripts/compare_method_metrics.py \
        results/eval_spline/libero_object/episodes.npz \
        results/eval_spline/libero_spatial/episodes.npz \
        results/eval_spline/libero_goal/episodes.npz \
        results/eval_spline/libero_10/episodes.npz

    # Add extra npz as a named method (e.g. a different processor's proc_actions)
    python libero_analysis/scripts/compare_method_metrics.py \
        results/eval_spline/libero_object/episodes.npz \
        --extra "AJAS-C500:results/eval_ajas_laptop/c500/libero_object/episodes.npz:proc"

Options
-------
    --lp-cutoffs    Space-separated LP cutoff frequencies in Hz (default: 2 4 6)
    --lp-order      Butterworth filter order (default: 2)
    --lp-causal     Use causal lfilter (default); --no-lp-causal uses zero-phase filtfilt
    --skip-gripper  Exclude joint index 6 from all metrics (default: True)
    --joints        Comma-separated joint indices to include (overrides --skip-gripper)
    --save          Path to save CSV summary (optional)
    --suite-label   Label for the suite (default: inferred from path)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, filtfilt

DT  = 1.0 / 30.0
FPS = 30.0


def macd(actions, dt):
    return float(np.abs(np.diff(actions, axis=0)).mean())

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

def direction_change_count(actions, dt):
    diffs = np.diff(actions, axis=0)
    signs = np.sign(diffs)
    changes = (signs[1:] * signs[:-1]) < 0
    return float(changes.mean()) if actions.shape[0] >= 3 else 0.0

def spectral_entropy(actions, dt):
    scores = []
    N, n_bins = actions.shape[0], actions.shape[0] // 2 + 1
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        F = np.abs(np.fft.rfft(x)) ** 2
        total = F.sum()
        if total < 1e-30:
            scores.append(0.0); continue
        p = F / total
        p = p[p > 1e-30]
        scores.append(float(np.clip(-np.sum(p * np.log2(p)) / np.log2(n_bins), 0.0, 1.0)))
    return float(np.mean(scores))

def weighted_mean_frequency(actions, dt):
    fs = 1.0 / dt
    scores = []
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        F = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=dt)
        scores.append(float((2.0 / (len(freqs) * fs)) * np.sum(F * freqs)))
    return float(np.mean(scores))

def signal_energy_ratio(actions, dt, threshold_hz=1.0):
    scores = []
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        F = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=dt)
        total = F.sum()
        if total < 1e-30:
            scores.append(0.0); continue
        scores.append(float(1.0 - F[freqs <= threshold_hz].sum() / total))
    return float(np.mean(scores))

def autocorrelation_decay_rate(actions, dt):
    scores = []
    max_lag = min(actions.shape[0] // 2, 50)
    if max_lag < 3:
        return 0.0
    lags = np.arange(max_lag) * dt
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        var = np.var(x)
        if var < 1e-30:
            scores.append(0.0); continue
        ac = np.array([np.mean(x[:len(x)-k] * x[k:]) / var for k in range(max_lag)])
        valid = ac > 1e-6
        if valid.sum() < 3:
            scores.append(float(max(0.0, (ac[0] - ac[-1]) / (lags[-1] + 1e-9)))); continue
        lam = -np.polyfit(lags[valid], np.log(ac[valid]), 1)[0]
        scores.append(float(max(0.0, lam)))
    return float(np.mean(scores))

DT  = 1.0 / 30.0
FPS = 30.0

METRICS = {
    "MACD":  macd,
    "Jerk":  normalised_jerk,
    "DCC":   direction_change_count,
    "SE":    spectral_entropy,
    "Sm":    weighted_mean_frequency,
    "1-SER": signal_energy_ratio,
    "ADR":   autocorrelation_decay_rate,
}


# ── helpers ────────────────────────────────────────────────────────────────────

def apply_lp(ep: np.ndarray, cutoff_hz: float, order: int = 2,
             causal: bool = True) -> np.ndarray:
    nyq = 0.5 * FPS
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    fn   = lfilter if causal else filtfilt
    out  = ep.copy()
    for j in range(ep.shape[1]):
        out[:, j] = fn(b, a, ep[:, j])
    return out


def split_episodes(actions: np.ndarray, ep_lengths: np.ndarray) -> list[np.ndarray]:
    eps, start = [], 0
    for L in ep_lengths:
        eps.append(actions[start:start + L])
        start += L
    return eps


def score(ep: np.ndarray, joint_mask: np.ndarray | None) -> dict[str, float]:
    a = ep[:, joint_mask] if joint_mask is not None else ep
    return {name: fn(a, DT) for name, fn in METRICS.items()}


def mean_scores(episode_scores: list[dict]) -> dict[str, float]:
    keys = list(METRICS.keys())
    return {k: float(np.mean([s[k] for s in episode_scores])) for k in keys}


def std_scores(episode_scores: list[dict]) -> dict[str, float]:
    keys = list(METRICS.keys())
    return {k: float(np.std([s[k] for s in episode_scores])) for k in keys}


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+", help="episodes.npz file(s) to load")
    p.add_argument("--extra", action="append", default=[],
                   metavar="LABEL:PATH:raw_or_proc",
                   help="Extra npz to add as a named method")
    p.add_argument("--lp-cutoffs", nargs="+", type=float, default=[2.0, 4.0, 6.0])
    p.add_argument("--lp-order",   type=int,   default=2)
    p.add_argument("--no-lp-causal", dest="lp_causal", action="store_false",
                   help="Use zero-phase filtfilt instead of causal lfilter")
    p.set_defaults(lp_causal=True)
    p.add_argument("--skip-gripper", action="store_true", default=True)
    p.add_argument("--no-skip-gripper", dest="skip_gripper", action="store_false")
    p.add_argument("--joints", type=str, default=None,
                   help="Comma-separated joint indices to include (e.g. 0,1,2,3,4,5)")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--suite-label", type=str, default=None)
    args = p.parse_args()

    # ── load primary npz files ─────────────────────────────────────────────────
    all_raw_eps:   list[np.ndarray] = []
    all_proc_eps:  list[np.ndarray] = []

    for path in args.npz:
        f = np.load(path, allow_pickle=True)
        eps_raw  = split_episodes(f["raw_actions"],  f["ep_lengths"])
        eps_proc = split_episodes(f["proc_actions"], f["ep_lengths"])
        all_raw_eps.extend(eps_raw)
        all_proc_eps.extend(eps_proc)
        label = args.suite_label or Path(path).parent.name
        print(f"Loaded {len(eps_raw)} episodes from {path}  ({label})")

    n_eps = len(all_raw_eps)
    print(f"Total: {n_eps} episodes\n")

    # ── determine joint mask ───────────────────────────────────────────────────
    action_dim = all_raw_eps[0].shape[1]
    if args.joints:
        joint_mask = np.array([int(j) for j in args.joints.split(",")])
    elif args.skip_gripper and action_dim > 6:
        joint_mask = np.arange(6)   # exclude joint 6 (gripper)
    else:
        joint_mask = None

    if joint_mask is not None:
        print(f"Using joints: {joint_mask.tolist()}")
    else:
        print(f"Using all {action_dim} joints")
    print()

    # ── build methods dict ─────────────────────────────────────────────────────
    # label -> list of episode np.ndarray
    methods: dict[str, list[np.ndarray]] = {}
    methods["Raw"]   = all_raw_eps
    methods["Spline"] = all_proc_eps

    # LP post-hoc applied to raw actions
    lp_mode = "causal" if args.lp_causal else "zero-phase"
    for hz in args.lp_cutoffs:
        key = f"LP {hz:.0f}Hz ({lp_mode})"
        methods[key] = [apply_lp(ep, hz, args.lp_order, args.lp_causal)
                        for ep in all_raw_eps]

    # extra named methods
    for spec in args.extra:
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"  WARNING: --extra format is LABEL:PATH:raw_or_proc, got {spec!r}")
            continue
        label, path, field = parts
        f = np.load(path, allow_pickle=True)
        key_name = "raw_actions" if field.startswith("r") else "proc_actions"
        eps = split_episodes(f[key_name], f["ep_lengths"])
        methods[label] = eps
        print(f"Extra: {label} ({len(eps)} episodes from {path})")

    # ── compute metrics ────────────────────────────────────────────────────────
    print("Computing metrics...")
    results: dict[str, dict[str, float]] = {}
    stds:    dict[str, dict[str, float]] = {}

    for method, eps in methods.items():
        ep_scores = [score(ep, joint_mask) for ep in eps]
        results[method] = mean_scores(ep_scores)
        stds[method]    = std_scores(ep_scores)
        print(f"  {method:<30}  done")

    # ── print table ───────────────────────────────────────────────────────────
    metric_names = list(METRICS.keys())
    col_w = 13

    print()
    print("=" * (30 + col_w * len(metric_names)))
    label_str = f"  Suite: {args.suite_label}" if args.suite_label else f"  {n_eps} episodes"
    print(f"METRIC COMPARISON  |  {label_str}  |  joints={'all' if joint_mask is None else list(joint_mask)}")
    print(f"Higher = more jitter for ALL metrics  |  LP mode: {lp_mode}")
    print("=" * (30 + col_w * len(metric_names)))

    header = f"{'Method':<30}" + "".join(f"{m:>{col_w}}" for m in metric_names)
    sep    = "-" * len(header)
    print(header)
    print(sep)

    for method, scores in results.items():
        row = f"{method:<30}" + "".join(f"{scores[m]:>{col_w}.5f}" for m in metric_names)
        print(row)

    print(sep)

    # ── % reduction from Raw ──────────────────────────────────────────────────
    raw_scores = results["Raw"]
    print(f"\n{'% reduction from Raw (positive = less jitter)'}")
    print(header)
    print(sep)

    for method, scores in results.items():
        if method == "Raw":
            continue
        pcts = {m: 100.0 * (raw_scores[m] - scores[m]) / (abs(raw_scores[m]) + 1e-12)
                for m in metric_names}
        row = f"{method:<30}" + "".join(f"{pcts[m]:>{col_w}.1f}%" for m in metric_names)
        print(row)

    print(sep)

    # ── per-metric best method ────────────────────────────────────────────────
    print("\nBest method per metric (lowest jitter, excluding Raw):")
    non_raw = {k: v for k, v in results.items() if k != "Raw"}
    for m in metric_names:
        best = min(non_raw, key=lambda k: non_raw[k][m])
        best_pct = 100.0 * (raw_scores[m] - non_raw[best][m]) / (abs(raw_scores[m]) + 1e-12)
        print(f"  {m:<8}  →  {best:<30}  ({best_pct:+.1f}% from raw)")

    # ── save CSV ──────────────────────────────────────────────────────────────
    if args.save:
        import csv
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method"] + metric_names)
            for method, scores in results.items():
                writer.writerow([method] + [scores[m] for m in metric_names])
        print(f"\nSaved CSV: {out}")


if __name__ == "__main__":
    main()
