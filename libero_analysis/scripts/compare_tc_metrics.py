"""
TC metric comparison across smooth vs jittery episodes.

Evaluates 7 temporal consistency metrics on:
  - Real recordings: BrandonAL/smooth_and_jittery_so101
      episode 0 = smooth, episode 1 = jittery (higher mean |Δaction|)
  - 4 synthetic datasets: pure sine + sine + increasing white-noise jitter

For each metric, reports the score on each dataset and computes a
discriminative power score:  D = (jitter - smooth) / (|smooth| + ε)
Higher D = better separation between smooth and jittery.

Metrics
-------
1.  MACD   Mean Absolute Consecutive Difference
2.  Jerk   Normalised Jerk  (rate of acceleration change)
3.  DCC    Direction Change Count  (sign changes in consecutive diffs)
4.  SE     Spectral Entropy
5.  Sm     Weighted Mean Frequency  (existing metric)
6.  SER    Signal Energy Ratio  (existing metric)
7.  ADR    Autocorrelation Decay Rate

Output
------
  libero_analysis/claude_results/metric_comparison.txt   human-readable table
  libero_analysis/claude_results/metric_comparison.json  raw numbers

Usage
-----
    conda activate lerobot
    python libero_analysis/compare_tc_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── sampling config ────────────────────────────────────────────────────────────
FPS = 30.0
DT  = 1.0 / FPS
SER_THRESHOLD_HZ = 1.0   # for SER metric (matches existing temporal_consistency.py)


# ══════════════════════════════════════════════════════════════════════════════
# Metric implementations
# All functions accept:
#   actions : np.ndarray  shape (T, n_joints)
#   dt      : float       seconds per step
# Return a single float (mean across joints unless noted).
# Convention: HIGHER = MORE JITTER (so all metrics point the same way).
# SER is inverted at the end since it is higher-is-smoother.
# ══════════════════════════════════════════════════════════════════════════════

def macd(actions: np.ndarray, dt: float) -> float:
    """Mean Absolute Consecutive Difference (mean across joints).

    MACD = mean( |a[t+1] - a[t]| )
    Higher → more jitter (larger step-to-step changes).
    """
    diffs = np.abs(np.diff(actions, axis=0))   # (T-1, n_joints)
    return float(diffs.mean())


def normalised_jerk(actions: np.ndarray, dt: float) -> float:
    """Normalised Jerk — mean absolute third derivative, scaled by dt³.

    Jerk ≈ Δ³a / dt³  (finite-difference approximation).
    Normalised by the range of the signal so it is scale-independent.
    Higher → more jitter.
    """
    a = actions  # (T, n_joints)
    rng = np.ptp(a, axis=0)  # peak-to-peak per joint; avoid div-by-zero
    rng = np.where(rng < 1e-9, 1.0, rng)

    scores = []
    for j in range(a.shape[1]):
        x = a[:, j] / rng[j]
        if len(x) < 4:
            scores.append(0.0)
            continue
        jk = np.diff(x, n=3) / (dt ** 3)
        scores.append(float(np.abs(jk).mean()))
    return float(np.mean(scores))


def direction_change_count(actions: np.ndarray, dt: float) -> float:
    """Direction Change Count — fraction of steps where velocity reverses sign.

    DCC = mean over joints of ( # sign changes in Δa ) / (T-2)
    Higher → more direction reversals → more jitter.
    """
    diffs = np.diff(actions, axis=0)     # (T-1, n_joints)
    signs = np.sign(diffs)
    # sign change at step t ↔ sign[t] != sign[t-1]  (and neither is 0)
    changes = (signs[1:] * signs[:-1]) < 0   # (T-2, n_joints)
    T = actions.shape[0]
    if T < 3:
        return 0.0
    return float(changes.mean())


def spectral_entropy(actions: np.ndarray, dt: float) -> float:
    """Spectral Entropy — Shannon entropy of the normalised power spectrum.

    SE = -Σ p(f) log2 p(f)   where p(f) = |FFT|² / Σ|FFT|²
    Higher SE → energy spread across many frequencies → more jitter.
    Normalised to [0, 1] by dividing by log2(N/2+1).
    """
    scores = []
    N = actions.shape[0]
    n_bins = N // 2 + 1
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        F = np.abs(np.fft.rfft(x)) ** 2
        total = F.sum()
        if total < 1e-30:
            scores.append(0.0)
            continue
        p = F / total
        p = p[p > 1e-30]
        se = -np.sum(p * np.log2(p))
        se_norm = se / np.log2(n_bins)   # normalise to [0,1]
        scores.append(float(np.clip(se_norm, 0.0, 1.0)))
    return float(np.mean(scores))


def weighted_mean_frequency(actions: np.ndarray, dt: float) -> float:
    """Sm — Weighted Mean Frequency (existing metric, reproduced here).

    Sm = (2 / (N * fs)) * Σ |F(f)| * f
    Higher → more high-frequency content → more jitter.
    """
    fs = 1.0 / dt
    scores = []
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        F = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=dt)
        n = len(freqs)
        sm = (2.0 / (n * fs)) * np.sum(F * freqs)
        scores.append(float(sm))
    return float(np.mean(scores))


def signal_energy_ratio(actions: np.ndarray, dt: float,
                         threshold_hz: float = SER_THRESHOLD_HZ) -> float:
    """SER — Signal Energy Ratio (existing metric, inverted for table convention).

    Raw SER = energy below threshold / total energy.
    Higher raw SER = smoother.
    Here we return 1 - SER so higher = more jitter (consistent with other metrics).
    """
    scores = []
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        F = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=dt)
        total = F.sum()
        if total < 1e-30:
            scores.append(0.0)
            continue
        ser = F[freqs <= threshold_hz].sum() / total
        scores.append(float(1.0 - ser))   # invert: higher = jitterier
    return float(np.mean(scores))


def autocorrelation_decay_rate(actions: np.ndarray, dt: float) -> float:
    """ADR — Autocorrelation Decay Rate.

    Fits an exponential to the normalised autocorrelation and returns the
    decay constant λ (in s⁻¹).  Higher λ → faster decay → more jitter
    (smooth signals stay correlated over longer lags).

    Uses lags 0 to min(T//2, 50).
    """
    scores = []
    max_lag = min(actions.shape[0] // 2, 50)
    if max_lag < 3:
        return 0.0

    lags = np.arange(max_lag) * dt

    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        var = np.var(x)
        if var < 1e-30:
            scores.append(0.0)
            continue

        # Normalised autocorrelation at each lag
        ac = np.array([
            np.mean(x[:len(x) - k] * x[k:]) / var
            for k in range(max_lag)
        ])

        # Fit log|ac| ~ -λ * lag  (linear in log space, ignoring where ac ≤ 0)
        valid = ac > 1e-6
        if valid.sum() < 3:
            # Fallback: use slope of ac directly
            scores.append(float(max(0.0, (ac[0] - ac[-1]) / (lags[-1] + 1e-9))))
            continue

        log_ac = np.log(ac[valid])
        lags_valid = lags[valid]
        # Least-squares slope
        lam = -np.polyfit(lags_valid, log_ac, 1)[0]
        scores.append(float(max(0.0, lam)))

    return float(np.mean(scores))


# ── metric registry ────────────────────────────────────────────────────────────

METRICS = {
    "MACD":  macd,
    "Jerk":  normalised_jerk,
    "DCC":   direction_change_count,
    "SE":    spectral_entropy,
    "Sm":    weighted_mean_frequency,
    "1-SER": signal_energy_ratio,   # inverted; higher = jitterier
    "ADR":   autocorrelation_decay_rate,
}


# ── synthetic dataset generator ────────────────────────────────────────────────

def make_synthetic_datasets(T: int = 500, n_joints: int = 6,
                             fps: float = FPS) -> dict[str, np.ndarray]:
    """Generate 4 synthetic datasets of increasing jitter.

    Returns dict label → actions (T, n_joints).
    """
    rng = np.random.default_rng(42)
    t = np.arange(T) / fps

    # Base: 1 Hz sine wave per joint (slightly different frequency per joint)
    freqs = np.linspace(0.5, 1.5, n_joints)
    base = np.stack([np.sin(2 * np.pi * f * t) for f in freqs], axis=1)  # (T, n_joints)

    datasets = {}
    noise_levels = [0.0, 0.1, 0.5, 2.0]
    labels      = ["synthetic_clean", "synthetic_low_jitter",
                   "synthetic_med_jitter", "synthetic_high_jitter"]

    for label, sigma in zip(labels, noise_levels):
        noise = rng.standard_normal((T, n_joints)) * sigma
        datasets[label] = (base + noise).astype(np.float32)

    return datasets


# ── scoring ────────────────────────────────────────────────────────────────────

def score_all(actions: np.ndarray, dt: float) -> dict[str, float]:
    return {name: fn(actions, dt) for name, fn in METRICS.items()}


def discriminative_power(smooth_score: float, jitter_score: float) -> float:
    """D = (jitter - smooth) / (|smooth| + ε).  Higher = better separation."""
    return (jitter_score - smooth_score) / (abs(smooth_score) + 1e-9)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path("libero_analysis/claude_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BrandonAL/smooth_and_jittery_so101 ...")
    ds = load_dataset("BrandonAL/smooth_and_jittery_so101", split="train")

    ep0, ep1 = [], []
    for row in ds:
        (ep0 if row["episode_index"] == 0 else ep1).append(row["action"])
    ep0 = np.array(ep0, dtype=np.float32)
    ep1 = np.array(ep1, dtype=np.float32)

    # Determine which is smooth and which is jittery from mean |diff|
    macd0 = np.abs(np.diff(ep0, axis=0)).mean()
    macd1 = np.abs(np.diff(ep1, axis=0)).mean()
    if macd0 < macd1:
        smooth_real, jittery_real = ep0, ep1
        smooth_label, jittery_label = "ep0", "ep1"
    else:
        smooth_real, jittery_real = ep1, ep0
        smooth_label, jittery_label = "ep1", "ep0"
    print(f"  Smooth = episode {smooth_label}  ({len(smooth_real)} frames)")
    print(f"  Jittery = episode {jittery_label}  ({len(jittery_real)} frames)")

    print("Generating synthetic datasets ...")
    synthetics = make_synthetic_datasets(T=500, n_joints=ep0.shape[1], fps=FPS)

    # All datasets in evaluation order
    datasets: dict[str, np.ndarray] = {
        f"real_smooth  ({smooth_label})":  smooth_real,
        f"real_jittery ({jittery_label})": jittery_real,
        **synthetics,
    }

    print("Computing metrics ...")
    results: dict[str, dict[str, float]] = {}
    for label, actions in datasets.items():
        results[label] = score_all(actions, DT)
        print(f"  {label:<40}  done ({actions.shape[0]} frames)")

    # ── discriminative power (real smooth vs real jittery) ──────────────────
    smooth_key  = f"real_smooth  ({smooth_label})"
    jittery_key = f"real_jittery ({jittery_label})"
    disc = {}
    for metric in METRICS:
        d = discriminative_power(results[smooth_key][metric],
                                 results[jittery_key][metric])
        disc[metric] = d

    ranked = sorted(disc.items(), key=lambda x: x[1], reverse=True)

    # ── monotonicity check on synthetic series ───────────────────────────────
    synth_order = ["synthetic_clean", "synthetic_low_jitter",
                   "synthetic_med_jitter", "synthetic_high_jitter"]
    mono = {}
    for metric in METRICS:
        vals = [results[k][metric] for k in synth_order]
        # Monotonically increasing (more jitter = higher score)?
        is_mono = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        mono[metric] = is_mono

    # ══════════════════════════════════════════════════════════════════════════
    # Print table
    # ══════════════════════════════════════════════════════════════════════════
    col_w = 14
    metric_names = list(METRICS.keys())

    header = f"{'Dataset':<42}" + "".join(f"{m:>{col_w}}" for m in metric_names)
    sep    = "-" * len(header)

    lines = [
        "TC METRIC COMPARISON",
        "=" * len(header),
        "",
        "Convention: higher value = more jitter for ALL metrics",
        "  (1-SER is listed rather than SER so all metrics point the same way)",
        "",
        header,
        sep,
    ]

    for label, scores in results.items():
        row = f"{label:<42}" + "".join(f"{scores[m]:>{col_w}.5f}" for m in metric_names)
        lines.append(row)

    lines += [
        sep,
        f"{'Discrim. power (real)':<42}" +
        "".join(f"{disc[m]:>{col_w}.3f}" for m in metric_names),
        f"{'Monotone on synthetic?':<42}" +
        "".join(f"{'YES' if mono[m] else 'NO':>{col_w}}" for m in metric_names),
        "",
        "RANKING by discriminative power (real smooth vs real jittery):",
        "-" * 50,
    ]

    for rank, (metric, d) in enumerate(ranked, 1):
        mono_str = "monotone" if mono[metric] else "NOT monotone"
        lines.append(f"  #{rank:>2}  {metric:<8}  D = {d:+.3f}   [{mono_str} on synthetic]")

    lines += [
        "",
        "Notes:",
        "  D > 0  : metric correctly identifies jittery > smooth",
        "  D < 0  : metric is inverted (would need flipping)",
        "  D near 0: poor discriminative power",
        "",
        f"Dataset details:",
        f"  real_smooth:  {smooth_real.shape[0]} frames, {smooth_real.shape[1]} joints, 30 fps",
        f"  real_jittery: {jittery_real.shape[0]} frames, {jittery_real.shape[1]} joints, 30 fps",
        f"  synthetic:    500 frames, {ep0.shape[1]} joints, sine + white noise σ = 0 / 0.1 / 0.5 / 2.0",
        f"  SER threshold: {SER_THRESHOLD_HZ} Hz",
    ]

    output = "\n".join(lines)
    print("\n" + output)

    # Save
    txt_path = out_dir / "metric_comparison.txt"
    txt_path.write_text(output)
    print(f"\nSaved: {txt_path}")

    json_path = out_dir / "metric_comparison.json"
    json_path.write_text(json.dumps({
        "results":     results,
        "disc_power":  disc,
        "monotone":    mono,
        "ranking":     [{"rank": i+1, "metric": m, "disc_power": d, "monotone": mono[m]}
                        for i, (m, d) in enumerate(ranked)],
        "config": {"fps": FPS, "dt": DT, "ser_threshold_hz": SER_THRESHOLD_HZ},
    }, indent=2))
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
