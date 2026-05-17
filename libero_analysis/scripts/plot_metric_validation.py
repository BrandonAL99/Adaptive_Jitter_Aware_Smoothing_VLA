"""
Metric validation plots for the thesis report.

Figure 1 — Monotonicity & sensitivity:
  X: noise sigma (synthetic datasets, sine + white noise)
  Y: metric score normalised to [0, 1] per metric
  One line per metric.

Figure 2 — Real episode table:
  Prints a LaTeX table of raw metric values for real_smooth vs real_jittery.

Usage
-----
    conda activate lerobot
    python libero_analysis/plot_metric_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── reuse metric implementations from compare_tc_metrics ──────────────────────
from libero_analysis.compare_tc_metrics import (
    METRICS, score_all, make_synthetic_datasets, DT, FPS
)

# ── config ─────────────────────────────────────────────────────────────────────
SIGMAS = [round(i * 0.2, 1) for i in range(11)]  # 0.0, 0.2, 0.4, ... 2.0
T      = 500
SEED   = 42
OUT_DIR = Path("libero_analysis/claude_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_LABELS = {
    "MACD":  "MACD",
    "Jerk":  "Jerk",
    "DCC":   "DCC",
    "SE":    "Spectral Entropy",
    "Sm":    "Weighted Mean Freq (Sm)",
    "1-SER": "1 − SER",
    "ADR":   "ADR",
}

COLORS = plt.cm.tab10.colors


# ── synthetic sweep ────────────────────────────────────────────────────────────

def sweep_sigmas(sigmas: list[float], T: int, seed: int) -> dict[str, list[float]]:
    """Return {metric_name: [score_at_sigma_0, score_at_sigma_1, ...]}."""
    rng = np.random.default_rng(seed)
    n_joints = 6
    t = np.arange(T) / FPS
    freqs = np.linspace(0.5, 1.5, n_joints)
    base = np.stack([np.sin(2 * np.pi * f * t) for f in freqs], axis=1).astype(np.float32)

    scores_by_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    for sigma in sigmas:
        noise = rng.standard_normal((T, n_joints)).astype(np.float32) * sigma
        actions = base + noise
        s = score_all(actions, DT)
        for m, v in s.items():
            scores_by_metric[m].append(v)

    return scores_by_metric


def normalise(vals: list[float]) -> np.ndarray:
    a = np.array(vals, dtype=float)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


# ── plot ───────────────────────────────────────────────────────────────────────

def plot_monotonicity(sigmas, scores_by_metric, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (metric, vals) in enumerate(scores_by_metric.items()):
        norm_vals = normalise(vals)
        ax.plot(sigmas, norm_vals,
                marker="o", markersize=4,
                color=COLORS[i % len(COLORS)],
                label=METRIC_LABELS[metric])

    ax.set_xlabel("Noise standard deviation (σ)", fontsize=12)
    ax.set_ylabel("Normalised metric score", fontsize=12)
    ax.set_title("Metric monotonicity and sensitivity\n"
                 "Synthetic signal: sine wave + white noise", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


def save_raw_csv(sigmas, scores_by_metric, out_path: Path):
    metric_names = list(scores_by_metric.keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma"] + metric_names)
        for i, sigma in enumerate(sigmas):
            writer.writerow([sigma] + [scores_by_metric[m][i] for m in metric_names])
    print(f"Saved: {out_path}")


def print_sigma_table(sigmas, scores_by_metric):
    metric_names = list(scores_by_metric.keys())
    col = 12
    header = f"{'sigma':<8}" + "".join(f"{m:>{col}}" for m in metric_names)
    sep = "-" * len(header)
    print("\nRaw metric scores vs sigma")
    print(sep)
    print(header)
    print(sep)
    for i, sigma in enumerate(sigmas):
        row = f"{sigma:<8.3f}" + "".join(f"{scores_by_metric[m][i]:>{col}.5f}" for m in metric_names)
        print(row)
    print(sep)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Sweeping σ across {SIGMAS} ...")
    scores_by_metric = sweep_sigmas(SIGMAS, T, SEED)

    plot_monotonicity(SIGMAS, scores_by_metric,
                      OUT_DIR / "metric_monotonicity.png")
    save_raw_csv(SIGMAS, scores_by_metric,
                 OUT_DIR / "metric_monotonicity_raw.csv")
    print_sigma_table(SIGMAS, scores_by_metric)


if __name__ == "__main__":
    main()
