"""
Plot training steps vs success rate from results/eval_results.json.

Usage:
    python simulation/scripts/plot_eval.py
    python simulation/scripts/plot_eval.py --results results/eval_results.json --out results/eval_plot.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = Path("results/eval_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS_FILE))
    parser.add_argument("--out",     default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    data.sort(key=lambda r: r["step"])
    steps  = [r["step"]      for r in data]
    scores = [r["mean_score"] for r in data]

    # Per-config breakdown
    config_names = list(data[0]["config_scores"].keys()) if data else []
    config_means = {
        name: [r["config_scores"][name]["mean"] for r in data]
        for name in config_names
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ── Top: overall mean score ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, scores, "o-", color="black", linewidth=2, markersize=5, label="Mean score")
    ax.axhline(1.0, color="green",  linestyle="--", alpha=0.4, label="Perfect (1.0)")
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.4, label="Grasp only (0.5)")
    ax.set_ylabel("Mean score")
    ax.set_title("Pick-place evaluation: score vs training steps")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Bottom: per-config breakdown ───────────────────────────────────────
    ax = axes[1]
    cmap = plt.cm.get_cmap("tab10", len(config_names))
    for i, name in enumerate(config_names):
        ax.plot(steps, config_means[name], "o--", color=cmap(i),
                linewidth=1, markersize=4, alpha=0.8, label=name)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean score per config")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
