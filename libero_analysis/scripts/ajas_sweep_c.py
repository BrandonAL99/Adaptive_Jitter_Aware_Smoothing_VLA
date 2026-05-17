"""
Offline C-value sweep for AJAS tuning.

Applies AJAS with a range of C values to existing episodes.npz,
measures Sm before and after per joint, and prints a comparison table.
No simulation required.

Usage
-----
    conda activate lerobot
    python libero_analysis/ajas_sweep_c.py
    python libero_analysis/ajas_sweep_c.py results/eval_chunk50_test/episodes.npz
"""

import sys
import numpy as np
from pathlib import Path

DT    = 1.0 / 30.0
CHUNK = 50
K_MIN = 5
K_MAX = 20

C_VALUES = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]


# ── metric & smoothing ─────────────────────────────────────────────────────────

def sm_per_joint(actions: np.ndarray) -> np.ndarray:
    """Sm for each joint independently. Returns shape (action_dim,)."""
    fs = 1.0 / DT
    scores = []
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        F = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=DT)
        n = len(freqs)
        scores.append((2.0 / (n * fs)) * float(np.sum(F * freqs)))
    return np.array(scores)


def ajas_smooth_per_joint(chunk: np.ndarray, C: float) -> np.ndarray:
    """Apply AJAS independently per joint using per-joint Sm → k mapping."""
    T, D = chunk.shape
    sm   = sm_per_joint(chunk)
    smoothed = chunk.copy()

    for j in range(D):
        k = int(np.clip(round(C * sm[j]), K_MIN, K_MAX))
        indices = list(range(0, T, k))
        if indices[-1] != T - 1:
            indices.append(T - 1)
        for i in range(len(indices) - 1):
            s, e = indices[i], indices[i + 1]
            smoothed[s:e + 1, j] = np.linspace(chunk[s, j], chunk[e, j], e - s + 1)

    return smoothed


# ── sweep ──────────────────────────────────────────────────────────────────────

def sweep(npz_path: str):
    npz = np.load(npz_path, allow_pickle=True)
    raw_actions = npz['raw_actions']
    ep_lengths  = npz['ep_lengths'].astype(int)
    ep_starts   = np.concatenate([[0], np.cumsum(ep_lengths[:-1])])
    action_dim  = raw_actions.shape[1]

    # Collect all CHUNK-sized windows across all episodes
    chunks = []
    for e in range(len(ep_lengths)):
        s, L = ep_starts[e], ep_lengths[e]
        data = raw_actions[s:s + L]
        for i in range(0, L - CHUNK + 1, CHUNK):
            chunks.append(data[i:i + CHUNK])
    n_chunks = len(chunks)
    print(f"Loaded {len(ep_lengths)} episodes → {n_chunks} chunks of {CHUNK} steps\n")

    # Raw Sm
    raw_sm = np.array([sm_per_joint(c) for c in chunks])   # (n_chunks, action_dim)
    raw_mean  = raw_sm.mean()
    raw_joint = raw_sm.mean(axis=0)

    # Sweep C values
    results = []
    for C in C_VALUES:
        proc_sm_all = []
        k_all = []
        for chunk in chunks:
            sm  = sm_per_joint(chunk)
            ks  = np.clip(np.round(C * sm).astype(int), K_MIN, K_MAX)
            proc = ajas_smooth_per_joint(chunk, C)
            proc_sm_all.append(sm_per_joint(proc))
            k_all.append(ks)

        proc_sm   = np.array(proc_sm_all)   # (n_chunks, action_dim)
        k_arr     = np.array(k_all)         # (n_chunks, action_dim)
        proc_mean = proc_sm.mean()
        reduction = (raw_mean - proc_mean) / raw_mean * 100
        k_mean    = k_arr.mean()

        results.append({
            'C':         C,
            'proc_mean': proc_mean,
            'reduction': reduction,
            'k_mean':    k_mean,
            'proc_joint': proc_sm.mean(axis=0),
            'k_joint':    k_arr.mean(axis=0),
        })

    # ── print table ────────────────────────────────────────────────────────────
    col = 11
    jcol = 8

    print(f"Raw Sm (×10⁻³):  mean={raw_mean*1e3:.3f}   "
          + "  ".join(f"J{j+1}={raw_joint[j]*1e3:.3f}" for j in range(action_dim)))
    print()

    header = (f"{'C':>6}  {'Sm mean':>{col}}  {'Reduction':>{col}}  {'k mean':>{col}}"
              + "".join(f"  {'J'+str(j+1)+' Sm':>{jcol}}" for j in range(action_dim))
              + "".join(f"  {'J'+str(j+1)+' k':>{jcol}}" for j in range(action_dim)))
    print(header)
    print("-" * len(header))

    for r in results:
        row = (f"{r['C']:>6}  "
               f"{r['proc_mean']*1e3:>{col}.3f}  "
               f"{r['reduction']:>{col}.1f}%  "
               f"{r['k_mean']:>{col}.1f}"
               + "".join(f"  {r['proc_joint'][j]*1e3:>{jcol}.3f}" for j in range(action_dim))
               + "".join(f"  {r['k_joint'][j]:>{jcol}.1f}" for j in range(action_dim)))
        print(row)

    print()
    print(f"Lowpass-2 reference Sm ≈ 0.680 ×10⁻³  (from pseudo code table)")
    print(f"Lowpass-4 reference Sm ≈ 0.800 ×10⁻³")
    print(f"K_MIN={K_MIN}  K_MAX={K_MAX}  CHUNK={CHUNK}")


if __name__ == '__main__':
    default = 'results/eval_chunk50_test/episodes.npz'
    path = sys.argv[1] if len(sys.argv) > 1 else default
    sweep(path)
