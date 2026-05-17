"""
AJAS Tuning Tool

Interactive visualisation of raw vs AJAS-smoothed action chunks.
Use this to determine the best mapping from Sm → look-ahead distance k.

Controls
--------
  Episode slider  : select episode
  Chunk slider    : select 50-step chunk within episode
  k radio buttons : look-ahead stride (keyframe spacing)

The info panel shows the Sm value for the current chunk so you can
record which k looks best at each Sm level.

Usage
-----
    conda activate lerobot
    python libero_analysis/ajas_tuning.py
    python libero_analysis/ajas_tuning.py results/eval_chunk50_test/episodes.npz
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons
from pathlib import Path

DT      = 1.0 / 30.0
CHUNK   = 50
K_OPTIONS = [1, 2, 3, 5, 7, 10, 15, 20]


# ── metrics & smoothing ────────────────────────────────────────────────────────

def compute_sm(actions: np.ndarray) -> np.ndarray:
    """Returns Sm per joint, shape (action_dim,)."""
    fs = 1.0 / DT
    scores = []
    for j in range(actions.shape[1]):
        x = actions[:, j] - np.mean(actions[:, j])
        F = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=DT)
        n = len(freqs)
        scores.append((2.0 / (n * fs)) * float(np.sum(F * freqs)))
    return np.array(scores)


def ajas_smooth(chunk: np.ndarray, k: int) -> np.ndarray:
    """Linear interpolation between every k-th keyframe."""
    T, D = chunk.shape
    indices = list(range(0, T, k))
    if indices[-1] != T - 1:
        indices.append(T - 1)
    smoothed = np.empty_like(chunk)
    for i in range(len(indices) - 1):
        s, e = indices[i], indices[i + 1]
        for d in range(D):
            smoothed[s:e + 1, d] = np.linspace(chunk[s, d], chunk[e, d], e - s + 1)
    return smoothed


# ── main ───────────────────────────────────────────────────────────────────────

def main(npz_path: str):
    npz = np.load(npz_path, allow_pickle=True)
    raw_actions = npz['raw_actions']          # (N, action_dim)
    ep_lengths  = npz['ep_lengths'].astype(int)
    task_ids    = npz['task_ids']
    suites      = npz['suites']

    n_eps      = len(ep_lengths)
    ep_starts  = np.concatenate([[0], np.cumsum(ep_lengths[:-1])])
    action_dim = raw_actions.shape[1]

    # Split each episode into CHUNK-sized windows
    ep_chunks: list[list[np.ndarray]] = []
    for e in range(n_eps):
        s, L = ep_starts[e], ep_lengths[e]
        data = raw_actions[s:s + L]
        windows = [data[i:i + CHUNK] for i in range(0, L - CHUNK + 1, CHUNK)]
        if not windows:
            windows = [data]
        ep_chunks.append(windows)

    state = {'ep': 0, 'chunk': 0, 'k': 5}

    # ── figure layout ──────────────────────────────────────────────────────────
    n_rows = (action_dim + 2) // 3          # enough rows for action_dim subplots
    fig = plt.figure(figsize=(15, 3 * n_rows + 1))
    gs  = gridspec.GridSpec(n_rows, 3, figure=fig,
                            hspace=0.55, wspace=0.35,
                            left=0.06, right=0.74, top=0.93, bottom=0.08)

    joint_axes = []
    for i in range(action_dim):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        joint_axes.append(ax)

    # Widget panel (right side)
    ax_ep    = fig.add_axes([0.80, 0.82, 0.16, 0.03])
    ax_chunk = fig.add_axes([0.80, 0.74, 0.16, 0.03])
    ax_radio = fig.add_axes([0.80, 0.32, 0.16, 0.38])
    ax_info  = fig.add_axes([0.80, 0.08, 0.16, 0.20])
    ax_info.axis('off')

    sl_ep    = Slider(ax_ep,    'Episode', 0, n_eps - 1, valinit=0, valstep=1)
    sl_chunk = Slider(ax_chunk, 'Chunk',   0,
                      max(len(ep_chunks[0]) - 1, 0), valinit=0, valstep=1)
    radio    = RadioButtons(ax_radio, [str(k) for k in K_OPTIONS],
                            active=K_OPTIONS.index(5))
    ax_radio.set_title('k (stride)', fontsize=9)

    info_text = ax_info.text(0.05, 0.98, '', transform=ax_info.transAxes,
                              va='top', fontsize=9, family='monospace')

    # Lines per joint
    raw_lines, proc_lines = [], []
    for i, ax in enumerate(joint_axes):
        rl, = ax.plot([], [], color='steelblue',  lw=1.2, label='raw')
        pl, = ax.plot([], [], color='darkorange', lw=1.5, ls='--', label='AJAS')
        raw_lines.append(rl)
        proc_lines.append(pl)
        ax.set_title(f'Joint {i + 1}', fontsize=8)
        ax.set_xlabel('timestep', fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    # ── update ─────────────────────────────────────────────────────────────────
    def redraw():
        e = state['ep']
        c = min(state['chunk'], len(ep_chunks[e]) - 1)
        state['chunk'] = c
        k = state['k']

        chunk    = ep_chunks[e][c]
        smoothed = ajas_smooth(chunk, k)
        sm       = compute_sm(chunk)
        t        = np.arange(len(chunk))

        for i, ax in enumerate(joint_axes):
            raw_lines[i].set_data(t, chunk[:, i])
            proc_lines[i].set_data(t, smoothed[:, i])
            ax.relim()
            ax.autoscale_view()
            ax.set_title(f'Joint {i+1}   Sm={sm[i]*1e3:.3f}×10⁻³', fontsize=8)

        keyframes = list(range(0, len(chunk), k))
        if keyframes[-1] != len(chunk) - 1:
            keyframes.append(len(chunk) - 1)

        sm_lines = '\n'.join(
            f'  J{j+1}: {sm[j]*1e3:.3f}' for j in range(len(sm))
        )
        info_text.set_text(
            f'Episode : {e}\n'
            f'Chunk   : {c} / {len(ep_chunks[e]) - 1}\n'
            f'k       : {k}  ({len(keyframes)} keyframes)\n'
            f'Task id : {task_ids[e]}\n'
            f'Suite   : {suites[e]}\n'
            f'Sm ×10⁻³ per joint:\n'
            f'{sm_lines}\n'
            f'  mean: {sm.mean()*1e3:.3f}'
        )
        fig.canvas.draw_idle()

    def on_ep(val):
        state['ep'] = int(sl_ep.val)
        n_chunks = len(ep_chunks[state['ep']])
        sl_chunk.valmax = max(n_chunks - 1, 0)
        sl_chunk.set_val(0)
        state['chunk'] = 0
        redraw()

    def on_chunk(val):
        state['chunk'] = int(sl_chunk.val)
        redraw()

    def on_k(label):
        state['k'] = int(label)
        redraw()

    sl_ep.on_changed(on_ep)
    sl_chunk.on_changed(on_chunk)
    radio.on_clicked(on_k)

    fig.suptitle(
        'AJAS Tuning  —  raw (blue) vs AJAS-smoothed (orange dashed)',
        fontsize=11)
    redraw()
    plt.show()


if __name__ == '__main__':
    default = 'results/eval_chunk50_test/episodes.npz'
    path = sys.argv[1] if len(sys.argv) > 1 else default
    main(path)
