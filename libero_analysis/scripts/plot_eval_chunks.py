"""
Eval episode plotter.

Loads one or two episodes.npz files and plots raw_actions vs proc_actions
overlaid per joint. The Sm and k rows show local sliding-window values computed
at every timestep, with a slider to adjust the window size live.

Usage
-----
# Single file
python libero_analysis/scripts/plot_eval_chunks.py results/eval_ajas_laptop_kmin1/c500/libero_object/episodes.npz

# Two files — compare datasets
python libero_analysis/scripts/plot_eval_chunks.py \\
    results/eval_baseline_chunk50/libero_object/episodes.npz \\
    results/eval_ajas_laptop_kmin1/c500/libero_object/episodes.npz

# Custom AJAS params
python libero_analysis/scripts/plot_eval_chunks.py episodes.npz --C 500 --k-min 1 --k-max 40
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, CheckButtons
from pathlib import Path

CHUNK       = 50
DT          = 1.0 / 30.0
JOINT_NAMES = [f"J{i+1}" for i in range(7)]
JOINT_COLORS = [plt.cm.tab10(i / 10) for i in range(7)]

STYLES = [
    ("steelblue",  "darkorange", "A: "),
    ("seagreen",   "crimson",    "B: "),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _sm_per_joint(chunk: np.ndarray) -> np.ndarray:
    fs = 1.0 / DT
    scores = []
    for j in range(chunk.shape[1]):
        x = chunk[:, j] - np.mean(chunk[:, j])
        F = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=DT)
        n = len(freqs)
        scores.append((2.0 / (n * fs)) * float(np.sum(F * freqs)))
    return np.array(scores, dtype=np.float32)


def _sm_per_joint_norm(chunk: np.ndarray) -> np.ndarray:
    """Sm with per-joint std-normalisation (visualisation only — not used for k)."""
    fs = 1.0 / DT
    scores = []
    for j in range(chunk.shape[1]):
        x = chunk[:, j] - np.mean(chunk[:, j])
        std = np.std(x)
        if std > 1e-9:
            x = x / std
        F = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=DT)
        n = len(freqs)
        scores.append((2.0 / (n * fs)) * float(np.sum(F * freqs)))
    return np.array(scores, dtype=np.float32)


def compute_sliding_sm_k(raw_ep: np.ndarray, C: float, k_min: int, k_max: int, sm_window: int,
                          sm_fn=None):
    """Compute Sm and k at every timestep using a forward sliding window.

    Returns sm_arr and k_arr, each shape (L, action_dim).
    At timesteps near the end where the window is short, falls back to the
    last valid computation.
    sm_fn controls which Sm function is used; defaults to _sm_per_joint (raw).
    """
    if sm_fn is None:
        sm_fn = _sm_per_joint
    L, action_dim = raw_ep.shape
    sm_arr = np.zeros((L, action_dim), dtype=np.float32)
    k_arr  = np.zeros((L, action_dim), dtype=np.int32)
    last_valid_sm = np.zeros(action_dim, dtype=np.float32)

    for t in range(L):
        window = raw_ep[t:min(L, t + sm_window)]
        if len(window) >= 4:
            sm = sm_fn(window)
            last_valid_sm = sm
        else:
            sm = last_valid_sm
        sm_arr[t] = sm
        k_arr[t]  = np.clip(np.round(C * sm).astype(int), k_min, k_max)

    return sm_arr, k_arr


def load_npz(path):
    npz     = np.load(path, allow_pickle=True)
    raw     = npz["raw_actions"]
    proc    = npz["proc_actions"]
    lengths = npz["ep_lengths"].astype(int)
    starts  = np.concatenate([[0], np.cumsum(lengths[:-1])])

    episodes = []
    for e in range(len(lengths)):
        s, L = starts[e], lengths[e]
        episodes.append((raw[s:s + L], proc[s:s + L]))

    return dict(
        episodes=episodes,
        success=npz["successes"],
        tasks=npz["task_ids"],
        suites=npz["suites"],
        n_eps=len(lengths),
        label=Path(path).parts[-2],
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main(paths, C=2000.0, k_min=1, k_max=40, sm_window_init=15):
    datasets   = [load_npz(p) for p in paths]
    n_ds       = len(datasets)
    action_dim = 7

    state = {"ep": 0, "show": [True] * (n_ds * 2), "t0": 0, "t1": 500, "sm_win": sm_window_init,
             "norm_sm": False}

    # ── figure layout ────────────────────────────────────────────────────────
    n_jrows = (action_dim + 2) // 3
    fig = plt.figure(figsize=(16, 3 * n_jrows + 4))

    gs = gridspec.GridSpec(
        n_jrows + 2, 3,
        figure=fig,
        height_ratios=[1] * n_jrows + [0.55, 0.55],
        hspace=0.65, wspace=0.35,
        left=0.06, right=0.72, top=0.93, bottom=0.06,
    )

    joint_axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(action_dim)]
    ax_sm = fig.add_subplot(gs[n_jrows,     :])
    ax_k  = fig.add_subplot(gs[n_jrows + 1, :])
    all_axes = joint_axes + [ax_sm, ax_k]

    # Right-panel widgets — compact to fit sm_window slider
    ax_ep     = fig.add_axes([0.78, 0.91, 0.18, 0.023])
    ax_tstart = fig.add_axes([0.78, 0.86, 0.18, 0.023])
    ax_tend   = fig.add_axes([0.78, 0.81, 0.18, 0.023])
    ax_smwin  = fig.add_axes([0.78, 0.76, 0.18, 0.023])
    ax_norm   = fig.add_axes([0.78, 0.73, 0.18, 0.025])
    ax_check  = fig.add_axes([0.78, 0.40, 0.18, 0.32])
    ax_info   = fig.add_axes([0.78, 0.06, 0.18, 0.30])
    ax_info.axis("off")

    n_eps_max = max(ds["n_eps"] for ds in datasets)
    sl_ep     = Slider(ax_ep,    "Episode",   0, n_eps_max - 1,  valinit=0,              valstep=1)
    sl_tstart = Slider(ax_tstart,"T start",   0, 999,             valinit=0,              valstep=1)
    sl_tend   = Slider(ax_tend,  "T end",     1, 1000,            valinit=500,            valstep=1)
    sl_smwin  = Slider(ax_smwin, "Sm window", 4, CHUNK,           valinit=sm_window_init, valstep=1)
    norm_check = CheckButtons(ax_norm, ["Normalise Sm"], [False])

    check_labels = []
    for i, ds in enumerate(datasets):
        pfx = STYLES[i][2]
        check_labels += [f"{pfx}raw ({ds['label']})", f"{pfx}proc ({ds['label']})"]
    check = CheckButtons(ax_check, check_labels, state["show"])

    info_text = ax_info.text(0.03, 0.97, "", transform=ax_info.transAxes,
                              va="top", fontsize=7.5, family="monospace")

    # ── action lines ─────────────────────────────────────────────────────────
    lines = []
    for i, ds in enumerate(datasets):
        raw_c, proc_c, pfx = STYLES[i]
        ds_lines = [[], []]
        for j, ax in enumerate(joint_axes):
            rl, = ax.plot([], [], color=raw_c,  lw=1.2,
                          label=f"{pfx}raw" if j == 0 else "_")
            pl, = ax.plot([], [], color=proc_c, lw=1.5, ls="--",
                          label=f"{pfx}proc" if j == 0 else "_")
            ds_lines[0].append(rl)
            ds_lines[1].append(pl)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")
        lines.append(ds_lines)

    # ── Sm / k sliding-window lines — per joint (J7/gripper excluded) ───────────
    SM_SKIP = {6}   # gripper Sm dominates scale; exclude from metric plots
    sm_lines, k_lines = [], []
    for j in range(action_dim):
        label = JOINT_NAMES[j] if j not in SM_SKIP else "_"
        sl, = ax_sm.plot([], [], color=JOINT_COLORS[j], lw=1.2, label=label,
                         visible=(j not in SM_SKIP))
        kl, = ax_k.plot([], [],  color=JOINT_COLORS[j], lw=1.2,
                        visible=(j not in SM_SKIP))
        sm_lines.append(sl)
        k_lines.append(kl)

    ax_sm.set_ylabel("Sm  (local window)", fontsize=8)
    ax_sm.set_xlabel("timestep", fontsize=7)
    ax_sm.tick_params(labelsize=7)
    ax_sm.legend(fontsize=6, ncol=7, loc="upper right")

    ax_k.set_ylabel("k  (local)", fontsize=8)
    ax_k.set_xlabel("timestep", fontsize=7)
    ax_k.tick_params(labelsize=7)

    for j, ax in enumerate(joint_axes):
        ax.set_title(JOINT_NAMES[j], fontsize=8)
        ax.set_xlabel("timestep", fontsize=7)
        ax.tick_params(labelsize=7)

    vline_cols = [[] for _ in all_axes]

    # ── redraw ───────────────────────────────────────────────────────────────
    def redraw():
        e      = state["ep"]
        t0     = state["t0"]
        t1     = state["t1"]
        sm_win = state["sm_win"]
        info_lines = []
        max_len = 0

        for i, ds in enumerate(datasets):
            e_i = min(e, ds["n_eps"] - 1)
            raw_ep, proc_ep = ds["episodes"][e_i]
            max_len = max(max_len, len(raw_ep))
            t_ep = np.arange(len(raw_ep))
            show_raw  = state["show"][i * 2]
            show_proc = state["show"][i * 2 + 1]

            for j in range(action_dim):
                lines[i][0][j].set_data(
                    t_ep if show_raw  else [], raw_ep[:, j]  if show_raw  else [])
                lines[i][1][j].set_data(
                    t_ep if show_proc else [], proc_ep[:, j] if show_proc else [])

            n_chunks = int(np.ceil(len(raw_ep) / CHUNK))
            info_lines.append(
                f"{STYLES[i][2]}{ds['label']}\n"
                f"  ep={e_i}  len={len(raw_ep)}\n"
                f"  chunks={n_chunks}\n"
                f"  task={ds['tasks'][e_i]} "
                f"{'✓' if ds['success'][e_i] else '✗'}\n"
                f"  suite={ds['suites'][e_i]}\n"
            )

        # Sliding-window Sm / k from dataset A
        e_0 = min(e, datasets[0]["n_eps"] - 1)
        raw_ep0, _ = datasets[0]["episodes"][e_0]
        t_ax = np.arange(len(raw_ep0))
        # k always from raw Sm (C is calibrated for it); display can be normalised
        _, k_arr = compute_sliding_sm_k(raw_ep0, C, k_min, k_max, sm_win)
        disp_fn = _sm_per_joint_norm if state["norm_sm"] else _sm_per_joint
        sm_disp, _ = compute_sliding_sm_k(raw_ep0, C, k_min, k_max, sm_win, sm_fn=disp_fn)
        for j in range(action_dim):
            if j in SM_SKIP:
                continue
            sm_lines[j].set_data(t_ax, sm_disp[:, j])
            k_lines[j].set_data(t_ax, k_arr[:, j])

        # Chunk vlines + relim + window
        for ai, ax in enumerate(all_axes):
            for vl in vline_cols[ai]:
                vl.remove()
            vline_cols[ai].clear()
            for x in range(CHUNK, max_len, CHUNK):
                vl = ax.axvline(x, color="grey", lw=0.5, ls=":", alpha=0.5)
                vline_cols[ai].append(vl)
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(t0, t1)

        info_text.set_text("\n".join(info_lines))
        fig.canvas.draw_idle()

    def apply_window():
        t0 = max(0, int(sl_tstart.val))
        t1 = int(sl_tend.val)
        if t1 <= t0:
            t1 = t0 + 1
        state["t0"], state["t1"] = t0, t1
        for ax in all_axes:
            ax.set_xlim(t0, t1)
        fig.canvas.draw_idle()

    def on_ep(val):
        e = int(sl_ep.val)
        state["ep"] = e
        max_len = max(
            len(ds["episodes"][min(e, ds["n_eps"] - 1)][0]) for ds in datasets
        )
        sl_tstart.set_val(0)
        sl_tend.valmax = max_len
        sl_tend.ax.set_xlim(1, max_len)
        sl_tend.set_val(max_len)
        state["t0"], state["t1"] = 0, max_len
        redraw()

    def on_smwin(val):
        state["sm_win"] = int(sl_smwin.val)
        redraw()

    def on_norm(label):
        state["norm_sm"] = not state["norm_sm"]
        ax_sm.set_ylabel(
            "Sm  (std-normalised)" if state["norm_sm"] else "Sm  (local window)", fontsize=8)
        redraw()

    sl_ep.on_changed(on_ep)
    sl_tstart.on_changed(lambda v: apply_window())
    sl_tend.on_changed(lambda v: apply_window())
    sl_smwin.on_changed(on_smwin)
    norm_check.on_clicked(on_norm)
    check.on_clicked(lambda label: (
        state["show"].__setitem__(check_labels.index(label),
                                  not state["show"][check_labels.index(label)]),
        redraw()
    ))

    title = "  vs  ".join(f"{Path(p).parts[-3]}/{Path(p).parts[-2]}" for p in paths)
    fig.suptitle(
        f"Episode overlay — {title}\n"
        f"raw (solid) vs proc (dashed) | grey = chunk boundaries | "
        f"Sm/k: C={C} k_min={k_min} k_max={k_max}",
        fontsize=9,
    )
    on_ep(0)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive episode plotter")
    parser.add_argument("paths",       nargs="+", help="Path(s) to episodes.npz")
    parser.add_argument("--C",         type=float, default=2000.0)
    parser.add_argument("--k-min",     type=int,   default=1)
    parser.add_argument("--k-max",     type=int,   default=40)
    parser.add_argument("--sm-window", type=int,   default=15,
                        help="Initial sliding-window size for local Sm (default: 15)")
    args = parser.parse_args()
    main(args.paths, C=args.C, k_min=args.k_min, k_max=args.k_max,
         sm_window_init=args.sm_window)
