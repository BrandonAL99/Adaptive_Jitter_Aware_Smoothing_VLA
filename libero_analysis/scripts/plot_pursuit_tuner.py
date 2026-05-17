"""
Processor comparison tuner.

Loads one or more episodes.npz files and overlays proc_actions from each file
plus three live-tunable processors: low-pass filter, pure pursuit, AJAS spline.

For pursuit and spline, dots mark the lookahead / keyframe positions.

Usage
-----
# Primary file only (raw + proc from file + live traces)
python libero_analysis/scripts/plot_pursuit_tuner.py PRIMARY.npz

# With extra named files (one checkbox per file)
python libero_analysis/scripts/plot_pursuit_tuner.py PRIMARY.npz \\
    --file "LP 2Hz:results/eval_lowpass/lp2/libero_object/episodes.npz" \\
    --file "LP 6Hz:results/eval_lowpass/lp6/libero_object/episodes.npz" \\
    --file "Spline:results/eval_spline/libero_object/episodes.npz"
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, CheckButtons
from pathlib import Path
from scipy.signal import butter, filtfilt, lfilter
from scipy.interpolate import PchipInterpolator

DT           = 1.0 / 30.0
FS           = 1.0 / DT
CHUNK        = 50
JOINT_NAMES  = [f"J{i+1}" for i in range(7)]
JOINT_COLORS = [plt.cm.tab10(i / 10) for i in range(7)]
SKIP_JOINTS  = {6}

# colours for live traces
COL_RAW     = ("steelblue",    "-",   1.2)
COL_LP      = ("darkorange",   "--",  1.5)
COL_PURSUIT = ("seagreen",     ":",   1.8)
COL_SPLINE  = ("mediumpurple", "-.",  1.5)
FILE_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3",
               "#a65628", "#f781bf", "#999999"]


# ── processors ─────────────────────────────────────────────────────────────────

def _sm_norm(chunk: np.ndarray) -> np.ndarray:
    fs = FS
    scores = []
    for j in range(chunk.shape[1]):
        x   = chunk[:, j] - np.mean(chunk[:, j])
        std = np.std(x)
        if std > 1e-9:
            x = x / std
        F     = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=DT)
        n     = len(freqs)
        scores.append((2.0 / (n * fs)) * float(np.sum(F * freqs)))
    return np.array(scores, dtype=np.float32)


def apply_lowpass(ep, cutoff_hz, order=4, causal=True):
    nyq = 0.5 * FS
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    fn   = lfilter if causal else filtfilt
    out  = ep.copy()
    for j in range(ep.shape[1]):
        out[:, j] = fn(b, a, ep[:, j])
    return out


def simulate_pursuit(raw_ep, C, P, k_min, k_max, sm_window, kf_window=1):
    """Returns (sim, sm_arr, k_arr, dot_t, dot_v) where dot_* are keyframe positions."""
    L, D = raw_ep.shape
    sim    = np.zeros_like(raw_ep)
    sm_arr = np.zeros((L, D), dtype=np.float32)
    k_arr  = np.zeros((L, D), dtype=np.int32)
    dot_t  = {j: [] for j in range(D)}   # lookahead dot x (target timestep)
    dot_v  = {j: [] for j in range(D)}   # lookahead dot y (target value)

    cur     = raw_ep[0].copy()
    last_sm = np.zeros(D, dtype=np.float32)

    for t in range(L):
        win = raw_ep[t:min(L, t + sm_window)]
        if len(win) >= 4:
            last_sm = _sm_norm(win)
        ks = np.clip(np.round(C * last_sm).astype(int), k_min, k_max)
        sm_arr[t] = last_sm
        k_arr[t]  = ks

        action = raw_ep[t].copy()
        for j, k in enumerate(ks):
            if j in SKIP_JOINTS:
                continue
            centre = min(t + int(k), L - 1)
            lo     = max(0, centre - kf_window)
            hi     = min(L - 1, centre + kf_window)
            target = float(np.mean(raw_ep[lo:hi + 1, j]))
            action[j] = cur[j] + P * (target - cur[j])
            dot_t[j].append(centre)
            dot_v[j].append(target)

        sim[t] = action
        cur    = action.copy()

    dot_t = {j: np.array(v) for j, v in dot_t.items()}
    dot_v = {j: np.array(v) for j, v in dot_v.items()}
    return sim, sm_arr, k_arr, dot_t, dot_v


def apply_spline(raw_ep, C, k_min, k_max, sm_window, kf_window):
    """Returns (smoothed, kf_t, kf_v) where kf_* are keyframe positions per joint."""
    L, D     = raw_ep.shape
    smoothed = raw_ep.copy()
    kf_t     = {j: [] for j in range(D)}
    kf_v     = {j: [] for j in range(D)}

    for t_start in range(0, L, CHUNK):
        chunk = raw_ep[t_start:min(t_start + CHUNK, L)]
        cL    = len(chunk)
        if cL < 4:
            smoothed[t_start:t_start + cL] = chunk
            continue

        for j in range(D):
            if j in SKIP_JOINTS:
                continue

            last_sm_j = 0.0
            t = 0
            kf_idx = [0]
            while t < cL - 1:
                win = chunk[t:min(cL, t + sm_window)]
                if len(win) >= 4:
                    last_sm_j = float(_sm_norm(win)[j])
                k      = int(np.clip(round(C * last_sm_j), k_min, k_max))
                next_t = min(t + k, cL - 1)
                kf_idx.append(next_t)
                t = next_t
            if kf_idx[-1] != cL - 1:
                kf_idx.append(cL - 1)

            n_kf   = len(kf_idx)
            kf_pos = []
            for ii, ki in enumerate(kf_idx):
                lo_gap = (ki - kf_idx[ii - 1]) // 2 if ii > 0         else ki
                hi_gap = (kf_idx[ii + 1] - ki) // 2 if ii < n_kf - 1 else cL - 1 - ki
                eff_w  = min(kf_window, lo_gap, hi_gap)
                lo     = max(0,       ki - eff_w)
                hi     = min(cL - 1,  ki + eff_w)
                kf_pos.append(float(np.mean(chunk[lo:hi + 1, j])))

            kf_t_arr = np.array(kf_idx, dtype=float)
            kf_p_arr = np.array(kf_pos, dtype=float)
            cs       = PchipInterpolator(kf_t_arr, kf_p_arr)
            smoothed[t_start:t_start + cL, j] = cs(np.arange(cL))

            for ki, kp in zip(kf_idx, kf_pos):
                kf_t[j].append(t_start + ki)
                kf_v[j].append(kp)

    kf_t = {j: np.array(v) for j, v in kf_t.items()}
    kf_v = {j: np.array(v) for j, v in kf_v.items()}
    return smoothed.astype(np.float32), kf_t, kf_v


# ── npz loader ─────────────────────────────────────────────────────────────────

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
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main(primary_path, extra_files, C_init, P_init, k_min, k_max, sm_window_init):
    primary = load_npz(primary_path)
    n_eps   = primary["n_eps"]
    action_dim = 7

    # load extra files — align to primary by episode index
    file_datasets = []   # list of (label, colour, episodes_list)
    for i, (label, path) in enumerate(extra_files):
        ds  = load_npz(path)
        col = FILE_COLORS[i % len(FILE_COLORS)]
        file_datasets.append((label, col, ds["episodes"]))

    state = {
        "ep":  0, "t0": 0, "t1": 500,
        "lp_cutoff":   4.0,  "lp_causal": True,
        "C":  C_init,  "P": P_init, "sm_win": sm_window_init,
        "sp_C": 100.0, "sp_sm": 15, "sp_kfw": 2,
        # visibility
        "show_raw":     True,
        "show_lp":      True,
        "show_pursuit": True,
        "show_spline":  True,
        "show_pursuit_dots": True,
        "show_spline_dots":  True,
    }
    for label, _, _ in file_datasets:
        state[f"show_file_{label}"] = True

    # ── layout ───────────────────────────────────────────────────────────────
    n_jrows = (action_dim + 2) // 3
    fig = plt.figure(figsize=(20, 4 * n_jrows + 5))

    gs = gridspec.GridSpec(
        n_jrows + 2, 3, figure=fig,
        height_ratios=[1] * n_jrows + [0.55, 0.55],
        hspace=0.65, wspace=0.35,
        left=0.05, right=0.70, top=0.93, bottom=0.06,
    )
    joint_axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(action_dim)]
    ax_sm = fig.add_subplot(gs[n_jrows,     :])
    ax_k  = fig.add_subplot(gs[n_jrows + 1, :])
    all_axes = joint_axes + [ax_sm, ax_k]

    # ── right-panel sliders ───────────────────────────────────────────────────
    SL, SW, SH = 0.745, 0.225, 0.022
    y = 0.945

    def sl_ax():
        nonlocal y
        y -= 0.032
        return fig.add_axes([SL, y, SW, SH])

    ax_ep     = sl_ax()
    ax_tstart = sl_ax()
    ax_tend   = sl_ax()
    y -= 0.008   # gap before LP
    ax_lp     = sl_ax()
    y -= 0.008
    ax_C      = sl_ax()
    ax_P      = sl_ax()
    y -= 0.008
    ax_sp_c   = sl_ax()
    ax_sp_sm  = sl_ax()
    ax_sp_kfw = sl_ax()

    sl_ep     = Slider(ax_ep,     "Episode",      0, n_eps - 1, valinit=0,   valstep=1)
    sl_tstart = Slider(ax_tstart, "T start",      0, 999,        valinit=0,   valstep=1)
    sl_tend   = Slider(ax_tend,   "T end",        1, 1000,       valinit=500, valstep=1)
    sl_lp     = Slider(ax_lp,     "LP cutoff Hz", 0.5, 14.0,    valinit=4.0, valstep=0.5)
    sl_C      = Slider(ax_C,      "Pursuit C",    100, 15000,    valinit=C_init,        valstep=100)
    sl_P      = Slider(ax_P,      "Pursuit P",    0.05, 1.0,     valinit=P_init,        valstep=0.05)
    sl_sp_c   = Slider(ax_sp_c,   "Spline C",     100, 5000,     valinit=state["sp_C"], valstep=100)
    sl_sp_sm  = Slider(ax_sp_sm,  "Spline sm_win",  4,   50,     valinit=state["sp_sm"],valstep=1)
    sl_sp_kfw = Slider(ax_sp_kfw, "Spline kf_win",  0,    5,     valinit=state["sp_kfw"],valstep=1)

    # ── checkboxes ───────────────────────────────────────────────────────────
    # Traces checkbox
    trace_labels = (
        ["raw", "LP (live)", "Pursuit (live)", "Spline (live)"]
        + [lbl for lbl, _, _ in file_datasets]
    )
    trace_inits  = [True] * len(trace_labels)
    trace_ch     = 0.032 * len(trace_labels) + 0.02
    y -= 0.015
    ax_traces = fig.add_axes([SL, y - trace_ch, SW, trace_ch])
    check_traces = CheckButtons(ax_traces, trace_labels, trace_inits)
    y -= trace_ch

    # Dots checkbox
    dot_labels = ["Pursuit dots", "Spline dots"]
    dot_ch     = 0.032 * len(dot_labels) + 0.02
    y -= 0.01
    ax_dots = fig.add_axes([SL, y - dot_ch, SW, dot_ch])
    check_dots = CheckButtons(ax_dots, dot_labels, [True, True])

    # info text
    ax_info = fig.add_axes([SL, 0.04, SW, max(0.03, y - dot_ch - 0.055)])
    ax_info.axis("off")
    info_text = ax_info.text(0.03, 0.97, "", transform=ax_info.transAxes,
                             va="top", fontsize=7.5, family="monospace")

    # ── plot lines ───────────────────────────────────────────────────────────
    raw_lines     = []
    lp_lines      = []
    pursuit_lines = []
    spline_lines  = []
    file_lines    = [[] for _ in file_datasets]
    pursuit_dots  = []
    spline_dots   = []

    for j, ax in enumerate(joint_axes):
        c, ls, lw = COL_RAW
        rl, = ax.plot([], [], color=c, lw=lw, ls=ls, label="raw" if j == 0 else "_")
        raw_lines.append(rl)

        c, ls, lw = COL_LP
        ll, = ax.plot([], [], color=c, lw=lw, ls=ls, label="LP" if j == 0 else "_")
        lp_lines.append(ll)

        c, ls, lw = COL_PURSUIT
        pl, = ax.plot([], [], color=c, lw=lw, ls=ls, label="Pursuit" if j == 0 else "_")
        pursuit_lines.append(pl)
        pd, = ax.plot([], [], "o", color=c, ms=2, alpha=0.5, zorder=5,
                      label="_")
        pursuit_dots.append(pd)

        c, ls, lw = COL_SPLINE
        sl2, = ax.plot([], [], color=c, lw=lw, ls=ls, label="Spline" if j == 0 else "_")
        spline_lines.append(sl2)
        sd, = ax.plot([], [], "o", color=c, ms=3, zorder=5, label="_")
        spline_dots.append(sd)

        for i, (lbl, col, _) in enumerate(file_datasets):
            fl, = ax.plot([], [], color=col, lw=1.4, ls="--",
                          label=lbl if j == 0 else "_")
            file_lines[i].append(fl)

        ax.set_title(JOINT_NAMES[j], fontsize=8)
        ax.set_xlabel("timestep", fontsize=7)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")

    # Sm / k lines
    sm_lines, k_lines = [], []
    for j in range(action_dim):
        lbl = JOINT_NAMES[j] if j not in SKIP_JOINTS else "_"
        sl3, = ax_sm.plot([], [], color=JOINT_COLORS[j], lw=1.2, label=lbl,
                          visible=(j not in SKIP_JOINTS))
        kl,  = ax_k.plot([], [],  color=JOINT_COLORS[j], lw=1.2,
                          visible=(j not in SKIP_JOINTS))
        sm_lines.append(sl3)
        k_lines.append(kl)

    ax_sm.set_ylabel("Sm norm (pursuit window)", fontsize=8)
    ax_sm.set_xlabel("timestep", fontsize=7)
    ax_sm.tick_params(labelsize=7)
    ax_sm.legend(fontsize=6, ncol=7, loc="upper right")
    ax_k.set_ylabel("k  (pursuit lookahead)", fontsize=8)
    ax_k.set_xlabel("timestep", fontsize=7)
    ax_k.tick_params(labelsize=7)

    vline_cols = [[] for _ in all_axes]

    # ── redraw ───────────────────────────────────────────────────────────────
    def redraw():
        e      = state["ep"]
        t0, t1 = state["t0"], state["t1"]
        t_ax   = np.arange(len(primary["episodes"][e][0]))

        raw_ep, _ = primary["episodes"][e]
        L = len(raw_ep)

        # compute live traces
        lp_ep   = apply_lowpass(raw_ep, state["lp_cutoff"], causal=state["lp_causal"])
        pur_ep, sm_arr, k_arr, pur_dt, pur_dv = simulate_pursuit(
            raw_ep, state["C"], state["P"], k_min, k_max, state["sm_win"])
        sp_ep, sp_kt, sp_kv = apply_spline(
            raw_ep, state["sp_C"], k_min, k_max, state["sp_sm"], state["sp_kfw"])

        def vis(key):
            return state.get(key, True)

        for j in range(action_dim):
            def _set(line, data, key):
                show = t_ax if vis(key) else []
                line.set_data(show, data[:, j] if vis(key) else [])

            _set(raw_lines[j],     raw_ep, "show_raw")
            _set(lp_lines[j],      lp_ep,  "show_lp")
            _set(pursuit_lines[j], pur_ep, "show_pursuit")
            _set(spline_lines[j],  sp_ep,  "show_spline")

            # pursuit dots
            if vis("show_pursuit") and vis("show_pursuit_dots") and j not in SKIP_JOINTS:
                dt, dv = pur_dt[j], pur_dv[j]
                mask = (dt >= t0) & (dt < t1)
                pursuit_dots[j].set_data(dt[mask], dv[mask])
            else:
                pursuit_dots[j].set_data([], [])

            # spline keyframe dots
            if vis("show_spline") and vis("show_spline_dots") and j not in SKIP_JOINTS:
                kt, kv = sp_kt.get(j, np.array([])), sp_kv.get(j, np.array([]))
                if len(kt):
                    mask = (kt >= t0) & (kt < t1)
                    spline_dots[j].set_data(kt[mask], kv[mask])
                else:
                    spline_dots[j].set_data([], [])
            else:
                spline_dots[j].set_data([], [])

            # file traces
            for i, (lbl, _, eps) in enumerate(file_datasets):
                key = f"show_file_{lbl}"
                if vis(key) and e < len(eps):
                    _, proc = eps[e]
                    ln = min(L, len(proc))
                    file_lines[i][j].set_data(t_ax[:ln], proc[:ln, j])
                else:
                    file_lines[i][j].set_data([], [])

        # Sm / k from pursuit computation
        for j in range(action_dim):
            if j in SKIP_JOINTS:
                continue
            sm_lines[j].set_data(t_ax, sm_arr[:, j])
            k_lines[j].set_data( t_ax, k_arr[:, j])

        # chunk vlines + axis limits
        for ai, ax in enumerate(all_axes):
            for vl in vline_cols[ai]:
                vl.remove()
            vline_cols[ai].clear()
            for x in range(CHUNK, L, CHUNK):
                vline_cols[ai].append(
                    ax.axvline(x, color="grey", lw=0.5, ls=":", alpha=0.4))
            ax.relim(); ax.autoscale_view()
            ax.set_xlim(t0, t1)
        ax_k.axhline(k_min, color="grey", lw=0.8, ls="--", alpha=0.4)
        ax_k.axhline(k_max, color="grey", lw=0.8, ls="--", alpha=0.4)

        info_text.set_text(
            f"ep={e}/{n_eps}  T={L}\n"
            f"task={primary['tasks'][e]} "
            f"{'✓' if primary['success'][e] else '✗'}\n"
            f"suite={primary['suites'][e]}\n"
        )
        fig.canvas.draw_idle()

    # ── callbacks ────────────────────────────────────────────────────────────
    def on_ep(val):
        e = int(sl_ep.val)
        state["ep"] = e
        L = len(primary["episodes"][e][0])
        sl_tend.valmax = L
        sl_tend.ax.set_xlim(1, L)
        sl_tend.set_val(L)
        state["t0"], state["t1"] = 0, L
        redraw()

    def apply_window():
        t0 = max(0, int(sl_tstart.val))
        t1 = max(t0 + 1, int(sl_tend.val))
        state["t0"], state["t1"] = t0, t1
        for ax in all_axes:
            ax.set_xlim(t0, t1)
        fig.canvas.draw_idle()

    def on_trace(label):
        mapping = {
            "raw":           "show_raw",
            "LP (live)":     "show_lp",
            "Pursuit (live)":"show_pursuit",
            "Spline (live)": "show_spline",
        }
        key = mapping.get(label, f"show_file_{label}")
        state[key] = not state[key]
        redraw()

    def on_dot(label):
        mapping = {"Pursuit dots": "show_pursuit_dots", "Spline dots": "show_spline_dots"}
        state[mapping[label]] = not state[mapping[label]]
        redraw()

    sl_ep.on_changed(on_ep)
    sl_tstart.on_changed(lambda v: apply_window())
    sl_tend.on_changed(lambda v: apply_window())
    sl_lp.on_changed(    lambda v: (state.__setitem__("lp_cutoff", float(v)),    redraw()))
    sl_C.on_changed(     lambda v: (state.__setitem__("C",         float(v)),    redraw()))
    sl_P.on_changed(     lambda v: (state.__setitem__("P",         float(v)),    redraw()))
    sl_sp_c.on_changed(  lambda v: (state.__setitem__("sp_C",      float(v)),    redraw()))
    sl_sp_sm.on_changed( lambda v: (state.__setitem__("sp_sm",     int(v)),      redraw()))
    sl_sp_kfw.on_changed(lambda v: (state.__setitem__("sp_kfw",    int(v)),      redraw()))
    check_traces.on_clicked(on_trace)
    check_dots.on_clicked(on_dot)

    fig.suptitle(
        f"{Path(primary_path).parts[-2]}   "
        f"blue=raw  orange=LP  green=Pursuit  purple=Spline",
        fontsize=9,
    )
    on_ep(0)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path",       help="Primary episodes.npz (provides raw_actions)")
    parser.add_argument("--file",     action="append", default=[], metavar="LABEL:PATH",
                        help="Extra npz files to overlay, e.g. 'LP 4Hz:results/.../episodes.npz'")
    parser.add_argument("--C",        type=float, default=3000.0)
    parser.add_argument("--P",        type=float, default=0.5)
    parser.add_argument("--k-min",    type=int,   default=1)
    parser.add_argument("--k-max",    type=int,   default=40)
    parser.add_argument("--sm-window",type=int,   default=15)
    args = parser.parse_args()

    extra = []
    for f in args.file:
        if ":" not in f:
            parser.error(f"--file must be LABEL:PATH, got: {f!r}")
        label, path = f.split(":", 1)
        extra.append((label, path))

    main(args.path, extra, args.C, args.P, args.k_min, args.k_max, args.sm_window)
