"""
Processor comparison plotter.

Loads an episodes.npz and applies all configured processors offline so you
can compare their outputs interactively.  Edit the CONFIG BLOCK below to
change processor parameters, then use the checkboxes to toggle traces.

Usage
-----
python libero_analysis/scripts/plot_processor_compare.py \
    results/eval_ajas_laptop_kmin1/c500/libero_object/episodes.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, CheckButtons
from pathlib import Path
from scipy.signal import butter, lfilter
from scipy.interpolate import PchipInterpolator

DT    = 1.0 / 30.0
CHUNK = 50
JOINT_NAMES  = [f"J{i+1}" for i in range(7)]
JOINT_COLORS = [plt.cm.tab10(i / 10) for i in range(7)]
SKIP_JOINTS  = {6}


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG BLOCK — edit parameters here, comment out processors you don't want
# ══════════════════════════════════════════════════════════════════════════════

PROCESSOR_CONFIGS = {
    "lowpass": {
        "cutoff": 3.0,   # Hz
        "order":  2,
    },
    "ajas": {
        "C":          500,
        "k_min":      1,
        "k_max":      40,
        "kf_window":  1,
    },
    "pursuit": {
        "C":          3000,
        "P":          0.5,
        "k_min":      1,
        "k_max":      40,
        "sm_window":  15,
        "kf_window":  1,
    },
    "ajas_spline": {
        "C":          5000,
        "k_min":      1,
        "k_max":      40,
        "sm_window":  15,   # forward window size for local Sm at each keyframe
        "kf_window":  1,
    },
}

# Colours for each processor (raw is always steelblue)
PROC_COLORS = {
    "raw":         ("steelblue",  "-",  1.2),
    "lowpass":     ("darkorange", "--", 1.5),
    "ajas":        ("seagreen",   ":",  1.8),
    "ajas_spline": ("mediumpurple", "--", 1.5),
    "pursuit":     ("crimson",    "-.", 1.5),
}

# ══════════════════════════════════════════════════════════════════════════════


# ── metric ─────────────────────────────────────────────────────────────────────

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
    """Sm with per-joint std-normalisation.
    Divides each joint's signal by its std before FFT so Sm measures
    relative jitter (fraction of signal range), not absolute amplitude.
    A joint with range ±0.06 rad gets the same Sm as one with ±1 rad
    if they are equally jittery relative to their own swing.
    """
    fs = 1.0 / DT
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


def _sliding_sm(raw_ep, sm_window):
    L, D = raw_ep.shape
    sm_arr = np.zeros((L, D), dtype=np.float32)
    last   = np.zeros(D, dtype=np.float32)
    for t in range(L):
        win = raw_ep[t:min(L, t + sm_window)]
        if len(win) >= 4:
            last = _sm_per_joint(win)
        sm_arr[t] = last
    return sm_arr


# ── processors ────────────────────────────────────────────────────────────────

def apply_lowpass(raw_ep, cutoff=3.0, order=2):
    nyq  = (1.0 / DT) / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    out  = raw_ep.copy()
    for j in range(raw_ep.shape[1]):
        out[:, j] = lfilter(b, a, raw_ep[:, j])
    return out


def apply_ajas(raw_ep, C=500, k_min=1, k_max=40, kf_window=1):
    L, D  = raw_ep.shape
    result = raw_ep.copy()
    for t in range(0, L, CHUNK):
        chunk = raw_ep[t:min(t + CHUNK, L)]
        cL    = len(chunk)
        if cL < 4:
            continue
        sm = _sm_per_joint(chunk)
        ks = np.clip(np.round(C * sm).astype(int), k_min, k_max)
        smoothed = chunk.copy()
        for j in range(D):
            if j in SKIP_JOINTS:
                continue
            k = int(ks[j])
            kf_idx = list(range(0, cL, k)) if k > 0 else list(range(cL))
            if kf_idx[-1] != cL - 1:
                kf_idx.append(cL - 1)
            kf_pos = []
            for ki in kf_idx:
                lo = max(0, ki - kf_window)
                hi = min(cL - 1, ki + kf_window)
                kf_pos.append(float(np.mean(chunk[lo:hi + 1, j])))
            smoothed[:, j] = np.interp(np.arange(cL), np.array(kf_idx), np.array(kf_pos))
        result[t:t + cL] = smoothed
    return result

def apply_ajas_spline(raw_ep, C=500, k_min=1, k_max=40, sm_window=15, kf_window=1):
    """AJAS with adaptive non-uniform keyframe placement (WIP — interpolation still linear).

    Instead of computing Sm once per chunk and using a fixed stride throughout,
    this version slides a window forward and re-evaluates jitter at each keyframe.
    The stride k adapts within the chunk:
      - jittery region  → large k  → jump far ahead  → sparse keyframes → smooth output
      - smooth region   → small k  → jump short       → dense keyframes  → detail preserved

    Algorithm per chunk per joint:
      1. Start at t=0
      2. Compute local Sm over the next sm_window steps from t
      3. Convert Sm → stride k  (high Sm → large k)
      4. Place next keyframe at t + k  (averaged over ±kf_window neighbours)
      5. Advance t to that keyframe and repeat from step 2
      6. Continue until end of chunk — keyframes are non-uniformly spaced
      7. Interpolate between keyframes to fill every timestep
         Uses PchipInterpolator for C1 monotone-preserving cubic interpolation
    """
    L, D   = raw_ep.shape
    result = raw_ep.copy()

    for t_start in range(0, L, CHUNK):
        chunk = raw_ep[t_start:min(t_start + CHUNK, L)]
        cL    = len(chunk)
        if cL < 4:
            continue

        smoothed = chunk.copy()

        for j in range(D):
            if j in SKIP_JOINTS:
                # Gripper (J7) skipped — smoothing it delays gripper open/close
                # and causes the robot to fail to grasp objects.
                continue

            # ── build adaptive keyframe list ──────────────────────────────
            kf_idx   = [0]   # always start at beginning of chunk
            last_sm_j = 0.0  # fallback if window is too short to compute Sm
            t = 0

            while t < cL - 1:
                # Step 2: local Sm from current position over next sm_window steps.
                # Using a forward window means k reflects upcoming jitter, not past.
                win = chunk[t:min(cL, t + sm_window)]
                if len(win) >= 4:
                    last_sm_j = float(_sm_per_joint_norm(win)[j])
                # else: too close to end of chunk, reuse last computed Sm

                # Step 3: convert Sm → stride k for this joint.
                # High Sm → large k → skip over jitter.
                # Low Sm  → small k → follow detail closely.
                k = int(np.clip(round(C * last_sm_j), k_min, k_max))

                # Step 4: next keyframe is k steps ahead (clamped to chunk end).
                next_t = min(t + k, cL - 1)
                kf_idx.append(next_t)
                t = next_t   # jump to new keyframe position

            # Always end at final timestep
            if kf_idx[-1] != cL - 1:
                kf_idx.append(cL - 1)

            # ── average ±kf_window neighbours at each keyframe ───────────
            # Cap the window at half the distance to the nearest neighbour so
            # overlapping windows can't wash out height differences between
            # adjacent keyframes (which causes horizontal splines in steep regions).
            kf_pos = []
            n_kf   = len(kf_idx)
            for ii, ki in enumerate(kf_idx):
                lo_gap = (ki - kf_idx[ii - 1]) // 2 if ii > 0           else ki
                hi_gap = (kf_idx[ii + 1] - ki) // 2 if ii < n_kf - 1   else cL - 1 - ki
                eff_w  = min(kf_window, lo_gap, hi_gap)
                lo     = max(0,       ki - eff_w)
                hi     = min(cL - 1,  ki + eff_w)
                kf_pos.append(float(np.mean(chunk[lo:hi + 1, j])))

            # ── spline interpolation between keyframes ────────────────────
            # PchipInterpolator: C1 monotone-preserving cubic — automatically
            # computes velocities that prevent overshooting and ensure the spline
            # rises/falls in the same direction as the keyframe positions.
            # No manual velocity estimation needed (avoids the near-zero-gradient
            # bug when keyframes straddle a steep but localised rise).
            kf_t = np.array(kf_idx, dtype=float)
            kf_p = np.array(kf_pos, dtype=float)
            cs   = PchipInterpolator(kf_t, kf_p)
            smoothed[:, j] = cs(np.arange(cL))

        result[t_start:t_start + cL] = smoothed
    return result


def get_ajas_spline_keyframes(raw_ep):
    """Return keyframe (t, value) arrays per joint for the current ajas_spline config.

    Used by the plotter to draw dots at interpolation points.
    Returns {joint_idx: (t_array, val_array)} across the full episode.
    """
    cfg = PROCESSOR_CONFIGS.get("ajas_spline", {})
    C         = cfg.get("C",         500)
    k_min     = cfg.get("k_min",     1)
    k_max     = cfg.get("k_max",     40)
    sm_window = cfg.get("sm_window", 15)
    kf_window = cfg.get("kf_window", 1)

    L, D = raw_ep.shape
    kf_ts  = {j: [] for j in range(D)}
    kf_vs  = {j: [] for j in range(D)}

    for t_start in range(0, L, CHUNK):
        chunk = raw_ep[t_start:min(t_start + CHUNK, L)]
        cL    = len(chunk)
        if cL < 4:
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
                    last_sm_j = float(_sm_per_joint_norm(win)[j])
                k   = int(np.clip(round(C * last_sm_j), k_min, k_max))
                next_t = min(t + k, cL - 1)
                kf_idx.append(next_t)
                t = next_t
            if kf_idx[-1] != cL - 1:
                kf_idx.append(cL - 1)

            n_kf = len(kf_idx)
            for ii, ki in enumerate(kf_idx):
                lo_gap = (ki - kf_idx[ii - 1]) // 2 if ii > 0         else ki
                hi_gap = (kf_idx[ii + 1] - ki) // 2 if ii < n_kf - 1 else cL - 1 - ki
                eff_w  = min(kf_window, lo_gap, hi_gap)
                lo     = max(0,       ki - eff_w)
                hi     = min(cL - 1,  ki + eff_w)
                kf_ts[j].append(t_start + ki)
                kf_vs[j].append(float(np.mean(chunk[lo:hi + 1, j])))

    return {j: (np.array(kf_ts[j]), np.array(kf_vs[j])) for j in range(D)}


def apply_pursuit(raw_ep, C=3000, P=0.5, k_min=1, k_max=40, sm_window=15, kf_window=1):
    L, D   = raw_ep.shape
    sim    = np.zeros_like(raw_ep)
    cur    = raw_ep[0].copy()
    last_sm = np.zeros(D, dtype=np.float32)
    for t in range(L):
        win = raw_ep[t:min(L, t + sm_window)]
        if len(win) >= 4:
            last_sm = _sm_per_joint(win)
        ks  = np.clip(np.round(C * last_sm).astype(int), k_min, k_max)
        act = raw_ep[t].copy()
        for j, k in enumerate(ks):
            if j in SKIP_JOINTS:
                continue
            centre = min(t + int(k), L - 1)
            lo = max(0, centre - kf_window)
            hi = min(L - 1, centre + kf_window)
            target = float(np.mean(raw_ep[lo:hi + 1, j]))
            act[j] = cur[j] + P * (target - cur[j])
        sim[t] = act
        cur = act.copy()
    return sim


APPLY_FN = {
    "lowpass":     apply_lowpass,
    "ajas":        apply_ajas,
    "ajas_spline": apply_ajas_spline,
    "pursuit":     apply_pursuit,
}


def apply_all(raw_ep):
    """Run every configured processor and return {name: array}.
    Also computes '_kf_spline' keyframe data if ajas_spline is configured.
    """
    results = {"raw": raw_ep}
    for name, cfg in PROCESSOR_CONFIGS.items():
        results[name] = APPLY_FN[name](raw_ep, **cfg)
    if "ajas_spline" in PROCESSOR_CONFIGS:
        results["_kf_spline"] = get_ajas_spline_keyframes(raw_ep)
    return results


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

def main(path, sm_window_init=15):
    ds = load_npz(path)
    action_dim  = 7
    proc_names  = ["raw"] + list(PROCESSOR_CONFIGS.keys())

    state = {
        "ep":        0,
        "t0":        0,
        "t1":        500,
        "sm_win":    sm_window_init,
        "show":      {n: True for n in proc_names},
        "show_dots": True,
        "cache":     {},   # ep_index → {proc_name: array}
    }

    def get_traces(e):
        if e not in state["cache"]:
            raw_ep, _ = ds["episodes"][e]
            state["cache"][e] = apply_all(raw_ep)
        return state["cache"][e]

    # ── layout ───────────────────────────────────────────────────────────────
    # Plots use the full figure width. The control panel floats in the
    # bottom-right corner as a white-backed overlay — no space is reserved
    # for it, so joint plots are full size.
    n_jrows = (action_dim + 2) // 3
    fig = plt.figure(figsize=(22, 4 * n_jrows + 5))
    gs  = gridspec.GridSpec(
        n_jrows + 2, 3, figure=fig,
        height_ratios=[1] * n_jrows + [0.55, 0.55],
        hspace=0.65, wspace=0.35,
        left=0.05, right=0.97, top=0.93, bottom=0.06,
    )
    joint_axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(action_dim)]
    ax_sm  = fig.add_subplot(gs[n_jrows,     :])
    ax_k   = fig.add_subplot(gs[n_jrows + 1, :])
    all_axes = joint_axes + [ax_sm, ax_k]

    # ── floating control panel ────────────────────────────────────────────────
    # White rectangle drawn on the figure so controls are readable over plots.
    PL, PW = 0.745, 0.245          # panel left, width
    SL, SW = PL + 0.01, PW - 0.02  # slider left, width (inset)
    SH     = 0.023                  # slider height
    GAP    = 0.033                  # vertical gap between sliders

    check_h = 0.038 * len(proc_names) + 0.02
    # stack from bottom: info → dots → checkboxes → sliders → top
    PB_info  = 0.035
    PB_dots  = PB_info  + 0.055
    PB_check = PB_dots  + check_h + 0.01
    PB_kfw   = PB_check + check_h + 0.01
    PB_spsm  = PB_kfw   + GAP
    PB_spc   = PB_spsm  + GAP
    PB_smwin = PB_spc   + GAP + 0.005
    PB_tend  = PB_smwin + GAP
    PB_tstart= PB_tend  + GAP
    PB_ep    = PB_tstart + GAP
    PH       = PB_ep    + SH + 0.025   # total panel height

    # white background patch
    fig.patches.append(plt.Rectangle(
        (PL - 0.005, 0.025), PW + 0.01, PH,
        transform=fig.transFigure, zorder=3,
        facecolor="white", edgecolor="#aaaaaa", linewidth=0.8,
    ))

    def _ax(pb):
        return fig.add_axes([SL, pb, SW, SH])

    ax_ep     = _ax(PB_ep)
    ax_tstart = _ax(PB_tstart)
    ax_tend   = _ax(PB_tend)
    ax_smwin  = _ax(PB_smwin)

    sp_cfg    = PROCESSOR_CONFIGS.get("ajas_spline", {})
    ax_sp_c   = _ax(PB_spc)
    ax_sp_sm  = _ax(PB_spsm)
    ax_sp_kfw = _ax(PB_kfw)

    ax_check  = fig.add_axes([SL, PB_check, SW, check_h])
    ax_kdots  = fig.add_axes([SL, PB_dots,  SW, 0.04])
    ax_info   = fig.add_axes([SL, PB_info,  SW, 0.045])
    ax_info.axis("off")

    sl_ep     = Slider(ax_ep,     "Episode",       0, ds["n_eps"] - 1, valinit=0,              valstep=1)
    sl_tstart = Slider(ax_tstart, "T start",       0, 999,              valinit=0,              valstep=1)
    sl_tend   = Slider(ax_tend,   "T end",         1, 1000,             valinit=500,            valstep=1)
    sl_smwin  = Slider(ax_smwin,  "Sm window",     4, CHUNK,            valinit=sm_window_init, valstep=1)
    sl_sp_c   = Slider(ax_sp_c,   "spline C",    100, 10000,            valinit=sp_cfg.get("C",        500),  valstep=100)
    sl_sp_sm  = Slider(ax_sp_sm,  "spline sm_win", 4, 50,               valinit=sp_cfg.get("sm_window", 15),  valstep=1)
    sl_sp_kfw = Slider(ax_sp_kfw, "spline kf_win", 0,  5,               valinit=sp_cfg.get("kf_window",  1),  valstep=1)

    check      = CheckButtons(ax_check, proc_names, [True] * len(proc_names))
    dots_check = CheckButtons(ax_kdots, ["keyframe dots"], [True])
    info_t = ax_info.text(0.03, 0.97, "", transform=ax_info.transAxes,
                          va="top", fontsize=7.5, family="monospace")

    # ── lines: one per processor per joint ───────────────────────────────────
    joint_lines = {n: [] for n in proc_names}
    for n in proc_names:
        col, ls, lw = PROC_COLORS.get(n, ("grey", "-", 1.2))
        for j, ax in enumerate(joint_axes):
            ln, = ax.plot([], [], color=col, lw=lw, ls=ls,
                          label=n if j == 0 else "_")
            joint_lines[n].append(ln)
            if j == 0:
                ax.legend(fontsize=6, loc="upper right")
        ax.set_title(JOINT_NAMES[j], fontsize=8)

    # ── keyframe dots for ajas_spline ────────────────────────────────────────
    kf_dots = []
    for j, ax in enumerate(joint_axes):
        dot, = ax.plot([], [], "o", color="mediumpurple", ms=2, zorder=5,
                       label="keyframes" if j == 0 else "_")
        kf_dots.append(dot)

    for ax in joint_axes:
        ax.set_xlabel("timestep", fontsize=7)
        ax.tick_params(labelsize=7)
    for j, ax in enumerate(joint_axes):
        ax.set_title(JOINT_NAMES[j], fontsize=8)

    # Sm / k lines from raw
    sm_lines = []
    k_lines  = []
    for j in range(action_dim):
        label = JOINT_NAMES[j] if j not in SKIP_JOINTS else "_"
        sl2, = ax_sm.plot([], [], color=JOINT_COLORS[j], lw=1.2, label=label,
                          visible=(j not in SKIP_JOINTS))
        kl,  = ax_k.plot([], [],  color=JOINT_COLORS[j], lw=1.2,
                          visible=(j not in SKIP_JOINTS))
        sm_lines.append(sl2)
        k_lines.append(kl)

    ax_sm.set_ylabel("Sm  (raw, local window)", fontsize=8)
    ax_sm.set_xlabel("timestep", fontsize=7)
    ax_sm.tick_params(labelsize=7)
    ax_sm.legend(fontsize=6, ncol=7, loc="upper right")
    ax_k.set_ylabel("k (AJAS / pursuit)", fontsize=8)
    ax_k.set_xlabel("timestep", fontsize=7)
    ax_k.tick_params(labelsize=7)

    vline_cols = [[] for _ in all_axes]

    # ── redraw ───────────────────────────────────────────────────────────────
    def redraw():
        e      = state["ep"]
        t0, t1 = state["t0"], state["t1"]
        sm_win = state["sm_win"]

        traces  = get_traces(e)
        raw_ep  = traces["raw"]
        L       = len(raw_ep)
        t_ax    = np.arange(L)

        for n in proc_names:
            arr     = traces[n]
            visible = state["show"][n]
            for j in range(action_dim):
                joint_lines[n][j].set_data(
                    t_ax if visible else [],
                    arr[:, j] if visible else [],
                )

        # Keyframe dots — shown when ajas_spline is visible
        kf_data     = traces.get("_kf_spline", {})
        show_spline = state["show"].get("ajas_spline", False)
        show_dots   = state["show_dots"]
        for j in range(action_dim):
            if show_spline and show_dots and j in kf_data:
                ts, vs = kf_data[j]
                kf_dots[j].set_data(ts, vs)
            else:
                kf_dots[j].set_data([], [])

        # Sm and k from raw
        sm_arr = _sliding_sm(raw_ep, sm_win)
        cfg_ajas    = PROCESSOR_CONFIGS.get("ajas",    {})
        cfg_pursuit = PROCESSOR_CONFIGS.get("pursuit", {})
        C_ref   = cfg_ajas.get("C",     cfg_pursuit.get("C",    2000))
        kmin    = cfg_ajas.get("k_min", cfg_pursuit.get("k_min", 1))
        kmax    = cfg_ajas.get("k_max", cfg_pursuit.get("k_max", 40))
        k_arr = np.clip(np.round(C_ref * sm_arr).astype(int), kmin, kmax)
        for j in range(action_dim):
            if j in SKIP_JOINTS:
                continue
            sm_lines[j].set_data(t_ax, sm_arr[:, j])
            k_lines[j].set_data( t_ax, k_arr[:, j])

        # Chunk vlines
        for ai, ax in enumerate(all_axes):
            for vl in vline_cols[ai]:
                vl.remove()
            vline_cols[ai].clear()
            for x in range(CHUNK, L, CHUNK):
                vl = ax.axvline(x, color="grey", lw=0.5, ls=":", alpha=0.5)
                vline_cols[ai].append(vl)
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(t0, t1)

        # Info
        sm_mean = float(np.mean(sm_arr[:, :6]))
        info_t.set_text(
            f"ep={e}  len={L}\n"
            f"task={ds['tasks'][e]} "
            f"{'✓' if ds['success'][e] else '✗'}\n"
            f"suite={ds['suites'][e]}\n\n"
            f"mean Sm={sm_mean:.5f}\n"
            f"chunks={int(np.ceil(L/CHUNK))}"
        )
        fig.canvas.draw_idle()

    def apply_window():
        t0 = max(0, int(sl_tstart.val))
        t1 = max(t0 + 1, int(sl_tend.val))
        state["t0"], state["t1"] = t0, t1
        for ax in all_axes:
            ax.set_xlim(t0, t1)
        fig.canvas.draw_idle()

    def on_ep(val):
        e = int(sl_ep.val)
        state["ep"] = e
        L = len(ds["episodes"][e][0])
        sl_tstart.set_val(0)
        sl_tend.valmax = L
        sl_tend.ax.set_xlim(1, L)
        sl_tend.set_val(L)
        state["t0"], state["t1"] = 0, L
        redraw()

    def on_check(label):
        state["show"][label] = not state["show"][label]
        redraw()

    def on_spline_param(_=None):
        PROCESSOR_CONFIGS["ajas_spline"].update({
            "C":         float(sl_sp_c.val),
            "sm_window": int(sl_sp_sm.val),
            "kf_window": int(sl_sp_kfw.val),
        })
        state["cache"].clear()   # invalidate so traces recompute with new params
        redraw()

    sl_ep.on_changed(on_ep)
    sl_tstart.on_changed(lambda v: apply_window())
    sl_tend.on_changed(lambda v: apply_window())
    sl_smwin.on_changed(lambda v: (state.__setitem__("sm_win", int(sl_smwin.val)), redraw()))
    sl_sp_c.on_changed(on_spline_param)
    sl_sp_sm.on_changed(on_spline_param)
    sl_sp_kfw.on_changed(on_spline_param)
    check.on_clicked(on_check)
    dots_check.on_clicked(lambda _: (
        state.__setitem__("show_dots", not state["show_dots"]), redraw()))

    fig.suptitle(
        f"Processor comparison — {Path(path).parts[-2]}\n"
        + "  ".join(f"{n}={PROC_COLORS[n][0]}" for n in proc_names if n in PROC_COLORS),
        fontsize=9,
    )
    on_ep(0)
    plt.show()


def plot_episode(path: str, episode: int = 0, t0: int = 0, t1: int | None = None,
                 save: str | None = None) -> None:
    """Static overlay plot for a single episode — no widgets.

    Applies every processor in PROCESSOR_CONFIGS and plots all traces per joint.

    Parameters
    ----------
    path    : path to episodes.npz
    episode : episode index (0-based)
    t0, t1  : timestep window (default: full episode)
    save    : if given, save figure to this path instead of showing
    """
    ds = load_npz(path)
    e  = episode % ds["n_eps"]
    raw_ep, proc_ep = ds["episodes"][e]
    L  = len(raw_ep)
    t1 = t1 if t1 is not None else L
    t_ax = np.arange(L)

    traces = apply_all(raw_ep)
    kf_data = traces.pop("_kf_spline", None)

    action_dim = raw_ep.shape[1]
    n_cols = 3
    n_rows = (action_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3 * n_rows),
                             sharex=False)
    axes = np.array(axes).flatten()

    for j in range(action_dim):
        ax = axes[j]
        for name, arr in traces.items():
            color, ls, lw = PROC_COLORS.get(name, ("grey", "-", 1.2))
            ax.plot(t_ax, arr[:, j], color=color, lw=lw, ls=ls,
                    label=name, alpha=0.85)
        if kf_data is not None and j in kf_data:
            kf_t, kf_v = kf_data[j]
            mask = (kf_t >= t0) & (kf_t < t1)
            ax.scatter(kf_t[mask], kf_v[mask],
                       color=PROC_COLORS["ajas_spline"][0],
                       s=8, zorder=5, label="_kf")
        # chunk boundaries
        for x in range(CHUNK, L, CHUNK):
            ax.axvline(x, color="grey", lw=0.5, ls=":", alpha=0.4)
        ax.set_xlim(t0, t1)
        ax.set_title(JOINT_NAMES[j], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("timestep", fontsize=7)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(action_dim, len(axes)):
        axes[j].set_visible(False)

    suite   = str(ds["suites"][e]) if hasattr(ds["suites"][e], "__str__") else ""
    success = "✓" if ds["success"][e] else "✗"
    fig.suptitle(
        f"{Path(path).parts[-2]}  |  ep={e}  task={ds['tasks'][e]}  {success}  {suite}\n"
        + "  ".join(f"{n}" for n in traces if n != "raw"),
        fontsize=9,
    )
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"Saved → {save}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare action processors on an episodes.npz")
    parser.add_argument("path",        help="Path to episodes.npz")
    parser.add_argument("--sm-window", type=int, default=15)
    parser.add_argument("--static",    action="store_true",
                        help="Static single-episode plot (no widgets)")
    parser.add_argument("--episode",   type=int, default=0,
                        help="Episode index for --static mode")
    parser.add_argument("--t0",        type=int, default=0)
    parser.add_argument("--t1",        type=int, default=None)
    parser.add_argument("--save",      default=None,
                        help="Save figure to this path instead of showing")
    args = parser.parse_args()

    if args.static:
        plot_episode(args.path, episode=args.episode,
                     t0=args.t0, t1=args.t1, save=args.save)
    else:
        main(args.path, sm_window_init=args.sm_window)
