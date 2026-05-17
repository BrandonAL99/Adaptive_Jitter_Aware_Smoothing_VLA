"""
Libero dataset plotter.

Load strategy:
  1. On dataset load — scan episode_index + success only (tiny columns, fast).
     Builds an index: global_ep -> (split, local_ep).
  2. On first access of an episode — stream just that episode's action column,
     stop as soon as episode_index changes.  Result is cached in memory.
  This means startup is near-instant and you only download what you view.

Views:
  Plot Actions    – raw / diff / jerk traces per joint
  Fourier         – FFT magnitude per joint + Sm / SER
  Consistency All – bar chart of Sm & SER across all episodes (loads all)
  Windowed Sm     – sliding-window Sm heatmap for one episode

Usage:
  python libero_analysis/libero_plotter.py
  python libero_analysis/libero_plotter.py --source BrandonAL/eval_my_smolvla_libero_object
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import TextBox, Button, CheckButtons
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libero_analysis.temporal_consistency import compute_metrics, compute_metrics_windowed

# ── Dataset catalogue ─────────────────────────────────────────────────────────
HF_SOURCES = [
    "BrandonAL/eval_my_smolvla_libero_object",
    "BrandonAL/eval_HFvla_smolvla_libero_object",
    "BrandonAL/eval_HFvla_smolvla_libero_spatial",
    "BrandonAL/eval_HFvla_smolvla_libero_goal",
    "BrandonAL/eval_HFvla_smolvla_libero_long",
    "BrandonAL/libero_eval_06022026v2",
    "BrandonAL/smolvla_libero_chunk50_test_v2",
]
DEFAULT_SOURCE = HF_SOURCES[0]

JOINT_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"]
COLORS = ["tab:blue", "tab:orange", "tab:green",
          "tab:red",  "tab:purple", "tab:brown", "tab:pink"]
DT = 1.0 / 30.0

# ── State ─────────────────────────────────────────────────────────────────────
_repo_id       = None
_ep_index      = {}   # global_ep -> (split, local_ep)
_success_cache = {}   # global_ep -> bool
_actions_cache = {}   # global_ep -> np.ndarray (n_joints, T)  — lazy
_obs_cache     = {}   # global_ep -> np.ndarray (n_joints, T)  — lazy (joint pos dims 20:27)
_label         = ""

def _loaded():
    return bool(_ep_index)

def _num_eps():
    return max(_ep_index.keys()) if _ep_index else 0

# ── HF helpers ────────────────────────────────────────────────────────────────
def _login():
    try:
        from huggingface_hub import login
        login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)
    except Exception:
        pass

def _stream(repo, split, cols):
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    return ds.select_columns(cols)

def _load(repo_id: str):
    """Build the episode index by reading only episode_index + success columns."""
    global _repo_id, _ep_index, _success_cache, _actions_cache, _label
    print(f"\nLoading index for {repo_id} ...", flush=True)
    _login()
    from datasets import disable_progress_bar
    disable_progress_bar()

    _ep_index      = {}
    _success_cache = {}
    _actions_cache = {}
    _obs_cache     = {}
    _repo_id       = repo_id
    global_ep      = 0

    for i in range(20):
        split = f"task_{i}"
        try:
            stream = _stream(repo_id, split, ["episode_index", "success"])
        except Exception:
            break
        local_seen = {}
        for row in stream:
            loc = int(row["episode_index"])
            if loc not in local_seen:
                local_seen[loc] = global_ep
                _ep_index[global_ep]      = (split, loc)
                _success_cache[global_ep] = bool(row["success"])
                global_ep += 1
            _success_cache[local_seen[loc]] = bool(row["success"])  # keep last value
        print(f"  {split}: {len(local_seen)} episodes", flush=True)

    _label = repo_id
    print(f"  Index built: {len(_ep_index)} episodes total")

def _fetch_actions(global_ep: int) -> np.ndarray:
    """Stream one episode's actions on demand and cache."""
    if global_ep in _actions_cache:
        return _actions_cache[global_ep]

    split, local_ep = _ep_index[global_ep]
    stream = _stream(_repo_id, split, ["episode_index", "action"])
    rows = []
    started = False
    for row in stream:
        loc = int(row["episode_index"])
        if loc == local_ep:
            rows.append(row["action"])
            started = True
        elif started:
            break   # episode ended, stop streaming

    result = np.array(rows).T  # (n_joints, T)
    _actions_cache[global_ep] = result
    return result

def _fetch_obs(global_ep: int) -> np.ndarray:
    """Stream one episode's robot_state on demand, return joint pos dims 20:27, cached."""
    if global_ep in _obs_cache:
        return _obs_cache[global_ep]

    split, local_ep = _ep_index[global_ep]
    stream = _stream(_repo_id, split, ["episode_index", "observation.robot_state"])
    rows = []
    started = False
    for row in stream:
        loc = int(row["episode_index"])
        if loc == local_ep:
            rows.append(row["observation.robot_state"][20:27])
            started = True
        elif started:
            break

    result = np.array(rows).T  # (7, T)
    _obs_cache[global_ep] = result
    return result

def _get_success(ep: int):
    return _success_cache.get(ep)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.canvas.manager.set_window_title("Libero Plotter")

gs_main = gridspec.GridSpec(7, 1, left=0.07, right=0.66,
                             top=0.93, bottom=0.27, hspace=0.12)
axs = [fig.add_subplot(gs_main[i]) for i in range(7)]
for i, ax in enumerate(axs):
    ax.set_ylabel(JOINT_NAMES[i], fontsize=8)
    ax.grid(True)
axs[-1].set_xlabel("Timestep")

secax = axs[0].secondary_xaxis("top",
    functions=(lambda x: x * DT, lambda t: t / DT))
secax.set_xlabel("Time (s)")

# Dataset list panel
ax_list = fig.add_axes([0.69, 0.27, 0.29, 0.66])
ax_list.set_facecolor("#f5f5f5")
_VISIBLE = 10
_list_st = {"offset": 0, "selected": 0}

def _draw_list():
    ax_list.cla(); ax_list.set_axis_off(); ax_list.set_facecolor("#f5f5f5")
    ax_list.set_title("Datasets  (click to load)", fontsize=8, pad=3)
    for i in range(min(_VISIBLE, len(HF_SOURCES) - _list_st["offset"])):
        idx = _list_st["offset"] + i
        lbl = HF_SOURCES[idx].split("/")[-1]
        y   = 1.0 - (i + 0.5) / _VISIBLE
        sel = idx == _list_st["selected"]
        ax_list.text(0.04, y, lbl, fontsize=7.5, va="center", ha="left",
                     transform=ax_list.transAxes,
                     color="white" if sel else "#222",
                     bbox=dict(boxstyle="round,pad=0.25",
                               facecolor="#2a6db5" if sel else "white",
                               edgecolor="#aaa", linewidth=0.5))
    fig.canvas.draw_idle()

def _on_list_click(event):
    if event.inaxes != ax_list:
        return
    bb   = ax_list.get_window_extent()
    item = int((1.0 - max(0.0, min(1.0, (event.y - bb.y0) / bb.height))) * _VISIBLE)
    idx  = _list_st["offset"] + item
    if 0 <= idx < len(HF_SOURCES):
        _list_st["selected"] = idx
        _draw_list()

fig.canvas.mpl_connect("button_press_event", _on_list_click)

# ── Controls ──────────────────────────────────────────────────────────────────
fig.text(0.07, 0.215, "Episode",      fontsize=9)
ep_box   = TextBox(fig.add_axes([0.18, 0.205, 0.05, 0.030]), None, initial="0")

fig.text(0.07, 0.168, "Start step",   fontsize=9)
s0_box   = TextBox(fig.add_axes([0.18, 0.158, 0.05, 0.030]), None, initial="0")

_end_lbl = fig.text(0.07, 0.121, "End step (max=?)", fontsize=9)
s1_box   = TextBox(fig.add_axes([0.18, 0.111, 0.05, 0.030]), None, initial="0")

fig.text(0.07, 0.074, "Window (Sm)", fontsize=9)
win_box  = TextBox(fig.add_axes([0.18, 0.064, 0.05, 0.030]), None, initial="20")

plot_opts = CheckButtons(fig.add_axes([0.30, 0.065, 0.20, 0.155]),
                         ["Raw", "Diff (vel)", "Jerk", "Show Obs"], [True, False, False, False])
for lbl in plot_opts.labels:
    lbl.set_fontsize(10)

def _btn(rect, label):
    return Button(fig.add_axes(rect), label)

btn_plot     = _btn([0.54, 0.190, 0.12, 0.033], "Plot Actions")
btn_fourier  = _btn([0.54, 0.150, 0.12, 0.033], "Fourier")
btn_cons_all = _btn([0.54, 0.110, 0.12, 0.033], "Consistency All")
btn_windowed = _btn([0.54, 0.070, 0.12, 0.033], "Windowed Sm")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _ep_idx(): return int(ep_box.text)
def _s0():     return int(s0_box.text)
def _s1():     return int(s1_box.text)
def _win():    return max(4, int(win_box.text))

def _prep_actions(ep, s0, s1):
    raw    = _fetch_actions(ep)[:, s0:s1]
    status = plot_opts.get_status()
    t      = np.arange(s0, s1)
    if status[1] or status[2]:
        raw = np.diff(raw, axis=1); t = t[:-1]
    if status[2]:
        raw = np.diff(raw, axis=1); t = t[:-1]
    return raw, t

def _clear_axes():
    for i, ax in enumerate(axs):
        ax.cla(); ax.set_ylabel(JOINT_NAMES[i], fontsize=8); ax.grid(True)
    axs[-1].set_xlabel("Timestep")

def _title(s):
    plt.suptitle(f"{_label}\n{s}", fontsize=9)

# ── Callbacks ─────────────────────────────────────────────────────────────────
def _on_ep_change(_):
    if not _loaded():
        return
    ep = _ep_idx()
    if ep > _num_eps():
        print(f"Episode {ep} out of range (max {_num_eps()})")
        return
    suc = _get_success(ep)
    s0_box.set_val("0")
    if ep in _actions_cache:
        T = _actions_cache[ep].shape[1]
        _end_lbl.set_text(f"End step (max={T-1})  success={suc}")
        s1_box.set_val(str(T - 1))
    else:
        _end_lbl.set_text(f"End step (max=?)  success={suc}")
        s1_box.set_val("500")

def _ensure_loaded():
    """Load index for the currently selected dataset if not already loaded."""
    repo = HF_SOURCES[_list_st["selected"]]
    if _repo_id != repo:
        _load(repo)
        ep_box.set_val("0")
        _draw_list()

def _load_and_refresh(source):
    _load(source)
    ep_box.set_val("0")
    _on_ep_change(None)
    _draw_list()
    _cb_plot(None)

def _cb_plot(_):
    _ensure_loaded()
    ep  = _ep_idx()
    raw = _fetch_actions(ep)
    T   = raw.shape[1]
    s0  = min(_s0(), T - 1)
    s1  = min(_s1(), T)
    _end_lbl.set_text(f"End step (max={T-1})  success={_get_success(ep)}")
    s1_box.set_val(str(T - 1))
    actions, t = _prep_actions(ep, s0, s1)
    status = plot_opts.get_status()
    mode   = "Jerk" if status[2] else "Diff" if status[1] else "Raw"
    show_obs = status[3]
    suc  = _get_success(ep)
    _clear_axes()
    for i, ax in enumerate(axs):
        ax.plot(t, actions[i], color=COLORS[i], linewidth=1.2, label="action")
        if show_obs:
            obs = _fetch_obs(ep)[:, s0:s1]
            ax.plot(t[:obs.shape[1]], obs[i], color=COLORS[i], linewidth=1.0,
                    linestyle="--", alpha=0.6, label="obs (joint pos)")
        ax.set_xlim(t[0], t[-1]); ax.margins(x=0)
    if show_obs:
        axs[0].legend(loc="upper right", fontsize=6)
    axs[-1].set_xlabel("Timestep")
    _title(f"Episode {ep}  success={suc} — {mode} actions (steps {s0}–{s1})")
    fig.canvas.draw_idle()

def _cb_fourier(_):
    _ensure_loaded()
    ep  = _ep_idx()
    raw = _fetch_actions(ep)
    T   = raw.shape[1]
    s0  = min(_s0(), T - 1)
    s1  = min(_s1(), T)
    _end_lbl.set_text(f"End step (max={T-1})  success={_get_success(ep)}")
    s1_box.set_val(str(T - 1))
    actions = raw[:, s0:s1]
    m       = compute_metrics(actions, DT)
    N       = actions.shape[1]
    f_hz    = np.fft.rfftfreq(N, d=DT)
    print(f"\nFourier ep{ep} steps {s0}–{s1}  Sm={m['sm_mean']:.6f}  SER={m['ser_mean']:.4f}")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {name}: Sm={m['sm_per_joint'][i]:.6f}  SER={m['ser_per_joint'][i]:.4f}")
    _clear_axes()
    for i, ax in enumerate(axs):
        F = np.abs(np.fft.rfft(actions[i] - np.mean(actions[i]))) / N
        ax.plot(f_hz, F, color=COLORS[i], linewidth=1.2, marker="o", markersize=2)
        ax.axvline(1.0, color="grey", lw=0.8, ls="--", alpha=0.6, label="1 Hz (SER)")
        ax.axvline(5.0, color="red",  lw=0.8, ls="--", alpha=0.5, label="5 Hz (jitter)")
        ax.set_ylabel(f"{JOINT_NAMES[i]} |F|", fontsize=7)
        ax.set_xlim(0, 15); ax.grid(True)
    axs[0].legend(loc="upper right", fontsize=6)
    axs[-1].set_xlabel("Frequency (Hz)")
    _title(f"Episode {ep} — Fourier  Sm={m['sm_mean']:.5f}  SER={m['ser_mean']:.3f}")
    fig.canvas.draw_idle()

def _cb_consistency_all(_):
    _ensure_loaded()
    n = _num_eps() + 1
    print(f"\nConsistency All ({n} episodes) — loading all action data ...")
    sms, sers, sucs = [], [], []
    for ep in range(n):
        m = compute_metrics(_fetch_actions(ep), DT)
        sms.append(m["sm_mean"]); sers.append(m["ser_mean"])
        s = _get_success(ep); sucs.append(s)
        print(f"  ep{ep:03d}  Sm={m['sm_mean']:.6f}  SER={m['ser_mean']:.4f}  success={s}")

    sms  = np.array(sms); sers = np.array(sers)
    clrs = ["tab:green" if s else "tab:red" if s is False else "tab:blue" for s in sucs]
    eps  = np.arange(n)

    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig2.canvas.manager.set_window_title("Consistency — All Episodes")
    axes2[0].bar(eps, sms, color=clrs, alpha=0.8)
    axes2[0].axhline(sms.mean(), color="k", lw=1.2, ls="--", label=f"mean={sms.mean():.5f}")
    axes2[0].set_ylabel("Sm  (lower = smoother)"); axes2[0].legend(fontsize=8); axes2[0].grid(axis="y", alpha=0.4)
    axes2[0].set_title(f"{_label}  (green=success, red=fail)")
    axes2[1].bar(eps, sers, color=clrs, alpha=0.8)
    axes2[1].axhline(sers.mean(), color="k", lw=1.2, ls="--", label=f"mean={sers.mean():.4f}")
    axes2[1].set_ylabel("SER  (higher = smoother)"); axes2[1].set_xlabel("Episode")
    axes2[1].legend(fontsize=8); axes2[1].grid(axis="y", alpha=0.4)
    plt.tight_layout()

    sr = sum(1 for s in sucs if s) / n
    print(f"\n  Sm  mean={sms.mean():.6f}  std={sms.std():.6f}")
    print(f"  SER mean={sers.mean():.4f}   std={sers.std():.4f}")
    print(f"  Success rate: {sr:.1%}  ({sum(1 for s in sucs if s)}/{n})")
    plt.show(block=False)

def _cb_windowed(_):
    _ensure_loaded()
    ep  = _ep_idx()
    raw = _fetch_actions(ep)
    T   = raw.shape[1]
    s0  = min(_s0(), T - 1)
    s1  = min(_s1(), T)
    _end_lbl.set_text(f"End step (max={T-1})  success={_get_success(ep)}")
    s1_box.set_val(str(T - 1))
    window  = _win()
    actions = raw[:, s0:s1]
    wins    = compute_metrics_windowed(actions, DT, window=window, step=1)
    if not wins:
        print("Not enough timesteps for windowed analysis."); return
    sm_mat = np.array([w["sm_per_joint"] for w in wins]).T
    starts = np.array([w["start"] + s0 for w in wins])
    fig3, axes3 = plt.subplots(2, 1, figsize=(14, 6))
    fig3.canvas.manager.set_window_title(f"Windowed Sm — ep{ep}")
    im = axes3[0].imshow(sm_mat, aspect="auto", origin="lower",
                         extent=[starts[0], starts[-1], -0.5, 6.5], cmap="hot_r")
    axes3[0].set_yticks(range(7)); axes3[0].set_yticklabels(JOINT_NAMES, fontsize=8)
    axes3[0].set_title(f"Sm heatmap — ep{ep}, window={window} ({window*DT:.2f}s)")
    plt.colorbar(im, ax=axes3[0], label="Sm")
    mean_sm = sm_mat.mean(axis=0)
    axes3[1].plot(starts, mean_sm, color="tab:blue", lw=1.2)
    axes3[1].fill_between(starts, mean_sm, alpha=0.25, color="tab:blue")
    axes3[1].set_ylabel("Mean Sm"); axes3[1].set_title("Mean Sm over time (higher = more jitter)")
    axes3[1].grid(True); axes3[1].set_xlabel("Timestep")
    plt.tight_layout(); plt.show(block=False)
    print(f"Windowed Sm ep{ep}: range {mean_sm.min():.6f} – {mean_sm.max():.6f}")

def _on_opts_click(label):
    # Raw/Diff/Jerk (indices 0-2) are mutually exclusive; Show Obs (index 3) is independent
    status = plot_opts.get_status()
    active_sig = [i for i in range(3) if status[i]]
    if len(active_sig) > 1:
        for i in active_sig[:-1]:
            plot_opts.set_active(i)

def _on_key(event):
    if event.key == "q":
        plt.close("all")

# ── Wire up ───────────────────────────────────────────────────────────────────
btn_plot.on_clicked(_cb_plot)
btn_fourier.on_clicked(_cb_fourier)
btn_cons_all.on_clicked(_cb_consistency_all)
btn_windowed.on_clicked(_cb_windowed)
ep_box.on_submit(_on_ep_change)
plot_opts.on_clicked(_on_opts_click)
fig.canvas.mpl_connect("key_press_event", _on_key)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args()
    if args.source in HF_SOURCES:
        _list_st["selected"] = HF_SOURCES.index(args.source)
    _draw_list()
    plt.show()