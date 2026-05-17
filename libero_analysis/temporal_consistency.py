"""
Temporal consistency metrics for action sequences.

Two metrics are provided, both derived from the FFT of each joint's action signal:

  Sm  (weighted mean frequency)
      Sm = (2 / (n * fs)) * sum( |F(f)| * f )
      Range [0, 1].  Higher → more high-frequency content → more jitter.

  SER (Signal Energy Ratio)
      SER = energy below `threshold_hz` / total energy
      Range [0, 1].  Higher → more energy in low frequencies → smoother.

Primary entry-point
-------------------
compute_metrics(actions, dt) -> dict
    actions : np.ndarray, shape (n_joints, T)  or  (T, n_joints)
    dt      : float, seconds per timestep (e.g. 1/30)

Returns a dict with keys:
    'sm_per_joint'   : list[float]  – Sm for each joint
    'ser_per_joint'  : list[float]  – SER for each joint
    'sm_mean'        : float        – mean Sm across joints
    'sm_max'         : float        – max  Sm across joints
    'ser_mean'       : float        – mean SER across joints
    'ser_min'        : float        – min  SER across joints (worst joint)
    'n_joints'       : int
    'T'              : int          – number of timesteps
    'dt'             : float

Windowed helper
---------------
compute_metrics_windowed(actions, dt, window, step) -> list[dict]
    Slides a window over the time axis and returns one metrics-dict per window.
    Each dict also contains 'start' and 'end' (inclusive timestep indices).
"""

import numpy as np


def compute_metrics(actions: np.ndarray, dt: float, ser_threshold_hz: float = 1.0) -> dict:
    """Compute Sm and SER temporal-consistency metrics for an action sequence.

    Parameters
    ----------
    actions : np.ndarray
        Shape ``(n_joints, T)`` or ``(T, n_joints)``.  Values are joint
        positions / actions at each timestep.
    dt : float
        Seconds per timestep.
    ser_threshold_hz : float
        Frequency boundary (Hz) for the SER low-energy band.  Default 1.0 Hz.

    Returns
    -------
    dict
        See module docstring for full key descriptions.
    """
    actions = np.asarray(actions, dtype=float)

    # Normalise to (n_joints, T)
    if actions.ndim == 1:
        actions = actions[np.newaxis, :]
    if actions.shape[0] > actions.shape[1]:
        actions = actions.T

    n_joints, T = actions.shape
    fs = 1.0 / dt
    f_hz = np.fft.rfftfreq(T, d=dt)
    n = len(f_hz)  # number of FFT bins

    sm_per_joint = []
    ser_per_joint = []

    for i in range(n_joints):
        x = actions[i] - np.mean(actions[i])
        F = np.abs(np.fft.rfft(x)) / T  # normalised magnitude spectrum

        # Sm: weighted mean frequency
        Sm = (2.0 / (n * fs)) * np.sum(F * f_hz)
        sm_per_joint.append(float(Sm))

        # SER: fraction of energy below threshold_hz
        total_energy = np.sum(F ** 2)
        if total_energy > 0:
            ser = np.sum(F[f_hz <= ser_threshold_hz] ** 2) / total_energy
        else:
            ser = 1.0  # silent signal is perfectly consistent
        ser_per_joint.append(float(ser))

    return {
        "sm_per_joint": sm_per_joint,
        "ser_per_joint": ser_per_joint,
        "sm_mean": float(np.mean(sm_per_joint)),
        "sm_max": float(np.max(sm_per_joint)),
        "ser_mean": float(np.mean(ser_per_joint)),
        "ser_min": float(np.min(ser_per_joint)),
        "n_joints": n_joints,
        "T": T,
        "dt": dt,
    }


def compute_metrics_windowed(
    actions: np.ndarray,
    dt: float,
    window: int,
    step: int = 1,
    ser_threshold_hz: float = 1.0,
) -> list:
    """Slide a window over the action sequence and compute metrics per window.

    Parameters
    ----------
    actions : np.ndarray
        Shape ``(n_joints, T)`` or ``(T, n_joints)``.
    dt : float
        Seconds per timestep.
    window : int
        Number of timesteps in each window.
    step : int
        Number of timesteps to advance between windows.
    ser_threshold_hz : float
        Passed through to :func:`compute_metrics`.

    Returns
    -------
    list[dict]
        One dict per window, each containing all keys from
        :func:`compute_metrics` plus ``'start'`` and ``'end'`` timestep indices.
    """
    actions = np.asarray(actions, dtype=float)
    if actions.shape[0] > actions.shape[1]:
        actions = actions.T  # -> (n_joints, T)

    T = actions.shape[1]
    results = []
    for start in range(0, T - window + 1, step):
        end = start + window
        m = compute_metrics(actions[:, start:end], dt, ser_threshold_hz)
        m["start"] = start
        m["end"] = end - 1
        results.append(m)
    return results


def print_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print a metrics dict returned by :func:`compute_metrics`."""
    header = f"  {label}" if label else ""
    print(f"\nTemporal consistency metrics{header}")
    print(f"  Timesteps : {metrics['T']}  |  dt = {metrics['dt']:.4f}s  |  joints = {metrics['n_joints']}")
    print(f"  Sm  mean={metrics['sm_mean']:.6f}   max={metrics['sm_max']:.6f}  (lower = smoother)")
    print(f"  SER mean={metrics['ser_mean']:.4f}    min={metrics['ser_min']:.4f}  (higher = smoother)")
    joint_names = ["sh_pan", "sh_lift", "el_flex", "wr_flex", "wr_roll", "gripper"]
    names = joint_names[:metrics['n_joints']] if metrics['n_joints'] <= 6 else [str(i) for i in range(metrics['n_joints'])]
    col = "  {:>10}" * metrics['n_joints']
    print("  " + "".join(f"{n:>10}" for n in names))
    print("  Sm  " + "".join(f"{v:>10.6f}" for v in metrics['sm_per_joint']))
    print("  SER " + "".join(f"{v:>10.4f}" for v in metrics['ser_per_joint']))