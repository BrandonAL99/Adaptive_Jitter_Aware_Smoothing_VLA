"""AJAS with adaptive non-uniform keyframe placement and cubic spline interpolation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from .base import ChunkProcessor, sm_per_joint_norm


class AJASSplineProcessor(ChunkProcessor):
    """AJAS variant with adaptive keyframe placement and cubic Hermite spline interpolation.

    Improvements over AJASProcessor:
      1. Normalised Sm — divides each joint by its std before computing Sm so
         joints with different amplitude ranges are treated equally.
      2. Adaptive non-uniform keyframes — instead of a fixed stride k for the whole
         chunk, a sliding window re-evaluates local Sm at each keyframe and jumps
         k steps ahead.  Jittery regions get sparse keyframes (large k = skip over
         noise), smooth regions get dense keyframes (small k = preserve detail).
      3. Cubic Hermite spline — interpolates between keyframes matching velocity at
         each boundary, giving C1-continuous output with no sharp corners.

    Parameters
    ----------
    C          : float  Sm → k scaling constant (calibrated for normalised Sm,
                        so values ~50–200 are typical rather than 500–5000).
    k_min      : int    Minimum stride.
    k_max      : int    Maximum stride.
    dt         : float  Seconds per timestep.
    skip_joints: tuple  Joints left unsmoothed (default: gripper index 6).
    kf_window  : int    ±w neighbours averaged at each keyframe (default 2).
    sm_window  : int    Forward window size for local Sm computation (default 15).
    """

    def __init__(
        self,
        C: float = 100.0,
        k_min: int = 1,
        k_max: int = 40,
        dt: float = 1.0 / 30,
        skip_joints: tuple[int, ...] = (6,),
        kf_window: int = 2,
        sm_window: int = 15,
    ) -> None:
        self.C           = C
        self.k_min       = k_min
        self.k_max       = k_max
        self.dt          = dt
        self.skip_joints = set(skip_joints)
        self.kf_window   = kf_window
        self.sm_window   = sm_window

        self.last_sm: np.ndarray | None = None
        self.last_k:  np.ndarray | None = None

    def reset(self) -> None:
        self.last_sm    = None
        self.last_k     = None
        self._raw_chunk = None
        self._raw_idx   = 0

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        T, action_dim = chunk.shape
        smoothed = chunk.copy()

        for j in range(action_dim):
            if j in self.skip_joints:
                continue

            # ── adaptive keyframe placement ───────────────────────────────
            # Re-evaluate local Sm at each keyframe using a forward window.
            # k adapts within the chunk: large k in jittery regions, small k
            # in smooth regions.
            kf_idx    = [0]
            last_sm_j = 0.0
            t         = 0

            while t < T - 1:
                win = chunk[t:min(T, t + self.sm_window)]
                if len(win) >= 4:
                    last_sm_j = float(sm_per_joint_norm(win, self.dt)[j])
                k      = int(np.clip(round(self.C * last_sm_j), self.k_min, self.k_max))
                next_t = min(t + k, T - 1)
                kf_idx.append(next_t)
                t = next_t

            if kf_idx[-1] != T - 1:
                kf_idx.append(T - 1)

            # ── average ±kf_window neighbours at each keyframe ────────────
            # Cap the window at half the gap to the nearest neighbour so that
            # overlapping windows don't wash out height differences between
            # adjacent keyframes (which produces horizontal splines in steep regions).
            kf_pos = []
            n_kf   = len(kf_idx)
            for ii, ki in enumerate(kf_idx):
                lo_gap = (ki - kf_idx[ii - 1]) // 2 if ii > 0           else ki
                hi_gap = (kf_idx[ii + 1] - ki) // 2 if ii < n_kf - 1   else T - 1 - ki
                eff_w  = min(self.kf_window, lo_gap, hi_gap)
                lo     = max(0,     ki - eff_w)
                hi     = min(T - 1, ki + eff_w)
                kf_pos.append(float(chunk[lo:hi + 1, j].mean()))

            # ── monotone cubic spline ─────────────────────────────────────
            # PchipInterpolator: C1 continuous, monotone-preserving — no overshoot,
            # no horizontal artefact when keyframes straddle a steep rise/fall.
            kf_t = np.array(kf_idx, dtype=float)
            kf_p = np.array(kf_pos, dtype=float)
            cs   = PchipInterpolator(kf_t, kf_p)
            smoothed[:, j] = cs(np.arange(T))

        # Store for raw-action recording in eval script
        global_sm = sm_per_joint_norm(chunk, self.dt)
        self.last_sm = global_sm
        self.last_k  = np.clip(np.round(self.C * global_sm).astype(int),
                               self.k_min, self.k_max)

        return smoothed.astype(np.float32)

    def __repr__(self) -> str:
        skip = f", skip_joints={sorted(self.skip_joints)}" if self.skip_joints else ""
        return (
            f"AJASSplineProcessor(C={self.C}, k_min={self.k_min}, "
            f"k_max={self.k_max}, dt={self.dt:.4f}, "
            f"kf_window={self.kf_window}, sm_window={self.sm_window}{skip})"
        )
