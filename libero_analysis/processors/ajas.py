"""Adaptive Jitter-Aware Action Smoothing (AJAS) — open-loop chunk processor."""

from __future__ import annotations

import numpy as np
from .base import ChunkProcessor, sm_per_joint


class AJASProcessor(ChunkProcessor):
    """Adaptive Jitter-Aware Action Smoothing (AJAS).

    Per-joint adaptive smoothing of the VLA action chunk.
    Jitter is measured via the Weighted Mean Frequency (Sm) and mapped to a
    stride k, then the chunk is linearly interpolated between keyframes spaced
    k steps apart.

        k_j = clip(round(C * Sm_j), k_min, k_max)

    Higher Sm (more jitter) → larger k → fewer, more widely-spaced keyframes
    → smoother interpolated output for that joint.

    Optionally averages ±kf_window neighbours at each keyframe before
    interpolating, so the target is a local centroid rather than a single
    noisy point.

    Parameters
    ----------
    C          : float  Sm → k scaling constant.
    k_min      : int    Minimum stride (floor).
    k_max      : int    Maximum stride (cap).
    dt         : float  Seconds per timestep (1/30 for LIBERO).
    skip_joints: tuple  Joint indices left unsmoothed (e.g. gripper).
    kf_window  : int    ±w neighbours averaged at each keyframe (0 = single point).
    """

    def __init__(
        self,
        C: float = 2000.0,
        k_min: int = 5,
        k_max: int = 40,
        dt: float = 1.0 / 30,
        skip_joints: tuple[int, ...] = (),
        kf_window: int = 0,
    ) -> None:
        self.C           = C
        self.k_min       = k_min
        self.k_max       = k_max
        self.dt          = dt
        self.fs          = 1.0 / dt
        self.skip_joints = set(skip_joints)
        self.kf_window   = kf_window

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

        sm = sm_per_joint(chunk, self.dt)
        ks = np.clip(np.round(self.C * sm).astype(int), self.k_min, self.k_max)
        self.last_sm = sm
        self.last_k  = ks

        smoothed = chunk.copy()
        w = self.kf_window

        for j in range(action_dim):
            if j in self.skip_joints:
                continue
            k = int(ks[j])
            indices = list(range(0, T, k))
            if indices[-1] != T - 1:
                indices.append(T - 1)

            if w == 0:
                kf_vals = [chunk[idx, j] for idx in indices]
            else:
                kf_vals = [
                    chunk[max(0, idx - w):min(T, idx + w + 1), j].mean()
                    for idx in indices
                ]

            for i in range(len(indices) - 1):
                s, e = indices[i], indices[i + 1]
                smoothed[s:e + 1, j] = np.linspace(kf_vals[i], kf_vals[i + 1], e - s + 1)

        return smoothed.astype(np.float32)

    def __repr__(self) -> str:
        skip = f", skip_joints={sorted(self.skip_joints)}" if self.skip_joints else ""
        win  = f", kf_window={self.kf_window}" if self.kf_window else ""
        return (
            f"AJASProcessor(C={self.C}, k_min={self.k_min}, "
            f"k_max={self.k_max}, dt={self.dt:.4f}{skip}{win})"
        )
