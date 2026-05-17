"""Pure-pursuit closed-loop action processor."""

from __future__ import annotations

import numpy as np
from .base import ChunkProcessor, sm_per_joint


class PurePursuitProcessor(ChunkProcessor):
    """Closed-loop pure-pursuit action smoothing.

    Mirrors the mobile-robotics pure-pursuit algorithm:
      - The VLA chunk is the "path" (T waypoints in joint space).
      - At each execution step the controller looks ahead k_j steps along the
        path and drives the joint proportionally toward that target using the
        *observed* current joint angle.

    Per step, for joint j:

        action_j = q_j  +  P * (path[t + k_j]_j  -  q_j)

    where q_j is the observed joint position and path[t + k_j]_j is the
    denormalised lookahead waypoint averaged over ±kf_window neighbours.

    k_j is computed adaptively from the Sm (Weighted Mean Frequency) of each
    joint in the chunk:

        k_j = clip(round(C * Sm_j), k_min, k_max)

    High-jitter joints get larger k → lookahead is further ahead → smoother
    commanded motion.  This is closed-loop: if the robot lags the path it
    automatically closes the gap; if it overshoots it slows down.

    Parameters
    ----------
    C          : float   Sm → k scaling constant.
    k_min      : int     Minimum lookahead steps.
    k_max      : int     Maximum lookahead steps.
    P          : float   Proportional gain (0.5 = move halfway to lookahead).
    dt         : float   Seconds per timestep (1/30 for LIBERO).
    skip_joints: tuple   Joints left unsmoothed (default: gripper, index 6).
    kf_window  : int     ±w neighbours averaged at lookahead point (default: 1).
    """

    def __init__(
        self,
        C: float = 2000.0,
        k_min: int = 1,
        k_max: int = 10,
        P: float = 0.5,
        dt: float = 1.0 / 30,
        skip_joints: tuple[int, ...] = (6,),
        kf_window: int = 1,
        sm_window: int = 15,
    ) -> None:
        self.C           = C
        self.k_min       = k_min
        self.k_max       = k_max
        self.P           = P
        self.dt          = dt
        self.fs          = 1.0 / dt
        self.skip_joints = set(skip_joints)
        self.kf_window   = kf_window
        self.sm_window   = sm_window  # steps ahead used to compute local Sm; 0 = whole-chunk global k

        self.last_sm: np.ndarray | None = None
        self.last_k:  np.ndarray | None = None
        self._raw_chunk: np.ndarray | None = None
        self._raw_idx: int = 0
        # Per-step values exposed after each pursue() call for external logging
        self.last_step_ks:  np.ndarray | None = None
        self.last_step_sm:  np.ndarray | None = None

    def reset(self) -> None:
        self.last_sm       = None
        self.last_k        = None
        self._raw_chunk    = None
        self._raw_idx      = 0
        self.last_step_ks  = None
        self.last_step_sm  = None

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """Compute Sm → k and store the path. Returns chunk UNCHANGED.

        Actual smoothing happens per-step in pursue(), called from
        run_episode() with the current observed joint angles.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        sm = sm_per_joint(chunk, self.dt)
        self.last_sm = sm
        self.last_k  = np.clip(np.round(self.C * sm).astype(int), self.k_min, self.k_max)
        return chunk

    def pursue(
        self,
        current_joints: np.ndarray,
        raw_action_denorm: np.ndarray,
        step_idx: int,
        postprocessor,
    ) -> np.ndarray:
        """Compute the pure-pursuit action for one execution step.

        Parameters
        ----------
        current_joints    : np.ndarray (action_dim,)  observed joint positions.
        raw_action_denorm : np.ndarray (action_dim,)  raw denormalised action
                            (used as-is for skip_joints).
        step_idx          : int  current execution index within the chunk.
        postprocessor     : callable  normalised → actual joint angles.
        """
        import torch

        T      = self._raw_chunk.shape[0]
        action = raw_action_denorm.copy()

        # Recompute k locally from upcoming path if sm_window is set,
        # otherwise fall back to the global k computed in __call__().
        used_sm: np.ndarray
        if self.sm_window > 0:
            win_end      = min(T, step_idx + self.sm_window)
            local_chunk  = self._raw_chunk[step_idx:win_end]
            if len(local_chunk) >= 4:
                local_sm = sm_per_joint(local_chunk, self.dt)
                ks       = np.clip(np.round(self.C * local_sm).astype(int), self.k_min, self.k_max)
                used_sm  = local_sm
            else:
                ks      = self.last_k
                used_sm = self.last_sm if self.last_sm is not None else np.zeros(action.shape[0])
        else:
            ks      = self.last_k
            used_sm = self.last_sm if self.last_sm is not None else np.zeros(action.shape[0])

        self.last_step_ks = ks.astype(np.float32)
        self.last_step_sm = used_sm.astype(np.float32)

        for j, k in enumerate(ks):
            if j in self.skip_joints:
                continue
            centre = min(step_idx + int(k), T - 1)
            lo     = max(0, centre - self.kf_window)
            hi     = min(T - 1, centre + self.kf_window)
            target_j = np.mean([
                postprocessor(torch.from_numpy(self._raw_chunk[t:t + 1]))
                .squeeze(0).cpu().numpy()[j]
                for t in range(lo, hi + 1)
            ])
            action[j] = current_joints[j] + self.P * (target_j - current_joints[j])

        return action.astype(np.float32)

    def __repr__(self) -> str:
        skip = f", skip_joints={sorted(self.skip_joints)}" if self.skip_joints else ""
        win = f", kf_window={self.kf_window}" if self.kf_window else ""
        smw = f", sm_window={self.sm_window}" if self.sm_window else ""
        return (
            f"PurePursuitProcessor(C={self.C}, k_min={self.k_min}, "
            f"k_max={self.k_max}, P={self.P}, dt={self.dt:.4f}{skip}{win}{smw})"
        )
