from __future__ import annotations

import numpy as np
from .base import ActionProcessor


class LowPassFilterProcessor(ActionProcessor):
    """Causal Butterworth low-pass filter applied to each action dimension.

    Uses scipy's second-order-sections (SOS) form for numerical stability.
    Filter state (zi) is maintained across timesteps and initialised on the
    first action to avoid start-up transients.

    Parameters
    ----------
    cutoff_hz : float  -3 dB cut-off frequency in Hz (3–6 Hz typical for 30 Hz robot control).
    fs        : float  Sampling frequency in Hz (30 for LIBERO).
    order     : int    Butterworth order (2 recommended; 4 max for real-time).
    """

    def __init__(self, cutoff_hz: float = 3.0, fs: float = 30.0, order: int = 2,
                 skip_joints: tuple = ()) -> None:
        from scipy.signal import butter, sosfilt_zi
        self.cutoff_hz    = cutoff_hz
        self.fs           = fs
        self.order        = order
        self.skip_joints  = set(skip_joints)
        self.sos          = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
        self._zi_template = sosfilt_zi(self.sos)
        self._zi: np.ndarray | None = None

    def reset(self) -> None:
        self._zi = None

    def __call__(self, action: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt
        action     = np.asarray(action, dtype=np.float32)
        action_dim = action.shape[0]

        if self._zi is None:
            self._zi = np.stack(
                [self._zi_template * action[i] for i in range(action_dim)], axis=-1
            )

        filtered = np.empty_like(action)
        for i in range(action_dim):
            if i in self.skip_joints:
                filtered[i] = action[i]
            else:
                out, new_zi       = sosfilt(self.sos, action[i:i + 1], zi=self._zi[:, :, i])
                filtered[i]       = out[0]
                self._zi[:, :, i] = new_zi

        return filtered.astype(np.float32)

    def __repr__(self) -> str:
        skip = f", skip_joints={tuple(sorted(self.skip_joints))}" if self.skip_joints else ""
        return (
            f"LowPassFilterProcessor(cutoff_hz={self.cutoff_hz}, "
            f"fs={self.fs}, order={self.order}{skip})"
        )
