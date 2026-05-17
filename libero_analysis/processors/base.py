"""Base classes and shared utilities for action processors."""

from __future__ import annotations

import numpy as np


def sm_per_joint(chunk: np.ndarray, dt: float = 1.0 / 30) -> np.ndarray:
    """Weighted mean frequency (Sm) per joint.

    Parameters
    ----------
    chunk : np.ndarray, shape (T, action_dim)
    dt    : float  seconds per timestep

    Returns
    -------
    np.ndarray, shape (action_dim,)  — Sm values, higher = more jitter
    """
    fs = 1.0 / dt
    scores = []
    for j in range(chunk.shape[1]):
        x = chunk[:, j] - np.mean(chunk[:, j])
        F = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=dt)
        n = len(freqs)
        scores.append((2.0 / (n * fs)) * float(np.sum(F * freqs)))
    return np.array(scores, dtype=np.float32)


def sm_per_joint_norm(chunk: np.ndarray, dt: float = 1.0 / 30) -> np.ndarray:
    """Sm with per-joint std-normalisation.

    Divides each joint's signal by its std before FFT so Sm reflects
    relative jitter (fraction of signal range) rather than absolute amplitude.
    A joint oscillating over ±0.06 rad gets the same Sm as one oscillating
    over ±1 rad if they are equally jittery relative to their own swing.
    Use this instead of sm_per_joint when joints have very different ranges.
    """
    fs = 1.0 / dt
    scores = []
    for j in range(chunk.shape[1]):
        x   = chunk[:, j] - np.mean(chunk[:, j])
        std = np.std(x)
        if std > 1e-9:
            x = x / std
        F     = np.abs(np.fft.rfft(x)) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=dt)
        n     = len(freqs)
        scores.append((2.0 / (n * fs)) * float(np.sum(F * freqs)))
    return np.array(scores, dtype=np.float32)


class ActionProcessor:
    """Abstract base class for per-action processors.

    Applied to one action at a time after the policy postprocessor.
    Interface:
        processor = SomeProcessor(...)
        processor.reset()               # call at episode start
        action_out = processor(action)  # shape (action_dim,)
    """

    def reset(self) -> None:
        """Reset internal state. Called at the start of each episode."""

    def __call__(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ChunkProcessor:
    """Abstract base class for chunk-level processors.

    Applied to the full N-step action chunk from the VLA before execution.
    SmolVLA generates chunk_size=50 actions per forward pass.
    The eval script hooks the policy via wrap_policy_with_chunk_processor().

    Interface:
        processor = SomeChunkProcessor(...)
        processor.reset()              # call at episode start
        chunk_out = processor(chunk)   # (T, action_dim) → (T, action_dim)
    """

    def reset(self) -> None:
        """Reset internal state. Called at the start of each episode."""

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
