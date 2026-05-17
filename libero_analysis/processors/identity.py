from __future__ import annotations

import numpy as np
from .base import ActionProcessor


class IdentityProcessor(ActionProcessor):
    """Pass-through: returns the action unchanged."""

    def __call__(self, action: np.ndarray) -> np.ndarray:
        return action
