"""Action processors for LIBERO evaluation.

Available processors
--------------------
identity  - IdentityProcessor        no-op passthrough
lowpass   - LowPassFilterProcessor   causal Butterworth IIR (per-action)
ajas      - AJASProcessor            Adaptive Jitter-Aware Smoothing (chunk, open-loop)
pursuit   - PurePursuitProcessor     Pure-pursuit closed-loop controller (chunk)

Usage
-----
from libero_analysis.processors import get_processor, list_processors

proc = get_processor("ajas", C=500, k_min=1, k_max=40, skip_joints=(6,))
"""

from .base        import ActionProcessor, ChunkProcessor, sm_per_joint
from .identity    import IdentityProcessor
from .lowpass     import LowPassFilterProcessor
from .ajas        import AJASProcessor
from .ajas_spline import AJASSplineProcessor
from .pursuit     import PurePursuitProcessor
from .registry    import get_processor, list_processors, wrap_policy_with_chunk_processor

__all__ = [
    "ActionProcessor",
    "ChunkProcessor",
    "sm_per_joint",
    "IdentityProcessor",
    "LowPassFilterProcessor",
    "AJASProcessor",
    "AJASSplineProcessor",
    "PurePursuitProcessor",
    "get_processor",
    "list_processors",
    "wrap_policy_with_chunk_processor",
]
