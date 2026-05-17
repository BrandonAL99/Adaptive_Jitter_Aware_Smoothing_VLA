"""Processor registry, factory, and policy hook."""

from __future__ import annotations

from .base        import ActionProcessor, ChunkProcessor
from .identity    import IdentityProcessor
from .lowpass     import LowPassFilterProcessor
from .ajas        import AJASProcessor
from .ajas_spline import AJASSplineProcessor
from .pursuit     import PurePursuitProcessor

_REGISTRY: dict[str, type] = {
    "identity":    IdentityProcessor,
    "lowpass":     LowPassFilterProcessor,
    "ajas":        AJASProcessor,
    "ajas_spline": AJASSplineProcessor,
    "pursuit":     PurePursuitProcessor,
}


def list_processors() -> list[str]:
    """Return the names of all registered processors."""
    return list(_REGISTRY.keys())


def get_processor(name: str, **kwargs) -> ActionProcessor | ChunkProcessor:
    """Instantiate a processor by name.

    Parameters
    ----------
    name   : one of list_processors()
    kwargs : passed to the processor constructor
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown processor '{name}'. Available: {list_processors()}")
    cls = _REGISTRY[name]
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()


def wrap_policy_with_chunk_processor(policy, chunk_processor: ChunkProcessor):
    """Monkey-patch SmolVLA's _get_action_chunk to apply a ChunkProcessor.

    The hook:
      1. Calls the original _get_action_chunk to get the raw (B, T, action_dim) tensor.
      2. Passes the (T, action_dim) numpy array to chunk_processor.
      3. Stores the raw chunk in chunk_processor._raw_chunk for raw-action recording.
      4. Returns the processed chunk as a tensor with the same dtype/device.
    """
    import torch

    chunk_processor._raw_chunk = None
    chunk_processor._raw_idx   = 0

    original_get_chunk = policy._get_action_chunk

    def hooked_get_chunk(batch, noise=None, **kwargs):
        chunk_tensor = original_get_chunk(batch, noise, **kwargs)
        chunk_np     = chunk_tensor[0].cpu().numpy()
        processed_np = chunk_processor(chunk_np)
        chunk_processor._raw_chunk = chunk_np.copy()
        chunk_processor._raw_idx   = 0
        return (
            torch.from_numpy(processed_np)
            .unsqueeze(0)
            .to(dtype=chunk_tensor.dtype, device=chunk_tensor.device)
        )

    policy._get_action_chunk = hooked_get_chunk
    return policy
