"""Version-compatible activation checkpoint wrapper."""

import inspect

from torch.utils.checkpoint import checkpoint as torch_checkpoint


try:
    _SUPPORTS_USE_REENTRANT = (
        "use_reentrant" in inspect.signature(torch_checkpoint).parameters
    )
except (TypeError, ValueError):
    _SUPPORTS_USE_REENTRANT = False


def activation_checkpoint(function, *args):
    """Use legacy reentrant semantics explicitly when PyTorch supports it."""

    if _SUPPORTS_USE_REENTRANT:
        return torch_checkpoint(function, *args, use_reentrant=True)
    return torch_checkpoint(function, *args)


__all__ = ["activation_checkpoint"]
