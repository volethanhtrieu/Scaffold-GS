"""Small, explicit PyTorch runtime controls for reproducible GPU profiles."""

from __future__ import annotations

from typing import Dict

import torch

from arguments import str2bool, validate_tf32_mode


def configure_torch_runtime(
    *,
    tf32_mode: str = "default",
    cudnn_benchmark: bool = False,
) -> Dict[str, object]:
    """Apply opt-in CUDA math settings and return their resolved values.

    ``tf32_mode="default"`` leaves the installed PyTorch defaults unchanged.
    ``"enabled"`` allows TF32 for float32 matmuls and cuDNN convolutions,
    while ``"disabled"`` explicitly requests IEEE float32 behavior.  The
    output dtype remains float32; only internal CUDA math is affected.
    """

    mode = validate_tf32_mode(tf32_mode)
    benchmark = str2bool(cudnn_benchmark)

    # PyTorch 2.9+ exposes the explicit ``fp32_precision`` policy.  Prefer it
    # when available because mixing that API with the legacy ``allow_tf32``
    # switches is deprecated. Older 1.x/2.x builds use the latter controls.
    has_explicit_precision_api = all(
        hasattr(backend, "fp32_precision")
        for backend in (
            torch.backends.cuda.matmul,
            torch.backends.cudnn,
        )
    )
    if mode != "default":
        enabled = mode == "enabled"
        if has_explicit_precision_api:
            precision = "tf32" if enabled else "ieee"
            torch.backends.cuda.matmul.fp32_precision = precision
            torch.backends.cudnn.fp32_precision = precision
        else:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(
                    "high" if enabled else "highest"
                )
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.backends.cudnn.allow_tf32 = enabled

    torch.backends.cudnn.benchmark = benchmark

    resolved = {
        "tf32_mode": mode,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    if has_explicit_precision_api:
        resolved["cuda_matmul_fp32_precision"] = str(
            torch.backends.cuda.matmul.fp32_precision
        )
        resolved["cudnn_fp32_precision"] = str(
            torch.backends.cudnn.fp32_precision
        )
    else:
        resolved["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    if not has_explicit_precision_api and hasattr(
        torch, "get_float32_matmul_precision"
    ):
        resolved["float32_matmul_precision"] = (
            torch.get_float32_matmul_precision()
        )
    elif not has_explicit_precision_api:
        resolved["cuda_matmul_allow_tf32"] = bool(
            torch.backends.cuda.matmul.allow_tf32
        )
    return resolved


__all__ = ["configure_torch_runtime"]
