#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def _as_bchw(image: torch.Tensor, name: str):
    """Normalize a CHW/BCHW image to BCHW and return whether it was unbatched."""

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if image.ndim == 3:
        return image.unsqueeze(0), True
    if image.ndim == 4:
        return image, False
    raise ValueError(
        f"{name} must use CHW or BCHW layout (received {tuple(image.shape)})"
    )


def _sobel_gradient_maps(image: torch.Tensor):
    """Return horizontal/vertical Sobel maps for a BCHW floating tensor."""

    channels = image.shape[1]
    # PAPER: fixed 3x3 Sobel kernels highlight horizontal and vertical
    # texture/geometry changes.  They are constructed on the input device and
    # dtype, never assumed to be CUDA tensors.
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    sobel_x = sobel_x.expand(channels, 1, 3, 3)
    sobel_y = sobel_y.expand(channels, 1, 3, 3)
    gradient_x = F.conv2d(image, sobel_x, padding=1, groups=channels)
    gradient_y = F.conv2d(image, sobel_y, padding=1, groups=channels)
    return gradient_x, gradient_y


def selective_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute SOGS's selective gradient loss for CHW or BCHW images.

    The Sobel operator is applied independently to every channel.  Following
    Eq. (14)--(16), each channel's absolute horizontal/vertical discrepancy is
    summed over ``H x W`` and scaled by ``1/sqrt(HW)``; the pixel-wise absolute
    discrepancies also serve as dynamic region weights.  ``prediction`` and
    ``target`` must have identical shape and finite floating-point values.
    ``reduction`` may be ``"mean"`` or ``"sum"`` and always returns a scalar.
    """

    prediction_bchw, _ = _as_bchw(prediction, "prediction")
    target_bchw, _ = _as_bchw(target, "target")
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes "
            f"(received {tuple(prediction.shape)} and {tuple(target.shape)})"
        )
    if prediction_bchw.shape[1] < 1:
        raise ValueError("images must contain at least one channel")
    if prediction_bchw.shape[-2] < 1 or prediction_bchw.shape[-1] < 1:
        raise ValueError("images must have positive height and width")
    if not prediction_bchw.is_floating_point() or not target_bchw.is_floating_point():
        raise ValueError("prediction and target must use floating-point dtypes")
    if prediction_bchw.device != target_bchw.device:
        raise ValueError("prediction and target must be on the same device")
    if not torch.isfinite(prediction_bchw).all().item() or not torch.isfinite(
        target_bchw
    ).all().item():
        raise ValueError("prediction and target must not contain NaN or infinite values")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    # INFERENCE: promote half-precision Sobel convolution/accumulation to
    # float32.  This avoids unsupported CPU half conv2d operations and
    # overflow while preserving gradients back to the source tensors.
    work_dtype = (
        torch.float32
        if prediction_bchw.dtype in (torch.float16, torch.bfloat16)
        else prediction_bchw.dtype
    )
    prediction_bchw = prediction_bchw.to(dtype=work_dtype)
    target_bchw = target_bchw.to(dtype=work_dtype)
    prediction_gradient_x, prediction_gradient_y = _sobel_gradient_maps(
        prediction_bchw
    )
    target_gradient_x, target_gradient_y = _sobel_gradient_maps(target_bchw)

    difference_x = prediction_gradient_x - target_gradient_x
    difference_y = prediction_gradient_y - target_gradient_y
    # PAPER: Eq. (14) scales the spatial absolute-error sum by 1/sqrt(HW);
    # Eq. (15) reuses the pixel-wise absolute error as a dynamic weight map.
    weight_x = difference_x.abs()
    weight_y = difference_y.abs()
    height, width = prediction_bchw.shape[-2:]
    spatial_scale = float(height * width) ** 0.5
    discrepancy_x = weight_x.sum(dim=(-2, -1), keepdim=True) / spatial_scale
    discrepancy_y = weight_y.sum(dim=(-2, -1), keepdim=True) / spatial_scale
    loss_map = weight_x * discrepancy_x + weight_y * discrepancy_y
    if reduction == "sum":
        loss = loss_map.sum()
    else:
        loss = loss_map.mean()
    # A finite check catches overflow in very low-precision execution without
    # silently propagating an invalid training objective.
    if not torch.isfinite(loss).all().item():
        raise ValueError("selective gradient loss became NaN or infinite")
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
