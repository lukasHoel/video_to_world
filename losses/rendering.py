"""
Rendering / image reconstruction losses shared across training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PixelReconstructionLossWeights:
    l1: float = 0.1
    lpips: float = 0.1


def init_lpips(device: torch.device | str, net: str = "vgg"):
    """
    Initialize LPIPS (frozen, eval mode).

    Kept as a tiny helper so training scripts don't repeat boilerplate.
    """
    import lpips

    lpips_fn = lpips.LPIPS(net=net).to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False
    return lpips_fn


def pixel_reconstruction_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    *,
    lpips_fn=None,
    weights: PixelReconstructionLossWeights = PixelReconstructionLossWeights(),
    clamp_max: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Standard pixel-wise reconstruction loss used in this repo:
        MSE + 0.1 * L1 + lpips_weight * LPIPS

    Args:
        rendered: (B, 3, H, W) in [0, 1]
        gt:       (B, 3, H, W) in [0, 1]
    """
    loss_mse = F.mse_loss(rendered, gt)
    loss_l1 = F.l1_loss(rendered, gt)

    loss_lpips = torch.tensor(0.0, device=rendered.device)
    if lpips_fn is not None and weights.lpips != 0:
        # LPIPS expects images in [-1, 1]
        loss_lpips = lpips_fn(rendered * 2 - 1, gt * 2 - 1).mean()

    total = loss_mse + weights.l1 * loss_l1 + weights.lpips * loss_lpips
    if clamp_max is not None:
        total = total.clamp(max=clamp_max)

    return total, {
        "mse": loss_mse,
        "l1": loss_l1,
        "lpips": loss_lpips,
    }
