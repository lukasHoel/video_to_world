"""
2D Gaussian Splatting specific losses.

These losses complement the existing rendering and geometric losses with
regularization terms specific to the 2DGS representation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normal_consistency_loss(
    render_normals: torch.Tensor,
    surf_normals: torch.Tensor,
    render_alphas: torch.Tensor,
    alpha_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Penalize disagreement between rendered Gaussian normals and
    depth-derived surface normals.

    Args:
        render_normals: (1, H, W, 3) normals from Gaussian orientations
        surf_normals:   (1, H, W, 3) normals derived from rendered depth
        render_alphas:  (1, H, W, 1) alpha/opacity accumulation
        alpha_threshold: only compute loss where alpha exceeds this
    Returns:
        scalar loss
    """
    mask = (render_alphas > alpha_threshold).squeeze(-1)  # (1, H, W)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=render_normals.device)

    # Cosine similarity (1 - dot = 0 when aligned, 2 when opposite)
    dot = (render_normals * surf_normals).sum(dim=-1)  # (1, H, W)
    loss = (1.0 - dot)[mask]
    return loss.mean()


def distortion_loss(render_distort: torch.Tensor) -> torch.Tensor:
    """
    Encourage compact ray-space distributions (from gsplat 2DGS output).

    Args:
        render_distort: (1, H, W, 1) distortion values from rasterization_2dgs
    Returns:
        scalar loss
    """
    return render_distort.mean()


def opacity_regularization_loss(opacities: torch.Tensor) -> torch.Tensor:
    """
    Encourage binary opacities for clean geometry.

    Applies a loss that is minimized when opacity is 0 or 1:
        L = -log(p) - log(1-p)  (binary cross-entropy with itself)

    Args:
        opacities: (N, 1) sigmoid-activated opacities in (0, 1)
    Returns:
        scalar loss
    """
    eps = 1e-6
    opacities = opacities.clamp(eps, 1.0 - eps)
    return (-opacities * torch.log(opacities) - (1 - opacities) * torch.log(1 - opacities)).mean()


def scale_regularization_loss(
    scales: torch.Tensor,
    max_log_scale: float = 2.0,
) -> torch.Tensor:
    """
    Prevent Gaussians from growing too large.

    Penalizes log-space scales that exceed max_log_scale.

    Args:
        scales: (N, 3) log-space scales (before exp)
        max_log_scale: threshold above which penalty kicks in
    Returns:
        scalar loss
    """
    excess = torch.relu(scales - max_log_scale)
    return excess.mean()


def depth_loss(
    rendered_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha_mask: torch.Tensor,
    alpha_threshold: float = 0.5,
) -> torch.Tensor:
    """L1 loss between rendered expected depth and GT depth in camera z-space.

    Both the rendered expected depth (from ``gsplat`` with ``render_mode="RGB+ED"``)
    and the GT depth (from the DA3 NPZ) are camera z-depth — the z-coordinate
    in camera space.  They can be compared directly.

    The loss is only computed at pixels where:
      - ``valid_mask`` is True (GT depth is finite, positive, and above conf threshold)
      - rendered alpha exceeds ``alpha_threshold`` (the Gaussian model covers this pixel)

    Args:
        rendered_depth: (1, H, W, 1) expected depth from gsplat.
        gt_depth:       (H, W) raw depth map.
        valid_mask:     (H, W) bool tensor.
        alpha_mask:     (1, H, W, 1) rendered alpha accumulation.
        alpha_threshold: minimum alpha for a pixel to contribute.

    Returns:
        Scalar L1 loss, or 0 if fewer than 10 valid pixels.
    """
    rd = rendered_depth[0, :, :, 0]  # (H, W)
    ra = alpha_mask[0, :, :, 0]  # (H, W)

    mask = valid_mask & (ra > alpha_threshold)
    if mask.sum() < 10:
        return torch.tensor(0.0, device=rendered_depth.device)

    return F.l1_loss(rd[mask], gt_depth[mask])
