from __future__ import annotations

import torch


def colors_to_intensity(colors: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB or single-channel colors to a 1D intensity tensor.

    Accepts:
      - (..., 3) RGB in [0,1] or [0,255]
      - (..., 1) or (...) single-channel

    Returns:
      - flattened (N,) float tensor
    """
    if colors.ndim == 1:
        c = colors.view(-1, 1)
    else:
        c = colors.reshape(-1, colors.shape[-1])

    if c.shape[1] == 1:
        intensity = c[:, 0]
    else:
        r = c[:, 0].float()
        g = c[:, 1].float()
        b = c[:, 2].float()
        intensity = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return intensity


def build_intrinsic_matrix(
    log_focal: torch.Tensor,
    pp_norm: torch.Tensor,
    imsizes: torch.Tensor,
    min_focal: torch.Tensor,
    max_focal: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a (3,3) intrinsic matrix from learnable log-focal and normalised principal point."""
    f = log_focal.exp().clamp(min=min_focal, max=max_focal)
    pp = pp_norm * imsizes

    K = torch.eye(3, device=log_focal.device, dtype=log_focal.dtype).clone()
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = pp[0]
    K[1, 2] = pp[1]
    return K
