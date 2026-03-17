"""
Point-cloud downsampling utilities.

``voxel_grid_downsample`` averages points inside uniform voxel cells.

``downsample_to_target`` incrementally grows the voxel size starting from
0.001 m until the point count is within ±50 % of the target — works well
for highly non-uniform indoor scenes.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def voxel_grid_downsample(
    points: torch.Tensor,
    colors: torch.Tensor | None = None,
    voxel_size: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Downsample a point cloud by averaging within voxel cells.

    Args:
        points: (N, 3) point positions.
        colors: (N, 3) per-point colors (optional).
        voxel_size: Edge length of each voxel.

    Returns:
        (downsampled_points, downsampled_colors) — colours are ``None`` when
        the input ``colors`` is ``None``.
    """
    coords = torch.floor(points / voxel_size)
    _, inverse = torch.unique(coords, dim=0, return_inverse=True)

    num_voxels = inverse.max() + 1
    downsampled = torch.zeros((num_voxels, 3), dtype=points.dtype, device=points.device)
    downsampled.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), points)

    downsampled_colors: torch.Tensor | None = None
    if colors is not None:
        downsampled_colors = torch.zeros((num_voxels, 3), dtype=colors.dtype, device=colors.device)
        downsampled_colors.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), colors)

    counts = torch.bincount(inverse).float().unsqueeze(1)
    downsampled = downsampled / counts
    if downsampled_colors is not None:
        downsampled_colors = downsampled_colors / counts

    return downsampled, downsampled_colors


def downsample_to_target(
    points: torch.Tensor,
    colors: torch.Tensor | None = None,
    target_count: int = 500_000,
    start_voxel: float = 0.001,
    voxel_step: float = 0.001,
    tolerance: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Downsample a point cloud to approximately *target_count* points.

    Starts with a tiny voxel (1 mm) and linearly increases the size by
    *voxel_step* each iteration until the output count is within
    ``target_count * (1 ± tolerance)``.  Returns as soon as the tolerance
    is reached — no random subsampling, so the spatial structure produced
    by voxel averaging is fully deterministic and reproducible.

    Args:
        points: (N, 3) point positions.
        colors: (N, 3) per-point colours (optional, averaged per voxel).
        target_count: Desired number of output points.
        start_voxel: Initial voxel edge-length in metres (default 1 mm).
        voxel_step: Increment added to voxel size each iteration (metres).
        tolerance: Acceptable relative deviation from *target_count*
            (default 0.5 → ±50 %).

    Returns:
        (downsampled_points, downsampled_colors)
    """
    N = points.shape[0]
    if N <= target_count:
        return points, colors

    lo = target_count * (1.0 - tolerance)
    hi = target_count * (1.0 + tolerance)

    vs = start_voxel
    best_pts, best_cols = points, colors

    while True:
        ds_pts, ds_cols = voxel_grid_downsample(points, colors, voxel_size=vs)
        count = ds_pts.shape[0]
        logger.info(
            "downsample_to_target: voxel=%.4f → %d points (target %d ± %.0f%%)",
            vs,
            count,
            target_count,
            tolerance * 100,
        )

        if count <= hi:
            # We've entered or passed below the tolerance window.
            if count >= lo:
                # Inside window — perfect, use this result.
                best_pts, best_cols = ds_pts, ds_cols
            else:
                # Overshot below the window.  Use whichever is closer to
                # target: the current result or the previous (larger) one.
                prev_diff = abs(best_pts.shape[0] - target_count)
                curr_diff = abs(count - target_count)
                if curr_diff < prev_diff:
                    best_pts, best_cols = ds_pts, ds_cols
            break

        # Still above the window — remember this as the best-so-far and
        # increase the voxel size.
        best_pts, best_cols = ds_pts, ds_cols
        vs += voxel_step

    logger.info(
        "downsample_to_target: %d → %d points (target %d, final voxel %.4f)",
        N,
        best_pts.shape[0],
        target_count,
        vs,
    )

    return best_pts, best_cols
