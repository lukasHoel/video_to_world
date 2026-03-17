from __future__ import annotations

import time
from typing import Any, Iterable, Tuple

import numpy as np
import open3d as o3d
import torch

from utils.knn import build_kdtree, build_torch_kdtree
from utils.logging import get_logger
from utils.normals import estimate_normals


logger = get_logger(__name__)


def _coords_to_hash(coords: torch.Tensor, hash_scale: float) -> torch.Tensor:
    """Convert 3D voxel coordinates to 1D hash integers."""
    return (coords[:, 0] * hash_scale * (2**21) + coords[:, 1] * hash_scale + coords[:, 2]).to(torch.int64)


def merge_new_points_with_model(
    model_points: torch.Tensor,
    model_colors: torch.Tensor | None,
    model_normals: torch.Tensor,
    new_points: torch.Tensor,
    new_colors: torch.Tensor | None,
    voxel_size: float = 0.01,
    color_thresh: float | None = None,
    verbose: bool = False,
    downsample_new_points: bool = True,
    voxel_size_downsample: float | None = None,
    knn_backend: str = "cpu_kdtree",
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, Any]:
    """Merge new points into an existing model without modifying existing points."""
    device = model_points.device
    dtype = model_points.dtype

    if voxel_size_downsample is None:
        voxel_size_downsample = voxel_size

    if verbose:
        logger.info("[merge_new_points_with_model] Starting merge:")
        logger.info(
            "  Model points: %d, New points: %d",
            model_points.shape[0],
            new_points.shape[0],
        )
        logger.info("  Voxel size: %s, Color threshold: %s", voxel_size, color_thresh)
        logger.info(
            "  Downsample new points: %s, Downsample voxel size: %s",
            downsample_new_points,
            voxel_size_downsample,
        )

    if new_points.numel() == 0:
        empty_mask = torch.zeros(0, device=device, dtype=torch.bool)
        return model_points, model_colors, model_normals, empty_mask, None

    skip_voxel_matching = (
        color_thresh is not None and color_thresh < 0 and model_colors is not None and new_colors is not None
    )

    if skip_voxel_matching:
        keep_mask = torch.ones(new_points.shape[0], device=device, dtype=torch.bool)
        is_in_existing_voxel = torch.zeros(new_points.shape[0], device=device, dtype=torch.bool)
        removed_by_color_thresh = torch.zeros(new_points.shape[0], device=device, dtype=torch.bool)

        if verbose:
            logger.info("  Skipping voxel matching (color_thresh < 0, all points will be kept)")
    else:
        existing_coords = torch.floor(model_points / voxel_size).to(torch.int64)
        new_coords = torch.floor(new_points / voxel_size).to(torch.int64)
        unique_existing_coords, unique_indices = torch.unique(existing_coords, dim=0, return_inverse=True)

        arange = torch.arange(existing_coords.shape[0], device=device, dtype=torch.float32)
        first_occurrence_idx = torch.full(
            (len(unique_existing_coords),),
            float("inf"),
            device=device,
            dtype=torch.float32,
        )
        first_occurrence_idx.index_reduce_(0, unique_indices, arange, reduce="amin", include_self=False)
        first_occurrence_idx = first_occurrence_idx.to(torch.long)

        max_coord_val = max(
            existing_coords.abs().max().item() if existing_coords.numel() > 0 else 0,
            new_coords.abs().max().item() if new_coords.numel() > 0 else 0,
        )
        hash_scale = min(2**21, (2**63 - 1) // (max_coord_val * 3 + 1)) if max_coord_val > 0 else 2**21

        unique_existing_hash = _coords_to_hash(unique_existing_coords, hash_scale)
        new_coords_hash = _coords_to_hash(new_coords, hash_scale)

        sorted_unique_hash, sort_idx = torch.sort(unique_existing_hash)
        sorted_first_occurrence = first_occurrence_idx[sort_idx]
        search_positions = torch.searchsorted(sorted_unique_hash, new_coords_hash, right=False)
        valid_mask = search_positions < len(sorted_unique_hash)
        is_in_existing_voxel = valid_mask & (
            sorted_unique_hash[torch.clamp(search_positions, 0, len(sorted_unique_hash) - 1)] == new_coords_hash
        )

        keep_mask = torch.ones(new_points.shape[0], device=device, dtype=torch.bool)
        removed_by_color_thresh = torch.zeros(new_points.shape[0], device=device, dtype=torch.bool)

        if is_in_existing_voxel.any():
            if color_thresh is not None and model_colors is not None and new_colors is not None:
                matched_positions = torch.clamp(
                    search_positions[is_in_existing_voxel],
                    0,
                    len(sorted_unique_hash) - 1,
                )
                existing_indices = sorted_first_occurrence[matched_positions]
                color_diffs = torch.mean(
                    torch.abs(model_colors[existing_indices] - new_colors[is_in_existing_voxel]),
                    dim=1,
                )
                keep_mask[is_in_existing_voxel] = color_diffs > color_thresh
                removed_by_color_thresh[is_in_existing_voxel] = color_diffs <= color_thresh

                if verbose:
                    logger.info(
                        "  Points in existing voxels: %d",
                        int(is_in_existing_voxel.sum().item()),
                    )
                    logger.info(
                        "  Removed by color threshold: %d",
                        int(removed_by_color_thresh.sum().item()),
                    )
                    logger.info(
                        "  Kept (color diff > thresh): %d",
                        int(keep_mask[is_in_existing_voxel].sum().item()),
                    )
            else:
                keep_mask[is_in_existing_voxel] = False
                if verbose:
                    logger.info(
                        "  Points in existing voxels: %d (all discarded, no color check)",
                        int(is_in_existing_voxel.sum().item()),
                    )

    filtered_new_points = new_points[keep_mask]
    filtered_new_colors = new_colors[keep_mask] if new_colors is not None else None

    if verbose:
        logger.info("")
        logger.info("  === Point Filtering Breakdown ===")
        logger.info("  Total new points: %d", int(new_points.shape[0]))
        logger.info(
            "  Points in existing voxels: %d",
            int(is_in_existing_voxel.sum().item()),
        )
        logger.info(
            "    -> Removed by color threshold: %d",
            int(removed_by_color_thresh.sum().item()),
        )
        logger.info(
            "    -> Kept (color diff > thresh): %d",
            int((is_in_existing_voxel & keep_mask).sum().item()),
        )
        logger.info(
            "  Points in NEW voxels: %d",
            int((~is_in_existing_voxel).sum().item()),
        )
        logger.info(
            "  Total removed by filtering: %d",
            int((~keep_mask).sum().item()),
        )
        logger.info(
            "  Points kept after filtering: %d",
            int(keep_mask.sum().item()),
        )

    if filtered_new_points.shape[0] == 0:
        if verbose:
            logger.info("  No new points to merge after filtering")
        return model_points, model_colors, model_normals, keep_mask, None

    if downsample_new_points:
        filtered_new_coords_downsample = torch.floor(filtered_new_points / voxel_size_downsample).to(torch.int64)
        unique_new_coords, inverse_indices = torch.unique(filtered_new_coords_downsample, dim=0, return_inverse=True)
        num_unique_voxels = unique_new_coords.shape[0]

        if verbose:
            logger.info("")
            logger.info("  === Voxel Downsampling Breakdown ===")
            logger.info(
                "  Points before downsampling: %d",
                int(filtered_new_points.shape[0]),
            )
            logger.info("  Unique voxels: %d", int(num_unique_voxels))
            logger.info(
                "  Points removed by downsampling: %d",
                int(filtered_new_points.shape[0] - num_unique_voxels),
            )
            if num_unique_voxels > 0:
                logger.info(
                    "  Average points per voxel: %.2f",
                    float(filtered_new_points.shape[0] / num_unique_voxels),
                )

        new_centroids = torch.zeros((num_unique_voxels, 3), device=device, dtype=dtype)
        new_centroids.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), filtered_new_points)
        voxel_counts = torch.bincount(inverse_indices, minlength=num_unique_voxels).float()
        new_centroids /= voxel_counts.unsqueeze(1)

        new_centroid_colors = None
        if filtered_new_colors is not None:
            new_centroid_colors = torch.zeros(
                (num_unique_voxels, 3),
                device=device,
                dtype=filtered_new_colors.dtype,
            )
            new_centroid_colors.scatter_add_(
                0,
                inverse_indices.unsqueeze(1).expand(-1, 3),
                filtered_new_colors,
            )
            new_centroid_colors /= voxel_counts.unsqueeze(1)
    else:
        new_centroids = filtered_new_points
        new_centroid_colors = filtered_new_colors
        if verbose:
            logger.info("")
            logger.info("  === Downsampling Skipped ===")
            logger.info(
                "  Using all %d filtered points without downsampling",
                int(filtered_new_points.shape[0]),
            )

    updated_points = torch.cat([model_points, new_centroids], dim=0)

    kdtree_for_normals = None
    t_kdtree_build_start = time.perf_counter()
    if knn_backend == "cpu_kdtree":
        kdtree_for_normals = build_kdtree(updated_points)
    elif knn_backend == "gpu_kdtree":
        kdtree_for_normals = build_torch_kdtree(updated_points)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    t_kdtree_build_end = time.perf_counter()
    kdtree_build_time_ms = (t_kdtree_build_end - t_kdtree_build_start) * 1000.0
    if verbose:
        logger.info(
            "  KDTree build during merge (backend=%s) on %d points took %.1f ms",
            knn_backend,
            int(updated_points.shape[0]),
            kdtree_build_time_ms,
        )

    new_normals, _ = estimate_normals(
        updated_points,
        backend=knn_backend,
        start_idx=model_points.shape[0],
        prebuilt_tree=kdtree_for_normals,
    )
    updated_normals = torch.cat([model_normals, new_normals], dim=0)
    updated_colors = (
        torch.cat([model_colors, new_centroid_colors], dim=0)
        if (model_colors is not None and new_centroid_colors is not None)
        else model_colors
    )

    if verbose:
        logger.info("")
        logger.info("  === Final Summary ===")
        logger.info("  New centroids added: %d", int(new_centroids.shape[0]))
        logger.info("  Updated model points: %d", int(updated_points.shape[0]))

    return (
        updated_points,
        updated_colors,
        updated_normals,
        keep_mask,
        kdtree_for_normals,
    )


def merge_point_clouds(
    pcd_list: Iterable[o3d.geometry.PointCloud],
) -> o3d.geometry.PointCloud:
    """Merge a list of Open3D point clouds into a single point cloud."""
    points = []
    colors = []

    for pcd in pcd_list:
        points.append(np.asarray(pcd.points))
        if pcd.has_colors():
            colors.append(np.asarray(pcd.colors))

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(points))

    if colors:
        merged.colors = o3d.utility.Vector3dVector(np.vstack(colors))

    return merged
