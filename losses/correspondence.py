"""
Correspondence-based geometric loss.

This loss is intentionally independent of how correspondences were generated
(dense matcher, keypoints, tracks, manual annotations, ...). It only assumes
that correspondences provide:
- source pixel coordinates
- reference pixel coordinates
- an optional per-match weight/confidence
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from models.roma_matcher import get_local_indices_for_pixels_batch


def compute_correspondence_loss_with_model_segments(
    matches_data: list[Any],
    src_points_transformed: torch.Tensor,  # (M_src, 3) transformed source points (filtered)
    model_points: torch.Tensor,  # Full model point cloud
    model_colors: Optional[torch.Tensor],  # Full model colors (or None)
    model_frame_segments: list[tuple[int, int]],  # (start_idx, end_idx) for each frame in model
    ref_frame_indices: list[int],  # Frame indices corresponding to model_frame_segments
    src_valid_pixel_indices: torch.Tensor,  # (M_src,) valid pixel indices for source frame
    model_valid_pixel_indices_list: list[torch.Tensor],  # List of valid pixel indices for ref frames
    H: int,
    W: int,
    src_colors: Optional[torch.Tensor] = None,  # (M_src, 3)
    color_loss_weight: float = 0.0,
    max_corr_dist: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute a weighted correspondence loss using model segments.

    `matches_data` elements must have at least these attributes:
      - ref_frame_idx: int
      - kpts_src: (K,2) tensor of (x,y) pixel coords in source
      - kpts_ref: (K,2) tensor of (x,y) pixel coords in reference
      - certainty: (K,) tensor of weights/confidences in [0,1] (or any non-negative weight)

    Returns:
        geometry_loss: scalar
        color_loss: scalar
        num_matches: number of matches used (after filtering)
    """
    device = src_points_transformed.device

    if not matches_data:
        z = torch.tensor(0.0, device=device)
        return z, z, 0

    frame_to_segment_idx = {idx: i for i, idx in enumerate(ref_frame_indices)}

    weighted_dist_sum = None
    weight_sum = None
    global_conf_gate = None
    num_matches = 0

    weighted_color_sum = None

    for match_data in matches_data:
        ref_idx = int(getattr(match_data, "ref_frame_idx"))
        if ref_idx not in frame_to_segment_idx:
            continue

        segment_idx = frame_to_segment_idx[ref_idx]
        start_idx, end_idx = model_frame_segments[segment_idx]

        ref_points = model_points[start_idx:end_idx]
        ref_valid_indices = model_valid_pixel_indices_list[segment_idx]

        ref_colors = None
        if model_colors is not None:
            ref_colors = model_colors[start_idx:end_idx]

        kpts_src = getattr(match_data, "kpts_src")
        kpts_ref = getattr(match_data, "kpts_ref")
        weights = getattr(match_data, "certainty")

        src_local_idx, src_valid_mask = get_local_indices_for_pixels_batch(
            pixels_x=kpts_src[:, 0],
            pixels_y=kpts_src[:, 1],
            valid_pixel_indices=src_valid_pixel_indices,
            H=H,
            W=W,
        )
        ref_local_idx, ref_valid_mask = get_local_indices_for_pixels_batch(
            pixels_x=kpts_ref[:, 0],
            pixels_y=kpts_ref[:, 1],
            valid_pixel_indices=ref_valid_indices,
            H=H,
            W=W,
        )

        both_valid = src_valid_mask & ref_valid_mask
        if int(both_valid.sum().item()) == 0:
            continue

        src_idx_valid = src_local_idx[both_valid]
        ref_idx_valid = ref_local_idx[both_valid]

        src_pts_valid = src_points_transformed[src_idx_valid]
        ref_pts_valid = ref_points[ref_idx_valid]
        w = weights[both_valid]

        dists_sq = ((src_pts_valid - ref_pts_valid) ** 2).sum(dim=1)

        if max_corr_dist is not None:
            thresh_sq = max_corr_dist * max_corr_dist
            dist_mask = dists_sq < thresh_sq
            if int(dist_mask.sum().item()) == 0:
                continue
            dists_sq = dists_sq[dist_mask]
            w = w[dist_mask]
            src_idx_valid = src_idx_valid[dist_mask]
            ref_idx_valid = ref_idx_valid[dist_mask]

        if int(dists_sq.numel()) == 0:
            continue

        match_weight_sum = w.sum()
        match_weighted_dist_sum = (w * dists_sq).sum()
        match_max_w = w.max()

        if weighted_dist_sum is None:
            weighted_dist_sum = match_weighted_dist_sum
            weight_sum = match_weight_sum
            global_conf_gate = match_max_w
        else:
            weighted_dist_sum = weighted_dist_sum + match_weighted_dist_sum
            weight_sum = weight_sum + match_weight_sum
            global_conf_gate = torch.maximum(global_conf_gate, match_max_w)

        num_matches += int(dists_sq.numel())

        if color_loss_weight > 0 and src_colors is not None and ref_colors is not None:
            src_c_valid = src_colors[src_idx_valid]
            ref_c_valid = ref_colors[ref_idx_valid]
            color_diff_sq = ((src_c_valid - ref_c_valid) ** 2).sum(dim=1)
            match_weighted_color_sum = (w * color_diff_sq).sum()
            if weighted_color_sum is None:
                weighted_color_sum = match_weighted_color_sum
            else:
                weighted_color_sum = weighted_color_sum + match_weighted_color_sum

    if weighted_dist_sum is None or weight_sum is None or global_conf_gate is None or num_matches == 0:
        z = torch.tensor(0.0, device=device)
        return z, z, 0

    eps = 1e-8
    if weight_sum > 0:
        weighted_mean = weighted_dist_sum / weight_sum.clamp_min(eps)
        geometry_loss = global_conf_gate * weighted_mean
    else:
        geometry_loss = torch.tensor(0.0, device=device)

    if weighted_color_sum is not None and weight_sum > 0:
        color_weighted_mean = weighted_color_sum / weight_sum.clamp_min(eps)
        color_loss = color_loss_weight * global_conf_gate * color_weighted_mean
    else:
        color_loss = torch.tensor(0.0, device=device)

    return geometry_loss, color_loss, num_matches
