import os
import time

import open3d as o3d
import torch
from typing import Optional, Callable

from losses.tv import tv_loss
from data.data_loading import torch_to_o3d_pcd
from utils.image import colors_to_intensity
from utils.knn import query_knn_with_backend
from utils.normals import estimate_normals
from utils.geometry import se3_apply


def _apply_deformation(
    world_points: torch.Tensor,
    local_deform,
    global_rigid: torch.Tensor,
) -> torch.Tensor:
    """Apply local deformation then global rigid transform: world → canonical.

    Args:
        world_points: (N, 3) points in world space
        local_deform: Deformation module or callable returning (N, 6) SE3 twists
        global_rigid: (6,) SE3 twist for global rigid transform

    Returns:
        (N, 3) points in canonical space
    """
    # Local deformation
    xi_local = local_deform(world_points)  # (N, 6)
    pts_after_local = se3_apply(xi_local, world_points)  # (N, 3)

    # Global rigid
    pts_canonical = se3_apply(global_rigid, pts_after_local)  # (N, 3)

    return pts_canonical


def _apply_deformation_chunked(
    world_points: torch.Tensor,
    local_deform,
    global_rigid: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Chunked variant of `_apply_deformation` to limit peak memory (TCNN workspace)."""
    if chunk_size <= 0 or world_points.shape[0] <= chunk_size:
        return _apply_deformation(world_points, local_deform, global_rigid)
    outs = []
    for s in range(0, world_points.shape[0], chunk_size):
        pts = world_points[s : s + chunk_size]
        outs.append(_apply_deformation(pts, local_deform, global_rigid))
    return torch.cat(outs, dim=0)


def _build_model_segments(per_frame_world_points: list[torch.Tensor]) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start = 0
    for pts in per_frame_world_points:
        end = start + int(pts.shape[0])
        segs.append((start, end))
        start = end
    return segs


def _global_to_frame_local(
    global_idx: torch.Tensor,
    *,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global indices into (frame_id, local_idx) using segment boundaries."""
    # frame_id = smallest f where global_idx < seg_ends[f]
    #
    # Important: use right=True so that indices exactly on a segment boundary
    # (e.g. global_idx == seg_ends[f]) are assigned to the *next* frame, not f.
    # Example: if segs are [0,10), [10,20), then idx=10 belongs to frame 1.
    frame_id = torch.bucketize(global_idx, seg_ends, right=True)
    local_idx = global_idx - seg_starts[frame_id]
    return frame_id, local_idx


def _gather_deformed_points_for_global_indices(
    global_indices: torch.Tensor,
    *,
    per_frame_world_points: list[torch.Tensor],
    per_frame_local_deform: list,
    per_frame_global_rigid: list[torch.Tensor],
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
    deform_chunk_size: int,
) -> torch.Tensor:
    """Compute canonical points for a set of global indices with gradients."""
    device = global_indices.device
    frame_id, local_idx = _global_to_frame_local(global_indices, seg_starts=seg_starts, seg_ends=seg_ends)
    out = torch.empty((global_indices.shape[0], 3), device=device, dtype=per_frame_world_points[0].dtype)

    for f in range(len(per_frame_world_points)):
        mask_f = frame_id == f
        if not mask_f.any():
            continue
        local_f = local_idx[mask_f]
        # Safety check: if this triggers, something upstream produced invalid global indices.
        if (local_f < 0).any() or (local_f >= per_frame_world_points[f].shape[0]).any():
            bad = local_f[(local_f < 0) | (local_f >= per_frame_world_points[f].shape[0])]
            raise IndexError(
                f"Local indices out of bounds for frame {f}: "
                f"min={int(bad.min().item())}, max={int(bad.max().item())}, "
                f"frame_points={int(per_frame_world_points[f].shape[0])}."
            )
        uniq_local_f = torch.unique(local_f, sorted=True)
        pts_f = per_frame_world_points[f][uniq_local_f]
        deformed_f = _apply_deformation_chunked(
            pts_f,
            per_frame_local_deform[f],
            per_frame_global_rigid[f],
            chunk_size=deform_chunk_size,
        )
        pos = torch.searchsorted(uniq_local_f, local_f)
        out[mask_f] = deformed_f[pos]
    return out


def _sample_loo_pairs_two_stage(
    nn_idx_all: torch.Tensor,
    loo_valid_all: torch.Tensor,
    *,
    max_pairs: int,
    pairs_per_src: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (src_idx, tgt_idx) pairs without materializing all valid pairs.

    Sampling scheme:
    - sample src rows uniformly over points
    - sample one of the valid neighbors uniformly within that row using argmax trick

    Returns:
        src_idx: (S,)
        tgt_idx: (S,)
        weights: (S,) importance weights proportional to (#valid neighbors for src)
    """
    device = nn_idx_all.device
    N, K = nn_idx_all.shape
    if max_pairs <= 0:
        return (
            torch.empty((0,), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.float32),
        )

    # We sample sources in batches until we collect enough valid pairs.
    S_target = int(max_pairs)
    src_out: list[torch.Tensor] = []
    tgt_out: list[torch.Tensor] = []
    w_out: list[torch.Tensor] = []

    # Oversample factor to reduce loop iterations.
    batch_src = max(1024, min(N, S_target * 2))
    max_rounds = 50  # safety

    for _ in range(max_rounds):
        remaining = S_target - sum(int(x.numel()) for x in src_out)
        if remaining <= 0:
            break
        # Sample candidate sources uniformly.
        src = torch.randint(0, N, (batch_src,), device=device)
        valid_row = loo_valid_all[src]  # (B,K)
        counts = valid_row.sum(dim=1)  # (B,)
        has_any = counts > 0
        if not has_any.any():
            continue
        src = src[has_any]
        valid_row = valid_row[has_any]
        counts = counts[has_any]

        # Sample up to `pairs_per_src` neighbors per src.
        # Use argmax of iid Uniform(0,1) over valid positions ⇒ uniform among valids.
        reps = int(max(1, pairs_per_src))
        src_rep = src.repeat_interleave(reps)
        valid_rep = valid_row.repeat_interleave(reps, dim=0)
        counts_rep = counts.repeat_interleave(reps)

        rnd = torch.rand(valid_rep.shape, device=device)
        rnd = torch.where(valid_rep, rnd, torch.full_like(rnd, -1.0))
        col = rnd.argmax(dim=1)  # (B*reps,)
        # Note: if a row had no valid, it was filtered out already.
        tgt = nn_idx_all[src_rep, col]

        # Keep only as many as we still need.
        take = min(int(tgt.shape[0]), remaining)
        if take <= 0:
            break
        src_out.append(src_rep[:take])
        tgt_out.append(tgt[:take])
        w_out.append(counts_rep[:take].to(torch.float32))

    if not src_out:
        return (
            torch.empty((0,), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.float32),
        )

    src_idx = torch.cat(src_out, dim=0)
    tgt_idx = torch.cat(tgt_out, dim=0)
    weights = torch.cat(w_out, dim=0)
    return src_idx, tgt_idx, weights


def _sample_uniform_in_bbox(
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample points uniformly within a bounding box."""
    rand = torch.rand(n_samples, 3, device=device)
    return bbox_min + rand * (bbox_max - bbox_min)


def global_opt(
    per_frame_world_points: list[torch.Tensor],
    per_frame_world_colors: list[torch.Tensor],
    per_frame_global_rigid: list[torch.Tensor],
    per_frame_local_deform: list,
    # LOO consensus loss parameters
    loo_loss_weight: float = 1.0,
    loo_k_neighbors: int = 5,
    loo_max_corr_dist: float = 0.05,
    loo_normal_k: int = 20,
    loo_kdtree_rebuild_every: int = 10,
    knn_backend: str = "cpu_kdtree",
    # Anchoring parameters
    anchor_loss_weight: float = 0.1,
    anchor_n_samples: int = 4096,
    # TV parameters
    tv_reg: float = 1e-4,
    tv_voxel_size: float = 0.05,
    tv_every_k: int = 1,
    tv_sample_ratio: float | None = None,
    bbox_min: torch.Tensor | None = None,
    bbox_max: torch.Tensor | None = None,
    # LOO Color ICP-style photometric term (geometry-only, no color optimisation)
    loo_color_icp_weight: float = 0.0,
    loo_color_icp_k: int = 20,
    loo_color_icp_max_color_dist: float | None = None,
    # Surface-sharpening losses (reuse LOO KNN data)
    thin_shell_weight: float = 0.0,
    # Subsampling / memory controls
    loo_max_pairs_per_iter: int | None = None,
    loo_pairs_per_src: int = 1,
    deform_chunk_size: int = 200_000,
    # Optimization parameters
    lr: float = 1e-3,
    n_iters: int = 200,
    # Logging
    tb_writer=None,
    save_intermediate_dir: str | None = None,
    save_intermediate_every_n: int = 20,
    # Optional: hook for external progress bars/logging.
    # Called with progress events:
    #  - Stage events: it=0 and metrics contains {"stage": str, ...}
    #  - Iteration events: it>=1 and metrics contains per-iter metrics.
    progress_callback: Optional[Callable[[int, dict], None]] = None,
):
    """
    Global optimization: Leave-One-Out Consensus + Joint Deformation Optimization.

    Overview:
    - Uses Leave-One-Out (LOO) consensus loss: for frame i, compare D_i(P_i) against
      the model built from ALL OTHER frames (excluding frame i's own segment)
    - Both sides of correspondence losses have gradients (joint optimization)
    - Frame 0 is completely frozen (gauge fix)
    - Anchoring regularization prevents drift from initial deformations
    - Optional photometric consistency term (Colored-ICP-style; fixed observed colors)

    Returns:
        dict with keys:
            canonical_points: (N_total, 3) reconstructed canonical point cloud
            canonical_colors: (N_total, 3) colors (observed; fixed)
            per_frame_global_rigid: Updated global rigid transforms
            per_frame_local_deform: Updated local deformation modules
            model_frame_segments: List of (start, end) tuples for each frame in canonical
    """
    device = per_frame_world_points[0].device
    num_frames = len(per_frame_world_points)

    # Canonical model segments are fixed as long as per-frame point sets are fixed.
    model_frame_segments = _build_model_segments(per_frame_world_points)
    seg_starts = torch.tensor([s for s, _e in model_frame_segments], device=device, dtype=torch.long)
    seg_ends = torch.tensor([e for _s, e in model_frame_segments], device=device, dtype=torch.long)
    n_total_points = int(seg_ends[-1].item()) if len(seg_ends) > 0 else 0

    if progress_callback is not None:
        progress_callback(
            0,
            {
                "stage": "init",
                "num_frames": int(num_frames),
                "n_iters": int(n_iters),
                "loo_k": int(loo_k_neighbors),
                "loo_max_corr_dist": float(loo_max_corr_dist),
                "knn_backend": str(knn_backend),
            },
        )

    # Create intermediate save directory if needed
    if save_intermediate_dir is not None:
        os.makedirs(save_intermediate_dir, exist_ok=True)

    # =========================================================================
    # Setup optimization: Frame 0 is frozen (gauge fix), frames 1..N are optimized
    # =========================================================================

    # Build optimizer parameters (frames 1..N only)
    opt_params = []
    for i in range(1, num_frames):
        # Global rigid
        if not per_frame_global_rigid[i].requires_grad:
            per_frame_global_rigid[i] = per_frame_global_rigid[i].clone().detach().requires_grad_(True)
        opt_params.append(per_frame_global_rigid[i])

        # Local deform parameters
        if isinstance(per_frame_local_deform[i], torch.nn.Module):
            for p_idx, p in enumerate(per_frame_local_deform[i].parameters()):
                opt_params.append(p)

    if not opt_params:
        raise ValueError("No optimizable parameters found (need at least 2 frames)")

    # Build optimizer
    param_groups = [{"params": opt_params, "lr": lr}]

    optimizer = torch.optim.Adam(param_groups)

    # =========================================================================
    # Initialize anchoring: snapshot initial deformation outputs
    # =========================================================================
    if anchor_loss_weight > 0:
        anchoring_data = {}
        if progress_callback is not None:
            progress_callback(0, {"stage": "anchoring_init_start", "num_frames": int(num_frames - 1)})

        for i in range(1, num_frames):
            # Use per-frame bbox when available (needed for c2w
            # convention where each DeformationGrid has its own camera-space
            # bbox).  Falls back to the global bbox for the old convention.
            module_i = per_frame_local_deform[i]
            if isinstance(module_i, torch.nn.Module) and hasattr(module_i, "bbox_min"):
                frame_bbox_min = module_i.bbox_min
                frame_bbox_max = module_i.bbox_max
            else:
                frame_bbox_min = bbox_min
                frame_bbox_max = bbox_max
            anchor_pts_i = _sample_uniform_in_bbox(frame_bbox_min, frame_bbox_max, anchor_n_samples, device)
            with torch.no_grad():
                xi_local_init = module_i(anchor_pts_i).clone()
                xi_global_init = per_frame_global_rigid[i].clone()
            anchoring_data[i] = {
                "anchor_pts": anchor_pts_i,
                "xi_local": xi_local_init,
                "xi_global": xi_global_init,
            }
        if progress_callback is not None:
            progress_callback(
                0,
                {"stage": "anchoring_init_end", "num_frames": int(num_frames - 1), "n_samples": int(anchor_n_samples)},
            )

    # =========================================================================
    # Initialize LOO normals + KD-tree
    # =========================================================================
    _need_knn = loo_loss_weight > 0 or thin_shell_weight > 0 or loo_color_icp_weight > 0
    model_normals = None
    loo_tree = None
    model_color_grad = None
    model_intensity = None
    if _need_knn:
        with torch.no_grad():
            if progress_callback is not None:
                progress_callback(0, {"stage": "knn_init_start"})
            # Build an initial canonical model from all frames.
            per_frame_aligned_init = []
            for f in range(num_frames):
                aligned_f = _apply_deformation_chunked(
                    per_frame_world_points[f],
                    per_frame_local_deform[f],
                    per_frame_global_rigid[f],
                    chunk_size=deform_chunk_size,
                )
                per_frame_aligned_init.append(aligned_f)
            model_points_init = torch.cat(per_frame_aligned_init, dim=0)  # (N_total, 3)

            # Initial normals and KD-tree for LOO.
            model_normals, loo_tree = estimate_normals(model_points_init.detach(), k=loo_normal_k, backend=knn_backend)
            if progress_callback is not None:
                progress_callback(0, {"stage": "knn_init_end", "num_model": int(model_points_init.shape[0])})

            # ------------------------------------------------------------------
            # Precompute color gradients on the initial canonical model for
            # a Colored-ICP-style LOO photometric term (if enabled).
            # This uses a similar construction as non_rigid_icp / colored_icp_adam:
            #   (∑ u u^T + w n n^T) d_p = ∑ u ΔI
            # where u are tangent-plane offsets and ΔI neighbour intensity diffs.
            # ------------------------------------------------------------------
            if loo_color_icp_weight > 0:
                if progress_callback is not None:
                    progress_callback(0, {"stage": "loo_color_precompute_start", "k": int(loo_color_icp_k)})
                # Concatenate per-frame colors to match model_points_init layout.
                model_colors_init = torch.cat(per_frame_world_colors, dim=0)  # (N_total, 3)
                # Convert RGB to scalar intensity in [0,1].
                model_intensity = colors_to_intensity(model_colors_init).to(
                    device=device, dtype=model_points_init.dtype
                )

                K_neighbors = loo_color_icp_k + 1  # include self, then drop it
                nn_idxs_color, _ = query_knn_with_backend(
                    model_points_init.detach(),
                    model_points_init.detach(),
                    K=K_neighbors,
                    backend=knn_backend,
                    cpu_tree=loo_tree if knn_backend == "cpu_kdtree" else None,
                    gpu_tree=loo_tree if knn_backend == "gpu_kdtree" else None,
                )
                if nn_idxs_color.dim() == 1:
                    raise ValueError("loo_color_icp_k must be >= 1 to estimate color gradients.")
                # Drop self neighbour (assumed first)
                nn_idxs_color = nn_idxs_color[:, 1:]  # (N_total, loo_color_icp_k)

                # Compute color gradients in chunks to avoid allocating (N_total,k,3)
                # tensors all at once (which can OOM for large models).
                N_total = int(model_points_init.shape[0])
                model_color_grad = torch.empty((N_total, 3), device=device, dtype=model_points_init.dtype)

                w_ortho = float(loo_color_icp_k)
                eps = 1e-4
                I3 = torch.eye(3, device=device, dtype=model_points_init.dtype).view(1, 3, 3)
                chunk = 200_000

                for s in range(0, N_total, chunk):
                    e = min(s + chunk, N_total)
                    idxs = nn_idxs_color[s:e]  # (B,k)

                    neigh_pos = model_points_init[idxs]  # (B,k,3)
                    neigh_I = model_intensity[idxs]  # (B,k)

                    ref_pos = model_points_init[s:e]  # (B,3)
                    ref_norm = model_normals[s:e]  # (B,3)

                    ref_pos_exp = ref_pos.unsqueeze(1)  # (B,1,3)
                    ref_norm_exp = ref_norm.unsqueeze(1)  # (B,1,3)

                    delta = neigh_pos - ref_pos_exp  # (B,k,3)
                    dot = (delta * ref_norm_exp).sum(dim=2, keepdim=True)  # (B,k,1)
                    u = delta - dot * ref_norm_exp  # (B,k,3)

                    delta_I = (neigh_I - model_intensity[s:e].unsqueeze(1)).unsqueeze(2)  # (B,k,1)

                    U_t = u.transpose(1, 2)  # (B,3,k)
                    A = U_t @ u  # (B,3,3)
                    b = U_t @ delta_I  # (B,3,1)

                    n_outer = ref_norm.unsqueeze(2) * ref_norm.unsqueeze(1)  # (B,3,3)
                    A_reg = A + w_ortho * n_outer + eps * I3
                    model_color_grad[s:e] = torch.linalg.solve(A_reg, b).squeeze(2)  # (B,3)

                if progress_callback is not None:
                    progress_callback(0, {"stage": "loo_color_precompute_end"})

    # =========================================================================
    # Main optimization loop
    # =========================================================================

    for it in range(n_iters):
        optimizer.zero_grad()

        # -------------------------------------------------------------
        # Per-iteration timing (for performance profiling)
        # -------------------------------------------------------------
        iter_t_start = time.perf_counter()
        t_est_normals = 0.0
        t_knn = 0.0
        t_backward = 0.0

        # ---------------------------------------------------------------------
        # Build detached model for KNN/normals (correspondences are non-diff)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            per_frame_aligned_det = []
            for f in range(num_frames):
                aligned_f = _apply_deformation_chunked(
                    per_frame_world_points[f],
                    per_frame_local_deform[f],
                    per_frame_global_rigid[f],
                    chunk_size=deform_chunk_size,
                )
                per_frame_aligned_det.append(aligned_f)
            model_points_det = torch.cat(per_frame_aligned_det, dim=0)  # (N_total,3) detached

        # Track losses
        total_loo_loss = torch.tensor(0.0, device=device)
        total_thin_shell = torch.tensor(0.0, device=device)
        total_anchor_loss = torch.tensor(0.0, device=device)
        total_tv_loss = torch.tensor(0.0, device=device)

        loo_n_valid = 0

        # ---------------------------------------------------------------------
        # KNN query + LOO valid mask (shared by LOO consensus, normal
        # consistency, thin-shell, and color losses).
        # ---------------------------------------------------------------------
        nn_idx_all = None
        nn_d2_all = None
        loo_valid_all = None

        if _need_knn:
            t_knn_start = time.perf_counter()
            # Single global KNN over the entire model to avoid per-frame calls.
            nn_idx_all, nn_d2_all = query_knn_with_backend(
                model_points_det,
                model_points_det,
                K=loo_k_neighbors,
                backend=knn_backend,
                cpu_tree=loo_tree if knn_backend == "cpu_kdtree" else None,
                gpu_tree=loo_tree if knn_backend == "gpu_kdtree" else None,
            )
            t_knn += time.perf_counter() - t_knn_start

            # Build the global LOO + distance mask (N, K): True where the
            # neighbour is from a *different* frame and within max_corr_dist.
            nn_dists_sqrt_all = torch.sqrt(torch.clamp_min(nn_d2_all, 0.0))
            dist_mask_all = nn_dists_sqrt_all < loo_max_corr_dist

            loo_valid_all = torch.zeros_like(nn_idx_all, dtype=torch.bool)
            for f in range(num_frames):
                seg_start, seg_end = model_frame_segments[f]
                rows = slice(seg_start, seg_end)
                loo_mask_f = (nn_idx_all[rows] < seg_start) | (nn_idx_all[rows] >= seg_end)
                loo_valid_all[rows] = loo_mask_f & dist_mask_all[rows]

        # ---------------------------------------------------------------------
        # If requested, use stochastic pair subsampling for LOO/ThinShell/ColorICP
        # ---------------------------------------------------------------------
        use_pair_subsampling = (
            loo_max_pairs_per_iter is not None
            and int(loo_max_pairs_per_iter) > 0
            and loo_valid_all is not None
            and nn_idx_all is not None
            and (loo_loss_weight > 0 or thin_shell_weight > 0 or loo_color_icp_weight > 0)
        )

        # ---------------------------------------------------------------------
        # Point-to-Plane Loss (LOO consensus + thin-shell surface sharpening).
        # Both terms use the same LOO KNN structure:
        #   LOO:        ((p_i - p_j) . n_j)^2   -- point-to-plane at neighbor
        #   Thin-shell: ((p_j - p_i) . n_i)^2   -- point-to-plane at source
        # They are computed together in one pass for efficiency.
        # ---------------------------------------------------------------------
        if (loo_loss_weight > 0 or thin_shell_weight > 0) and loo_valid_all is not None:
            if use_pair_subsampling:
                src_idx_s, tgt_idx_s, w_s = _sample_loo_pairs_two_stage(
                    nn_idx_all,
                    loo_valid_all,
                    max_pairs=int(loo_max_pairs_per_iter),
                    pairs_per_src=int(loo_pairs_per_src),
                )
                if src_idx_s.numel() > 0:
                    src_pts = _gather_deformed_points_for_global_indices(
                        src_idx_s,
                        per_frame_world_points=per_frame_world_points,
                        per_frame_local_deform=per_frame_local_deform,
                        per_frame_global_rigid=per_frame_global_rigid,
                        seg_starts=seg_starts,
                        seg_ends=seg_ends,
                        deform_chunk_size=deform_chunk_size,
                    )
                    tgt_pts = _gather_deformed_points_for_global_indices(
                        tgt_idx_s,
                        per_frame_world_points=per_frame_world_points,
                        per_frame_local_deform=per_frame_local_deform,
                        per_frame_global_rigid=per_frame_global_rigid,
                        seg_starts=seg_starts,
                        seg_ends=seg_ends,
                        deform_chunk_size=deform_chunk_size,
                    )

                    tgt_normals = model_normals[tgt_idx_s]  # detached
                    src_normals = model_normals[src_idx_s]  # detached
                    displacement = src_pts - tgt_pts

                    # Importance-weighted estimator for mean over all valid pairs:
                    # p(src,tgt) ∝ 1/N * 1/m_src  ⇒ weight ∝ m_src
                    w = torch.clamp_min(w_s, 1.0)
                    w_sum = w.sum().clamp_min(1.0)

                    if loo_loss_weight > 0:
                        p2p_at_tgt = (displacement * tgt_normals).sum(dim=-1)
                        total_loo_loss = ((p2p_at_tgt * p2p_at_tgt) * w).sum() / w_sum
                    if thin_shell_weight > 0:
                        p2p_at_src = (displacement * src_normals).sum(dim=-1)
                        total_thin_shell = ((p2p_at_src * p2p_at_src) * w).sum() / w_sum

                    loo_n_valid = int(src_idx_s.numel())
            else:
                # Full (no subsampling): build full model with gradients.
                per_frame_aligned = []
                for f in range(num_frames):
                    per_frame_aligned.append(
                        _apply_deformation_chunked(
                            per_frame_world_points[f],
                            per_frame_local_deform[f],
                            per_frame_global_rigid[f],
                            chunk_size=deform_chunk_size,
                        )
                    )
                model_points = torch.cat(per_frame_aligned, dim=0)  # (N_total, 3), grad

                for f in range(num_frames):
                    xprime_f = per_frame_aligned[f]  # (M_f, 3), grad → deform_f
                    seg_start, seg_end = model_frame_segments[f]

                    if seg_end - seg_start == 0:
                        continue

                    valid = loo_valid_all[seg_start:seg_end]  # (M_f, K)
                    if valid.sum() == 0:
                        continue

                    nn_idx = nn_idx_all[seg_start:seg_end]
                    valid_src_rows, _valid_cols = torch.where(valid)
                    tgt_idx = nn_idx[valid]  # indices into model_points

                    src_pts = xprime_f[valid_src_rows]  # (N_valid, 3), grad → deform_f
                    tgt_pts = model_points[tgt_idx]  # (N_valid, 3), grad → deform_j
                    tgt_normals = model_normals[tgt_idx]  # (N_valid, 3), detached
                    src_normals = model_normals[seg_start + valid_src_rows]  # (N_valid, 3), detached

                    displacement = src_pts - tgt_pts  # (N_valid, 3)

                    if loo_loss_weight > 0:
                        p2p_at_tgt = (displacement * tgt_normals).sum(dim=-1)
                        total_loo_loss = total_loo_loss + (p2p_at_tgt**2).sum()

                    if thin_shell_weight > 0:
                        p2p_at_src = (displacement * src_normals).sum(dim=-1)
                        total_thin_shell = total_thin_shell + (p2p_at_src**2).sum()

                    loo_n_valid += tgt_idx.numel()

                if loo_n_valid > 0:
                    if loo_loss_weight > 0:
                        total_loo_loss = total_loo_loss / loo_n_valid
                    if thin_shell_weight > 0:
                        total_thin_shell = total_thin_shell / loo_n_valid

        # ---------------------------------------------------------------------
        # Anchoring Regularization: Penalize deviation from initial deformations
        # ---------------------------------------------------------------------
        if anchor_loss_weight > 0:
            for i in range(1, num_frames):
                # Local deform anchoring (per-frame anchor points)
                anchor_pts_i = anchoring_data[i]["anchor_pts"]
                xi_local_current = per_frame_local_deform[i](anchor_pts_i)
                xi_local_init = anchoring_data[i]["xi_local"]  # detached snapshot
                total_anchor_loss = total_anchor_loss + ((xi_local_current - xi_local_init) ** 2).mean()

                # Global rigid anchoring
                xi_global_current = per_frame_global_rigid[i]
                xi_global_init = anchoring_data[i]["xi_global"]  # detached snapshot
                total_anchor_loss = total_anchor_loss + ((xi_global_current - xi_global_init) ** 2).mean()

            # Normalize by number of frames
            total_anchor_loss = total_anchor_loss / (num_frames - 1)

        # ---------------------------------------------------------------------
        # TV Regularization
        # ---------------------------------------------------------------------
        if tv_reg > 0 and (tv_every_k <= 1 or (it % tv_every_k) == 0):
            local_modules = [
                per_frame_local_deform[i]
                for i in range(1, num_frames)
                if isinstance(per_frame_local_deform[i], torch.nn.Module)
            ]
            if local_modules:
                # Subsample to at most 5 modules for efficiency
                num_to_use = min(5, len(local_modules))
                if num_to_use < len(local_modules):
                    perm = torch.randperm(len(local_modules), device=device)[:num_to_use]
                    selected_modules = [local_modules[int(i)] for i in perm]
                else:
                    selected_modules = local_modules

                tv_energies = []
                for module in selected_modules:
                    # Use per-module bbox when available (needed for
                    # c2w convention where each frame's
                    # DeformationGrid has its own camera-space bbox).
                    _tv_bmin = module.bbox_min if hasattr(module, "bbox_min") else bbox_min
                    _tv_bmax = module.bbox_max if hasattr(module, "bbox_max") else bbox_max
                    energy = tv_loss(
                        _tv_bmin,
                        _tv_bmax,
                        tv_voxel_size,
                        module,
                        sample_ratio=tv_sample_ratio,
                    )
                    tv_energies.append(energy)

                if tv_energies:
                    total_tv_loss = torch.stack(tv_energies).mean()

        # ---------------------------------------------------------------------
        # LOO Colored-ICP-style photometric term (geometry-only)
        # ---------------------------------------------------------------------
        total_loo_color_icp = torch.tensor(0.0, device=device)
        if (
            loo_color_icp_weight > 0
            and model_color_grad is not None
            and model_intensity is not None
            and loo_valid_all is not None
        ):
            if use_pair_subsampling:
                # Reuse the same subsampled pairs as geometry when possible.
                # (If geometry weights are 0, sample again here.)
                if "src_idx_s" in locals() and "tgt_idx_s" in locals() and src_idx_s.numel() > 0:
                    src_idx_c = src_idx_s
                    tgt_idx_c = tgt_idx_s
                    w_c = w_s
                    # src_pts/tgt_pts already computed above if geometry term ran; recompute lazily otherwise.
                    if "src_pts" not in locals() or "tgt_pts" not in locals():
                        src_pts = _gather_deformed_points_for_global_indices(
                            src_idx_c,
                            per_frame_world_points=per_frame_world_points,
                            per_frame_local_deform=per_frame_local_deform,
                            per_frame_global_rigid=per_frame_global_rigid,
                            seg_starts=seg_starts,
                            seg_ends=seg_ends,
                            deform_chunk_size=deform_chunk_size,
                        )
                        tgt_pts = _gather_deformed_points_for_global_indices(
                            tgt_idx_c,
                            per_frame_world_points=per_frame_world_points,
                            per_frame_local_deform=per_frame_local_deform,
                            per_frame_global_rigid=per_frame_global_rigid,
                            seg_starts=seg_starts,
                            seg_ends=seg_ends,
                            deform_chunk_size=deform_chunk_size,
                        )
                else:
                    src_idx_c, tgt_idx_c, w_c = _sample_loo_pairs_two_stage(
                        nn_idx_all,
                        loo_valid_all,
                        max_pairs=int(loo_max_pairs_per_iter),
                        pairs_per_src=int(loo_pairs_per_src),
                    )
                    if src_idx_c.numel() > 0:
                        src_pts = _gather_deformed_points_for_global_indices(
                            src_idx_c,
                            per_frame_world_points=per_frame_world_points,
                            per_frame_local_deform=per_frame_local_deform,
                            per_frame_global_rigid=per_frame_global_rigid,
                            seg_starts=seg_starts,
                            seg_ends=seg_ends,
                            deform_chunk_size=deform_chunk_size,
                        )
                        tgt_pts = _gather_deformed_points_for_global_indices(
                            tgt_idx_c,
                            per_frame_world_points=per_frame_world_points,
                            per_frame_local_deform=per_frame_local_deform,
                            per_frame_global_rigid=per_frame_global_rigid,
                            seg_starts=seg_starts,
                            seg_ends=seg_ends,
                            deform_chunk_size=deform_chunk_size,
                        )

                if src_idx_c.numel() > 0:
                    n_p = model_normals[tgt_idx_c]  # (S,3), detached
                    d_p = model_color_grad[tgt_idx_c]  # (S,3), detached

                    diff_qp = src_pts - tgt_pts  # (S,3)
                    dot_qp = (diff_qp * n_p).sum(dim=1, keepdim=True)
                    u_q = diff_qp - dot_qp * n_p  # (S,3)

                    I_p = model_intensity[tgt_idx_c]  # (S,)
                    I_q = model_intensity[src_idx_c]  # (S,)

                    valid = torch.ones_like(I_p, dtype=torch.bool, device=device)
                    if loo_color_icp_max_color_dist is not None:
                        valid = (I_p - I_q).abs() <= loo_color_icp_max_color_dist

                    if valid.any():
                        C_hat = I_p + (d_p * u_q).sum(dim=1)
                        r_C = C_hat - I_q
                        r2 = r_C[valid] * r_C[valid]
                        w = torch.clamp_min(w_c[valid].to(torch.float32), 1.0)
                        total_loo_color_icp = (r2 * w).sum() / w.sum().clamp_min(1.0)
            else:
                # Full (no subsampling) colored ICP term.
                # Requires `model_points` with gradients; build if not already built above.
                if "model_points" not in locals():
                    per_frame_aligned = []
                    for f in range(num_frames):
                        per_frame_aligned.append(
                            _apply_deformation_chunked(
                                per_frame_world_points[f],
                                per_frame_local_deform[f],
                                per_frame_global_rigid[f],
                                chunk_size=deform_chunk_size,
                            )
                        )
                    model_points = torch.cat(per_frame_aligned, dim=0)

                N_total, K = nn_idx_all.shape
                src_idx = torch.arange(N_total, device=device).unsqueeze(1).expand(N_total, K)
                tgt_idx = nn_idx_all  # (N_total,K)

                p = model_points[tgt_idx]  # (N_total,K,3) targets
                q = model_points[src_idx]  # (N_total,K,3) sources
                n_p = model_normals[tgt_idx]  # (N_total,K,3)
                d_p = model_color_grad[tgt_idx]  # (N_total,K,3)

                diff_qp = q - p
                dot_qp = (diff_qp * n_p).sum(dim=2, keepdim=True)
                u_q = diff_qp - dot_qp * n_p  # (N_total,K,3)

                I_p = model_intensity[tgt_idx]  # (N_total,K)
                I_q = model_intensity[src_idx]  # (N_total,K)

                valid = loo_valid_all.clone()
                if loo_color_icp_max_color_dist is not None:
                    color_diff = (I_p - I_q).abs()
                    color_mask = color_diff <= loo_color_icp_max_color_dist
                    valid = valid & color_mask

                if valid.any():
                    C_hat = I_p + (d_p * u_q).sum(dim=2)  # (N_total,K)
                    r_C = C_hat - I_q
                    r_C_valid = r_C[valid]
                    total_loo_color_icp = (r_C_valid * r_C_valid).mean()

        # ---------------------------------------------------------------------
        # Total loss and optimization step (single backward)
        # ---------------------------------------------------------------------
        total_loss = (
            loo_loss_weight * total_loo_loss
            + anchor_loss_weight * total_anchor_loss
            + tv_reg * total_tv_loss
            + thin_shell_weight * total_thin_shell
            + loo_color_icp_weight * total_loo_color_icp
        )

        if total_loss.requires_grad:
            t_back_start = time.perf_counter()
            total_loss.backward()

            all_params_for_clip = list(opt_params)
            for p in all_params_for_clip:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

            total_grad_norm = torch.nn.utils.clip_grad_norm_(all_params_for_clip, 1.0)

            optimizer.step()
            t_backward += time.perf_counter() - t_back_start
        else:
            total_grad_norm = torch.tensor(0.0)

        # ---------------------------------------------------------------------
        # Periodically rebuild LOO normals + KD-tree
        # ---------------------------------------------------------------------
        with torch.no_grad():
            if it % loo_kdtree_rebuild_every == 0 and _need_knn:
                if progress_callback is not None:
                    progress_callback(0, {"stage": "knn_rebuild_start", "iter": int(it + 1)})
                current_model = model_points_det
                # Rebuild normals and KD-tree together; estimate_normals returns both.
                t0 = time.perf_counter()
                model_normals, loo_tree = estimate_normals(current_model, k=loo_normal_k, backend=knn_backend)
                t_est_normals += time.perf_counter() - t0
                if progress_callback is not None:
                    progress_callback(
                        0,
                        {
                            "stage": "knn_rebuild_end",
                            "iter": int(it + 1),
                            "time_est_normals_s": float(t_est_normals),
                        },
                    )

        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------
        if tb_writer is not None:
            tb_writer.add_scalar("global_opt/loss_total", float(total_loss.detach()), it)
            tb_writer.add_scalar("global_opt/loss_loo", float(total_loo_loss.detach()), it)
            tb_writer.add_scalar("global_opt/loss_anchor", float(total_anchor_loss.detach()), it)
            tb_writer.add_scalar("global_opt/loss_tv", float(total_tv_loss.detach()), it)
            tb_writer.add_scalar("global_opt/loss_thin_shell", float(total_thin_shell.detach()), it)
            tb_writer.add_scalar(
                "global_opt/loss_loo_color_icp",
                float(total_loo_color_icp.detach()),
                it,
            )
            tb_writer.add_scalar("global_opt/loo_n_valid", float(loo_n_valid), it)
            tb_writer.add_scalar("global_opt/loo_color_icp_n_valid", float(loo_n_valid), it)
            tb_writer.add_scalar("global_opt/grad_norm", float(total_grad_norm), it)
            # Timing to TensorBoard (seconds)
            iter_elapsed = time.perf_counter() - iter_t_start
            tb_writer.add_scalar("global_opt/time_iter", float(iter_elapsed), it)
            tb_writer.add_scalar("global_opt/time_est_normals", float(t_est_normals), it)
            tb_writer.add_scalar("global_opt/time_knn", float(t_knn), it)
            tb_writer.add_scalar("global_opt/time_backward", float(t_backward), it)

        if progress_callback is not None:
            progress_callback(
                it + 1,
                {
                    "loss_total": float(total_loss.detach()),
                    "loss_loo": float(total_loo_loss.detach()),
                    "loss_anchor": float(total_anchor_loss.detach()),
                    "loss_tv": float(total_tv_loss.detach()),
                    "loss_thin_shell": float(total_thin_shell.detach()),
                    "loss_loo_color_icp": float(total_loo_color_icp.detach()),
                    "loo_n_valid": int(loo_n_valid),
                    "grad_norm": float(total_grad_norm.detach()) if isinstance(total_grad_norm, torch.Tensor) else float(total_grad_norm),
                    "time_iter_s": float(time.perf_counter() - iter_t_start),
                    "time_normals_s": float(t_est_normals),
                    "time_knn_s": float(t_knn),
                    "time_back_s": float(t_backward),
                },
            )

        # ---------------------------------------------------------------------
        # Save intermediate results
        # ---------------------------------------------------------------------
        if save_intermediate_dir is not None and (it + 1) % save_intermediate_every_n == 0:
            intermediate_path = os.path.join(save_intermediate_dir, f"iter_{it + 1:05d}")
            os.makedirs(intermediate_path, exist_ok=True)

            # Reconstruct and save canonical
            with torch.no_grad():
                canon_pts = torch.cat(
                    [
                        _apply_deformation(
                            per_frame_world_points[f],
                            per_frame_local_deform[f],
                            per_frame_global_rigid[f],
                        )
                        for f in range(num_frames)
                    ],
                    dim=0,
                )
                canon_colors = (
                    torch.cat(per_frame_world_colors, dim=0) if per_frame_world_colors[0] is not None else None
                )

            pcl = torch_to_o3d_pcd(canon_pts, canon_colors)
            o3d.io.write_point_cloud(os.path.join(intermediate_path, "aligned_points.ply"), pcl)

            # Save deformations
            for i in range(num_frames):
                torch.save(
                    per_frame_global_rigid[i].detach(),
                    os.path.join(intermediate_path, f"per_frame_global_rigid_{i:05d}.pt"),
                )
            for i in range(1, num_frames):
                if isinstance(per_frame_local_deform[i], torch.nn.Module):
                    torch.save(
                        per_frame_local_deform[i].state_dict(),
                        os.path.join(intermediate_path, f"per_frame_local_deform_{i:05d}.pt"),
                    )

            if progress_callback is not None:
                progress_callback(0, {"stage": "saved_intermediate", "path": str(intermediate_path), "iter": int(it + 1)})

    # =========================================================================
    # Final output: reconstruct canonical from optimized deformations
    # =========================================================================
    with torch.no_grad():
        final_aligned = []
        final_segments = []
        start_idx = 0

        for f in range(num_frames):
            aligned_f = _apply_deformation(
                per_frame_world_points[f],
                per_frame_local_deform[f],
                per_frame_global_rigid[f],
            )
            final_aligned.append(aligned_f)
            end_idx = start_idx + aligned_f.shape[0]
            final_segments.append((start_idx, end_idx))
            start_idx = end_idx

        canonical_points = torch.cat(final_aligned, dim=0)
        canonical_colors = torch.cat(per_frame_world_colors, dim=0) if per_frame_world_colors[0] is not None else None

    if progress_callback is not None:
        progress_callback(
            0,
            {"stage": "finished", "num_frames": int(num_frames), "num_canonical_points": int(canonical_points.shape[0])},
        )

    result = {
        "canonical_points": canonical_points,
        "canonical_colors": canonical_colors,
        "per_frame_global_rigid": per_frame_global_rigid,
        "per_frame_local_deform": per_frame_local_deform,
        "model_frame_segments": final_segments,
    }

    return result
