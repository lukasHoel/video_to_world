import time

import torch
from typing import Optional
from typing import Callable

from models.deformation import DeformationGrid
from utils.image import colors_to_intensity
from utils.geometry import se3_apply, se3_exp
from losses.tv import tv_loss
from losses.correspondence import compute_correspondence_loss_with_model_segments
from utils.knn import (
    build_kdtree,
    build_torch_kdtree,
    query_knn_with_backend,
)
from utils.normals import estimate_normals


# ---------------------------
# Full autodiff non-rigid ICP loop
# ---------------------------
def non_rigid_icp(
    src,
    ref,
    # RoMa matching parameters - using model segments directly
    # Note: ref is the model, so we index into it using roma_model_frame_segments
    roma_matches_data=None,
    roma_model_frame_segments=None,  # List of (start_idx, end_idx) tuples for each frame in ref (model)
    roma_ref_frame_indices=None,  # Frame indices corresponding to roma_model_frame_segments
    roma_src_valid_pixel_indices=None,  # (M_src,) valid pixel indices for source frame
    roma_model_valid_pixel_indices_list=None,  # List of valid pixel indices for ref frames (filtered)
    roma_loss_weight=0.0,
    roma_max_corr_dist=None,
    image_height=None,
    image_width=None,
    n_iter=30,
    lr=0.05,
    knn_backend: str = "cpu_kdtree",
    method="point2point",
    ref_normals=None,
    normal_k=20,
    max_corr_dist=None,
    local_twist_reg=1e-4,
    tv_reg=1e-4,
    chunk=50000,
    bbox_min=None,
    bbox_max=None,
    tb_writer=None,
    tb_prefix: str = "",
    metrics_out: Optional[dict] = None,
    tv_voxel_size: float = 0.05,
    tv_every_k: int = 1,
    tv_sample_ratio: Optional[float] = None,
    ref_kd_tree=None,
    # Early stopping: stop when both data loss and roma loss (if used) stop improving
    early_stopping_patience: Optional[int] = 5,
    early_stopping_min_iters: int = 5,
    early_stopping_min_delta: float = 1e-5,
    xi_global_init: Optional[torch.Tensor] = None,
    # DeformationGrid capacity parameters
    deform_log2_hashmap_size: int = 19,
    deform_num_levels: int = 16,
    deform_n_neurons: int = 64,
    deform_n_hidden_layers: int = 2,
    deform_min_res: int = 16,
    deform_max_res: int = 1024,
    # Colored ICP-style additional photometric term (optional).
    # When > 0 and colors are provided, we add a color residual term similar
    # in spirit to Park et al. / Open3D's ColoredICP.
    color_icp_weight: float = 0.0,
    src_colors: Optional[torch.Tensor] = None,
    ref_colors: Optional[torch.Tensor] = None,
    color_icp_k: int = 20,
    color_icp_max_color_dist: Optional[float] = None,
    # Per-point loss output (for downstream filtering / diagnostics).
    # When True AND metrics_out is not None, appends per-point loss tensors
    # to metrics_out after the optimisation loop (one final NN query).
    compute_per_point_losses: bool = False,
    # Optional: hook for external progress bars/logging.
    # Called with progress events:
    #  - Stage events: it=0 and metrics contains {"stage": str, ...}
    #  - Iteration events: it>=1 and metrics contains per-iter losses.
    progress_callback: Optional[Callable[[int, dict], None]] = None,
):
    """
    src, ref: (1, H, W, 3) and (N, H, W, 3) torch tensors (float32)
    Note: this ICP optimizes geometry only (no color optimization / photometric loss).
    roma_matches_data: optional list of RoMaMatchData for RoMa-based matching
    roma_model_frame_segments: list of (start_idx, end_idx) tuples for each frame in ref (model)
    roma_ref_frame_indices: frame indices corresponding to model_frame_segments
    roma_src_valid_pixel_indices: (M_src,) valid pixel indices for source frame
    roma_model_valid_pixel_indices_list: list of valid pixel indices for ref frames (filtered after merge)
    roma_loss_weight: weight for RoMa matching geometry loss
    roma_max_corr_dist: maximum correspondence distance for RoMa matches
    image_height, image_width: image dimensions for pixel-to-3D mapping (required for RoMa)
    TV controls:
      tv_voxel_size: voxel size for TV neighbor definition (used when sampling at input points).
      tv_every_k: compute TV loss every k iters (1 = every iter).
      tv_sample_ratio: if in (0,1), randomly subsample this fraction of input points for TV loss.
    Early stopping (data + roma; stop only when both have stopped improving):
      early_stopping_patience: stop after this many consecutive iters with no improvement
        in both data loss and roma loss (if roma is used). None = disabled (run all n_iter).
      early_stopping_min_iters: minimum iterations before early stopping can trigger.
      early_stopping_min_delta: loss must decrease by more than this to count as improved.
      xi_global_init: optional (6,) SE3 twist to initialise the global rigid transform.
        When None (default), xi_global starts at zeros (identity).

    Returns:
      src_final (1, H, W, 3), xi_global (6,) tensor (cpu), deform (module)
    """

    device = src.device

    src_flat = src.view(-1, 3)
    ref_flat = ref.view(-1, 3)
    if ref_normals is not None:
        ref_normals = ref_normals.view(-1, 3)

    # Compute bounding box from source and reference point clouds if not provided
    if bbox_min is None or bbox_max is None:
        all_pts = torch.cat([src_flat, ref_flat], dim=0)
        bbox_min = all_pts.min(dim=0)[0]
        bbox_max = all_pts.max(dim=0)[0]

        # Add small padding to avoid edge cases
        padding = (bbox_max - bbox_min) * 0.01
        bbox_min = bbox_min - padding
        bbox_max = bbox_max + padding
    else:
        # Ensure bbox is on the correct device and dtype
        bbox_min = bbox_min.to(device).to(src.dtype)
        bbox_max = bbox_max.to(device).to(src.dtype)

    # global se(3) parameter (omega (3), v (3))
    if xi_global_init is not None:
        xi_global = torch.nn.Parameter(xi_global_init.clone().to(device=device, dtype=src.dtype))
    else:
        xi_global = torch.nn.Parameter(torch.zeros(6, device=device, dtype=src.dtype))

    # deformation grid
    deform = DeformationGrid(
        bbox_min,
        bbox_max,
        min_res=deform_min_res,
        max_res=deform_max_res,
        num_levels=deform_num_levels,
        log2_hashmap_size=deform_log2_hashmap_size,
        n_neurons=deform_n_neurons,
        n_hidden_layers=deform_n_hidden_layers,
    ).to(device)

    # optimizer
    opt_params = [xi_global]
    if any(p.requires_grad for p in deform.parameters()):
        opt_params += list(deform.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    # Pre-build KDTree once and reuse (CPU or GPU depending on backend).
    # If a pre-built tree is provided, reuse it (saves time when the same model is used).
    tree = None
    kdtree_build_ms: float = 0.0
    if ref_kd_tree is not None:
        tree = ref_kd_tree
    elif knn_backend in ("cpu_kdtree", "gpu_kdtree"):
        if progress_callback is not None:
            progress_callback(0, {"stage": "kdtree_build_start", "backend": knn_backend, "num_ref": int(ref_flat.shape[0])})
        t_build_start = time.perf_counter()
        if knn_backend == "cpu_kdtree":
            tree = build_kdtree(ref_flat)
        elif knn_backend == "gpu_kdtree":
            tree = build_torch_kdtree(ref_flat)

        # Make sure all device work tied to the build is finished.
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        t_build_end = time.perf_counter()
        kdtree_build_ms = (t_build_end - t_build_start) * 1000.0
        if progress_callback is not None:
            progress_callback(
                0,
                {
                    "stage": "kdtree_build_end",
                    "backend": knn_backend,
                    "num_ref": int(ref_flat.shape[0]),
                    "kdtree_build_ms": float(kdtree_build_ms),
                },
            )

    # compute normals if using point-to-plane
    if method == "point2plane" and ref_normals is None:
        if progress_callback is not None:
            progress_callback(0, {"stage": "estimate_normals_start", "k": int(normal_k), "num_ref": int(ref_flat.shape[0])})
        ref_normals, _ = estimate_normals(ref_flat, k=normal_k, backend=knn_backend)
        if progress_callback is not None:
            progress_callback(0, {"stage": "estimate_normals_end"})

    # Precompute color gradients on the (fixed) reference, if requested.
    use_color_icp = (
        color_icp_weight is not None and color_icp_weight > 0.0 and ref_colors is not None and src_colors is not None
    )
    ref_color_grad = None
    ref_intensity = None
    src_intensity = None
    if use_color_icp:
        if progress_callback is not None:
            progress_callback(
                0,
                {
                    "stage": "color_icp_precompute_start",
                    "k": int(color_icp_k),
                    "num_ref": int(ref_flat.shape[0]),
                    "max_color_dist": float(color_icp_max_color_dist) if color_icp_max_color_dist is not None else None,
                },
            )
        # Flatten colors and convert to intensities.
        src_intensity = colors_to_intensity(src_colors).to(device=device, dtype=src.dtype)
        ref_intensity = colors_to_intensity(ref_colors).to(device=device, dtype=src.dtype)

        # Use same KNN backend/tree as geometry; reuse KDTree when possible.
        K_neighbors = color_icp_k + 1  # include self, then drop it
        cpu_tree_for_color = None
        gpu_tree_for_color = None
        if knn_backend == "cpu_kdtree":
            cpu_tree_for_color = ref_kd_tree if ref_kd_tree is not None else build_kdtree(ref_flat)
        elif knn_backend == "gpu_kdtree":
            gpu_tree_for_color = ref_kd_tree if ref_kd_tree is not None else build_torch_kdtree(ref_flat)

        nn_idxs, _ = query_knn_with_backend(
            ref_flat,
            ref_flat,
            K=K_neighbors,
            backend=knn_backend,
            chunk=50_000,
            cpu_tree=cpu_tree_for_color,
            gpu_tree=gpu_tree_for_color,
        )
        if nn_idxs.dim() == 1:
            raise ValueError("color_icp_k must be >= 1 to estimate color gradients.")
        nn_idxs = nn_idxs[:, 1:]  # drop self

        neigh_pos = ref_flat[nn_idxs]  # (Nr,k,3)
        neigh_I = ref_intensity[nn_idxs]  # (Nr,k)

        ref_pos_exp = ref_flat.unsqueeze(1)  # (Nr,1,3)
        ref_norm_exp = ref_normals.unsqueeze(1)  # (Nr,1,3)
        delta = neigh_pos - ref_pos_exp  # (Nr,k,3)
        dot = (delta * ref_norm_exp).sum(dim=2, keepdim=True)  # (Nr,k,1)
        u = delta - dot * ref_norm_exp  # (Nr,k,3)

        delta_I = (neigh_I - ref_intensity.unsqueeze(1)).unsqueeze(2)  # (Nr,k,1)

        U_t = u.transpose(1, 2)  # (Nr,3,k)
        A = U_t @ u  # (Nr,3,3)
        b = U_t @ delta_I  # (Nr,3,1)

        w_ortho = float(color_icp_k)
        n = ref_normals  # (Nr,3)
        n_outer = n.unsqueeze(2) * n.unsqueeze(1)  # (Nr,3,3)

        eps = 1e-4
        I3 = torch.eye(3, device=device, dtype=src.dtype).view(1, 3, 3)
        A_reg = A + w_ortho * n_outer + eps * I3
        ref_color_grad = torch.linalg.solve(A_reg, b).squeeze(2)  # (Nr,3)
        if progress_callback is not None:
            progress_callback(0, {"stage": "color_icp_precompute_end"})

    last_metrics = None
    iters_completed = 0

    # Early stopping: stop only when BOTH data loss AND roma loss (if used) have stopped improving
    use_early_stopping = early_stopping_patience is not None and early_stopping_patience > 0
    use_roma_for_early_stop = (
        roma_matches_data is not None
        and roma_model_frame_segments is not None
        and roma_src_valid_pixel_indices is not None
        and roma_model_valid_pixel_indices_list is not None
        and roma_loss_weight > 0
        and image_height is not None
        and image_width is not None
    )
    use_color_icp_for_early_stop = use_color_icp
    best_data_loss = float("inf")
    best_roma_loss = float("inf")
    best_color_icp_loss = float("inf")
    no_improve_count = 0

    # Per-iteration timing accumulators (for TensorBoard mean stats)
    timing_knn_ms = []
    timing_geom_ms = []
    timing_roma_ms = []
    timing_opt_ms = []
    timing_total_ms = []

    for it in range(n_iter):
        # ------------------------------------------------------------------
        # Timing helpers (for tb / metrics / callback)
        # ------------------------------------------------------------------
        def _sync_if_cuda():
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)

        do_timing = tb_writer is not None or metrics_out is not None
        if do_timing:
            _sync_if_cuda()
            t_iter_start = time.perf_counter()
            t_after_knn = None
            t_after_geom_loss = None
            t_after_roma_loss = None
            t_after_optim = None
        Rg, tg = se3_exp(xi_global.unsqueeze(0))
        Rg = Rg[0]
        tg = tg[0]

        xi_local = deform(src_flat)  # (N,6)
        p_local = se3_apply(xi_local, src_flat)  # (N,3)
        src_transformed = (p_local @ Rg.t()) + tg.view(1, 3)

        # Find correspondences using unified helper.
        idxs, d2 = query_knn_with_backend(
            src_transformed,
            ref_flat,
            K=1,
            backend=knn_backend,
            chunk=chunk,
            cpu_tree=tree if knn_backend == "cpu_kdtree" else None,
            gpu_tree=tree if knn_backend == "gpu_kdtree" else None,
        )

        if do_timing:
            _sync_if_cuda()
            t_after_knn = time.perf_counter()

        perc_used = 100.0
        if max_corr_dist is not None:
            thresh2 = max_corr_dist * max_corr_dist
            mask = d2 < thresh2
            if mask.sum() < 3:
                break
            src_used = src_transformed[mask]
            tgt_used = ref_flat[idxs[mask]]
            d2_used = d2[mask]
            perc_used = 100.0 * (mask.sum().item() / mask.shape[0])
            if method == "point2plane":
                normals_used = ref_normals[idxs[mask]]
        else:
            src_used = src_transformed
            tgt_used = ref_flat[idxs]
            d2_used = d2
            if method == "point2plane":
                normals_used = ref_normals[idxs]

        if method == "point2point":
            diff = src_used - tgt_used
            loss_data = (diff * diff).sum(dim=1).mean()
        elif method == "point2plane":
            rel = src_used - tgt_used
            proj = (rel * normals_used).sum(dim=1)
            loss_data = (proj * proj).mean()
        else:
            raise ValueError("Unknown method")

        if local_twist_reg > 0:
            loss_reg = local_twist_reg * (xi_local.norm(dim=1).mean())
        else:
            loss_reg = torch.tensor(0.0, device=device)

        if tv_reg > 0 and (tv_every_k is None or tv_every_k <= 1 or (it % tv_every_k) == 0):
            # Sample TV loss at input points and their neighbors (memory-friendly)
            loss_tv = tv_reg * tv_loss(
                bbox_min,
                bbox_max,
                tv_voxel_size,
                deform,
                sample_ratio=tv_sample_ratio,  # Randomly subsample input points if provided
                input_points=src_flat,  # Sample at input point locations
                num_jittered_points=4,  # Can be increased for more regularization
                jitter_scale=None,  # Defaults to voxel_size * 0.5
            )
        else:
            loss_tv = torch.tensor(0.0, device=device)

        if do_timing:
            _sync_if_cuda()
            t_after_geom_loss = time.perf_counter()

        # RoMa matching loss
        loss_roma = torch.tensor(0.0, device=device)
        num_roma_matches = 0
        if (
            roma_matches_data is not None
            and roma_model_frame_segments is not None
            and roma_src_valid_pixel_indices is not None
            and roma_model_valid_pixel_indices_list is not None
            and roma_loss_weight > 0
            and image_height is not None
            and image_width is not None
        ):
            # Compute RoMa matching loss using model segments directly
            # ref_flat IS the model, so we index into it using roma_model_frame_segments
            loss_roma, _loss_roma_color, num_roma_matches = compute_correspondence_loss_with_model_segments(
                matches_data=roma_matches_data,
                src_points_transformed=src_transformed.reshape(-1, 3),
                model_points=ref_flat,  # ref is the model
                model_colors=None,  # ICP does not use RoMa color loss
                model_frame_segments=roma_model_frame_segments,
                ref_frame_indices=roma_ref_frame_indices,
                src_valid_pixel_indices=roma_src_valid_pixel_indices,
                model_valid_pixel_indices_list=roma_model_valid_pixel_indices_list,
                H=image_height,
                W=image_width,
                src_colors=None,  # ICP does not use RoMa color loss
                color_loss_weight=0.0,
                max_corr_dist=roma_max_corr_dist,
            )
            loss_roma = roma_loss_weight * loss_roma

        if do_timing:
            _sync_if_cuda()
            t_after_roma_loss = time.perf_counter()

        # Colored ICP-style color loss (photometric term)
        loss_color_icp = torch.tensor(0.0, device=device)
        if use_color_icp:
            # We apply a color residual similar to Park et al., but as an extra
            # term so it does not disrupt the existing geometric weighting.
            # For each correspondence (p = tgt_used, q = src_used):
            #   u = (q - p) - ((q - p)·n_p) n_p
            #   C_p(q') ≈ I_p + d_p · u
            #   r_C = C_p(q') - I_q
            # and we minimize mean(r_C^2).
            # Optionally filter by max_color_dist (analogous to max_corr_dist):
            # exclude correspondences where |I_p - I_q| > color_icp_max_color_dist.
            tgt_idx_used = idxs[mask] if max_corr_dist is not None else idxs
            p = tgt_used  # (K,3)
            q = src_used  # (K,3)
            n_used = normals_used  # (K,3)
            d_p_used = ref_color_grad[tgt_idx_used]  # (K,3)

            diff_qp = q - p  # (K,3)
            dot_qp = (diff_qp * n_used).sum(dim=1, keepdim=True)  # (K,1)
            u_q = diff_qp - dot_qp * n_used  # (K,3)

            I_p = ref_intensity[tgt_idx_used]  # (K,)
            I_q = src_intensity[mask] if max_corr_dist is not None else src_intensity

            if color_icp_max_color_dist is not None:
                color_diff = (I_p - I_q).abs()
                color_mask = color_diff <= color_icp_max_color_dist
                if color_mask.sum() >= 3:
                    I_p = I_p[color_mask]
                    I_q = I_q[color_mask]
                    d_p_used = d_p_used[color_mask]
                    u_q = u_q[color_mask]
                # else: keep all correspondences (too few after filtering)

            C_hat = I_p + (d_p_used * u_q).sum(dim=1)
            r_C = C_hat - I_q

            loss_color_icp = color_icp_weight * (r_C * r_C).mean()

        loss = loss_data + loss_reg + loss_tv + loss_roma + loss_color_icp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if do_timing:
            _sync_if_cuda()
            t_after_optim = time.perf_counter()

        do_metrics = tb_writer is not None or metrics_out is not None or progress_callback is not None
        if do_metrics:
            used = int(src_used.shape[0])
            rmse = torch.sqrt(d2_used.mean())
            diff = src_used - tgt_used
            rmse_v2 = torch.sqrt((diff * diff).sum(dim=1).mean())
            rmse = rmse_v2
            roma_loss_val = loss_roma.item() if isinstance(loss_roma, torch.Tensor) else loss_roma
            color_icp_val = loss_color_icp.item() if isinstance(loss_color_icp, torch.Tensor) else float(loss_color_icp)

            # Optional per-iteration timing breakdown (for tb summary).
            if (
                t_iter_start is not None
                and t_after_knn is not None
                and t_after_geom_loss is not None
                and t_after_roma_loss is not None
                and t_after_optim is not None
            ):
                dt_knn = (t_after_knn - t_iter_start) * 1000.0
                dt_geom = (t_after_geom_loss - t_after_knn) * 1000.0
                dt_roma = (t_after_roma_loss - t_after_geom_loss) * 1000.0
                dt_optim = (t_after_optim - t_after_roma_loss) * 1000.0
                dt_total = (t_after_optim - t_iter_start) * 1000.0

                timing_knn_ms.append(dt_knn)
                timing_geom_ms.append(dt_geom)
                timing_roma_ms.append(dt_roma)
                timing_opt_ms.append(dt_optim)
                timing_total_ms.append(dt_total)

            if tb_writer is not None:
                prefix = tb_prefix.strip("/")

                def _tag(name: str) -> str:
                    return f"{prefix}/{name}" if prefix else name

                def _to_float(x) -> float:
                    if isinstance(x, torch.Tensor):
                        return float(x.detach().item())
                    return float(x)

                step = it + 1
                tb_writer.add_scalar(_tag("loss"), _to_float(loss), step)
                tb_writer.add_scalar(_tag("loss/data"), _to_float(loss_data), step)
                tb_writer.add_scalar(_tag("loss/reg"), _to_float(loss_reg), step)
                tb_writer.add_scalar(_tag("loss/tv"), _to_float(loss_tv), step)
                tb_writer.add_scalar(_tag("loss/roma"), _to_float(loss_roma), step)
                tb_writer.add_scalar(_tag("loss/color_icp"), _to_float(loss_color_icp), step)
                tb_writer.add_scalar(_tag("rmse"), _to_float(rmse), step)
                tb_writer.add_scalar(_tag("fitness"), float(perc_used), step)
                tb_writer.add_scalar(_tag("num_used"), float(used), step)
                tb_writer.add_scalar(_tag("num_roma_matches"), float(num_roma_matches), step)

            last_metrics = {
                "loss": float(loss.detach().item()),
                "loss_data": float(loss_data.detach().item()),
                "loss_reg": float(loss_reg.detach().item()) if isinstance(loss_reg, torch.Tensor) else float(loss_reg),
                "loss_tv": float(loss_tv.detach().item()) if isinstance(loss_tv, torch.Tensor) else float(loss_tv),
                "loss_roma": float(loss_roma.detach().item())
                if isinstance(loss_roma, torch.Tensor)
                else float(loss_roma),
                "loss_color_icp": float(loss_color_icp.detach().item())
                if isinstance(loss_color_icp, torch.Tensor)
                else float(loss_color_icp),
                "rmse": float(rmse.detach().item()) if isinstance(rmse, torch.Tensor) else float(rmse),
                "fitness": float(perc_used),
                "num_used": float(used),
                "num_roma_matches": float(num_roma_matches),
            }
            iters_completed = it + 1

            if progress_callback is not None:
                progress_callback(iters_completed, last_metrics)

            # Early stopping: stop only when both data loss and roma loss (if used) have stopped improving
            if use_early_stopping:
                cur_data = last_metrics["loss_data"]
                cur_roma = last_metrics["loss_roma"]
                cur_color = last_metrics["loss_color_icp"]
                data_improved = (best_data_loss - cur_data) >= early_stopping_min_delta
                roma_improved = (
                    (best_roma_loss - cur_roma) >= early_stopping_min_delta if use_roma_for_early_stop else False
                )
                color_icp_improved = (
                    (best_color_icp_loss - cur_color) >= early_stopping_min_delta
                    if use_color_icp_for_early_stop
                    else False
                )
                still_improving = (
                    data_improved
                    or (use_roma_for_early_stop and roma_improved)
                    or (use_color_icp_for_early_stop and color_icp_improved)
                )
                if still_improving:
                    if data_improved:
                        best_data_loss = cur_data
                    if use_roma_for_early_stop and roma_improved:
                        best_roma_loss = cur_roma
                    if use_color_icp_for_early_stop and color_icp_improved:
                        best_color_icp_loss = cur_color
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if iters_completed >= early_stopping_min_iters and no_improve_count >= early_stopping_patience:
                    break

    # After optimization loop: log mean timing statistics to TensorBoard
    if tb_writer is not None and len(timing_total_ms) > 0:
        prefix = tb_prefix.strip("/")

        def _tag(name: str) -> str:
            return f"{prefix}/{name}" if prefix else name

        def _mean(xs):
            return float(sum(xs) / len(xs))

        # KDTree build time (single build per ICP call)
        if kdtree_build_ms > 0.0:
            tb_writer.add_scalar(_tag("timing/kdtree_build_ms"), float(kdtree_build_ms), 0)

        tb_writer.add_scalar(_tag("timing/knn_ms_mean"), _mean(timing_knn_ms), 0)
        tb_writer.add_scalar(_tag("timing/geom_ms_mean"), _mean(timing_geom_ms), 0)
        tb_writer.add_scalar(_tag("timing/roma_ms_mean"), _mean(timing_roma_ms), 0)
        tb_writer.add_scalar(_tag("timing/opt_ms_mean"), _mean(timing_opt_ms), 0)
        tb_writer.add_scalar(_tag("timing/total_ms_mean"), _mean(timing_total_ms), 0)

    # final transformed source (using final xi_global and deform)
    with torch.no_grad():
        xi_local = deform(src_flat)
        src_local = se3_apply(xi_local, src_flat)

        Rg, tg = se3_exp(xi_global.unsqueeze(0))
        Rg = Rg[0]
        tg = tg[0]
        src_final = (src_local @ Rg.t()) + tg.view(1, 3)

    src_final = src_final.reshape(*src.shape)

    if metrics_out is not None:
        metrics_out.clear()
        metrics_out["iters_completed"] = int(iters_completed)
        metrics_out["kdtree_build_ms"] = float(kdtree_build_ms)
        if last_metrics is not None:
            metrics_out.update(last_metrics)

    # ------------------------------------------------------------------
    # Per-point loss computation (one final NN query, no gradient)
    # ------------------------------------------------------------------
    if compute_per_point_losses and metrics_out is not None:
        with torch.no_grad():
            src_final_flat = src_final.reshape(-1, 3)

            idxs_pp, d2_pp = query_knn_with_backend(
                src_final_flat,
                ref_flat,
                K=1,
                backend=knn_backend,
                chunk=chunk,
                cpu_tree=tree if knn_backend == "cpu_kdtree" else None,
                gpu_tree=tree if knn_backend == "gpu_kdtree" else None,
            )

            if max_corr_dist is not None:
                valid_pp = d2_pp < (max_corr_dist * max_corr_dist)
            else:
                valid_pp = torch.ones(src_final_flat.shape[0], dtype=torch.bool, device=device)

            tgt_pp = ref_flat[idxs_pp]

            # Geometric loss per point
            if method == "point2plane" and ref_normals is not None:
                normals_pp = ref_normals[idxs_pp]
                rel_pp = src_final_flat - tgt_pp
                proj_pp = (rel_pp * normals_pp).sum(dim=1)
                geom_loss_pp = proj_pp * proj_pp
            else:
                diff_pp = src_final_flat - tgt_pp
                geom_loss_pp = (diff_pp * diff_pp).sum(dim=1)

            # Color ICP loss per point
            color_loss_pp = torch.zeros(src_final_flat.shape[0], device=device, dtype=src.dtype)
            if use_color_icp and ref_color_grad is not None and ref_normals is not None:
                n_pp = ref_normals[idxs_pp]
                d_p_pp = ref_color_grad[idxs_pp]

                diff_qp_pp = src_final_flat - tgt_pp
                dot_qp_pp = (diff_qp_pp * n_pp).sum(dim=1, keepdim=True)
                u_q_pp = diff_qp_pp - dot_qp_pp * n_pp

                I_p_pp = ref_intensity[idxs_pp]
                I_q_pp = src_intensity

                C_hat_pp = I_p_pp + (d_p_pp * u_q_pp).sum(dim=1)
                r_C_pp = C_hat_pp - I_q_pp
                color_loss_pp = r_C_pp * r_C_pp

            metrics_out["per_point_geom_loss"] = geom_loss_pp
            metrics_out["per_point_color_loss"] = color_loss_pp
            metrics_out["per_point_valid_mask"] = valid_pp

    return src_final, xi_global, deform
