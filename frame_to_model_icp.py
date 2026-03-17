"""
Stage 1: Frame-to-model non-rigid ICP alignment.

Incrementally registers each frame's depth-lifted point cloud into a shared
canonical (world) coordinate system.

Per-frame parameterisation:
  - Points live in camera space (depth * K^{-1}).
  - The c2w extrinsic IS the global rigid transform (xi_global, SE3 twist).
  - A local deformation grid handles non-rigid residuals in camera space.

Forward chain:  camera_pts -> local_deform_i(.) -> se3_apply(c2w_i, .) -> canonical
Frame 0 defines the canonical frame and is frozen (gauge fix).
"""

import json
import os
import time

import numpy as np
import open3d as o3d
import torch
import tyro
from tqdm.auto import tqdm

from dataclasses import asdict, replace

from algos.icp import colored_icp_adam
from algos.non_rigid_icp import non_rigid_icp
from data.data_loading import load_data, torch_to_o3d_pcd
from models.roma_matcher import (
    MatchHistory,
    RoMaMatcherWrapper,
    compute_roma_matches_for_frame,
    _get_cache_key,
    _get_cache_path,
    load_cached_matches,
    save_matches_to_cache,
)
from utils.logging import get_logger, try_create_tensorboard_writer, tb_log_hparams
from utils.normals import estimate_normals
from utils.geometry import se3_apply, se3_exp, se3_log
from utils.pointcloud import merge_new_points_with_model, merge_point_clouds

from configs.stage1_align import FrameToModelICPConfig


logger = get_logger(__name__)

_PBAR_POSTFIX_MIN_SECONDS = 0.2
_PBAR_POSTFIX_MIN_ITERS = 10


class ICPLossTracker:
    """Accumulates per-frame ICP loss statistics for adaptive point filtering.

    After each frame's ICP, call ``update()`` with the raw per-point losses
    (before any filtering). Then call ``get_adaptive_thresholds()`` to obtain
    robust, history-aware thresholds for the current frame.
    """

    PERCENTILE_KEYS = ("p75", "p90", "p95", "p99")

    def __init__(self) -> None:
        self.frame_stats: list[dict] = []

    def __len__(self) -> int:
        return len(self.frame_stats)

    @staticmethod
    def _summarise(t: torch.Tensor) -> dict:
        """Compute summary statistics for a 1-D tensor of losses."""
        if t.numel() == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        qs = torch.tensor([0.75, 0.90, 0.95, 0.99], device=t.device, dtype=t.dtype)
        pvals = torch.quantile(t.float(), qs)
        return {
            "mean": t.mean().item(),
            "std": t.std().item() if t.numel() > 1 else 0.0,
            "median": t.median().item(),
            "p75": pvals[0].item(),
            "p90": pvals[1].item(),
            "p95": pvals[2].item(),
            "p99": pvals[3].item(),
        }

    def update(
        self,
        geom_losses: torch.Tensor,
        color_losses: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict:
        """Record per-point loss statistics for one frame."""
        vg = geom_losses[valid_mask]
        vc = color_losses[valid_mask]

        gs = self._summarise(vg)
        cs = self._summarise(vc)
        self.frame_stats.append({"geom": gs, "color": cs})

        return {
            "geom": gs,
            "color": cs,
            "num_valid": int(valid_mask.sum().item()),
            "num_total": int(valid_mask.shape[0]),
        }

    def get_adaptive_thresholds(
        self,
        geom_sigma: float = 3.0,
        color_sigma: float = 3.0,
        base_percentile: str = "p95",
    ) -> tuple[float | None, float | None]:
        """Compute robust adaptive thresholds from frame history."""
        if len(self.frame_stats) < 1:
            return None, None

        geom_vals = np.array([s["geom"][base_percentile] for s in self.frame_stats])
        color_vals = np.array([s["color"][base_percentile] for s in self.frame_stats])

        def _robust_thresh(vals: np.ndarray, sigma: float) -> float:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            return med + sigma * 1.4826 * max(mad, 1e-12)

        return _robust_thresh(geom_vals, geom_sigma), _robust_thresh(color_vals, color_sigma)


# ================================================================
#                         MAIN FUNCTION
# ================================================================
def main(config: FrameToModelICPConfig):
    """Run frame-to-model non-rigid ICP alignment (Stage 1)."""

    # Sentinel defaults: allow higher-level pipelines to decide presets, but keep
    # this stage's standalone CLI behavior unchanged.
    if config.icp_early_stopping_min_delta is None:
        config = replace(config, icp_early_stopping_min_delta=5e-6)

    # Load point clouds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Set out_path if not specified
    out_path = config.out_path
    if out_path is None:
        out_path = os.path.join(
            config.root_path,
            (
                f"frame_to_model_icp_{config.alignment.num_frames}_{config.alignment.stride}"
                f"_offset{config.alignment.offset}{config.out_suffix if config.out_suffix else ''}"
            ),
        )
    os.makedirs(out_path, exist_ok=True)

    # Persist Stage-1-produced config for downstream stages.
    # Canonical location (per repo convention): after_non_rigid_icp/config.json
    stage1_ckpt_dir = os.path.join(out_path, "after_non_rigid_icp")
    os.makedirs(stage1_ckpt_dir, exist_ok=True)
    with open(os.path.join(stage1_ckpt_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    tb_writer = None
    if config.tensorboard:
        tb_base = config.tensorboard_log_dir or os.path.join(out_path, "tensorboard")
        tb_dir = tb_base
        tb_writer = try_create_tensorboard_writer(tb_dir)
        if tb_writer is not None:
            logger.info("TensorBoard logging to: %s", tb_dir)
            tb_writer.add_text("run/root_path", str(config.root_path), 0)
            tb_writer.add_text("run/out_path", str(out_path), 0)
            tb_writer.add_scalar("run/num_frames", float(config.alignment.num_frames), 0)
            tb_writer.add_scalar("run/stride", float(config.alignment.stride), 0)

            tb_log_hparams(
                tb_writer,
                asdict(config),
                step=0,
            )

    # Load original data (needed for both ICP and bundle adjustment)
    (
        pcls,
        extrinsics,
        intrinsics,
        images,
        valid_pixel_indices,
        _depth_conf,
        _depth_maps,
        _orig_images,
        _orig_intrinsics,
    ) = load_data(
        config.root_path,
        config.alignment.num_frames,
        config.alignment.stride,
        device,
        config.alignment.conf_thresh_percentile,
        conf_mode=config.alignment.conf_mode,
        conf_local_percentile=config.alignment.conf_local_percentile,
        conf_global_percentile=config.alignment.conf_global_percentile,
        voxel_size=config.alignment.conf_voxel_size,
        voxel_min_count_percentile=config.alignment.conf_voxel_min_count_percentile,
        offset=config.alignment.offset,
    )

    # ------------------------------------------------------------------
    # Convert world-space points to camera-space and compute c2w SE3 twists.
    # In this variant the per-frame c2w IS the global rigid transform that
    # non_rigid_icp optimises, so local_deform operates in camera space.
    # ------------------------------------------------------------------
    per_frame_camera_pts: list[torch.Tensor] = []
    per_frame_camera_colors: list[torch.Tensor] = []
    per_frame_c2w_se3: list[torch.Tensor] = []
    original_extrinsics_w2c: list[torch.Tensor] = []  # keep for reference

    for _fi in range(len(pcls)):
        world_pts = torch.from_numpy(np.array(pcls[_fi].points)).to(device=device, dtype=torch.float32)
        world_cols = torch.from_numpy(np.array(pcls[_fi].colors)).to(device=device, dtype=torch.float32)

        # Build (4,4) w2c from original extrinsic (expected (3,4) or (4,4) numpy)
        _ext = np.asarray(extrinsics[_fi], dtype=np.float32)
        w2c_44 = np.eye(4, dtype=np.float32)
        w2c_44[: _ext.shape[0], : _ext.shape[1]] = _ext
        w2c_torch = torch.from_numpy(w2c_44).to(device)
        original_extrinsics_w2c.append(w2c_torch)

        # World → camera
        cam_pts = (w2c_torch[:3, :3] @ world_pts.T).T + w2c_torch[:3, 3]
        per_frame_camera_pts.append(cam_pts)
        per_frame_camera_colors.append(world_cols)

        # c2w as SE3 twist (will serve as xi_global_init for non_rigid_icp)
        c2w = torch.linalg.inv(w2c_torch)
        xi_c2w = se3_log(c2w[:3, :3], c2w[:3, 3])
        per_frame_c2w_se3.append(xi_c2w)

    logger.info(
        "Converted %d frames from world-space to camera-space.  c2w SE3 norms: %s",
        len(per_frame_c2w_se3),
        [f"{float(x.norm()):.4f}" for x in per_frame_c2w_se3],
    )

    # RoMa matching requires valid pixel indices (for segment bookkeeping + filtering).
    use_roma_matching = bool(config.roma.use_roma_matching and valid_pixel_indices is not None)
    if config.roma.use_roma_matching and not use_roma_matching:
        logger.warning("RoMa matching requested but valid_pixel_indices not available. Disabling RoMa matching.")
        if tb_writer is not None:
            tb_writer.add_scalar("hparams/effective_use_roma_matching", 0.0, 0)

    # Initialize RoMa matcher if requested
    roma_matcher = None
    match_history = None
    roma_cache_path = None
    cached_roma_matches = None
    if use_roma_matching:
        logger.info(
            "Initializing RoMa matcher (version=%s, model=%s)...",
            config.roma.roma_version,
            config.roma.roma_model,
        )
        roma_matcher = RoMaMatcherWrapper(
            device=device,
            model_type=config.roma.roma_model,
            version=config.roma.roma_version,
        )
        match_history = MatchHistory()

        # Set up caching
        num_frames = len(pcls)
        cache_key = _get_cache_key(
            roma_version=config.roma.roma_version,
            roma_model=config.roma.roma_model,
            num_samples_per_pair=config.roma.roma_num_samples,
            certainty_threshold=config.roma.roma_certainty_threshold,
            num_frames=num_frames,
            stride=config.alignment.stride,
            reference_selection_mode=config.roma.roma_reference_sampling,
        )
        roma_cache_path = _get_cache_path(config.root_path, cache_key)
        cached_roma_matches = load_cached_matches(roma_cache_path, device=device)
        if cached_roma_matches:
            logger.info(f"Loaded {len(cached_roma_matches)} cached ROMA match pairs from {roma_cache_path}")

        logger.info("RoMa matcher initialized successfully")

    # Run Non-Rigid ICP iteratively
    # save downsampled merge of all pcls
    merged = merge_point_clouds(pcls)
    # merged = merged.voxel_down_sample(voxel_size=0.01)
    o3d.io.write_point_cloud(os.path.join(out_path, "before_non_rigid_icp.ply"), merged)

    # Calculate world-space bbox from merged point cloud (used for normal
    # estimation & merge logic; DeformationGrid bbox is per-frame in camera space).
    merged_points = torch.from_numpy(np.array(merged.points)).to(device).to(torch.float32)
    bbox_min = merged_points.min(dim=0)[0]
    bbox_max = merged_points.max(dim=0)[0]
    # Add small padding to avoid edge cases
    padding = (bbox_max - bbox_min) * 0.01
    bbox_min = bbox_min - padding
    bbox_max = bbox_max + padding
    logger.info(
        "Using world-space bbox from merged point cloud: min=%s, max=%s",
        bbox_min.cpu().numpy(),
        bbox_max.cpu().numpy(),
    )

    # --- Frame 0: canonical model via c2w_0 applied to camera-space points ---
    # The canonical model lives in world space = se3_apply(c2w_0, camera_pts_0).
    model = se3_apply(per_frame_c2w_se3[0], per_frame_camera_pts[0])
    model_colors = per_frame_camera_colors[0].clone()
    voxel_size = 0.05  # keep consistent voxel size across the pipeline
    # Initial normals for the canonical model
    model_normals, model_kd_tree = estimate_normals(model.reshape(-1, 3), backend=config.knn_backend)

    # Frame 0's global deform IS its c2w (frozen gauge fix); local deform is identity.
    def dummy_deform(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 6), device=x.device, dtype=torch.float32)

    per_frame_global_deform = [per_frame_c2w_se3[0].clone()]  # c2w_0 (frozen)
    per_frame_local_deform = [dummy_deform]
    ref_frame_indexes = [0]

    # Track frame segments within the model: (start_idx, end_idx) for each frame
    # This allows us to index into model directly instead of maintaining separate copies
    model_frame_segments = [(0, model.shape[0])]  # First frame is the initial model
    # Track valid pixel indices for each frame's segment in the model
    # These get updated when merge_new_points_with_model filters out points
    model_valid_pixel_indices_list = [valid_pixel_indices[0].clone()] if valid_pixel_indices else []

    if tb_writer is not None:
        tb_writer.add_scalar("model/num_points_after_merge", float(model.shape[0]), 0)

    # Accumulate per-frame timings to summarize at the end of the ICP stage
    total_roma_ms = 0.0
    total_icp_ms = 0.0
    total_merge_ms = 0.0
    num_timed_frames = 0

    loss_tracker = ICPLossTracker() if config.filter_points else None

    frames_pbar = tqdm(
        range(1, len(pcls)),
        desc="Frames",
        position=0,
        leave=True,
        dynamic_ncols=True,
        mininterval=_PBAR_POSTFIX_MIN_SECONDS,
    )
    for i in frames_pbar:
        # --- Use camera-space points for this frame ---
        pcl_cam = per_frame_camera_pts[i]  # camera-space source
        pcl_colors = per_frame_camera_colors[i]
        # World-space version (via initial c2w) for visualisation / merge preview only
        pcl_world = se3_apply(per_frame_c2w_se3[i], pcl_cam)

        # --------------------------------------------------------------
        # Multi-scale rigid colored ICP initialisation (world space)
        # --------------------------------------------------------------
        voxel_radius = [0.04, 0.02]
        max_iter_icp = [150, 70]

        src_pcd_full = torch_to_o3d_pcd(pcl_world, pcl_colors)
        ref_pcd_full = torch_to_o3d_pcd(model, model_colors)

        R_acc = torch.eye(3, device=device, dtype=pcl_world.dtype)
        t_acc = torch.zeros(3, device=device, dtype=pcl_world.dtype)

        rigid_desc = f"Rigid ICP f{i:05d}"
        with tqdm(
            total=int(sum(max_iter_icp)),
            desc=rigid_desc,
            position=1,
            leave=False,
            dynamic_ncols=True,
            mininterval=_PBAR_POSTFIX_MIN_SECONDS,
        ) as rigid_pbar:
            last_it_total = 0
            last_postfix_it = 0
            last_postfix_t = time.perf_counter()

            def _rigid_progress_cb(it_total_done: int, m: dict) -> None:
                nonlocal last_it_total
                stage = m.get("stage")
                if stage is not None and int(it_total_done) == 0:
                    if stage == "estimate_normals_start":
                        rigid_pbar.set_postfix_str(
                            f"estimating normals | k={int(m.get('k', 0))} | Nref={int(m.get('num_ref', 0))}"
                        )
                    elif stage == "estimate_normals_end":
                        rigid_pbar.set_postfix_str("normals ready")
                    elif stage == "color_grad_precompute_start":
                        rigid_pbar.set_postfix_str(
                            f"color precompute | k={int(m.get('k', 0))} | Nref={int(m.get('num_ref', 0))}"
                        )
                    elif stage == "color_grad_precompute_end":
                        rigid_pbar.set_postfix_str("color grads ready")
                    elif stage == "kdtree_build_start":
                        rigid_pbar.set_postfix_str(
                            f"building KDTree ({m.get('backend')}) | Nref={int(m.get('num_ref', 0))}"
                        )
                    elif stage == "kdtree_build_end":
                        rigid_pbar.set_postfix_str(
                            f"KDTree ready ({m.get('backend')}) | Nref={int(m.get('num_ref', 0))}"
                        )
                    else:
                        rigid_pbar.set_postfix_str(str(stage))
                    return

                step = max(int(it_total_done) - int(last_it_total), 0)
                if step > 0:
                    rigid_pbar.update(step)
                    last_it_total = int(it_total_done)
                nonlocal last_postfix_it, last_postfix_t
                now = time.perf_counter()
                if (int(it_total_done) - int(last_postfix_it)) < _PBAR_POSTFIX_MIN_ITERS and (
                    now - last_postfix_t
                ) < _PBAR_POSTFIX_MIN_SECONDS:
                    return
                last_postfix_it = int(it_total_done)
                last_postfix_t = now
                rigid_pbar.set_postfix(
                    loss=f"{m.get('loss', 0.0):.3e}",
                    geo=f"{m.get('loss_geo', 0.0):.3e}",
                    col=f"{m.get('loss_color', 0.0):.3e}",
                    rmse=f"{m.get('rmse', 0.0):.3e}",
                    fit=f"{m.get('fitness', 0.0):.1f}%",
                    inl=int(m.get("inliers", 0)),
                    refresh=False,
                )

            cum_base = 0
            for scale, (radius, iters) in enumerate(zip(voxel_radius, max_iter_icp)):
                rigid_pbar.set_postfix_str(f"scale={scale} r={radius:.3f}")

                src_down_pcd = src_pcd_full.voxel_down_sample(radius)
                ref_down_pcd = ref_pcd_full.voxel_down_sample(radius)

                src_down_np = np.asarray(src_down_pcd.points, dtype=np.float32)
                ref_down_np = np.asarray(ref_down_pcd.points, dtype=np.float32)
                src_down = torch.from_numpy(src_down_np).to(device=device, dtype=pcl_world.dtype)
                ref_down = torch.from_numpy(ref_down_np).to(device=device, dtype=pcl_world.dtype)

                if src_down_pcd.has_colors():
                    src_down_colors_np = np.asarray(src_down_pcd.colors, dtype=np.float32)
                else:
                    src_down_colors_np = np.ones_like(src_down_np, dtype=np.float32)
                if ref_down_pcd.has_colors():
                    ref_down_colors_np = np.asarray(ref_down_pcd.colors, dtype=np.float32)
                else:
                    ref_down_colors_np = np.ones_like(ref_down_np, dtype=np.float32)

                src_down_colors = torch.from_numpy(src_down_colors_np).to(device=device)
                ref_down_colors = torch.from_numpy(ref_down_colors_np).to(device=device)

                src_down_transformed = (src_down @ R_acc.t()) + t_acc.view(1, 3)

                def _scale_cb(it_done: int, m: dict) -> None:
                    if int(it_done) == 0:
                        _rigid_progress_cb(0, m)
                        return
                    _rigid_progress_cb(cum_base + int(it_done), m)

                aligned_down, R_delta, t_delta = colored_icp_adam(
                    src_down_transformed,
                    src_down_colors,
                    ref_down,
                    ref_down_colors,
                    n_iter=iters,
                    lr=config.icp_lr,
                    knn_backend=config.knn_backend,
                    max_corr_dist=radius,
                    lambda_geometric=0.968 if config.icp_color_icp_weight > 0.0 else 1.0,
                    color_k=config.icp_color_icp_k,
                    progress_callback=_scale_cb,
                )

                R_acc = R_delta @ R_acc
                t_acc = R_delta @ t_acc + t_delta
                # Ensure progress advances even if early-stopped.
                cum_base += int(iters)
                rigid_pbar.n = min(int(rigid_pbar.total), int(cum_base))
                rigid_pbar.refresh()

        # compose refined rigid init with original c2w for this frame
        R_init, t_init = se3_exp(per_frame_c2w_se3[i].unsqueeze(0))
        R_init = R_init[0]
        t_init = t_init[0]
        R_c2w_refined = R_acc @ R_init
        t_c2w_refined = R_acc @ t_init + t_acc
        xi_global_init = se3_log(R_c2w_refined, t_c2w_refined)
        # xi_global_init = per_frame_c2w_se3[i]

        # Per-frame camera-space bbox for the DeformationGrid
        bbox_cam_min = pcl_cam.min(dim=0)[0]
        bbox_cam_max = pcl_cam.max(dim=0)[0]
        _cam_padding = (bbox_cam_max - bbox_cam_min) * 0.01
        bbox_cam_min = bbox_cam_min - _cam_padding
        bbox_cam_max = bbox_cam_max + _cam_padding

        # --------------------------------------------------------------
        # Compute RoMa matches for this frame against previous frames
        # --------------------------------------------------------------
        t_roma_start = time.perf_counter()
        roma_matches_for_frame = None
        if roma_matcher is not None and match_history is not None:
            frames_pbar.set_postfix_str("RoMa matching...")

            # Compute/load matches (returns original unfiltered matches)
            roma_matches_for_frame = compute_roma_matches_for_frame(
                roma_matcher=roma_matcher,
                images=images,
                current_frame_idx=i,
                max_references=config.roma.roma_max_references,
                num_samples_per_pair=config.roma.roma_num_samples,
                certainty_threshold=config.roma.roma_certainty_threshold,
                cache_path=roma_cache_path,
                cached_matches=cached_roma_matches,
                reference_selection_mode=config.roma.roma_reference_sampling,
            )

            # Save any newly computed matches to disk
            if roma_matches_for_frame and roma_cache_path is not None:
                new_pairs = []
                if cached_roma_matches is None:
                    cached_roma_matches = {}
                for match in roma_matches_for_frame:
                    pair = (match.src_frame_idx, match.ref_frame_idx)
                    if pair not in cached_roma_matches:
                        cached_roma_matches[pair] = match
                        new_pairs.append(match)
                if new_pairs:
                    save_matches_to_cache(
                        roma_cache_path,
                        new_pairs,
                        existing_cache=cached_roma_matches,
                    )

            if roma_matches_for_frame:
                match_history.add_matches(roma_matches_for_frame)
                total_matches = sum(len(m.kpts_src) for m in roma_matches_for_frame)
                frames_pbar.set_postfix_str(
                    f"RoMa matches={total_matches} pairs={len(roma_matches_for_frame)}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("roma/total_matches", float(total_matches), i)
                    tb_writer.add_scalar("roma/num_pairs", float(len(roma_matches_for_frame)), i)
            else:
                frames_pbar.set_postfix_str("RoMa matches=0")
                if tb_writer is not None:
                    tb_writer.add_scalar("roma/total_matches", 0.0, i)
                    tb_writer.add_scalar("roma/num_pairs", 0.0, i)

        t_roma_end = time.perf_counter()

        # --------------------------------------------------------------
        # Non-rigid ICP for this frame (camera-space src, c2w init)
        # --------------------------------------------------------------
        icp_metrics = {}
        t_icp_start = time.perf_counter()
        nonrigid_desc = f"Non-rigid ICP f{i:05d}"
        with tqdm(
            total=int(config.icp_n_iter),
            desc=nonrigid_desc,
            position=1,
            leave=False,
            dynamic_ncols=True,
            mininterval=_PBAR_POSTFIX_MIN_SECONDS,
        ) as icp_pbar:
            last_iter = 0
            last_postfix_it = 0
            last_postfix_t = time.perf_counter()

            def _icp_progress_cb(it_done: int, m: dict) -> None:
                nonlocal last_iter
                stage = m.get("stage")
                if stage is not None and int(it_done) == 0:
                    if stage == "kdtree_build_start":
                        icp_pbar.set_postfix_str(
                            f"building KDTree ({m.get('backend')}) | Nref={int(m.get('num_ref', 0))}"
                        )
                    elif stage == "kdtree_build_end":
                        ms = m.get("kdtree_build_ms", 0.0)
                        icp_pbar.set_postfix_str(
                            f"KDTree built ({m.get('backend')}) | {float(ms):.1f} ms | Nref={int(m.get('num_ref', 0))}"
                        )
                    elif stage == "estimate_normals_start":
                        icp_pbar.set_postfix_str(
                            f"estimating normals | k={int(m.get('k', 0))} | Nref={int(m.get('num_ref', 0))}"
                        )
                    elif stage == "estimate_normals_end":
                        icp_pbar.set_postfix_str("normals ready")
                    elif stage == "color_icp_precompute_start":
                        icp_pbar.set_postfix_str(
                            f"color precompute | k={int(m.get('k', 0))} | Nref={int(m.get('num_ref', 0))}"
                        )
                    elif stage == "color_icp_precompute_end":
                        icp_pbar.set_postfix_str("color grads ready")
                    else:
                        icp_pbar.set_postfix_str(str(stage))
                    return

                step = max(int(it_done) - int(last_iter), 0)
                if step > 0:
                    icp_pbar.update(step)
                    last_iter = int(it_done)
                nonlocal last_postfix_it, last_postfix_t
                now = time.perf_counter()
                if (int(it_done) - int(last_postfix_it)) < _PBAR_POSTFIX_MIN_ITERS and (
                    now - last_postfix_t
                ) < _PBAR_POSTFIX_MIN_SECONDS:
                    return
                last_postfix_it = int(it_done)
                last_postfix_t = now
                icp_pbar.set_postfix(
                    loss=f"{m.get('loss', 0.0):.3e}",
                    data=f"{m.get('loss_data', 0.0):.3e}",
                    tv=f"{m.get('loss_tv', 0.0):.3e}",
                    roma=f"{m.get('loss_roma', 0.0):.3e}",
                    col=f"{m.get('loss_color_icp', 0.0):.3e}",
                    rmse=f"{m.get('rmse', 0.0):.3e}",
                    fit=f"{m.get('fitness', 0.0):.1f}%",
                    used=int(m.get("num_used", 0)),
                    refresh=False,
                )

            aligned_src_to_ref_nr, global_deform, local_deform = non_rigid_icp(
                src=pcl_cam,
                ref=model,
                ref_normals=model_normals,
                roma_matches_data=roma_matches_for_frame,
                roma_model_frame_segments=model_frame_segments.copy(),
                roma_ref_frame_indices=ref_frame_indexes.copy(),
                roma_src_valid_pixel_indices=valid_pixel_indices[i] if valid_pixel_indices else None,
                roma_model_valid_pixel_indices_list=model_valid_pixel_indices_list.copy() if valid_pixel_indices else None,
                roma_loss_weight=config.roma.roma_loss_weight if use_roma_matching else 0.0,
                roma_max_corr_dist=config.roma.roma_max_corr_dist,
                image_height=images.shape[2],
                image_width=images.shape[3],
                n_iter=config.icp_n_iter,
                early_stopping_patience=config.icp_early_stopping_patience,
                early_stopping_min_iters=config.icp_early_stopping_min_iters,
                early_stopping_min_delta=config.icp_early_stopping_min_delta,
                lr=config.icp_lr,
                method=config.icp_method,
                max_corr_dist=config.max_corr_dist,
                knn_backend=config.knn_backend,
                local_twist_reg=config.icp_local_twist_reg,
                tv_reg=config.icp_tv_reg,
                bbox_min=bbox_cam_min,
                bbox_max=bbox_cam_max,
                tb_writer=tb_writer,
                tb_prefix=f"frame_{i:05d}",
                metrics_out=icp_metrics,
                tv_voxel_size=config.icp_tv_voxel_size,
                tv_every_k=config.icp_tv_every_k,
                tv_sample_ratio=config.icp_tv_sample_ratio,
                ref_kd_tree=model_kd_tree,
                xi_global_init=xi_global_init,
                color_icp_weight=config.icp_color_icp_weight,
                color_icp_max_color_dist=config.icp_color_icp_max_color_dist,
                color_icp_k=config.icp_color_icp_k,
                src_colors=pcl_colors,
                ref_colors=model_colors,
                deform_log2_hashmap_size=config.deform_log2_hashmap_size,
                deform_num_levels=config.deform_num_levels,
                deform_n_neurons=config.deform_n_neurons,
                deform_n_hidden_layers=config.deform_n_hidden_layers,
                deform_min_res=config.deform_min_res,
                deform_max_res=config.deform_max_res,
                compute_per_point_losses=config.filter_points,
                progress_callback=_icp_progress_cb,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_icp_end = time.perf_counter()

        if tb_writer is not None and icp_metrics.get("iters_completed", 0) > 0:
            tb_writer.add_scalar("icp_final/loss", float(icp_metrics.get("loss", 0.0)), i)
            tb_writer.add_scalar("icp_final/loss_data", float(icp_metrics.get("loss_data", 0.0)), i)
            tb_writer.add_scalar("icp_final/loss_reg", float(icp_metrics.get("loss_reg", 0.0)), i)
            tb_writer.add_scalar("icp_final/loss_tv", float(icp_metrics.get("loss_tv", 0.0)), i)
            tb_writer.add_scalar(
                "icp_final/loss_semantic",
                float(icp_metrics.get("loss_semantic", 0.0)),
                i,
            )
            tb_writer.add_scalar("icp_final/loss_roma", float(icp_metrics.get("loss_roma", 0.0)), i)
            tb_writer.add_scalar("icp_final/rmse", float(icp_metrics.get("rmse", 0.0)), i)
            tb_writer.add_scalar("icp_final/fitness", float(icp_metrics.get("fitness", 0.0)), i)
            tb_writer.add_scalar("icp_final/num_used", float(icp_metrics.get("num_used", 0.0)), i)
            tb_writer.add_scalar(
                "icp_final/num_roma_matches",
                float(icp_metrics.get("num_roma_matches", 0.0)),
                i,
            )
            tb_writer.add_scalar(
                "icp_final/iters_completed",
                float(icp_metrics.get("iters_completed", 0.0)),
                i,
            )

        ref_frame_indexes.append(i)
        per_frame_global_deform.append(global_deform)
        per_frame_local_deform.append(local_deform)

        # --------------------------------------------------------------
        # Per-point loss filtering (post-ICP, pre-merge)
        # --------------------------------------------------------------
        icp_keep_mask = None  # None ⇒ no ICP-level filtering applied
        if config.filter_points and loss_tracker is not None:
            pp_geom = icp_metrics.get("per_point_geom_loss")
            pp_color = icp_metrics.get("per_point_color_loss")
            pp_valid = icp_metrics.get("per_point_valid_mask")

            if pp_geom is not None and pp_color is not None and pp_valid is not None:
                n_before = int(pp_valid.shape[0])

                # 1) Record unfiltered stats for this frame
                frame_loss_stats = loss_tracker.update(pp_geom, pp_color, pp_valid)
                logger.info(
                    "[Filter][Frame %d] loss stats — geom: mean=%.3e p95=%.3e | "
                    "color: mean=%.3e p95=%.3e | valid: %d / %d",
                    i,
                    frame_loss_stats["geom"]["mean"],
                    frame_loss_stats["geom"]["p95"],
                    frame_loss_stats["color"]["mean"],
                    frame_loss_stats["color"]["p95"],
                    frame_loss_stats["num_valid"],
                    frame_loss_stats["num_total"],
                )

                # 2) Build icp_keep_mask.
                # IMPORTANT: points without any correspondence (pp_valid == False)
                # should still be KEPT, since they can add new content to the scene.
                # We therefore:
                #   - Use pp_valid only for statistics and thresholding,
                #   - But start from "keep everything" and never drop pp_valid == False.
                icp_keep_mask = torch.ones_like(pp_valid, dtype=torch.bool)
                n_removed_invalid = 0
                n_removed_thresh = 0
                n_removed_topk = 0

                # (a) Adaptive threshold (needs enough history)
                geom_thresh, color_thresh = None, None
                if len(loss_tracker) >= config.filter_min_frames:
                    geom_thresh, color_thresh = loss_tracker.get_adaptive_thresholds(
                        geom_sigma=config.filter_geom_sigma,
                        color_sigma=config.filter_color_sigma,
                        base_percentile=config.filter_base_percentile,
                    )
                if geom_thresh is not None and color_thresh is not None:
                    above_geom = pp_geom > geom_thresh
                    above_color = pp_color > color_thresh
                    # Only ever drop points that actually had correspondences.
                    thresh_remove = (above_geom | above_color) & icp_keep_mask & pp_valid
                    n_removed_thresh = int(thresh_remove.sum().item())
                    icp_keep_mask = icp_keep_mask & ~thresh_remove

                # (b) Top-k removal (always available)
                valid_for_topk = pp_valid & icp_keep_mask
                if config.filter_worst_pct > 0.0 and valid_for_topk.sum() > 0:
                    combined = pp_geom + config.icp_color_icp_weight * pp_color
                    # Exclude non-correspondence points and already-removed points
                    combined[~valid_for_topk] = -float("inf")
                    valid_combined = combined[valid_for_topk]
                    if valid_combined.numel() > 0:
                        cutoff = torch.quantile(
                            valid_combined.float(),
                            1.0 - config.filter_worst_pct,
                        )
                        topk_remove = (combined > cutoff) & valid_for_topk
                        n_removed_topk = int(topk_remove.sum().item())
                        icp_keep_mask = icp_keep_mask & ~topk_remove

                n_removed_total = n_before - int(icp_keep_mask.sum().item())
                keep_ratio = icp_keep_mask.sum().item() / max(n_before, 1)
                logger.info(
                    "[Filter][Frame %d] removed %d / %d points (%.1f%% kept) — "
                    "invalid=%d, threshold=%d (geom=%.3e, color=%.3e), topk=%d",
                    i,
                    n_removed_total,
                    n_before,
                    keep_ratio * 100.0,
                    n_removed_invalid,
                    n_removed_thresh,
                    geom_thresh if geom_thresh is not None else float("nan"),
                    color_thresh if color_thresh is not None else float("nan"),
                    n_removed_topk,
                )

                if tb_writer is not None:
                    tb_writer.add_scalar("filter/num_removed_invalid", float(n_removed_invalid), i)
                    tb_writer.add_scalar("filter/num_removed_threshold", float(n_removed_thresh), i)
                    tb_writer.add_scalar("filter/num_removed_topk", float(n_removed_topk), i)
                    tb_writer.add_scalar("filter/num_removed_total", float(n_removed_total), i)
                    tb_writer.add_scalar("filter/keep_ratio", float(keep_ratio), i)
                    if geom_thresh is not None:
                        tb_writer.add_scalar("filter/geom_threshold", float(geom_thresh), i)
                    if color_thresh is not None:
                        tb_writer.add_scalar("filter/color_threshold", float(color_thresh), i)
                    tb_writer.add_scalar("filter/geom_mean", float(frame_loss_stats["geom"]["mean"]), i)
                    tb_writer.add_scalar("filter/color_mean", float(frame_loss_stats["color"]["mean"]), i)
                    tb_writer.add_scalar("filter/geom_p95", float(frame_loss_stats["geom"]["p95"]), i)
                    tb_writer.add_scalar("filter/color_p95", float(frame_loss_stats["color"]["p95"]), i)

                # Apply ICP-level filter to the aligned points and colors
                aligned_src_to_ref_nr = aligned_src_to_ref_nr.reshape(-1, 3)[icp_keep_mask]
                pcl_colors = pcl_colors[icp_keep_mask]

        # Track the start index for this frame's points in the model
        frame_start_idx = model.shape[0]

        # --------------------------------------------------------------
        # Merge new points into the existing model (including normals)
        # --------------------------------------------------------------
        t_merge_start = time.perf_counter()
        model, model_colors, model_normals, merge_keep_mask, model_kd_tree = merge_new_points_with_model(
            model_points=model,
            model_colors=model_colors,
            model_normals=model_normals,
            new_points=aligned_src_to_ref_nr,
            new_colors=pcl_colors,
            voxel_size=voxel_size,
            color_thresh=-1,  # 0.15,  # in [0,1] color space; keep points when colors disagree
            verbose=False,
            downsample_new_points=False,
            voxel_size_downsample=voxel_size * 0.1,
            knn_backend=config.knn_backend,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_merge_end = time.perf_counter()

        if model_kd_tree is not None:
            logger.info(
                "KDTree built during merge for next ICP iteration on %d points",
                int(model.shape[0]),
            )

        # --------------------------------------------------------------
        # Per-frame timing summary
        # --------------------------------------------------------------
        t_roma_ms = (t_roma_end - t_roma_start) * 1000.0
        t_icp_ms = (t_icp_end - t_icp_start) * 1000.0
        t_merge_ms = (t_merge_end - t_merge_start) * 1000.0
        total_roma_ms += t_roma_ms
        total_icp_ms += t_icp_ms
        total_merge_ms += t_merge_ms
        num_timed_frames += 1

        logger.info(
            "[Timing][Frame %d] RoMa=%.1f ms | ICP=%.1f ms | Merge=%.1f ms | Total=%.1f ms",
            i,
            t_roma_ms,
            t_icp_ms,
            t_merge_ms,
            t_roma_ms + t_icp_ms + t_merge_ms,
        )

        # Also log per-frame timings to TensorBoard for detailed analysis
        if tb_writer is not None:
            tb_writer.add_scalar("timing/frame_roma_ms", float(t_roma_ms), i)
            tb_writer.add_scalar("timing/frame_icp_ms", float(t_icp_ms), i)
            tb_writer.add_scalar("timing/frame_merge_ms", float(t_merge_ms), i)
            tb_writer.add_scalar(
                "timing/frame_total_ms",
                float(t_roma_ms + t_icp_ms + t_merge_ms),
                i,
            )

        # Track frame segment in model: (start_idx, end_idx)
        frame_end_idx = model.shape[0]
        model_frame_segments.append((frame_start_idx, frame_end_idx))

        # Update valid pixel indices for this frame (compose ICP + merge masks)
        if valid_pixel_indices:
            if icp_keep_mask is not None:
                filtered_vpi = valid_pixel_indices[i][icp_keep_mask]
            else:
                filtered_vpi = valid_pixel_indices[i]
            final_pixel_indices = filtered_vpi[merge_keep_mask]
            model_valid_pixel_indices_list.append(final_pixel_indices)

        # Model growth is controlled by only adding points in empty voxels
        # via merge_new_points_with_model (rather than downsampling the full model).
        frames_pbar.set_postfix_str(f"model_pts={int(model.shape[0])}")
        if tb_writer is not None:
            tb_writer.add_scalar("model/num_points_after_merge", float(model.shape[0]), i)
            tb_writer.add_scalar("merge/num_new_points_kept", float(int(merge_keep_mask.sum().item())), i)
            tb_writer.add_scalar(
                "merge/keep_ratio",
                float(merge_keep_mask.to(torch.float32).mean().item()) if merge_keep_mask.numel() > 0 else 0.0,
                i,
            )
            tb_writer.flush()

        if config.save_intermediate_every > 0 and (i % config.save_intermediate_every) == 0:
            o3d.io.write_point_cloud(
                os.path.join(out_path, f"after_non_rigid_icp_step_{i:05d}.ply"),
                torch_to_o3d_pcd(model, model_colors),
            )

    # After processing all frames with non-rigid ICP, log mean timings.
    if tb_writer is not None and num_timed_frames > 0:
        mean_roma_ms = total_roma_ms / num_timed_frames
        mean_icp_ms = total_icp_ms / num_timed_frames
        mean_merge_ms = total_merge_ms / num_timed_frames
        mean_total_ms = (total_roma_ms + total_icp_ms + total_merge_ms) / num_timed_frames

        tb_writer.add_scalar("timing/mean_roma_ms", float(mean_roma_ms), 0)
        tb_writer.add_scalar("timing/mean_icp_ms", float(mean_icp_ms), 0)
        tb_writer.add_scalar("timing/mean_merge_ms", float(mean_merge_ms), 0)
        tb_writer.add_scalar("timing/mean_total_ms", float(mean_total_ms), 0)

    model = model.reshape(-1, 3)
    model_colors = model_colors.reshape(-1, 3)

    # Save current point cloud
    pcl_after_non_rigid_icp = torch_to_o3d_pcd(model, model_colors)
    sub_path = os.path.join(out_path, "after_non_rigid_icp")
    os.makedirs(sub_path, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(sub_path, "aligned_points.ply"), pcl_after_non_rigid_icp)
    for i in range(len(per_frame_global_deform)):
        torch.save(
            per_frame_global_deform[i],
            os.path.join(sub_path, f"per_frame_global_deform_{i:05d}.pt"),
        )
    for i in range(1, len(per_frame_local_deform)):  # skip the first one since it's the dummy
        torch.save(
            per_frame_local_deform[i].state_dict(),
            os.path.join(sub_path, f"per_frame_local_deform_{i:05d}.pt"),
        )

    # Save RoMa match history if available
    if match_history is not None and len(match_history) > 0:
        match_history_data = {
            "frame_pairs": [(m.src_frame_idx, m.ref_frame_idx) for m in match_history.all_matches],
            "kpts_src": [m.kpts_src.cpu() for m in match_history.all_matches],
            "kpts_ref": [m.kpts_ref.cpu() for m in match_history.all_matches],
            "certainty": [m.certainty.cpu() for m in match_history.all_matches],
        }
        torch.save(match_history_data, os.path.join(sub_path, "roma_match_history.pt"))
        logger.info(f"Saved RoMa match history with {len(match_history)} match sets to {sub_path}")

    # Save model frame segments and filtered pixel indices for downstream stages
    # These exist regardless of whether RoMa was used and encode ICPLossTracker
    # filtering + voxel merge decisions.
    if model_frame_segments is not None and model_valid_pixel_indices_list is not None:
        torch.save(model_frame_segments, os.path.join(sub_path, "model_frame_segments.pt"))
        torch.save(
            [idx.cpu() for idx in model_valid_pixel_indices_list],
            os.path.join(sub_path, "model_valid_pixel_indices_list.pt"),
        )
        logger.info(f"Saved model frame segments and pixel indices to {sub_path}")

    # Save convention metadata so downstream consumers know the parameterisation
    convention = {
        "variant": "c2w",
        "global_deform_is": "c2w",
        "local_deform_space": "camera",
        "description": (
            "per_frame_global_deform contains full c2w SE3 twists (not corrections). "
            "local_deform operates in camera space.  To reconstruct canonical points: "
            "canonical = se3_apply(c2w_i, local_deform_i(camera_pts_i))."
        ),
    }
    with open(os.path.join(sub_path, "convention.json"), "w") as _fconv:
        json.dump(convention, _fconv, indent=2)

    # Save original extrinsics for reference
    torch.save(
        [w2c.cpu() for w2c in original_extrinsics_w2c],
        os.path.join(sub_path, "original_extrinsics_w2c.pt"),
    )
    logger.info("Saved convention metadata and original extrinsics to %s", sub_path)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    tyro.cli(main)
