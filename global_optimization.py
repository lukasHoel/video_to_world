"""
Stage 2: Global optimization via leave-one-out consensus.

Loads a Stage 1 checkpoint and jointly optimises all per-frame deformations to
sharpen the canonical point cloud using LOO geometric losses, colored-ICP
photometric terms, and thin-shell surface regularisation.
"""

from __future__ import annotations

import json
import os

import numpy as np
import open3d as o3d
import torch
import tyro
from tqdm.auto import tqdm
import time

from algos.global_optimization import global_opt
from configs.stage2_global_optimization import GlobalOptimizationConfig
from data.checkpoint_loading import (
    AlignmentDataParams,
    load_alignment_data_params,
    load_deformation_checkpoints,
    load_roma_model_index_data,
)
from data.data_loading import load_data, torch_to_o3d_pcd
from utils.logging import get_logger, try_create_tensorboard_writer, tb_log_hparams


logger = get_logger(__name__)


def main(config: GlobalOptimizationConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # -------------------------------------------------------------------------
    # Resolve paths and load checkpoints
    # -------------------------------------------------------------------------
    run_dir = os.path.join(config.root_path, config.run)
    checkpoint_dir = os.path.join(run_dir, config.checkpoint_subdir)
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    # Verify input convention (REQUIRED -- must be c2w)
    conv_path = os.path.join(checkpoint_dir, "convention.json")
    if os.path.exists(conv_path):
        with open(conv_path) as f:
            input_convention = json.load(f)
        logger.info("Input checkpoint convention: %s", input_convention)
        if input_convention.get("global_deform_is") != "c2w":
            raise ValueError(
                f"Input checkpoint uses convention '{input_convention.get('global_deform_is')}', "
                f"but this runner requires 'c2w' (from frame_to_model_icp)."
            )
    else:
        raise ValueError(
            f"No convention.json found in {checkpoint_dir}. "
            f"This runner requires checkpoints from frame_to_model_icp.py "
            f"(which saves convention.json)."
        )

    logger.info("Loading deformations from: %s", checkpoint_dir)
    per_frame_global_deform, per_frame_local_deform, bbox_min, bbox_max = load_deformation_checkpoints(
        checkpoint_dir,
        device=device,
        first_local="dummy",
        allow_rigid_fallback=True,
    )

    # -------------------------------------------------------------------------
    # Load model index data (model frame segments + filtered pixel indices)
    # -------------------------------------------------------------------------
    model_valid_pixel_indices_list = None
    model_frame_segments = None

    index_data = load_roma_model_index_data(checkpoint_dir, device=device)
    if index_data is None:
        raise ValueError(
            "Stage 2 requires RoMa/ICP-filtered index masking data, but none was found. "
            "Missing RoMa index data in checkpoint. Re-run Stage 1 with RoMa index saving enabled "
            "(or ensure the checkpoint directory contains the RoMa model index outputs)."
        )
    model_frame_segments = index_data.model_frame_segments
    model_valid_pixel_indices_list = index_data.model_valid_pixel_indices_list
    if model_frame_segments is None or model_valid_pixel_indices_list is None:
        raise ValueError(
            "Stage 2 requires RoMa/ICP-filtered index masking data, but loaded index data is incomplete "
            "(missing model_frame_segments and/or model_valid_pixel_indices_list)."
        )
    logger.info(
        "Loaded model index data: %d frames with pre-filtered pixel indices",
        len(model_valid_pixel_indices_list),
    )

    # -------------------------------------------------------------------------
    # Load per-frame data and convert world-space points to camera space
    # -------------------------------------------------------------------------
    # Reuse the original frame_to_model_icp config so that confidence / voxel
    # filtering and frame sampling exactly match the Stage 1 run.
    align_params: AlignmentDataParams = load_alignment_data_params(
        root_path=config.root_path,
        run=config.run,
    )

    (
        pcls,
        extrinsics,
        _intrinsics,
        images,
        valid_pixel_indices,
        _depth_conf,
        _depth_maps,
        _orig_images,
        _orig_intrinsics,
    ) = load_data(
        config.root_path,
        align_params.num_frames,
        align_params.stride,
        device,
        align_params.conf_thresh_percentile,
        conf_mode=align_params.conf_mode,
        conf_local_percentile=align_params.conf_local_percentile,
        conf_global_percentile=align_params.conf_global_percentile,
        voxel_size=align_params.conf_voxel_size,
        voxel_min_count_percentile=align_params.conf_voxel_min_count_percentile,
        offset=align_params.offset,
    )

    num_frames = len(pcls)
    logger.info("Loaded %d frames", num_frames)

    # Convert world-space point clouds to camera space using extrinsics.
    # The per-frame c2w IS the global rigid
    # transform and local_deform operates in camera space.
    per_frame_camera_pts: list[torch.Tensor] = []
    per_frame_camera_colors: list[torch.Tensor] = []
    original_extrinsics_w2c: list[torch.Tensor] = []

    for i in range(num_frames):
        world_pts = torch.from_numpy(np.asarray(pcls[i].points)).to(device=device, dtype=torch.float32)
        world_cols = torch.from_numpy(np.asarray(pcls[i].colors)).to(device=device, dtype=torch.float32)

        # Build (4,4) w2c from original extrinsic (expected (3,4) or (4,4) numpy)
        _ext = np.asarray(extrinsics[i], dtype=np.float32)
        w2c_44 = np.eye(4, dtype=np.float32)
        w2c_44[: _ext.shape[0], : _ext.shape[1]] = _ext
        w2c_torch = torch.from_numpy(w2c_44).to(device)
        original_extrinsics_w2c.append(w2c_torch)

        # World -> camera
        cam_pts = (w2c_torch[:3, :3] @ world_pts.T).T + w2c_torch[:3, 3]
        per_frame_camera_pts.append(cam_pts)
        per_frame_camera_colors.append(world_cols)

    # If we have RoMa model index data, reconstruct the exact
    # ICP+merge-filtered per-frame point sets so global optimization starts from the same
    # canonical model as frame_to_model_icp.
    if (
        model_frame_segments is not None
        and model_valid_pixel_indices_list is not None
        and valid_pixel_indices is not None
    ):
        if len(model_frame_segments) != num_frames or len(model_valid_pixel_indices_list) != num_frames:
            raise ValueError(
                "Stage 2 requires RoMa/ICP-filtered index masking for all frames, but the number of frames "
                f"does not match: loaded_frames={num_frames}, "
                f"index_segments={len(model_frame_segments)}, "
                f"index_pixel_lists={len(model_valid_pixel_indices_list)}. "
                "Re-run Stage 1 / regenerate the checkpoint with consistent frame sampling (num_frames/stride/offset)."
            )
        if len(valid_pixel_indices) != num_frames:
            raise ValueError(
                "Stage 2 requires valid_pixel_indices for all loaded frames to apply RoMa/ICP masking, but got "
                f"len(valid_pixel_indices)={len(valid_pixel_indices)} vs loaded_frames={num_frames}."
            )
        frames_to_use = num_frames
        for i in range(frames_to_use):
            vpi = valid_pixel_indices[i].to(device)
            model_vpi = model_valid_pixel_indices_list[i].to(device)

            # Build mask: keep only those points whose pixel indices survived
            # ICP loss + merge filtering in the original run.
            vpi_sorted, order = torch.sort(vpi)
            mv_sorted, _ = torch.sort(model_vpi)
            pos = torch.searchsorted(vpi_sorted, mv_sorted)
            pos_clamped = torch.clamp(pos, 0, vpi_sorted.numel() - 1)
            matches = vpi_sorted[pos_clamped] == mv_sorted
            if not matches.all():
                missing = int((~matches).sum().item())
                total = int(mv_sorted.numel())
                raise ValueError(
                    "Stage 2 requires exact RoMa/ICP-filtered index masking, but some model pixel indices "
                    f"were not found in the currently loaded valid_pixel_indices for frame {i}: "
                    f"missing={missing} of total_model_indices={total}. "
                    "This usually indicates a mismatch between Stage 1 and Stage 2 data loading/filtering "
                    "(confidence thresholding/voxel filtering/stride/offset)."
                )
            mask_sorted = torch.zeros_like(vpi_sorted, dtype=torch.bool)
            if matches.any():
                mask_sorted[pos_clamped[matches]] = True
            keep_mask = torch.zeros_like(vpi, dtype=torch.bool)
            keep_mask[order] = mask_sorted

            # Sanity check against checkpoint model_frame_segments
            seg_start, seg_end = model_frame_segments[i]
            seg_len = int(seg_end - seg_start)
            kept = int(keep_mask.sum().item())
            if kept != seg_len:
                raise ValueError(
                    "Stage 2 requires exact RoMa/ICP-filtered index masking, but the number of kept points "
                    f"does not match the checkpoint segment length for frame {i}: kept={kept}, seg_len={seg_len}. "
                    "Re-run Stage 1 / regenerate the checkpoint to ensure consistency."
                )

            per_frame_camera_pts[i] = per_frame_camera_pts[i][keep_mask]
            per_frame_camera_colors[i] = per_frame_camera_colors[i][keep_mask]
    else:
        raise ValueError(
            "Stage 2 requires RoMa/ICP-filtered index masking, but it could not be applied. "
            "Missing one of: model_frame_segments, model_valid_pixel_indices_list, or valid_pixel_indices."
        )

    # -------------------------------------------------------------------------
    # Setup output directory and TensorBoard
    # -------------------------------------------------------------------------
    out_path = os.path.join(run_dir, config.out_subdir)
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    tb_writer = None
    if config.tensorboard:
        tb_dir = config.tensorboard_log_dir or os.path.join(out_path, "tensorboard")
        tb_writer = try_create_tensorboard_writer(tb_dir)
        if tb_writer is not None:
            logger.info("TensorBoard logging to: %s", tb_dir)
            tb_log_hparams(
                tb_writer,
                {
                    "root_path": config.root_path,
                    "run": config.run,
                    "num_frames": num_frames,
                    "stride": align_params.stride,
                    "convention": "c2w",
                    "loo_loss_weight": config.loo_loss_weight,
                    "loo_k_neighbors": config.loo_k_neighbors,
                    "loo_max_corr_dist": config.loo_max_corr_dist,
                    "loo_normal_k": config.loo_normal_k,
                    "loo_kdtree_rebuild_every": config.loo_kdtree_rebuild_every,
                    "knn_backend": config.knn_backend,
                    "anchor_loss_weight": config.anchor_loss_weight,
                    "tv_reg": config.tv_reg,
                    "loo_color_icp_weight": config.loo_color_icp_weight,
                    "loo_color_icp_k": config.loo_color_icp_k,
                    "loo_color_icp_max_color_dist": config.loo_color_icp_max_color_dist,
                    "thin_shell_weight": config.thin_shell_weight,
                    "lr": config.lr,
                    "n_iters": config.n_iters,
                    "device": device,
                },
                step=0,
            )

    # -------------------------------------------------------------------------
    # Run global optimization (camera-space points + c2w global rigid)
    # -------------------------------------------------------------------------
    # global_opt() applies local_deform(pts) then se3_apply(global_rigid, ...).
    # With camera-space input and c2w as global_rigid, the output is correctly
    # in canonical / world space.
    with tqdm(
        total=int(config.n_iters),
        desc="Global optimization",
        position=0,
        leave=True,
        dynamic_ncols=True,
        mininterval=0.2,
    ) as opt_pbar:
        last_it = 0
        last_postfix_it = 0
        last_postfix_t = time.perf_counter()

        def _opt_progress_cb(it_done: int, m: dict) -> None:
            nonlocal last_it
            stage = m.get("stage")
            if stage is not None and int(it_done) == 0:
                if stage == "knn_init_start":
                    opt_pbar.set_postfix_str("initializing KNN/normals...")
                elif stage == "knn_init_end":
                    opt_pbar.set_postfix_str(f"KNN ready | Nmodel={int(m.get('num_model', 0))}")
                elif stage == "loo_color_precompute_start":
                    opt_pbar.set_postfix_str(f"precomputing color grads | k={int(m.get('k', 0))}")
                elif stage == "loo_color_precompute_end":
                    opt_pbar.set_postfix_str("color grads ready")
                elif stage == "anchoring_init_start":
                    opt_pbar.set_postfix_str(f"anchoring init | Nframes={int(m.get('num_frames', 0))}")
                elif stage == "anchoring_init_end":
                    opt_pbar.set_postfix_str(
                        f"anchoring ready | Nframes={int(m.get('num_frames', 0))} Ns={int(m.get('n_samples', 0))}"
                    )
                elif stage == "knn_rebuild_start":
                    opt_pbar.set_postfix_str(f"rebuilding KNN/normals | iter={int(m.get('iter', 0))}")
                elif stage == "knn_rebuild_end":
                    opt_pbar.set_postfix_str(f"KNN rebuilt | iter={int(m.get('iter', 0))}")
                elif stage == "saved_intermediate":
                    opt_pbar.set_postfix_str(f"saved intermediate | iter={int(m.get('iter', 0))}")
                elif stage == "finished":
                    opt_pbar.set_postfix_str(f"done | Ncanon={int(m.get('num_canonical_points', 0))}")
                else:
                    opt_pbar.set_postfix_str(str(stage))
                return

            step = max(int(it_done) - int(last_it), 0)
            if step > 0:
                opt_pbar.update(step)
                last_it = int(it_done)
            nonlocal last_postfix_it, last_postfix_t
            now = time.perf_counter()
            if (int(it_done) - int(last_postfix_it)) < 10 and (now - last_postfix_t) < 0.2:
                return
            last_postfix_it = int(it_done)
            last_postfix_t = now
            opt_pbar.set_postfix(
                tot=f"{m.get('loss_total', 0.0):.3e}",
                loo=f"{m.get('loss_loo', 0.0):.3e}",
                anc=f"{m.get('loss_anchor', 0.0):.3e}",
                tv=f"{m.get('loss_tv', 0.0):.3e}",
                col=f"{m.get('loss_loo_color_icp', 0.0):.3e}",
                n=int(m.get("loo_n_valid", 0)),
                g=f"{m.get('grad_norm', 0.0):.2e}",
                t=f"{m.get('time_iter_s', 0.0):.2f}s",
                refresh=False,
            )

        nrba_result = global_opt(
        per_frame_world_points=per_frame_camera_pts,
        per_frame_world_colors=per_frame_camera_colors,
        per_frame_global_rigid=per_frame_global_deform,
        per_frame_local_deform=per_frame_local_deform,
        loo_loss_weight=config.loo_loss_weight,
        loo_k_neighbors=config.loo_k_neighbors,
        loo_max_corr_dist=config.loo_max_corr_dist,
        loo_normal_k=config.loo_normal_k,
        loo_kdtree_rebuild_every=config.loo_kdtree_rebuild_every,
        knn_backend=config.knn_backend,
        anchor_loss_weight=config.anchor_loss_weight,
        anchor_n_samples=config.anchor_n_samples,
        tv_reg=config.tv_reg,
        tv_voxel_size=config.tv_voxel_size,
        tv_every_k=config.tv_every_k,
        tv_sample_ratio=config.tv_sample_ratio,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        loo_color_icp_weight=config.loo_color_icp_weight,
        loo_color_icp_k=config.loo_color_icp_k,
        loo_color_icp_max_color_dist=config.loo_color_icp_max_color_dist,
        thin_shell_weight=config.thin_shell_weight,
        loo_max_pairs_per_iter=config.loo_max_pairs_per_iter,
        loo_pairs_per_src=config.loo_pairs_per_src,
        deform_chunk_size=config.deform_chunk_size,
        lr=config.lr,
        n_iters=config.n_iters,
        tb_writer=tb_writer,
        save_intermediate_dir=os.path.join(out_path, "intermediate") if config.tensorboard else None,
        save_intermediate_every_n=config.save_intermediate_every_n,
        progress_callback=_opt_progress_cb,
    )

    # Unpack results
    canonical_points = nrba_result["canonical_points"]
    canonical_colors = nrba_result["canonical_colors"]
    per_frame_global_rigid = nrba_result["per_frame_global_rigid"]
    per_frame_local_deform = nrba_result["per_frame_local_deform"]
    model_frame_segments = nrba_result["model_frame_segments"]

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    pcl_after_nrba = torch_to_o3d_pcd(canonical_points, canonical_colors)
    o3d.io.write_point_cloud(os.path.join(out_path, "aligned_points.ply"), pcl_after_nrba)

    for i in range(len(per_frame_global_rigid)):
        torch.save(
            per_frame_global_rigid[i].detach(),
            os.path.join(out_path, f"per_frame_global_rigid_{i:05d}.pt"),
        )
    for i in range(1, len(per_frame_local_deform)):
        if isinstance(per_frame_local_deform[i], torch.nn.Module):
            torch.save(
                per_frame_local_deform[i].state_dict(),
                os.path.join(out_path, f"per_frame_local_deform_{i:05d}.pt"),
            )

    # Save model frame segments
    torch.save(model_frame_segments, os.path.join(out_path, "model_frame_segments.pt"))

    # Save convention metadata so downstream consumers know the parameterisation
    convention = {
        "variant": "c2w",
        "global_deform_is": "c2w",
        "local_deform_space": "camera",
        "description": (
            "per_frame_global_rigid contains full c2w SE3 twists (not corrections). "
            "local_deform operates in camera space. "
            "canonical = se3_apply(c2w_i, local_deform_i(camera_pts_i))."
        ),
    }
    with open(os.path.join(out_path, "convention.json"), "w") as _fconv:
        json.dump(convention, _fconv, indent=2)

    # Save original extrinsics for reference
    torch.save(
        [w2c.cpu() for w2c in original_extrinsics_w2c],
        os.path.join(out_path, "original_extrinsics_w2c.pt"),
    )

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    logger.info("Global optimization finished. Outputs written to %s", out_path)


if __name__ == "__main__":
    tyro.cli(main)
