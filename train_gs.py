"""
Stage 3.2: Train a 2DGS or 3DGS model on the aligned canonical point cloud.

Initialises SH0 from canonical RGB and renders via gsplat.
The ``--config.renderer`` flag selects between 2DGS and 3DGS backends.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
import subprocess
import sys

import cv2
import numpy as np
import open3d as o3d
import torch
import tyro
from tqdm.auto import tqdm

from configs.stage3_gs import GSConfig
from data.checkpoint_loading import (
    AlignmentDataParams,
    load_alignment_data_params,
    load_aligned_point_cloud,
    load_deformation_checkpoints,
    load_inverse_local_from_checkpoint,
)
from data.data_loading import load_data, load_da3_original_images_from_folder
from losses import init_lpips
from models.canonical_gs_model import CanonicalGSModel, sh0_to_rgb
from utils.export_checkpoint_to_ply import (
    ExportGSCheckpointToPlyConfig,
    main as export_gs_checkpoint_to_ply,
)
from utils.downsample import downsample_to_target
from utils.logging import get_logger, try_create_tensorboard_writer, tb_log_hparams
from utils.normals import estimate_normals
from utils.knn import query_knn_with_backend

logger = get_logger(__name__)

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


# ==========================================================================
# Main
# ==========================================================================


def main(config: GSConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load data & checkpoint
    # ------------------------------------------------------------------
    nrba_dir = os.path.join(config.root_path, config.run, config.global_opt_subdir)
    if not os.path.exists(nrba_dir):
        raise FileNotFoundError(f"Global optimization directory not found: {nrba_dir}")

    convention_path = os.path.join(nrba_dir, "convention.json")
    if os.path.exists(convention_path):
        with open(convention_path) as f:
            conv = json.load(f)
        if conv.get("global_deform_is") != "c2w":
            raise ValueError(f"Expected c2w convention, got: {conv}")
        logger.info("Convention verified: c2w")
    else:
        logger.warning("No convention.json — assuming c2w convention")

    logger.info("Loading aligned point cloud...")
    canonical_pts, canonical_cols = load_aligned_point_cloud(nrba_dir, device)
    logger.info("Loaded %d canonical points", canonical_pts.shape[0])

    if canonical_pts.shape[0] > config.target_num_points:
        logger.info(
            "Downsampling from %d to ~%d points...",
            canonical_pts.shape[0],
            config.target_num_points,
        )
        canonical_pts, canonical_cols = downsample_to_target(
            canonical_pts,
            canonical_cols,
            target_count=config.target_num_points,
        )
        logger.info("After downsampling: %d points", canonical_pts.shape[0])

    per_frame_global_deform, per_frame_local_deform, bbox_min, bbox_max = load_deformation_checkpoints(
        nrba_dir,
        device,
        first_local="none",
        allow_rigid_fallback=True,
    )
    num_deform_frames = len(per_frame_global_deform)

    # Reuse the exact data loading / confidence-filtering configuration from the
    # original frame_to_model_icp Stage 1 run so that Stage 3.2 is consistent
    # with the alignment stage.
    align_params: AlignmentDataParams = load_alignment_data_params(
        root_path=config.root_path,
        run=config.run,
    )

    load_data_kwargs: dict = dict(
        conf_thresh_percentile=align_params.conf_thresh_percentile,
        conf_mode=align_params.conf_mode,
        conf_local_percentile=align_params.conf_local_percentile,
        conf_global_percentile=align_params.conf_global_percentile,
        voxel_size=align_params.conf_voxel_size,
        voxel_min_count_percentile=align_params.conf_voxel_min_count_percentile,
        offset=align_params.offset,
    )
    logger.info(
        "Using alignment data params for GS training: "
        "num_frames=%d, stride=%d, offset=%d, conf_thresh_percentile=%.1f, "
        "conf_mode=%s, conf_local_percentile=%s, conf_global_percentile=%s, "
        "conf_voxel_size=%.4f, conf_voxel_min_count_percentile=%s",
        align_params.num_frames,
        align_params.stride,
        align_params.offset,
        align_params.conf_thresh_percentile,
        align_params.conf_mode,
        str(align_params.conf_local_percentile),
        str(align_params.conf_global_percentile),
        align_params.conf_voxel_size,
        str(align_params.conf_voxel_min_count_percentile),
    )

    (
        pcls,
        extrinsics_np,
        intrinsics_np,
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
        **load_data_kwargs,
        load_original_images_and_intrinsics=False,
    )
    if config.original_images_dir:
        folder_images, folder_intrinsics = load_da3_original_images_from_folder(
            config.root_path,
            config.original_images_dir,
            num_frames=align_params.num_frames,
            stride=align_params.stride,
            device=device,
        )
        images = folder_images
        intrinsics_np = folder_intrinsics
    num_frames = min(len(pcls), num_deform_frames)
    H, W = images.shape[2], images.shape[3]
    logger.info("Loaded %d frames (%d x %d)", num_frames, H, W)

    intrinsics_list: list[torch.Tensor] = []
    for i in range(num_frames):
        intrinsics_list.append(torch.from_numpy(intrinsics_np[i]).to(device).float())

    gt_images = images[:num_frames]  # (N, 3, H, W)

    if not config.inverse_deform_dir:
        raise ValueError("--config.inverse-deform-dir is required")
    inverse_deform_net, _inv_cfg = load_inverse_local_from_checkpoint(config.inverse_deform_dir, device=device)
    inverse_deform_net.eval()
    for p in inverse_deform_net.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    init_normals = None
    knn_dists = None
    kdtree = None

    if config.normal_k > 0:
        logger.info(
            "Estimating normals (k=%d) for %d canonical points...",
            config.normal_k,
            canonical_pts.shape[0],
        )
        init_normals, kdtree = estimate_normals(
            canonical_pts,
            k=config.normal_k,
            backend="cpu_kdtree",
        )
        init_normals = init_normals.cpu()
        logger.info("Normal estimation done.")

    if config.scale_init == "knn":
        logger.info(
            "Computing KNN distances (K=%d) for %d points...",
            config.knn_neighbors,
            canonical_pts.shape[0],
        )
        _, d2 = query_knn_with_backend(
            canonical_pts,
            canonical_pts,
            K=config.knn_neighbors,
            backend="cpu_kdtree",
            cpu_tree=kdtree,
        )
        knn_dists = d2[:, 1:].mean(dim=-1).sqrt().cpu()
        logger.info(
            "KNN dists: median=%.5f, min=%.5f, max=%.5f",
            knn_dists.median().item(),
            knn_dists.min().item(),
            knn_dists.max().item(),
        )

    model = CanonicalGSModel(
        canonical_points=canonical_pts,
        canonical_colors=canonical_cols,
        per_frame_global_deform=per_frame_global_deform[:num_frames],
        per_frame_local_deform=per_frame_local_deform[:num_frames],
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        height=H,
        width=W,
        renderer=config.renderer,
        optimize_cams=config.optimize_cams,
        optimize_positions=config.optimize_positions,
        deform_rotations=config.deform_inverse_rotations,
        initial_opacity=config.initial_opacity,
        initial_scale=config.initial_scale,
        initial_flat_ratio=config.initial_flat_ratio,
        near_plane=0.01,
        far_plane=1e10,
        inverse_deform_net=inverse_deform_net,
        knn_dists=knn_dists,
        init_normals=init_normals,
        sh_degree=config.sh_degree,
    ).to(device)

    # ------------------------------------------------------------------
    # 3. Optimiser
    # ------------------------------------------------------------------
    param_groups = []

    param_groups.append(
        {"params": [model.sh_dc], "lr": config.lr_sh0, "name": "sh0"},
    )
    if model.sh_rest is not None:
        param_groups.append(
            {"params": [model.sh_rest], "lr": config.lr_shN, "name": "shN"},
        )
    param_groups += [
        {
            "params": [model.logit_opacities],
            "lr": config.lr_opacities,
            "name": "opacities",
        },
        {"params": [model.log_scales], "lr": config.lr_scales, "name": "scales"},
        {"params": [model.quats], "lr": config.lr_quats, "name": "quats"},
    ]

    if config.optimize_positions:
        param_groups.append(
            {
                "params": [model.canonical_points],
                "lr": config.lr_positions,
                "name": "positions",
            },
        )

    if config.optimize_cams:
        cam_params = [p for p in model.per_frame_c2w.parameters() if p.requires_grad]
        if cam_params:
            param_groups.append({"params": cam_params, "lr": config.lr_cams, "name": "cams"})

    optimizer = torch.optim.Adam(param_groups) if param_groups else None
    scheduler = None
    if optimizer is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_iters,
            eta_min=config.lr_sh0 * config.lr_decay,
        )

    lpips_fn = init_lpips(device) if config.lpips_weight > 0 else None

    # ------------------------------------------------------------------
    # 4. Output & logging
    # ------------------------------------------------------------------
    if config.out_dir is None:
        config.out_dir = os.path.join(
            config.root_path,
            config.run,
            f"gs_{config.renderer}",
        )
    os.makedirs(config.out_dir, exist_ok=True)

    writer = None
    if config.tensorboard:
        tb_dir = os.path.join(config.out_dir, "tensorboard")
        writer = try_create_tensorboard_writer(tb_dir)

    with open(os.path.join(config.out_dir, "config.json"), "w") as f:
        json.dump(
            {
                k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                for k, v in vars(config).items()
            },
            f,
            indent=2,
        )

    logger.info("Output directory: %s", config.out_dir)
    logger.info("Renderer: %s, sh_degree=%d", config.renderer, config.sh_degree)
    logger.info("Trainable parameters:")
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        logger.info("  %s: %d params, lr=%.2e", pg["name"], n_params, pg["lr"])

    if writer is not None:
        tb_log_hparams(
            writer,
            {
                "root_path": config.root_path,
                "run": config.run,
                "renderer": config.renderer,
                "sh_degree": config.sh_degree,
                "target_num_points": config.target_num_points,
                "optimize_cams": config.optimize_cams,
                "optimize_positions": config.optimize_positions,
                "lr_colors": config.lr_colors,
                "lr_opacities": config.lr_opacities,
                "lr_scales": config.lr_scales,
                "lr_quats": config.lr_quats,
                "lr_sh0": config.lr_sh0,
                "lr_shN": config.lr_shN,
                "num_iters": config.num_iters,
                "frames_per_iter": config.frames_per_iter,
                "num_frames": align_params.num_frames,
                "stride": align_params.stride,
                "conf_thresh_percentile": align_params.conf_thresh_percentile,
            },
            step=0,
        )

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    is_2dgs = config.renderer == "2dgs"
    model.train()
    pbar = tqdm(range(config.num_iters), desc=f"GS training ({config.renderer})")

    for it in pbar:
        if config.sh_full_from_iter > 0 and it == config.sh_full_from_iter:
            for p in model.per_frame_c2w.parameters():
                p.requires_grad = False
            if config.sh_freeze_means_when_full_sh:
                model.canonical_points.requires_grad = False
            logger.info(
                "At iter %d: froze cams (and means=%s), full SH enabled.",
                it,
                config.sh_freeze_means_when_full_sh,
            )

        optimizer.zero_grad()

        frame_indices = torch.randint(0, num_frames, (config.frames_per_iter,))
        total_loss = torch.tensor(0.0, device=device)
        render_result = None
        sh_reg_loss = torch.tensor(0.0, device=device)

        for fi in frame_indices:
            fi = fi.item()
            K = intrinsics_list[fi]

            render_result = model.render_frame(fi, K, use_inverse_deform=True)
            rendered_rgb = render_result["rgb"]
            gt = gt_images[fi : fi + 1]

            loss_l1 = (rendered_rgb - gt).abs().mean()
            loss_lpips = torch.tensor(0.0, device=device)
            if lpips_fn is not None and config.lpips_weight > 0:
                loss_lpips = lpips_fn(
                    rendered_rgb.clamp(0, 1) * 2 - 1,
                    gt.clamp(0, 1) * 2 - 1,
                ).mean()

            frame_loss = config.l1_weight * loss_l1 + config.lpips_weight * loss_lpips

            # ---------- Regularisation ----------
            reg_loss = torch.tensor(0.0, device=device)

            if config.scale_reg_weight > 0:
                scales = model.log_scales.exp()
                reg_loss = reg_loss + config.scale_reg_weight * scales.mean()

            if is_2dgs:
                if config.opacity_reg_weight > 0:
                    opacities = torch.sigmoid(model.logit_opacities)
                    reg_loss = reg_loss + config.opacity_reg_weight * (opacities * (1 - opacities)).mean()

                if config.normal_consistency_weight > 0 and render_result is not None and "normals" in render_result:
                    from losses.gaussian import normal_consistency_loss

                    normals = render_result["normals"]
                    surf_normals = render_result["surf_normals"]
                    alphas = render_result.get("alpha")
                    if normals is not None and surf_normals is not None and alphas is not None:
                        nc_loss = normal_consistency_loss(
                            normals,
                            surf_normals.unsqueeze(0),
                            alphas.permute(0, 2, 3, 1),
                        )
                        reg_loss = reg_loss + config.normal_consistency_weight * nc_loss

                if config.distortion_weight > 0 and render_result is not None and "distort" in render_result:
                    from losses import distortion_loss

                    distort = render_result["distort"]
                    if distort is not None:
                        dl = distortion_loss(distort)
                        reg_loss = reg_loss + config.distortion_weight * dl

                if config.alpha_reg_weight > 0 and render_result is not None and "alpha" in render_result:
                    alpha = render_result["alpha"]
                    if alpha is not None:
                        reg_loss = reg_loss + config.alpha_reg_weight * (1.0 - alpha).mean()

            if config.sh_reg_weight > 0.0 and hasattr(model, "sh_rest") and model.sh_rest is not None:
                sh_reg = (model.sh_rest**2).mean()
                sh_reg_loss = sh_reg_loss + sh_reg / config.frames_per_iter
                reg_loss = reg_loss + config.sh_reg_weight * sh_reg

            frame_loss = frame_loss + reg_loss
            total_loss = total_loss + frame_loss / config.frames_per_iter
        total_loss.backward()

        # SH gradient freezing schedule
        if (
            hasattr(model, "sh_rest")
            and model.sh_rest is not None
            and model.sh_rest.grad is not None
            and config.sh_degree > 0
        ):
            if config.sh_full_from_iter > 0 and it < config.sh_full_from_iter:
                model.sh_rest.grad.zero_()
            elif config.sh_increase_every > 0 and config.sh_full_from_iter == 0:
                L = int(config.sh_degree)
                total_rest = model.sh_rest.shape[1]
                if total_rest == L * (L + 2):
                    bands_unlocked = min(max(it // config.sh_increase_every, 0), L)
                    active_non_dc = bands_unlocked * (bands_unlocked + 2)
                    if active_non_dc < total_rest:
                        model.sh_rest.grad[:, active_non_dc:, :].zero_()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # ---------- Logging ----------
        if it % config.log_every == 0 or it == config.num_iters - 1:
            pbar.set_postfix({"loss": f"{total_loss.item():.5f}"})

            if writer is not None:
                writer.add_scalar("train/loss", total_loss.item(), it)
                if config.sh_reg_weight > 0.0:
                    writer.add_scalar("train/sh_reg_loss", sh_reg_loss.item(), it)
                if scheduler is not None:
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], it)

                if it % (config.log_every * 5) == 0 and render_result is not None:
                    with torch.no_grad():
                        vis_fi = 0
                        vis_K = intrinsics_list[vis_fi]
                        vis_result = model.render_frame(vis_fi, vis_K, use_inverse_deform=True)
                        writer.add_image("vis/rendered", vis_result["rgb"][0].clamp(0, 1), it)
                        writer.add_image("vis/gt", gt_images[vis_fi], it)

        # ---------- Save ----------
        if (it > 0 and it % config.save_every == 0) or it == config.num_iters - 1:
            ckpt_path = os.path.join(config.out_dir, f"checkpoint_{it:06d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

        # ---------- Eval ----------
        if (it > 0 and it % config.eval_every == 0) or it == config.num_iters - 1:
            _evaluate(
                model,
                gt_images,
                intrinsics_list,
                num_frames,
                config,
                it,
                writer,
                num_frames_to_save=5 if it < config.num_iters - 1 else 1000,
            )

    # ------------------------------------------------------------------
    # 6. Final save
    # ------------------------------------------------------------------
    final_path = os.path.join(config.out_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info("Training complete.  Final model: %s", final_path)

    final_pts = model.canonical_points.detach().cpu().numpy()
    if hasattr(model, "sh_coeffs") and model.sh_coeffs is not None:
        sh0 = model.sh_coeffs[:, 0, :]
        final_cols = sh0_to_rgb(sh0).detach().cpu().numpy().clip(0, 1)
    else:
        final_cols = sh0_to_rgb(model.sh_dc.detach()[:, 0, :]).cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pts)
    pcd.colors = o3d.utility.Vector3dVector(final_cols)
    o3d.io.write_point_cloud(os.path.join(config.out_dir, "canonical_points_final.ply"), pcd)

    if writer is not None:
        writer.close()

    # If we trained a 3DGS model, automatically export a 3DGS-compatible PLY
    # checkpoint alongside the final model for convenient downstream viewing.
    if config.renderer == "3dgs":
        try:
            export_cfg = ExportGSCheckpointToPlyConfig(
                root_path=config.root_path,
                run=config.run,
                checkpoint_dir=config.out_dir,
                global_opt_subdir=config.global_opt_subdir,
            )
            export_gs_checkpoint_to_ply(export_cfg)
        except Exception as exc:
            logger.error("Automatic 3DGS PLY export failed: %s", exc)

    logger.info("Done.")

    # ------------------------------------------------------------------
    # 9. Optional automatic evaluation
    # ------------------------------------------------------------------
    if config.auto_eval:
        cmd = [
            sys.executable,
            "-m",
            "eval_gs",
            "--config.root-path",
            config.root_path,
            "--config.run",
            config.run,
            "--config.checkpoint-dir",
            config.out_dir,
        ]
        if config.original_images_dir:
            cmd.extend(["--config.original-images-dir", config.original_images_dir])
        try:
            logger.info("Running automatic eval via: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
        except Exception as e:
            logger.error("Automatic eval failed: %s", e)


@torch.no_grad()
def _evaluate(
    model: CanonicalGSModel,
    gt_images: torch.Tensor,
    intrinsics_list: list[torch.Tensor],
    num_frames: int,
    config: GSConfig,
    global_step: int,
    writer: SummaryWriter | None,
    num_frames_to_save: int = 5,
):
    """Run eval on all frames and log PSNR."""
    model.eval()
    psnrs = []

    eval_dir = os.path.join(config.out_dir, f"eval_{global_step:06d}")
    os.makedirs(eval_dir, exist_ok=True)

    for fi in range(num_frames):
        K = intrinsics_list[fi]
        result = model.render_frame(fi, K, use_inverse_deform=True)
        rendered = result["rgb"]  # (1, 3, H, W)
        gt = gt_images[fi : fi + 1]

        mse = ((rendered.clamp(0, 1) - gt) ** 2).mean()
        psnr = -10.0 * torch.log10(mse.clamp(min=1e-10)).item()
        psnrs.append(psnr)

        if fi < num_frames_to_save:
            img_np = (rendered[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(eval_dir, f"rendered_{fi:05d}.png"),
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
            )
            gt_np = (gt[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(eval_dir, f"gt_{fi:05d}.png"),
                cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR),
            )

            rendered_canon = model.render_frame(fi, K, use_inverse_deform=False)["rgb"]
            rendered_canon_np = (rendered_canon[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(eval_dir, f"rendered_canon_{fi:05d}.png"),
                cv2.cvtColor(rendered_canon_np, cv2.COLOR_RGB2BGR),
            )

    avg_psnr = np.mean(psnrs)
    logger.info("Eval [step %d]: avg PSNR = %.2f dB", global_step, avg_psnr)

    if writer is not None:
        writer.add_scalar("eval/psnr_avg", avg_psnr, global_step)
        for fi, p in enumerate(psnrs):
            writer.add_scalar(f"eval/psnr_frame_{fi:03d}", p, global_step)

    model.train()


if __name__ == "__main__":
    tyro.cli(main)
