"""
Stage 3.1: Train a view-conditioned inverse deformation network.

Learns to map canonical points back to per-view camera space (inverse of the
forward alignment from Stages 1+2). Used at Stage 3.2 to warp Gaussians into
each view's coordinate frame for supervision.
"""

from __future__ import annotations

import json
import os

from dataclasses import replace
from typing import Optional, TYPE_CHECKING

import numpy as np
import open3d as o3d
import torch
import tyro
from tqdm.auto import tqdm

from configs.stage3_inverse_deformation import TrainInverseDeformationConfig
from models.deformation import FullInverseDeformationModel
from data.checkpoint_loading import (
    AlignmentDataParams,
    load_alignment_data_params,
    load_deformation_checkpoints as _load_deformation_checkpoints_impl,
)
from data.data_loading import load_data
from utils.knn import (
    build_kdtree,
    build_torch_kdtree,
    query_knn_with_backend,
)
from utils.logging import get_logger, try_create_tensorboard_writer, tb_log_hparams

logger = get_logger(__name__)

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


# ---------------------------
# Training Data Generation
# ---------------------------
def generate_forward_pairs(
    model: FullInverseDeformationModel,
    per_view_cam_points: list[torch.Tensor],
    num_samples_per_view: int = 10000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate exact forward pairs by sampling camera-space points from each view
    and warping them to canonical space.

    Returns:
        cam_pts: (N, 3) original camera-space points
        canonical_pts: (N, 3) forward-warped canonical-space points
        view_indices: (N,) view index for each point
    """
    all_cam_pts = []
    all_canonical_pts = []
    all_view_indices = []

    device = per_view_cam_points[0].device

    for view_idx, pts in enumerate(per_view_cam_points):
        n_pts = pts.shape[0]
        if n_pts > num_samples_per_view:
            indices = torch.randperm(n_pts, device=device)[:num_samples_per_view]
            sampled_pts = pts[indices]
        else:
            sampled_pts = pts

        with torch.no_grad():
            canonical_pts = model.forward_deform(sampled_pts, view_idx)

        all_cam_pts.append(sampled_pts)
        all_canonical_pts.append(canonical_pts)
        all_view_indices.append(torch.full((sampled_pts.shape[0],), view_idx, device=device, dtype=torch.long))

    return (
        torch.cat(all_cam_pts, dim=0),
        torch.cat(all_canonical_pts, dim=0),
        torch.cat(all_view_indices, dim=0),
    )


def generate_interpolated_samples(
    model: FullInverseDeformationModel,
    per_view_cam_points: list[torch.Tensor],
    num_samples_per_view: int = 5000,
    k_neighbors: int = 8,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Generate interpolated interior samples in camera space,
    then forward-warp to canonical space.
    """
    from scipy.spatial import cKDTree

    all_interp_cam_pts = []
    all_interp_canonical_pts = []
    all_view_indices = []

    device = per_view_cam_points[0].device

    for view_idx, pts in enumerate(per_view_cam_points):
        pts_np = pts.detach().cpu().numpy()
        n_pts = pts.shape[0]

        if n_pts < k_neighbors + 1:
            continue

        tree = cKDTree(pts_np)

        num_anchors = min(num_samples_per_view, n_pts)
        anchor_indices = torch.randperm(n_pts, device=device)[:num_anchors]
        anchor_pts = pts[anchor_indices]

        _, neighbor_indices = tree.query(anchor_pts.cpu().numpy(), k=k_neighbors + 1)
        neighbor_indices = torch.from_numpy(neighbor_indices[:, 1:]).to(device)

        random_neighbor_idx = torch.randint(0, k_neighbors, (num_anchors,), device=device)
        selected_neighbors = neighbor_indices[torch.arange(num_anchors, device=device), random_neighbor_idx]
        neighbor_pts = pts[selected_neighbors]

        t = torch.rand(num_anchors, 1, device=device)
        interp_pts = anchor_pts * (1 - t) + neighbor_pts * t

        with torch.no_grad():
            interp_canonical = model.forward_deform(interp_pts, view_idx)

        all_interp_cam_pts.append(interp_pts)
        all_interp_canonical_pts.append(interp_canonical)
        all_view_indices.append(torch.full((interp_pts.shape[0],), view_idx, device=device, dtype=torch.long))

    if len(all_interp_cam_pts) == 0:
        return None, None, None

    return (
        torch.cat(all_interp_cam_pts, dim=0),
        torch.cat(all_interp_canonical_pts, dim=0),
        torch.cat(all_view_indices, dim=0),
    )


# ---------------------------
# Loss Functions
# ---------------------------
def compute_inverse_warp_loss(
    model: FullInverseDeformationModel,
    canonical_pts: torch.Tensor,
    target_cam_pts: torch.Tensor,
    view_indices: torch.Tensor,
) -> torch.Tensor:
    """
    L2 loss: inverse-local should map canonical → camera_corrected ≈ target_cam_pts.
    """
    pred_cam_pts = model.inverse_deform_to_camera(canonical_pts, view_indices)
    loss = ((pred_cam_pts - target_cam_pts) ** 2).sum(dim=-1).mean()
    return loss


def compute_cycle_consistency_loss(
    model: FullInverseDeformationModel,
    canonical_pts: torch.Tensor,
    view_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Cycle consistency: forward(inverse_to_camera(canonical)) ≈ canonical.

    Path: canonical → c2w^{-1} → inv_local → local → c2w → should ≈ canonical
    """
    pred_cam_pts = model.inverse_deform_to_camera(canonical_pts, view_indices)

    unique_views = torch.unique(view_indices)
    cycle_loss = torch.tensor(0.0, device=canonical_pts.device)
    total_pts = 0

    for view_idx in unique_views:
        mask = view_indices == view_idx
        cam_pts = pred_cam_pts[mask]
        original_canonical = canonical_pts[mask]

        # Forward: cam_pts → local → c2w → canonical (delegate to model helper)
        reconstructed_canonical = model.forward_deform(cam_pts, view_idx.item())

        cycle_loss = cycle_loss + ((reconstructed_canonical - original_canonical) ** 2).sum(dim=-1).sum()
        total_pts += cam_pts.shape[0]

    return cycle_loss / max(total_pts, 1)


def compute_twist_magnitude_loss(
    model: FullInverseDeformationModel,
    canonical_pts: torch.Tensor,
    view_indices: torch.Tensor,
) -> torch.Tensor:
    """Regularisation on SE(3) twist magnitude."""
    xi_inv = model.get_inverse_twist(canonical_pts, view_indices)
    return (xi_inv**2).sum(dim=-1).mean()


def compute_spatial_smoothness_loss(
    model: FullInverseDeformationModel,
    canonical_pts: torch.Tensor,
    view_indices: torch.Tensor,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Spatial smoothness: nearby points should have similar SE(3) twists."""
    perturbations = torch.randn_like(canonical_pts) * epsilon
    perturbed_pts = canonical_pts + perturbations
    perturbed_pts = torch.clamp(perturbed_pts, model.bbox_min, model.bbox_max)

    xi_orig = model.get_inverse_twist(canonical_pts, view_indices)
    xi_perturbed = model.get_inverse_twist(perturbed_pts, view_indices)

    diff = xi_orig - xi_perturbed
    return (diff**2).sum(dim=-1).mean()


# ---------------------------
# Training Loop
# ---------------------------
def train_inverse_deformation(
    model: FullInverseDeformationModel,
    per_view_cam_points: list[torch.Tensor],
    n_epochs: int = 100,
    batch_size: int = 8192,
    lr: float = 1e-3,
    cycle_weight: float = 0.1,
    magnitude_weight: float = 1e-3,
    smoothness_weight: float = 1e-3,
    num_forward_samples: int = 10000,
    num_interp_samples: int = 5000,
    regenerate_every: int = 10,
    writer=None,
    log_dir: Optional[str] = None,
) -> FullInverseDeformationModel:
    """Train the inverse deformation model."""
    device = per_view_cam_points[0].device

    if writer is None and log_dir is not None:
        writer = try_create_tensorboard_writer(log_dir)

    optimizer = torch.optim.Adam(model.inverse_local.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    global_step = 0

    # Generate initial training data
    logger.info("Generating initial training data...")
    cam_pts, canonical_pts, view_indices = generate_forward_pairs(model, per_view_cam_points, num_forward_samples)
    interp_cam_pts, interp_canonical_pts, interp_view_indices = generate_interpolated_samples(
        model, per_view_cam_points, num_interp_samples
    )

    if interp_cam_pts is not None:
        all_cam_pts = torch.cat([cam_pts, interp_cam_pts], dim=0)
        all_canonical_pts = torch.cat([canonical_pts, interp_canonical_pts], dim=0)
        all_view_indices = torch.cat([view_indices, interp_view_indices], dim=0)
    else:
        all_cam_pts = cam_pts
        all_canonical_pts = canonical_pts
        all_view_indices = view_indices

    logger.info("Total training samples: %d", all_cam_pts.shape[0])

    epoch_pbar = tqdm(range(n_epochs), desc="Training")

    for epoch in epoch_pbar:
        # Regenerate data periodically
        if epoch > 0 and epoch % regenerate_every == 0:
            logger.info("Regenerating training data at epoch %d...", epoch)
            cam_pts, canonical_pts, view_indices = generate_forward_pairs(
                model, per_view_cam_points, num_forward_samples
            )
            interp_cam_pts, interp_canonical_pts, interp_view_indices = generate_interpolated_samples(
                model, per_view_cam_points, num_interp_samples
            )
            if interp_cam_pts is not None:
                all_cam_pts = torch.cat([cam_pts, interp_cam_pts], dim=0)
                all_canonical_pts = torch.cat([canonical_pts, interp_canonical_pts], dim=0)
                all_view_indices = torch.cat([view_indices, interp_view_indices], dim=0)
            else:
                all_cam_pts = cam_pts
                all_canonical_pts = canonical_pts
                all_view_indices = view_indices

        # Shuffle
        perm = torch.randperm(all_cam_pts.shape[0], device=device)
        all_cam_pts = all_cam_pts[perm]
        all_canonical_pts = all_canonical_pts[perm]
        all_view_indices = all_view_indices[perm]

        epoch_loss = 0.0
        epoch_inverse_loss = 0.0
        epoch_cycle_loss = 0.0
        epoch_mag_loss = 0.0
        epoch_smooth_loss = 0.0
        num_batches = 0
        num_batches_total = (all_cam_pts.shape[0] + batch_size - 1) // batch_size

        batch_pbar = tqdm(
            range(0, all_cam_pts.shape[0], batch_size),
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            total=num_batches_total,
            leave=False,
        )

        for batch_idx, i in enumerate(batch_pbar):
            b_cam = all_cam_pts[i : i + batch_size]
            b_canonical = all_canonical_pts[i : i + batch_size]
            b_views = all_view_indices[i : i + batch_size]

            optimizer.zero_grad()

            loss_inverse = compute_inverse_warp_loss(model, b_canonical, b_cam, b_views)
            loss_cycle = compute_cycle_consistency_loss(model, b_canonical, b_views)
            loss_magnitude = compute_twist_magnitude_loss(model, b_canonical, b_views)
            loss_smoothness = compute_spatial_smoothness_loss(model, b_canonical, b_views)

            loss = (
                loss_inverse
                + cycle_weight * loss_cycle
                + magnitude_weight * loss_magnitude
                + smoothness_weight * loss_smoothness
            )

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_inverse_loss += loss_inverse.item()
            epoch_cycle_loss += loss_cycle.item()
            epoch_mag_loss += loss_magnitude.item()
            epoch_smooth_loss += loss_smoothness.item()
            num_batches += 1

            if (batch_idx + 1) % 1 == 0 or batch_idx == num_batches_total - 1:
                running_avg = epoch_loss / num_batches
                batch_pbar.set_postfix(
                    {
                        "loss": f"{running_avg:.6f}",
                        "inv": f"{epoch_inverse_loss / num_batches:.6f}",
                        "cycle": f"{epoch_cycle_loss / num_batches:.6f}",
                    }
                )

            if writer is not None:
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/loss_inverse", loss_inverse.item(), global_step)
                writer.add_scalar("train/loss_cycle", loss_cycle.item(), global_step)
                writer.add_scalar("train/loss_magnitude", loss_magnitude.item(), global_step)
                writer.add_scalar("train/loss_smoothness", loss_smoothness.item(), global_step)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)

            global_step += 1

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        epoch_pbar.set_postfix(
            {
                "loss": f"{avg_loss:.6f}",
                "inv": f"{epoch_inverse_loss / num_batches:.6f}",
                "cycle": f"{epoch_cycle_loss / num_batches:.6f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )

        if writer is not None:
            writer.add_scalar("epoch/loss", avg_loss, epoch)
            writer.add_scalar("epoch/learning_rate", scheduler.get_last_lr()[0], epoch)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            logger.info(
                "Epoch %d/%d | Loss: %.6f | Inverse: %.6f | Cycle: %.6f | Mag: %.6f | Smooth: %.6f | LR: %.2e",
                epoch + 1,
                n_epochs,
                avg_loss,
                epoch_inverse_loss / num_batches,
                epoch_cycle_loss / num_batches,
                epoch_mag_loss / num_batches,
                epoch_smooth_loss / num_batches,
                scheduler.get_last_lr()[0],
            )

    epoch_pbar.close()
    return model


@torch.no_grad()
def validate_roundtrip_per_view(
    model: FullInverseDeformationModel,
    per_view_cam_points: list[torch.Tensor],
    per_view_world_points: list[torch.Tensor],
    out_path: str,
    writer: SummaryWriter | None = None,
    save_plys: bool = True,
    knn_backend: str = "cpu_kdtree",
) -> dict[str, float]:
    """
    Round-trip validation:
        cam_pts → forward_deform → canonical → inverse_deform_to_camera → cam_pts_hat

    Additionally saves:
        - input points in world/canonical space
        - inverse-deformed canonical points (canonical space without local deformation)
    """
    model.eval()
    val_dir = os.path.join(out_path, "validation_roundtrip")
    os.makedirs(val_dir, exist_ok=True)

    rmses_direct: list[float] = []
    rmses_nn: list[float] = []

    logger.info("Running round-trip per-view validation (saving PLYs: %s)...", save_plys)

    for view_idx, (cam_pts, world_pts) in enumerate(
        tqdm(
            zip(per_view_cam_points, per_view_world_points),
            desc="Validation",
            leave=False,
        )
    ):
        if cam_pts.numel() == 0:
            continue

        canonical_pts = model.forward_deform(cam_pts, view_idx)
        view_idx_tensor = torch.full(
            (canonical_pts.shape[0],),
            view_idx,
            device=canonical_pts.device,
            dtype=torch.long,
        )
        cam_pts_hat = model.inverse_deform_to_camera(canonical_pts, view_idx_tensor)

        # Canonical-space roundtrip without local deformation:
        # canonical → inverse_deform → canonical_no_local
        canonical_pts_no_local = model.inverse_deform(canonical_pts, view_idx_tensor)

        direct_rmse = torch.sqrt(((cam_pts_hat - cam_pts) ** 2).sum(dim=-1).mean()).item()

        if knn_backend == "cpu_kdtree":
            tree = build_kdtree(cam_pts)
        elif knn_backend == "gpu_kdtree":
            tree = build_torch_kdtree(cam_pts)
        else:
            tree = None
        _, d2 = query_knn_with_backend(
            cam_pts_hat,
            cam_pts,
            K=1,
            backend=knn_backend,
            cpu_tree=tree if knn_backend == "cpu_kdtree" else None,
            gpu_tree=tree if knn_backend == "gpu_kdtree" else None,
        )
        nn_rmse = torch.sqrt(d2.mean()).item()

        rmses_direct.append(direct_rmse)
        rmses_nn.append(nn_rmse)

        if writer is not None:
            writer.add_scalar("validation_roundtrip/per_view_direct_rmse", direct_rmse, view_idx)
            writer.add_scalar("validation_roundtrip/per_view_nn_rmse", nn_rmse, view_idx)

        if save_plys:
            # Keep lightweight Open3D saves for debugging; no shared helper needed.
            def _write_pcd(pts: torch.Tensor, out_file: str, color: tuple[float, float, float]) -> None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy().reshape(-1, 3))
                pcd.paint_uniform_color(list(color))
                o3d.io.write_point_cloud(out_file, pcd)

            _write_pcd(
                cam_pts,
                os.path.join(val_dir, f"view_{view_idx:05d}_input_cam.ply"),
                (1.0, 0.0, 0.0),
            )
            _write_pcd(
                canonical_pts,
                os.path.join(val_dir, f"view_{view_idx:05d}_canonical.ply"),
                (0.0, 1.0, 0.0),
            )
            _write_pcd(
                cam_pts_hat,
                os.path.join(val_dir, f"view_{view_idx:05d}_roundtrip_cam.ply"),
                (0.0, 0.0, 1.0),
            )
            _write_pcd(
                world_pts,
                os.path.join(val_dir, f"view_{view_idx:05d}_input_world.ply"),
                (1.0, 1.0, 0.0),
            )
            _write_pcd(
                canonical_pts_no_local,
                os.path.join(val_dir, f"view_{view_idx:05d}_canonical_no_local.ply"),
                (1.0, 0.0, 1.0),
            )

        logger.info(
            "Val view %d | direct RMSE: %.6e | nn RMSE: %.6e | N=%d",
            view_idx,
            direct_rmse,
            nn_rmse,
            cam_pts.shape[0],
        )

    if len(rmses_direct) == 0:
        return {
            k: float("nan")
            for k in [
                "direct_rmse_mean",
                "direct_rmse_median",
                "direct_rmse_max",
                "nn_rmse_mean",
                "nn_rmse_median",
                "nn_rmse_max",
            ]
        }

    direct_arr = np.asarray(rmses_direct, dtype=np.float64)
    nn_arr = np.asarray(rmses_nn, dtype=np.float64)
    metrics = {
        "direct_rmse_mean": float(np.mean(direct_arr)),
        "direct_rmse_median": float(np.median(direct_arr)),
        "direct_rmse_max": float(np.max(direct_arr)),
        "nn_rmse_mean": float(np.mean(nn_arr)),
        "nn_rmse_median": float(np.median(nn_arr)),
        "nn_rmse_max": float(np.max(nn_arr)),
    }
    logger.info(
        "Round-trip validation summary | "
        "direct RMSE mean/median/max: %.6e / %.6e / %.6e | "
        "nn RMSE mean/median/max: %.6e / %.6e / %.6e",
        metrics["direct_rmse_mean"],
        metrics["direct_rmse_median"],
        metrics["direct_rmse_max"],
        metrics["nn_rmse_mean"],
        metrics["nn_rmse_median"],
        metrics["nn_rmse_max"],
    )

    if writer is not None:
        for k, v in metrics.items():
            writer.add_scalar(f"validation_roundtrip/summary_{k}", v, 0)

    return metrics


# ---------------------------
# Main Entry Point
# ---------------------------
def main(config: TrainInverseDeformationConfig):
    """Main training function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Sentinel defaults: allow higher-level pipelines to decide presets, but keep
    # this stage's standalone CLI behavior unchanged.
    if config.n_epochs is None:
        config = replace(config, n_epochs=30)

    checkpoint_dir = os.path.join(config.root_path, config.run, config.checkpoint_subdir)
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Validate convention
    convention_path = os.path.join(checkpoint_dir, "convention.json")
    if os.path.exists(convention_path):
        with open(convention_path, "r") as f:
            conv = json.load(f)
        if conv.get("global_deform_is") != "c2w":
            raise ValueError(f"Expected c2w convention (global_deform_is='c2w'), got: {conv}")
        logger.info("Convention verified: c2w (global_deform_is=c2w)")
    else:
        logger.warning("No convention.json found — proceeding assuming c2w convention")

    # Load deformation checkpoints
    logger.info("Loading checkpoints from: %s", checkpoint_dir)
    per_frame_global_deform, per_frame_local_deform, bbox_min, bbox_max = _load_deformation_checkpoints_impl(
        checkpoint_dir,
        device,
        first_local="none",
        allow_rigid_fallback=True,
    )

    num_views = len(per_frame_global_deform)
    logger.info("Number of views: %d", num_views)
    logger.info("Bounding box: %s to %s", bbox_min.cpu().numpy(), bbox_max.cpu().numpy())

    # Load data configuration used during the original alignment run so that
    # confidence filtering and frame sampling are identical to the checkpoint
    # that produced these deformations.
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
        "Using alignment data params for inverse deformation: "
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

    # Load per-frame point clouds and convert to camera space
    logger.info("Loading per-frame point clouds...")
    (
        pcls,
        extrinsics,
        intrinsics,
        images,
        _valid_pixel_indices,
        _depth_conf,
        _depth_maps,
        _orig_images,
        _orig_intrinsics,
    ) = load_data(
        config.root_path,
        num_frames=align_params.num_frames,
        stride=align_params.stride,
        device=device,
        **load_data_kwargs,
    )

    per_view_cam_points = []
    per_view_world_points = []
    for i, pcl in enumerate(pcls[:num_views]):
        world_pts = torch.from_numpy(np.array(pcl.points)).to(device).float()
        # Convert world → camera using original extrinsics
        w2c = torch.from_numpy(extrinsics[i]).to(device).float()
        if w2c.shape == (3, 4):
            w2c_4x4 = torch.eye(4, device=device)
            w2c_4x4[:3, :4] = w2c
            w2c = w2c_4x4
        cam_pts = (w2c[:3, :3] @ world_pts.T).T + w2c[:3, 3]
        per_view_cam_points.append(cam_pts)
        per_view_world_points.append(world_pts)

    logger.info("Loaded %d point clouds (camera space)", len(per_view_cam_points))

    # Create the model
    model = FullInverseDeformationModel(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        num_views=num_views,
        per_view_global_deform=per_frame_global_deform,
        per_view_local_deform=per_frame_local_deform,
        view_embed_dim=config.view_embed_dim,
        min_res=config.min_res,
        max_res=config.max_res,
        num_levels=config.num_levels,
        log2_hashmap_size=config.log2_hashmap_size,
        n_neurons=config.n_neurons,
        n_hidden_layers=config.n_hidden_layers,
    ).to(device)

    logger.info("Created inverse deformation model")

    # Output directory
    if config.out_path is None:
        out_path = os.path.join(
            os.path.dirname(checkpoint_dir),
            "inverse_deformation",
        )
    else:
        out_path = config.out_path

    os.makedirs(out_path, exist_ok=True)
    tb_log_dir = os.path.join(out_path, "tensorboard")
    writer = None
    if config.tensorboard:
        writer = try_create_tensorboard_writer(tb_log_dir)
        if writer is not None:
            logger.info("TensorBoard logs: %s", tb_log_dir)
            tb_log_hparams(
                writer,
                {
                    "root_path": config.root_path,
                    "run": config.run,
                    "checkpoint_subdir": config.checkpoint_subdir,
                    "num_views": num_views,
                    "view_embed_dim": config.view_embed_dim,
                    "min_res": config.min_res,
                    "max_res": config.max_res,
                    "num_levels": config.num_levels,
                    "log2_hashmap_size": config.log2_hashmap_size,
                    "n_neurons": config.n_neurons,
                    "n_hidden_layers": config.n_hidden_layers,
                    "n_epochs": config.n_epochs,
                    "batch_size": config.batch_size,
                    "lr": config.lr,
                    "cycle_weight": config.cycle_weight,
                    "magnitude_weight": config.magnitude_weight,
                    "smoothness_weight": config.smoothness_weight,
                },
                step=0,
            )

    # Train
    logger.info("Starting training...")
    model = train_inverse_deformation(
        model=model,
        per_view_cam_points=per_view_cam_points,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        cycle_weight=config.cycle_weight,
        magnitude_weight=config.magnitude_weight,
        smoothness_weight=config.smoothness_weight,
        num_forward_samples=config.num_forward_samples,
        num_interp_samples=config.num_interp_samples,
        regenerate_every=config.regenerate_every,
        writer=writer,
        log_dir=tb_log_dir,
    )

    # Save
    torch.save(model.inverse_local.state_dict(), os.path.join(out_path, "inverse_local.pt"))

    torch.save(
        {
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "num_views": num_views,
            "view_embed_dim": config.view_embed_dim,
            "min_res": config.min_res,
            "max_res": config.max_res,
            "num_levels": config.num_levels,
            "log2_hashmap_size": config.log2_hashmap_size,
            "n_neurons": config.n_neurons,
            "n_hidden_layers": config.n_hidden_layers,
            "convention": "c2w",
        },
        os.path.join(out_path, "config.pt"),
    )

    # Save convention metadata
    with open(os.path.join(out_path, "convention.json"), "w") as f:
        json.dump(
            {
                "variant": "c2w",
                "global_deform_is": "c2w",
                "local_deform_space": "camera",
            },
            f,
            indent=2,
        )

    logger.info("Saved trained model to: %s", out_path)

    # Validation
    logger.info("Running validation...")
    model.eval()
    with torch.no_grad():
        cam_pts, canonical_pts, view_indices = generate_forward_pairs(
            model,
            per_view_cam_points,
            num_samples_per_view=5000,
        )
        pred_cam_pts = model.inverse_deform_to_camera(canonical_pts, view_indices)
        error = torch.sqrt(((pred_cam_pts - cam_pts) ** 2).sum(dim=-1))
        logger.info(
            "Validation error — Mean: %.6f, Median: %.6f, Max: %.6f",
            error.mean().item(),
            error.median().item(),
            error.max().item(),
        )
        if writer is not None:
            writer.add_scalar("validation/error_mean", error.mean().item(), 0)
            writer.add_scalar("validation/error_median", error.median().item(), 0)
            writer.add_scalar("validation/error_max", error.max().item(), 0)

    _ = validate_roundtrip_per_view(
        model=model,
        per_view_cam_points=per_view_cam_points,
        per_view_world_points=per_view_world_points,
        out_path=out_path,
        writer=writer,
        save_plys=config.save_validation_plys,
        knn_backend=config.knn_backend,
    )

    if writer is not None:
        writer.close()

    return model


if __name__ == "__main__":
    tyro.cli(main)
