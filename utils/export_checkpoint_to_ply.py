from __future__ import annotations

import os

import numpy as np
import torch
import tyro
from plyfile import PlyData, PlyElement

from configs.utils import ExportGSCheckpointToPlyConfig
from data.data_loading import load_da3_camera_images
from eval_gs import EvalGSConfig, _build_model, _find_checkpoint_path
from utils.logging import get_logger


logger = get_logger(__name__)


@torch.no_grad()
def main(config: ExportGSCheckpointToPlyConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.out_ply is None:
        config.out_ply = os.path.join(config.checkpoint_dir, "splats_3dgs.ply")

    ckpt_path = _find_checkpoint_path(config.checkpoint_dir)
    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Need a rendering resolution to construct the model.
    images, _poses_c2w, intrinsics = load_da3_camera_images(
        config.root_path,
        num_frames=1,
        stride=1,
        device=device,
        use_original_images_and_intrinsics=False,
    )
    H = int(images.shape[2])
    W = int(images.shape[3])

    eval_cfg = EvalGSConfig(
        root_path=config.root_path,
        run=config.run,
        checkpoint_dir=config.checkpoint_dir,
        global_opt_subdir=config.global_opt_subdir,
        render_gs_video_path=False,
        render_input_poses=False,
        render_optimised_poses=False,
        save_images=False,
        save_video=False,
    )
    model = _build_model(eval_cfg, device, height=H, width=W)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Only 3DGS exports are supported (matches eval_3dgs_ply / view_3dgs format).
    if model.renderer_type != "3dgs":
        raise ValueError("export_checkpoint_to_ply currently supports only renderer='3dgs'.")
    if model.sh_coeffs is None or model.sh_degree <= 0:
        raise ValueError(
            "3DGS PLY export requires SH colors (sh_degree>0). Train with renderer='3dgs' and sh_degree>0."
        )

    pts = model.canonical_points.detach()  # (N, 3)
    sh_coeffs = model.sh_coeffs.detach()  # (N, K, 3)
    N, K, _ = sh_coeffs.shape

    # Optional subsampling
    if config.max_points > 0 and N > config.max_points:
        perm = torch.randperm(N, device=pts.device)[: config.max_points]
        pts = pts[perm]
        sh_coeffs = sh_coeffs[perm]
        N = pts.shape[0]

    # SH DC (first band) → f_dc_0..2
    sh0 = sh_coeffs[:, 0, :]  # (N, 3)

    # Higher-order SH bands (flattened as f_rest_0..)
    if K > 1:
        sh_rest = sh_coeffs[:, 1:, :].reshape(N, -1)  # (N, (K-1)*3)
        n_f_rest = sh_rest.shape[1]
    else:
        sh_rest = torch.zeros((N, 0), device=pts.device, dtype=pts.dtype)
        n_f_rest = 0

    # Scales and opacities in log/logit form (standard 3DGS convention).
    log_scales = model.log_scales.detach().reshape(N, 3)
    logit_opacities = model.logit_opacities.detach().reshape(N)
    quats = model.quats.detach().reshape(N, 4)

    # Build structured array for PLY vertex properties.
    dtype_fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    for i in range(n_f_rest):
        dtype_fields.append((f"f_rest_{i}", "f4"))
    dtype_fields.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )

    vertex = np.empty(N, dtype=np.dtype(dtype_fields))
    pts_np = pts.detach().cpu().numpy().astype(np.float32)
    sh0_np = sh0.detach().cpu().numpy().astype(np.float32)
    sh_rest_np = sh_rest.detach().cpu().numpy().astype(np.float32) if n_f_rest > 0 else None
    log_scales_np = log_scales.detach().cpu().numpy().astype(np.float32)
    logit_op_np = logit_opacities.detach().cpu().numpy().astype(np.float32)
    quats_np = quats.detach().cpu().numpy().astype(np.float32)

    vertex["x"] = pts_np[:, 0]
    vertex["y"] = pts_np[:, 1]
    vertex["z"] = pts_np[:, 2]
    vertex["f_dc_0"] = sh0_np[:, 0]
    vertex["f_dc_1"] = sh0_np[:, 1]
    vertex["f_dc_2"] = sh0_np[:, 2]
    if n_f_rest > 0 and sh_rest_np is not None:
        for i in range(n_f_rest):
            vertex[f"f_rest_{i}"] = sh_rest_np[:, i]
    vertex["opacity"] = logit_op_np
    vertex["scale_0"] = log_scales_np[:, 0]
    vertex["scale_1"] = log_scales_np[:, 1]
    vertex["scale_2"] = log_scales_np[:, 2]
    vertex["rot_0"] = quats_np[:, 0]
    vertex["rot_1"] = quats_np[:, 1]
    vertex["rot_2"] = quats_np[:, 2]
    vertex["rot_3"] = quats_np[:, 3]

    ply = PlyData([PlyElement.describe(vertex, "vertex")], text=False)
    os.makedirs(os.path.dirname(config.out_ply) or ".", exist_ok=True)
    ply.write(config.out_ply)
    logger.info(
        "Wrote 3DGS PLY %s (%d splats, sh_degree=%d)",
        config.out_ply,
        int(N),
        int(model.sh_degree),
    )


if __name__ == "__main__":
    tyro.cli(main)
