from __future__ import annotations

import os

import numpy as np
import torch
import tyro
from plyfile import PlyData, PlyElement

from configs.utils import ExportGSCheckpointToPlyConfig
from data.data_loading import load_da3_camera_images
from eval_gs import (
    EvalGSConfig,
    _build_model,
    _find_checkpoint_path,
    _load_training_config,
)
from utils.logging import get_logger


logger = get_logger(__name__)

# C0 coefficient for the real SH basis (same as models/canonical_gs_model.py and
# the Antimatter15 / PlayCanvas readers).
SH_C0 = 0.28209479177387814

# SuperSplat's read-ply.ts maps the *flat* f_rest_* column count to SH bands:
# 0 -> deg 0, 9 -> deg 1, 24 -> deg 2, 45 -> deg 3 (channel-major, 3 * per-channel).
SUPERSPLAT_REST_COUNTS = {0, 9, 24, 45}


def _quats_unit(quats: np.ndarray) -> np.ndarray:
    """Normalize (N, 4) quaternions; degenerate ones fall back to identity."""
    out = quats.copy()
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    ok = norms[..., 0] > 1e-12
    out[ok] = out[ok] / norms[ok]
    out[~ok] = np.array([1.0, 0.0, 0.0, 0.0])
    return out


def _degree_from_rest(n_f_rest: int) -> int:
    """Recover sh_degree from the *flat* number of higher-order SH coefficients.

    Flat count = 3 * ((deg + 1) ** 2 - 1), so deg = sqrt(flat / 3 + 1) - 1.
    """
    if n_f_rest <= 0:
        return 0
    per_channel = n_f_rest // 3
    return int(round(np.sqrt(per_channel + 1) - 1))


def _write_ply(
    path: str,
    *,
    pts: np.ndarray,
    sh0: np.ndarray,
    sh_rest: np.ndarray | None,
    opacity_logit: np.ndarray,
    log_scales: np.ndarray,
    quats: np.ndarray,
    renderer: str,
) -> None:
    """
    Write a SuperSplat-compatible binary PLY.

    Property layout matches the standard 3DGS PLY (x,y,z, f_dc_*, f_rest_*,
    opacity, scale_*, rot_*). scale_* are log-scales, opacity is a logit, and
    rot_* is an (w,x,y,z) quaternion. For 2DGS scenes the `scale_2` property is
    omitted and a `SplatRenderMode: 2dgs` comment is added so SuperSplat renders
    zero-thickness surfels (see playcanvas/splat-transform read-ply.ts).
    """
    N = pts.shape[0]
    n_f_rest = sh_rest.shape[1] if sh_rest is not None else 0
    is_2dgs = renderer == "2dgs"

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
    dtype_fields.append(("opacity", "f4"))
    dtype_fields += [("scale_0", "f4"), ("scale_1", "f4")]
    if not is_2dgs:
        dtype_fields.append(("scale_2", "f4"))
    dtype_fields += [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    vertex = np.empty(N, dtype=np.dtype(dtype_fields))
    vertex["x"], vertex["y"], vertex["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"] = sh0[:, 0], sh0[:, 1], sh0[:, 2]
    if n_f_rest > 0 and sh_rest is not None:
        for i in range(n_f_rest):
            vertex[f"f_rest_{i}"] = sh_rest[:, i]
    vertex["opacity"] = opacity_logit
    vertex["scale_0"], vertex["scale_1"] = log_scales[:, 0], log_scales[:, 1]
    if not is_2dgs:
        vertex["scale_2"] = log_scales[:, 2]
    vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"] = (
        quats[:, 0],
        quats[:, 1],
        quats[:, 2],
        quats[:, 3],
    )

    comments = ["SplatRenderMode: 2dgs"] if is_2dgs else []
    ply = PlyData([PlyElement.describe(vertex, "vertex")], text=False, comments=comments)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ply.write(path)
    logger.info(
        "Wrote PLY %s (%d splats, renderer=%s, sh_degree=%d)",
        path,
        N,
        renderer,
        _degree_from_rest(n_f_rest),
    )


def _write_splat(
    path: str,
    *,
    pts: np.ndarray,
    sh0: np.ndarray,
    opacity_logit: np.ndarray,
    log_scales: np.ndarray,
    quats: np.ndarray,
) -> None:
    """
    Write an Antimatter15 / PlayCanvas `.splat` binary (32 bytes per splat).

    Record layout: position (3xf32) + scale (3xf32, linear) + rgba (4xu8) +
    packed quaternion (4xu8, `(q/||q||)*128+128`). Mirrors the official
    antimatter15 `convert.py` and round-trips exactly through SuperSplat's
    `read-splat.ts`. Splats are sorted big-and-opaque-first for progressive
    loaders.
    """
    N = pts.shape[0]
    scales = np.exp(log_scales)
    opac = 1.0 / (1.0 + np.exp(-opacity_logit))
    rgb = (0.5 + SH_C0 * sh0).clip(0.0, 1.0)

    order = np.argsort(-np.exp(log_scales.sum(axis=-1)) / (1.0 + np.exp(-opacity_logit)))
    pts = pts[order]
    scales = scales[order]
    rgb = rgb[order]
    opac = opac[order]
    quats = quats[order]

    rec = np.empty(
        N,
        dtype=[
            ("pos", np.float32, (3,)),
            ("scale", np.float32, (3,)),
            ("color", np.uint8, (4,)),
            ("rot", np.uint8, (4,)),
        ],
    )
    rec["pos"] = pts
    rec["scale"] = scales
    rec["color"] = np.concatenate([rgb * 255.0, opac[:, None] * 255.0], axis=1).clip(0, 255).astype(np.uint8)
    rec["rot"] = (quats * 128.0 + 128.0).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec.tofile(path)
    logger.info("Wrote SPLAT %s (%d splats)", path, N)


@torch.no_grad()
def export_model_to_files(
    model,
    *,
    renderer: str,
    out_ply: str | None,
    out_splat: str | None,
    max_points: int = -1,
    seed: int = 0,
) -> None:
    """Export an *already-loaded* GS model to SuperSplat PLY / Antimatter15 .splat.

    This is the in-memory variant of `main`: it does not reload anything from
    disk, so callers (the live viewer, or a paused training run) can export the
    exact state they currently hold.
    """
    if model.sh_coeffs is None:
        raise ValueError("Gaussian model has no SH colors; cannot export.")

    pts = model.canonical_points.detach()  # (N, 3)
    sh_coeffs = model.sh_coeffs.detach()  # (N, K, 3)
    N, K, _ = sh_coeffs.shape

    n_f_rest = (K - 1) * 3 if K > 1 else 0
    if out_ply is not None and n_f_rest not in SUPERSPLAT_REST_COUNTS:
        raise ValueError(
            f"sh_degree={int(model.sh_degree)} produces {n_f_rest} f_rest_* columns, "
            f"which SuperSplat cannot read (must be one of {sorted(SUPERSPLAT_REST_COUNTS)}). "
            "Retrain with sh_degree in 0..3 or use --config.format splat."
        )

    # Optional subsampling (deterministic via seed).
    if max_points > 0 and N > max_points:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)[: max_points]
        pts = pts[perm]
        sh_coeffs = sh_coeffs[perm]
        N = pts.shape[0]

    pts_np = pts.detach().cpu().numpy().astype(np.float32)
    sh0_np = sh_coeffs[:, 0, :].detach().cpu().numpy().astype(np.float32)
    sh_rest_np = (
        sh_coeffs[:, 1:, :].reshape(N, -1).detach().cpu().numpy().astype(np.float32)
        if K > 1
        else None
    )
    log_scales_np = model.log_scales.detach().reshape(N, 3).cpu().numpy().astype(np.float32)
    logit_op_np = model.logit_opacities.detach().reshape(N).cpu().numpy().astype(np.float32)
    quats_np = _quats_unit(model.quats.detach().reshape(N, 4).cpu().numpy().astype(np.float32))

    if out_ply is not None:
        _write_ply(
            out_ply,
            pts=pts_np,
            sh0=sh0_np,
            sh_rest=sh_rest_np,
            opacity_logit=logit_op_np,
            log_scales=log_scales_np,
            quats=quats_np,
            renderer=renderer,
        )
    if out_splat is not None:
        _write_splat(
            out_splat,
            pts=pts_np,
            sh0=sh0_np,
            opacity_logit=logit_op_np,
            log_scales=log_scales_np,
            quats=quats_np,
        )


@torch.no_grad()
def main(config: ExportGSCheckpointToPlyConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cfg = _load_training_config(config.checkpoint_dir)
    renderer = str(train_cfg.get("renderer", "2dgs"))
    if renderer not in ("2dgs", "3dgs"):
        logger.warning("Unknown renderer %r in checkpoint config; defaulting to '3dgs'.", renderer)
        renderer = "3dgs"

    if config.out_ply is None and config.format in ("ply", "both"):
        config.out_ply = os.path.join(config.checkpoint_dir, f"splats_{renderer}.ply")
    if config.out_splat is None and config.format in ("splat", "both"):
        config.out_splat = os.path.join(config.checkpoint_dir, f"splats_{renderer}.splat")

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

    export_model_to_files(
        model,
        renderer=renderer,
        out_ply=config.out_ply,
        out_splat=config.out_splat,
        max_points=config.max_points,
        seed=config.seed,
    )


if __name__ == "__main__":
    tyro.cli(main)
