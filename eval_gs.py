"""
Evaluate a trained 2DGS / 3DGS model on input, optimised, and custom camera paths.

Renders images + MP4 videos from a checkpoint produced by train_gs.py.

Usage:
  python -m eval_gs \\
      --root-path /path/to/da3_scene \\
      --run frame_to_model_icp_50_2_offset0 \\
      --checkpoint-dir /path/to/gs_run_dir
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
import tyro

from configs.eval_gs import EvalGSConfig
from data.checkpoint_loading import (
    load_aligned_point_cloud,
    load_deformation_checkpoints,
)
from data.data_loading import (
    load_da3_camera_images,
    load_da3_original_images_from_folder,
    load_nerf_transforms_json,
)
from models.canonical_gs_model import CanonicalGSModel
from utils.downsample import downsample_to_target


def _find_checkpoint_path(checkpoint_dir: str) -> str:
    """Pick the final GS checkpoint file from a directory."""
    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    if os.path.exists(final_path):
        return final_path

    # Fallback to latest checkpoint_*.pt (if present)
    candidates = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
    candidates.sort()
    return os.path.join(checkpoint_dir, candidates[-1])


def _w2c_from_c2w(c2w: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of a rigid c2w (4,4)."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    R_w2c = R.transpose(0, 1)
    t_w2c = -(R_w2c @ t)
    w2c = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c
    return w2c


def _load_training_config(checkpoint_dir: str) -> dict:
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r") as f:
        return json.load(f)


def _build_model(
    config: EvalGSConfig,
    device: str,
    *,
    height: int,
    width: int,
) -> CanonicalGSModel:
    # Load global-optimization checkpoint pieces
    global_opt_dir = os.path.join(config.root_path, config.run, config.global_opt_subdir)
    if not os.path.exists(global_opt_dir):
        raise FileNotFoundError(f"Global-optimization directory not found: {global_opt_dir}")

    canonical_pts, canonical_cols = load_aligned_point_cloud(global_opt_dir, device)

    # Use training config from checkpoint dir so model structure matches (renderer, SH degree, target_num_points, etc.)
    train_cfg = _load_training_config(config.checkpoint_dir)

    # Downsample — must mirror training's target_num_points if you want strict weight loading.
    target_num_points = int(train_cfg.get("target_num_points", config.target_num_points))
    if canonical_pts.shape[0] > target_num_points:
        canonical_pts, canonical_cols = downsample_to_target(
            canonical_pts, canonical_cols, target_count=target_num_points
        )

    per_frame_global_deform, per_frame_local_deform, bbox_min, bbox_max = load_deformation_checkpoints(
        global_opt_dir, device, first_local="none", allow_rigid_fallback=True
    )

    # Respect the number of frames actually used during GS training. If the
    # training config recorded a specific num_frames, clamp to that; otherwise
    # keep the full deformation sequence.
    num_frames_cfg = train_cfg.get("num_frames", None)
    if isinstance(num_frames_cfg, int) and num_frames_cfg > 0:
        per_frame_global_deform = per_frame_global_deform[:num_frames_cfg]
        per_frame_local_deform = per_frame_local_deform[:num_frames_cfg]

    renderer = train_cfg.get("renderer", "2dgs")
    sh_degree = int(train_cfg.get("sh_degree", 0))

    model = CanonicalGSModel(
        canonical_points=canonical_pts,
        canonical_colors=canonical_cols,
        per_frame_global_deform=per_frame_global_deform,
        per_frame_local_deform=per_frame_local_deform,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        height=height,
        width=width,
        renderer=renderer,
        optimize_cams=False,
        optimize_positions=False,
        sh_degree=sh_degree,
    ).to(device)

    return model


@torch.no_grad()
def main(config: EvalGSConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if config.transforms_path is None:
        config.transforms_path = os.path.join(config.root_path, "gs_video", "0000_extend_transforms.json")

    if config.out_dir is None:
        config.out_dir = os.path.join(config.checkpoint_dir, "gs_video_eval")
    os.makedirs(config.out_dir, exist_ok=True)

    # Load training config to auto-infer original image settings when possible.
    train_cfg = _load_training_config(config.checkpoint_dir)
    if not config.original_images_dir:
        orig_dir = train_cfg.get("original_images_dir", "")
        if isinstance(orig_dir, str):
            config.original_images_dir = orig_dir
    use_folder_originals = bool(config.original_images_dir)

    # ------------------------------------------------------------------
    # 1. Load gs_video / transforms camera path
    # ------------------------------------------------------------------
    ow = None if config.override_width <= 0 else int(config.override_width)
    oh = None if config.override_height <= 0 else int(config.override_height)

    poses_c2w_gs: torch.Tensor | None = None
    intrinsics_gs: torch.Tensor | None = None
    w: int | None = None
    h: int | None = None

    if config.render_gs_video_path:
        poses_c2w_gs, intrinsics_gs, w, h = load_nerf_transforms_json(
            config.transforms_path,
            device=device,
            override_width=ow,
            override_height=oh,
            blender_opengl_to_opencv=True,
        )

        if config.max_frames > 0:
            poses_c2w_gs = poses_c2w_gs[: config.max_frames]
            intrinsics_gs = intrinsics_gs[: config.max_frames]

        num_poses_gs = poses_c2w_gs.shape[0]
        print(f"Loaded {num_poses_gs} gs_video camera poses from: {config.transforms_path}")

    # ------------------------------------------------------------------
    # 2. Fallback resolution from DA3 exports if needed
    # ------------------------------------------------------------------
    if w is None or h is None or use_folder_originals:
        assert config.override_width < 0 and config.override_height < 0, (
            "Override width and height must not be specified if using original images"
        )
        try:
            if use_folder_originals:
                images_da3, intrinsics_list = load_da3_original_images_from_folder(
                    config.root_path,
                    config.original_images_dir,
                    num_frames=1,
                    stride=1,
                    device=device,
                )
                h, w = int(images_da3.shape[2]), int(images_da3.shape[3])
                if intrinsics_gs is not None and intrinsics_list:
                    K0 = torch.from_numpy(intrinsics_list[0]).to(device=device, dtype=torch.float32)
                    intrinsics_gs = K0.unsqueeze(0).repeat(intrinsics_gs.shape[0], 1, 1)
            else:
                images_da3, _, intrinsics_da3 = load_da3_camera_images(
                    config.root_path,
                    num_frames=1,
                    stride=1,
                    device=device,
                    use_original_images_and_intrinsics=False,
                )
                h, w = int(images_da3.shape[2]), int(images_da3.shape[3])
                intrinsics_gs = intrinsics_da3.repeat(intrinsics_gs.shape[0], 1, 1)
        except Exception:
            raise ValueError(
                "Could not determine rendering resolution. "
                "Your transforms JSON has no w/h; please pass --override-width/--override-height."
            )

    print(f"Rendering at {w}x{h} (fps={config.fps})")

    # ------------------------------------------------------------------
    # 3. Build + load model (config and checkpoint structure from checkpoint_dir)
    # ------------------------------------------------------------------
    ckpt_path = _find_checkpoint_path(config.checkpoint_dir)
    print(f"Loading GS checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _build_model(config, device, height=int(h), width=int(w))
    model.load_state_dict(state, strict=False)
    model.eval()

    # Helper: factory for MP4 writer
    def _make_video_writer(name: str):
        if not config.save_video:
            return None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path = os.path.join(config.out_dir, f"{name}.mp4")
        return cv2.VideoWriter(path, fourcc, config.fps, (int(w), int(h)))

    # ------------------------------------------------------------------
    # 4. Render gs_video / transforms path
    # ------------------------------------------------------------------
    if config.render_gs_video_path and poses_c2w_gs is not None:
        frames_dir = os.path.join(config.out_dir, "frames_gs_video")
        if config.save_images:
            os.makedirs(frames_dir, exist_ok=True)

        video_writer = _make_video_writer("render_gs_video")
        num_poses = poses_c2w_gs.shape[0]

        for i in range(num_poses):
            c2w = poses_c2w_gs[i]
            K = intrinsics_gs[i]
            viewmat = _w2c_from_c2w(c2w)

            if model.renderer_type == "2dgs":
                out = model.render_2dgs(model.canonical_points, viewmat, K)
            elif model.renderer_type == "3dgs":
                out = model.render_3dgs(model.canonical_points, viewmat, K)
            else:
                raise ValueError(f"Unknown renderer_type: {model.renderer_type!r}")

            rgb = out["rgb"][0].clamp(0, 1)  # (3, H, W)
            if config.white_background and "alpha" in out:
                alpha = out["alpha"][0]  # (1, H, W)
                rgb = rgb * alpha + (1.0 - alpha)
                rgb = rgb.clamp(0, 1)
            img_rgb = (rgb.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if config.save_images:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{i:05d}.png"), img_bgr)
            if video_writer is not None:
                video_writer.write(img_bgr)

            if (i + 1) % 50 == 0 or (i + 1) == num_poses:
                print(f"[gs_video] Rendered {i + 1}/{num_poses}")

        if video_writer is not None:
            video_writer.release()

    # ------------------------------------------------------------------
    # 5. Render input DA3 poses (from results.npz)
    # ------------------------------------------------------------------
    if config.render_input_poses:
        _, poses_c2w_in, _ = load_da3_camera_images(
            config.root_path,
            num_frames=10_000,
            stride=1,
            device=device,
            use_original_images_and_intrinsics=False,
        )
        if use_folder_originals:
            images_in, intrinsics_list = load_da3_original_images_from_folder(
                config.root_path,
                config.original_images_dir,
                num_frames=poses_c2w_in.shape[0],
                stride=1,
                device=device,
            )
            intrinsics_in = torch.stack(
                [torch.from_numpy(K).to(device=device, dtype=torch.float32) for K in intrinsics_list],
                dim=0,
            )
        else:
            images_in, _, intrinsics_in = load_da3_camera_images(
                config.root_path,
                num_frames=10_000,
                stride=1,
                device=device,
                use_original_images_and_intrinsics=False,
            )
        num_in = poses_c2w_in.shape[0]
        H_in, W_in = int(images_in.shape[2]), int(images_in.shape[3])

        sx = float(w) / float(W_in)
        sy = float(h) / float(H_in)

        frames_dir_in = os.path.join(config.out_dir, "frames_input_poses")
        if config.save_images:
            os.makedirs(frames_dir_in, exist_ok=True)
        video_writer_in = _make_video_writer("render_input_poses")

        for i in range(num_in):
            K0 = intrinsics_in[i].clone()
            K0[0, 0] *= sx
            K0[1, 1] *= sy
            K0[0, 2] *= sx
            K0[1, 2] *= sy
            viewmat = _w2c_from_c2w(poses_c2w_in[i])

            if model.renderer_type == "2dgs":
                out = model.render_2dgs(model.canonical_points, viewmat, K0)
            elif model.renderer_type == "3dgs":
                out = model.render_3dgs(model.canonical_points, viewmat, K0)
            else:
                raise ValueError(f"Unknown renderer_type: {model.renderer_type!r}")

            rgb = out["rgb"][0].clamp(0, 1)
            if config.white_background and "alpha" in out:
                alpha = out["alpha"][0]
                rgb = rgb * alpha + (1.0 - alpha)
                rgb = rgb.clamp(0, 1)
            img_rgb = (rgb.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if config.save_images:
                cv2.imwrite(os.path.join(frames_dir_in, f"frame_{i:05d}.png"), img_bgr)
            if video_writer_in is not None:
                video_writer_in.write(img_bgr)

            if (i + 1) % 50 == 0 or (i + 1) == num_in:
                print(f"[input_poses] Rendered {i + 1}/{num_in}")

        if video_writer_in is not None:
            video_writer_in.release()

    # ------------------------------------------------------------------
    # 6. Render optimised GS checkpoint poses (per_frame_c2w)
    # ------------------------------------------------------------------
    if config.render_optimised_poses:
        try:
            _, poses_c2w_in, _ = load_da3_camera_images(
                config.root_path,
                num_frames=10_000,
                stride=1,
                device=device,
                use_original_images_and_intrinsics=False,
            )
            if use_folder_originals:
                images_in, intrinsics_list = load_da3_original_images_from_folder(
                    config.root_path,
                    config.original_images_dir,
                    num_frames=poses_c2w_in.shape[0],
                    stride=1,
                    device=device,
                )
                intrinsics_in = torch.stack(
                    [torch.from_numpy(K).to(device=device, dtype=torch.float32) for K in intrinsics_list],
                    dim=0,
                )
            else:
                images_in, _, intrinsics_in = load_da3_camera_images(
                    config.root_path,
                    num_frames=10_000,
                    stride=1,
                    device=device,
                    use_original_images_and_intrinsics=False,
                )
            num_in = poses_c2w_in.shape[0]
            H_in, W_in = int(images_in.shape[2]), int(images_in.shape[3])
            sx = float(w) / float(W_in)
            sy = float(h) / float(H_in)
        except Exception:
            intrinsics_in = None
            num_in = 0

        num_opt = len(model.per_frame_c2w)
        num_render = num_opt if intrinsics_in is None else min(num_opt, num_in)

        if num_render > 0 and intrinsics_in is not None:
            frames_dir_opt = os.path.join(config.out_dir, "frames_optimised_poses")
            if config.save_images:
                os.makedirs(frames_dir_opt, exist_ok=True)
            video_writer_opt = _make_video_writer("render_optimised_poses")

            for i in range(num_render):
                K0 = intrinsics_in[i].clone()
                K0[0, 0] *= sx
                K0[1, 1] *= sy
                K0[0, 2] *= sx
                K0[1, 2] *= sy

                if model.renderer_type == "2dgs":
                    viewmat = model.get_viewmat(i)
                    out = model.render_2dgs(model.canonical_points, viewmat, K0)
                elif model.renderer_type == "3dgs":
                    viewmat = model.get_viewmat(i)
                    out = model.render_3dgs(model.canonical_points, viewmat, K0)
                else:
                    raise ValueError(f"Unknown renderer_type: {model.renderer_type!r}")

                rgb = out["rgb"][0].clamp(0, 1)
                if config.white_background and "alpha" in out:
                    alpha = out["alpha"][0]
                    rgb = rgb * alpha + (1.0 - alpha)
                    rgb = rgb.clamp(0, 1)
                img_rgb = (rgb.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                if config.save_images:
                    cv2.imwrite(os.path.join(frames_dir_opt, f"frame_{i:05d}.png"), img_bgr)
                if video_writer_opt is not None:
                    video_writer_opt.write(img_bgr)

                if (i + 1) % 50 == 0 or (i + 1) == num_render:
                    print(f"[optimised_poses] Rendered {i + 1}/{num_render}")

            if video_writer_opt is not None:
                video_writer_opt.release()

    with open(os.path.join(config.out_dir, "eval_config.json"), "w") as f:
        json.dump({k: v for k, v in vars(config).items()}, f, indent=2)

    print(f"Done. Outputs saved to: {config.out_dir}")


if __name__ == "__main__":
    tyro.cli(main)
