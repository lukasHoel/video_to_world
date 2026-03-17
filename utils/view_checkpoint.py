from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from configs.utils import ViewGSCheckpointConfig
from data.data_loading import load_da3_camera_images
from eval_gs import EvalGSConfig, _build_model, _find_checkpoint_path
from utils.logging import get_logger


logger = get_logger(__name__)


def _depth_to_color(depth_norm: torch.Tensor) -> torch.Tensor:
    """Simple turbo-ish depth visualization. Input (H, W, 1) -> (H, W, 3)."""
    d = depth_norm.squeeze(-1)
    r = (1.0 - 2.0 * (d - 0.5).abs()).clamp(0, 1)
    g = (1.0 - 2.0 * (d - 0.25).abs()).clamp(0, 1)
    b = (1.0 - 2.0 * d).clamp(0, 1)
    return torch.stack([r, g, b], dim=-1)


def _rotation_matrix_to_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion."""
    # Numerically stable branch-based conversion.
    m00, m01, m02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
    m10, m11, m12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
    m20, m21, m22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (w, x, y, z)


def _add_camera_frames(
    server,
    c2w_list: list[torch.Tensor],
    *,
    scale: float,
    group: str = "/cameras_training",
) -> None:
    """Add camera coordinate frames (axes) to the viser scene."""
    for i, c2w in enumerate(c2w_list):
        c2w_np = c2w.detach().cpu().numpy()
        wxyz = np.array(_rotation_matrix_to_wxyz(c2w_np[:3, :3]), dtype=np.float32)
        position = c2w_np[:3, 3].astype(np.float32)
        server.scene.add_frame(
            f"{group}/cam_{i:04d}",
            wxyz=wxyz,
            position=position,
            axes_length=float(scale),
            axes_radius=float(scale) * 0.05,
        )


@torch.no_grad()
def main(config: ViewGSCheckpointConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = _find_checkpoint_path(config.checkpoint_dir)
    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Need a rendering resolution to construct the model. Use DA3 exports (cheap).
    images, poses_c2w, intrinsics = load_da3_camera_images(
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

    # -------------------------------------------------------------------------
    # Viser + nerfview viewer (interactive rendering, like old repo scripts)
    # -------------------------------------------------------------------------
    try:
        import viser
        from nerfview import CameraState, Viewer, RenderTabState
    except Exception as exc:
        raise ImportError(
            "This viewer requires 'viser' and 'nerfview'. "
            "Install them in your environment, then re-run."
        ) from exc

    renderer_type = model.renderer_type
    logger.info("Checkpoint renderer: %s (sh_degree=%d)", renderer_type, int(model.sh_degree))

    means_full = model.canonical_points.detach()
    quats_full = F.normalize(model.quats.detach(), p=2, dim=-1)
    scales_full = model.log_scales.detach().exp()
    opacities_full = torch.sigmoid(model.logit_opacities.detach()).squeeze(-1)
    colors_full = model.sh_coeffs.detach()  # (N, K, 3) for both 2DGS/3DGS

    class ViewerGSRenderTabState(RenderTabState):
        total_gs_count: int = 0
        rendered_gs_count: int = 0
        near_plane: float = float(config.near_plane)
        far_plane: float = float(min(config.far_plane, 1e6))
        render_mode: str = "rgb"
        backgrounds: tuple[float, float, float] = (255.0, 255.0, 255.0)
        rasterize_mode: str = "classic"
        max_sh_degree: int = int(model.sh_degree)

    class ViewerGS(Viewer):
        def __init__(self, server, render_fn, output_dir, mode="rendering"):
            super().__init__(server, render_fn, output_dir, mode)
            label = "2DGS Viewer" if renderer_type == "2dgs" else "3DGS Viewer"
            server.gui.set_panel_label(label)

        def _init_rendering_tab(self):
            self.render_tab_state = ViewerGSRenderTabState()
            self._rendering_tab_handles = {}
            self._rendering_folder = self.server.gui.add_folder("Rendering")

        def _populate_rendering_tab(self):
            server = self.server
            with self._rendering_folder:
                with server.gui.add_folder("GS"):
                    total_number = server.gui.add_number(
                        "Total splats",
                        initial_value=self.render_tab_state.total_gs_count,
                        disabled=True,
                    )
                    rendered_number = server.gui.add_number(
                        "Rendered",
                        initial_value=self.render_tab_state.rendered_gs_count,
                        disabled=True,
                    )
                    near_far = server.gui.add_vector2(
                        "Near / Far",
                        initial_value=(self.render_tab_state.near_plane, self.render_tab_state.far_plane),
                        min=(1e-3, 1.0),
                        max=(1.0, 1e6),
                        step=0.01,
                    )

                    @near_far.on_update
                    def _(_) -> None:
                        self.render_tab_state.near_plane = float(near_far.value[0])
                        self.render_tab_state.far_plane = float(near_far.value[1])
                        self.rerender(_)

                    bg_color = server.gui.add_rgb(
                        "Background",
                        initial_value=self.render_tab_state.backgrounds,
                    )

                    @bg_color.on_update
                    def _(_) -> None:
                        self.render_tab_state.backgrounds = bg_color.value
                        self.rerender(_)

                    if renderer_type == "2dgs":
                        mode_dd = server.gui.add_dropdown(
                            "Render mode",
                            ("rgb", "depth", "normal", "alpha"),
                            initial_value=self.render_tab_state.render_mode,
                        )
                    else:
                        mode_dd = server.gui.add_dropdown(
                            "Render mode",
                            ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                            initial_value=self.render_tab_state.render_mode,
                        )

                    @mode_dd.on_update
                    def _(_) -> None:
                        self.render_tab_state.render_mode = mode_dd.value
                        self.rerender(_)

                    if renderer_type == "3dgs":
                        rasterize_dd = server.gui.add_dropdown(
                            "Rasterize mode",
                            ("classic", "antialiased"),
                            initial_value=self.render_tab_state.rasterize_mode,
                        )

                        @rasterize_dd.on_update
                        def _(_) -> None:
                            self.render_tab_state.rasterize_mode = rasterize_dd.value
                            self.rerender(_)

                        max_sh = server.gui.add_slider(
                            "Max SH degree",
                            min=0,
                            max=max(0, int(model.sh_degree)),
                            step=1,
                            initial_value=int(model.sh_degree),
                        )

                        @max_sh.on_update
                        def _(_) -> None:
                            self.render_tab_state.max_sh_degree = int(max_sh.value)
                            self.rerender(_)

            self._rendering_tab_handles.update(
                {
                    "total_number": total_number,
                    "rendered_number": rendered_number,
                }
            )
            super()._populate_rendering_tab()

        def _after_render(self):
            self._rendering_tab_handles["total_number"].value = self.render_tab_state.total_gs_count
            self._rendering_tab_handles["rendered_number"].value = self.render_tab_state.rendered_gs_count

    # Import gsplat in main thread (avoid issues with torch.distributed init in worker thread).
    if renderer_type == "3dgs":
        from gsplat.rendering import rasterization as rasterization_3d
    else:
        from gsplat import rasterization_2dgs

    @torch.no_grad()
    def render_fn(camera_state: CameraState, render_tab_state: RenderTabState) -> np.ndarray:
        assert isinstance(render_tab_state, ViewerGSRenderTabState)

        if render_tab_state.preview_render:
            width = int(render_tab_state.render_width)
            height = int(render_tab_state.render_height)
        else:
            width = int(render_tab_state.viewer_width)
            height = int(render_tab_state.viewer_height)

        # Camera matrices
        c2w = torch.from_numpy(camera_state.c2w).to(device=device, dtype=torch.float32)
        K = torch.from_numpy(camera_state.get_K((width, height))).to(device=device, dtype=torch.float32)

        # Closed-form w2c for stability
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -(R_w2c @ t_c2w)
        viewmat = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
        viewmat[:3, :3] = R_w2c
        viewmat[:3, 3] = t_w2c

        bg_rgb = torch.tensor(render_tab_state.backgrounds, device=c2w.device, dtype=torch.float32) / 255.0

        if renderer_type == "2dgs":
            render_mode_map = {
                "rgb": "RGB+ED",
                "depth": "RGB+ED",
                "normal": "RGB+ED",
                "alpha": "RGB+ED",
            }
            (
                render_colors,
                render_alphas,
                render_normals,
                _surf_normals,
                _render_distort,
                _render_median,
                info,
            ) = rasterization_2dgs(
                means=means_full,
                quats=quats_full,
                scales=scales_full,
                opacities=opacities_full,
                colors=colors_full,
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                near_plane=float(render_tab_state.near_plane),
                far_plane=float(render_tab_state.far_plane),
                render_mode=render_mode_map[render_tab_state.render_mode],
                backgrounds=bg_rgb.unsqueeze(0),
                sh_degree=int(model.sh_degree),
            )

            render_tab_state.total_gs_count = int(means_full.shape[0])
            render_tab_state.rendered_gs_count = int((info["radii"] > 0).all(-1).sum().item())

            if render_tab_state.render_mode == "rgb":
                out = render_colors[0, ..., :3].clamp(0, 1)
            elif render_tab_state.render_mode == "depth":
                depth = render_colors[0, ..., 3:4]
                near = depth[depth > 0].min() if (depth > 0).any() else depth.min()
                far = depth.max()
                depth_norm = (depth - near) / (far - near + 1e-10)
                out = _depth_to_color(depth_norm.clamp(0, 1))
            elif render_tab_state.render_mode == "normal":
                if render_normals is not None and render_normals.numel() > 0:
                    out = (render_normals[0] * 0.5 + 0.5).clamp(0, 1)
                else:
                    out = render_colors[0, ..., :3].clamp(0, 1)
            elif render_tab_state.render_mode == "alpha":
                a = render_alphas[0]
                out = a.expand(-1, -1, 3)
            else:
                out = render_colors[0, ..., :3].clamp(0, 1)

            return out.detach().cpu().numpy()

        # 3DGS
        render_mode_map_3d = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }
        packed = True
        bg = bg_rgb if packed else bg_rgb.unsqueeze(0)

        render_colors, render_alphas, info = rasterization_3d(
            means=means_full,
            quats=quats_full,
            scales=scales_full,
            opacities=opacities_full,
            colors=colors_full,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            near_plane=float(render_tab_state.near_plane),
            far_plane=float(render_tab_state.far_plane),
            packed=packed,
            rasterize_mode=render_tab_state.rasterize_mode,
            sh_degree=min(int(render_tab_state.max_sh_degree), int(model.sh_degree)),
            backgrounds=bg,
            render_mode=render_mode_map_3d[render_tab_state.render_mode],
        )

        render_tab_state.total_gs_count = int(means_full.shape[0])
        render_tab_state.rendered_gs_count = int((info["radii"] > 0).all(dim=-1).sum().item())

        if render_tab_state.render_mode == "rgb":
            out = render_colors[0, ..., 0:3].clamp(0, 1)
            return out.detach().cpu().numpy()

        if render_tab_state.render_mode in ("depth(accumulated)", "depth(expected)"):
            depth = render_colors[0, ..., 0:1]
            d_min = depth.min().item()
            d_max = depth.max().item()
            depth_norm = (depth - d_min) / (d_max - d_min + 1e-10)
            out = _depth_to_color(depth_norm.clamp(0, 1))
            return out.detach().cpu().numpy()

        if render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            out = alpha.expand(-1, -1, 3)
            return out.detach().cpu().numpy()

        return render_colors[0, ..., 0:3].clamp(0, 1).detach().cpu().numpy()

    server = viser.ViserServer(port=int(config.port), verbose=False)

    # Add training camera frames if available (from optimised per_frame_c2w).
    if bool(config.show_cameras):
        c2w_list = [model.get_c2w(i).detach().cpu() for i in range(len(model.per_frame_c2w))]
        if c2w_list:
            _add_camera_frames(
                server,
                c2w_list,
                scale=float(config.training_cameras_scale),
                group="/cameras_training",
            )

    # Viewer output dir (screenshots / camera paths) lives next to checkpoint.
    output_dir = Path(os.path.dirname(ckpt_path))
    viewer = ViewerGS(
        server=server,
        render_fn=render_fn,
        output_dir=output_dir,
        mode="rendering",
    )

    # Initial state.
    viewer.render_tab_state.near_plane = float(config.near_plane)
    viewer.render_tab_state.far_plane = float(min(config.far_plane, 1e6))
    if renderer_type == "3dgs":
        viewer.render_tab_state.rasterize_mode = "classic"
        viewer.render_tab_state.max_sh_degree = int(model.sh_degree)

    logger.info("Viewer running on http://localhost:%d — Ctrl+C to exit.", int(config.port))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    tyro.cli(main)
