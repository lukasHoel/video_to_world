"""
Canonical GS model for 2DGS / 3DGS rendering of aligned point clouds.

Renders a merged canonical point cloud via gsplat (2DGS or 3DGS) using
inverse-deformed Gaussians for per-view supervision.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from models.deformation import DeformationGrid, ViewConditionedInverseDeformation
from utils.geometry import (
    quaternion_multiply,
    rotation_matrix_to_quaternion,
    se3_apply,
    se3_exp,
    se3_inverse,
)

logger = logging.getLogger(__name__)


SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb01: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0,1] to SH DC coefficient."""
    return (rgb01.clamp(0, 1) - 0.5) / SH_C0


def sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    """Convert SH DC coefficient to RGB in [0,1]."""
    return (sh0 * SH_C0 + 0.5).clamp(0, 1)


class CanonicalGSModel(nn.Module):
    """
    Canonical novel-view synthesis model operating on a single merged &
    downsampled canonical point cloud.

    Parameters
    ----------
    canonical_points : (N, 3) — merged canonical points in world space
    canonical_colors : (N, 3) — per-point RGB in [0, 1]
    per_frame_global_deform : list of (6,) c2w SE3 twist tensors
    per_frame_local_deform : list of DeformationGrid | None
    bbox_min, bbox_max : (3,) bounding box of local deform networks
    height, width : image dimensions
    renderer : '2dgs' or '3dgs'
    optimize_cams : if True, c2w twists are learnable (frame 0 frozen)
    optimize_positions : if True, canonical point positions are learnable
    """

    def __init__(
        self,
        canonical_points: torch.Tensor,
        canonical_colors: torch.Tensor,
        per_frame_global_deform: list[torch.Tensor],
        per_frame_local_deform: list,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        height: int,
        width: int,
        renderer: Literal["2dgs", "3dgs"] = "2dgs",
        optimize_cams: bool = True,
        optimize_positions: bool = True,
        # 2DGS defaults
        deform_rotations: bool = True,
        initial_opacity: float = 0.5,
        initial_scale: float = 0.005,
        initial_flat_ratio: float = 0.1,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        # SH colors (2DGS only)
        sh_degree: int = 0,
        # Inverse deform (B2)
        inverse_deform_net: Optional[ViewConditionedInverseDeformation] = None,
        # KNN-based scale init (per-point average distance to K neighbours)
        knn_dists: Optional[torch.Tensor] = None,
        # Normal-based quaternion init (per-point unit normals)
        init_normals: Optional[torch.Tensor] = None,
        # Per-frame point index segments for xcanon_all / pfgt losses: [(start_0, end_0), ...]
        model_frame_segments: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()

        self.num_frames = len(per_frame_global_deform)
        self.model_frame_segments = model_frame_segments  # optional [(s0, e0), (s1, e1), ...]
        self.height = height
        self.width = width
        self.renderer_type = renderer
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.deform_rotations = deform_rotations
        requested_sh_degree = max(int(sh_degree), 0)
        self.sh_degree = requested_sh_degree
        self.is_3dgs = renderer == "3dgs"

        N = canonical_points.shape[0]
        logger.info("CanonicalGSModel: %d points, renderer=%s", N, renderer)

        self.register_buffer("bbox_min", bbox_min.clone().detach())
        self.register_buffer("bbox_max", bbox_max.clone().detach())

        # --- Canonical point positions ---
        self.canonical_points = nn.Parameter(
            canonical_points.clone(),
            requires_grad=optimize_positions,
        )

        # --- Per-point learnable attributes ---
        if renderer in ("2dgs", "3dgs"):
            # SH color representation (always enabled for both 2DGS and 3DGS).
            # We store SH0 and higher-order bands
            # as separate Parameters for training, and expose ``sh_coeffs`` for rendering.
            K_sh = (self.sh_degree + 1) ** 2  # sh_degree==0 -> K_sh==1
            sh0 = rgb_to_sh0(canonical_colors)  # (N, 3)
            self.sh_dc = nn.Parameter(sh0.unsqueeze(1))  # (N, 1, 3)
            if K_sh > 1:
                sh_rest_init = torch.zeros(
                    (N, K_sh - 1, 3),
                    dtype=canonical_colors.dtype,
                    device=canonical_colors.device,
                )
                self.sh_rest = nn.Parameter(sh_rest_init)
            else:
                self.sh_rest = None
            logger.info(
                "Initialised SH colors: degree=%d, K=%d",
                self.sh_degree,
                K_sh,
            )

            logit_op = torch.log(torch.tensor(initial_opacity) / (1.0 - torch.tensor(initial_opacity))).item()
            self.logit_opacities = nn.Parameter(torch.full((N, 1), logit_op))

            if knn_dists is not None:
                # Per-point scale = KNN distance (natural overlap scale)
                log_xy = torch.log(knn_dists.clamp(min=1e-7))
                log_z = torch.log((knn_dists * initial_flat_ratio).clamp(min=1e-7))
                self.log_scales = nn.Parameter(torch.stack([log_xy, log_xy, log_z], dim=-1))
                logger.info(
                    "KNN-based scale init: median=%.5f, min=%.5f, max=%.5f",
                    knn_dists.median().item(),
                    knn_dists.min().item(),
                    knn_dists.max().item(),
                )
            else:
                log_xy = torch.log(torch.tensor(initial_scale).clamp(min=1e-7))
                log_z = torch.log(torch.tensor(initial_scale * initial_flat_ratio).clamp(min=1e-7))
                self.log_scales = nn.Parameter(
                    torch.stack(
                        [
                            torch.full((N,), log_xy),
                            torch.full((N,), log_xy),
                            torch.full((N,), log_z),
                        ],
                        dim=-1,
                    )
                )

            # Quaternion initialisation
            if init_normals is not None:
                from utils.geometry import normal_to_quaternion

                quats_init = normal_to_quaternion(init_normals)
                logger.info("Normal-based quaternion init from %d normals", N)
            else:
                # Fallback: identity + small noise
                quats_init = torch.zeros(N, 4)
                quats_init[:, 0] = 1.0
                quats_init += torch.randn_like(quats_init) * 0.01
                quats_init = quats_init / quats_init.norm(dim=-1, keepdim=True)
            self.quats = nn.Parameter(quats_init)

        # --- Camera parameters (c2w SE3 twists) ---
        self.per_frame_c2w = nn.ParameterList()
        for i, xi in enumerate(per_frame_global_deform):
            requires_grad = optimize_cams and (i > 0)  # freeze frame 0
            self.per_frame_c2w.append(nn.Parameter(xi.clone().detach(), requires_grad=requires_grad))

        # --- Frozen local deformations (for deform-weight computation in B1) ---
        self.per_frame_local_deform = nn.ModuleList()
        for deform in per_frame_local_deform:
            if isinstance(deform, DeformationGrid):
                for p in deform.parameters():
                    p.requires_grad = False
                self.per_frame_local_deform.append(deform)
            else:
                self.per_frame_local_deform.append(None)

        # --- Frozen inverse deformation (for B2) ---
        self.inverse_deform = inverse_deform_net
        if self.inverse_deform is not None:
            for p in self.inverse_deform.parameters():
                p.requires_grad = False

        # Precompute inverse globals
        self._update_inverse_globals()

    @property
    def sh_coeffs(self) -> torch.Tensor:
        """
        Concatenated SH coefficients tensor of shape (N, K, 3), where the
        first band (index 0) is SH0.
        """
        if self.sh_rest is None:
            return self.sh_dc
        return torch.cat([self.sh_dc, self.sh_rest], dim=1)

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _update_inverse_globals(self):
        inv_globals = []
        for xi in self.per_frame_c2w:
            inv_globals.append(se3_inverse(xi.detach()))
        self.register_buffer(
            "inv_global_deforms",
            torch.stack(inv_globals),
            persistent=False,
        )

    def get_c2w(self, frame_idx: int) -> torch.Tensor:
        """Return (4,4) camera-to-world matrix for a frame."""
        R_c2w, t_c2w = se3_exp(self.per_frame_c2w[frame_idx])
        c2w = torch.eye(4, device=R_c2w.device, dtype=R_c2w.dtype)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w
        return c2w

    def get_viewmat(self, frame_idx: int) -> torch.Tensor:
        """Return (4,4) world-to-camera matrix for a frame.

        Uses the closed-form rigid-body inverse (R^T, -R^T t) instead of
        ``linalg.inv`` for numerical robustness.
        """
        R_c2w, t_c2w = se3_exp(self.per_frame_c2w[frame_idx])
        R_w2c = R_c2w.T
        t_w2c = -(R_w2c @ t_c2w)
        w2c = torch.eye(4, device=R_c2w.device, dtype=R_c2w.dtype)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        return w2c

    # ------------------------------------------------------------------
    # Inverse deformation
    # ------------------------------------------------------------------

    def inverse_deform_points(
        self,
        frame_idx: int,
        point_range: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Apply frozen inverse deformation to canonical points for frame
        *frame_idx*.

        Returns (N, 3) or (n_subset, 3) world-space points. If point_range=(s, e)
        is given, only canonical_points[s:e] are deformed.
        """
        assert self.inverse_deform is not None, "inverse_deform_net not loaded"
        pts_can = (
            self.canonical_points[point_range[0] : point_range[1]] if point_range is not None else self.canonical_points
        )
        N = pts_can.shape[0]
        xi_inv_global = self.inv_global_deforms[frame_idx]

        # Step 1: canonical → camera (detached)
        pts_cam = se3_apply(xi_inv_global.detach(), pts_can)

        # Step 2: inverse local
        view_idx = torch.full((N,), frame_idx, device=pts_cam.device, dtype=torch.long)
        pts_cam_corrected = self.inverse_deform.inverse_warp(pts_cam, view_idx)

        # Step 3: camera → world via c2w (gradients flow through c2w)
        pts_world = se3_apply(self.per_frame_c2w[frame_idx], pts_cam_corrected)
        return pts_world

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_2dgs(
        self,
        means: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        extra_colors: torch.Tensor | None = None,
        point_range: Optional[Tuple[int, int]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Render using 2DGS (gsplat).

        Args:
            means: (N, 3) or (n_subset, 3) point positions in world space
            viewmat: (4, 4) world-to-camera
            K: (3, 3) intrinsics
            extra_colors: (N, C) additional per-point channels to render
            point_range: when set, slice per-point attributes to [s, e) to match means

        Returns dict with keys: rgb, alpha, depth, extras (optional).
        """
        from gsplat import rasterization_2dgs

        sh_coeffs = self.sh_coeffs

        if point_range is not None:
            s, e = point_range
            quats = self.quats[s:e] / self.quats[s:e].norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scales = self.log_scales[s:e].exp()
            opacities = torch.sigmoid(self.logit_opacities[s:e]).squeeze(-1)
            sh_colors = sh_coeffs[s:e]
        else:
            quats = self.quats / self.quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scales = self.log_scales.exp()
            opacities = torch.sigmoid(self.logit_opacities).squeeze(-1)
            sh_colors = sh_coeffs

        if extra_colors is not None:
            raise NotImplementedError("extra_colors is not supported when using SH colors.")

        (
            render_colors,
            render_alphas,
            render_normals,
            surf_normals,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=self.width,
            height=self.height,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            sh_degree=self.sh_degree,
            render_mode="RGB+ED",
        )

        rgb = render_colors[..., :3].permute(0, 3, 1, 2)  # (1, 3, H, W)
        depth = render_colors[..., 3:4].permute(0, 3, 1, 2)  # (1, 1, H, W)
        alpha = render_alphas.permute(0, 3, 1, 2)  # (1, 1, H, W)

        result = {
            "rgb": rgb,
            "alpha": alpha,
            "depth": depth,
            "normals": render_normals,
            "surf_normals": surf_normals,
            "distort": render_distort,
            "median_depth": render_median,
            "info": info,
        }

        return result

    def render_3dgs(
        self,
        means: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        point_range: Optional[Tuple[int, int]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Render using 3DGS (gsplat 3D rasterizer).

        Args:
            means: (N, 3) or (n_subset, 3) point positions in world space
            viewmat: (4, 4) world-to-camera
            K: (3, 3) intrinsics
            point_range: when set, slice per-point attributes to [s, e) to match means
        """
        from gsplat.rendering import rasterization

        # SH colors are always used. Degree 0 corresponds to SH0-only (DC band).

        if point_range is not None:
            s, e = point_range
            quats = self.quats[s:e] / self.quats[s:e].norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scales = self.log_scales[s:e].exp()
            opacities = torch.sigmoid(self.logit_opacities[s:e]).squeeze(-1)
            sh_colors = self.sh_coeffs[s:e]
        else:
            quats = self.quats / self.quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scales = self.log_scales.exp()
            opacities = torch.sigmoid(self.logit_opacities).squeeze(-1)
            sh_colors = self.sh_coeffs

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=self.width,
            height=self.height,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            sh_degree=self.sh_degree,
            render_mode="RGB",
        )

        rgb = render_colors.permute(0, 3, 1, 2)  # (1, 3, H, W)
        alpha = render_alphas.permute(0, 3, 1, 2)  # (1, 1, H, W)

        result = {
            "rgb": rgb,
            "alpha": alpha,
            "depth": None,
            "normals": None,
            "surf_normals": None,
            "distort": None,
            "median_depth": None,
            "info": info,
        }

        return result

    # ------------------------------------------------------------------
    # High-level render entry points
    # ------------------------------------------------------------------

    def render_frame(
        self,
        frame_idx: int,
        K: torch.Tensor,
        use_inverse_deform: bool = False,
        frame_points_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Render canonical points from camera *frame_idx*.

        If ``use_inverse_deform=True`` (B2 mode), applies inverse deformation
        before rendering. If ``frame_points_only=True`` and ``model_frame_segments``
        is set, only points belonging to this frame are rendered (for xcanon_all / pfgt losses).
        """
        self._update_inverse_globals()

        point_range: Optional[Tuple[int, int]] = None
        if frame_points_only and self.model_frame_segments is not None:
            point_range = self.model_frame_segments[frame_idx]

        use_inv_gaussians = (
            use_inverse_deform
            and self.renderer_type in ("2dgs", "3dgs")
            and self.inverse_deform is not None
            and self.deform_rotations
        )

        if use_inv_gaussians:
            means, quats, log_scales = self.inverse_deform_gaussians(frame_idx, point_range=point_range)
            opacities = (
                torch.sigmoid(self.logit_opacities[point_range[0] : point_range[1]]).squeeze(-1)
                if point_range is not None
                else torch.sigmoid(self.logit_opacities).squeeze(-1)
            )
            viewmat = self.get_viewmat(frame_idx)
            if self.renderer_type == "2dgs":
                if point_range is not None:
                    s, e = point_range
                    colors = self.sh_coeffs[s:e]  # (N, K, 3)
                else:
                    colors = self.sh_coeffs  # (N, K, 3)
                return self._rasterize_2dgs_from_gaussians(means, quats, log_scales, opacities, colors, viewmat, K)
            else:
                if point_range is not None:
                    s, e = point_range
                    sh_colors = self.sh_coeffs[s:e]
                else:
                    sh_colors = self.sh_coeffs
                return self._rasterize_3dgs_from_gaussians(means, quats, log_scales, opacities, sh_colors, viewmat, K)

        if use_inverse_deform:
            pts = self.inverse_deform_points(frame_idx, point_range=point_range)
        else:
            pts = (
                self.canonical_points[point_range[0] : point_range[1]]
                if point_range is not None
                else self.canonical_points
            )

        viewmat = self.get_viewmat(frame_idx)

        if self.renderer_type == "2dgs":
            return self.render_2dgs(pts, viewmat, K, point_range=point_range)
        elif self.renderer_type == "3dgs":
            return self.render_3dgs(pts, viewmat, K, point_range=point_range)
        else:
            raise ValueError(f"Unknown renderer_type: {self.renderer_type!r}")

    # ------------------------------------------------------------------
    # Inverse-deformed Gaussians
    # ------------------------------------------------------------------

    def inverse_deform_gaussians(
        self,
        frame_idx: int,
        point_range: Optional[Tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform canonical Gaussian attributes by undoing local deformation
        and transforming back to canonical/world space.

        Flow: canonical → camera (inv_global) → camera_corrected (inv_local) → world (c2w)

        Returns:
            deformed_means:  (N, 3) or (n_subset, 3) in world space
            deformed_quats:  (N, 4) or (n_subset, 4)
            deformed_scales: (N, 3) or (n_subset, 3) in log-space
        If point_range=(s, e) is given, only points [s:e] are processed.
        """
        assert self.inverse_deform is not None, "inverse_deform_net not loaded"

        if point_range is not None:
            s, e = point_range
            means = self.canonical_points[s:e]
            quats = self.quats[s:e]
            scales = self.log_scales[s:e]
        else:
            means = self.canonical_points
            quats = self.quats
            scales = self.log_scales

        N = means.shape[0]
        device = means.device
        frame_idx_tensor = torch.tensor(frame_idx, device=device)

        # --- Step 1: deform means (matching inverse_deform_points flow) ---
        xi_inv_global = self.inv_global_deforms[frame_idx]
        R_inv_global, t_inv_global = se3_exp(xi_inv_global.unsqueeze(0))
        R_inv_global = R_inv_global[0]  # (3, 3)
        t_inv_global = t_inv_global[0]  # (3,)

        # Canonical → camera (detached)
        pts_cam = (R_inv_global.detach() @ means.unsqueeze(-1)).squeeze(-1) + t_inv_global.detach()

        # Camera → camera_corrected (undo local deformation)
        view_idx = frame_idx_tensor.expand(N)
        xi_inv_local = self.inverse_deform(pts_cam, view_idx)  # (N, 6)
        R_inv_local, t_inv_local = se3_exp(xi_inv_local)  # (N, 3, 3), (N, 3)
        pts_cam_corrected = (R_inv_local @ pts_cam.unsqueeze(-1)).squeeze(-1) + t_inv_local

        # Camera_corrected → world via c2w (gradients flow through c2w)
        xi_c2w = self.per_frame_c2w[frame_idx]
        R_c2w, t_c2w = se3_exp(xi_c2w.unsqueeze(0))
        R_c2w = R_c2w[0]  # (3, 3)
        t_c2w = t_c2w[0]  # (3,)
        deformed_means = (R_c2w @ pts_cam_corrected.unsqueeze(-1)).squeeze(-1) + t_c2w

        # --- Step 2: rotate quaternions (R_c2w @ R_inv_local @ R_inv_global) ---
        if self.deform_rotations:
            # R_inv_global is detached, R_inv_local is differentiable, R_c2w is differentiable
            R_total = R_c2w.unsqueeze(0).expand(N, -1, -1) @ R_inv_local @ R_inv_global.unsqueeze(0).expand(N, -1, -1)
            R_quat = rotation_matrix_to_quaternion(R_total)  # (N, 4)
            deformed_quats = quaternion_multiply(R_quat, quats)
        else:
            deformed_quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        deformed_scales = scales

        return deformed_means, deformed_quats, deformed_scales

    def _rasterize_2dgs_from_gaussians(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Low-level 2DGS rasterisation from explicit Gaussian params."""
        from gsplat import rasterization_2dgs

        quats_norm = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scales = log_scales.exp()
        opacities_act = opacities
        colors_act = colors  # expected shape (N, K, 3)
        (
            render_colors,
            render_alphas,
            render_normals,
            surf_normals,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=quats_norm,
            scales=scales,
            opacities=opacities_act,
            colors=colors_act,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=self.width,
            height=self.height,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            sh_degree=self.sh_degree,
            render_mode="RGB+ED",
        )

        rgb = render_colors[..., :3].permute(0, 3, 1, 2)  # (1, 3, H, W)
        depth = render_colors[..., 3:4].permute(0, 3, 1, 2)  # (1, 1, H, W)
        alpha = render_alphas.permute(0, 3, 1, 2)  # (1, 1, H, W)

        result = {
            "rgb": rgb,
            "alpha": alpha,
            "depth": depth,
            "normals": render_normals,
            "surf_normals": surf_normals,
            "distort": render_distort,
            "median_depth": render_median,
            "info": info,
        }

        return result

    def _rasterize_3dgs_from_gaussians(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        opacities: torch.Tensor,
        sh_colors: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Low-level 3DGS rasterisation from explicit Gaussian params."""
        from gsplat.rendering import rasterization

        assert sh_colors.ndim == 3, "3DGS rasterisation requires SH colors shaped (N, K, 3)."

        quats_norm = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scales = log_scales.exp()
        opacities_act = opacities

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats_norm,
            scales=scales,
            opacities=opacities_act,
            colors=sh_colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=self.width,
            height=self.height,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            sh_degree=self.sh_degree,
            render_mode="RGB",
        )

        rgb = render_colors.permute(0, 3, 1, 2)  # (1, 3, H, W)
        alpha = render_alphas.permute(0, 3, 1, 2)  # (1, 1, H, W)

        result = {
            "rgb": rgb,
            "alpha": alpha,
            "depth": None,
            "normals": None,
            "surf_normals": None,
            "distort": None,
            "median_depth": None,
            "info": info,
        }

        return result
