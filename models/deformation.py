"""
Forward and inverse 3D deformation models based on SE(3) twists.

- DeformationGrid: hash-grid MLP predicting per-point forward SE(3) twists.
- ViewConditionedInverseDeformation: view-conditioned MLP predicting inverse twists.
- FullInverseDeformationModel: wraps global c2w + local deformation grids with a
  learned inverse, used for training and rendering.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils.geometry import se3_apply, se3_inverse


def _compute_growth_factor(min_res: int, max_res: int, num_levels: int) -> float:
    """Shared helper to compute per-level scale for HashGrid encodings."""
    if num_levels <= 1:
        return 1.0
    return float(np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)))


def _require_tcnn(context: str):
    """Import tinycudann with a consistent error message."""
    try:
        import tinycudann as tcnn  # type: ignore[import]
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"tinycudann is required for {context}. Install tiny-cuda-nn / tinycudann (see project docs)."
        ) from e
    return tcnn


def _normalize_points_to_unit_bbox(
    pts: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize 3D points to the [0,1]^3 cube given a bounding box.

    Used by both forward (DeformationGrid) and inverse (ViewConditionedInverseDeformation)
    deformation networks to share consistent normalization.
    """
    bbox_size = bbox_max - bbox_min
    pts_normalized = (pts - bbox_min.unsqueeze(0)) / bbox_size.unsqueeze(0)
    return torch.clamp(pts_normalized, 0.0, 1.0)


class DeformationGrid(nn.Module):
    """
    HashGrid-based local deformation model.

    Predicts an SE(3) twist per point: xi_local = [omega (3), v (3)].
    Translation is predicted in normalized space and scaled to world space
    using the bbox size.
    """

    def __init__(
        self,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        min_res: int = 16,
        max_res: int = 2048,
        num_levels: int = 16,
        log2_hashmap_size: int = 19,
        n_neurons: int = 64,
        n_hidden_layers: int = 2,
    ):
        super().__init__()

        tcnn = _require_tcnn("DeformationGrid")

        # Store bbox for normalization
        self.register_buffer("bbox_min", bbox_min.clone().detach())
        self.register_buffer("bbox_max", bbox_max.clone().detach())

        growth_factor = _compute_growth_factor(min_res, max_res, num_levels)

        # Hashgrid network that outputs 6 values (se3 parameters)
        self.network = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=6,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": growth_factor,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers,
            },
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Predict local SE(3) parameters for each input point.

        Args:
            pts: (N, 3) points in world space
        Returns:
            xi_local: (N, 6) se3 twist parameters in world space
        """
        # Normalize points to [0, 1] range using bbox
        pts_normalized = _normalize_points_to_unit_bbox(pts, self.bbox_min, self.bbox_max)

        # Pass through hashgrid network (predicts in normalized space)
        xi_local_normalized = self.network(pts_normalized)

        # Split into rotation (omega) and translation (v) components
        omega = xi_local_normalized[..., :3]  # rotation, scale-independent
        v_normalized = xi_local_normalized[..., 3:]  # translation in normalized space

        # Scale translation component to world space
        bbox_size = self.bbox_max - self.bbox_min
        v_world = v_normalized * bbox_size.unsqueeze(0)

        # Combine back into se3 parameters
        xi_local = torch.cat([omega, v_world], dim=-1)

        return xi_local


# ---------------------------
# View-Conditioned Inverse Deformation Network
# ---------------------------
class ViewConditionedInverseDeformation(nn.Module):
    """
    A shared-weight network that takes aligned-space points and a view-conditioning
    code, and predicts a local SE(3) twist to map back to view space.

    Similar to the forward DeformationGrid, this predicts 6 SE(3) parameters
    (omega for rotation, v for translation) and applies:
        view_space_pts = R_inv @ aligned_pts + t_inv
    where R_inv, t_inv = se3_exp(xi_inv)
    """

    def __init__(
        self,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        num_views: int,
        view_embed_dim: int = 32,
        min_res: int = 16,
        max_res: int = 512,
        num_levels: int = 16,
        log2_hashmap_size: int = 19,
        n_neurons: int = 64,
        n_hidden_layers: int = 3,
    ):
        super().__init__()

        tcnn = _require_tcnn("ViewConditionedInverseDeformation")

        self.register_buffer("bbox_min", bbox_min.clone().detach())
        self.register_buffer("bbox_max", bbox_max.clone().detach())
        self.num_views = num_views
        self.view_embed_dim = view_embed_dim

        # Learnable view embeddings
        self.view_embeddings = nn.Embedding(num_views, view_embed_dim)
        nn.init.normal_(self.view_embeddings.weight, mean=0.0, std=0.01)

        # Compute growth factor for hashgrid
        growth_factor = _compute_growth_factor(min_res, max_res, num_levels)

        # Spatial encoding using hashgrid
        self.spatial_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": growth_factor,
            },
        )

        # MLP that takes concatenated [spatial_features, view_embedding] and outputs SE(3) twist (6,)
        spatial_feature_dim = num_levels * 2  # n_levels * n_features_per_level
        input_dim = spatial_feature_dim + view_embed_dim

        self.mlp = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=6,  # SE(3) twist: omega (3) for rotation, v (3) for translation
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers,
            },
        )

    def forward(self, aligned_pts: torch.Tensor, view_idx: torch.Tensor) -> torch.Tensor:
        """
        Predict the inverse deformation SE(3) twist.

        Args:
            aligned_pts: (N, 3) points in aligned/canonical space
            view_idx: (N,) or scalar view indices

        Returns:
            xi_inv: (N, 6) SE(3) twist parameters [omega (3), v (3)]
                    omega: rotation axis-angle
                    v: translation (scaled by bbox size)
        """
        N = aligned_pts.shape[0]

        # Normalize points to [0, 1] for hashgrid
        pts_normalized = _normalize_points_to_unit_bbox(aligned_pts, self.bbox_min, self.bbox_max)

        # Get spatial features from hashgrid
        spatial_features = self.spatial_encoding(pts_normalized)  # (N, spatial_feature_dim)

        # Get view embeddings
        if view_idx.ndim == 0:
            view_idx = view_idx.expand(N)
        view_embed = self.view_embeddings(view_idx)  # (N, view_embed_dim)

        # Concatenate and pass through MLP
        combined = torch.cat([spatial_features, view_embed], dim=-1)
        xi_normalized = self.mlp(combined)  # (N, 6) in normalized space

        # Split into rotation (omega) and translation (v) components
        omega = xi_normalized[..., :3]  # rotation, scale-independent
        v_normalized = xi_normalized[..., 3:]  # translation in normalized space

        # Scale translation component to world space
        bbox_size = self.bbox_max - self.bbox_min
        v_world = v_normalized * bbox_size

        # Combine back into SE(3) parameters
        xi_inv = torch.cat([omega, v_world], dim=-1)

        return xi_inv

    def inverse_warp(self, aligned_pts: torch.Tensor, view_idx: torch.Tensor) -> torch.Tensor:
        """
        Warp aligned-space points back to view space using SE(3) transformation.

        view_space_pts = R_inv @ aligned_pts + t_inv
        where R_inv, t_inv = se3_exp(xi_inv)
        """
        xi_inv = self.forward(aligned_pts, view_idx)  # (N, 6)
        # Apply SE(3) transformation
        view_pts = se3_apply(xi_inv, aligned_pts)
        return view_pts


class FullInverseDeformationModel(nn.Module):
    """
    Wraps frozen forward deformations (c2w + local grid) with a learned inverse.

    Forward:  camera_pts -> local_deform(.) -> se3_apply(c2w, .) -> canonical
    Inverse:  canonical -> c2w^{-1} [detached] -> inv_local(., view) -> camera
    """

    def __init__(
        self,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        num_views: int,
        per_view_global_deform: list[torch.Tensor],
        per_view_local_deform: list,
        view_embed_dim: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.num_views = num_views
        self.register_buffer("bbox_min", bbox_min.clone().detach())
        self.register_buffer("bbox_max", bbox_max.clone().detach())

        # Store c2w SE3 twists (frozen, used for forward deform and inverse path)
        self.per_view_global_deform = nn.ParameterList(
            [nn.Parameter(xi.clone().detach(), requires_grad=False) for xi in per_view_global_deform]
        )

        # Store local deformation networks (frozen, operate in camera space)
        self.per_view_local_deform = nn.ModuleList()
        for deform in per_view_local_deform:
            if isinstance(deform, DeformationGrid):
                for p in deform.parameters():
                    p.requires_grad = False
                self.per_view_local_deform.append(deform)
            else:
                self.per_view_local_deform.append(None)

        # Precompute inverse globals (c2w^{-1})
        self._precompute_inverse_globals()

        # Learnable inverse local deformation network (shared across views)
        self.inverse_local = ViewConditionedInverseDeformation(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            num_views=num_views,
            view_embed_dim=view_embed_dim,
            **kwargs,
        )

    def _precompute_inverse_globals(self):
        """Precompute inverse of c2w SE3 twists."""
        inv_globals = []
        for xi in self.per_view_global_deform:
            xi_inv = se3_inverse(xi.data)
            inv_globals.append(xi_inv)
        self.register_buffer("inv_global_deforms", torch.stack(inv_globals))

    def forward_deform(self, camera_pts: torch.Tensor, view_idx: int) -> torch.Tensor:
        """
        Forward deformation:
            camera_pts  →  local_deform(camera_pts)  →  se3_apply(c2w, ·)  →  canonical/world
        """
        local_deform = self.per_view_local_deform[view_idx]
        if local_deform is not None:
            xi_local = local_deform(camera_pts)
            pts_after_local = se3_apply(xi_local, camera_pts)
        else:
            pts_after_local = camera_pts

        xi_c2w = self.per_view_global_deform[view_idx]
        canonical_pts = se3_apply(xi_c2w, pts_after_local)
        return canonical_pts

    def inverse_deform(self, canonical_pts: torch.Tensor, view_idx: torch.Tensor) -> torch.Tensor:
        """
        Inverse deformation:
            canonical  →  c2w^{-1} [detached]  →  inv_local  →  c2w [grad]  →  world

        The first step to camera space is *detached* so the inverse-local network
        sees a fixed camera-space input.  The final c2w application lets gradients
        flow through the global rigid transform.
        """
        N = canonical_pts.shape[0]
        if view_idx.ndim == 0:
            view_idx_expanded = view_idx.expand(N)
            xi_inv_global = self.inv_global_deforms[view_idx]
            xi_global = self.per_view_global_deform[view_idx.item()]
        else:
            view_idx_expanded = view_idx
            xi_inv_global = self.inv_global_deforms[view_idx]
            xi_global = None  # batched — not supported for gradient path

        # Step 1: canonical → camera via detached c2w^{-1}
        pts_cam = se3_apply(xi_inv_global.detach(), canonical_pts)

        # Step 2: inverse local deformation in camera space
        pts_cam_corrected = self.inverse_local.inverse_warp(pts_cam, view_idx_expanded)

        # Step 3: camera → world via c2w (gradients flow through c2w)
        if xi_global is not None:
            pts_world = se3_apply(xi_global, pts_cam_corrected)
        else:
            # Batched per-point: each point may have a different view
            # Gather per-point c2w twists
            all_c2w = torch.stack([p.data for p in self.per_view_global_deform])
            xi_per_point = all_c2w[view_idx]  # (N, 6)
            pts_world = se3_apply(xi_per_point, pts_cam_corrected)

        return pts_world

    def inverse_deform_to_camera(
        self,
        canonical_pts: torch.Tensor,
        view_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inverse deformation stopping in camera space (without the final c2w).

        Useful for training: returns the camera-space points that should match
        the original observation.
        """
        N = canonical_pts.shape[0]
        if view_idx.ndim == 0:
            view_idx_expanded = view_idx.expand(N)
            xi_inv_global = self.inv_global_deforms[view_idx]
        else:
            view_idx_expanded = view_idx
            xi_inv_global = self.inv_global_deforms[view_idx]

        # canonical → camera (detached)
        pts_cam = se3_apply(xi_inv_global.detach(), canonical_pts)

        # inverse local deformation in camera space
        pts_cam_corrected = self.inverse_local.inverse_warp(pts_cam, view_idx_expanded)
        return pts_cam_corrected

    def get_inverse_twist(self, aligned_pts: torch.Tensor, view_idx: torch.Tensor) -> torch.Tensor:
        """
        Get the learned inverse-local SE(3) twist for regularisation.
        """
        N = aligned_pts.shape[0]
        if view_idx.ndim == 0:
            xi_inv_global = self.inv_global_deforms[view_idx]
            view_idx_expanded = view_idx.expand(N)
        else:
            xi_inv_global = self.inv_global_deforms[view_idx]
            view_idx_expanded = view_idx

        pts_cam = se3_apply(xi_inv_global.detach(), aligned_pts)
        xi_inv_local = self.inverse_local(pts_cam, view_idx_expanded)
        return xi_inv_local
