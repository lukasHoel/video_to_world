"""
Total Variation (TV) regularization loss for SE(3) deformation fields.
"""

from __future__ import annotations

import torch


def build_voxel_grid(bounds_min: torch.Tensor, bounds_max: torch.Tensor, voxel_size: float):
    """
    Build a regular voxel grid of sample points.

    This version places samples at voxel centres, suitable for generic
    continuous fields (e.g. hash-grid based `DeformationGrid`).

    Args:
        bounds_min, bounds_max: (3,)

    Returns:
        pts: (N,3) voxel centers
        dims: (Nx,Ny,Nz)
    """
    extent = bounds_max - bounds_min
    dims = torch.ceil(extent / voxel_size).long()

    xs = torch.linspace(
        bounds_min[0] + voxel_size * 0.5,
        bounds_min[0] + voxel_size * (dims[0] - 0.5),
        int(dims[0]),
        device=bounds_min.device,
        dtype=bounds_min.dtype,
    )
    ys = torch.linspace(
        bounds_min[1] + voxel_size * 0.5,
        bounds_min[1] + voxel_size * (dims[1] - 0.5),
        int(dims[1]),
        device=bounds_min.device,
        dtype=bounds_min.dtype,
    )
    zs = torch.linspace(
        bounds_min[2] + voxel_size * 0.5,
        bounds_min[2] + voxel_size * (dims[2] - 0.5),
        int(dims[2]),
        device=bounds_min.device,
        dtype=bounds_min.dtype,
    )

    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
    pts = grid.reshape(-1, 3)
    return pts, dims


def voxel_neighbors_6(dims: torch.Tensor | tuple[int, int, int], device: torch.device):
    """
    dims = (Nx,Ny,Nz)
    Returns:
        idx_i: (M,) indices of voxel i
        idx_j: (M,) indices of neighbors j
    """
    Nx, Ny, Nz = (int(dims[0]), int(dims[1]), int(dims[2]))
    N = Nx * Ny * Nz

    base = torch.arange(N, device=device)
    ix = base // (Ny * Nz)
    iy = (base % (Ny * Nz)) // Nz
    iz = base % Nz

    offs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    idx_i = []
    idx_j = []
    for ox, oy, oz in offs:
        nx = ix + ox
        ny = iy + oy
        nz = iz + oz

        valid = (nx >= 0) & (nx < Nx) & (ny >= 0) & (ny < Ny) & (nz >= 0) & (nz < Nz)
        i_idx = base[valid]
        j_idx = nx[valid] * Ny * Nz + ny[valid] * Nz + nz[valid]
        idx_i.append(i_idx)
        idx_j.append(j_idx)

    return torch.cat(idx_i, dim=0), torch.cat(idx_j, dim=0)


def tv_loss(
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    voxel_size: float,
    deform,
    sample_ratio: float | None = None,
    input_points: torch.Tensor | None = None,
    num_jittered_points: int = 0,
    jitter_scale: float | None = None,
) -> torch.Tensor:
    """
    TV-style regularizer on a continuous SE(3) deformation field.

    If `input_points` is provided, samples the deformation at:
    1. Input point locations (optionally subsampled by sample_ratio)
    2. Neighbors of input points (6-neighborhood defined by voxel_size)
    3. N randomly jittered points around input points (if num_jittered_points > 0)

    Otherwise, falls back to the original voxel grid sampling approach.

    Args:
        bounds_min, bounds_max: (3,) bounding box
        voxel_size: Size of voxel for neighbor definition
        deform: Deformation network (DeformationGrid)
        sample_ratio: If in (0,1) and input_points is provided, randomly subsample this fraction of input points.
                      If None or 1.0, use all input points. For voxel grid fallback, subsamples voxels.
        input_points: (N, 3) optional input points to sample at
        num_jittered_points: Number of randomly jittered points per input point (default: 0)
        jitter_scale: Scale of jitter noise (default: voxel_size * 0.5)
    """
    device = bounds_min.device

    # New approach: sample at input points and their neighbors
    if input_points is not None:
        input_points = input_points.reshape(-1, 3).to(device=device, dtype=bounds_min.dtype)
        N_input = input_points.shape[0]

        if N_input == 0:
            return torch.tensor(0.0, device=device, dtype=bounds_min.dtype)

        # Randomly subsample input points if sample_ratio is provided
        if sample_ratio is not None:
            sample_ratio = float(sample_ratio)
            if 0.0 < sample_ratio < 1.0:
                num_keep = max(1, int(N_input * sample_ratio))
                perm = torch.randperm(N_input, device=device)[:num_keep]
                input_points = input_points[perm]
                N_input = input_points.shape[0]

        # Define 6-neighborhood offsets
        offsets = torch.tensor(
            [
                [voxel_size, 0, 0],
                [-voxel_size, 0, 0],
                [0, voxel_size, 0],
                [0, -voxel_size, 0],
                [0, 0, voxel_size],
                [0, 0, -voxel_size],
            ],
            device=device,
            dtype=bounds_min.dtype,
        )  # (6, 3)

        # Collect all points that need TV constraints (input points + jittered points)
        all_center_pts = [input_points]

        # Generate jittered points if requested
        if num_jittered_points > 0:
            if jitter_scale is None:
                jitter_scale = voxel_size * 0.5

            # Generate random jitter: (N_input, num_jittered_points, 3)
            jitter = (
                torch.randn(
                    N_input,
                    num_jittered_points,
                    3,
                    device=device,
                    dtype=bounds_min.dtype,
                )
                * jitter_scale
            )

            # Add jitter to input points
            jittered_pts = input_points.unsqueeze(1) + jitter  # (N_input, num_jittered_points, 3)
            jittered_pts = jittered_pts.reshape(-1, 3)  # (N_input * num_jittered_points, 3)
            all_center_pts.append(jittered_pts)

        # Concatenate all center points (input + jittered)
        center_pts = torch.cat(all_center_pts, dim=0)  # (N_center, 3)
        N_center = center_pts.shape[0]

        # Compute neighbors for all center points: (N_center, 6, 3)
        center_pts_expanded = center_pts.unsqueeze(1)  # (N_center, 1, 3)
        offsets_expanded = offsets.unsqueeze(0)  # (1, 6, 3)
        neighbor_pts = center_pts_expanded + offsets_expanded  # (N_center, 6, 3)
        neighbor_pts = neighbor_pts.reshape(-1, 3)  # (N_center * 6, 3)

        # Concatenate center points and neighbors for evaluation
        all_sample_pts = torch.cat([center_pts, neighbor_pts], dim=0)  # (N_center + N_center*6, 3)

        # Evaluate deformation at all sample points
        twist_all = deform(all_sample_pts)  # (N_center + N_center*6, 6)

        # Split twists into center and neighbor twists
        twist_center = twist_all[:N_center]  # (N_center, 6)
        twist_neighbor = twist_all[N_center:]  # (N_center * 6, 6)
        twist_neighbor = twist_neighbor.reshape(N_center, 6, 6)  # (N_center, 6, 6)

        # Vectorized edge construction: each center point connects to its 6 neighbors
        # Expand center twists to match neighbor structure: (N_center, 6, 6)
        twist_center_expanded = twist_center.unsqueeze(1).expand(-1, 6, -1)  # (N_center, 6, 6)

        # Compute differences: (N_center, 6, 6)
        diff = twist_center_expanded - twist_neighbor

        # Compute TV loss: total variation on twists
        loss_tv = (diff**2).sum(dim=-1).mean()  # Sum over 6 twist channels, then mean over all edges

        return loss_tv

    # Generic TV for continuous fields (e.g. hash-grid DeformationGrid).
    # We use a TV-style objective on twists, but build neighbors only for a
    # (possibly) subsampled set of voxels to avoid O(N) neighbor construction.

    extent = bounds_max - bounds_min
    dims = torch.ceil(extent / voxel_size).long().to(device)  # (3,)
    Nx, Ny, Nz = int(dims[0]), int(dims[1]), int(dims[2])
    N = Nx * Ny * Nz

    if N == 0:
        return torch.tensor(0.0, device=device, dtype=bounds_min.dtype)

    # Base voxel indices [0, N)
    base = torch.arange(N, device=device, dtype=torch.long)

    # Optional subsampling of voxels (rather than edges) to build a sparser
    # neighborhood graph.
    sr = float(sample_ratio) if sample_ratio is not None else 1.0
    if 0.0 < sr < 1.0:
        num_keep = max(1, int(N * sr))
        perm = torch.randperm(N, device=device)[:num_keep]
        base = base[perm]

    # Convert flat indices -> (ix,iy,iz)
    NyNz = Ny * Nz
    ix = base // NyNz
    iy = (base % NyNz) // Nz
    iz = base % Nz

    # Build 6-neighborhood edges only for these sampled voxels.
    offs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    idx_i_list = []
    idx_j_list = []
    for ox, oy, oz in offs:
        nx = ix + ox
        ny = iy + oy
        nz = iz + oz
        valid = (nx >= 0) & (nx < Nx) & (ny >= 0) & (ny < Ny) & (nz >= 0) & (nz < Nz)
        if not torch.any(valid):
            continue
        i_idx = base[valid]
        j_idx = nx[valid] * NyNz + ny[valid] * Nz + nz[valid]
        idx_i_list.append(i_idx)
        idx_j_list.append(j_idx)

    if not idx_i_list:
        return torch.tensor(0.0, device=device, dtype=bounds_min.dtype)

    idx_i = torch.cat(idx_i_list, dim=0)
    idx_j = torch.cat(idx_j_list, dim=0)

    # Evaluate deformation only at unique voxel centers appearing in edges.
    uniq = torch.unique(torch.cat([idx_i, idx_j], dim=0), sorted=False)
    # Map uniq indices to 3D voxel centers (same convention as build_voxel_grid).
    ux = uniq // NyNz
    uy = (uniq % NyNz) // Nz
    uz = uniq % Nz
    voxel_size_t = bounds_min.new_tensor(voxel_size)
    pts_uniq = torch.stack(
        [
            bounds_min[0] + voxel_size_t * (ux.to(bounds_min.dtype) + 0.5),
            bounds_min[1] + voxel_size_t * (uy.to(bounds_min.dtype) + 0.5),
            bounds_min[2] + voxel_size_t * (uz.to(bounds_min.dtype) + 0.5),
        ],
        dim=-1,
    )  # (U,3)

    twist_uniq = deform(pts_uniq)  # (U,6)

    # Map original voxel indices -> compact [0..U-1] indices
    map_full = torch.full((N,), -1, device=device, dtype=torch.long)
    map_full[uniq] = torch.arange(uniq.shape[0], device=device, dtype=torch.long)
    ii = map_full[idx_i]
    jj = map_full[idx_j]

    diff = twist_uniq[ii] - twist_uniq[jj]  # (M,6)
    loss_tv = (diff**2).sum(-1).mean()
    return loss_tv
