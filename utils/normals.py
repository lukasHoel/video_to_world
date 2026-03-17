"""
Normal estimation utilities.

These helpers are used by ICP variants (point-to-plane) and related algorithms.

Supports multiple backends for neighbor search / normal estimation:
  - 'cpu_kdtree': scipy KDTree + batched SVD
  - 'gpu_kdtree': torch_kdtree GPU KDTree + batched SVD
"""

from __future__ import annotations

from typing import Literal

import time

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

from utils.logging import get_logger

logger = get_logger(__name__)


# Reuse the same backend naming as utils.knn for consistency.
NormalBackend = Literal["cpu_kdtree", "gpu_kdtree"]


def _estimate_normals_open3d_fallback(ref: torch.Tensor, *, k: int = 20) -> torch.Tensor:
    """Fallback to Open3D normal estimation for problematic cases."""
    device = ref.device
    dtype = ref.dtype

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ref.detach().cpu().numpy())
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k),
        fast_normal_computation=False,
    )
    normals_np = np.asarray(pcd.normals)
    return torch.from_numpy(normals_np).to(device=device, dtype=dtype)


def estimate_normals(
    ref: torch.Tensor,
    *,
    k: int = 20,
    orient_towards: torch.Tensor | None = None,
    backend: NormalBackend = "cpu_kdtree",
    start_idx: int = 0,
    end_idx: int | None = None,
    prebuilt_tree=None,
) -> tuple[torch.Tensor, any]:
    """
    Estimate normals for `ref`.

    Backend selection:
      - 'cpu_kdtree': CPU KDTree for neighbors + GPU SVD for PCA (default).
      - 'gpu_kdtree': GPU torch_kdtree for neighbors + GPU SVD for PCA.

    Args:
        start_idx: Only compute normals for points starting from this index (default: 0).
                   All points are still used for neighbor search.
        end_idx: Only compute normals up to (but not including) this index (default: None = all points).
        prebuilt_tree: Optional pre-built KDTree (cKDTree for cpu_kdtree, torch_kdtree for gpu_kdtree).
                      If provided, this tree will be reused instead of building a new one.
    """
    device = ref.device
    M = ref.shape[0]
    if end_idx is None:
        end_idx = M
    compute_start = max(0, start_idx)
    compute_end = min(M, end_idx)
    num_compute = compute_end - compute_start

    # For 'cpu_kdtree' and 'gpu_kdtree', use KDTree+SVD implementation.
    # We build the tree on all points, but only query neighbors for the requested range.
    if backend == "cpu_kdtree":
        ref_np = ref.detach().cpu().numpy().astype(np.float64)

        if prebuilt_tree is not None:
            tree = prebuilt_tree
            t_query_start = time.perf_counter()
            # Query neighbors only for the points we need to compute normals for
            ref_query = ref_np[compute_start:compute_end]
            _, knn_idx = tree.query(ref_query, k=k, workers=-1)  # (num_compute,k)
            t_query_end = time.perf_counter()
            logger.info(
                "estimate_normals[cpu_kdtree]: Reused pre-built KDTree; KNN query (k=%d) on %d points took %.1f ms; computing normals for %d points (indices %d-%d)",
                int(k),
                int(num_compute),
                (t_query_end - t_query_start) * 1000.0,
                int(num_compute),
                int(compute_start),
                int(compute_end),
            )
        else:
            t_build_start = time.perf_counter()
            tree = cKDTree(ref_np)
            t_build_end = time.perf_counter()

            t_query_start = time.perf_counter()
            # Query neighbors only for the points we need to compute normals for
            ref_query = ref_np[compute_start:compute_end]
            _, knn_idx = tree.query(ref_query, k=k, workers=-1)  # (num_compute,k)
            t_query_end = time.perf_counter()

            logger.info(
                "estimate_normals[cpu_kdtree]: KDTree build on %d points took %.1f ms; KNN query (k=%d) on %d points took %.1f ms; computing normals for %d points (indices %d-%d)",
                int(M),
                (t_build_end - t_build_start) * 1000.0,
                int(k),
                int(num_compute),
                (t_query_end - t_query_start) * 1000.0,
                int(num_compute),
                int(compute_start),
                int(compute_end),
            )

        knn_idx = torch.from_numpy(knn_idx.astype(np.int64)).to(device)
    elif backend == "gpu_kdtree":
        if prebuilt_tree is not None:
            tree = prebuilt_tree
            t_query_start = time.perf_counter()
            # Query neighbors only for the points we need to compute normals for
            ref_query = ref[compute_start:compute_end]
            _, knn_idx = tree.query(ref_query, nr_nns_searches=k)  # (num_compute,k)
            if ref.device.type == "cuda":
                torch.cuda.synchronize(device=ref.device)
            t_query_end = time.perf_counter()
            logger.info(
                "estimate_normals[gpu_kdtree]: Reused pre-built KDTree; KNN query (k=%d) on %d points took %.1f ms; computing normals for %d points (indices %d-%d)",
                int(k),
                int(num_compute),
                (t_query_end - t_query_start) * 1000.0,
                int(num_compute),
                int(compute_start),
                int(compute_end),
            )
        else:
            from utils.knn import build_torch_kdtree

            t_build_start = time.perf_counter()
            tree = build_torch_kdtree(ref)
            if ref.device.type == "cuda":
                torch.cuda.synchronize(device=ref.device)
            t_build_end = time.perf_counter()

            t_query_start = time.perf_counter()
            # Query neighbors only for the points we need to compute normals for
            ref_query = ref[compute_start:compute_end]
            _, knn_idx = tree.query(ref_query, nr_nns_searches=k)  # (num_compute,k)
            if ref.device.type == "cuda":
                torch.cuda.synchronize(device=ref.device)
            t_query_end = time.perf_counter()

            logger.info(
                "estimate_normals[gpu_kdtree]: KDTree build on %d points took %.1f ms; KNN query (k=%d) on %d points took %.1f ms; computing normals for %d points (indices %d-%d)",
                int(M),
                (t_build_end - t_build_start) * 1000.0,
                int(k),
                int(num_compute),
                (t_query_end - t_query_start) * 1000.0,
                int(num_compute),
                int(compute_start),
                int(compute_end),
            )

        # torch_kdtree returns (num_compute, k) shape, ensure it's the right dtype
        knn_idx = knn_idx.long()
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    # Only allocate normals for the range we're computing
    normals = torch.zeros((num_compute, 3), device=device, dtype=ref.dtype)
    degenerate_mask = torch.zeros(num_compute, dtype=torch.bool, device=device)

    t_normal_compute_start = time.perf_counter()
    batch_size = 100_000
    for i in range(0, num_compute, batch_size):
        end_idx_batch = min(i + batch_size, num_compute)
        B = end_idx_batch - i
        actual_idx = compute_start + i

        batch_knn_idx = knn_idx[i:end_idx_batch]  # (B, k)
        neighbors = ref[batch_knn_idx.reshape(-1)].reshape(B, k, 3)
        centroid = neighbors.mean(dim=1, keepdim=True)
        X_batch = neighbors - centroid

        X_norm = torch.linalg.norm(X_batch, dim=2)
        batch_degenerate = X_norm.max(dim=1)[0] < 1e-8
        degenerate_mask[i:end_idx_batch] = batch_degenerate

        try:
            _, _, Vt_batch = torch.linalg.svd(X_batch, full_matrices=False)
            normals_batch = Vt_batch[:, -1, :]
            norm_normals = torch.linalg.norm(normals_batch, dim=1, keepdim=True)
            valid_norm = norm_normals.squeeze(1) > 1e-8
            normals_batch[valid_norm] = normals_batch[valid_norm] / norm_normals[valid_norm]
            normals_batch[~valid_norm] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=ref.dtype)
            normals[i:end_idx_batch] = normals_batch
        except RuntimeError as e:
            # Some CUDA backends can throw on invalid values; fallback for this batch.
            logger.warning(
                "Falling back to Open3D normals for batch %d-%d (%s)",
                actual_idx,
                actual_idx + B,
                e,
            )
            try:
                normals[i:end_idx_batch] = _estimate_normals_open3d_fallback(ref[actual_idx : actual_idx + B], k=k)
            except Exception as e2:
                logger.warning("Open3D fallback failed (%s); using default normals", e2)
                normals[i:end_idx_batch] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=ref.dtype)

    if degenerate_mask.any():
        degenerate_indices_local = torch.where(degenerate_mask)[0]
        degenerate_indices_global = compute_start + degenerate_indices_local
        try:
            normals_degen = _estimate_normals_open3d_fallback(ref[degenerate_indices_global], k=k)
            normals[degenerate_indices_local] = normals_degen
        except Exception as e:
            logger.warning("Open3D fallback failed for degenerate cases (%s); using defaults", e)
            normals[degenerate_indices_local] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=ref.dtype)

    if orient_towards is not None:
        ot = orient_towards.to(device)
        # Only orient normals for the range we computed
        ref_computed = ref[compute_start:compute_end]
        if ot.ndim == 1 and ot.shape[0] == 3:
            viewvec = ot.unsqueeze(0) - ref_computed
            flip_mask = (normals * viewvec).sum(dim=1) < 0
            normals[flip_mask] *= -1
        elif ot.shape == (M, 3):
            ot_computed = ot[compute_start:compute_end]
            flip_mask = (normals * ot_computed).sum(dim=1) < 0
            normals[flip_mask] *= -1
        else:
            raise ValueError("orient_towards must be (3,) or (M,3)")

    if torch.isnan(normals).any() or torch.isinf(normals).any():
        nan_mask = torch.isnan(normals).any(dim=1) | torch.isinf(normals).any(dim=1)
        if nan_mask.any():
            nan_indices_global = compute_start + torch.where(nan_mask)[0]
            try:
                normals[nan_mask] = _estimate_normals_open3d_fallback(ref[nan_indices_global], k=k)
            except Exception as e:
                logger.warning("Open3D fallback failed for NaN/inf normals (%s); using defaults", e)
                normals[nan_mask] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=ref.dtype)

    t_normal_compute_end = time.perf_counter()
    normal_compute_time_ms = (t_normal_compute_end - t_normal_compute_start) * 1000.0
    logger.info(
        "estimate_normals[%s]: Normal computation (SVD/PCA) for %d points took %.1f ms",
        backend,
        int(num_compute),
        normal_compute_time_ms,
    )

    # Return normals and the tree (built or pre-built) so caller can reuse it
    return normals, tree if backend in ("cpu_kdtree", "gpu_kdtree") else None
