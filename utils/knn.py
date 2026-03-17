"""
Nearest-neighbor helpers (KNN).

These are used across alignment / ICP code and losses. Kept in `utils/` so algorithm
implementations can stay focused on optimization/control-flow.

Supports:
  - 'cpu_kdtree': scipy cKDTree on CPU
  - 'gpu_kdtree': torch_kdtree KD-tree on GPU
selectable via a single backend string flag.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree


def nearest_neighbors_torch_kdtree(
    src: torch.Tensor, ref: torch.Tensor, *, K: int = 1, tree=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU KD-tree nearest neighbors using torch_kdtree (https://github.com/thomgrand/torch_kdtree).

    Tree is built on CPU then queried on GPU. Pass pre-built tree to avoid rebuilding
    when ref is static across many queries (e.g. ICP iterations).

    Args:
        src: (N, 3) query points on GPU
        ref: (M, 3) reference points on GPU
        K: number of nearest neighbors (default 1)
        tree: pre-built KD-tree from build_torch_kdtree(ref), or None to build

    Returns:
        idxs: (N,) or (N, K) indices into ref
        d2: (N,) or (N, K) squared distances
    """
    from torch_kdtree import build_kd_tree

    if tree is None:
        tree = build_kd_tree(ref)
    dists, inds = tree.query(src, nr_nns_searches=K)
    if K == 1:
        inds = inds.squeeze(-1) if inds.dim() > 1 else inds
        dists = dists.squeeze(-1) if dists.dim() > 1 else dists
    return inds, dists


def build_torch_kdtree(ref: torch.Tensor):
    """Build torch_kdtree for repeated queries. Ref must be on CUDA."""
    from torch_kdtree import build_kd_tree

    return build_kd_tree(ref)


def nearest_neighbors(
    src: torch.Tensor, ref: torch.Tensor, *, chunk: int = 50_000
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Brute-force nearest neighbors using chunked `torch.cdist` (GPU-friendly fallback).

    Args:
        src: (N,3)
        ref: (M,3)
        chunk: chunk size over src points

    Returns:
        idxs: (N,) indices into ref
        d2:   (N,) squared distances
    """
    N = src.shape[0]
    device = src.device

    idxs = torch.empty(N, dtype=torch.long, device=device)
    d2 = torch.empty(N, device=device)

    for i in range(0, N, chunk):
        s = src[i : i + chunk]  # (C, 3)
        dist = torch.cdist(s, ref)  # (C, M)
        dmin, argmin = torch.min(dist, dim=1)
        idxs[i : i + chunk] = argmin
        d2[i : i + chunk] = dmin * dmin

    return idxs, d2


def build_kdtree(ref: torch.Tensor) -> cKDTree:
    """
    Build a scipy cKDTree for a reference point cloud.

    Args:
        ref: (M,3) torch tensor (any device)

    Returns:
        scipy.spatial.cKDTree built on CPU float64 numpy points
    """
    ref_np = ref.detach().cpu().numpy().astype(np.float64)
    return cKDTree(ref_np)


def nearest_neighbors_kdtree(src: torch.Tensor, kdtree: cKDTree, *, K: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Nearest neighbors using scipy's vectorized cKDTree queries.

    Args:
        src: (N,3) torch tensor (any device)
        kdtree: cKDTree built on ref
        K: number of nearest neighbors (default 1)

    Returns:
        idxs: (N,) or (N, K) long indices into ref
        d2:   (N,) or (N, K) float32 squared distances
    """
    device = src.device

    src_np = src.detach().cpu().numpy().astype(np.float64)
    distances, indices = kdtree.query(src_np, k=K, workers=-1)

    if K == 1:
        # scipy returns shape (N,) when k=1
        idxs = torch.from_numpy(indices.astype(np.int64)).to(device)
        d2 = torch.from_numpy((distances**2).astype(np.float32)).to(device)
    else:
        # (N, K)
        idxs = torch.from_numpy(indices.astype(np.int64)).to(device)
        d2 = torch.from_numpy((distances**2).astype(np.float32)).to(device)
    return idxs, d2


# ---------------------------------------------------------------------
# Unified KNN backend helper
# ---------------------------------------------------------------------

KNNBackend = Literal["cpu_kdtree", "gpu_kdtree"]


def query_knn_with_backend(
    src: torch.Tensor,
    ref: torch.Tensor,
    *,
    K: int = 1,
    backend: KNNBackend = "cpu_kdtree",
    chunk: int = 50_000,
    cpu_tree: Optional[cKDTree] = None,
    gpu_tree=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified interface to query nearest neighbors with a chosen backend.

    Args:
        src: (N, 3) query points
        ref: (M, 3) reference points
        K: number of neighbors (currently only K=1 is used by callers)
        backend:
            - 'cpu_kdtree': scipy cKDTree on CPU
            - 'gpu_kdtree': torch_kdtree KD-tree on GPU (if available)
        chunk: chunk size for brute-force torch.cdist fallback (unused for KD-tree backends)
        cpu_tree: optional pre-built cKDTree for 'cpu_kdtree' (built once and reused)
        gpu_tree: optional pre-built torch_kdtree for 'gpu_kdtree' (built once and reused)

    Returns:
        idxs: (N,) or (N, K) indices into ref
        d2:   (N,) or (N, K) squared distances
    """
    if backend == "cpu_kdtree":
        if cpu_tree is None:
            cpu_tree = build_kdtree(ref)
        return nearest_neighbors_kdtree(src, cpu_tree, K=K)

    if backend == "gpu_kdtree":
        if gpu_tree is None:
            gpu_tree = build_torch_kdtree(ref)
        return nearest_neighbors_torch_kdtree(src, ref, K=K, tree=gpu_tree)

    raise ValueError(f"Unknown KNN backend: {backend!r}")
