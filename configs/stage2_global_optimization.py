from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from configs.common import KnnBackendConfig, TensorboardConfig


@dataclass
class GlobalOptimizationConfig(TensorboardConfig, KnnBackendConfig):
    """Stage 2: Global optimization (LOO consensus + joint deformation)."""

    # KNN backend. GPU KD-tree default (large-N throughput).
    knn_backend: str = "gpu_kdtree"

    # Paths
    root_path: str = ""
    """Dataset root."""

    run: str = ""
    """Stage 1 run name (e.g. `frame_to_model_icp_50_2_offset0`)."""

    checkpoint_subdir: str = "after_non_rigid_icp"
    """Input checkpoint subdir (within run dir)."""

    out_subdir: str = "after_global_optimization"
    """Output subdir (within run dir)."""

    # LOO consensus (geometry)
    loo_loss_weight: float = 1.0
    loo_k_neighbors: int = 5
    """KNN candidate pool size per point (LOO filtering applied after KNN).

    Notes:
    - Larger pool → higher probability of at least one valid cross-frame match.
    - Too small (e.g. 1) → many rows with zero valid LOO neighbours.
    """
    loo_max_corr_dist: float = 0.01 * 3.125
    """Max correspondence distance (LOO filtering)."""
    loo_normal_k: int = 20
    """Normal estimation K (PCA/SVD over KNN neighbourhood)."""
    loo_kdtree_rebuild_every: int = 50
    """KNN/normals rebuild period (iterations)."""

    loo_max_pairs_per_iter: Optional[int] = 200_000
    """LOO pair budget per iteration (None disables subsampling; full objective).

    Importance-weighted estimator; expectation matches full loss.
    Trade-off: smaller → lower memory/compute, higher gradient variance.
    """

    loo_pairs_per_src: int = 1
    """Subsampling: max #pairs per sampled source row.

    Distinct from `loo_k_neighbors`:
    - `loo_k_neighbors`: candidate pool size (KNN query)
    - `loo_pairs_per_src`: draws from valid subset of that pool

    Constraint: `loo_pairs_per_src <= loo_k_neighbors`.
    Typical: 1–2.
    """

    deform_chunk_size: int = 200_000
    """Deformation forward chunk size (TCNN peak-memory control)."""

    # Anchoring (stability / drift control)
    anchor_loss_weight: float = 1000.0
    """Anchor weight (MSE on twists at random points)."""
    anchor_n_samples: int = 4096
    """Anchor samples per frame."""

    # TV regularization (optional)
    tv_reg: float = 0.0
    """TV weight."""
    tv_voxel_size: float = 0.05
    """Voxel size for TV sampling."""
    tv_every_k: int = 1
    """TV evaluation period (iterations)."""
    tv_sample_ratio: Optional[float] = 0.1
    """TV random subsampling ratio (None: full)."""

    # LOO photometric term (Colored-ICP style; geometry-only)
    loo_color_icp_weight: float = 0.05
    """Photometric weight."""

    loo_color_icp_k: int = 10
    """KNN size for tangent-plane color gradient fit."""

    loo_color_icp_max_color_dist: Optional[float] = 0.1
    """Optional |I_p - I_q| gate (intensity; [0,1])."""

    # Surface sharpening (reuse LOO KNN structure)
    thin_shell_weight: float = 1000.0
    """Thin-shell weight."""

    # Optimization
    lr: float = 1e-3
    """Adam learning rate."""
    n_iters: int = 150
    """Iterations."""

    save_intermediate_every_n: int = 50
