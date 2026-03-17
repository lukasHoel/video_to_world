from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from configs.common import AlignmentDataConfig, KnnBackendConfig, TensorboardConfig
from configs.roma import RomaConfig


@dataclass
class FrameToModelICPConfig(TensorboardConfig, KnnBackendConfig):
    # ---- Paths / naming ----
    root_path: str = ""
    out_path: Optional[str] = None
    out_suffix: str = ""

    # ---- Shared data-loading params (persisted for downstream stages) ----
    alignment: AlignmentDataConfig = AlignmentDataConfig()

    # ---- RoMa matching ----
    roma: RomaConfig = RomaConfig()

    # ---- Shared correspondence threshold ----
    max_corr_dist: float = 0.03

    # ---- Non-rigid ICP hyperparameters ----
    icp_n_iter: int = 100
    icp_early_stopping_patience: Optional[int] = 5
    icp_early_stopping_min_iters: int = 25
    icp_early_stopping_min_delta: Optional[float] = None
    icp_lr: float = 1e-3
    icp_method: str = "point2plane"
    icp_local_twist_reg: float = 0.0
    icp_tv_reg: float = 50.0
    icp_tv_voxel_size: float = 0.05
    icp_tv_every_k: int = 1
    icp_tv_sample_ratio: Optional[float] = 0.1
    icp_color_icp_weight: float = 0.05
    icp_color_icp_max_color_dist: Optional[float] = 0.1
    icp_color_icp_k: int = 10

    save_intermediate_every: int = 10

    # ---- DeformationGrid capacity parameters ----
    deform_log2_hashmap_size: int = 19
    deform_num_levels: int = 16
    deform_n_neurons: int = 64
    deform_n_hidden_layers: int = 2
    deform_min_res: int = 16
    deform_max_res: int = 2048

    # ---- Point filtering (post-ICP, pre-merge) ----
    filter_points: bool = True
    filter_geom_sigma: float = 2.5
    filter_color_sigma: float = 1.5
    filter_worst_pct: float = 0.2
    filter_min_frames: int = 2
    filter_base_percentile: str = "p75"
