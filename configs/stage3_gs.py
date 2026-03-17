from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from configs.common import TensorboardConfig


@dataclass
class GSConfig(TensorboardConfig):
    """Configuration for Gaussian splatting training (2DGS or 3DGS)."""

    # ---- Paths ----
    root_path: str = ""
    """Root data directory."""

    run: str = ""
    """Stage 1 run name (e.g. frame_to_model_icp_50_2_offset0)."""

    global_opt_subdir: str = "after_global_optimization"
    """Subdirectory inside the run directory for Stage 2 checkpoints."""

    original_images_dir: str = ""
    """Path to folder with original-resolution images (optional)."""

    out_dir: Optional[str] = None
    """Output directory. Default: auto-generated under run directory."""

    # ---- Inverse deformation ----
    inverse_deform_dir: str = ""
    """Path to directory with trained inverse deformation (inverse_local.pt + config.pt)."""

    # ---- Rendering backend ----
    renderer: Literal["2dgs", "3dgs"] = "2dgs"

    # ---- SH configuration ----
    sh_degree: int = 3
    sh_increase_every: int = 0
    sh_full_from_iter: int = 5000
    sh_freeze_means_when_full_sh: bool = True
    sh_reg_weight: float = 10.0

    # ---- Downsampling ----
    target_num_points: int = 4_000_000

    # ---- Camera optimisation ----
    optimize_cams: bool = True
    lr_cams: float = 1e-4

    # ---- Point optimisation ----
    optimize_positions: bool = True
    lr_positions: float = 1e-5

    # ---- Learning rates ----
    lr_colors: float = 2.5e-3
    lr_opacities: float = 5e-2
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20.0

    # ---- Inverse-deformation of Gaussians ----
    deform_inverse_rotations: bool = True

    # ---- Gaussian init ----
    initial_opacity: float = 0.5
    initial_scale: float = 0.005
    initial_flat_ratio: float = 0.1
    scale_init: Literal["fixed", "knn"] = "knn"
    knn_neighbors: int = 4
    normal_k: int = 20

    # ---- Loss ----
    l1_weight: float = 0.8
    lpips_weight: float = 0.2

    # ---- Regularisation ----
    opacity_reg_weight: float = 0.0
    scale_reg_weight: float = 0.0
    normal_consistency_weight: float = 0.05
    distortion_weight: float = 0.01
    alpha_reg_weight: float = 0.0

    # ---- Training ----
    num_iters: int = 15000
    frames_per_iter: int = 1

    log_every: int = 50
    save_every: int = 5000
    eval_every: int = 1000

    # ---- Scheduler ----
    lr_decay: float = 0.1

    # ---- Evaluation ----
    auto_eval: bool = True
