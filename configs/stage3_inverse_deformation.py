from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from configs.common import KnnBackendConfig, TensorboardConfig


@dataclass
class TrainInverseDeformationConfig(TensorboardConfig, KnnBackendConfig):
    """Configuration for training inverse deformation network (Stage 3.1)."""

    root_path: str = ""
    """Root path containing the data."""

    run: str = ""
    """Run name (e.g., frame_to_model_icp_50_2_offset0)."""

    checkpoint_subdir: str = "after_global_optimization"
    """Subdirectory containing checkpoints."""

    # Training parameters
    n_epochs: Optional[int] = None
    batch_size: int = 8192
    lr: float = 1e-3
    cycle_weight: float = 0.1
    magnitude_weight: float = 1e-3
    smoothness_weight: float = 1e-3
    num_forward_samples: int = 10000
    num_interp_samples: int = 5000
    regenerate_every: int = 10

    # Network architecture
    view_embed_dim: int = 32
    min_res: int = 16
    max_res: int = 2048
    num_levels: int = 16
    log2_hashmap_size: int = 19
    n_neurons: int = 64
    n_hidden_layers: int = 3

    # Output
    out_path: Optional[str] = None
    save_validation_plys: bool = True
