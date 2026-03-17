from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RomaConfig:
    use_roma_matching: bool = True
    roma_version: str = "v2"
    roma_model: str = "indoor"
    roma_num_samples: int = 5000
    roma_certainty_threshold: float = 0.5
    roma_max_references: int = 20
    roma_reference_sampling: str = "recent_and_strided"
    roma_loss_weight: float = 1.0
    roma_max_corr_dist: Optional[float] = 1.0
