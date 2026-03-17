from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExportGSCheckpointToPlyConfig:
    root_path: str = ""
    run: str = ""
    checkpoint_dir: str = ""
    global_opt_subdir: str = "after_global_optimization"

    out_ply: Optional[str] = None
    """Output PLY file path. Default: <checkpoint_dir>/splats_3dgs.ply"""

    max_points: int = -1
    """If > 0, randomly subsample to at most this many points before writing."""


@dataclass
class ViewGSCheckpointConfig:
    root_path: str = ""
    run: str = ""
    checkpoint_dir: str = ""
    global_opt_subdir: str = "after_global_optimization"

    port: int = 8080
    """Viewer web port (Viser)."""

    show_cameras: bool = True
    """Show training camera frames (from optimised `per_frame_c2w.*` in the checkpoint)."""

    training_cameras_scale: float = 0.06
    """Axis scale for training camera frames."""

    near_plane: float = 0.01
    far_plane: float = 1e10
    """Initial near/far values for the interactive renderer."""
