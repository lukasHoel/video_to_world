from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ExportGSCheckpointToPlyConfig:
    root_path: str = ""
    run: str = ""
    checkpoint_dir: str = ""
    global_opt_subdir: str = "after_global_optimization"

    format: Literal["ply", "splat", "both"] = "both"
    """Which file format(s) to write.

    - 'ply': standard 3DGS PLY (SuperSplat-compatible, keeps SH colors)
    - 'splat': compact Antimatter15/PlayCanvas .splat binary (SuperSplat-compatible, SH0 only)
    - 'both': write both files.
    """

    out_ply: Optional[str] = None
    """Output PLY file path. Default: <checkpoint_dir>/splats_<renderer>.ply"""

    out_splat: Optional[str] = None
    """Output .splat file path. Default: <checkpoint_dir>/splats_<renderer>.splat"""

    max_points: int = -1
    """If > 0, randomly subsample to at most this many points before writing."""

    seed: int = 0
    """Seed for the random subsampling used by `max_points`."""


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

    watch: bool = False
    """Live mode: poll `checkpoint_dir` for new checkpoint files and hot-reload
    the scene automatically (new `checkpoint_*.pt`, `preview_checkpoint.pt`,
    `checkpoint_paused.pt`, or `model_final.pt`)."""

    watch_interval: float = 5.0
    """Polling interval (seconds) in watch mode."""

    wait_for_first: bool = False
    """In watch mode, keep polling (without erroring) until a checkpoint appears.
    Useful for attaching to a training run before its first checkpoint save."""

    export_on_load: Optional[str] = None
    """If set to 'ply', 'splat', or 'both', write the currently loaded model
    state to the checkpoint directory as soon as it is loaded / reloaded
    (e.g. `splats_<renderer>.ply` / `.splat`)."""
