from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalGSConfig:
    # ---- Paths ----
    root_path: str = ""
    """DA3 scene root path (contains exports/npz and optionally gs_video/...)."""

    run: str = ""
    """Run name (frame_to_model_icp_...)."""

    checkpoint_dir: str = ""
    """Directory produced by train_gs.py (contains model_final.pt + config.json)."""

    global_opt_subdir: str = "after_global_optimization"
    """Global-optimization checkpoint subdirectory inside `root_path/run/`."""

    transforms_path: Optional[str] = None
    """Path to transforms JSON. Default: <root_path>/gs_video/0000_extend_transforms.json"""

    # ---- Output ----
    out_dir: Optional[str] = None
    """Output directory. Default: <checkpoint_dir>/gs_video_eval"""

    original_images_dir: str = ""
    """Optional path to folder with original-resolution images for GT comparison."""

    save_images: bool = True
    save_video: bool = True
    fps: int = 30
    max_frames: int = -1
    """If > 0, only render first max_frames poses (for gs_video path)."""

    # ---- Which pose families to render ----
    render_gs_video_path: bool = True
    render_input_poses: bool = True
    render_optimised_poses: bool = True

    # ---- Rendering resolution overrides (optional) ----
    override_width: int = -1
    override_height: int = -1

    # ---- Canonical point cloud ----
    target_num_points: int = 1_000_000
    white_background: bool = False
