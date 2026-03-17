import json
import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import open3d as o3d
import torch

from configs.common import AlignmentDataConfig
from models.deformation import DeformationGrid, ViewConditionedInverseDeformation
from models.roma_matcher import MatchHistory, RoMaMatchData

logger = logging.getLogger(__name__)

FirstLocalMode = Literal["none", "dummy"]


def write_point_cloud_ply(
    path: str,
    points: torch.Tensor,
    colors: torch.Tensor | None = None,
    *,
    uniform_color: tuple[float, float, float] | None = None,
) -> None:
    """Write a point cloud to a PLY file.

    This is used for checkpoint exports and lightweight debugging visualisations.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy().reshape(-1, 3))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy().reshape(-1, 3).clip(0, 1))
    if uniform_color is not None:
        pcd.paint_uniform_color(list(uniform_color))
    o3d.io.write_point_cloud(path, pcd)


def load_aligned_point_cloud(
    checkpoint_dir: str,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load aligned_points.ply from a checkpoint directory. Returns (N,3) points, (N,3) colors."""
    ply_path = os.path.join(checkpoint_dir, "aligned_points.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"aligned_points.ply not found: {ply_path}")

    aligned_pcd = o3d.io.read_point_cloud(ply_path)
    points = torch.from_numpy(np.array(aligned_pcd.points)).to(device).to(torch.float32).reshape(-1, 3)
    colors = torch.from_numpy(np.array(aligned_pcd.colors)).to(device).to(torch.float32).reshape(-1, 3)
    return points, colors


def load_deformation_checkpoints(
    checkpoint_dir: str,
    device: str = "cuda",
    *,
    first_local: FirstLocalMode = "none",
    allow_rigid_fallback: bool = True,
) -> tuple[list[torch.Tensor], list, torch.Tensor, torch.Tensor]:
    """
    Load per-frame deformation checkpoints (global SE3 twists + local deformation grids).

    Returns (per_frame_global_deform, per_frame_local_deform, bbox_min, bbox_max).
    """
    # -----------------------
    # Global deformations
    # -----------------------
    per_frame_global_deform: list[torch.Tensor] = []
    i = 0
    while True:
        path = os.path.join(checkpoint_dir, f"per_frame_global_deform_{i:05d}.pt")
        if not os.path.exists(path) and allow_rigid_fallback:
            path = os.path.join(checkpoint_dir, f"per_frame_global_rigid_{i:05d}.pt")
        if not os.path.exists(path):
            break
        per_frame_global_deform.append(torch.load(path, weights_only=False, map_location=device))
        i += 1

    if len(per_frame_global_deform) == 0:
        raise ValueError(f"No global deform files found in {checkpoint_dir}")

    logger.info("Loaded %d global deformations", len(per_frame_global_deform))

    # -----------------------
    # Local deformations
    # -----------------------
    if first_local == "none":
        per_frame_local_deform: list = [None]
    elif first_local == "dummy":
        per_frame_local_deform = [lambda x: torch.zeros((x.shape[0], 6), device=x.device, dtype=torch.float32)]
    else:
        raise ValueError(f"Unknown first_local mode: {first_local}")

    bbox_min: torch.Tensor | None = None
    bbox_max: torch.Tensor | None = None

    i = 1
    while True:
        path = os.path.join(checkpoint_dir, f"per_frame_local_deform_{i:05d}.pt")
        if not os.path.exists(path):
            break

        state_dict = torch.load(path, weights_only=False, map_location=device)

        if bbox_min is None or bbox_max is None:
            if "bbox_min" not in state_dict or "bbox_max" not in state_dict:
                raise ValueError(f"State dict at {path} missing bbox_min/bbox_max")
            bbox_min = state_dict["bbox_min"].to(device)
            bbox_max = state_dict["bbox_max"].to(device)

        deform_grid = DeformationGrid(bbox_min, bbox_max, max_res=2048).to(device)
        deform_grid.load_state_dict(state_dict)
        per_frame_local_deform.append(deform_grid)
        i += 1

    logger.info("Loaded %d local deformations (including first)", len(per_frame_local_deform))

    if bbox_min is None or bbox_max is None:
        raise ValueError("Could not extract bbox from local deformation checkpoints")

    return per_frame_global_deform, per_frame_local_deform, bbox_min, bbox_max


def load_roma_match_history(checkpoint_dir: str, device: str = "cuda") -> MatchHistory | None:
    """Load roma_match_history.pt if present, else None."""
    path = os.path.join(checkpoint_dir, "roma_match_history.pt")
    if not os.path.exists(path):
        return None

    data = torch.load(path, weights_only=False, map_location=device)
    mh = MatchHistory()
    for i in range(len(data["frame_pairs"])):
        src_idx, ref_idx = data["frame_pairs"][i]
        mh.add_matches(
            [
                RoMaMatchData(
                    src_frame_idx=src_idx,
                    ref_frame_idx=ref_idx,
                    kpts_src=data["kpts_src"][i].to(device),
                    kpts_ref=data["kpts_ref"][i].to(device),
                    certainty=data["certainty"][i].to(device),
                )
            ]
        )
    return mh


@dataclass(frozen=True)
class RomaModelIndexData:
    model_frame_segments: torch.Tensor
    model_valid_pixel_indices_list: list[torch.Tensor]


def load_roma_model_index_data(checkpoint_dir: str, device: str = "cuda") -> RomaModelIndexData | None:
    """Load model_frame_segments + model_valid_pixel_indices_list if present, else None."""
    segments_path = os.path.join(checkpoint_dir, "model_frame_segments.pt")
    pixel_indices_path = os.path.join(checkpoint_dir, "model_valid_pixel_indices_list.pt")
    if not (os.path.exists(segments_path) and os.path.exists(pixel_indices_path)):
        return None

    model_frame_segments = torch.load(segments_path, weights_only=False, map_location=device)
    model_valid_pixel_indices_list = [
        idx.to(device) for idx in torch.load(pixel_indices_path, weights_only=False, map_location=device)
    ]
    return RomaModelIndexData(
        model_frame_segments=model_frame_segments,
        model_valid_pixel_indices_list=model_valid_pixel_indices_list,
    )


@dataclass(frozen=True)
class AlignmentDataParams(AlignmentDataConfig):
    """Shared alignment data config parameters."""


def load_json_config(path: str, *, required: bool = True) -> dict:
    """
    Load a JSON config file.

    If required=True and the file does not exist or cannot be parsed, raises a
    descriptive ValueError.
    """
    if not os.path.exists(path):
        if required:
            raise ValueError(f"Required config file not found: {path}")
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        if required:
            raise ValueError(f"Failed to load or parse config JSON at {path}: {exc}") from exc
        logger.warning("Failed to load/parse optional config at %s (%s)", path, exc)
        return {}


def load_stage_config(
    root_path: str,
    run: str,
    subdir: Optional[str] = None,
    filename: str = "config.json",
    *,
    required: bool = True,
) -> dict:
    """
    Resolve and load a stage config JSON under:
        <root_path>/<run>/[subdir]/<filename>
    """
    run_dir = os.path.join(root_path, run)
    cfg_path = os.path.join(run_dir, subdir, filename) if subdir else os.path.join(run_dir, filename)
    return load_json_config(cfg_path, required=required)


def load_alignment_data_params(root_path: str, run: str) -> AlignmentDataParams:
    """
    Load alignment (Stage 1) data parameters from the Stage 1 persisted config.

    This is REQUIRED for downstream stages; if the config is missing or does not
    contain the expected keys, a ValueError is raised with a clear message.
    """
    cfg = load_stage_config(
        root_path,
        run,
        subdir="after_non_rigid_icp",
        filename="config.json",
        required=True,
    )

    # New unified schema: nested under "alignment".
    if "alignment" not in cfg or not isinstance(cfg["alignment"], dict):
        raise ValueError(
            f"Stage 1 config is missing 'alignment' dict for run='{run}'. Re-run Stage 1 (frame_to_model_icp.py)."
        )

    a = cfg["alignment"]

    missing_keys = [
        k
        for k in (
            "num_frames",
            "stride",
            "offset",
            "conf_thresh_percentile",
            "conf_mode",
            "conf_voxel_size",
        )
        if k not in a
    ]
    if missing_keys:
        raise ValueError(
            "Alignment config.json is missing required keys "
            f"{missing_keys} for run='{run}'. Make sure Stage 1 "
            "(frame_to_model_icp.py) completed successfully."
        )

    num_frames = int(a["num_frames"])
    stride = int(a["stride"])
    offset = int(a["offset"])

    conf_thresh_percentile = float(a["conf_thresh_percentile"])
    conf_mode = str(a["conf_mode"])
    conf_voxel_size = float(a["conf_voxel_size"])

    conf_local_percentile = a.get("conf_local_percentile", None)
    if conf_local_percentile is not None:
        conf_local_percentile = float(conf_local_percentile)

    conf_global_percentile = a.get("conf_global_percentile", None)
    if conf_global_percentile is not None:
        conf_global_percentile = float(conf_global_percentile)

    conf_voxel_min_count_percentile = a.get("conf_voxel_min_count_percentile", None)
    if conf_voxel_min_count_percentile is not None:
        conf_voxel_min_count_percentile = float(conf_voxel_min_count_percentile)

    logger.info(
        "Loaded alignment data params for run '%s': "
        "num_frames=%d, stride=%d, offset=%d, conf_thresh_percentile=%.1f, "
        "conf_mode=%s, conf_local_percentile=%s, conf_global_percentile=%s, "
        "conf_voxel_size=%.4f, conf_voxel_min_count_percentile=%s",
        run,
        num_frames,
        stride,
        offset,
        conf_thresh_percentile,
        conf_mode,
        str(conf_local_percentile),
        str(conf_global_percentile),
        conf_voxel_size,
        str(conf_voxel_min_count_percentile),
    )

    return AlignmentDataParams(
        num_frames=num_frames,
        stride=stride,
        offset=offset,
        conf_thresh_percentile=conf_thresh_percentile,
        conf_mode=conf_mode,
        conf_local_percentile=conf_local_percentile,
        conf_global_percentile=conf_global_percentile,
        conf_voxel_size=conf_voxel_size,
        conf_voxel_min_count_percentile=conf_voxel_min_count_percentile,
    )


def load_inverse_local_from_checkpoint(
    checkpoint_dir: str,
    device: str | torch.device = "cuda",
) -> tuple[ViewConditionedInverseDeformation, dict]:
    """
    Load a pretrained ViewConditionedInverseDeformation from a train_inverse_deformation run.

    Expects files:
      - config.pt  (hyperparameters + bbox)
      - inverse_local.pt  (state dict for the inverse network)
    """
    config_path = os.path.join(checkpoint_dir, "config.pt")
    weights_path = os.path.join(checkpoint_dir, "inverse_local.pt")
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find inverse deformation config/weights in {checkpoint_dir}")

    cfg = torch.load(config_path, map_location=device)

    bbox_min = cfg["bbox_min"].to(device)
    bbox_max = cfg["bbox_max"].to(device)
    num_views = int(cfg["num_views"])

    inverse_local = ViewConditionedInverseDeformation(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        num_views=num_views,
        view_embed_dim=int(cfg["view_embed_dim"]),
        min_res=int(cfg["min_res"]),
        max_res=int(cfg["max_res"]),
        num_levels=int(cfg["num_levels"]),
        log2_hashmap_size=int(cfg["log2_hashmap_size"]),
        n_neurons=int(cfg["n_neurons"]),
        n_hidden_layers=int(cfg["n_hidden_layers"]),
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device)
    inverse_local.load_state_dict(state_dict)

    logger.info(
        "Loaded pretrained inverse deformation network from %s (num_views=%d, view_embed_dim=%d)",
        checkpoint_dir,
        num_views,
        int(cfg["view_embed_dim"]),
    )

    return inverse_local, cfg
