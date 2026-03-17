"""
End-to-end pipeline: DA3 data --> canonical reconstruction --> 2DGS/3DGS.

Chains Stage 1 (iterative alignment), Stage 2 (global optimization),
Stage 3.1 (inverse deformation training), and Stage 3.2 (Gaussian splatting)
for a single scene.

Usage:
    python run_reconstruction.py \
        --config.root-path /path/to/da3_scene \
        --config.stage1.alignment.num-frames 50 --config.stage1.alignment.stride 2 \
        --config.mode fast           # or "extensive"
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Iterable, Literal, Optional

import tyro

from configs.stage1_align import FrameToModelICPConfig
from configs.stage2_global_optimization import GlobalOptimizationConfig
from configs.stage3_gs import GSConfig
from configs.stage3_inverse_deformation import TrainInverseDeformationConfig


@dataclass
class PipelineConfig:
    """End-to-end pipeline configuration."""

    # ---- Stage 0 inputs (optional) ----
    input_video: Optional[str] = None
    """Optional path to an input video. If set, Stage 0 runs `preprocess_video.py` to create DA3 outputs."""

    frames_dir: Optional[str] = None
    """Optional path to an existing frames folder. If set, Stage 0 runs `preprocess_video.py` on this folder."""

    scene_root: Optional[str] = None
    """Optional override for where Stage 0 writes DA3 outputs (a.k.a. the scene root).

    If omitted and `input_video` is set, defaults to the input video path without extension
    (e.g. /path/to/video.mp4 -> /path/to/video).
    If omitted and `frames_dir` is set, defaults to <frames_dir>_preprocessed
    (e.g. /path/to/frames -> /path/to/frames_preprocessed).
    """

    preprocess_overwrite: bool = False
    """Force rerunning Stage 0 preprocessing even if outputs exist."""

    preprocess_max_frames: int = 100
    """Stage 0: maximum number of frames to run DA3 on."""

    preprocess_max_stride: int = 8
    """Stage 0: maximum stride between frames when subsampling."""

    preprocess_image_ext: str = "png"
    """Stage 0: frame file extension (used for extraction + folder globbing)."""

    preprocess_model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    """Stage 0: DA3 model name/path (HuggingFace repo or local)."""

    # ---- Scene root (required if Stage 0 is not used) ----
    root_path: Optional[str] = None
    """Root path to the DA3 scene data (must contain exports/npz/results.npz).

    If Stage 0 runs, this is set automatically to the produced scene root.
    """

    mode: Literal["fast", "extensive"] = "fast"
    """Pipeline mode.

    - fast: skip Stage 2, train inverse deformation for fewer epochs, run 3DGS only (shorter)
    - extensive: run Stage 2, train inverse deformation longer, run both 2DGS + 3DGS (longer)
    """

    renderer: Optional[Literal["2dgs", "3dgs", "both"]] = None
    """Which Gaussian splatting backend(s) to train.

    If set, this is honored in both modes.
    If None, `mode` decides the renderer(s):
    - fast: trains 3DGS only
    - extensive: trains both 2DGS + 3DGS
    """

    output_root: Optional[str] = None
    """Override output root (default: creates run dirs inside root_path)."""

    skip_alignment: bool = False
    """Skip Stage 1 if a checkpoint already exists (provide --alignment-run)."""

    alignment_run: Optional[str] = None
    """Existing alignment run name to reuse (skips Stage 1)."""

    skip_inverse_deform: bool = False
    """Skip Stage 3.1 if inverse deformation is already trained."""

    inverse_deform_dir: Optional[str] = None
    """Existing inverse deformation directory to reuse (skips Stage 3.1)."""

    # ---- Stage configs (defaults live in /configs) ----
    stage1: FrameToModelICPConfig = field(default_factory=FrameToModelICPConfig)
    """Stage 1: frame-to-model non-rigid ICP."""

    stage2: GlobalOptimizationConfig = field(default_factory=GlobalOptimizationConfig)
    """Stage 2: global optimization."""

    stage31: TrainInverseDeformationConfig = field(default_factory=TrainInverseDeformationConfig)
    """Stage 3.1: inverse deformation training."""

    gs: GSConfig = field(default_factory=GSConfig)
    """Stage 3.2: Gaussian splatting training (base config, renderer set per run)."""

    dry_run: bool = False
    """Print commands without executing."""


def _run(cmd: list[str], dry_run: bool = False) -> None:
    """Run a command, printing it first."""
    cmd_str = " ".join(cmd)
    print(f"\n{'=' * 80}")
    print(f"[PIPELINE] {cmd_str}")
    print(f"{'=' * 80}\n")
    if not dry_run:
        subprocess.run(cmd, check=True)


def _find_subdir(parent: str, prefix: str) -> Optional[str]:
    """Find a subdirectory matching a prefix."""
    if not os.path.isdir(parent):
        return None
    candidates = sorted(
        [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d)) and d.startswith(prefix)],
    )
    if candidates:
        return candidates[-1]
    return None


def _kebab(s: str) -> str:
    return s.replace("_", "-")


def _is_primitive(v: Any) -> bool:
    return isinstance(v, (str, int, float))


def _encode_scalar(v: Any) -> str:
    # Keep tyro-friendly representations.
    if isinstance(v, bool):
        raise TypeError("bool handled separately")
    if _is_primitive(v):
        return str(v)
    raise TypeError(f"Unsupported scalar type for CLI encoding: {type(v)}")


def _iter_config_overrides(
    *,
    prefix: str,
    cfg: Any,
    defaults: Any,
) -> Iterable[str]:
    """
    Convert dataclass differences to tyro-style CLI args.

    This intentionally only supports the config types used in this repo (nested dataclasses,
    primitives, Optional[primitive], and bool flags).
    """
    if not is_dataclass(cfg) or not is_dataclass(defaults):
        raise TypeError("cfg/defaults must be dataclass instances")

    for f in fields(cfg):
        name = f.name
        v = getattr(cfg, name)
        dv = getattr(defaults, name)

        # Recurse into nested dataclasses.
        if is_dataclass(v) and is_dataclass(dv):
            yield from _iter_config_overrides(
                prefix=f"{prefix}.{_kebab(name)}",
                cfg=v,
                defaults=dv,
            )
            continue

        if v == dv:
            continue

        flag_base = f"{prefix}.{_kebab(name)}"

        # Bool flags use tyro's --foo / --no-foo convention.
        if isinstance(v, bool):
            if v:
                yield f"--{flag_base}"
            else:
                yield f"--{prefix}.no-{_kebab(name)}"
            continue

        # Optional values.
        if v is None:
            # We could support explicit None via tyro's special syntax, but we don't need it here.
            continue

        yield f"--{flag_base}"
        yield _encode_scalar(v)


def main(config: PipelineConfig) -> None:
    python = sys.executable

    # ---- Stage 0: DA3 preprocessing (optional) ----
    # Determine whether we're in "direct scene" mode (root_path given) or "raw input" mode.
    if config.input_video and config.frames_dir:
        raise ValueError("Provide only one of --config.input-video or --config.frames-dir (not both).")

    root_path: Optional[str] = None
    if config.input_video is not None or config.frames_dir is not None:
        if config.scene_root is None or str(config.scene_root).strip() == "":
            if config.input_video is not None:
                scene_root = os.path.splitext(os.path.abspath(config.input_video))[0]
            else:
                frames_dir_abs = os.path.abspath(config.frames_dir)  # type: ignore[arg-type]
                scene_root = f"{frames_dir_abs}_preprocessed"
        else:
            scene_root = os.path.abspath(config.scene_root)

        npz_path = os.path.join(scene_root, "exports", "npz", "results.npz")
        need_preprocess = config.preprocess_overwrite or (not os.path.exists(npz_path))
        if need_preprocess:
            print("[PIPELINE] === Stage 0: DA3 Preprocessing ===")
            stage0_cmd = [
                python,
                os.path.join(os.path.dirname(__file__), "preprocess_video.py"),
                "--scene_root",
                scene_root,
                "--model_name",
                config.preprocess_model_name,
                "--image_ext",
                config.preprocess_image_ext,
                "--max_frames",
                str(config.preprocess_max_frames),
                "--max_stride",
                str(config.preprocess_max_stride),
            ]
            if config.input_video is not None:
                stage0_cmd += ["--input_video", os.path.abspath(config.input_video)]
            else:
                stage0_cmd += ["--frames_dir", os.path.abspath(config.frames_dir)]  # type: ignore[arg-type]

            if config.preprocess_overwrite:
                stage0_cmd += ["--overwrite"]

            _run(stage0_cmd, config.dry_run)
        else:
            print(f"[PIPELINE] Skipping Stage 0 (found existing NPZ): {npz_path}")

        root_path = scene_root
    elif config.root_path is not None and str(config.root_path).strip() != "":
        root_path = os.path.abspath(config.root_path)
    else:
        raise ValueError(
            "Must provide either --config.root-path (existing DA3 scene) or "
            "--config.input-video / --config.frames-dir (run Stage 0 preprocessing)."
        )

    # From this point on, the pipeline operates on an existing scene root.
    python = sys.executable
    root_path = os.path.abspath(root_path)

    # Keep pipeline convenience flags but let /configs be the source of truth for defaults.
    # These are the dynamic fields that must always be injected.
    stage1_cfg = config.stage1
    stage2_cfg = config.stage2
    stage31_cfg = config.stage31
    gs_cfg_base = config.gs

    # ---- Mode presets (applied before any stage runs) ----
    # Intentionally: mode acts as a high-level preset for the pipeline. Power users can still
    # override lower-level settings via the nested --config.stage* flags.
    if config.mode == "fast":
        if stage1_cfg.icp_early_stopping_min_delta is None:
            stage1_cfg = replace(stage1_cfg, icp_early_stopping_min_delta=5e-5)
        if stage31_cfg.n_epochs is None:
            stage31_cfg = replace(stage31_cfg, n_epochs=15)
        mode_default_renderers: tuple[str, ...] = ("3dgs",)
        gs_iters_by_renderer = {"2dgs": 10_000, "3dgs": 10_000}
        run_stage2 = False
    elif config.mode == "extensive":
        if stage1_cfg.icp_early_stopping_min_delta is None:
            stage1_cfg = replace(stage1_cfg, icp_early_stopping_min_delta=5e-6)
        if stage31_cfg.n_epochs is None:
            stage31_cfg = replace(stage31_cfg, n_epochs=30)
        mode_default_renderers = ("2dgs", "3dgs")
        gs_iters_by_renderer = {"2dgs": 15_000, "3dgs": 15_000}
        run_stage2 = True
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    # If the user explicitly requested a renderer, honor it in both modes.
    if config.renderer is None:
        effective_renderers = mode_default_renderers
    elif config.renderer == "both":
        effective_renderers = ("2dgs", "3dgs")
    else:
        effective_renderers = (config.renderer,)

    stage1_cfg = replace(stage1_cfg, root_path=root_path)
    stage1_cfg = replace(
        stage1_cfg,
        alignment=replace(
            stage1_cfg.alignment,
            # If the user set these via --config.stage1.alignment.*, tyro already updated them.
            # Otherwise, we keep the stage1 defaults.
            num_frames=stage1_cfg.alignment.num_frames,
            stride=stage1_cfg.alignment.stride,
        ),
    )

    # ---- Stage 1: Iterative Alignment ----
    if config.alignment_run and config.skip_alignment:
        run_name = config.alignment_run
        print(f"[PIPELINE] Skipping Stage 1, reusing: {run_name}")
    elif config.alignment_run:
        run_name = config.alignment_run
        print(f"[PIPELINE] Reusing existing alignment run: {run_name}")
    else:
        print("[PIPELINE] === Stage 1: Iterative Alignment ===")
        stage1_cmd = [python, "-m", "frame_to_model_icp"]
        stage1_cmd += list(
            _iter_config_overrides(
                prefix="config",
                cfg=stage1_cfg,
                defaults=FrameToModelICPConfig(),
            )
        )
        _run(stage1_cmd, config.dry_run)

        run_name = _find_subdir(root_path, "frame_to_model_icp_")
        if run_name is None:
            raise RuntimeError("Stage 1 did not produce output directory")
        print(f"[PIPELINE] Stage 1 produced: {run_name}")

    run_dir = os.path.join(root_path, run_name)

    # ---- Stage 2: Global Optimization ----
    if run_stage2:
        print("[PIPELINE] === Stage 2: Global Optimization ===")
        stage2_cfg = replace(stage2_cfg, root_path=root_path, run=run_name)
        stage2_cmd = [python, "-m", "global_optimization"]
        stage2_cmd += list(
            _iter_config_overrides(
                prefix="config",
                cfg=stage2_cfg,
                defaults=GlobalOptimizationConfig(),
            )
        )
        _run(stage2_cmd, config.dry_run)
    else:
        print("[PIPELINE] Skipping Stage 2")

    # Determine which checkpoint subdir downstream stages should consume.
    # - If Stage 2 ran, downstream should default to Stage 2 outputs.
    # - If Stage 2 is skipped, downstream should default to the inputs Stage 2 would have consumed
    #   (typically Stage 1 outputs in `after_non_rigid_icp`).
    downstream_ckpt_subdir = stage2_cfg.out_subdir if run_stage2 else stage2_cfg.checkpoint_subdir

    # ---- Stage 3.1: Inverse Deformation Training ----
    inverse_deform_dir = config.inverse_deform_dir
    if not config.skip_inverse_deform and inverse_deform_dir is None:
        print("[PIPELINE] === Stage 3.1: Inverse Deformation Training ===")
        # Respect explicit CLI overrides: only inject defaults if the user didn't set a value.
        default_stage31 = TrainInverseDeformationConfig()
        stage31_checkpoint_subdir = (
            downstream_ckpt_subdir
            if stage31_cfg.checkpoint_subdir == default_stage31.checkpoint_subdir
            else stage31_cfg.checkpoint_subdir
        )
        stage31_cfg = replace(
            stage31_cfg,
            root_path=root_path,
            run=run_name,
            checkpoint_subdir=stage31_checkpoint_subdir,
        )
        stage31_cmd = [python, "-m", "train_inverse_deformation"]
        stage31_cmd += list(
            _iter_config_overrides(
                prefix="config",
                cfg=stage31_cfg,
                defaults=TrainInverseDeformationConfig(),
            )
        )
        _run(stage31_cmd, config.dry_run)

        inverse_deform_dir = os.path.join(run_dir, "inverse_deformation")
        if not config.dry_run and not os.path.isdir(inverse_deform_dir):
            raise RuntimeError(f"Stage 3.1 did not produce output: {inverse_deform_dir}")
    elif inverse_deform_dir is None:
        inverse_deform_dir = os.path.join(run_dir, "inverse_deformation")
        if not os.path.isdir(inverse_deform_dir):
            raise RuntimeError(f"No inverse deformation directory found: {inverse_deform_dir}")

    print(f"[PIPELINE] Using inverse deformation: {inverse_deform_dir}")

    # ---- Stage 3.2: Gaussian Splatting ----
    def _run_gs(renderer: str):
        print(f"[PIPELINE] === Stage 3.2: {renderer.upper()} Training ===")
        # Respect explicit CLI overrides: only inject defaults if the user didn't set a value.
        default_gs = GSConfig()
        gs_global_opt_subdir = (
            downstream_ckpt_subdir
            if gs_cfg_base.global_opt_subdir == default_gs.global_opt_subdir
            else gs_cfg_base.global_opt_subdir
        )
        gs_cfg = replace(
            gs_cfg_base,
            root_path=root_path,
            run=run_name,
            global_opt_subdir=gs_global_opt_subdir,
            inverse_deform_dir=inverse_deform_dir,
            renderer=renderer,  # type: ignore[arg-type]
            num_iters=gs_iters_by_renderer.get(renderer, gs_cfg_base.num_iters),
        )
        # Default to using the original (preprocess) frames when available.
        # These are typically written by Stage 0 to <scene_root>/frames_subsampled and align
        # with DA3's NPZ indexing (0..N-1).
        if gs_cfg.original_images_dir == default_gs.original_images_dir:
            default_orig_dir = os.path.join(root_path, "frames_subsampled")
            if os.path.isdir(default_orig_dir):
                gs_cfg = replace(gs_cfg, original_images_dir=default_orig_dir)
        stage32_cmd = [python, "-m", "train_gs"]
        stage32_cmd += list(
            _iter_config_overrides(
                prefix="config",
                cfg=gs_cfg,
                defaults=GSConfig(),
            )
        )
        _run(stage32_cmd, config.dry_run)

    if "2dgs" in effective_renderers:
        _run_gs("2dgs")
    if "3dgs" in effective_renderers:
        _run_gs("3dgs")

    print(f"\n[PIPELINE] Done! All outputs are in: {run_dir}")


if __name__ == "__main__":
    tyro.cli(main)
