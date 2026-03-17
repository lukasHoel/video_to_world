#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import subprocess
import json
from typing import Optional

import torch


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def subsample_frames(
    images: list[str],
    max_frames: Optional[int] = None,
    max_stride: Optional[int] = None,
) -> tuple[list[str], int]:
    total_frames = len(images)

    # If we have fewer or equal frames than requested, just use all with stride 1.
    # This also respects the "max_stride is an upper bound" semantics.
    if max_frames is not None and total_frames <= max_frames:
        return images, 1

    # If no constraints, return everything
    if max_frames is None and max_stride is None:
        return images, 1

    # If max_frames is not specified, treat it as "use everything" and ignore max_stride.
    # (Current CLI always passes max_frames when subsampling is desired.)
    if max_frames is None:
        return images, 1

    # Normalise max_stride: None or <1 means "no effective upper bound"
    if max_stride is None or max_stride < 1:
        max_stride = total_frames

    # Ideal average stride to hit max_frames over total_frames
    ideal_stride = total_frames / float(max_frames)

    if ideal_stride <= max_stride:
        # We can (approximately) span the whole sequence.
        # Pick the smallest integer stride that still gives us at least max_frames,
        # i.e. floor(ideal_stride), but at least 1.
        stride = max(1, total_frames // max_frames)
    else:
        # Hard max_stride constraint prevents covering the full range uniformly.
        # Enforce max_stride strictly and truncate coverage.
        stride = max_stride

    indices = list(range(0, total_frames, stride))
    if len(indices) > max_frames:
        indices = indices[:max_frames]

    return [images[i] for i in indices], stride


def extract_frames(
    input_video: str,
    frames_dir: str,
    image_ext: str = "png",
) -> None:
    os.makedirs(frames_dir, exist_ok=True)

    # Clear destination if it already has files, to avoid mixing runs.
    existing = glob.glob(os.path.join(frames_dir, f"*.{image_ext}"))
    if existing:
        for p in existing:
            os.remove(p)

    out_pattern = os.path.join(frames_dir, f"%06d.{image_ext}")
    cmd = ["ffmpeg", "-y", "-i", input_video]
    cmd += ["-vsync", "0", out_pattern]
    _run(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Preprocess a video (or frame folder) with Depth Anything 3 (DA3).\n"
            "Outputs are written into <scene_root>/exports/npz/results.npz and <scene_root>/gs_video/."
        )
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_video", type=str, default=None, help="Path to input video.")
    g.add_argument("--frames_dir", type=str, default=None, help="Path to an existing frames folder.")

    p.add_argument(
        "--scene_root",
        type=str,
        default=None,
        help=(
            "Scene output directory root. If omitted:\n"
            "  - with --input_video: defaults to the input video path without extension "
            "(e.g. /path/to/video.mp4 -> /path/to/video)\n"
            "  - with --frames_dir: defaults to <frames_dir>_preprocessed "
            "(e.g. /path/to/frames -> /path/to/frames_preprocessed)"
        ),
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="DA3 model name/path (HuggingFace repo or local).",
    )
    p.add_argument("--image_ext", type=str, default="png", help="Frame file extension.")
    p.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum number of frames to run DA3 on.",
    )
    p.add_argument(
        "--max_stride",
        type=int,
        default=8,
        help="Maximum stride between frames when subsampling.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DA3 outputs under scene_root.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.scene_root is None or str(args.scene_root).strip() == "":
        if args.input_video is not None:
            scene_root = os.path.splitext(os.path.abspath(args.input_video))[0]
        else:
            # Default for an existing frames folder: sibling scene dir next to the folder.
            frames_dir_abs = os.path.abspath(args.frames_dir)
            scene_root = f"{frames_dir_abs}_preprocessed"
    else:
        scene_root = os.path.abspath(args.scene_root)

    os.makedirs(scene_root, exist_ok=True)

    source_frames_dir: str
    if args.input_video is not None:
        frames_dir = os.path.join(scene_root, "frames")
        extract_frames(
            input_video=os.path.abspath(args.input_video),
            frames_dir=frames_dir,
            image_ext=args.image_ext,
        )
        source_frames_dir = frames_dir
    else:
        frames_dir = os.path.abspath(args.frames_dir)
        if not os.path.isdir(frames_dir):
            raise ValueError(f"frames_dir '{frames_dir}' is not a directory.")
        source_frames_dir = frames_dir

    images = sorted(glob.glob(os.path.join(frames_dir, f"*.{args.image_ext}")))
    if not images:
        raise ValueError(f"No '*.{args.image_ext}' frames found in '{frames_dir}'.")

    selected_images, stride = subsample_frames(
        images,
        max_frames=args.max_frames,
        max_stride=args.max_stride,
    )
    print(f"Subsampled frames: N={len(selected_images)} (stride {stride})")

    # Materialize the selected frames into a dedicated folder so downstream code
    # can reliably "refer back" to the exact frames DA3 was run on.
    used_frames_dir = os.path.join(scene_root, "frames_subsampled")
    os.makedirs(used_frames_dir, exist_ok=True)
    # Clear destination if it already has files, to avoid mixing runs.
    existing = glob.glob(os.path.join(used_frames_dir, f"*.{args.image_ext}"))
    if existing:
        for p in existing:
            os.remove(p)

    for i, src_path in enumerate(selected_images):
        dst_path = os.path.join(used_frames_dir, f"{i:06d}.{args.image_ext}")
        shutil.copy2(src_path, dst_path)

    images_for_da3 = sorted(glob.glob(os.path.join(used_frames_dir, f"*.{args.image_ext}")))
    if len(images_for_da3) != len(selected_images):
        raise RuntimeError(
            f"Failed to materialize subsampled frames: expected {len(selected_images)} "
            f"but found {len(images_for_da3)} in '{used_frames_dir}'."
        )

    # Record which frames were used so downstream loaders can find "original" images.
    meta_path = os.path.join(scene_root, "preprocess_frames.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "frames_dir": used_frames_dir,
                "source_frames_dir": source_frames_dir,
                "image_ext": args.image_ext,
                "source": ("input_video" if args.input_video is not None else "frames_dir"),
                "max_frames": args.max_frames,
                "max_stride": args.max_stride,
                "actual_stride": stride,
                "num_frames_used": len(images_for_da3),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    # Optionally clear previous outputs (but keep frames).
    if args.overwrite:
        for rel in ["exports", "gs_video", "gs_ply", "glb", "depth_vis", "feat_vis", "colmap"]:
            p = os.path.join(scene_root, rel)
            if os.path.isdir(p):
                shutil.rmtree(p)

    # Import DA3 only after env is set up.
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DepthAnything3.from_pretrained(args.model_name)
    model = model.to(device=device)
    model.eval()

    export_format = "npz-gs_video"
    model.inference(
        image=images_for_da3,
        export_dir=scene_root,
        export_format=export_format,
        infer_gs=True,
        align_to_input_ext_scale=False,
    )

    npz_path = os.path.join(scene_root, "exports", "npz", "results.npz")
    print(f"DA3 preprocessing complete.\n- NPZ: {npz_path}\n- GS video: {os.path.join(scene_root, 'gs_video')}")


if __name__ == "__main__":
    main()

