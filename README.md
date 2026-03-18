# World Reconstruction From Inconsistent Views
Our method reconstructs 3D worlds from video diffusion models using non-rigid alignment to resolve inherent 3D inconsistencies in the generated sequences.

This is the official repository that contains source code for the paper *World Reconstruction From Inconsistent Views*.

[[arXiv](https://arxiv.org/abs/2603.16736)] [[Project Page](https://lukashoel.github.io/video_to_world/)] [[Video](https://www.youtube.com/watch?v=qXnUwhVmBzA)]

![Teaser](./assets/teaser.jpg)

If you find World Reconstruction From Inconsistent Views useful for your work please cite:
```
@misc{hoellein2026worldreconstructioninconsistentviews,
      title={World Reconstruction From Inconsistent Views}, 
      author={Lukas H{\"o}llein and Matthias Nie{\ss}ner},
      year={2026},
      eprint={2603.16736},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.16736}, 
}
```

## Prepare Environment

Clone this repository and create the conda environment:

```bash
git clone --branch main --single-branch https://github.com/lukasHoel/video_to_world
cd video_to_world

conda create -n video_to_world python=3.10
conda activate video_to_world
```

First, set up [DepthAnything-3](https://github.com/ByteDance-Seed/depth-anything-3):

```bash
mkdir -p third_party

# Clone DA3
git clone https://github.com/DepthAnything/Depth-Anything-3.git third_party/Depth-Anything-3
git -C third_party/Depth-Anything-3 checkout 2c21ea849ceec7b469a3e62ea0c0e270afc3281a

# Install DA3 + deps (minimal set for npz + gs_video)
pip install xformers torch>=2 torchvision
pip install -e third_party/Depth-Anything-3

# Apply the trajectory-export patch
git -C third_party/Depth-Anything-3 apply ../../patches/da3-export-trajectory.patch
```

Install `gsplat`:

```bash
pip install --no-build-isolation \
  "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"
```

Install `tinycudann`:

```bash
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```

Then install the remaining dependencies:

```bash
pip install open3d opencv-python scipy tyro tqdm tensorboard
pip install lpips romav2 romatch viser nerfview
```

## Quickstart

Reconstruct a 3D world from a single MP4 (generated from a video model):

```bash
python run_reconstruction.py --config.input-video /path/to/video.mp4
```

Alternatively, run the full pipeline from a folder of frames:

```bash
python run_reconstruction.py --config.frames-dir /path/to/frames
```

### Presets: fast vs extensive

`run_reconstruction.py` supports two presets via `--config.mode`:

- **fast (default)**: skips global optimization, trains backward deformation for 15 epochs, terminates ICP with `icp_early_stopping_min_delta=5e-5`, trains 3DGS for 10k iterations.
- **extensive**: runs all stages, trains backward deformation for 30 epochs, terminates ICP with `icp_early_stopping_min_delta=5e-6`, trains both 2DGS and 3DGS for 15k iterations each.

Use `--config.renderer [2dgs,3dgs,both]` to select which type of Gaussian Splatting scene is optimized.

## Running Individual Stages

### Stage 0: DA3 preprocessing (video / frames → pointcloud)

```bash
python preprocess_video.py --input_video /path/to/video.mp4
```

This estimates per-frame pointclouds using DepthAnyting-3 and saves the results to `<scene_root> = /path/to/video` (overwrite via `--scene_root /path/to/da3_scene`).

Subsampling of frames is controlled by `--max_frames` (default: 100) and `--max_stride` (default: 8).
The script extracts all frames to `<scene_root>/frames/`, then writes the selected subset (renumbered from `000000.*`) to `<scene_root>/frames_subsampled/` and runs DepthAnyting-3 on that folder.
This constrains memory of DA3 to the available budget (choose fewer frames for smaller GPUs).
Please consult the original repository for more information regarding memory.
If the scene contains much more frames, one can use [DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md) to predict per-frame pointclouds for all frames.

**Expected scene layout**:

```
<scene_root>/
  exports/
    npz/
      results.npz          # Contains: depth (N,H,W), conf (N,H,W),
                            #   extrinsics (N,3,4) w2c, intrinsics (N,3,3),
                            #   image (N,H,W,3) uint8
  gs_video/
    *.mp4                  # flythrough video of naive DA3 reconstruction
    *_transforms.json       # exported camera trajectory (used later for evaluation)
  frames/                   # extracted original frames
  frames_subsampled/         # renumbered subset used for DA3
```

The `results.npz` file is the primary input for all subsequent stages.

### Stage 1: Iterative Non-rigid Frame-to-model ICP

This non-rigidly aligns the per-frame DA3 point clouds into a single canonical frame and writes the aligned canonical point cloud plus per-frame deformation fields.

```bash
python -m frame_to_model_icp --config.root-path <scene_root>
```

#### Frame subsampling: `N`, `stride`, `offset`

Stage 1 can optionally align only a subset of frames from `exports/npz/results.npz`. The run folder name encodes the chosen subset:

- **`--config.alignment.num-frames` (`N`)**: number of frames used by Stage 1 (default: 50).
- **`--config.alignment.stride`**: take every `stride`-th frame from the underlying sequence (default: 2).
- **`--config.alignment.offset`**: starting index into the underlying sequence (default: 0).

**Output**: `<scene_root>/frame_to_model_icp_<N>_<stride>_offset<offset>/` containing:
- `after_non_rigid_icp/` -- per-frame SE(3) twists, deformation grids, merged point cloud
- `after_non_rigid_icp/config.json` -- run configuration

### Stage 2: Global Optimization

This jointly refines all per-frame deformations in a single optimization to further sharpen and flatten the canonical point cloud.

```bash
python -m global_optimization --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset>
```

**Output**: `<align_run>/after_global_optimization/` containing refined deformations and canonical point clouds.

### Stage 3.1: Inverse Deformation Training

This trains an inverse deformation network that maps canonical-space points back into each frame’s camera space to enable deformation-aware rendering losses.

```bash
python -m train_inverse_deformation \
    --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset> \
    --config.checkpoint-subdir after_global_optimization
```

**Output**: `<align_run>/inverse_deformation/` containing `inverse_local.pt` and `config.pt`.

### Stage 3.2: Gaussian Splatting Training

This optimizes a 2DGS/3DGS scene initialized from the canonical point cloud while using the inverse deformation network to warp Gaussians per frame during training.

```bash
python -m train_gs \
    --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset> \
    --config.global-opt-subdir after_global_optimization \
    --config.inverse-deform-dir <align_run>/inverse_deformation \
    --config.original-images-dir <scene_root>/frames_subsampled
```

Use `--config.renderer 3dgs` for 3D Gaussian Splatting instead (default: 2DGS).

**Output**: `<align_run>/gs_<renderer>/` containing Gaussian checkpoint, rendered images, and evaluation metrics.

### Evaluation / Novel-View Rendering

This renders novel views from a trained GS checkpoint using the evaluation camera trajectory (e.g. the DA3-exported `_transforms.json`).

```bash
python -m eval_gs \
    --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset> \
    --config.checkpoint-dir <align_run>/gs_<renderer>
```

**Output**: `<align_run>/gs_<renderer>/gs_video_eval/` containing rendered images and MP4 videos along the evaluation camera path (override with `--config.out-dir`).

## Utilities

### Export a trained 3DGS checkpoint to PLY

```bash
python -m utils.export_checkpoint_to_ply \
    --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset> \
    --config.checkpoint-dir <align_run>/gs_<renderer>
```

**Output**: a 3DGS PLY file at `--config.out-ply` (default: `<align_run>/gs_3dgs/splats_3dgs.ply`).

### View a checkpoint (interactive)

```bash
python -m utils.view_checkpoint \
    --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset> \
    --config.checkpoint-dir <align_run>/gs_<renderer>
```

This launches an interactive viewer (Viser + nerfview) for both 2DGS and 3DGS checkpoints. By default it runs on `localhost:8080` (override with `--config.port`).

## Configuration

All hyperparameters live in dataclasses under `configs/`.
They can be modified via CLI parameters for detailed configuration of the individual stages.

| File | Stage | Description |
|------|-------|-------------|
| `configs/stage1_align.py` | 1 | Iterative Non-rigid Frame-to-model ICP (`FrameToModelICPConfig`) |
| `configs/stage2_global_optimization.py` | 2 | Global optimization |
| `configs/stage3_inverse_deformation.py` | 3.1 | Inverse deformation |
| `configs/stage3_gs.py` | 3.2 | Gaussian splatting (2DGS / 3DGS) |

## Acknowledgements

Our work builds on top of amazing open-source projects. We thank the authors for making their code available.

- [Depth Anything 3 (DA3)](https://github.com/DepthAnything/Depth-Anything-3): per-frame depth/point cloud prediction (Stage 0 input).
- [RoMa](https://github.com/Parskatt/RoMa): robust dense feature matching used for correspondences during alignment.
- [gsplat](https://github.com/nerfstudio-project/gsplat): Gaussian splatting rasterizer used for 2DGS/3DGS training and rendering.
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn): hash-grid encodings used by the deformation networks.

