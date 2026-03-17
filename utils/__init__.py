"""
Small shared utilities used across the codebase.

This package intentionally exposes a minimal, stable surface for scripts.
Most modules can still be imported directly (e.g. `from utils.knn import ...`).
"""

from .logging import get_logger, tb_log_hparams, try_create_tensorboard_writer

from .image import build_intrinsic_matrix, colors_to_intensity

from .geometry import (
    compose_rt,
    compose_se3,
    hat,
    normal_to_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    rt_apply,
    se3_apply,
    se3_exp,
    se3_inverse,
    se3_log,
    so3_exp,
    so3_left_jacobian,
    so3_left_jacobian_inv,
    so3_log,
    vee,
)

__all__ = [
    # logging
    "get_logger",
    "try_create_tensorboard_writer",
    "tb_log_hparams",
    # image/camera
    "colors_to_intensity",
    "build_intrinsic_matrix",
    # geometry
    "hat",
    "vee",
    "so3_exp",
    "so3_log",
    "so3_left_jacobian",
    "so3_left_jacobian_inv",
    "se3_exp",
    "se3_log",
    "se3_inverse",
    "se3_apply",
    "rt_apply",
    "compose_rt",
    "compose_se3",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "normal_to_quaternion",
]
