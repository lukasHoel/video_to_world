"""
Rigid-motion and Lie group utilities shared across video_3d_consistency.

Conventions used throughout this repo:
- Points are often represented as row vectors `(N,3)` and transformed as:
    p' = p @ R.T + t
  which is equivalent to the standard column-vector form:
    p'_col = R @ p_col + t

This module provides small-angle-stable implementations for:
- SO(3) exp/log
- SO(3) left Jacobian and its inverse
- SE(3) exp and SE(3) log (via R,t -> xi)
- Common helpers: invert SE(3), apply SE(3) to points, compose SE(3)

It also includes quaternion helpers used by Gaussian splatting.
Quaternions are stored as (w, x, y, z) with w being the scalar part.
"""

from __future__ import annotations

from typing import Tuple

import torch


# -------------------------------------------------------------------------
# Quaternion utilities
# -------------------------------------------------------------------------
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion to rotation matrix."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)

    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))

    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to unit quaternion (Shepperd's method)."""
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 0: trace is largest
    s0 = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2
    q0 = torch.stack(
        [
            0.25 * s0,
            (R[:, 2, 1] - R[:, 1, 2]) / s0,
            (R[:, 0, 2] - R[:, 2, 0]) / s0,
            (R[:, 1, 0] - R[:, 0, 1]) / s0,
        ],
        dim=-1,
    )

    # Case 1: R[0,0] is largest diagonal
    s1 = torch.sqrt((1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-10)) * 2
    q1 = torch.stack(
        [
            (R[:, 2, 1] - R[:, 1, 2]) / s1,
            0.25 * s1,
            (R[:, 0, 1] + R[:, 1, 0]) / s1,
            (R[:, 0, 2] + R[:, 2, 0]) / s1,
        ],
        dim=-1,
    )

    # Case 2: R[1,1] is largest diagonal
    s2 = torch.sqrt((1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).clamp(min=1e-10)) * 2
    q2 = torch.stack(
        [
            (R[:, 0, 2] - R[:, 2, 0]) / s2,
            (R[:, 0, 1] + R[:, 1, 0]) / s2,
            0.25 * s2,
            (R[:, 1, 2] + R[:, 2, 1]) / s2,
        ],
        dim=-1,
    )

    # Case 3: R[2,2] is largest diagonal
    s3 = torch.sqrt((1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).clamp(min=1e-10)) * 2
    q3 = torch.stack(
        [
            (R[:, 1, 0] - R[:, 0, 1]) / s3,
            (R[:, 0, 2] + R[:, 2, 0]) / s3,
            (R[:, 1, 2] + R[:, 2, 1]) / s3,
            0.25 * s3,
        ],
        dim=-1,
    )

    diag = torch.stack([trace, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], dim=-1)
    idx = diag.argmax(dim=-1)

    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    q = torch.where(idx.unsqueeze(-1) == 0, q0, q)
    q = torch.where(idx.unsqueeze(-1) == 1, q1, q)
    q = torch.where(idx.unsqueeze(-1) == 2, q2, q)
    q = torch.where(idx.unsqueeze(-1) == 3, q3, q)

    # Ensure w > 0 convention
    q = q * torch.sign(q[:, :1]).clamp(min=1e-8)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return q.reshape(batch_shape + (4,))


def normal_to_quaternion(normals: torch.Tensor) -> torch.Tensor:
    """Compute quaternion that rotates z-axis [0,0,1] to the given normal."""
    batch_shape = normals.shape[:-1]
    n = normals.reshape(-1, 3)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    dot = n[:, 2]
    cross_x = -n[:, 1]
    cross_y = n[:, 0]
    cross_z = torch.zeros_like(dot)

    w = 1.0 + dot
    q = torch.stack([w, cross_x, cross_y, cross_z], dim=-1)

    anti = dot < -0.999
    if anti.any():
        q_anti = torch.tensor([0.0, 1.0, 0.0, 0.0], device=n.device, dtype=n.dtype)
        q_anti = q_anti.unsqueeze(0).expand(n.shape[0], -1)
        q = torch.where(anti.unsqueeze(-1), q_anti, q)

    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return q.reshape(batch_shape + (4,))


# -------------------------------------------------------------------------
# SO(3) / SE(3) utilities
# -------------------------------------------------------------------------
def hat(omega: torch.Tensor) -> torch.Tensor:
    """omega: (...,3) -> (...,3,3) skew-symmetric"""
    ox, oy, oz = omega[..., 0], omega[..., 1], omega[..., 2]
    out = torch.zeros(omega.shape[:-1] + (3, 3), device=omega.device, dtype=omega.dtype)
    out[..., 0, 1] = -oz
    out[..., 0, 2] = oy
    out[..., 1, 0] = oz
    out[..., 1, 2] = -ox
    out[..., 2, 0] = -oy
    out[..., 2, 1] = ox
    return out


def vee(omega_hat: torch.Tensor) -> torch.Tensor:
    """Inverse of hat: extracts 3D vector from skew-symmetric matrix."""
    omega = torch.zeros(omega_hat.shape[:-2] + (3,), device=omega_hat.device, dtype=omega_hat.dtype)
    omega[..., 0] = omega_hat[..., 2, 1]
    omega[..., 1] = omega_hat[..., 0, 2]
    omega[..., 2] = omega_hat[..., 1, 0]
    return omega


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula with small-angle handling."""
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    small = theta < 1e-8
    theta_safe = torch.where(small, torch.ones_like(theta), theta)

    axis = omega / theta_safe
    K = hat(axis)

    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(K.shape)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    R = eye + sin_t * K + (1 - cos_t) * (K @ K)

    if small.any():
        omega_hat = hat(omega)
        R_small = eye + omega_hat + 0.5 * (omega_hat @ omega_hat)
        R = torch.where(small[..., None], R_small, R)
    return R


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Logarithm map of SO(3) (inverse of `so3_exp`)."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    trace = torch.clamp(trace, -1.0, 3.0)

    theta = torch.acos((trace - 1.0) / 2.0)
    small = theta < 1e-8

    R_minus_RT = R - R.transpose(-2, -1)
    omega = vee(R_minus_RT)

    sin_theta = torch.sin(theta)
    sin_theta_safe = torch.where(small, torch.ones_like(sin_theta), sin_theta)
    coeff = theta.unsqueeze(-1) / (2.0 * sin_theta_safe.unsqueeze(-1) + 1e-12)
    omega = coeff * omega

    if small.any():
        eye = torch.eye(3, device=R.device, dtype=R.dtype).expand(R.shape)
        omega_small = vee(R - eye)
        omega = torch.where(small.unsqueeze(-1), omega_small, omega)
    return omega


def so3_left_jacobian(omega: torch.Tensor) -> torch.Tensor:
    """Left Jacobian J(omega) for SO(3)."""
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    small = theta < 1e-8
    theta_safe = torch.where(small, torch.ones_like(theta), theta)

    omega_hat = hat(omega)
    omega_hat_sq = omega_hat @ omega_hat
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega_hat.shape)

    a = (1 - torch.cos(theta_safe)) / (theta_safe**2)
    b = (theta_safe - torch.sin(theta_safe)) / (theta_safe**3)
    J = eye + a[..., None] * omega_hat + b[..., None] * omega_hat_sq

    if small.any():
        J_small = eye + 0.5 * omega_hat + (1.0 / 6.0) * omega_hat_sq
        J = torch.where(small[..., None], J_small, J)
    return J


def so3_left_jacobian_inv(omega: torch.Tensor) -> torch.Tensor:
    """Inverse left Jacobian J^{-1}(omega) for SO(3)."""
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    small = theta < 1e-8
    theta_safe = torch.where(small, torch.ones_like(theta), theta)

    omega_hat = hat(omega)
    omega_hat_sq = omega_hat @ omega_hat
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega_hat.shape)

    theta_half = theta_safe / 2.0
    cot_theta_half = torch.cos(theta_half) / (torch.sin(theta_half) + 1e-12)
    coeff = (1.0 - theta_half * cot_theta_half) / (theta_safe**2 + 1e-12)

    J_inv = eye - 0.5 * omega_hat + coeff[..., None] * omega_hat_sq

    if small.any():
        J_inv_small = eye - 0.5 * omega_hat + (1.0 / 12.0) * omega_hat_sq
        J_inv = torch.where(small[..., None], J_inv_small, J_inv)
    return J_inv


def se3_exp(xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """xi: (...,6) where xi[..., :3]=omega, xi[..., 3:]=v -> returns (R,t)."""
    omega = xi[..., :3]
    v = xi[..., 3:]
    R = so3_exp(omega)
    J = so3_left_jacobian(omega)
    t = torch.matmul(J, v.unsqueeze(-1)).squeeze(-1)
    return R, t


def se3_log(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """SE(3) logarithm map: inverse of `se3_exp`."""
    omega = so3_log(R)
    J_inv = so3_left_jacobian_inv(omega)
    v = torch.matmul(J_inv, t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([omega, v], dim=-1)


def se3_inverse(xi: torch.Tensor) -> torch.Tensor:
    """Inverse of an SE(3) transformation represented as xi (...,6)."""
    R, t = se3_exp(xi)
    R_inv = R.transpose(-2, -1)
    t_inv = (-(R_inv @ t.unsqueeze(-1))).squeeze(-1)
    return se3_log(R_inv, t_inv)


def se3_apply(xi: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply SE(3) transformation(s) to point(s)."""
    R, t = se3_exp(xi)
    return (R @ pts.unsqueeze(-1)).squeeze(-1) + t


def rt_apply(R: torch.Tensor, t: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply a rigid transform (R, t) to row-vector points."""
    return (R @ pts.unsqueeze(-1)).squeeze(-1) + t


def compose_rt(
    R_left: torch.Tensor,
    t_left: torch.Tensor,
    R_right: torch.Tensor,
    t_right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose two rigid transforms in (R,t) form (column-vector convention)."""
    R = R_left @ R_right
    t = (R_left @ t_right.unsqueeze(-1)).squeeze(-1) + t_left
    return R, t


def compose_se3(xi_left: torch.Tensor, xi_right: torch.Tensor) -> torch.Tensor:
    """Compose two SE(3) transforms given as twists."""
    R_left, t_left = se3_exp(xi_left)
    R_right, t_right = se3_exp(xi_right)
    R, t = compose_rt(R_left, t_left, R_right, t_right)
    return se3_log(R, t)
