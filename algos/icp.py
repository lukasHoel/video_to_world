import torch
from typing import Optional, Callable

from utils.image import colors_to_intensity
from utils.knn import (
    build_kdtree,
    build_torch_kdtree,
    query_knn_with_backend,
)
from utils.normals import estimate_normals
from utils.geometry import se3_exp


def colored_icp_adam(
    src_pts: torch.Tensor,
    src_colors: torch.Tensor,
    ref_pts: torch.Tensor,
    ref_colors: torch.Tensor,
    *,
    n_iter: int = 30,
    lr: float = 1e-2,
    knn_backend: str = "cpu_kdtree",
    ref_normals: torch.Tensor | None = None,
    normal_k: int = 20,
    color_k: int = 20,
    max_corr_dist: float | None = None,
    lambda_geometric: float = 0.95,
    chunk: int = 50_000,
    # Optional: hook for external progress bars/logging.
    # Called with progress events:
    #  - Stage events: it=0 and metrics contains {"stage": str, ...}
    #  - Iteration events: it>=1 and metrics contains per-iter metrics.
    progress_callback: Optional[Callable[[int, dict], None]] = None,
):
    """
    Rigid colored point cloud registration using Adam, inspired by Park et al. (ICCV 2017).

    This optimizes a joint objective in the spirit of Park et al. and the
    reference implementation in `color_icp`:

        E(T) = λ E_G(T) + (1 - λ) E_C(T)

    but applied as weights on *residuals* (as in the Gauss–Newton version):

        loss = mean( (sqrt(λ) r_G)^2 + (sqrt(1-λ) r_C)^2 )

    where:
      - E_G is the standard point-to-plane ICP objective.
      - E_C is the photometric term defined on a virtual tangent plane at each target point:
            E_C(T) = sum_{(p,q)} ( C_p(f(T q)) - C(q) )^2
        with C_p(·) approximated by a first-order Taylor expansion using precomputed
        per-point color gradients on the target cloud.

    Args:
        src_pts: (Ns,3) source points (will be transformed).
        src_colors: (Ns,3) or (Ns,) source colors.
        ref_pts: (Nr,3) target / reference points.
        ref_colors: (Nr,3) or (Nr,) target colors.
        n_iter: Adam iterations.
        lr: Adam learning rate.
        knn_backend: 'cpu_kdtree' or 'gpu_kdtree'.
        ref_normals: optional (Nr,3) target normals; if None, they are estimated.
        normal_k: neighborhood size for normal estimation (if needed).
        color_k: number of neighbors per target point for color gradient estimation.
        max_corr_dist: optional max correspondence distance (in same units as coords).
        lambda_geometric: λ in [0,1]; λ≈1 → geometry-dominated, λ≈0 → color-dominated.
        chunk: chunk size for brute-force KNN fallbacks.

    Returns:
        src_aligned: (Ns,3) transformed source points.
        R: (3,3) rotation matrix.
        t: (3,) translation vector.
    """
    device = src_pts.device
    dtype = src_pts.dtype

    src_pts = src_pts.to(device=device, dtype=dtype)
    ref_pts = ref_pts.to(device=device, dtype=dtype)
    src_colors = src_colors.to(device=device)
    ref_colors = ref_colors.to(device=device)

    # Intensity channels for photometric term
    src_I = colors_to_intensity(src_colors)
    ref_I = colors_to_intensity(ref_colors)

    Ns = src_pts.shape[0]
    Nr = ref_pts.shape[0]

    # ------------------------------------------------------------------
    # Target normals (for point-to-plane geometry and tangent planes)
    # ------------------------------------------------------------------
    if ref_normals is None:
        if progress_callback is not None:
            progress_callback(0, {"stage": "estimate_normals_start", "k": int(normal_k), "num_ref": int(Nr)})
        ref_normals, normal_tree = estimate_normals(ref_pts, k=normal_k, backend=knn_backend)
        if progress_callback is not None:
            progress_callback(0, {"stage": "estimate_normals_end"})
    else:
        normal_tree = None
        ref_normals = ref_normals.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Precompute per-target color gradients on tangent planes (d_p in Park et al.).
    # We approximate them via linear least squares over K nearest neighbors on ref.
    # ------------------------------------------------------------------
    if progress_callback is not None:
        progress_callback(0, {"stage": "color_grad_precompute_start", "k": int(color_k), "num_ref": int(Nr)})
    K_neighbors = color_k + 1  # include self, then drop it

    # Reuse pre-built KDTree from normals if possible.
    cpu_tree = normal_tree if knn_backend == "cpu_kdtree" else None
    gpu_tree = normal_tree if knn_backend == "gpu_kdtree" else None

    nn_idxs, _ = query_knn_with_backend(
        ref_pts,
        ref_pts,
        K=K_neighbors,
        backend=knn_backend,
        chunk=chunk,
        cpu_tree=cpu_tree,
        gpu_tree=gpu_tree,
    )  # (Nr, K_neighbors)

    if nn_idxs.dim() == 1:
        # K_neighbors == 1 degenerate case; we require at least 2.
        raise ValueError("color_k must be >= 1 to estimate color gradients.")

    # Drop self neighbor (assumed first)
    nn_idxs = nn_idxs[:, 1:]  # (Nr, color_k)

    # Gather neighbor positions and intensities
    ref_pts_expanded = ref_pts.unsqueeze(1)  # (Nr,1,3)
    ref_normals_expanded = ref_normals.unsqueeze(1)  # (Nr,1,3)
    neigh_pos = ref_pts[nn_idxs]  # (Nr,color_k,3)
    neigh_I = ref_I[nn_idxs]  # (Nr,color_k)

    # Tangent-plane offsets u_ij for each neighbor (Eq. 9 projection)
    delta = neigh_pos - ref_pts_expanded  # (Nr,color_k,3)
    dot = (delta * ref_normals_expanded).sum(dim=2, keepdim=True)  # (Nr,color_k,1)
    u = delta - dot * ref_normals_expanded  # (Nr,color_k,3), in tangent plane

    # Intensity differences C(p') - C(p)
    delta_I = neigh_I - ref_I.unsqueeze(1)  # (Nr,color_k)
    delta_I = delta_I.unsqueeze(2)  # (Nr,color_k,1)

    # Solve for d_p with an orthogonality constraint d_p·n_p ≈ 0, similar to
    # prepareColorGradient() in the reference implementation:
    #   (∑ u u^T + w n_p n_p^T) d_p = ∑ u ΔI
    U_t = u.transpose(1, 2)  # (Nr,3,color_k)
    A = U_t @ u  # (Nr,3,3) = ∑ u u^T
    b = U_t @ delta_I  # (Nr,3,1) = ∑ u ΔI

    # Add orthogonality term w * n_p n_p^T
    w_ortho = float(color_k)  # similar magnitude to (nn_size - 1)
    n = ref_normals  # (Nr,3)
    n_outer = n.unsqueeze(2) * n.unsqueeze(1)  # (Nr,3,3)

    eps = 1e-4
    I3 = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3)
    A_reg = A + w_ortho * n_outer + eps * I3  # (Nr,3,3)
    d_p = torch.linalg.solve(A_reg, b).squeeze(2)  # (Nr,3)
    if progress_callback is not None:
        progress_callback(0, {"stage": "color_grad_precompute_end"})

    # ------------------------------------------------------------------
    # Adam optimization on a single global SE(3) twist xi (6,)
    # ------------------------------------------------------------------
    xi = torch.nn.Parameter(torch.zeros(6, device=device, dtype=dtype))
    optimizer = torch.optim.Adam([xi], lr=lr)

    # Reuse KDTree for correspondences as well.
    if knn_backend == "cpu_kdtree" and cpu_tree is None:
        if progress_callback is not None:
            progress_callback(0, {"stage": "kdtree_build_start", "backend": knn_backend, "num_ref": int(Nr)})
        cpu_tree = build_kdtree(ref_pts)
        if progress_callback is not None:
            progress_callback(0, {"stage": "kdtree_build_end", "backend": knn_backend, "num_ref": int(Nr)})
    if knn_backend == "gpu_kdtree" and gpu_tree is None:
        if progress_callback is not None:
            progress_callback(0, {"stage": "kdtree_build_start", "backend": knn_backend, "num_ref": int(Nr)})
        gpu_tree = build_torch_kdtree(ref_pts)
        if progress_callback is not None:
            progress_callback(0, {"stage": "kdtree_build_end", "backend": knn_backend, "num_ref": int(Nr)})

    for it in range(n_iter):
        # Current SE(3) transform from twist
        Rg, tg = se3_exp(xi.unsqueeze(0))
        Rg = Rg[0]  # (3,3)
        tg = tg[0]  # (3,)

        src_transformed = (src_pts @ Rg.t()) + tg.view(1, 3)  # (Ns,3)

        # Nearest neighbors on reference for transformed source points
        idxs, d2 = query_knn_with_backend(
            src_transformed,
            ref_pts,
            K=1,
            backend=knn_backend,
            chunk=chunk,
            cpu_tree=cpu_tree if knn_backend == "cpu_kdtree" else None,
            gpu_tree=gpu_tree if knn_backend == "gpu_kdtree" else None,
        )

        # Mask by max correspondence distance if provided
        if max_corr_dist is not None:
            thresh2 = max_corr_dist * max_corr_dist
            mask = d2 < thresh2
            if mask.sum() < 3:
                break
            src_used = src_transformed[mask]
            tgt_used = ref_pts[idxs[mask]]
            normals_used = ref_normals[idxs[mask]]
            I_src_used = src_I[mask]
            I_tgt_used = ref_I[idxs[mask]]
            d_p_used = d_p[idxs[mask]]
            d2_used = d2[mask]
            perc_used = 100.0 * (mask.sum().item() / Ns)
            inliers = int(mask.sum().item())
        else:
            src_used = src_transformed
            tgt_used = ref_pts[idxs]
            normals_used = ref_normals[idxs]
            I_src_used = src_I
            I_tgt_used = ref_I[idxs]
            d_p_used = d_p[idxs]
            d2_used = d2
            perc_used = 100.0
            inliers = int(src_used.shape[0])

        # Geometric residual r_G: point-to-plane ICP (as in non_rigid_icp.py)
        rel = src_used - tgt_used
        proj = (rel * normals_used).sum(dim=1)  # (K,)
        # Photometric residual r_C: virtual tangent plane at each target point p
        # q' = projection of q (= src_used) onto tangent plane of p (= tgt_used)
        diff = src_used - tgt_used  # (K,3)
        dot_q = (diff * normals_used).sum(dim=1, keepdim=True)  # (K,1)
        u_q = diff - dot_q * normals_used  # (K,3) in tangent plane at p

        # C_p(q') ≈ C(p) + d_p · u_q
        C_hat = I_tgt_used + (d_p_used * u_q).sum(dim=1)  # (K,)
        resid_color = C_hat - I_src_used  # (K,)

        # Weight residuals like the Gauss–Newton reference:
        # loss = mean( (sqrt(λ) r_G)^2 + (sqrt(1-λ) r_C)^2 )
        lam = float(lambda_geometric)
        lam = max(0.0, min(1.0, lam))
        w_g = lam**0.5
        w_c = (1.0 - lam) ** 0.5

        loss_geo = (proj * proj).mean()
        loss_color = (resid_color * resid_color).mean()

        loss = ((w_g * proj) ** 2 + (w_c * resid_color) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rmse = torch.sqrt(d2_used.mean())
        if progress_callback is not None:
            progress_callback(
                it + 1,
                {
                    "loss": float(loss.detach().item()),
                    "loss_geo": float(loss_geo.detach().item()),
                    "loss_color": float(loss_color.detach().item()),
                    "rmse": float(rmse.detach().item()),
                    "fitness": float(perc_used),
                    "inliers": int(inliers),
                },
            )

    # Final transform
    with torch.no_grad():
        Rg, tg = se3_exp(xi.unsqueeze(0))
        Rg = Rg[0]
        tg = tg[0]
        src_aligned = (src_pts @ Rg.t()) + tg.view(1, 3)

    return src_aligned, Rg, tg
