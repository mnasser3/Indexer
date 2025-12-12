"""
Crystal Lattice Indexer (CPU Implementation)

This module implements a crystallographic indexing algorithm for determining crystal
orientation from diffraction peaks. It uses quaternion-based rotation representations,
lattice basis reduction, symmetry operations, and differential evolution optimization
to find the optimal crystal orientation matrix.

Key Features:
- Quaternion-based orientation handling with batch processing support
- Korkine-Zolotarev lattice reduction for improved indexing stability
- Symmetry-aware clustering and canonicalization
- Trimmed mean loss functions robust to outliers
- Target-space neighbor (TSN) polishing for small datasets
- Multi-threaded differential evolution optimization
"""

import os
import numpy as np
from functools import partial
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import gemmi
from numba import njit
import math


# ==============================================================================
# Quaternion Operations
# ==============================================================================

def quat_normalize(q):
    """Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion as array of shape (..., 4) in (w, x, y, z) format
        
    Returns:
        Normalized quaternion with same shape as input
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n

def quat_to_R(q):
    """Convert a single quaternion to a 3x3 rotation matrix.
    
    Args:
        q: Quaternion (w, x, y, z) as array of length 4
        
    Returns:
        3x3 rotation matrix as numpy array
    """
    q = quat_normalize(q)
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),     1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1*q1 + q2*q2)]
    ], dtype=np.float64)

@njit(fastmath=True)
def _quat_to_R_batch_core(quats):
    """JIT-compiled core function for batch quaternion to rotation matrix conversion.
    
    Args:
        quats: Array of shape (B, 4) containing B quaternions
        
    Returns:
        Array of shape (B, 3, 3) containing B rotation matrices
    """
    B = quats.shape[0]
    R = np.empty((B, 3, 3), np.float64)

    for i in range(B):
        w = quats[i, 0]
        x = quats[i, 1]
        y = quats[i, 2]
        z = quats[i, 3]

        n = math.sqrt(w*w + x*x + y*y + z*z)
        if n > 0.0:
            w /= n
            x /= n
            y /= n
            z /= n

        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z

        R[i, 0, 0] = 1.0 - 2.0 * (yy + zz)
        R[i, 0, 1] = 2.0 * (xy - wz)
        R[i, 0, 2] = 2.0 * (xz + wy)

        R[i, 1, 0] = 2.0 * (xy + wz)
        R[i, 1, 1] = 1.0 - 2.0 * (xx + zz)
        R[i, 1, 2] = 2.0 * (yz - wx)

        R[i, 2, 0] = 2.0 * (xz - wy)
        R[i, 2, 1] = 2.0 * (yz + wx)
        R[i, 2, 2] = 1.0 - 2.0 * (xx + yy)

    return R


def quat_to_R_batch(quats):
    """Convert batch of quaternions to rotation matrices.
    
    Args:
        quats: Array of shape (B, 4) or (4,) containing quaternions
        
    Returns:
        Array of shape (B, 3, 3) containing rotation matrices
        
    Raises:
        ValueError: If input shape is not (B, 4) or (4,)
    """
    q = np.asarray(quats, dtype=np.float64)
    if q.ndim == 1:
        if q.shape[0] != 4:
            raise ValueError(f"Expected quaternion of length 4, got shape {q.shape}")
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat_to_R_batch expects shape (B,4) or (4,), got {q.shape}")
    return _quat_to_R_batch_core(q)


# ==============================================================================
# Lattice Reduction and Basis Operations
# ==============================================================================

def _shortest_vector_enum_vec(B, Hmax=6):
    """Find shortest lattice vector using enumeration over a grid.
    
    Args:
        B: 3x3 lattice basis matrix
        Hmax: Maximum integer coordinate to search
        
    Returns:
        Tuple of (H, V, norm_squared) where H is the integer coefficients,
        V is the vector, and norm_squared is its squared length
    """
    r = np.arange(-Hmax, Hmax + 1)
    H = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    H = H[np.any(H != 0, axis=1)]
    V = B @ H.T
    n2 = np.einsum("ij,ij->j", V, V)
    k = int(np.argmin(n2))
    return H[k], V[:, k], float(n2[k])

def _gauss_reduce_2d(B2):
    """Perform 2D Gaussian lattice reduction.
    
    Args:
        B2: 2x2 or (3, 2) lattice basis matrix
        
    Returns:
        Tuple of (G, U2) where G is the reduced basis and U2 is the
        unimodular transformation matrix
    """
    G = B2.copy().astype(np.float64)
    U2 = np.eye(2, dtype=int)
    while True:
        denom = max(np.dot(G[:, 0], G[:, 0]), 1e-18)
        mu = np.dot(G[:, 0], G[:, 1]) / denom
        k = int(np.rint(mu))
        if k != 0:
            G[:, 1] -= k * G[:, 0]
            U2[:, 1] -= k * U2[:, 0]
        if np.dot(G[:, 1], G[:, 1]) < np.dot(G[:, 0], G[:, 0]) - 1e-18:
            G[:, [0, 1]] = G[:, [1, 0]]
            U2[:, [0, 1]] = U2[:, [1, 0]]
        else:
            break
    return G, U2

def kz_reduce_integer_3d(B, Hmax=6):
    """Perform Korkine-Zolotarev reduction on a 3D integer lattice basis.
    
    This reduction algorithm finds a more orthogonal basis which improves
    the stability and accuracy of lattice rounding operations.
    
    Args:
        B: 3x3 lattice basis matrix
        Hmax: Maximum search radius for shortest vector
        
    Returns:
        Tuple of (B_kz, U) where B_kz is the reduced basis and U is the
        integer unimodular transformation matrix such that B_kz = B @ U
    """
    B = np.asarray(B, np.float64)
    h1, b1, _ = _shortest_vector_enum_vec(B, Hmax=Hmax)
    U = np.eye(3, dtype=int)
    U[:, 0] = h1
    if np.linalg.matrix_rank(U) < 3:
        U[:, 1] = np.array([0, 1, 0]); U[:, 2] = np.array([0, 0, 1])
        if np.linalg.matrix_rank(U) < 3:
            U[:, 1] = np.array([1, 0, 0]); U[:, 2] = np.array([0, 1, 0])
    for j in (1, 2):
        b_j = B @ U[:, j]
        denom = max(np.dot(b1, b1), 1e-18)
        q = int(np.rint(np.dot(b_j, b1) / denom))
        U[:, j] -= q * U[:, 0]
    B1 = B @ U
    G2, U2 = _gauss_reduce_2d(B1[:, 1:3])
    U[:, 1:3] = U[:, 1:3] @ U2
    B_kz = B @ U
    return B_kz, U

@njit(fastmath=True)
def _babai_nearest_batched_core(R_B, M):
    """JIT-compiled Babai nearest plane algorithm for batch lattice rounding.
    
    Efficiently rounds multiple vectors to nearest lattice points using
    the QR decomposition of the lattice basis.
    
    Args:
        R_B: Upper triangular R matrix from QR decomposition of basis
        M: Array of shape (K, 3, N) containing vectors to round
        
    Returns:
        Array of shape (K, 3, N) containing integer lattice coordinates
    """
    r00 = R_B[0, 0]
    r11 = R_B[1, 1]
    r22 = R_B[2, 2]
    r01 = R_B[0, 1]
    r02 = R_B[0, 2]
    r12 = R_B[1, 2]

    K = M.shape[0]
    N = M.shape[2]
    H = np.empty((K, 3, N), np.int64)

    for k in range(K):
        for n in range(N):
            y2 = M[k, 2, n]
            h2 = int(np.rint(y2 / r22))

            y1 = M[k, 1, n] - r12 * h2
            y0 = M[k, 0, n] - r02 * h2

            h1 = int(np.rint(y1 / r11))
            y0 = y0 - r01 * h1

            h0 = int(np.rint(y0 / r00))

            H[k, 0, n] = h0
            H[k, 1, n] = h1
            H[k, 2, n] = h2

    return H


def babai_nearest_batched(R_B, M):
    """Batch lattice rounding using Babai nearest plane algorithm.
    
    Args:
        R_B: Upper triangular matrix from QR decomposition
        M: Array of shape (K, 3, N) to round
        
    Returns:
        Integer lattice coordinates of shape (K, 3, N)
        
    Raises:
        ValueError: If M doesn't have shape (K, 3, N)
    """
    R_B = np.asarray(R_B, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 3 or M.shape[1] != 3:
        raise ValueError(f"babai_nearest_batched expects M with shape (K,3,N), got {M.shape}")
    return _babai_nearest_batched_core(R_B, M)


# ==============================================================================
# Crystal Symmetry Operations
# ==============================================================================

def get_rots_mats_symm(unit_cell):
    """Determine lattice symmetry operations for a unit cell.
    
    Uses GEMMI library to find symmetry operations and converts them to
    reciprocal space rotation matrices.
    
    Args:
        unit_cell: Tuple of (a, b, c, alpha, beta, gamma) unit cell parameters
        
    Returns:
        Array of shape (S, 3, 3) containing S symmetry rotation matrices
        in reciprocal space
    """
    uc = gemmi.UnitCell(*unit_cell)
    ops = gemmi.find_lattice_symmetry(uc, 'P', 3.0)
    B = np.array(uc.frac.mat).T
    B_inv = np.array(uc.orth.mat).T
    rot_list = []
    for op in ops:
        R_frac = np.array(op.rot, dtype=np.float64) / op.DEN
        R_recip = B @ R_frac.T @ B_inv
        rot_list.append(R_recip.astype(np.float64))
    if not rot_list:
        rot_list = [np.eye(3, dtype=np.float64)]
    return np.stack(rot_list, axis=0)


# ==============================================================================
# Loss Functions
# ==============================================================================

def loss_trimmed_batched(
    quats,
    Q,
    B_red,
    Q_B_T,
    R_B,
    S_ops_stack,
    PB_ops_stack,
    kappa=0.40,
):
    """Compute trimmed mean loss for batch of quaternions with symmetry.
    
    Evaluates multiple orientations simultaneously, considering all symmetry
    operations and using a trimmed mean to be robust to outliers.
    
    Args:
        quats: Array of shape (B, 4) containing quaternions to evaluate
        Q: Array of shape (3, N) containing N observed peak positions
        B_red: Reduced lattice basis matrix (3, 3)
        Q_B_T: Transpose of Q matrix from QR decomposition
        R_B: R matrix from QR decomposition
        S_ops_stack: Symmetry operations in reduced basis (S, 3, 3)
        PB_ops_stack: Symmetry operations composed with basis (S, 3, 3)
        kappa: Trimming fraction (0.4 = keep best 40% of residuals)
        
    Returns:
        Array of shape (B,) containing loss values for each quaternion
    """
    quats = np.ascontiguousarray(quats, dtype=np.float64)
    quats /= (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12)

    Rk = quat_to_R_batch(quats)
    A  = np.einsum("bji,jn->bin", Rk, Q, optimize=True)

    Bsz, _, N = A.shape
    S = S_ops_stack.shape[0]

    A_sym = np.einsum("sij,bjn->sbin", S_ops_stack, A, optimize=True)
    A_resh  = A_sym.reshape(S * Bsz, 3, N)
    H_sym   = babai_nearest_batched(R_B, A_resh).reshape(S, Bsz, 3, N)
    Y_sym = np.einsum("sij,sbjn->sbin", PB_ops_stack, H_sym, optimize=True)

    diff = A[None, :, :, :] - Y_sym
    r2   = np.einsum("sbin,sbin->sbn", diff, diff, optimize=True)
    r2_min = np.min(r2, axis=0)

    h = int(np.ceil(kappa * N))
    part = np.partition(r2_min, h - 1, axis=1)[:, :h]
    t2_vec = np.quantile(part, 0.90, axis=1)
    clipped = np.minimum(part, t2_vec[:, None])
    out = clipped.mean(axis=1)
    return out

def loss_trimmed_scalar(q, Q, B_red, Q_B_T, R_B, S_ops_stack, PB_ops_stack, kappa=0.40):
    """Scalar version of trimmed loss for single quaternion (used in optimization).
    
    Args:
        q: Single quaternion of shape (4,)
        Q, B_red, Q_B_T, R_B, S_ops_stack, PB_ops_stack, kappa: Same as loss_trimmed_batched
        
    Returns:
        Scalar loss value
    """
    q = np.asarray(q, dtype=np.float64)
    return float(loss_trimmed_batched(q[None, :], Q, B_red, Q_B_T, R_B, S_ops_stack, PB_ops_stack, kappa)[0])

def loss_mse_batched(quats, Q, B_red, Q_B_T, R_B, S_ops_stack, PB_ops_stack):
    """Compute mean squared error loss for batch of quaternions with symmetry.
    
    Unlike trimmed loss, this uses all residuals without trimming. Faster but
    less robust to outliers.
    
    Args:
        quats: Array of shape (B, 4) containing quaternions
        Q: Observed peaks of shape (3, N)
        B_red, Q_B_T, R_B, S_ops_stack, PB_ops_stack: Same as loss_trimmed_batched
        
    Returns:
        Array of shape (B,) containing MSE loss for each quaternion
    """
    quats = np.ascontiguousarray(quats, dtype=np.float64)
    quats /= (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12)
    Q_sq_mean = float(np.mean(np.sum(Q * Q, axis=0)))

    Rk = quat_to_R_batch(quats)
    A  = np.einsum("bji,jn->bin", Rk, Q, optimize=True)

    best = np.full(len(quats), np.inf, dtype=np.float64)
    for s in range(S_ops_stack.shape[0]):
        A_sym = np.einsum("ij,bjn->bin", S_ops_stack[s], A, optimize=True)
        Hp_sym = babai_nearest_batched(R_B, A_sym)
        Y_sym = np.einsum("ij,bjn->bin", PB_ops_stack[s], Hp_sym, optimize=True)
        YY_mean = np.mean(np.sum(Y_sym * Y_sym, axis=1), axis=1)
        AY_mean = np.mean(np.sum(A      * Y_sym, axis=1), axis=1)
        val = Q_sq_mean + YY_mean - 2.0 * AY_mean
        best = np.minimum(best, val)
    return best

def R_to_quat(R):
    """Convert a 3x3 rotation matrix to a quaternion.
    
    Uses numerically stable conversion based on trace or largest diagonal element.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion (w, x, y, z) as numpy array of length 4
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (R[2,1] - R[1,2]) / S
        q2 = (R[0,2] - R[2,0]) / S
        q3 = (R[1,0] - R[0,1]) / S
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            q0 = (R[2,1] - R[1,2]) / S
            q1 = 0.25 * S
            q2 = (R[0,1] + R[1,0]) / S
            q3 = (R[0,2] + R[2,0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            q0 = (R[0,2] - R[2,0]) / S
            q1 = (R[0,1] + R[1,0]) / S
            q2 = 0.25 * S
            q3 = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            q0 = (R[1,0] - R[0,1]) / S
            q1 = (R[0,2] + R[2,0]) / S
            q2 = (R[1,2] + R[2,1]) / S
            q3 = 0.25 * S
    return quat_normalize(np.array([q0, q1, q2, q3], dtype=np.float64))


# ==============================================================================
# Polishing and Refinement
# ==============================================================================

def polish_R_smallN_tsn(
    R_init, Q, consts, *,
    kappa=0.40, delta=1, winsor_q=None,
    fit_alpha=True, alpha_clip=(0.97, 1.03), steps=2,
    indices=None
):
    """Polish rotation matrix using target-space neighbor search (TSN).
    
    Refines the orientation by searching nearby integer lattice points (neighbors
    in target space) to improve indexing accuracy. Particularly effective for
    small numbers of peaks.
    
    Args:
        R_init: Initial 3x3 rotation matrix
        Q: Observed peaks of shape (3, N)
        consts: Dictionary of precomputed constants from prepare_constants()
        kappa: Trimming fraction for robust fitting
        delta: Search radius in target space (±delta in each direction)
        winsor_q: Quantile for Winsorization (None to disable)
        fit_alpha: Whether to fit a scale factor
        alpha_clip: (min, max) bounds for scale factor
        steps: Number of refinement iterations
        indices: Subset of peak indices to use (None = all peaks)
        
    Returns:
        Tuple of (R, H, s_best, alpha) where:
        - R: Refined rotation matrix
        - H: Integer Miller indices for all peaks
        - s_best: Best symmetry operation index
        - alpha: Fitted scale factor
    """
    Q = np.asarray(Q, np.float64)
    N = Q.shape[1]
    I = np.arange(N, dtype=int) if indices is None else np.asarray(indices, dtype=int)
    h = int(np.ceil(kappa * len(I)))

    R_B     = consts["R_B"]
    S_ops   = consts["S_ops_stack"]
    PB_ops  = consts["PB_ops_stack"]
    S       = S_ops.shape[0]

    R = np.asarray(R_init, np.float64)
    alpha = 1.0
    H_out = None
    s_best = 0

    for _ in range(max(1, int(steps))):
        # Rotate observed peaks to reciprocal lattice frame
        A = R @ Q
        A_I = A[:, I]
        # Apply symmetry operations
        A_sym_I = np.einsum("sij,jn->sin", S_ops, A_I, optimize=True)

        # Round to nearest lattice points
        H_babai_I = babai_nearest_batched(R_B, A_sym_I)
        # Transform back to reciprocal space
        Y0_I = np.einsum("sij,sjn->sin", PB_ops, H_babai_I, optimize=True)
        # Compute residual errors
        E_I  = A_sym_I - Y0_I
        E2_I = np.einsum("sin,sin->sn", E_I, E_I, optimize=True)

        # Target-space neighbor search: try nearby integer lattice points
        if delta > 0:
            D_tsn  = consts.get("D_tsn", None)
            PB_D   = consts.get("PB_D", None)
            dY2    = consts.get("dY2", None)
            # Generate neighbor offsets if not precomputed or if delta changed
            if D_tsn is None or PB_D is None or dY2 is None or consts.get("delta_tsn", None) != delta:
                offs = np.arange(-delta, delta + 1, dtype=int)
                # Create all integer offset combinations in [-delta, delta]^3
                D_tsn = np.stack(
                    np.meshgrid(offs, offs, offs, indexing="ij"),
                    axis=-1
                ).reshape(-1, 3).T
                PB_D = np.einsum("sij,jk->sik", PB_ops, D_tsn, optimize=True)
                dY2  = np.einsum("sik,sik->sk", PB_D, PB_D, optimize=True)

            # Find best neighbor for each peak
            dot  = np.einsum("sin,sik->skn", E_I, PB_D, optimize=True)
            r2_k = E2_I[:, None, :] + dY2[:, :, None] - 2.0 * dot
            k_star_I = np.argmin(r2_k, axis=1)  # Best neighbor index
            r2_s_I   = np.min(r2_k, axis=1)     # Best residual per symmetry
        else:
            k_star_I = None
            r2_s_I = E2_I        
        
        S_, M = r2_s_I.shape
        assert S_ == S

        # Compute trimmed mean: keep only best kappa fraction of residuals
        h = int(np.ceil(kappa * len(I)))

        # Partially sort to get h smallest residuals efficiently
        part = np.partition(r2_s_I, h - 1, axis=1)[:, :h]

        # Optionally apply Winsorization to reduce influence of outliers
        if winsor_q is not None and 0.0 < winsor_q < 1.0 and h >= 3:
            t2_vec = np.quantile(part, winsor_q, axis=1)
            clipped_all = np.minimum(part, t2_vec[:, None])
        else:
            clipped_all = part

        # Select best symmetry operation
        loss_s = clipped_all.mean(axis=1)

        s_best = int(np.argmin(loss_s))

        r2_best = r2_s_I[s_best]
        idx_h_sub = np.argpartition(r2_best, h - 1)[:h]
        sel = r2_best[idx_h_sub]
        if winsor_q is not None and 0.0 < winsor_q < 1.0 and h >= 3:
            t2 = float(np.quantile(sel, winsor_q))
            sel = np.minimum(sel, t2)
        best_loss = float(np.mean(sel))

        idx_h = I[idx_h_sub]

        # Index all peaks using best symmetry operation
        A_sym_all = S_ops[s_best] @ A
        H_all = babai_nearest_batched(R_B, A_sym_all[None, ...])[0]

        # Apply neighbor offsets if using TSN
        if delta > 0:
            offs = np.arange(-delta, delta + 1, dtype=int)
            D = np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), axis=-1).reshape(-1, 3).T
            ks = k_star_I[s_best]
            H_all[:, I] = (H_all[:, I] + D[:, ks]).astype(int)

        Y_all = PB_ops[s_best] @ H_all

        # Optionally fit a uniform scale factor
        if fit_alpha:
            num = float(np.sum(Q[:, idx_h] * (R @ Y_all[:, idx_h])))
            den = float(np.sum(Y_all[:, idx_h] * Y_all[:, idx_h])) + 1e-18
            alpha = num / den
            lo, hi = alpha_clip
            alpha = float(np.minimum(hi, np.maximum(lo, alpha)))
        else:
            alpha = 1.0

        # Procrustes alignment: find optimal rotation R given correspondences
        Sxy = (Q[:, idx_h] @ (alpha * Y_all[:, idx_h]).T)
        U, _, Vt = np.linalg.svd(Sxy, full_matrices=False)
        # Ensure proper rotation (det = +1)
        R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt

        H_out = H_all

    return R, H_out, s_best, alpha

def hardest_peak_indices(R, Q, consts, K):
    """Identify the K peaks with largest indexing residuals.
    
    Used to focus refinement on the most problematic peaks in large datasets.
    
    Args:
        R: Current rotation matrix
        Q: All observed peaks of shape (3, N)
        consts: Dictionary of precomputed constants
        K: Number of hardest peaks to return
        
    Returns:
        Array of K peak indices with largest residuals
    """
    S_ops, PB_ops, R_B = consts["S_ops_stack"], consts["PB_ops_stack"], consts["R_B"]
    A = R @ Q
    A_sym = np.einsum("sij,jn->sin", S_ops, A, optimize=True)
    H_b = babai_nearest_batched(R_B, A_sym)
    Y0  = np.einsum("sij,sjn->sin", PB_ops, H_b, optimize=True)
    diff = A[None, :, :] - Y0
    r2   = np.einsum("sin,sin->sn", diff, diff, optimize=True)
    r2_min = np.min(r2, axis=0)
    K = int(min(K, r2_min.shape[0]))
    return np.argpartition(r2_min, -K)[-K:]

def loss_small_n_batched(
    quats, Q, B_red, Q_B_T, R_B, S_ops_stack, PB_ops_stack,
    D, PB_D, kappa=0.40, delta=1
):
    """Loss function with target-space neighbor search for small datasets.
    
    Evaluates loss while considering nearby lattice points in target space,
    which can improve accuracy for datasets with few peaks.
    
    Args:
        quats: Array of shape (B, 4) containing quaternions
        Q: Observed peaks of shape (3, N)
        B_red, Q_B_T, R_B, S_ops_stack, PB_ops_stack: Precomputed constants
        D: Offset vectors for target-space search of shape (3, K)
        PB_D: Precomputed PB @ D of shape (S, 3, K)
        kappa: Trimming fraction
        delta: Search radius (must match D and PB_D)
        
    Returns:
        Array of shape (B,) containing loss values
    """
    quats = np.ascontiguousarray(quats, dtype=np.float64)
    quats /= (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12)
    Rk = quat_to_R_batch(quats)
    A  = np.einsum("bji,jn->bin", Rk, Q, optimize=True)
    Bsz, _, N = A.shape
    S = S_ops_stack.shape[0]
    A_sym = np.einsum("sij,bjn->sbin", S_ops_stack, A, optimize=True)
    A_resh  = A_sym.reshape(S * Bsz, 3, N)
    H_babai = babai_nearest_batched(R_B, A_resh).reshape(S, Bsz, 3, N)
    Y0 = np.einsum("sij,sbjn->sbin", PB_ops_stack, H_babai, optimize=True)
    E  = A[None, :, :, :] - Y0
    E2 = np.einsum("sbin,sbin->sbn", E, E, optimize=True)
    dY2  = np.einsum("sik,sik->sk", PB_D, PB_D, optimize=True)
    dot = np.einsum("sbin,sik->sbkn", E, PB_D, optimize=True)
    r2_k = E2[:, :, None, :] + dY2[:, None, :, None] - 2.0 * dot
    r2_best_per_sym = np.min(r2_k, axis=2)
    r2_min = np.min(r2_best_per_sym, axis=0)
    Bsz, N = r2_min.shape
    h = int(np.ceil(kappa * N))
    part = np.partition(r2_min, h - 1, axis=1)[:, :h]  
    out = part.mean(axis=1).astype(np.float64)   
    return out


# ==============================================================================
# Setup and Preparation
# ==============================================================================

def prepare_constants(B, unit_cell, Hmax=6, delta_tsn=1):
    """Precompute all constants needed for efficient indexing.
    
    This function performs lattice reduction, QR decomposition, symmetry
    analysis, and precomputes various matrices used repeatedly in the
    optimization process.
    
    Args:
        B: 3x3 reciprocal lattice basis matrix in Angstrom^-1
        unit_cell: Tuple of (a, b, c, alpha, beta, gamma) in Angstroms and degrees
        Hmax: Maximum search radius for lattice reduction
        delta_tsn: Radius for target-space neighbor precomputation (None to skip)
        
    Returns:
        Dictionary containing:
        - B_red: Reduced lattice basis
        - U: Transformation matrix (B_red = B @ U)
        - Q_B_T: Transpose of Q from QR decomposition
        - R_B: R from QR decomposition  
        - S_ops_stack: Symmetry operations in reduced basis
        - PB_ops_stack: Symmetry operations in real space
        - rot_mats: Reciprocal space symmetry matrices
        - delta_tsn: The delta value used
        - D_tsn, PB_D, dY2: Precomputed neighbor offsets (if delta_tsn > 0)
    """
    B = np.ascontiguousarray(B, dtype=np.float64)
    # Apply Korkine-Zolotarev reduction for more orthogonal basis
    B_red, U = kz_reduce_integer_3d(B, Hmax=Hmax)
    # QR decomposition for efficient Babai rounding
    Q_B, R_B = np.linalg.qr(B_red)
    # Ensure R has positive diagonal for consistency
    s = np.sign(np.diag(R_B)); s[s == 0] = 1
    D = np.diag(s)
    Q_B  = Q_B @ D
    R_B  = D @ R_B
    Q_B_T = Q_B.T
    # Get lattice symmetry operations and sort by trace (identity first)
    rot_mats = get_rots_mats_symm(unit_cell)
    order = np.argsort([float(np.trace(R)) for R in rot_mats])
    rot_mats = rot_mats[order]
    # Precompute symmetry operations in reduced and real space bases
    S_ops_stack  = np.stack([Q_B_T @ R_sym.T for R_sym in rot_mats], axis=0)
    PB_ops_stack = np.stack([R_sym @ B_red   for R_sym in rot_mats], axis=0)
    if delta_tsn is not None and delta_tsn > 0:
        offs = np.arange(-delta_tsn, delta_tsn + 1, dtype=int)
        D_tsn = np.stack(
            np.meshgrid(offs, offs, offs, indexing="ij"),
            axis=-1
        ).reshape(-1, 3).T
        PB_D = np.einsum("sij,jk->sik", PB_ops_stack, D_tsn, optimize=True)
        dY2  = np.einsum("sik,sik->sk", PB_D, PB_D, optimize=True)
    else:
        D_tsn, PB_D, dY2 = None, None, None
    return dict(
        B_red=B_red, U=U, Q_B_T=Q_B_T, R_B=R_B,
        S_ops_stack=S_ops_stack, PB_ops_stack=PB_ops_stack,
        rot_mats=rot_mats,
        delta_tsn=delta_tsn,
        D_tsn=D_tsn, PB_D=PB_D, dY2=dY2,
    )


# ==============================================================================
# Main Optimization Function
# ==============================================================================

def global_optimize_via_de_prepared(
    Q, consts,
    obj="mse_symm_trimmed_auto",
    kappa=0.40,
    tol=3e-4,
    maxiter=1500,
    popsize=21,
    polish=True,
    updating="deferred",
    workers=1,
    strategy="randtobest1bin",
    init_mode="niched",
    seed_factor=3,
    niche_radius_deg=3.2,
    elite_per_niche=3,
    immigrants_frac=0.10,
    jitter_floor_deg=1.0,
    jitter_scale=0.8,
    sobol_seed=None,
    de_seed=None,
    smallN_tsn_polish=True,
    smallN_threshold=25,
    callback_every=25,
    tsn_delta=1,
    tsn_steps=1,
    tsn_winsor_q=None,
    tsn_fit_alpha=False,
    midN_threshold=80,
    midN_tsn_topK=48,
    callback_every_largeN=75,
):
    """Global optimization of crystal orientation using differential evolution.
    
    This is the main indexing function. It searches for the optimal crystal
    orientation that best explains the observed diffraction peaks, accounting
    for crystal symmetry and using robust statistics.
    
    Args:
        Q: Observed peak positions in reciprocal space, shape (3, N)
        consts: Dictionary from prepare_constants()
        obj: Objective function type:
            - "mse_symm_trimmed_auto": Trimmed mean (default, most robust)
            - "mse_symm": Standard MSE (faster, less robust)
            - "mse_small_n": TSN-enhanced for small N
        kappa: Trimming fraction (e.g., 0.40 = use best 40% of residuals)
        tol: Convergence tolerance for differential evolution
        maxiter: Maximum iterations for DE
        popsize: Population size multiplier for DE
        polish: Whether to polish final result
        updating: DE update strategy ('deferred' or 'immediate')
        workers: Number of parallel workers (1 = serial)
        strategy: DE mutation strategy
        init_mode: Initialization mode (not currently used)
        seed_factor: Population seeding parameter (not currently used)
        niche_radius_deg: Angular radius for niching (not currently used)
        elite_per_niche: Elites per niche (not currently used)
        immigrants_frac: Immigration fraction (not currently used)
        jitter_floor_deg: Minimum jitter angle (not currently used)
        jitter_scale: Jitter scaling factor (not currently used)
        sobol_seed: Seed for Sobol sequence (not currently used)
        de_seed: Random seed for differential evolution
        smallN_tsn_polish: Enable TSN polishing for small datasets
        smallN_threshold: Peak count threshold for "small" dataset
        callback_every: Callback frequency for small datasets
        tsn_delta: TSN search radius
        tsn_steps: Number of TSN refinement steps
        tsn_winsor_q: Quantile for Winsorization in TSN
        tsn_fit_alpha: Whether to fit scale in TSN
        midN_threshold: Upper threshold for medium-sized datasets
        midN_tsn_topK: Number of hardest peaks to refine for medium N
        callback_every_largeN: Callback frequency for large datasets
        
    Returns:
        Tuple of (R_best, H_best, best_val, nit, message, success) where:
        - R_best: Optimal 3x3 rotation matrix
        - H_best: Integer Miller indices for all peaks, shape (3, N)
        - best_val: Final loss value
        - nit: Number of iterations performed
        - message: Convergence message from optimizer
        - success: Boolean indicating successful convergence
    """
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    B_red, U, Q_B_T, R_B = consts["B_red"], consts["U"], consts["Q_B_T"], consts["R_B"]
    S_ops_stack, PB_ops_stack = consts["S_ops_stack"], consts["PB_ops_stack"]
    rot_mats = consts["rot_mats"]
    N = Q.shape[1]
    use_smallN = smallN_tsn_polish and (N <= smallN_threshold)
    use_midN   = (N > smallN_threshold) and (N <= midN_threshold)
    
    # Select appropriate loss function based on objective type
    if obj == "mse_symm_trimmed_auto":
        batched_loss = partial(
            loss_trimmed_batched,
            Q=Q, B_red=B_red, Q_B_T=Q_B_T, R_B=R_B,
            S_ops_stack=S_ops_stack, PB_ops_stack=PB_ops_stack,
            kappa=kappa,
        )
    elif obj == "mse_symm":
        batched_loss = partial(
            loss_mse_batched,
            Q=Q, B_red=B_red, Q_B_T=Q_B_T, R_B=R_B,
            S_ops_stack=S_ops_stack, PB_ops_stack=PB_ops_stack,
        )
    elif obj == "mse_small_n":
        D_tsn  = consts["D_tsn"]
        PB_D   = consts["PB_D"]
        batched_loss = partial(
            loss_small_n_batched,
            Q=Q, B_red=B_red, Q_B_T=Q_B_T, R_B=R_B,
            S_ops_stack=S_ops_stack, PB_ops_stack=PB_ops_stack,
            D=D_tsn, PB_D=PB_D,
            kappa=kappa,
        )
    else:
        raise ValueError(f"Unknown objective type: {obj}")
    
    # Use Latin hypercube sampling for initial population
    init = 'latinhypercube'

    # Vectorized objective function for efficient batch evaluation
    def objective_vec(pop):
        pop = np.asarray(pop, dtype=np.float64)
        if pop.ndim == 1:
            pop = pop[None, :]
        elif pop.ndim == 2:
            if pop.shape[1] == 4:
                pass
            elif pop.shape[0] == 4:
                pop = pop.T
            else:
                raise AssertionError(f"Unexpected shape {pop.shape}; expected (NP,4) or (4,NP)")
        else:
            raise AssertionError(f"Unexpected ndim {pop.ndim}")
        vals = batched_loss(pop)
        return vals
    
    # Shadow variables to track best solution found during callbacks
    best_q_shadow = None
    best_val_shadow = np.inf
    gen_counter = {"g": 0}
    best_tracker = {
        "best_val": np.inf,
        "last_improve_gen": 0,
    }
    patience = 600        # Stop if no improvement for this many generations
    target_loss = 1e-7    # Stop if loss drops below this threshold
    
    def de_callback(xk, convergence):
        """Callback function called during DE iterations for refinement and early stopping."""
        gen_counter["g"] += 1
        
        # Determine callback frequency based on dataset size
        gen_counter["g"] += 1
        if use_smallN:
            if (callback_every is None) or (gen_counter["g"] % max(1, int(callback_every)) != 0):
                return False
        elif use_midN:
            if (callback_every_largeN is None) or (gen_counter["g"] % max(1, int(callback_every_largeN)) != 0):
                return False
        else:
            if (callback_every_largeN is None) or (gen_counter["g"] % max(1, int(callback_every_largeN)) != 0):
                return False
        qk = np.asarray(xk, dtype=np.float64); qk /= (np.linalg.norm(qk) + 1e-12)
        valk = float(batched_loss(qk[None, :])[0])
        Rk = quat_to_R(qk)
        candidates = [(qk, valk)]
        
        # Apply appropriate polishing based on dataset size
        if use_smallN:
            for fit_alpha in (False, tsn_fit_alpha):
                Rp, _, _, _ = polish_R_smallN_tsn(Rk, Q, consts,
                        kappa=kappa, delta=tsn_delta, winsor_q=tsn_winsor_q,
                        fit_alpha=fit_alpha, steps=tsn_steps, indices=None)
                qp = R_to_quat(Rp)
                vp = float(batched_loss(qp[None, :])[0])
                candidates.append((qp, vp))
        elif use_midN:
            Ihard = hardest_peak_indices(Rk, Q, consts, midN_tsn_topK)
            for fit_alpha in (False, tsn_fit_alpha):
                Rp, _, _, _ = polish_R_smallN_tsn(Rk, Q, consts,
                        kappa=kappa, delta=tsn_delta, winsor_q=None,
                        fit_alpha=fit_alpha, steps=2, indices=Ihard)
                qp = R_to_quat(Rp)
                vp = float(batched_loss(qp[None, :])[0])
                candidates.append((qp, vp))
        else:
            Rp, _, _, _ = polish_R_smallN_tsn(Rk, Q, consts,
                    kappa=kappa, delta=0, winsor_q=None,
                    fit_alpha=tsn_fit_alpha, steps=1, indices=None)
            qp = R_to_quat(Rp)
            vp = float(batched_loss(qp[None, :])[0])
            candidates.append((qp, vp))
        best_local_q, best_local_v = min(candidates, key=lambda t: t[1])
        nonlocal best_q_shadow, best_val_shadow
        if best_local_v < best_tracker["best_val"] - 1e-9:
            best_tracker["best_val"] = best_local_v
            best_tracker["last_improve_gen"] = gen_counter["g"]
        if best_local_v < best_val_shadow - 1e-12:
            best_q_shadow, best_val_shadow = best_local_q, best_local_v
        
        # Early stopping criteria
        if (target_loss is not None) and (best_tracker["best_val"] <= target_loss):
            return True
        if gen_counter["g"] - best_tracker["last_improve_gen"] >= patience:
            print("DE callback: stopping early")
            return True
        return False
    
    # Run differential evolution optimizer
    result = differential_evolution(
        objective_vec,
        bounds=[(-1, 1)] * 4,
        tol=tol, maxiter=maxiter, popsize=popsize,
        polish=polish,
        updating=updating,
        workers=workers,
        vectorized=True,
        strategy=strategy,
        init=init,
        seed=de_seed,
        callback=de_callback,
        recombination=0.7,
        mutation=(0.5, 1)
    )
    
    # Select best result between DE output and callback-polished solution
    q_best_de = np.array(result.x, dtype=np.float64)
    q_best_de /= (np.linalg.norm(q_best_de) + 1e-12)
    val_best_de = float(batched_loss(q_best_de[None, :])[0])
    if best_q_shadow is not None and best_val_shadow < val_best_de - 1e-12:
        q_best = best_q_shadow
        best_val = best_val_shadow
    else:
        q_best = q_best_de
        best_val = val_best_de
    
    # Convert final quaternion to rotation matrix and compute Miller indices
    R_best = quat_to_R(q_best)
    M_final = Q_B_T @ (R_best.T @ Q)
    Hp_final = babai_nearest_batched(R_B, M_final[None, :, :])[0]
    H_best = consts["U"] @ Hp_final
    return R_best, H_best, best_val, result.nit, result.message, result.success
