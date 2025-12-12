"""
Crystal Lattice Indexer (GPU Implementation)

This module implements a GPU-accelerated crystallographic indexing algorithm using PyTorch.
It determines crystal orientation from diffraction peaks using quaternion-based rotation
representations, lattice basis reduction, symmetry operations, and differential evolution
optimization with full GPU acceleration.

Key Features:
- GPU-accelerated differential evolution optimizer in PyTorch
- Batch processing of multiple indexing problems simultaneously
- Quaternion and rotation matrix operations on GPU
- Target-space neighbor (TSN) polishing with GPU acceleration
- Symmetry-aware loss functions with trimmed mean for robustness
- Multi-problem optimization for processing multiple crystals in parallel
"""

import time
import math
from collections import namedtuple
import numpy as np
import torch
import gemmi
from numba import njit

# Named tuple for differential evolution results
DEOptimizeResult = namedtuple(
    "DEOptimizeResult", ["x", "fun", "nit", "nfev", "message", "success"]
)


# ==============================================================================
# Differential Evolution Optimizer (GPU-Accelerated)
# ==============================================================================

def _sample_indices_triplets(npop, device):
    """Sample three distinct random indices for each individual in population.
    
    Used in DE mutation to select r0, r1, r2 where all three are different
    from each other and from the target individual.
    
    Args:
        npop: Population size
        device: PyTorch device (cuda or cpu)
        
    Returns:
        Tuple of (r0, r1, r2) tensors of shape (npop,) containing distinct indices
    """
    idx = torch.arange(npop, device=device)

    # Generate initial random indices
    r0 = torch.randint(0, npop, (npop,), device=device)
    r1 = torch.randint(0, npop, (npop,), device=device)
    r2 = torch.randint(0, npop, (npop,), device=device)

    # Resample until all indices are distinct
    while True:
        bad = (
            (r0 == idx) | (r1 == idx) | (r2 == idx) |
            (r0 == r1) | (r0 == r2) | (r1 == r2)
        )
        if not bad.any():
            break
        num_bad = int(bad.sum().item())
        r0_new = torch.randint(0, npop, (num_bad,), device=device)
        r1_new = torch.randint(0, npop, (num_bad,), device=device)
        r2_new = torch.randint(0, npop, (num_bad,), device=device)
        r0[bad] = r0_new
        r1[bad] = r1_new
        r2[bad] = r2_new

    return r0, r1, r2


def _latin_hypercube(n, d, device, dtype):
    """Generate Latin hypercube samples on GPU.
    
    Creates a stratified random sample where each dimension is divided into
    n equal intervals with one sample per interval.
    
    Args:
        n: Number of samples
        d: Number of dimensions
        device: PyTorch device
        dtype: PyTorch data type
        
    Returns:
        Tensor of shape (n, d) with values in [0, 1]
    """
    samples = torch.empty((n, d), device=device, dtype=dtype)
    for j in range(d):
        perm = torch.randperm(n, device=device)
        jitter = torch.rand(n, device=device, dtype=dtype)
        samples[:, j] = (perm.to(dtype) + jitter) / float(n)
    return samples


def differential_evolution_torch(
    func,
    bounds,
    args=(),
    strategy="randtobest1bin",
    maxiter=1000,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1.0),
    recombination=0.7,
    seed=None,
    callback=None,
    disp=False,
    atol=0.0,
    init="latinhypercube",
    device=None,
    dtype=torch.float32,
    profile=False,
    n_problems=1,  
):
    """GPU-accelerated differential evolution optimizer using PyTorch.
    
    Implements differential evolution algorithm entirely on GPU for fast optimization
    of multiple problems simultaneously. Supports vectorized objective functions.
    
    Args:
        func: Objective function taking (P*npop, dim) tensor and returning (P*npop,) losses
        bounds: List of (lower, upper) tuples for each dimension
        args: Additional arguments for func (not typically used with vectorized functions)
        strategy: Mutation strategy ('best1bin' or 'randtobest1bin')
        maxiter: Maximum number of iterations
        popsize: Population size multiplier (actual pop = popsize * dim)
        tol: Relative tolerance for convergence
        mutation: Mutation constant (scalar or (min, max) tuple for dithering)
        recombination: Crossover probability
        seed: Random seed
        callback: Callback function called each iteration
        disp: Whether to display progress
        atol: Absolute tolerance for convergence
        init: Initialization method ('latinhypercube', 'random', or array)
        device: PyTorch device ('cuda' or 'cpu')
        dtype: PyTorch data type
        profile: Whether to print timing breakdown
        n_problems: Number of independent problems to solve simultaneously
        
    Returns:
        DEOptimizeResult with x, fun, nit, nfev, message, success
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = (isinstance(device, str) and device.startswith("cuda")) or (
        isinstance(device, torch.device) and device.type == "cuda"
    )

    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    if strategy not in ("best1bin", "randtobest1bin"):
        raise ValueError("strategy must be 'best1bin' or 'randtobest1bin'")

    # Parse bounds and convert to tensors
    bounds = list(bounds)
    if len(bounds) == 0:
        raise ValueError("bounds must be non-empty")

    dim = len(bounds)
    lower = torch.tensor([b[0] for b in bounds], device=device, dtype=dtype)
    upper = torch.tensor([b[1] for b in bounds], device=device, dtype=dtype)
    diff = upper - lower
    if torch.any(diff <= 0):
        raise ValueError("Each bound must satisfy lower < upper")

    center = 0.5 * (upper + lower)
    recip_diff = 1.0 / diff
    macheps = torch.finfo(dtype).eps

    # Scale parameters from [0,1] to actual bounds
    def _scale_parameters(u):
        return center + (u - 0.5) * diff

    # Unscale parameters from bounds to [0,1]
    def _unscale_parameters(x):
        return (x - center) * recip_diff + 0.5

    # Setup mutation parameter (constant or dithering)
    if isinstance(mutation, (tuple, list)):
        if len(mutation) != 2:
            raise ValueError("mutation tuple must have length 2")
        m_lo, m_hi = float(min(mutation)), float(max(mutation))
        dither = (m_lo, m_hi)
        scale = 0.5 * (m_lo + m_hi)
    else:
        scale = float(mutation)
        dither = None

    cr = float(recombination)

    # Determine population size from init parameter
    if isinstance(init, str):
        if popsize < 1:
            raise ValueError("popsize must be >= 1")
        npop = popsize * dim
        init_arr = None
    else:
        init_arr = torch.as_tensor(init, dtype=dtype)
        if init_arr.ndim == 2:
            if init_arr.shape[1] != dim:
                raise ValueError(f"init array must have shape (S, {dim}) or (P,S,{dim})")
            if init_arr.shape[0] < 5:
                raise ValueError("init array must have at least 5 individuals")
            npop = int(init_arr.shape[0])
        elif init_arr.ndim == 3:
            if init_arr.shape[2] != dim:
                raise ValueError(f"init array must have shape (S, {dim}) or (P,S,{dim})")
            if init_arr.shape[1] < 5:
                raise ValueError("init array must have at least 5 individuals per problem")
            if init_arr.shape[0] != n_problems:
                raise ValueError("init first dim must equal n_problems")
            npop = int(init_arr.shape[1])
        else:
            raise ValueError(f"init array must have ndim 2 or 3, got {init_arr.ndim}")

    # Initialize population
    if isinstance(init, str):
        if init == "latinhypercube":
            samples = _latin_hypercube(n_problems * npop, dim, device=device, dtype=dtype)
            pop = samples.view(n_problems, npop, dim)
        elif init == "random":
            pop = torch.rand((n_problems, npop, dim), device=device, dtype=dtype)
        else:
            raise ValueError("init must be 'latinhypercube', 'random', or array-like")
    else:
        if init_arr.ndim == 2:
            x0 = init_arr.to(device=device, dtype=dtype)
            u0 = _unscale_parameters(x0).clamp(0.0, 1.0)
            pop = u0.unsqueeze(0).repeat(n_problems, 1, 1)
        else:
            x0 = init_arr.to(device=device, dtype=dtype)
            u0 = _unscale_parameters(x0).clamp(0.0, 1.0)
            pop = u0

    trial_pop = torch.empty_like(pop)

    if profile:
        t_total0 = time.perf_counter()
        t_sample = t_mutcross = t_bounds = t_obj = 0.0
        t_sel = t_conv = t_cb = 0.0

        def _tic():
            if is_cuda:
                torch.cuda.synchronize()
            return time.perf_counter()

        def _toc(t0):
            if is_cuda:
                torch.cuda.synchronize()
            return time.perf_counter() - t0
    else:
        t_sample = t_mutcross = t_bounds = t_obj = 0.0
        t_sel = t_conv = t_cb = 0.0
        t_total0 = None

        def _tic():
            return 0.0

        def _toc(t0):
            return 0.0

    P = n_problems

    # Evaluate initial population
    x_pop = _scale_parameters(pop)
    x_flat = x_pop.reshape(P * npop, dim)
    t0 = _tic()
    energies_flat = func(x_flat, *args) if args else func(x_flat)
    t_obj += _toc(t0)

    if not isinstance(energies_flat, torch.Tensor):
        raise TypeError("func must return a torch.Tensor")
    energies_flat = energies_flat.to(device=device, dtype=dtype).view(-1)
    if energies_flat.numel() != P * npop:
        raise RuntimeError(
            f"Vectorized objective must return shape ({P*npop},), got {tuple(energies_flat.shape)}"
        )

    energies = energies_flat.view(P, npop)

    nfev = 1
    nit = 0
    success = False
    message = ""

    best_idx = torch.argmin(energies, dim=1)
    best_u = pop[torch.arange(P, device=device), best_idx]
    best_energy = energies[torch.arange(P, device=device), best_idx]

    # Main differential evolution loop
    while nit < maxiter:
        nit += 1

        # Update mutation scale if dithering
        if dither is not None:
            scale = float(
                torch.empty(1, device=device, dtype=dtype)
                .uniform_(dither[0], dither[1])
                .item()
            )

        t0 = _tic()
        r0, r1, r2 = _sample_indices_triplets(npop, device=device)
        t_sample += _toc(t0)

        t0 = _tic()
        best_idx = torch.argmin(energies, dim=1)
        best_u = pop[torch.arange(P, device=device), best_idx]
        best_energy = energies[torch.arange(P, device=device), best_idx]

        best_u_exp = best_u.unsqueeze(1)

        # Apply mutation strategy
        if strategy == "best1bin":
            # Mutation: best + scale * (r0 - r1)
            mutant = best_u_exp + scale * (pop[:, r0, :] - pop[:, r1, :])
        elif strategy == "randtobest1bin":
            # Mutation: r0 + scale * (best - r0) + scale * (r1 - r2)
            base = pop[:, r0, :]
            mutant = base + scale * (best_u_exp - base)
            mutant = mutant + scale * (pop[:, r1, :] - pop[:, r2, :])
        else:
            raise RuntimeError("Unsupported strategy")

        # Binomial crossover
        cross = (torch.rand((P, npop, dim), device=device) < cr)
        # Ensure at least one parameter is mutated
        fill_idx = torch.randint(0, dim, (P, npop), device=device)
        p_idx = torch.arange(P, device=device).unsqueeze(1).expand(P, npop)
        i_idx = torch.arange(npop, device=device).unsqueeze(0).expand(P, npop)
        cross[p_idx, i_idx, fill_idx] = True

        trial_pop[:] = torch.where(cross, mutant, pop)
        t_mutcross += _toc(t0)

        # Handle out-of-bounds individuals
        t0 = _tic()
        mask = (trial_pop < 0.0) | (trial_pop > 1.0)
        if mask.any():
            num_bad = int(mask.sum().item())
            trial_pop[mask] = torch.rand(num_bad, device=device, dtype=dtype)
        t_bounds += _toc(t0)

        # Evaluate trial population
        t0 = _tic()
        x_trial = _scale_parameters(trial_pop)
        x_trial_flat = x_trial.reshape(P * npop, dim)
        trial_energies_flat = func(x_trial_flat, *args) if args else func(x_trial_flat)
        t_obj += _toc(t0)

        if not isinstance(trial_energies_flat, torch.Tensor):
            raise TypeError("func must return a torch.Tensor")
        trial_energies_flat = trial_energies_flat.to(device=device, dtype=dtype).view(-1)
        if trial_energies_flat.numel() != P * npop:
            raise RuntimeError(
                f"Vectorized objective must return shape ({P*npop},), got {tuple(trial_energies_flat.shape)}"
            )
        nfev += 1
        trial_energies = trial_energies_flat.view(P, npop)

        t0 = _tic()
        improved = trial_energies <= energies
        if improved.any():
            pop[improved] = trial_pop[improved]
            energies[improved] = trial_energies[improved]
        t_sel += _toc(t0)

        t0 = _tic()
        energies_all = energies.view(-1)
        if torch.isinf(energies_all).any():
            conv = float("inf")
            converged = False
        else:
            mean = energies_all.mean()
            std = energies_all.std()
            conv_tensor = std / (torch.abs(mean) + macheps)
            conv = float(conv_tensor.item())
            converged = bool(std <= atol + tol * torch.abs(mean))
        t_conv += _toc(t0)

        if callback is not None:
            t0 = _tic()
            best_idx = torch.argmin(energies, dim=1)
            best_u = pop[torch.arange(P, device=device), best_idx]
            best_energy = energies[torch.arange(P, device=device), best_idx]

            xk_batch = _scale_parameters(best_u).detach().cpu().numpy()
            if P == 1:
                xk_cb = xk_batch[0]
            else:
                xk_cb = xk_batch

            stop = callback(xk_cb, conv)
            t_cb += _toc(t0)
            if stop:
                success = True
                message = "Callback terminated optimization."
                break

        if disp:
            if P == 1:
                print(f"iter {nit}, best f = {float(best_energy[0]):.6g}, conv = {conv:.3g}")
            else:
                print(
                    f"iter {nit}, best f (min over problems) = "
                    f"{float(best_energy.min().item()):.6g}, conv = {conv:.3g}"
                )

        if converged:
            success = True
            message = "Optimization terminated successfully."
            break

    if not success and message == "":
        message = "Maximum number of iterations has been exceeded."

    best_idx = torch.argmin(energies, dim=1)
    best_u = pop[torch.arange(P, device=device), best_idx]
    best_energy = energies[torch.arange(P, device=device), best_idx]
    x_best = _scale_parameters(best_u).detach().cpu().numpy()

    if profile:
        if is_cuda:
            torch.cuda.synchronize()
        t_total = time.perf_counter() - t_total0
        def pct(x):
            return (x / t_total * 100.0) if t_total > 0 else 0.0
        print("\n[DE Torch profile]")
        print(f"  total       = {t_total:.4f} s (100.0%)")
        print(f"  sample idx  = {t_sample:.4f} s ({pct(t_sample):5.1f}%)")
        print(f"  mut+cross   = {t_mutcross:.4f} s ({pct(t_mutcross):5.1f}%)")
        print(f"  bounds      = {t_bounds:.4f} s ({pct(t_bounds):5.1f}%)")
        print(f"  objective   = {t_obj:.4f} s ({pct(t_obj):5.1f}%)")
        print(f"  selection   = {t_sel:.4f} s ({pct(t_sel):5.1f}%)")
        print(f"  convergence = {t_conv:.4f} s ({pct(t_conv):5.1f}%)")
        print(f"  callback    = {t_cb:.4f} s ({pct(t_cb):5.1f}%)")

    fun = best_energy.detach().cpu().numpy()
    
    if P == 1:
        x_best = x_best[0]
        fun = float(fun[0])

    return DEOptimizeResult(
        x=x_best,
        fun=fun,
        nit=nit,
        nfev=nfev,
        message=message,
        success=success,
    )


# ==============================================================================
# Quaternion Operations (NumPy for CPU)
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


# ==============================================================================
# Lattice Reduction and Babai Rounding (NumPy/Numba)
# ==============================================================================

@njit(fastmath=True)
def _babai_nearest_batched_core(R_B, M):
    """JIT-compiled Babai nearest plane algorithm for batch lattice rounding.
    
    Args:
        R_B: Upper triangular R matrix from QR decomposition
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


def _shortest_vector_enum_vec(B, Hmax=6):
    """Find shortest lattice vector using enumeration over a grid.
    
    Args:
        B: 3x3 lattice basis matrix
        Hmax: Maximum integer coordinate to search
        
    Returns:
        Tuple of (H, V, norm_squared) where H is integer coefficients,
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
# Quaternion and Rotation Operations (PyTorch for GPU)
# ==============================================================================

def quat_to_R_batch_torch(quats, eps=1e-12):
    """Convert batch of quaternions to rotation matrices on GPU.
    
    Args:
        quats: Quaternions of shape (B, 4) or (4,) in (w, x, y, z) format
        eps: Small value to prevent division by zero
        
    Returns:
        Rotation matrices of shape (B, 3, 3)
        
    Raises:
        ValueError: If input shape is invalid
    """
    q = quats
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat_to_R_batch_torch expects (B,4) or (4,), got {tuple(q.shape)}")

    n = q.norm(dim=-1, keepdim=True) + eps
    q = q / n
    w, x, y, z = q.unbind(-1)

    B = q.shape[0]
    R = torch.empty((B, 3, 3), device=q.device, dtype=q.dtype)

    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)

    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)

    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)

    return R


def babai_nearest_batched_torch(R_B, M):
    """GPU-accelerated Babai nearest plane algorithm using PyTorch.
    
    Args:
        R_B: Upper triangular R matrix from QR decomposition (tensor)
        M: Tensor of shape (K, 3, N) containing vectors to round
        
    Returns:
        Integer lattice coordinates as tensor of shape (K, 3, N)
        
    Raises:
        ValueError: If M doesn't have shape (K, 3, N)
    """
    if M.ndim != 3 or M.shape[1] != 3:
        raise ValueError(f"babai_nearest_batched_torch expects M with shape (K,3,N), got {tuple(M.shape)}")

    r00 = R_B[0, 0]
    r11 = R_B[1, 1]
    r22 = R_B[2, 2]
    r01 = R_B[0, 1]
    r02 = R_B[0, 2]
    r12 = R_B[1, 2]

    y2 = M[:, 2, :]
    h2 = torch.round(y2 / r22)

    y1 = M[:, 1, :] - r12 * h2
    y0 = M[:, 0, :] - r02 * h2

    h1 = torch.round(y1 / r11)
    y0 = y0 - r01 * h1

    h0 = torch.round(y0 / r00)

    H = torch.stack([h0, h1, h2], dim=1).to(torch.int64)
    return H


# ==============================================================================
# Loss Functions (GPU-Accelerated)
# ==============================================================================

def build_loss_trimmed_multi_torch(
    Q_batch_t,
    R_B_t,
    S_t,
    PB_t,
    kappa=0.40,
    device="cuda",
    dtype=torch.float64,
):
    """Build GPU-accelerated trimmed mean loss function for multiple problems.
    
    Creates a closure that evaluates orientation quality using a trimmed mean
    of residuals, which is robust to outliers. Processes multiple indexing
    problems in parallel on GPU.
    
    Args:
        Q_batch_t: Observed peaks tensor of shape (P, 3, N) for P problems
        R_B_t: Upper triangular matrix from QR decomposition
        S_t: Symmetry operations in reduced basis (S, 3, 3)
        PB_t: Symmetry operations in real space (S, 3, 3)
        kappa: Trimming fraction (0.4 = keep best 40% of residuals)
        device: PyTorch device
        dtype: PyTorch data type
        
    Returns:
        Loss function taking (P*npop, 4) quaternions and returning (P*npop,) losses
    """
    P, _, N = Q_batch_t.shape
    S = int(S_t.shape[0])
    h = int(math.ceil(kappa * N))

    @torch.no_grad()
    def loss_fn(quats_flat):
        q = quats_flat.to(device=device, dtype=dtype)
        Btot = q.shape[0]
        if Btot % P != 0:
            raise ValueError(f"Expected Btot multiple of P={P}, got {Btot}")
        npop = Btot // P

        q = q.view(P, npop, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)

        Rk_flat = quat_to_R_batch_torch(q.view(-1, 4))
        Rk = Rk_flat.view(P, npop, 3, 3)

        A = torch.einsum("pbji,pjn->pbin", Rk, Q_batch_t)

        A_sym = torch.einsum("sij,pbjn->spbin", S_t, A)

        A_resh = A_sym.reshape(S * P * npop, 3, N)
        H_sym = babai_nearest_batched_torch(R_B_t, A_resh)
        H_sym = H_sym.view(S, P, npop, 3, N)

        Y_sym = torch.einsum(
            "sij,spbjn->spbin",
            PB_t,
            H_sym.to(dtype=PB_t.dtype),
        )

        diff = A.unsqueeze(0) - Y_sym
        r2 = torch.einsum("spbin,spbin->spbn", diff, diff)
        r2_min, _ = torch.min(r2, dim=0)

        part, _ = torch.topk(r2_min, k=h, dim=2, largest=False)
        t2_vec = torch.quantile(part, 0.90, dim=2)
        clipped = torch.minimum(part, t2_vec.unsqueeze(-1))
        out = clipped.mean(dim=2)

        return out.reshape(P * npop)

    return loss_fn


def build_loss_mse_multi_torch(
    Q_batch_t,
    R_B_t,
    S_t,
    PB_t,
    device="cuda",
    dtype=torch.float64,
):
    """Build GPU-accelerated MSE loss function for multiple problems.
    
    Creates a loss function using standard mean squared error without trimming.
    Faster than trimmed version but less robust to outliers.
    
    Args:
        Q_batch_t: Observed peaks tensor of shape (P, 3, N)
        R_B_t: Upper triangular matrix from QR decomposition
        S_t: Symmetry operations in reduced basis (S, 3, 3)
        PB_t: Symmetry operations in real space (S, 3, 3)
        device: PyTorch device
        dtype: PyTorch data type
        
    Returns:
        Loss function taking (P*npop, 4) quaternions and returning (P*npop,) losses
    """
    P, _, N = Q_batch_t.shape

    Q_sq_mean = torch.mean(torch.sum(Q_batch_t * Q_batch_t, dim=1), dim=1)

    @torch.no_grad()
    def loss_fn(quats_flat):
        q = quats_flat.to(device=device, dtype=dtype)
        Btot = q.shape[0]
        if Btot % P != 0:
            raise ValueError(f"Expected Btot multiple of P={P}, got {Btot}")
        npop = Btot // P

        q = q.view(P, npop, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)

        Rk_flat = quat_to_R_batch_torch(q.view(-1, 4))
        Rk = Rk_flat.view(P, npop, 3, 3)

        A = torch.einsum("pbji,pjn->pbin", Rk, Q_batch_t)

        Bsz = npop
        S = S_t.shape[0]

        A_sym = torch.einsum("sij,pbjn->spbin", S_t, A)

        A_resh = A_sym.reshape(S * P * Bsz, 3, N)
        H_sym = babai_nearest_batched_torch(R_B_t, A_resh)
        H_sym = H_sym.view(S, P, Bsz, 3, N)

        Y_sym = torch.einsum(
            "sij,spbjn->spbin",
            PB_t,
            H_sym.to(dtype=PB_t.dtype),
        )

        YY = torch.sum(Y_sym * Y_sym, dim=3)
        YY_mean = torch.mean(YY, dim=3)

        AY = torch.sum(A.unsqueeze(0) * Y_sym, dim=3)
        AY_mean = torch.mean(AY, dim=3)

        Q_sq = Q_sq_mean.view(1, P, 1)

        val = Q_sq + YY_mean - 2.0 * AY_mean
        best, _ = torch.min(val, dim=0)

        return best.reshape(P * Bsz)

    return loss_fn


def build_loss_small_n_multi_torch(
    Q_batch_t,
    R_B_t,
    S_t,
    PB_t,
    dY2_t,
    PB_D_t,
    kappa=0.40,
    device="cuda",
    dtype=torch.float64,
):
    """Build loss function with target-space neighbor search for small datasets.
    
    Evaluates loss while considering nearby lattice points in target space,
    which can improve accuracy for datasets with few peaks.
    
    Args:
        Q_batch_t: Observed peaks tensor of shape (P, 3, N)
        R_B_t: Upper triangular matrix from QR decomposition
        S_t: Symmetry operations in reduced basis (S, 3, 3)
        PB_t: Symmetry operations in real space (S, 3, 3)
        dY2_t: Precomputed squared norms of neighbor offsets (S, K)
        PB_D_t: Precomputed neighbor offsets in real space (S, 3, K)
        kappa: Trimming fraction
        device: PyTorch device
        dtype: PyTorch data type
        
    Returns:
        Loss function taking (P*npop, 4) quaternions and returning (P*npop,) losses
    """
    P, _, N = Q_batch_t.shape
    S = S_t.shape[0]
    h = int(math.ceil(kappa * N))

    @torch.no_grad()
    def loss_fn(quats_flat):
        q = quats_flat.to(device=device, dtype=dtype)
        Btot = q.shape[0]
        if Btot % P != 0:
            raise ValueError(f"Expected Btot multiple of P={P}, got {Btot}")
        npop = Btot // P

        q = q.view(P, npop, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)

        Rk_flat = quat_to_R_batch_torch(q.view(-1, 4))
        Rk = Rk_flat.view(P, npop, 3, 3)

        A = torch.einsum("pbji,pjn->pbin", Rk, Q_batch_t)

        Bsz = npop

        A_sym = torch.einsum("sij,pbjn->spbin", S_t, A)

        A_resh = A_sym.reshape(S * P * Bsz, 3, N)
        H_babai = babai_nearest_batched_torch(R_B_t, A_resh).reshape(S, P, Bsz, 3, N)

        Y0 = torch.einsum(
            "sij,spbjn->spbin",
            PB_t,
            H_babai.to(dtype=PB_t.dtype),
        )

        E  = A.unsqueeze(0) - Y0
        E2 = torch.einsum("spbin,spbin->spbn", E, E)

        dot = torch.einsum("spbin,sik->spbkn", E, PB_D_t)
        r2_k = E2[:, :, :, None, :] + dY2_t[:, None, None, :, None] - 2.0 * dot

        r2_best_per_sym, _ = torch.min(r2_k, dim=3)
        r2_min, _ = torch.min(r2_best_per_sym, dim=0)

        part, _ = torch.topk(r2_min, k=h, dim=2, largest=False)
        out = part.mean(dim=2)
        return out.reshape(P * Bsz)

    return loss_fn


# ==============================================================================
# Conversion and Utility Functions (GPU)
# ==============================================================================

@torch.no_grad()
def R_batch_to_quat_batch_torch(
    R_batch,
    device=None,
    dtype=torch.float64,
    eps=1e-12,
):
    """Convert batch of rotation matrices to quaternions on GPU.
    
    Uses numerically stable conversion based on trace or largest diagonal element.
    
    Args:
        R_batch: Rotation matrices of shape (..., 3, 3)
        device: PyTorch device (inferred if None)
        dtype: PyTorch data type
        eps: Small value to prevent division by zero
        
    Returns:
        Quaternions of shape (..., 4) in (w, x, y, z) format
        
    Raises:
        ValueError: If input shape is invalid
    """
    if device is None:
        if isinstance(R_batch, torch.Tensor):
            device = R_batch.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    R = torch.as_tensor(R_batch, device=device, dtype=dtype)
    if R.ndim < 2 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R_batch_to_quat_batch_torch expects (...,3,3), got {tuple(R.shape)}")

    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    diag0 = R[..., 0, 0]
    diag1 = R[..., 1, 1]
    diag2 = R[..., 2, 2]

    t_pos = t > 0
    diag_stack = torch.stack([diag0, diag1, diag2], dim=-1)
    i_max = torch.argmax(diag_stack, dim=-1)

    m0 = (~t_pos) & (i_max == 0)
    m1 = (~t_pos) & (i_max == 1)
    m2 = (~t_pos) & (i_max == 2)

    q0 = torch.zeros_like(t, dtype=dtype, device=device)
    q1 = torch.zeros_like(t, dtype=dtype, device=device)
    q2 = torch.zeros_like(t, dtype=dtype, device=device)
    q3 = torch.zeros_like(t, dtype=dtype, device=device)

    if t_pos.any():
        S = torch.sqrt(t[t_pos] + 1.0) * 2.0
        q0[t_pos] = 0.25 * S
        q1[t_pos] = (R[t_pos, 2, 1] - R[t_pos, 1, 2]) / S
        q2[t_pos] = (R[t_pos, 0, 2] - R[t_pos, 2, 0]) / S
        q3[t_pos] = (R[t_pos, 1, 0] - R[t_pos, 0, 1]) / S

    if m0.any():
        S = torch.sqrt(1.0 + diag0[m0] - diag1[m0] - diag2[m0]) * 2.0
        q0[m0] = (R[m0, 2, 1] - R[m0, 1, 2]) / S
        q1[m0] = 0.25 * S
        q2[m0] = (R[m0, 0, 1] + R[m0, 1, 0]) / S
        q3[m0] = (R[m0, 0, 2] + R[m0, 2, 0]) / S

    if m1.any():
        S = torch.sqrt(1.0 + diag1[m1] - diag0[m1] - diag2[m1]) * 2.0
        q0[m1] = (R[m1, 0, 2] - R[m1, 2, 0]) / S
        q1[m1] = (R[m1, 0, 1] + R[m1, 1, 0]) / S
        q2[m1] = 0.25 * S
        q3[m1] = (R[m1, 1, 2] + R[m1, 2, 1]) / S

    if m2.any():
        S = torch.sqrt(1.0 + diag2[m2] - diag0[m2] - diag1[m2]) * 2.0
        q0[m2] = (R[m2, 1, 0] - R[m2, 0, 1]) / S
        q1[m2] = (R[m2, 0, 2] + R[m2, 2, 0]) / S
        q2[m2] = (R[m2, 1, 2] + R[m2, 2, 1]) / S
        q3[m2] = 0.25 * S

    q = torch.stack([q0, q1, q2, q3], dim=-1)
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    return q


@torch.no_grad()
def hardest_peak_indices_multi_torch(
    R_batch_t,
    Q_batch_t,
    S_ops,
    R_B_t,
    PB_ops,
    K,
    device="cuda",
    dtype=torch.float64,
):
    """Identify K peaks with largest indexing residuals for each problem.
    
    Used to focus refinement on the most problematic peaks in large datasets.
    
    Args:
        R_batch_t: Current rotation matrices of shape (P, 3, 3)
        Q_batch_t: All observed peaks of shape (P, 3, N)
        S_ops: Symmetry operations in reduced basis (S, 3, 3)
        R_B_t: Upper triangular matrix from QR
        PB_ops: Symmetry operations in real space (S, 3, 3)
        K: Number of hardest peaks to return per problem
        device: PyTorch device
        dtype: PyTorch data type
        
    Returns:
        Tensor of shape (P, K) containing indices of K hardest peaks per problem
    """
    P, _, N = Q_batch_t.shape
    S = S_ops.shape[0]

    A = torch.einsum("pij,pjn->pin", R_batch_t, Q_batch_t)

    A_sym = torch.einsum("sij,pjn->spin", S_ops, A)

    A_sym_flat = A_sym.reshape(-1, 3, N)
    H_b_flat = babai_nearest_batched_torch(R_B_t, A_sym_flat)
    H_b = H_b_flat.reshape(S, P, 3, N)

    Y0 = torch.einsum("sij,spjn->spin", PB_ops, H_b.to(dtype))

    A_exp = A.unsqueeze(0)
    diff = A_exp - Y0

    r2 = (diff * diff).sum(dim=2)

    r2_min, _ = torch.min(r2, dim=0)

    K_eff = int(min(K, N))
    idx = torch.topk(r2_min, k=K_eff, dim=1, largest=True).indices

    return idx


# ==============================================================================
# Polishing and Refinement (GPU)
# ==============================================================================

@torch.no_grad()
def polish_R_smallN_tsn_multi_torch(
    R_init_batch,
    Q_batch_t,
    S_ops,
    R_B_t,
    PB_ops,
    consts,
    *,
    kappa=0.40,
    PB_D=None,
    dY2=None,
    delta=1,
    winsor_q=None,
    fit_alpha=True,
    alpha_clip=(0.97, 1.03),
    steps=2,
    indices=None,
    device="cuda",
    dtype=torch.float64,
):
    """Polish rotation matrices using target-space neighbor search on GPU.
    
    GPU-accelerated version of TSN polishing that refines orientations by
    searching nearby integer lattice points. Processes multiple problems
    simultaneously.
    
    Args:
        R_init_batch: Initial rotation matrices of shape (P, 3, 3)
        Q_batch_t: Observed peaks of shape (P, 3, N)
        S_ops: Symmetry operations in reduced basis (S, 3, 3)
        R_B_t: Upper triangular matrix from QR
        PB_ops: Symmetry operations in real space (S, 3, 3)
        consts: Dictionary of precomputed constants
        kappa: Trimming fraction for robust fitting
        PB_D: Precomputed neighbor offsets (S, 3, K) or None
        dY2: Precomputed squared norms (S, K) or None
        delta: Search radius in target space (±delta in each direction)
        winsor_q: Quantile for Winsorization (None to disable)
        fit_alpha: Whether to fit a scale factor
        alpha_clip: (min, max) bounds for scale factor
        steps: Number of refinement iterations
        indices: Subset of peak indices to use (None = all peaks)
        device: PyTorch device
        dtype: PyTorch data type
        
    Returns:
        Tuple of (R, H, s_best, alpha) where:
        - R: Refined rotation matrices (P, 3, 3)
        - H: Integer Miller indices (P, 3, N)
        - s_best: Best symmetry operation indices (P,)
        - alpha: Fitted scale factors (P,)
    """
    R = R_init_batch.to(device=device, dtype=dtype)

    P, _, N = Q_batch_t.shape
    S = S_ops.shape[0]

    if indices is None:
        I = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(P, -1)
    else:
        I_raw = torch.as_tensor(indices, device=device, dtype=torch.long)
        if I_raw.dim() == 1:
            I = I_raw.unsqueeze(0).expand(P, -1)
        elif I_raw.dim() == 2:
            if I_raw.shape[0] != P:
                raise ValueError(f"indices has shape {I_raw.shape}, expected (P,M) or (M,)")
            I = I_raw
        else:
            raise ValueError(f"indices must have dim 1 or 2, got {I_raw.dim()}")

    M = int(I.shape[1])
    
    h = int(math.ceil(kappa * M))

    if delta > 0:
        offs = torch.arange(-delta, delta + 1, device=device, dtype=torch.int64)
        grid = torch.stack(torch.meshgrid(offs, offs, offs, indexing="ij"), dim=-1)
        D_t = grid.reshape(-1, 3).T.to(dtype)
    else:
        D_t = None
        PB_D = None
        dY2 = None

    if PB_D is not None:
        PB_D = PB_D.to(device=device, dtype=dtype)
    if dY2 is not None:
        dY2 = dY2.to(device=device, dtype=dtype)

    H_out = None
    s_best = torch.zeros(P, dtype=torch.long, device=device)
    alpha = torch.ones(P, device=device, dtype=dtype)

    for _ in range(max(1, int(steps))):
        A = torch.einsum("pij,pjn->pin", R, Q_batch_t)

        I_exp = I.unsqueeze(1).expand(-1, 3, -1)
        A_I = torch.gather(A, 2, I_exp)

        A_sym_I = torch.einsum("sij,pjn->spin", S_ops, A_I)

        A_sym_I_flat = A_sym_I.reshape(-1, 3, M)
        H_b_flat = babai_nearest_batched_torch(R_B_t, A_sym_I_flat)
        H_babai_I = H_b_flat.reshape(S, P, 3, M)

        Y0_I = torch.einsum("sij,spjn->spin", PB_ops, H_babai_I.to(dtype))

        E_I = A_sym_I - Y0_I
        E2_I = (E_I * E_I).sum(dim=2)

        if delta > 0 and (PB_D is not None) and (dY2 is not None):
            dot = torch.einsum("spim,sik->spkm", E_I, PB_D)
            r2_k = (
                E2_I[:, :, None, :]
                + dY2[:, None, :, None]
                - 2.0 * dot
            )
            r2_s_I, k_star_I = torch.min(r2_k, dim=2)
        else:
            r2_s_I = E2_I
            k_star_I = None

        part, _ = torch.topk(r2_s_I, k=h, dim=2, largest=False)

        if winsor_q is not None and 0.0 < winsor_q < 1.0 and h >= 3:
            t2_vec = torch.quantile(part, winsor_q, dim=2)
            clipped_all = torch.minimum(part, t2_vec.unsqueeze(-1))
        else:
            clipped_all = part

        loss_s = clipped_all.mean(dim=2)

        s_best = torch.argmin(loss_s, dim=0)

        r2_perm = r2_s_I.permute(1, 0, 2)
        r2_best = r2_perm[torch.arange(P, device=device), s_best, :]

        top_vals, idx_h_sub = torch.topk(r2_best, k=h, dim=1, largest=False)
        sel = top_vals.clone()

        if winsor_q is not None and 0.0 < winsor_q < 1.0 and h >= 3:
            t2 = torch.quantile(sel, winsor_q, dim=1)
            sel = torch.minimum(sel, t2.unsqueeze(-1))

        idx_h = torch.gather(I, 1, idx_h_sub)

        S_best = S_ops[s_best]
        A_sym_all = torch.einsum("pij,pjn->pin", S_best, A)

        H_all = babai_nearest_batched_torch(R_B_t, A_sym_all)
        H_all = H_all.clone()

        if delta > 0 and (k_star_I is not None) and (D_t is not None):
            k_star_perm = k_star_I.permute(1, 0, 2)
            ks = k_star_perm[torch.arange(P, device=device), s_best, :]

            K_total = int(D_t.shape[1])
            D_exp = D_t.unsqueeze(0).expand(P, 3, K_total)
            ks_exp = ks.unsqueeze(1).expand(-1, 3, -1)
            D_sub = torch.gather(D_exp, 2, ks_exp)

            H_all_f = H_all.to(dtype)

            idx_exp = I.unsqueeze(1).expand(-1, 3, -1)
            H_all_f.scatter_add_(2, idx_exp, D_sub)

            H_all = H_all_f.round().to(torch.int64)

        PB_best = PB_ops[s_best]
        Y_all = torch.einsum("pij,pjn->pin", PB_best, H_all.to(dtype))

        idx_h_exp = idx_h.unsqueeze(1).expand(-1, 3, -1)
        Q_in = torch.gather(Q_batch_t, 2, idx_h_exp)
        Y_in = torch.gather(Y_all,     2, idx_h_exp)

        if fit_alpha:
            Ry = torch.einsum("pij,pjn->pin", R, Y_in)

            num = (Q_in * Ry).sum(dim=(1, 2))
            den = (Y_in * Y_in).sum(dim=(1, 2)) + 1e-18
            alpha = num / den

            lo, hi = alpha_clip
            alpha = torch.clamp(alpha, lo, hi)
        else:
            alpha = torch.ones(P, device=device, dtype=dtype)

        alpha_exp = alpha.view(P, 1, 1)
        Y_in_scaled = alpha_exp * Y_in
        Sxy = torch.matmul(Q_in, Y_in_scaled.transpose(1, 2))

        U, S_vals, Vh = torch.linalg.svd(Sxy, full_matrices=False)
        M_procrustes = torch.matmul(U, Vh)
        detM = torch.linalg.det(M_procrustes)
        sign = torch.sign(detM)

        diag = torch.stack(
            [
                torch.ones_like(sign),
                torch.ones_like(sign),
                sign,
            ],
            dim=-1,
        )
        D_corr = torch.diag_embed(diag)
        R_update = torch.matmul(U, torch.matmul(D_corr, Vh))

        R = R_update
        H_out = H_all

    return R, H_out, s_best, alpha


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
    B_red, U = kz_reduce_integer_3d(B, Hmax=Hmax)
    Q_B, R_B = np.linalg.qr(B_red)
    s = np.sign(np.diag(R_B)); s[s == 0] = 1
    D = np.diag(s)
    Q_B  = Q_B @ D
    R_B  = D @ R_B
    Q_B_T = Q_B.T

    rot_mats = get_rots_mats_symm(unit_cell)
    order = np.argsort([float(np.trace(R)) for R in rot_mats])
    rot_mats = rot_mats[order]

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
# Main Optimization Function (GPU)
# ==============================================================================

def global_optimize_via_de_prepared(
    Q_batch, consts,
    obj="mse_symm_trimmed_auto",
    kappa=0.40,
    tol=3e-4,
    maxiter=1500,
    popsize=21,
    dtype=torch.float64,
    strategy="randtobest1bin",
    de_seed=None,
    smallN_tsn_polish=True,
    smallN_threshold=25,
    callback_every=25,
    tsn_delta=1,
    tsn_steps=1,
    tsn_winsor_q=None,
    midN_threshold=80,
    midN_tsn_topK=48,
    tsn_fit_alpha=False,
    callback_every_largeN=75,
):
    """GPU-accelerated global optimization of crystal orientation.
    
    Main indexing function using GPU-accelerated differential evolution to find
    optimal crystal orientations. Can process multiple problems simultaneously
    for maximum throughput.
    
    Args:
        Q_batch: Observed peak positions, shape (P, 3, N) or (3, N)
        consts: Dictionary from prepare_constants()
        obj: Objective function type:
            - "mse_symm_trimmed_auto": Trimmed mean (default, most robust)
            - "mse_symm": Standard MSE (faster, less robust)
            - "mse_small_n": TSN-enhanced for small N
        kappa: Trimming fraction (e.g., 0.40 = use best 40% of residuals)
        tol: Convergence tolerance for differential evolution
        maxiter: Maximum iterations for DE
        popsize: Population size multiplier for DE
        dtype: PyTorch data type (torch.float32 or torch.float64)
        strategy: DE mutation strategy
        de_seed: Random seed for differential evolution
        smallN_tsn_polish: Enable TSN polishing for small datasets
        smallN_threshold: Peak count threshold for "small" dataset
        callback_every: Callback frequency for small datasets
        tsn_delta: TSN search radius
        tsn_steps: Number of TSN refinement steps
        tsn_winsor_q: Quantile for Winsorization in TSN
        midN_threshold: Upper threshold for medium-sized datasets
        midN_tsn_topK: Number of hardest peaks to refine for medium N
        tsn_fit_alpha: Whether to fit scale in TSN
        callback_every_largeN: Callback frequency for large datasets
        
    Returns:
        For single problem (P=1):
            Tuple of (R_best, H_best, None, nit, message, success)
        For multiple problems (P>1):
            Tuple of (R_best_all, H_best_all, None, nit, message, success)
        where:
        - R_best: Optimal 3x3 rotation matrix (or (P,3,3) array)
        - H_best: Integer Miller indices (3, N) or (P, 3, N)
        - nit: Number of iterations performed
        - message: Convergence message
        - success: Boolean indicating successful convergence
    """
    device = "cuda"

    # Prepare observed peaks tensor
    Q_batch = np.ascontiguousarray(Q_batch, dtype=np.float64)
    if Q_batch.ndim == 2:
        Q_batch = Q_batch[None, ...]  # Add batch dimension if single problem
    if Q_batch.shape[1] != 3:
        raise ValueError(f"Q_batch must have shape (P,3,N) or (3,N), got {Q_batch.shape}")

    # Extract precomputed constants
    B_red, U, Q_B_T, R_B = consts["B_red"], consts["U"], consts["Q_B_T"], consts["R_B"]

    # Move data to GPU
    Q_batch_t = torch.as_tensor(Q_batch, device=device, dtype=dtype)
    P = int(Q_batch_t.shape[0])  # Number of problems
    N = int(Q_batch_t.shape[2])  # Number of peaks
    S_ops     = torch.as_tensor(consts["S_ops_stack"], device=device, dtype=dtype)
    PB_ops    = torch.as_tensor(consts["PB_ops_stack"], device=device, dtype=dtype)
    R_B_t     = torch.as_tensor(consts["R_B"], device=device, dtype=dtype)

    # Load precomputed target-space neighbor offsets if available
    PB_D_t = None
    dY2_t  = None
    if "PB_D" in consts and consts["PB_D"] is not None:
        PB_D_t = torch.as_tensor(consts["PB_D"], device=device, dtype=dtype)
    if "dY2" in consts and consts["dY2"] is not None:
        dY2_t  = torch.as_tensor(consts["dY2"], device=device, dtype=dtype)

    # Determine refinement strategy based on dataset size
    use_smallN = smallN_tsn_polish and (N <= smallN_threshold)
    use_midN   = (N > smallN_threshold) and (N <= midN_threshold)

    # Build appropriate loss function based on objective type
    if obj == "mse_symm_trimmed_auto":
        batched_loss_torch = build_loss_trimmed_multi_torch(
            Q_batch_t, R_B_t, S_ops, PB_ops,
            kappa=kappa, device=device, dtype=dtype
        )
    elif obj == "mse_symm":
        batched_loss_torch = build_loss_mse_multi_torch(
            Q_batch_t, R_B_t, S_ops, PB_ops,
            device=device, dtype=dtype
        )
    elif obj == "mse_small_n":
        if PB_D_t is None or dY2_t is None:
            raise ValueError("mse_small_n requires PB_D and dY2 in consts")
        batched_loss_torch = build_loss_small_n_multi_torch(
            Q_batch_t, R_B_t, S_ops, PB_ops,
            dY2_t, PB_D_t,
            kappa=kappa, device=device, dtype=dtype
        )
    else:
        raise ValueError(f"Unknown objective type: {obj}")

    init = "latinhypercube"

    # Early stopping parameters
    patience    = 600  # Stop if no improvement for this many generations
    target_loss = 1e-7  # Stop if loss drops below this threshold

    # Track best solution per problem
    best_tracker = {
        "best_val": np.full((P,), np.inf, dtype=np.float64),
        "last_improve_gen": np.zeros((P,), dtype=int),
        "done": np.zeros((P,), dtype=bool),
    }

    # Shadow variables to track best polished solutions
    best_q_shadow  = np.zeros((P, 4), dtype=np.float64)
    best_val_shadow = np.full((P,), np.inf, dtype=np.float64)

    gen_counter = {"g": 0}

    def de_callback(xk, convergence):
        """Callback for refinement and early stopping during DE iterations."""
        gen_counter["g"] += 1
        g = gen_counter["g"]

        # Determine callback frequency based on dataset size
        if use_smallN:
            if (callback_every is None) or (g % max(1, int(callback_every)) != 0):
                return False
        elif use_midN:
            if (callback_every_largeN is None) or (g % max(1, int(callback_every_largeN)) != 0):
                return False
        else:
            if (callback_every_largeN is None) or (g % max(1, int(callback_every_largeN)) != 0):
                return False

        # Prepare current best quaternions for polishing
        qk_np = np.asarray(xk, dtype=np.float64)

        if qk_np.ndim == 1:
            qk_np = qk_np.reshape(P, 4)

        qk_np /= (np.linalg.norm(qk_np, axis=1, keepdims=True) + 1e-12)

        qk_t = torch.from_numpy(qk_np).to(device=device, dtype=dtype)

        with torch.inference_mode():
            vals = batched_loss_torch(qk_t)
            Rk_t = quat_to_R_batch_torch(qk_t)

            # Collect candidates: original + polished versions
            candidates_q = [qk_np]
            candidates_v = [vals.detach().cpu().numpy()]

            # Apply polishing based on dataset size
            if use_smallN:
                # Small datasets: full TSN polishing on all peaks
                for fit_alpha_flag in (False, tsn_fit_alpha):
                    Rp_t, _, _, _ = polish_R_smallN_tsn_multi_torch(
                        Rk_t, Q_batch_t, S_ops, R_B_t, PB_ops, consts,
                        kappa=kappa,
                        PB_D=PB_D_t,
                        dY2=dY2_t,
                        delta=tsn_delta,
                        winsor_q=tsn_winsor_q,
                        fit_alpha=fit_alpha_flag,
                        steps=tsn_steps,
                        indices=None,
                        device=device,
                        dtype=dtype,
                    )
                    qp_t = R_batch_to_quat_batch_torch(Rp_t, device=device, dtype=dtype)
                    vp_t = batched_loss_torch(qp_t)
                    candidates_q.append(qp_t.detach().cpu().numpy())
                    candidates_v.append(vp_t.detach().cpu().numpy())

            elif use_midN:
                Ihard_t = hardest_peak_indices_multi_torch(
                    Rk_t,
                    Q_batch_t,
                    S_ops,
                    R_B_t,
                    PB_ops,
                    midN_tsn_topK,
                    device=device,
                    dtype=dtype,
                )

                for fit_alpha_flag in (False, tsn_fit_alpha):
                    Rp_t, _, _, _ = polish_R_smallN_tsn_multi_torch(
                        Rk_t, Q_batch_t, S_ops, R_B_t, PB_ops, consts,
                        kappa=kappa,
                        PB_D=PB_D_t,
                        dY2=dY2_t,
                        delta=tsn_delta,
                        winsor_q=None,
                        fit_alpha=fit_alpha_flag,
                        steps=2,
                        indices=Ihard_t,
                        device=device,
                        dtype=dtype,
                    )
                    qp_t = R_batch_to_quat_batch_torch(Rp_t, device=device, dtype=dtype)
                    vp_t = batched_loss_torch(qp_t)
                    candidates_q.append(qp_t.detach().cpu().numpy())
                    candidates_v.append(vp_t.detach().cpu().numpy())

            else:
                Rp_t, _, _, _ = polish_R_smallN_tsn_multi_torch(
                    Rk_t, Q_batch_t, S_ops, R_B_t, PB_ops, consts,
                    kappa=kappa,
                    PB_D=PB_D_t,
                    dY2=dY2_t,
                    delta=0,
                    winsor_q=None,
                    fit_alpha=tsn_fit_alpha,
                    steps=1,
                    indices=None,
                    device=device,
                    dtype=dtype,
                )
                qp_t = R_batch_to_quat_batch_torch(Rp_t, device=device, dtype=dtype)
                vp_t = batched_loss_torch(qp_t)
                candidates_q.append(qp_t.detach().cpu().numpy())
                candidates_v.append(vp_t.detach().cpu().numpy())

            # Select best candidate from all polished versions
            all_q = np.stack(candidates_q, axis=0)
            all_v = np.stack(candidates_v, axis=0)

            best_per_cand = all_v.argmin(axis=0)
            best_vals_gen = all_v[best_per_cand, np.arange(P)]
            best_q_gen    = all_q[best_per_cand, np.arange(P), :]

            # Update shadow best if improved
            if best_q_shadow is not None:
                improved = best_vals_gen < (best_val_shadow - 1e-9)
                best_val_shadow[improved] = best_vals_gen[improved]
                best_q_shadow[improved]   = best_q_gen[improved]

            bt = best_tracker

            not_done = ~bt["done"]

            # Update best tracker for improved problems
            improved = (best_vals_gen < (bt["best_val"] - 1e-9)) & not_done

            bt["best_val"][improved] = best_vals_gen[improved]
            bt["last_improve_gen"][improved] = g

            # Check early stopping criteria per problem
            if target_loss is not None:
                reached_target = (bt["best_val"] <= target_loss) & not_done
            else:
                reached_target = np.zeros_like(bt["done"], dtype=bool)

            stalled = (g - bt["last_improve_gen"] >= patience) & not_done

            newly_done = reached_target | stalled
            bt["done"][newly_done] = True

            # Stop if all problems converged
            if np.all(bt["done"]):
                print("DE callback: all problems finished; stopping early")
                return True

        return False

    # Run GPU-accelerated differential evolution
    result = differential_evolution_torch(
        func=batched_loss_torch,
        bounds=[(-1, 1)] * 4,
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        atol=0.0,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=de_seed,
        callback=de_callback,
        init=init,
        device=device,
        dtype=dtype,
        profile=False,
        n_problems=P,
    )

    # Extract best quaternions from DE result
    x_best = np.array(result.x, dtype=np.float64)
    if x_best.ndim == 1:
        x_best = x_best[None, :]

    q_best_de = x_best.copy()
    q_best_de /= (np.linalg.norm(q_best_de, axis=1, keepdims=True) + 1e-12)

    # For single problem, use shadow best if better
    if (P == 1) and (best_q_shadow is not None):
        q_shadow = np.asarray(best_q_shadow[0], dtype=np.float64)
        q_shadow /= (np.linalg.norm(q_shadow) + 1e-12)
        R_shadow = quat_to_R(q_shadow)

        if best_tracker["best_val"][0] < np.inf:
            q_best_de[0] = q_shadow

    # Convert quaternions to rotation matrices and compute Miller indices
    R_best_all = np.zeros((P, 3, 3), dtype=np.float64)
    H_best_all = np.zeros((P, 3, N), dtype=np.int64)

    for p in range(P):
        q = q_best_de[p]
        R_p = quat_to_R(q)
        R_best_all[p] = R_p

        # Final indexing using Babai rounding
        M_final_p = Q_B_T @ (R_p.T @ Q_batch[p])
        Hp_final_p = babai_nearest_batched(R_B, M_final_p[None, :, :])[0]
        H_best_all[p] = consts["U"] @ Hp_final_p

    if P == 1:
        return (
            R_best_all[0],
            H_best_all[0],
            None,
            result.nit,
            result.message,
            result.success,
        )
    else:
        return (
            R_best_all,
            H_best_all,
            None,
            result.nit,
            result.message,
            result.success,
        )
