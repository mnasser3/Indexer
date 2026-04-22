# Crystal Lattice Indexer

A crystallographic indexing algorithm for determining crystal orientation from diffraction peaks.

## Files

### `indexer_cpu.py`
CPU-based implementation of the crystal lattice indexer. This is the documented reference implementation with comprehensive docstrings explaining the algorithm. It uses:
- NumPy for numerical operations
- Numba JIT compilation for performance-critical sections
- SciPy's differential evolution optimizer
- Multi-threaded optimization support
- A command-line interface for single-pattern indexing with retry schedules over `kappa` and DE `strategy`

### `indexer_gpu.py`
GPU-accelerated implementation using PyTorch. This version provides significant speedup for large-scale optimization by:
- Running differential evolution entirely on GPU
- Batch processing multiple orientations in parallel
- Supporting multi-problem optimization (multiple diffraction patterns simultaneously)
- Custom CUDA-optimized operations for rotation and lattice rounding

## Usage

## CPU command-line usage

`indexer_cpu.py` can now be called directly from the terminal.

Example:

```bash
python indexer_cpu.py \
  --q-path /global/cfs/cdirs/lcls/mnasser/cxidb_62/Final_Qs/Q_chunk_000001.npy \
  --unit-cell 105.8 105.8 75.5 90.0 90.0 120.0 \
  --n-tries 6 \
  --kappas 0.93 0.90 0.3 0.6 0.75 0.60 \
  --strategies best1bin best1bin best1bin randtobest1bin best1bin randtobest1bin \
  --save-prefix ../DE_preds_noisy_200/chunk_000123
```

### Required arguments

- `--q-path`: path to a `.npy` file containing `Q` with shape `(N, 3)`
- `--unit-cell`: six unit-cell parameters  
  `a b c alpha beta gamma`

### Retry behavior

The CPU CLI runs multiple attempts in sequence.

For try `t`, it uses:
- `kappa = kappas[t]`
- `strategy = strategies[t]`

Acceptance is determined by the CrystFEL-style lattice check:
- compute `U = (B_inv @ (R.T @ Q.T)).T`
- mark a feature as sane if all three components are within `delta` of the nearest integer
- accept the solution if at least 50% of features are sane

The run stops at the first accepted attempt. If no attempt is accepted, the last attempt is returned.

### Outputs

If `--save-prefix` is provided, the script saves:

- `<prefix>_U.npy`: estimated rotation matrix `R`
- `<prefix>_H.npy`: estimated Miller indices `H`
- `<prefix>_attempts.txt`: summary of all tries and acceptance statistics

The terminal output also reports, for each try:
- strategy
- kappa
- acceptance status
- sane-feature fraction
- optimizer loss
- number of iterations
- DE convergence flag

## Python API usage

The core API is still available from Python.

### 1. Prepare constants from lattice parameters

```python
consts = prepare_constants(B, unit_cell, Hmax=6, delta_tsn=1)
```

### 2. Run optimization with observed peaks

```python
R_best, H_best, loss, nit, msg, success = global_optimize_via_de_prepared(
    Q, consts, obj="mse_symm_trimmed_auto", kappa=0.40
)
```

Where:
- `B`: `3 x 3` reciprocal lattice basis matrix
- `unit_cell`: `(a, b, c, alpha, beta, gamma)`
- `Q`: `3 x N` array of observed peak positions
- `R_best`: optimal `3 x 3` rotation matrix
- `H_best`: `3 x N` integer Miller indices

## Algorithm overview

The indexer uses differential evolution to search for the crystal orientation that best explains observed diffraction peaks. Key steps:

1. **Lattice reduction** via Korkine-Zolotarev algorithm for numerical stability
2. **Symmetry detection** using GEMMI
3. **Global optimization** with differential evolution in quaternion space
4. **Robust fitting** using a trimmed objective to handle outliers
5. **Refinement** with target-space neighbor search (TSN) for improved accuracy

## Dependencies

- `numpy`: array operations
- `scipy`: optimization and statistics
- `gemmi`: crystal symmetry operations
- `numba`: JIT compilation for the CPU version
- `torch`: GPU acceleration for the GPU version