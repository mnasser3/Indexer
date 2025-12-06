# Crystal Lattice Indexer

A crystallographic indexing algorithm for determining crystal orientation from diffraction peaks.

## Files

### `indexer_cpu.py`
CPU-based implementation of the crystal lattice indexer. This is the fully documented reference implementation with comprehensive docstrings explaining the algorithm. It uses:
- NumPy for numerical operations
- Numba JIT compilation for performance-critical sections
- SciPy's differential evolution optimizer
- Multi-threaded optimization support

### `indexer_gpu.py`
GPU-accelerated implementation using PyTorch. This version provides significant speedup for large-scale optimization by:
- Running differential evolution entirely on GPU
- Batch processing multiple orientations in parallel
- Supporting multi-problem optimization (multiple diffraction patterns simultaneously)
- Custom CUDA-optimized operations for rotation and lattice rounding

## Usage

Both implementations share the same core API:

1. **Prepare constants** from lattice parameters:
```python
consts = prepare_constants(B, unit_cell, Hmax=6, delta_tsn=1)
```

2. **Run optimization** with observed peaks:
```python
R_best, H_best, loss, nit, msg, success = global_optimize_via_de_prepared(
    Q, consts, obj="mse_symm_trimmed_auto", kappa=0.40
)
```

Where:
- `B`: 3×3 reciprocal lattice basis matrix
- `unit_cell`: (a, b, c, α, β, γ) parameters
- `Q`: 3×N array of observed peak positions
- `R_best`: Optimal 3×3 rotation matrix
- `H_best`: 3×N integer Miller indices

## Algorithm Overview

The indexer uses differential evolution to search for the crystal orientation that best explains observed diffraction peaks. Key steps:

1. **Lattice reduction** via Korkine-Zolotarev algorithm for numerical stability
2. **Symmetry detection** using GEMMI library
3. **Global optimization** with differential evolution in quaternion space
4. **Robust fitting** using trimmed mean to handle outliers
5. **Refinement** with target-space neighbor search (TSN) for improved accuracy

## Dependencies

- `numpy`: Array operations
- `scipy`: Optimization and statistics
- `gemmi`: Crystal symmetry operations
- `numba`: JIT compilation (CPU version)
- `torch`: GPU acceleration (GPU version)
