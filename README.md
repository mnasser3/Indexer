# Crystal Lattice Indexer

A crystallographic indexing algorithm for determining crystal orientation from diffraction peaks.

## Files

### `indexer_cpu.py`

CPU-based implementation of the crystal lattice indexer. This is the documented reference implementation with comprehensive docstrings explaining the algorithm. It uses:

- NumPy for numerical operations
- Numba JIT compilation for performance-critical sections
- SciPy's differential evolution optimizer
- Multi-threaded optimization support
- A command-line interface for single-pattern indexing with retry schedules over objective, `kappa`, and DE `strategy`

### `indexer_gpu.py`

GPU-accelerated implementation using PyTorch. This version provides significant speedup for large-scale optimization by:

- Running differential evolution entirely on GPU
- Batch processing multiple orientations in parallel
- Supporting multi-problem optimization across multiple diffraction patterns
- Custom CUDA-optimized operations for rotation and lattice rounding

## Usage

## CPU command-line usage

`indexer_cpu.py` can be called directly from the terminal or from a notebook.

### Single-run example (Jupyter / interactive)

```python
for i in range(300,301):
    q_path = f"./Final_Qs/Q_chunk_{i:06d}.npy"
    Q = np.load(q_path)
    if Q.shape[0] >= 25:
        objec = "mse_symm_trimmed_auto"
    !python -u indexer_cpu.py \
    --q-path "$q_path" \
    --obj $objec $objec $objec $objec mse_symm mse_symm \
    --unit-cell 105.8 105.8 75.5 90.0 90.0 120.0 \
    --n-tries 6 \
    --kappas 0.93 0.87 0.24 0.6 0.75 0.42 \
    --strategies best1bin best1bin best1bin randtobest1bin best1bin randtobest1bin
```

Notes:

- `--obj` is passed once per try and must match `--n-tries`
- In the example above:
  - the first four tries use `mse_symm_trimmed_auto`
  - the last two tries use `mse_symm`

### Required arguments

- `--q-path`: path to a `.npy` file containing `Q` with shape `(N, 3)`
- `--unit-cell`: six unit-cell parameters: `a b c alpha beta gamma`

### Retry behavior

The CPU CLI runs multiple attempts in sequence.

For try `t`, it uses:

- `obj = objs[t]`
- `kappa = kappas[t]`
- `strategy = strategies[t]`

Acceptance is determined by the CrystFEL-style lattice check:

- compute `U = (B_inv @ (R.T @ Q.T)).T`
- mark a feature as sane if all three components are within `delta = 0.25` of the nearest integer
- accept the solution if at least 50% of features are sane

The run stops at the first accepted attempt. If no attempt is accepted, the last attempt is returned.

### Outputs

If `--save-prefix` is provided, the script saves:

- `<prefix>_U.npy`: estimated rotation matrix `R`
- `<prefix>_H.npy`: estimated Miller indices `H`
- `<prefix>_attempts.txt`: summary of all tries and acceptance statistics

The terminal output also reports, for each try:

- objective
- strategy
- kappa
- acceptance status
- sane-feature fraction
- optimizer loss
- number of iterations
- DE convergence flag

## Parallel batch execution

Run multiple chunks in parallel using `ProcessPoolExecutor`.

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess, os, re
from pathlib import Path
import numpy as np

N = 100 #number of images to index
N_WORKERS = 128 #number of cpu cores to run in parallel

LOGDIR = Path("indexer_logs")
LOGDIR.mkdir(exist_ok=True)

def run_one(i):
    q_path = f"./Final_Qs/Q_chunk_{i:06d}.npy"
    log_path = LOGDIR / f"chunk_{i:06d}.log"
    Q = np.load(q_path)
    if Q.shape[0] >= 25:
        objec = "mse_symm_trimmed_auto"
    else:
        objec = "mse_small_n"

    cmd = [
        "python", "-u", "indexer_cpu.py",
        "--q-path", q_path,
        "--obj", objec, objec, objec, objec, "mse_symm", "mse_symm",
        "--unit-cell", "105.8", "105.8", "75.5", "90.0", "90.0", "120.0",
        "--n-tries", "6",
        "--kappas", "0.93", "0.87", "0.24", "0.6", "0.75", "0.42",
        "--strategies", "best1bin", "best1bin", "best1bin",
        "randtobest1bin", "best1bin", "randtobest1bin",
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    with open(log_path, "w") as f:
        p = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    text = log_path.read_text(errors="replace")

    accepted = "*HEURISTIC* accept on try" in text
    failed = not accepted

    return i, p.returncode, accepted, failed, log_path


accepted_count = 0
failed_count = 0

with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = [ex.submit(run_one, i) for i in range(6,6+N)]

    for fut in as_completed(futures):
        i, code, accepted, failed, log_path = fut.result()

        if accepted:
            accepted_count += 1
            print(f"[ACCEPTED] chunk {i:06d} | total accepted={accepted_count} failed={failed_count}")
        else:
            failed_count += 1
            print(f"[FAILED]   chunk {i:06d} | total accepted={accepted_count} failed={failed_count}")
            print(f"          log: {log_path}")

print()
print(f"Final accepted: {accepted_count}")
print(f"Final failed:   {failed_count}")
```

Notes:

- Each subprocess runs one indexing job
- Logs are written to `indexer_logs/chunk_XXXXXX.log`
- Threading is disabled inside each process to avoid oversubscription:
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`

## Python API usage

The core API is still available from Python.

### 1. Prepare constants from lattice parameters

```python
consts = prepare_constants(B, unit_cell, Hmax=6, delta_tsn=1)
```

### 2. Run optimization with observed peaks

```python
R_best, H_best, loss, nit, msg, success = global_optimize_via_de_prepared(
    Q,
    consts,
    obj="mse_symm_trimmed_auto",
    kappa=0.40,
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

1. Lattice reduction via Korkine-Zolotarev algorithm for numerical stability
2. Symmetry detection using GEMMI
3. Global optimization with differential evolution in quaternion space
4. Robust fitting using a trimmed objective to handle outliers
5. Refinement with target-space neighbor search for improved accuracy

## Dependencies

- `numpy`: array operations
- `scipy`: optimization and statistics
- `gemmi`: crystal symmetry operations
- `numba`: JIT compilation for the CPU version
- `torch`: GPU acceleration for the GPU version
