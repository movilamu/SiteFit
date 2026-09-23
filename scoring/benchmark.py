"""
scoring/benchmark.py

Performance & Latency Benchmark for Interactive Site Scoring (File 32).
=======================================================================

Verifies that `apply_weights()` from File 32 runs fast enough to deliver a
"live" interactive experience on slider adjustment (< 100ms per call).

Requirements:
1. Loads precomputed decision matrix once (File 31 / scoring/precompute_matrix.py output).
2. Calls apply_weights() 100 times with randomly varied weight vectors.
3. Computes and prints:
   - Total time for all 100 calls
   - Average time per call
   - Min time per call
   - Max time per call
4. Evaluates Pass/Fail against the 100ms threshold:
   - PASS: average latency < 100ms (UI feels live and continuous)
   - FAIL: average latency >= 100ms (warning: feels like a form submission)

Usage:
    python scoring/benchmark.py
    python scoring/benchmark.py --matrix path/to/precomputed_matrix.parquet
    python scoring/benchmark.py --runs 100 --method weighted_sum --threshold 100.0
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Import apply_weights from File 32 (with flexible module fallbacks)
# ----------------------------------------------------------------------------
try:
    from scoring.score import apply_weights
except ImportError:
    try:
        from score import apply_weights
    except ImportError:
        try:
            from File_32 import apply_weights
        except ImportError:
            apply_weights = None


# ----------------------------------------------------------------------------
# Precomputed Matrix Loader (File 31 integration)
# ----------------------------------------------------------------------------
CANDIDATE_DATA_PATHS = [
    Path("data/precomputed_matrix.parquet"),
    Path("scoring/precomputed_matrix.parquet"),
    Path("precomputed_matrix.parquet"),
    Path("data/normalized_matrix.parquet"),
    Path("data/precomputed_matrix.pkl"),
    Path("data/precomputed_matrix.npz"),
]


def load_precomputed_matrix(
    source: Optional[Union[str, Path]] = None,
    synthetic_sites: int = 5000,
    synthetic_criteria: int = 18,
    seed: int = 42,
) -> Tuple[Any, str]:
    """
    Load a precomputed normalized matrix once.

    Tries the following in priority order:
    1. Direct file path if supplied.
    2. File 31 loader function (`load_precomputed_matrix` / `load_matrix` in File 31).
    3. Common stored artifact locations on disk (.parquet, .feather, .npz, .pkl).
    4. Deterministic synthetic matrix representative of pilot dimensions
       (e.g., 5,000 sites x 18 criteria) if no stored file is found yet.

    Returns
    -------
    (matrix_object, source_description)
    """
    # 1. Explicit source path
    if source is not None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Specified matrix file not found: {path}")
        return _read_matrix_file(path), f"File: {path}"

    # 2. Try importing File 31 module and calling its loader
    for mod_name in ("scoring.precompute_matrix", "precompute_matrix", "File_31"):
        try:
            mod = importlib.import_module(mod_name)
            for fn_name in ("load_precomputed_matrix", "load_matrix", "load_normalized_matrix"):
                if hasattr(mod, fn_name):
                    loader = getattr(mod, fn_name)
                    matrix = loader()
                    return matrix, f"{mod_name}.{fn_name}()"
            if hasattr(mod, "NormalizedMatrixArtifact") and hasattr(mod.NormalizedMatrixArtifact, "load"):
                matrix = mod.NormalizedMatrixArtifact.load()
                return matrix, f"{mod_name}.NormalizedMatrixArtifact.load()"
        except Exception:
            continue

    # 3. Check standard artifact files on disk
    for candidate in CANDIDATE_DATA_PATHS:
        if candidate.exists():
            return _read_matrix_file(candidate), f"Stored artifact: {candidate}"

    # 4. Fallback: Generate realistic synthetic pilot matrix
    rng = np.random.default_rng(seed)
    columns = [f"criterion_{j:02d}" for j in range(synthetic_criteria)]
    index = [f"site_{i:05d}" for i in range(synthetic_sites)]
    # Normalized values in [0, 1]
    data = rng.uniform(0.0, 1.0, size=(synthetic_sites, synthetic_criteria))
    df = pd.DataFrame(data, index=index, columns=columns)
    return df, f"Synthetic Matrix ({synthetic_sites:,} sites × {synthetic_criteria} criteria)"


def _read_matrix_file(path: Path) -> Any:
    """Read a matrix from parquet, feather, csv, pickle, or numpy format."""
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        first_key = list(data.keys())[0]
        return data[first_key]
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported matrix file extension: {suffix}")


def _extract_matrix_info(matrix_obj: Any) -> Tuple[int, int, Sequence[str]]:
    """Extract (num_sites, num_criteria, column_names) without mutating object."""
    obj = matrix_obj
    if hasattr(obj, "matrix") and not isinstance(obj, (np.ndarray, pd.DataFrame)):
        obj = obj.matrix

    if isinstance(obj, pd.DataFrame):
        return obj.shape[0], obj.shape[1], list(obj.columns)

    arr = np.asarray(obj)
    if arr.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape {arr.shape}")
    cols = [f"c{j}" for j in range(arr.shape[1])]
    return arr.shape[0], arr.shape[1], cols


# ----------------------------------------------------------------------------
# Benchmark Engine
# ----------------------------------------------------------------------------
def generate_random_weights(
    n_runs: int,
    columns: Sequence[str],
    as_dict: bool = False,
    seed: int = 2026,
) -> List[Union[np.ndarray, Dict[str, float]]]:
    """
    Generate `n_runs` random weight vectors on the unit simplex (sum=1.0, w_i >= 0).
    Pre-generated so that vector generation time is excluded from benchmark timing.
    """
    rng = np.random.default_rng(seed)
    k = len(columns)
    # Dirichlet(1, ..., 1) generates uniform distributions over the simplex
    weight_samples = rng.dirichlet(np.ones(k), size=n_runs)

    vectors: List[Union[np.ndarray, Dict[str, float]]] = []
    for w in weight_samples:
        if as_dict:
            vectors.append({col: float(val) for col, val in zip(columns, w)})
        else:
            vectors.append(w)
    return vectors


def run_benchmark(
    matrix_source: Optional[Union[str, Path]] = None,
    n_runs: int = 100,
    method: str = "weighted_sum",
    threshold_ms: float = 100.0,
    weights_as_dict: bool = False,
    warmup_runs: int = 3,
    seed: int = 2026,
) -> Dict[str, Any]:
    """
    Execute the scoring benchmark.

    Parameters
    ----------
    matrix_source : str or Path, optional
        Path to precomputed matrix. If None, auto-resolves via File 31.
    n_runs : int
        Number of scoring calls to benchmark (default 100).
    method : str
        Aggregation method ('weighted_sum', 'weighted_product', 'distance_to_ideal').
    threshold_ms : float
        Maximum allowed average latency per call in ms (default 100.0 ms).
    weights_as_dict : bool
        Whether to pass weights as dict mapping criteria to weight.
    warmup_runs : int
        Cold-start cache warmup runs before timing starts.
    seed : int
        Seed for deterministic random weight vector generation.

    Returns
    -------
    dict with metrics: avg_ms, min_ms, max_ms, total_ms, pass_fail, throughput.
    """
    if apply_weights is None:
        raise ImportError(
            "Could not import `apply_weights` from scoring.score / score / File_32. "
            "Ensure scoring/score.py is in your PYTHONPATH."
        )

    # 1. Load precomputed matrix once
    matrix_obj, source_desc = load_precomputed_matrix(source=matrix_source, seed=seed)
    n_sites, n_criteria, columns = _extract_matrix_info(matrix_obj)

    # 2. Pre-generate n_runs randomly varied weight vectors
    weight_vectors = generate_random_weights(
        n_runs=n_runs,
        columns=columns,
        as_dict=weights_as_dict,
        seed=seed,
    )

    # 3. Warm-up runs to remove initial caching / JIT / paging artifacts
    for i in range(min(warmup_runs, n_runs)):
        apply_weights(matrix_obj, weight_vectors[i], method=method)

    # 4. Timed runs (100 calls)
    durations_sec: List[float] = []
    for w in weight_vectors:
        t0 = time.perf_counter()
        _ = apply_weights(matrix_obj, w, method=method)
        t1 = time.perf_counter()
        durations_sec.append(t1 - t0)

    # 5. Compute statistics
    durations_ms = [d * 1000.0 for d in durations_sec]
    total_time_ms = sum(durations_ms)
    avg_time_ms = total_time_ms / len(durations_ms)
    min_time_ms = min(durations_ms)
    max_time_ms = max(durations_ms)
    median_time_ms = float(np.median(durations_ms))
    p95_time_ms = float(np.percentile(durations_ms, 95))
    throughput = len(durations_sec) / sum(durations_sec)
    passed = avg_time_ms < threshold_ms

    results = {
        "passed": passed,
        "n_runs": n_runs,
        "n_sites": n_sites,
        "n_criteria": n_criteria,
        "source": source_desc,
        "method": method,
        "threshold_ms": threshold_ms,
        "total_time_ms": total_time_ms,
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "max_time_ms": max_time_ms,
        "median_time_ms": median_time_ms,
        "p95_time_ms": p95_time_ms,
        "throughput_calls_sec": throughput,
    }
    return results


def print_benchmark_report(results: Dict[str, Any]) -> None:
    """Print formatted summary table and Pass/Fail diagnosis."""
    passed = results["passed"]
    avg = results["avg_time_ms"]
    threshold = results["threshold_ms"]

    div_heavy = "=" * 78
    div_light = "-" * 78

    print(f"\n{div_heavy}")
    print("        SCORING LATENCY BENCHMARK REPORT (File 32 / apply_weights)")
    print(div_heavy)
    print(f" Matrix Source        : {results['source']}")
    print(f" Matrix Dimensions    : {results['n_sites']:,} sites × {results['n_criteria']} criteria")
    print(f" Aggregation Method   : {results['method']}")
    print(f" Iterations Evaluated : {results['n_runs']} calls (randomly varied weight vectors)")
    print(div_light)
    print(f" Total Time ({results['n_runs']} calls) : {results['total_time_ms']:>10.2f} ms  ({results['total_time_ms'] / 1000.0:.4f} s)")
    print(f" Average Time / Call  : {results['avg_time_ms']:>10.3f} ms")
    print(f" Min Time / Call      : {results['min_time_ms']:>10.3f} ms")
    print(f" Max Time / Call      : {results['max_time_ms']:>10.3f} ms")
    print(f" Median (p50) / Call  : {results['median_time_ms']:>10.3f} ms")
    print(f" 95th Percentile (p95): {results['p95_time_ms']:>10.3f} ms")
    print(f" Throughput           : {results['throughput_calls_sec']:>10.1f} calls/sec")
    print(div_light)
    print(f" Latency Target       : < {threshold:.2f} ms per interaction")

    if passed:
        print(
            f" STATUS: [PASS] Average latency {avg:.2f} ms is well below the {threshold:.2f} ms budget.\n"
            f"         Site scoring is fast enough to run continuously on every slider change\n"
            f"         with a live, fluid UI response."
        )
    else:
        print(
            f" STATUS: [FAIL] Average latency {avg:.2f} ms exceeds the {threshold:.2f} ms limit!\n"
            f"         WARNING: Interactions will feel like a sluggish form submission rather\n"
            f"         than a live interactive slider. Investigate matrix size, memory copying,\n"
            f"         or vectorization in the aggregation function."
        )
    print(f"{div_heavy}\n")


# ----------------------------------------------------------------------------
# Command-Line Entry Point
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark apply_weights() performance against sub-100ms live UI threshold."
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default=None,
        help="Path to precomputed matrix file (.parquet, .feather, .npz, .pkl).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of apply_weights() invocations (default: 100).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="weighted_sum",
        choices=["weighted_sum", "weighted_product", "distance_to_ideal"],
        help="Aggregation method to benchmark (default: weighted_sum).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Live UI latency threshold in milliseconds (default: 100.0 ms).",
    )
    parser.add_argument(
        "--as-dict",
        action="store_true",
        help="Pass weights as {col_name: weight} dictionary instead of numpy array.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible weight vectors.",
    )
    parser.add_argument(
        "--no-exit-code",
        action="store_true",
        help="Do not return non-zero exit code if benchmark fails.",
    )

    args = parser.parse_args()

    results = run_benchmark(
        matrix_source=args.matrix,
        n_runs=args.runs,
        method=args.method,
        threshold_ms=args.threshold,
        weights_as_dict=args.as_dict,
        seed=args.seed,
    )

    print_benchmark_report(results)

    if not results["passed"] and not args.no_exit_code:
        sys.exit(1)


if __name__ == "__main__":
    main()