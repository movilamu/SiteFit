"""
scoring/score.py

CHEAP, per-interaction aggregation over a File 31 precomputed matrix.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


try:
    from scoring.aggregate_weighted_sum import weighted_sum_score
    from scoring.aggregate_weighted_product import weighted_product_score
    from scoring.aggregate_distance_to_ideal import distance_to_ideal_score
except ImportError:
    try:
        from aggregate_weighted_sum import weighted_sum_score
        from aggregate_weighted_product import weighted_product_score
        from aggregate_distance_to_ideal import distance_to_ideal_score
    except ImportError:
        from File_27 import weighted_sum_score
        from File_28 import weighted_product_score
        from File_29 import distance_to_ideal_score


METHODS = {
    "weighted_sum": weighted_sum_score,
    "weighted_product": weighted_product_score,
    "distance_to_ideal": distance_to_ideal_score,
}


def apply_weights(
    normalized_matrix,
    global_weights,
    method: str = "weighted_sum",
):
    matrix, index, columns = _unwrap_precomputed_matrix(normalized_matrix)
    weights = _align_weights(global_weights, columns)

    key = method.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in METHODS:
        raise ValueError(
            f"Unknown aggregation method {method!r}. "
            f"Choose one of: {sorted(METHODS)}."
        )

    # Return empty series cleanly if candidate site inventory contains 0 sites
    if matrix.shape[0] == 0:
        return pd.Series([], index=index, dtype=float, name="score")

    scores = METHODS[key](matrix, weights)
    return pd.Series(np.asarray(scores, dtype=float), index=index, name="score")


def rank_sites(scores):
    series = _as_score_series(scores)
    ordered = series.sort_values(ascending=False, kind="mergesort")
    ranked = pd.DataFrame(
        {
            "rank": np.arange(1, len(ordered) + 1, dtype=int),
            "site_id": ordered.index,
            "score": ordered.to_numpy(dtype=float),
        }
    )
    return ranked.reset_index(drop=True)


def _unwrap_precomputed_matrix(normalized_matrix):
    """
    Extract (ndarray, index, columns) from File 31 output.
    Guarantees thread-safety (read-only array copy) and missing-value handling.
    """
    obj = normalized_matrix
    if hasattr(obj, "matrix") and not isinstance(obj, (np.ndarray, pd.DataFrame)):
        obj = obj.matrix

    if isinstance(obj, pd.DataFrame):
        matrix = obj.to_numpy(dtype=float, copy=True)
        index = obj.index
        columns = list(obj.columns)
    else:
        matrix = np.array(obj, dtype=float, copy=True)
        if matrix.ndim != 2:
            raise ValueError("normalized_matrix must be a 2-D matrix.")
        index = pd.RangeIndex(matrix.shape[0], name="site_id")
        columns = [f"c{j}" for j in range(matrix.shape[1])]

    # Fallback for missing/null criteria values:
    # Impute missing/NaN values column-by-column using the criterion's minimum
    # observed finite value (or 0.0 if all values for that criterion are NaN).
    if matrix.size > 0 and np.isnan(matrix).any():
        for col_idx in range(matrix.shape[1]):
            col = matrix[:, col_idx]
            nans = np.isnan(col)
            if np.any(nans):
                valid = col[~nans]
                fallback = float(np.min(valid)) if valid.size > 0 else 0.0
                col[nans] = fallback

    # Enforce read-only immutability to prevent race conditions during concurrent requests
    matrix.flags.writeable = False

    return matrix, index, columns


def _align_weights(global_weights, columns: Sequence[str]):
    if isinstance(global_weights, Mapping):
        if all(col in global_weights for col in columns):
            return np.asarray([global_weights[col] for col in columns], dtype=float)
        return np.asarray(list(global_weights.values()), dtype=float)
    return np.asarray(global_weights, dtype=float).reshape(-1)


def _as_score_series(scores) -> pd.Series:
    if isinstance(scores, pd.Series):
        return scores.astype(float)
    if isinstance(scores, pd.DataFrame):
        if "score" in scores.columns:
            idx = scores["site_id"] if "site_id" in scores.columns else scores.index
            return pd.Series(scores["score"].to_numpy(dtype=float), index=idx, name="score")
        if scores.shape[1] != 1:
            raise ValueError("scores DataFrame must have a 'score' column or a single column.")
        return scores.iloc[:, 0].astype(float)
    if isinstance(scores, Mapping):
        return pd.Series(scores, dtype=float, name="score")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    return pd.Series(arr, index=pd.RangeIndex(arr.size, name="site_id"), name="score")


__all__ = ["apply_weights", "rank_sites", "METHODS"]