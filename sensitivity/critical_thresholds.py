"""
sensitivity/critical_thresholds.py

Rank-sensitivity analysis over a File 31 precomputed matrix.

Given the current global weights (as consumed by File 32's
`apply_weights`), this module finds, for each criterion, how far that
criterion's weight can move -- holding the *relative proportions* of all
other weights fixed -- before the top-ranked site changes.

This is intentionally cheap: it calls File 32's `apply_weights` (an
O(sites x criteria) aggregation over an already-normalized matrix)
repeatedly inside a coarse-scan + bisection search. It never re-runs
File 31's constraint filtering or normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    from scoring.score import apply_weights, rank_sites, _unwrap_precomputed_matrix
except ImportError:  # running as a loose script alongside File 32
    try:
        from score import apply_weights, rank_sites, _unwrap_precomputed_matrix
    except ImportError:
        from File_32 import apply_weights, rank_sites, _unwrap_precomputed_matrix


# Number of coarse grid points scanned (per direction) before bisecting
# down to a precise threshold. Kept small since each point is one
# apply_weights() call, which is itself cheap by File 32's design.
_COARSE_STEPS = 40
_BISECTION_TOL = 1e-6
_MAX_BISECTION_ITERS = 40


@dataclass
class ThresholdResult:
    """Result of a single-criterion threshold search."""

    criterion: str
    current_weight: float
    current_top_site: Any
    direction: Optional[str]  # "increase", "decrease", or None if no flip found
    threshold_weight: Optional[float]
    delta: Optional[float]  # signed change from current_weight to threshold_weight
    new_top_site: Optional[Any]
    method: str

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "current_weight": self.current_weight,
            "current_top_site": self.current_top_site,
            "direction": self.direction,
            "threshold_weight": self.threshold_weight,
            "delta": self.delta,
            "new_top_site": self.new_top_site,
            "method": self.method,
        }


def _top_site(normalized_matrix, weights, method: str):
    """Score with the given weights and return (top_site_id, top_score)."""
    scores = apply_weights(normalized_matrix, weights, method=method)
    ranked = rank_sites(scores)
    top_row = ranked.iloc[0]
    return top_row["site_id"], float(top_row["score"])


def _as_weight_array(current_weights, columns: Sequence[str]) -> np.ndarray:
    """Mirror File 32's weight alignment so callers can pass dict or array."""
    if isinstance(current_weights, Mapping):
        if all(col in current_weights for col in columns):
            return np.asarray([current_weights[col] for col in columns], dtype=float)
        return np.asarray(list(current_weights.values()), dtype=float)
    return np.asarray(current_weights, dtype=float).reshape(-1)


def _rescale_weights(weights: np.ndarray, idx: int, new_value: float) -> np.ndarray:
    """
    Return a new weight vector with weights[idx] set to `new_value`, and all
    other weights scaled proportionally so the vector's total mass is
    preserved (i.e. relative proportions among the *other* criteria are
    held fixed).
    """
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    other_sum = total - weights[idx]

    new_weights = weights.copy()
    remaining = total - new_value

    if other_sum <= 0:
        # No mass to redistribute among the others (e.g. a degenerate
        # single-criterion weighting). Just set the target weight; leave
        # the rest at zero.
        new_weights[idx] = new_value
        return new_weights

    factor = remaining / other_sum
    new_weights = weights * factor
    new_weights[idx] = new_value
    return new_weights


def find_critical_threshold(
    normalized_matrix,
    current_weights,
    criterion_to_vary: Union[str, int],
    method: str = "weighted_sum",
    coarse_steps: int = _COARSE_STEPS,
) -> ThresholdResult:
    """
    Find how far `criterion_to_vary`'s weight can move -- holding other
    weights' relative proportions fixed -- before the top-ranked site
    changes.

    Parameters
    ----------
    normalized_matrix : ndarray, DataFrame, or NormalizedMatrixArtifact
        File 31 output, same as accepted by File 32's `apply_weights`.
    current_weights : array-like or mapping
        Current global leaf weights. Assumed non-negative; need not sum
        to 1 (the total mass is preserved during the search).
    criterion_to_vary : str or int
        Column name (if the matrix carries names) or positional index of
        the criterion whose weight is perturbed.
    method : str
        Aggregation method, passed straight through to `apply_weights`.
    coarse_steps : int
        Number of grid points scanned in each direction (increase /
        decrease) before bisecting. Higher values catch threshold
        crossings that a coarser grid might straddle and miss, at the
        cost of more `apply_weights` calls.

    Returns
    -------
    ThresholdResult
        The nearest weight value (in either direction) at which the top
        rank flips, which site takes over, and the direction of travel.
        If no flip is found within the criterion's feasible range
        ([0, total_weight_mass]), `direction`/`threshold_weight`/
        `new_top_site` are None.
    """
    _, index, columns = _unwrap_precomputed_matrix(normalized_matrix)
    weights = _as_weight_array(current_weights, columns)

    if isinstance(criterion_to_vary, str):
        if criterion_to_vary not in columns:
            raise ValueError(f"Unknown criterion {criterion_to_vary!r}. Choices: {columns}")
        idx = columns.index(criterion_to_vary)
        crit_name = criterion_to_vary
    else:
        idx = int(criterion_to_vary)
        crit_name = columns[idx] if idx < len(columns) else f"c{idx}"

    total_mass = float(weights.sum())
    current_value = float(weights[idx])

    baseline_top, _ = _top_site(normalized_matrix, weights, method)

    best: Optional[ThresholdResult] = None

    for direction, bound in (("increase", total_mass), ("decrease", 0.0)):
        candidate = _search_direction(
            normalized_matrix=normalized_matrix,
            weights=weights,
            idx=idx,
            crit_name=crit_name,
            current_value=current_value,
            baseline_top=baseline_top,
            bound=bound,
            direction=direction,
            method=method,
            coarse_steps=coarse_steps,
        )
        if candidate is None:
            continue
        threshold_weight, new_top = candidate
        delta = threshold_weight - current_value
        result = ThresholdResult(
            criterion=crit_name,
            current_weight=current_value,
            current_top_site=baseline_top,
            direction=direction,
            threshold_weight=threshold_weight,
            delta=delta,
            new_top_site=new_top,
            method=method,
        )
        if best is None or abs(result.delta) < abs(best.delta):
            best = result

    if best is not None:
        return best

    # No flip found in either direction within the feasible range.
    return ThresholdResult(
        criterion=crit_name,
        current_weight=current_value,
        current_top_site=baseline_top,
        direction=None,
        threshold_weight=None,
        delta=None,
        new_top_site=None,
        method=method,
    )


def _search_direction(
    normalized_matrix,
    weights: np.ndarray,
    idx: int,
    crit_name: str,
    current_value: float,
    baseline_top,
    bound: float,
    direction: str,
    method: str,
    coarse_steps: int,
):
    """
    Coarse-scan from current_value toward `bound`, find the first grid
    interval where the top site changes, then bisect within that
    interval for a precise threshold. Returns (threshold_value, new_top)
    or None if no flip is found before `bound`.
    """
    if np.isclose(current_value, bound):
        return None

    grid = np.linspace(current_value, bound, coarse_steps + 1)[1:]  # skip current point

    prev_value = current_value
    prev_top = baseline_top

    for value in grid:
        trial_weights = _rescale_weights(weights, idx, value)
        top_site, _ = _top_site(normalized_matrix, trial_weights, method)

        if top_site != prev_top:
            # Flip happened somewhere in (prev_value, value]; bisect.
            lo, hi = prev_value, value
            lo_top = prev_top
            hi_top = top_site
            for _ in range(_MAX_BISECTION_ITERS):
                if abs(hi - lo) < _BISECTION_TOL:
                    break
                mid = (lo + hi) / 2.0
                mid_weights = _rescale_weights(weights, idx, mid)
                mid_top, _ = _top_site(normalized_matrix, mid_weights, method)
                if mid_top == lo_top:
                    lo = mid
                else:
                    hi = mid
                    hi_top = mid_top
            return hi, hi_top

        prev_value = value
        prev_top = top_site

    return None


def compute_all_thresholds(
    normalized_matrix,
    current_weights,
    method: str = "weighted_sum",
    coarse_steps: int = _COARSE_STEPS,
) -> pd.DataFrame:
    """
    Run `find_critical_threshold` for every criterion in the matrix.

    Parameters
    ----------
    normalized_matrix : ndarray, DataFrame, or NormalizedMatrixArtifact
        Same as `find_critical_threshold`.
    current_weights : array-like or mapping
        Same as `find_critical_threshold`.
    method : str
        Aggregation method, applied uniformly across all criteria.
    coarse_steps : int
        Passed through to each per-criterion search.

    Returns
    -------
    pandas.DataFrame
        One row per criterion, columns:
        ``criterion``, ``current_weight``, ``current_top_site``,
        ``direction``, ``threshold_weight``, ``delta``, ``new_top_site``.
        Rows are sorted by ``abs(delta)`` ascending (most sensitive
        criteria -- i.e. smallest perturbation needed to flip the
        winner -- first). Criteria with no flip found are sorted last.
    """
    _, _, columns = _unwrap_precomputed_matrix(normalized_matrix)

    rows = []
    for col in columns:
        result = find_critical_threshold(
            normalized_matrix,
            current_weights,
            criterion_to_vary=col,
            method=method,
            coarse_steps=coarse_steps,
        )
        rows.append(result.to_dict())

    report = pd.DataFrame(rows)
    report["_abs_delta"] = report["delta"].abs()
    report = report.sort_values(
        by="_abs_delta", ascending=True, na_position="last", kind="mergesort"
    ).drop(columns="_abs_delta")
    return report.reset_index(drop=True)


__all__ = ["find_critical_threshold", "compute_all_thresholds", "ThresholdResult"]
