"""Sensitivity analysis for ranking stability under weight perturbations.

This module repeatedly applies File 32's ``apply_weights`` to small random
perturbations of the current global weights and measures how often each site
keeps its baseline rank (or remains in a configurable top-N).
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

try:
    from scoring.score import apply_weights, rank_sites
except ImportError:  # running as a loose script alongside File 32
    try:
        from score import apply_weights, rank_sites
    except ImportError:
        from File_32 import apply_weights, rank_sites


def sample_weight_perturbations(
    current_weights,
    n_samples: int = 200,
    perturbation_size: float = 0.1,
):
    """Generate randomly perturbed, normalized weight vectors.

    Each criterion weight is multiplied by a random factor uniformly sampled
    from ``[1 - perturbation_size, 1 + perturbation_size]`` and the resulting
    vector is renormalized to sum to 1.0.

    Parameters
    ----------
    current_weights : array-like or mapping
        Current criterion weights. Mapping values are perturbed in insertion
        order.
    n_samples : int, default=200
        Number of perturbed vectors to generate.
    perturbation_size : float, default=0.1
        Maximum relative perturbation per criterion. Must be in [0, 1].

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, n_criteria)``. Every row sums to 1.0.

    Raises
    ------
    ValueError
        If weights are invalid, have non-positive total weight, or the
        requested sample count / perturbation size is invalid.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")
    if not 0 <= perturbation_size <= 1:
        raise ValueError("perturbation_size must be between 0 and 1.")

    if isinstance(current_weights, Mapping):
        weights = np.asarray(list(current_weights.values()), dtype=float)
    else:
        weights = np.asarray(current_weights, dtype=float).reshape(-1)

    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("current_weights must contain at least one weight.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("current_weights must contain only finite values.")
    if np.any(weights < 0):
        raise ValueError("current_weights must be non-negative.")
    total = weights.sum()
    if total <= 0:
        raise ValueError("current_weights must have a positive total.")

    weights = weights / total

    if n_samples == 0:
        return np.empty((0, weights.size), dtype=float)

    factors = np.random.uniform(
        1.0 - perturbation_size,
        1.0 + perturbation_size,
        size=(n_samples, weights.size),
    )
    samples = weights[None, :] * factors
    row_totals = samples.sum(axis=1, keepdims=True)
    samples /= row_totals
    return samples


def rank_stability_report(
    normalized_matrix,
    current_weights,
    n_samples: int = 200,
    top_n: Optional[int] = None,
):
    """Report how stable each site's baseline rank is under weight perturbations.

    File 32's ``apply_weights`` is used for the baseline and for every
    perturbed weight vector. Rankings use File 32's ``rank_sites`` semantics:
    higher score is better and ties are resolved stably in input order.

    Parameters
    ----------
    normalized_matrix : ndarray, DataFrame, or File 31 matrix artifact
        Already-normalized decision matrix accepted by ``apply_weights``.
    current_weights : array-like or mapping
        Current global criterion weights.
    n_samples : int, default=200
        Number of perturbed weight vectors to evaluate.
    top_n : int or None, default=None
        If provided, report stability as the percentage of samples in which
        each site remains in the baseline top-N. If None, report the
        percentage of samples in which each site keeps its exact baseline
        rank.

    Returns
    -------
    pandas.DataFrame
        One row per site, containing ``site_id``, ``current_rank``,
        ``stability_percentage``, and ``stability_criterion``. When ``top_n``
        is supplied, ``top_n`` is also included.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be a positive integer or None.")
    if top_n is not None and int(top_n) != top_n:
        raise ValueError("top_n must be a positive integer or None.")
    top_n = int(top_n) if top_n is not None else None

    # File 32 accepts mappings aligned to DataFrame columns, but perturbation
    # sampling works on values and therefore preserves mapping insertion order.
    baseline_scores = apply_weights(normalized_matrix, current_weights)
    baseline_ranked = rank_sites(baseline_scores)

    if top_n is not None:
        top_n = min(top_n, len(baseline_ranked))
        baseline_target = set(baseline_ranked.iloc[:top_n]["site_id"])
        criterion = f"top_{top_n}"
    else:
        baseline_target = {
            row.site_id: int(row["rank"])
            for _, row in baseline_ranked.iterrows()
        }
        criterion = "exact_rank"

    weights = (
        list(current_weights.values())
        if isinstance(current_weights, Mapping)
        else current_weights
    )
    perturbations = sample_weight_perturbations(weights, n_samples=n_samples)

    hit_counts = pd.Series(0, index=baseline_ranked["site_id"], dtype=int)

    for perturbed_weights in perturbations:
        scores = apply_weights(normalized_matrix, perturbed_weights)
        ranked = rank_sites(scores)

        if top_n is not None:
            perturbed_top = set(ranked.iloc[:top_n]["site_id"])
            for site_id in baseline_target:
                if site_id in perturbed_top:
                    hit_counts.loc[site_id] += 1
        else:
            perturbed_ranks = dict(
                zip(ranked["site_id"], ranked["rank"])
            )
            for site_id, baseline_rank in baseline_target.items():
                if perturbed_ranks[site_id] == baseline_rank:
                    hit_counts.loc[site_id] += 1

    denominator = n_samples
    percentages = (
        hit_counts.to_numpy(dtype=float) / denominator * 100.0
        if denominator
        else np.zeros(len(hit_counts), dtype=float)
    )

    report = baseline_ranked[["site_id", "rank"]].rename(
        columns={"rank": "current_rank"}
    ).copy()
    report["stability_percentage"] = percentages
    report["stability_criterion"] = criterion

    if top_n is not None:
        report["top_n"] = top_n

    return report.reset_index(drop=True)


__all__ = [
    "sample_weight_perturbations",
    "rank_stability_report",
]
