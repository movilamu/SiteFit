# sensitivity/near_ties.py
"""
Identify groups of sites that should be treated as effectively tied rather
than strictly ordered.

This module is intended to sit downstream of File 32's ``rank_sites`` and
File 39's perturbation-based stability analysis.

A near-tie occurs when adjacent site scores differ by less than a chosen
uncertainty threshold. Instead of forcing an arbitrary ordering, tied sites
are returned as rank clusters for UI display.

Example
-------
>>> ranked = rank_sites(scores)
>>> identify_near_ties(ranked)
[
    {
        "cluster_rank": 1,
        "site_ids": ["A", "B"],
        "scores": [0.812, 0.809],
        "score_range": 0.003
    },
    {
        "cluster_rank": 3,
        "site_ids": ["C"],
        "scores": [0.781],
        "score_range": 0.0
    }
]
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def identify_near_ties(
    ranked_scores,
    uncertainty_threshold: Optional[float] = None,
):
    """
    Group sites whose scores differ by less than an uncertainty threshold.

    Parameters
    ----------
    ranked_scores : pandas.DataFrame
        Output from File 32's ``rank_sites()``.
        Must contain:
            - rank
            - site_id
            - score

        Rows should already be sorted best-first.

    uncertainty_threshold : float or None, default=None
        Maximum score difference for two sites to be treated as
        effectively tied.

        If None, a conservative default is used:

            threshold = 0.5% of score range

        This acts as a fallback when File 39 perturbation statistics are
        unavailable.

        Better practice:
        ----------------
        If File 39's perturbation sampling has been run, derive this value
        from observed score uncertainty, e.g.:

            threshold = median(score_std)

        where ``score_std`` is the standard deviation of each site's score
        across perturbation runs.

        Rationale:
        A ranking difference smaller than the natural variation introduced
        by plausible weight changes is not meaningfully distinguishable, so
        those sites should be presented as tied.

    Returns
    -------
    list of dict
        Ordered rank clusters.

        Each cluster contains:
            cluster_rank : int
                Rank of the first site in the cluster.

            site_ids : list
                Equivalent-ranked sites.

            scores : list
                Corresponding scores.

            score_range : float
                max(score) - min(score) within cluster.
    """
    if not isinstance(ranked_scores, pd.DataFrame):
        raise TypeError("ranked_scores must be a pandas DataFrame.")

    required = {"rank", "site_id", "score"}
    missing = required - set(ranked_scores.columns)

    if missing:
        raise ValueError(
            f"ranked_scores missing required columns: {sorted(missing)}"
        )

    ranked = ranked_scores.sort_values("rank").reset_index(drop=True)

    scores = ranked["score"].to_numpy(dtype=float)

    if uncertainty_threshold is None:
        # Fallback default:
        # Use 0.5% of the observed score spread.
        #
        # Example:
        # score range = 0.80 - 0.40 = 0.40
        # threshold = 0.002
        #
        # This is intentionally conservative.
        #
        # Preferred production behavior:
        # Replace this with a statistic derived from File 39's
        # perturbation outputs, such as:
        #
        #     median(site_score_std)
        #
        # because that directly measures ranking uncertainty under
        # realistic weight variation.
        score_range = float(scores.max() - scores.min())
        uncertainty_threshold = max(score_range * 0.005, 1e-6)

    if uncertainty_threshold < 0:
        raise ValueError(
            "uncertainty_threshold must be non-negative."
        )

    clusters = []

    current_sites = [ranked.iloc[0]["site_id"]]
    current_scores = [float(ranked.iloc[0]["score"])]
    cluster_start_rank = int(ranked.iloc[0]["rank"])

    for i in range(1, len(ranked)):
        current_score = float(ranked.iloc[i]["score"])

        cluster_min = min(current_scores)
        cluster_max = max(current_scores)

        new_min = min(cluster_min, current_score)
        new_max = max(cluster_max, current_score)

        # Entire cluster must remain within threshold
        if (new_max - new_min) <= uncertainty_threshold:
            current_sites.append(ranked.iloc[i]["site_id"])
            current_scores.append(current_score)
        else:
            clusters.append(
                {
                    "cluster_rank": cluster_start_rank,
                    "site_ids": current_sites,
                    "scores": current_scores,
                    "score_range": max(current_scores)
                    - min(current_scores),
                }
            )

            current_sites = [ranked.iloc[i]["site_id"]]
            current_scores = [current_score]
            cluster_start_rank = int(ranked.iloc[i]["rank"])

    # Final cluster
    clusters.append(
        {
            "cluster_rank": cluster_start_rank,
            "site_ids": current_sites,
            "scores": current_scores,
            "score_range": max(current_scores)
            - min(current_scores),
        }
    )

    return clusters


def derive_uncertainty_threshold(
    score_std: Sequence[float],
    percentile: float = 50.0,
):
    """
    Derive an uncertainty threshold from File 39 perturbation outputs.

    Parameters
    ----------
    score_std : sequence of float
        Standard deviation of each site's score across perturbation runs.

    percentile : float, default=50
        Percentile to use:
            50 -> median
            75 -> more conservative
            90 -> highly conservative

    Returns
    -------
    float
        Recommended near-tie threshold.
    """
    score_std = np.asarray(score_std, dtype=float)

    if score_std.size == 0:
        raise ValueError("score_std cannot be empty.")

    return float(np.percentile(score_std, percentile))


__all__ = [
    "identify_near_ties",
    "derive_uncertainty_threshold",
]