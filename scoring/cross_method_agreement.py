"""
scoring/cross_method_agreement.py

Cross-method ranking agreement diagnostics.

Three aggregation methods are used elsewhere in the scoring pipeline:

    - weighted_sum_score        (File 27, fully compensatory)
    - weighted_product_score    (File 28, partially compensatory)
    - distance_to_ideal_score   (File 29, TOPSIS-style)

Each method can produce a different ranking of sites because they encode
different assumptions about compensation between criteria. Rather than
silently picking one method's ranking as "the" answer, this module compares
the rankings produced by all three methods, quantifies how much they agree
(via pairwise Spearman rank correlation), and flags any sites whose rank
is unstable across methods so that they can be surfaced to the user instead
of being presented inside a single confident ranking.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np


def _to_ranks(scores: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D array of scores into 1-based ranks, where rank 1 is the
    best (highest) score. Ties are broken using the average-rank convention
    (the standard convention for Spearman correlation), so tied sites
    receive the mean of the ranks they would otherwise span.
    """
    # Sort descending (best score first) to get positions, then average
    # ranks for ties.
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]

    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # Positions i..j (0-based, in sorted order) are tied. Their 1-based
        # rank is the average of (i+1) .. (j+1).
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    return ranks


def _spearman(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """
    Spearman rank correlation computed directly as the Pearson correlation
    of two rank vectors. This formulation is robust to tied ranks (unlike
    the classic 1 - 6*sum(d^2)/(n*(n^2-1)) shortcut, which assumes no ties).
    """
    n = len(rank_a)
    if n < 2:
        # Correlation is undefined with fewer than two sites; treat as
        # perfect agreement since there is nothing to disagree about.
        return 1.0

    a = rank_a - rank_a.mean()
    b = rank_b - rank_b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        # All ranks identical (only possible if every site is tied on both
        # sides) -> perfect agreement.
        return 1.0
    return float(np.sum(a * b) / denom)


def compare_rankings(
    weighted_sum_result,
    weighted_product_result,
    distance_to_ideal_result,
    site_ids: Sequence = None,
    rank_diff_threshold: int = 3,
) -> Dict:
    """
    Compare site rankings produced by the three aggregation methods.

    Parameters
    ----------
    weighted_sum_result : array-like, shape (n_sites,)
        Scores from `weighted_sum_score` (File 27). Higher is better.
    weighted_product_result : array-like, shape (n_sites,)
        Scores from `weighted_product_score` (File 28). Higher is better.
    distance_to_ideal_result : array-like, shape (n_sites,)
        Scores from `distance_to_ideal_score` (File 29). Higher is better.
    site_ids : sequence, optional
        Labels for each site, in the same order as the score arrays. If
        omitted, sites are labeled by their positional index (0-based).
    rank_diff_threshold : int, default 3
        A site is flagged as "ranking-sensitive" if the spread between its
        best and worst rank across the three methods (max rank - min rank)
        is greater than or equal to this threshold. Lower thresholds flag
        more sites; this is intentionally configurable since "significant"
        disagreement depends on how the rankings will be used downstream.

    Returns
    -------
    dict with keys:
        "ranks" : dict mapping method name -> list of ranks (1 = best),
            one entry per site, in the original site order.
        "pairwise_spearman" : dict mapping "method_a__vs__method_b" ->
            Spearman rank correlation in [-1, 1].
        "overall_agreement" : float
            Mean of the three pairwise Spearman correlations. 1.0 means
            all three methods produce identical rankings; lower values
            indicate methods disagree about which sites are best.
        "flagged_sites" : list of dicts, one per ranking-sensitive site:
            {
                "site_id": ...,
                "ranks": {"weighted_sum": r1, "weighted_product": r2,
                          "distance_to_ideal": r3},
                "rank_spread": max_rank - min_rank,
            }
            Sorted by rank_spread, descending (most unstable first).

    Notes
    -----
    This function deliberately does not collapse the three methods into a
    single "winning" ranking. Its purpose is to surface disagreement, not
    to resolve it: sites whose relative standing depends heavily on which
    aggregation method is used should be flagged to the user rather than
    reported as though there were a single confident answer.
    """
    ws = np.asarray(weighted_sum_result, dtype=float).reshape(-1)
    wp = np.asarray(weighted_product_result, dtype=float).reshape(-1)
    dti = np.asarray(distance_to_ideal_result, dtype=float).reshape(-1)

    n_sites = ws.size
    if wp.size != n_sites or dti.size != n_sites:
        raise ValueError(
            "weighted_sum_result, weighted_product_result, and "
            "distance_to_ideal_result must all have the same length "
            f"(got {ws.size}, {wp.size}, {dti.size})."
        )
    if n_sites == 0:
        raise ValueError("Score arrays must contain at least one site.")

    if not (np.all(np.isfinite(ws)) and np.all(np.isfinite(wp)) and np.all(np.isfinite(dti))):
        raise ValueError("All score arrays must contain only finite values.")

    if site_ids is None:
        site_ids = list(range(n_sites))
    else:
        site_ids = list(site_ids)
        if len(site_ids) != n_sites:
            raise ValueError(
                f"site_ids must have length {n_sites}; got {len(site_ids)}."
            )

    if rank_diff_threshold < 0:
        raise ValueError("rank_diff_threshold must be non-negative.")

    method_scores: Mapping[str, np.ndarray] = {
        "weighted_sum": ws,
        "weighted_product": wp,
        "distance_to_ideal": dti,
    }
    method_names = list(method_scores.keys())

    method_ranks = {name: _to_ranks(scores) for name, scores in method_scores.items()}

    # Pairwise Spearman correlation between each pair of methods.
    pairwise_spearman: Dict[str, float] = {}
    correlations: List[float] = []
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            name_a, name_b = method_names[i], method_names[j]
            corr = _spearman(method_ranks[name_a], method_ranks[name_b])
            pairwise_spearman[f"{name_a}__vs__{name_b}"] = corr
            correlations.append(corr)

    overall_agreement = float(np.mean(correlations))

    # Flag sites whose rank spread across methods meets/exceeds the threshold.
    flagged_sites = []
    for idx in range(n_sites):
        ranks_for_site = {name: method_ranks[name][idx] for name in method_names}
        spread = max(ranks_for_site.values()) - min(ranks_for_site.values())
        if spread >= rank_diff_threshold:
            flagged_sites.append({
                "site_id": site_ids[idx],
                "ranks": ranks_for_site,
                "rank_spread": spread,
            })

    flagged_sites.sort(key=lambda entry: entry["rank_spread"], reverse=True)

    ranks_output = {
        name: [method_ranks[name][idx] for idx in range(n_sites)]
        for name in method_names
    }

    return {
        "ranks": ranks_output,
        "pairwise_spearman": pairwise_spearman,
        "overall_agreement": overall_agreement,
        "flagged_sites": flagged_sites,
    }


# =============================================================================
# Unit tests
# =============================================================================

if __name__ == "__main__":
    import unittest

    class TestCompareRankings(unittest.TestCase):
        def test_identical_rankings_full_agreement(self):
            # All three methods agree exactly on ranking -> correlation 1.0
            # everywhere, and nothing should be flagged.
            ws = [0.9, 0.6, 0.3]
            wp = [0.8, 0.5, 0.1]
            dti = [0.95, 0.55, 0.2]

            result = compare_rankings(ws, wp, dti, site_ids=["A", "B", "C"])

            self.assertAlmostEqual(result["overall_agreement"], 1.0, places=9)
            for corr in result["pairwise_spearman"].values():
                self.assertAlmostEqual(corr, 1.0, places=9)
            self.assertEqual(result["flagged_sites"], [])
            self.assertEqual(result["ranks"]["weighted_sum"], [1.0, 2.0, 3.0])

        def test_disagreement_flags_site(self):
            # Site C is best under weighted_sum but worst under the other
            # two methods -> large rank spread, should be flagged.
            ws = [0.5, 0.4, 0.9]     # rank: B=?, C best -> C:1, A:2, B:3
            wp = [0.9, 0.5, 0.1]     # A:1, B:2, C:3
            dti = [0.8, 0.6, 0.05]   # A:1, B:2, C:3

            result = compare_rankings(ws, wp, dti, site_ids=["A", "B", "C"],
                                       rank_diff_threshold=2)

            flagged_ids = [entry["site_id"] for entry in result["flagged_sites"]]
            self.assertIn("C", flagged_ids)
            self.assertLess(result["overall_agreement"], 1.0)

        def test_ties_produce_average_ranks(self):
            ws = [0.5, 0.5, 0.1]
            wp = [0.5, 0.5, 0.1]
            dti = [0.5, 0.5, 0.1]

            result = compare_rankings(ws, wp, dti)

            # Two-way tie for first place -> both get rank 1.5, last gets 3.
            self.assertEqual(result["ranks"]["weighted_sum"], [1.5, 1.5, 3.0])
            self.assertAlmostEqual(result["overall_agreement"], 1.0, places=9)

        def test_length_mismatch_raises(self):
            with self.assertRaises(ValueError):
                compare_rankings([0.1, 0.2], [0.1, 0.2, 0.3], [0.1, 0.2])

        def test_site_ids_length_mismatch_raises(self):
            with self.assertRaises(ValueError):
                compare_rankings([0.1, 0.2], [0.1, 0.2], [0.1, 0.2],
                                  site_ids=["only_one"])

        def test_default_site_ids_are_positional(self):
            result = compare_rankings([0.1, 0.9], [0.2, 0.8], [0.3, 0.7])
            self.assertEqual(result["flagged_sites"], [])
            # No error and ranks computed for both default-labeled sites.
            self.assertEqual(len(result["ranks"]["weighted_sum"]), 2)

    unittest.main(verbosity=2)
