"""
Criterion-correlation checks for the site-scoring pipeline.

Computes Pearson correlations across normalized criteria and flags pairs
whose absolute correlation meets/exceeds a configurable threshold.

Flagged pairs are candidates for human review as possible double-counting
of the same underlying signal. This module does not automatically merge,
remove, reweight, or otherwise alter the criteria hierarchy.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def check_criteria_correlation(
    normalized_matrix: pd.DataFrame,
    threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Check pairwise Pearson correlation between normalized criteria.

    Parameters
    ----------
    normalized_matrix : pandas.DataFrame
        Rows are sites and columns are criteria. Values should be the
        normalized [0, 1] criterion scores produced by normalize_all().
    threshold : float, default=0.8
        Absolute-correlation threshold used to flag potentially redundant
        criterion pairs. For example, 0.8 flags correlations >= +0.8
        and <= -0.8.

    Returns
    -------
    dict
        A report containing:
          - ``correlation_matrix``: DataFrame of pairwise correlations.
          - ``threshold``: threshold used.
          - ``flagged_pairs``: list of dictionaries with criterion names
            and correlation values.
          - ``flagged_pair_count``: number of flagged pairs.

    Notes
    -----
    A high absolute correlation is only a screening signal. It does not
    prove that two criteria measure the same construct. Flagged pairs
    should therefore be reviewed by a human before considering merging
    or discounting criteria.
    """
    if not isinstance(normalized_matrix, pd.DataFrame):
        normalized_matrix = pd.DataFrame(normalized_matrix)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")

    if normalized_matrix.shape[1] < 2:
        correlation_matrix = normalized_matrix.corr(method="pearson")
        return {
            "correlation_matrix": correlation_matrix,
            "threshold": threshold,
            "flagged_pairs": [],
            "flagged_pair_count": 0,
        }

    # Pearson correlation is calculated pairwise; pandas excludes NaNs
    # for each pair of criteria.
    correlation_matrix = normalized_matrix.corr(method="pearson")

    flagged_pairs: List[Dict[str, Any]] = []

    for criterion_a, criterion_b in combinations(correlation_matrix.columns, 2):
        correlation = correlation_matrix.loc[criterion_a, criterion_b]

        if pd.notna(correlation) and abs(float(correlation)) >= threshold:
            flagged_pairs.append(
                {
                    "criterion_a": criterion_a,
                    "criterion_b": criterion_b,
                    "correlation": float(correlation),
                    "absolute_correlation": abs(float(correlation)),
                }
            )

    flagged_pairs.sort(
        key=lambda pair: pair["absolute_correlation"],
        reverse=True,
    )

    return {
        "correlation_matrix": correlation_matrix,
        "threshold": threshold,
        "flagged_pairs": flagged_pairs,
        "flagged_pair_count": len(flagged_pairs),
    }


# =============================================================================
# Unit tests
# =============================================================================

if __name__ == "__main__":
    import unittest


    class TestCriteriaCorrelation(unittest.TestCase):
        def test_hand_verifiable_high_correlation_is_flagged(self):
            # criterion_b is exactly 2 * criterion_a, so Pearson r = 1.0.
            # criterion_c moves in the opposite direction, so r = -1.0.
            matrix = pd.DataFrame(
                {
                    "criterion_a": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "criterion_b": [0.0, 0.50, 1.0, 1.50, 2.0],
                    "criterion_c": [1.0, 0.75, 0.5, 0.25, 0.0],
                }
            )

            report = check_criteria_correlation(matrix, threshold=0.8)

            self.assertEqual(report["flagged_pair_count"], 3)
            self.assertAlmostEqual(
                report["correlation_matrix"].loc[
                    "criterion_a", "criterion_b"
                ],
                1.0,
            )

            pairs = {
                (item["criterion_a"], item["criterion_b"])
                for item in report["flagged_pairs"]
            }
            self.assertEqual(
                pairs,
                {
                    ("criterion_a", "criterion_b"),
                    ("criterion_a", "criterion_c"),
                    ("criterion_b", "criterion_c"),
                },
            )

    class TestThreshold(unittest.TestCase):
        def test_threshold_controls_flagging(self):
            matrix = pd.DataFrame(
                {
                    "a": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    # Strong, but not perfect, positive relationship.
                    "b": [0.1, 0.3, 0.35, 0.65, 0.7, 0.95],
                    "c": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                }
            )

            strict_report = check_criteria_correlation(matrix, threshold=0.99)
            loose_report = check_criteria_correlation(matrix, threshold=0.8)

            self.assertLess(
                strict_report["flagged_pair_count"],
                loose_report["flagged_pair_count"],
            )

    class TestEdgeCases(unittest.TestCase):
        def test_single_criterion(self):
            matrix = pd.DataFrame({"only_criterion": [0.0, 0.5, 1.0]})
            report = check_criteria_correlation(matrix)

            self.assertEqual(report["flagged_pairs"], [])
            self.assertEqual(report["flagged_pair_count"], 0)
            self.assertEqual(
                list(report["correlation_matrix"].columns),
                ["only_criterion"],
            )

        def test_invalid_threshold(self):
            matrix = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})

            with self.assertRaises(ValueError):
                check_criteria_correlation(matrix, threshold=1.1)

            with self.assertRaises(ValueError):
                check_criteria_correlation(matrix, threshold=-0.1)

        def test_nan_pairs_are_not_flagged(self):
            matrix = pd.DataFrame(
                {
                    "a": [1.0, 1.0, 1.0],
                    "b": [0.0, 0.5, 1.0],
                }
            )
            report = check_criteria_correlation(matrix, threshold=0.8)

            # Constant column has undefined Pearson correlation.
            self.assertEqual(report["flagged_pairs"], [])


    unittest.main(verbosity=2)
