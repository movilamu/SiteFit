"""
Weighted-sum aggregation for site scoring.

This method is fully compensatory: a strong score on one criterion can
offset a weak score on another. For a partial-compensation alternative,
see File 28 (weighted product).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def weighted_sum_score(normalized_matrix, global_weights):
    """
    Compute the weighted-sum score for each site.

    For site i:
        S_i = sum_j w_j * x_tilde_ij

    Parameters
    ----------
    normalized_matrix : array-like
        2-D matrix of normalized criterion scores, with one row per site
        and one column per criterion.
    global_weights : array-like or mapping
        Global weight for each criterion. If a mapping is supplied, its
        values are used in the mapping's iteration order.

    Returns
    -------
    numpy.ndarray
        One weighted-sum score per site.

    Notes
    -----
    This method is fully compensatory — a strong score on one criterion
    can offset a weak score on another. See File 28 (weighted product)
    for the partial-compensation alternative.
    """
    matrix = np.asarray(normalized_matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("normalized_matrix must be a 2-D matrix.")

    if isinstance(global_weights, Mapping):
        weights = np.asarray(list(global_weights.values()), dtype=float)
    else:
        weights = np.asarray(global_weights, dtype=float).reshape(-1)

    if weights.ndim != 1:
        raise ValueError("global_weights must be a 1-D sequence or mapping.")

    if matrix.shape[1] != weights.size:
        raise ValueError(
            "Number of global weights must match the number of criteria "
            f"in normalized_matrix ({weights.size} != {matrix.shape[1]})."
        )

    return matrix @ weights


# =============================================================================
# Unit tests
# =============================================================================

if __name__ == "__main__":
    import unittest

    class TestWeightedSumScore(unittest.TestCase):
        def test_hand_verifiable_example(self):
            # Two sites, two criteria.
            #
            # Site A: 0.8 and 0.4
            # Site B: 0.2 and 1.0
            #
            # Weights: 0.6 and 0.4
            #
            # A = (0.6 * 0.8) + (0.4 * 0.4) = 0.64
            # B = (0.6 * 0.2) + (0.4 * 1.0) = 0.52
            matrix = np.array([
                [0.8, 0.4],
                [0.2, 1.0],
            ])
            weights = np.array([0.6, 0.4])

            result = weighted_sum_score(matrix, weights)

            np.testing.assert_allclose(result, [0.64, 0.52])

        def test_mapping_weights(self):
            matrix = np.array([
                [1.0, 0.0],
                [0.0, 1.0],
            ])
            weights = {"criterion_a": 0.7, "criterion_b": 0.3}

            result = weighted_sum_score(matrix, weights)

            np.testing.assert_allclose(result, [0.7, 0.3])

        def test_dimension_mismatch(self):
            matrix = np.array([[0.5, 0.5]])
            weights = np.array([1.0])

            with self.assertRaises(ValueError):
                weighted_sum_score(matrix, weights)

    unittest.main(verbosity=2)
