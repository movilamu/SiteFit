"""
Weighted-product aggregation for site scoring.

This method is partially compensatory: a weak criterion is penalized more
sharply than under a weighted sum because a near-zero factor drags down the
whole product. A strong score on another criterion can compensate only
partially for that weakness.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


# A normalized score of exactly zero would make the entire product zero.
# Flooring at a tiny positive value keeps the calculation numerically usable
# while preserving the intended strong penalty for a zero/near-zero criterion.
EPSILON = 1e-12


def weighted_product_score(normalized_matrix, global_weights):
    """
    Compute the weighted-product score for each site.

    For site i:
        S_i = product_j (x_tilde_ij) ** w_j

    Parameters
    ----------
    normalized_matrix : array-like
        2-D matrix of normalized criterion scores, with one row per site and
        one column per criterion. Scores are expected to be in [0, 1].
    global_weights : array-like or mapping
        Global weight for each criterion. If a mapping is supplied, its
        values are used in the mapping's iteration order.

    Returns
    -------
    numpy.ndarray
        One weighted-product score per site.

    Notes
    -----
    This method penalizes weak criteria more sharply than weighted-sum
    aggregation. A near-zero factor drags the whole product down, so a strong
    score on another criterion can compensate only partially rather than
    fully.

    Exactly zero normalized scores are floored at EPSILON before exponentiation.
    Without this floor, any zero criterion would force the entire product to
    zero. The floor avoids that hard numerical collapse while retaining an
    effectively zero contribution at the scale of ordinary normalized scores.

    The implementation uses log-space internally:
        log(S_i) = sum_j w_j * log(max(x_tilde_ij, EPSILON))
    This is more numerically stable than directly multiplying many factors.
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

    if not np.all(np.isfinite(matrix)):
        raise ValueError("normalized_matrix must contain only finite values.")

    if not np.all(np.isfinite(weights)):
        raise ValueError("global_weights must contain only finite values.")

    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("normalized_matrix values must be in the [0, 1] range.")

    if np.any(weights < 0.0):
        raise ValueError("global_weights must be non-negative.")

    floored_matrix = np.maximum(matrix, EPSILON)
    log_scores = np.sum(np.log(floored_matrix) * weights, axis=1)

    return np.exp(log_scores)


# =============================================================================
# Unit tests
# =============================================================================

if __name__ == "__main__":
    import unittest

    class TestWeightedProductScore(unittest.TestCase):
        def test_same_hand_verifiable_example_as_file_27(self):
            # Same two-site, two-criterion example used by File 27
            # (aggregate_weighted_sum.py), so the methods can be compared
            # directly on identical input.
            #
            # Site A: 0.8 and 0.4
            # Site B: 0.2 and 1.0
            #
            # Weights: 0.6 and 0.4
            #
            # A = (0.8 ** 0.6) * (0.4 ** 0.4) ≈ 0.6063
            # B = (0.2 ** 0.6) * (1.0 ** 0.4) ≈ 0.3807
            matrix = np.array([
                [0.8, 0.4],
                [0.2, 1.0],
            ])
            weights = np.array([0.6, 0.4])

            result = weighted_product_score(matrix, weights)

            expected = np.array([
                (0.8 ** 0.6) * (0.4 ** 0.4),
                (0.2 ** 0.6) * (1.0 ** 0.4),
            ])
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(
                result,
                [0.6062866260, 0.3807307877],
                rtol=1e-9,
                atol=1e-9,
            )

        def test_zero_is_floored(self):
            matrix = np.array([[0.0, 1.0]])
            weights = np.array([0.6, 0.4])

            result = weighted_product_score(matrix, weights)

            expected = EPSILON ** 0.6
            self.assertGreater(result[0], 0.0)
            self.assertAlmostEqual(result[0], expected)

        def test_mapping_weights(self):
            matrix = np.array([
                [1.0, 0.5],
                [0.5, 1.0],
            ])
            weights = {"criterion_a": 0.7, "criterion_b": 0.3}

            result = weighted_product_score(matrix, weights)

            expected = np.array([
                1.0 ** 0.7 * 0.5 ** 0.3,
                0.5 ** 0.7 * 1.0 ** 0.3,
            ])
            np.testing.assert_allclose(result, expected)

        def test_dimension_mismatch(self):
            matrix = np.array([[0.5, 0.5]])
            weights = np.array([1.0])

            with self.assertRaises(ValueError):
                weighted_product_score(matrix, weights)

        def test_invalid_normalized_range(self):
            matrix = np.array([[1.1, 0.5]])
            weights = np.array([0.5, 0.5])

            with self.assertRaises(ValueError):
                weighted_product_score(matrix, weights)

    unittest.main(verbosity=2)
