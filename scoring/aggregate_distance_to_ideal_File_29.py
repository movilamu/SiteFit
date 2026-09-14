"""
scoring/aggregate_distance_to_ideal.py

TOPSIS-style distance-to-ideal aggregation.

Given a normalized [0, 1] decision matrix, the positive ideal is the
best normalized value observed for each criterion and the negative ideal
is the worst normalized value observed for each criterion. Global criterion
weights are applied to the squared distances before taking the Euclidean
distance.

The resulting score is relative closeness to the ideal:

    score_i = D_i^- / (D_i^+ + D_i^-)

where:
    D_i^+ = distance from site i to the ideal point
    D_i^- = distance from site i to the negative-ideal point

Higher scores are better and lie in [0, 1].
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def distance_to_ideal_score(normalized_matrix, global_weights):
    """
    Compute a TOPSIS-style distance-to-ideal score for each site.

    Parameters
    ----------
    normalized_matrix : array-like, shape (n_sites, n_criteria)
        Normalized criterion values. Criteria must already be oriented so
        that larger values are better (as produced by the normalization
        pipeline).
    global_weights : array-like or mapping
        Global criterion weights. If a mapping is supplied, its values are
        used in mapping insertion order. A mapping is therefore intended
        to be supplied in the same criterion order as normalized_matrix.

    Returns
    -------
    numpy.ndarray, shape (n_sites,)
        Relative closeness of every site to the ideal point. Higher is better.

    Notes
    -----
    For criterion j:

        ideal_j = max_i(x_ij)
        negative_ideal_j = min_i(x_ij)

    Weighted Euclidean distances are then:

        D_i^+ = sqrt(sum_j w_j * (x_ij - ideal_j)^2)
        D_i^- = sqrt(sum_j w_j * (x_ij - negative_ideal_j)^2)

    The final score is D_i^- / (D_i^+ + D_i^-).

    If all criteria are identical across all sites, both distances are zero
    and every site receives 0.5 because there is no basis for discrimination.
    """
    if hasattr(normalized_matrix, "to_numpy"):
        matrix = normalized_matrix.to_numpy(dtype=float)
        columns = list(normalized_matrix.columns)
    else:
        matrix = np.asarray(normalized_matrix, dtype=float)
        columns = None

    if matrix.ndim != 2:
        raise ValueError("normalized_matrix must be a 2-dimensional matrix.")

    n_sites, n_criteria = matrix.shape
    if n_sites == 0 or n_criteria == 0:
        raise ValueError("normalized_matrix must contain at least one site and one criterion.")

    if isinstance(global_weights, Mapping):
        if columns is not None:
            missing = [column for column in columns if column not in global_weights]
            if missing:
                raise ValueError(
                    f"global_weights is missing weights for criteria: {missing}"
                )
            weights = np.asarray([global_weights[column] for column in columns], dtype=float)
        else:
            weights = np.asarray(list(global_weights.values()), dtype=float)
    else:
        weights = np.asarray(global_weights, dtype=float)

    if weights.ndim != 1 or weights.size != n_criteria:
        raise ValueError(
            f"global_weights must contain exactly {n_criteria} weights; "
            f"got {weights.size if weights.ndim == 1 else 'non-1D input'}."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("normalized_matrix must contain only finite values.")

    if not np.all(np.isfinite(weights)):
        raise ValueError("global_weights must contain only finite values.")

    if np.any(weights < 0):
        raise ValueError("global_weights must be non-negative.")

    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("global_weights must have a positive sum.")

    # Normalize weights defensively. This preserves the result when weights
    # already sum to 1 while allowing callers to provide proportional weights.
    weights = weights / weight_sum

    ideal = np.max(matrix, axis=0)
    negative_ideal = np.min(matrix, axis=0)

    positive_distance = np.sqrt(
        np.sum(weights * np.square(matrix - ideal), axis=1)
    )
    negative_distance = np.sqrt(
        np.sum(weights * np.square(matrix - negative_ideal), axis=1)
    )

    denominator = positive_distance + negative_distance
    scores = np.divide(
        negative_distance,
        denominator,
        out=np.full(n_sites, 0.5, dtype=float),
        where=denominator > 0,
    )

    return scores


# =============================================================================
# Unit tests
# =============================================================================

if __name__ == "__main__":
    import unittest

    class TestDistanceToIdealScore(unittest.TestCase):
        def test_hand_verifiable_example(self):
            # Reuses the normalized values established by the normalization
            # example in File 24:
            #
            # resident_density:             [0.0, 0.5, 1.0]
            # rent (cost-oriented):         [1.0, 0.5, 0.0]
            # competitor_density targetband:[1.0, 0.0, 0.0]
            #
            # Use global weights proportional to the available example's
            # first three criteria. The implementation normalizes weights,
            # so proportional weights are valid.
            matrix = np.array([
                [0.0, 1.0, 1.0],
                [0.5, 0.5, 0.0],
                [1.0, 0.0, 0.0],
            ])
            weights = np.array([0.42, 0.18, 0.32])

            got = distance_to_ideal_score(matrix, weights)

            ideal = np.array([1.0, 1.0, 1.0])
            negative_ideal = np.array([0.0, 0.0, 0.0])
            w = weights / weights.sum()

            d_plus = np.sqrt(np.sum(w * (matrix - ideal) ** 2, axis=1))
            d_minus = np.sqrt(np.sum(w * (matrix - negative_ideal) ** 2, axis=1))
            expected = d_minus / (d_plus + d_minus)

            np.testing.assert_allclose(got, expected, atol=1e-12)
            # The expected values are determined by the TOPSIS formula;
            # do not impose an extra ranking assumption on the example.

        def test_dataframe_mapping_alignment(self):
            import pandas as pd

            matrix = pd.DataFrame({
                "resident_density": [0.0, 1.0],
                "rent": [1.0, 0.0],
            })
            weights = {
                "rent": 0.2,
                "resident_density": 0.8,
            }

            got = distance_to_ideal_score(matrix, weights)
            np.testing.assert_allclose(got, [1 / 3, 2 / 3], atol=1e-12)

        def test_all_sites_identical(self):
            matrix = np.array([
                [0.4, 0.7],
                [0.4, 0.7],
                [0.4, 0.7],
            ])

            got = distance_to_ideal_score(matrix, [0.6, 0.4])
            np.testing.assert_allclose(got, [0.5, 0.5, 0.5])

        def test_invalid_weight_count(self):
            matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
            with self.assertRaises(ValueError):
                distance_to_ideal_score(matrix, [1.0])

        def test_negative_weights_rejected(self):
            matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
            with self.assertRaises(ValueError):
                distance_to_ideal_score(matrix, [0.5, -0.5])

    unittest.main(verbosity=2)
