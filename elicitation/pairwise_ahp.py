"""
pairwise_ahp.py

Analytic Hierarchy Process (AHP) pairwise elicitation and weight derivation.

Given a set of pairwise ratio-scale judgments between criteria (e.g. "A is
3x more important than B"), this module:

    1. Assembles those judgments into a reciprocal comparison matrix
       (build_comparison_matrix).
    2. Derives criterion weights from that matrix's principal eigenvector
       (compute_weights_from_matrix).

Scope note: this module deliberately does NOT compute the consistency
ratio (CR) / consistency index (CI) used to flag incoherent judgments
(e.g. A > B > C > A cycles). That lives in File 35 and is imported
separately, e.g.:

    from elicitation.consistency import consistency_ratio

Typical usage:

    from elicitation.pairwise_ahp import build_comparison_matrix, compute_weights_from_matrix

    judgments = {
        ("cost", "quality"): 3,   # cost is 3x more important than quality
        ("cost", "speed"): 5,     # cost is 5x more important than speed
        ("quality", "speed"): 2,  # quality is 2x more important than speed
    }
    criteria = ["cost", "quality", "speed"]

    matrix = build_comparison_matrix(judgments, criteria)
    weights = compute_weights_from_matrix(matrix)
    # weights == {"cost": 0.633, "quality": 0.260, "speed": 0.106}  (approx)
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Pairwise judgments keyed by (criterion_a, criterion_b), value = how many
# times more important criterion_a is than criterion_b, on Saaty's 1-9
# ratio scale (1 = equal importance, 9 = extreme importance).
PairwiseJudgments = Dict[Tuple[str, str], float]


def build_comparison_matrix(
    pairwise_judgments: PairwiseJudgments,
    criteria: Sequence[str],
) -> np.ndarray:
    """
    Build a reciprocal pairwise comparison matrix from ratio-scale judgments.

    Args:
        pairwise_judgments: mapping from (criterion_a, criterion_b) -> ratio,
            where ratio is how many times more important criterion_a is than
            criterion_b. Only one direction of each pair needs to be
            supplied; the reciprocal (criterion_b, criterion_a) -> 1/ratio
            is filled in automatically. Supplying both directions is fine
            as long as they're consistent (the second is simply used as
            given rather than re-derived).
        criteria: the ordered list of criterion names. This fixes the row/
            column order of the returned matrix.

    Returns:
        An (n x n) numpy array `M` where `M[i, j]` is how many times more
        important `criteria[i]` is than `criteria[j]`. The diagonal is
        always 1.0 (a criterion is equally important to itself), and the
        matrix is reciprocal: `M[i, j] == 1 / M[j, i]`.

    Raises:
        KeyError: if neither (a, b) nor (b, a) is present in
            pairwise_judgments for some pair of distinct criteria.
    """
    n = len(criteria)
    index = {name: i for i, name in enumerate(criteria)}
    matrix = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = criteria[i], criteria[j]
            if (a, b) in pairwise_judgments:
                ratio = pairwise_judgments[(a, b)]
            elif (b, a) in pairwise_judgments:
                ratio = 1.0 / pairwise_judgments[(b, a)]
            else:
                raise KeyError(
                    f"No pairwise judgment supplied for ('{a}', '{b}') "
                    f"or its reciprocal ('{b}', '{a}')."
                )
            matrix[i, j] = ratio
            matrix[j, i] = 1.0 / ratio

    return matrix


def compute_weights_from_matrix(comparison_matrix: np.ndarray) -> np.ndarray:
    """
    Derive AHP weights as the principal (dominant) eigenvector of the
    comparison matrix, normalized to sum to 1.

    This is the standard eigenvector method: for a perfectly consistent
    reciprocal matrix, the principal eigenvector is exactly proportional to
    the true underlying weights. For real (imperfectly consistent)
    judgments, it's the best linear approximation. Consistency itself is
    NOT checked here -- see File 35.

    Args:
        comparison_matrix: an (n x n) reciprocal pairwise comparison matrix,
            as produced by build_comparison_matrix.

    Returns:
        A length-n numpy array of non-negative weights summing to 1.0, in
        the same row/column order as the input matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)

    # The principal eigenvalue of a positive reciprocal matrix is real,
    # positive, and >= n; it's the largest-magnitude eigenvalue.
    principal_index = np.argmax(eigenvalues.real)
    principal_vector = eigenvectors[:, principal_index].real

    # Sign of an eigenvector is arbitrary; normalize to positive, then to sum 1.
    if np.sum(principal_vector) < 0:
        principal_vector = -principal_vector

    weights = principal_vector / np.sum(principal_vector)
    return weights


def compute_weights_dict(
    comparison_matrix: np.ndarray, criteria: Sequence[str]
) -> Dict[str, float]:
    """
    Convenience wrapper: same as compute_weights_from_matrix, but returns a
    {criterion_name: weight} dict instead of a bare array, using `criteria`
    for labeling (must be in the same order used to build the matrix).
    """
    weights = compute_weights_from_matrix(comparison_matrix)
    return {name: float(w) for name, w in zip(criteria, weights)}


# =============================================================================
# Worked example (hand-verifiable)
# =============================================================================
#
# Three criteria: cost, quality, speed.
#
# Judgments (Saaty 1-9 scale):
#   cost   is 3x more important than quality
#   cost   is 5x more important than speed
#   quality is 2x more important than speed
#
# Comparison matrix (rows/cols in order cost, quality, speed):
#
#              cost     quality   speed
#   cost     [  1         3         5   ]
#   quality  [ 1/3        1         2   ]
#   speed    [ 1/5       1/2        1   ]
#
# This matrix is (numerically) perfectly consistent, since 3 * 2 = 6 ~ 5
# is the only "slack" -- actually cost/speed = 5 while (cost/quality) *
# (quality/speed) = 3 * 2 = 6, so it is NOT perfectly consistent (this is
# intentional -- real judgments rarely are, and that's exactly the kind of
# slack File 35's consistency check is for).
#
# Expected principal-eigenvector weights (rounded to 3 dp), as returned by
# compute_weights_from_matrix / compute_weights_dict:
#   cost:    0.648
#   quality: 0.230
#   speed:   0.122
#
# You can sanity-check these by hand with the column-average approximation
# method (normalize each column to sum to 1, then average across each row):
#
#   column sums: cost=1.533, quality=4.5, speed=8
#   normalized cost column:    [0.652, 0.217, 0.130]
#   normalized quality column: [0.667, 0.222, 0.111]
#   normalized speed column:   [0.625, 0.250, 0.125]
#   row averages (~weights):   [0.648, 0.230, 0.122]
#
# Here the column-average approximation and the exact eigenvector method
# agree to 3dp -- that's expected since this matrix is close to (though not
# perfectly) consistent.

if __name__ == "__main__":
    example_criteria = ["cost", "quality", "speed"]
    example_judgments: PairwiseJudgments = {
        ("cost", "quality"): 3,
        ("cost", "speed"): 5,
        ("quality", "speed"): 2,
    }

    example_matrix = build_comparison_matrix(example_judgments, example_criteria)
    print("Comparison matrix:")
    print(example_matrix)

    example_weights = compute_weights_dict(example_matrix, example_criteria)
    print("\nDerived weights (principal eigenvector method):")
    for name, w in example_weights.items():
        print(f"  {name:8s}: {w:.3f}")
