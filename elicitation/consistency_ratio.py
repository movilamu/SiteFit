"""
consistency_ratio.py

Consistency checking for AHP (Analytic Hierarchy Process) pairwise
comparison matrices, as built by File 34's
`elicitation.pairwise_ahp.build_comparison_matrix`.

Implements Saaty's standard consistency check:

    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI

where lambda_max is the principal eigenvalue of the comparison matrix,
n is the matrix size, and RI is the Random Index for size n (the average
CI of large samples of randomly generated reciprocal matrices).

A matrix is considered acceptably consistent if CR < 0.10 (10%). If it
isn't, `analyze_consistency` also reports which specific pairwise
judgments look most out of line with the rest, so the user knows what to
revisit rather than just being told "rejected."

Typical usage, paired with File 34:

    from elicitation.pairwise_ahp import build_comparison_matrix, compute_weights_from_matrix
    from elicitation.consistency_ratio import (
        principal_eigenvalue,
        compute_consistency_ratio,
        is_acceptable,
        analyze_consistency,
    )

    matrix = build_comparison_matrix(judgments, criteria)
    lambda_max = principal_eigenvalue(matrix)
    ci, cr = compute_consistency_ratio(matrix, lambda_max)

    if not is_acceptable(cr):
        print(analyze_consistency(matrix, criteria))
"""

from typing import List, Sequence, Tuple

import numpy as np

# Saaty's published Random Index (RI) table: average CI of randomly
# generated reciprocal matrices of size n, for n = 1..10. RI[0] and RI[1]
# are 0 because matrices of size 1 or 2 are always perfectly consistent
# (there's nothing to be inconsistent about).
RANDOM_INDEX = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

CR_THRESHOLD = 0.10


def principal_eigenvalue(comparison_matrix: np.ndarray) -> float:
    """
    Compute lambda_max, the principal (largest real part) eigenvalue of a
    reciprocal pairwise comparison matrix. Provided as a convenience so
    callers don't have to duplicate the eigendecomposition already done in
    File 34's compute_weights_from_matrix.
    """
    eigenvalues = np.linalg.eigvals(comparison_matrix)
    return float(np.max(eigenvalues.real))


def compute_consistency_ratio(
    comparison_matrix: np.ndarray, eigenvalue_max: float
) -> Tuple[float, float]:
    """
    Compute the Consistency Index (CI) and Consistency Ratio (CR) for a
    pairwise comparison matrix.

    Args:
        comparison_matrix: an (n x n) reciprocal pairwise comparison matrix.
        eigenvalue_max: the principal eigenvalue (lambda_max) of that
            matrix, e.g. from `principal_eigenvalue(comparison_matrix)`.

    Returns:
        (CI, CR) as a tuple of floats.

    Raises:
        ValueError: if n is outside the supported Random Index table
            range (1..10), or n < 1.
    """
    n = comparison_matrix.shape[0]
    if n not in RANDOM_INDEX:
        raise ValueError(
            f"No Random Index entry for matrix size n={n}; "
            f"supported sizes are 1..{max(RANDOM_INDEX)}."
        )

    if n <= 2:
        # Size 1 or 2 matrices are always perfectly consistent by
        # construction (no independent judgments to be inconsistent).
        return 0.0, 0.0

    ci = (eigenvalue_max - n) / (n - 1)
    ri = RANDOM_INDEX[n]
    cr = ci / ri if ri != 0 else 0.0
    return ci, cr


def is_acceptable(cr: float) -> bool:
    """
    Standard AHP acceptability threshold: CR < 0.10 (10%) is considered
    acceptably consistent. CR >= 0.10 means the judgments should be
    revisited.
    """
    return cr < CR_THRESHOLD


def most_inconsistent_pairs(
    comparison_matrix: np.ndarray,
    criteria: Sequence[str],
    weights: np.ndarray = None,
    top_n: int = 3,
) -> List[Tuple[str, str, float]]:
    """
    Identify which specific pairwise judgments deviate most from what the
    derived weights would predict -- i.e. the judgments most responsible
    for a high CR.

    For a perfectly consistent matrix, M[i, j] == w[i] / w[j] exactly. We
    rank each off-diagonal (i, j) pair (i < j) by how far its actual
    judgment M[i, j] is from the "implied" ratio w[i] / w[j], on a log
    scale (log ratio is symmetric and scale-appropriate for Saaty-scale
    judgments).

    Args:
        comparison_matrix: the (n x n) comparison matrix.
        criteria: ordered criterion names matching the matrix's rows/cols.
        weights: optional precomputed weight vector (e.g. from
            `compute_weights_from_matrix` in File 34). If not supplied,
            it's derived here via the same principal-eigenvector method.
        top_n: how many of the worst-offending pairs to return.

    Returns:
        A list of (criterion_a, criterion_b, deviation) tuples, sorted by
        descending deviation, where deviation is
        |log(M[i,j]) - log(w[i]/w[j])|. Larger deviation = more
        inconsistent / more worth revisiting.
    """
    n = comparison_matrix.shape[0]

    if weights is None:
        eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
        principal_index = np.argmax(eigenvalues.real)
        principal_vector = eigenvectors[:, principal_index].real
        if np.sum(principal_vector) < 0:
            principal_vector = -principal_vector
        weights = principal_vector / np.sum(principal_vector)

    deviations = []
    for i in range(n):
        for j in range(i + 1, n):
            implied_ratio = weights[i] / weights[j]
            actual_ratio = comparison_matrix[i, j]
            deviation = abs(np.log(actual_ratio) - np.log(implied_ratio))
            deviations.append((criteria[i], criteria[j], float(deviation)))

    deviations.sort(key=lambda t: t[2], reverse=True)
    return deviations[:top_n]


def analyze_consistency(
    comparison_matrix: np.ndarray,
    criteria: Sequence[str],
    top_n: int = 3,
) -> str:
    """
    Convenience wrapper that runs the full consistency check and produces
    a human-readable report: whether the matrix passes, its CR, and (if it
    fails) which pairwise judgments look most inconsistent and worth
    revisiting.

    Args:
        comparison_matrix: the (n x n) comparison matrix.
        criteria: ordered criterion names matching the matrix's rows/cols.
        top_n: how many worst-offending pairs to surface on failure.

    Returns:
        A multi-line human-readable summary string.
    """
    lambda_max = principal_eigenvalue(comparison_matrix)
    ci, cr = compute_consistency_ratio(comparison_matrix, lambda_max)

    lines = [
        f"lambda_max = {lambda_max:.4f}",
        f"CI = {ci:.4f}",
        f"CR = {cr:.4f}",
    ]

    if is_acceptable(cr):
        lines.append(f"Result: ACCEPTABLE (CR = {cr:.4f} < {CR_THRESHOLD:.2f}).")
    else:
        lines.append(
            f"Result: REJECTED (CR = {cr:.4f} >= {CR_THRESHOLD:.2f}). "
            f"Judgments should be revised."
        )
        offenders = most_inconsistent_pairs(comparison_matrix, criteria, top_n=top_n)
        lines.append("Most inconsistent pairwise judgments (worst first):")
        for a, b, deviation in offenders:
            actual = comparison_matrix[criteria.index(a), criteria.index(b)]
            lines.append(
                f"  - ({a}, {b}): current judgment = {actual:.3g}x "
                f"(log-deviation from implied weights = {deviation:.3f})"
            )

    return "\n".join(lines)


# =============================================================================
# Worked example -- matches File 34's cost/quality/speed example exactly
# =============================================================================
#
# Comparison matrix (rows/cols in order cost, quality, speed):
#
#              cost     quality   speed
#   cost     [  1         3         5   ]
#   quality  [ 1/3        1         2   ]
#   speed    [ 1/5       1/2        1   ]
#
# This is File 34's intentionally slightly-inconsistent example (cost is
# judged 5x speed directly, but 3x2=6x speed transitively through quality).
#
# By hand:
#   n = 3, RI(3) = 0.58
#   lambda_max ~= 3.0037   (principal eigenvalue of the matrix above)
#   CI = (lambda_max - n) / (n - 1) = (3.0037 - 3) / 2 ~= 0.00183
#   CR = CI / RI = 0.00183 / 0.58 ~= 0.0032
#
# CR ~= 0.0032 is far below the 0.10 threshold, so this matrix is
# ACCEPTABLE -- consistent with File 34's own note that the matrix is
# "close to (though not perfectly) consistent."

if __name__ == "__main__":
    example_criteria = ["cost", "quality", "speed"]
    example_matrix = np.array(
        [
            [1.0, 3.0, 5.0],
            [1.0 / 3.0, 1.0, 2.0],
            [1.0 / 5.0, 1.0 / 2.0, 1.0],
        ]
    )

    print("Comparison matrix:")
    print(example_matrix)
    print()
    print(analyze_consistency(example_matrix, example_criteria))
