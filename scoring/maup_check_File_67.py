"""
Diagnostic for the Modifiable Areal Unit Problem (MAUP).

This module is intentionally outside the live scoring pipeline. It re-runs
File 16's areal interpolation against an alternative set of spatial units,
then compares criterion values and site rankings.

The diagnostic is deliberately generic about the source criterion: callers
provide source polygons/values and the canonical and alternative target units.
"""

from typing import Any, Dict, Mapping, Sequence

try:
    from .interpolate import interpolate_to_analysis_unit
except ImportError:
    # File 16 may have a different filename in a consuming project. The
    # fallback keeps this module easy to adapt without hiding the dependency.
    try:
        from scoring.interpolate import interpolate_to_analysis_unit
    except ImportError:
        interpolate_to_analysis_unit = None


def _require_interpolator():
    if interpolate_to_analysis_unit is None:
        raise ImportError(
            "Could not import File 16's interpolate_to_analysis_unit(). "
            "Update the import path in maup_check.py to match File 16."
        )
    return interpolate_to_analysis_unit


def _rank(values: Sequence[float]) -> list[int]:
    """Return 1-based ranks, with larger values ranked better."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0] * len(values)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


def _rank_changes(
    baseline: Sequence[float], alternative: Sequence[float]
) -> Dict[str, Any]:
    baseline_ranks = _rank(baseline)
    alternative_ranks = _rank(alternative)
    changed = [
        i for i, (a, b) in enumerate(zip(baseline_ranks, alternative_ranks))
        if a != b
    ]

    n = len(baseline)
    return {
        "baseline_ranks": baseline_ranks,
        "alternative_ranks": alternative_ranks,
        "sites_with_changed_rank": len(changed),
        "rank_change_rate": (len(changed) / n) if n else 0.0,
        "mean_absolute_rank_change": (
            sum(abs(a - b) for a, b in zip(baseline_ranks, alternative_ranks)) / n
            if n else 0.0
        ),
    }


def _interpolate_units(
    demographic_source_data: Mapping[str, Any],
    target_units: Sequence[Any],
) -> list[float]:
    """
    Interpolate one criterion onto each target unit.

    Expected source-data keys:
      - source_polygons: source administrative polygons
      - source_values: criterion values for those polygons
      - population_surface: optional File 16 PopulationCell sequence
    """
    interpolate = _require_interpolator()

    source_polygons = demographic_source_data["source_polygons"]
    source_values = demographic_source_data["source_values"]
    population_surface = demographic_source_data.get("population_surface")

    return [
        interpolate(
            source_polygons=source_polygons,
            source_values=source_values,
            target_unit_geometry=unit,
            population_surface=population_surface,
        ).value
        for unit in target_units
    ]


def test_unit_sensitivity(
    demographic_source_data: Mapping[str, Any],
    alternative_units: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Re-run File 16 interpolation on an alternative spatial unit definition.

    Args:
        demographic_source_data:
            Mapping containing ``source_polygons`` and ``source_values``.
            Optionally contains File 16's ``population_surface``.
        alternative_units:
            Mapping with:
              - ``baseline``: canonical target-unit geometries.
              - ``alternative``: coarser/finer target-unit geometries.
              - optionally ``site_ids``: identifiers aligned to baseline units.
              - optionally ``alternative_to_baseline``: mapping from each
                alternative unit index to the baseline site indices it represents.

            The mapping is needed because a coarser/finer grid does not
            necessarily have the same number of units as the baseline.

    Returns:
        A diagnostic report containing interpolated criterion values, value
        changes, and ranking changes. The report also states whether the
        alternative geometry can be compared directly with the baseline.

    Notes:
        This is a periodic diagnostic/reporting tool, not part of the live
        pipeline. It surfaces MAUP risk; it does not decide whether a
        spatial-unit definition is correct.
    """
    baseline_units = alternative_units.get("baseline")
    comparison_units = alternative_units.get("alternative")

    if not baseline_units or not comparison_units:
        raise ValueError(
            "alternative_units must contain non-empty 'baseline' and "
            "'alternative' target-unit geometry sequences."
        )

    baseline_values = _interpolate_units(demographic_source_data, baseline_units)
    alternative_values = _interpolate_units(
        demographic_source_data, comparison_units
    )

    site_ids = alternative_units.get(
        "site_ids", list(range(len(baseline_units)))
    )

    report: Dict[str, Any] = {
        "diagnostic": "MAUP / unit sensitivity",
        "purpose": "Periodic diagnostic; not part of the live scoring pipeline.",
        "baseline_unit_count": len(baseline_values),
        "alternative_unit_count": len(alternative_values),
        "baseline_values": baseline_values,
        "alternative_values": alternative_values,
        "site_ids": site_ids,
    }

    # If both unit definitions represent the same sites one-to-one, compare
    # directly. Otherwise, aggregate alternative units into baseline sites
    # using the caller-supplied mapping before comparing rankings.
    if len(baseline_values) == len(alternative_values):
        comparable_values = alternative_values
        report["comparison_method"] = "direct_one_to_one"
    else:
        mapping = alternative_units.get("alternative_to_baseline")
        if mapping is None:
            report["comparison_method"] = "not_directly_comparable"
            report["warning"] = (
                "Baseline and alternative grids have different unit counts. "
                "Provide alternative_to_baseline to map/aggregate alternative "
                "units to baseline sites before interpreting rank changes."
            )
            return report

        comparable_values = [0.0] * len(baseline_values)
        for alt_index, baseline_index in enumerate(mapping):
            if not 0 <= baseline_index < len(comparable_values):
                raise ValueError(
                    f"alternative_to_baseline[{alt_index}]={baseline_index} "
                    "is outside the baseline unit range."
                )
            comparable_values[baseline_index] += alternative_values[alt_index]

        report["comparison_method"] = "mapped_aggregation"

    value_changes = [
        alternative - baseline
        for baseline, alternative in zip(baseline_values, comparable_values)
    ]
    report["value_change"] = value_changes
    report["mean_absolute_value_change"] = (
        sum(abs(x) for x in value_changes) / len(value_changes)
        if value_changes else 0.0
    )

    report["ranking"] = _rank_changes(baseline_values, comparable_values)
    report["maup_risk_flag"] = (
        report["ranking"]["rank_change_rate"] > 0.0
    )

    return report


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
