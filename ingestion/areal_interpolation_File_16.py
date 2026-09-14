"""
Areal interpolation utilities for reconciling source administrative geography
(e.g. Census 2011 wards/villages) onto canonical H3 analysis units.

WHY NAIVE AREA WEIGHTING IS WRONG
----------------------------------
The naive approach to "spreading" a source polygon's value onto a target
polygon is area-proportional interpolation:

    target_value += source_value * (overlap_area / source_area)

This implicitly assumes the value is distributed *uniformly* across the
source polygon's area. That assumption fails badly whenever the underlying
phenomenon is not uniform in space, which is the normal case for population
and population-derived variables:

  - A Census village polygon might contain one dense settlement cluster
    covering 5% of its area, with the remaining 95% being farmland, forest,
    or water with ~zero population.
  - If our target H3 cell happens to overlap only the empty 95%, naive area
    weighting will still assign it ~95% of the village's population, when
    the true share should be close to 0.
  - Conversely, a cell overlapping only the dense cluster would be
    under-credited.

The fix is population-weighted interpolation: instead of weighting the
overlap by its share of the source polygon's *area*, we weight it by its
share of the source polygon's *population* (area x local population
density, integrated over the overlap region using a population surface
such as gridded WorldPop/LandScan raster, or any point/cell layer with a
density field). This correctly routes value to where people actually are,
not where empty land happens to be.

Area weighting is only accurate when the source variable truly is uniform
over the source polygon (rare), or when no population_surface is available
(in which case it is used as a documented fallback, with the resulting
confidence score penalized accordingly).

WORKED EXAMPLE (verify by hand)
--------------------------------
Source village polygon V: a 2km x 1km rectangle, area = 2.0 km^2, with
total population 1000 people concentrated entirely in the left half
(x in [0,1], y in [0,1]) at uniform density; the right half (x in [1,2])
is uninhabited farmland.

Target H3 cell T: the right half of V, i.e. x in [1,2], y in [0,1],
area = 1.0 km^2. So overlap_area = 1.0 km^2 = 50% of the source area.

Naive area weighting:
    weight = overlap_area / source_area = 1.0 / 2.0 = 0.5
    interpolated_value = 1000 * 0.5 = 500 people   <-- WRONG, this half is empty

Population-weighted interpolation (population_surface tells us the left
half has density 1000 people/km^2 and the right half has density
0 people/km^2):
    population_in_overlap = 0 people/km^2 * 1.0 km^2 = 0
    population_in_source  = 1000 people/km^2 * 1.0 km^2 (left half)
                             + 0 * 1.0 km^2 (right half) = 1000
    weight = population_in_overlap / population_in_source = 0 / 1000 = 0.0
    interpolated_value = 1000 * 0.0 = 0 people     <-- CORRECT

This module's doctest-style example below (in interpolate_to_analysis_unit's
docstring) reproduces this scenario with a simpler two-cell population
surface and checks the numeric result.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


@dataclass
class PopulationCell:
    """A single cell of a population surface: a footprint polygon with a
    population density (people per unit area, in the same area units as
    the geometries, e.g. people per km^2 if coordinates are projected to km).
    """
    geometry: Polygon
    density: float  # people per unit area


@dataclass
class InterpolationResult:
    value: float
    confidence: float


def _safe_area(geom: BaseGeometry) -> float:
    return geom.area if geom is not None and not geom.is_empty else 0.0


def _population_in_region(region: BaseGeometry, population_surface: Sequence[PopulationCell]) -> float:
    """Integrate population (density * area) over `region` by intersecting it
    with each cell of the population surface. Cells that don't overlap the
    region contribute nothing.
    """
    total = 0.0
    for cell in population_surface:
        if cell.geometry is None or cell.geometry.is_empty:
            continue
        if not cell.geometry.intersects(region):
            continue
        overlap = cell.geometry.intersection(region)
        total += _safe_area(overlap) * cell.density
    return total


def interpolate_to_analysis_unit(
    source_polygons: Sequence[Polygon],
    source_values: Sequence[float],
    target_unit_geometry: Polygon,
    population_surface: Optional[Sequence[PopulationCell]] = None,
) -> InterpolationResult:
    """Distribute values held on `source_polygons` (e.g. Census villages/wards)
    onto `target_unit_geometry` (a canonical H3 analysis unit), weighted by
    each source polygon's share of overlap with the target.

    Args:
        source_polygons: Source geography polygons (same CRS/units as
            target_unit_geometry; use a projected CRS with linear units,
            e.g. metres or km, NOT raw lat/lon degrees, or area-based
            weighting will be distorted).
        source_values: The value held by each source polygon (e.g. total
            population, household count), same length/order as
            source_polygons.
        target_unit_geometry: The target H3 analysis unit polygon.
        population_surface: Optional sequence of PopulationCell giving a
            finer-grained population density surface. When provided,
            overlap weighting uses population mass (density * overlap area)
            instead of raw overlap area -- see module docstring for why
            this matters. When omitted, falls back to naive area weighting
            and the confidence score is penalized to reflect the added
            uncertainty.

    Returns:
        InterpolationResult(value, confidence) where:
          - value is the estimated total for the target unit, summed
            across all overlapping source polygons.
          - confidence is in [0, 1]. It is 1.0 only for a direct one-to-one
            geometric match (target fully covered by a single source
            polygon with no splitting required). It decreases as more
            source polygons must be split/blended to cover the target,
            and is further penalized when population_surface is
            unavailable (since area weighting is a weaker assumption).

    Raises:
        ValueError: if source_polygons and source_values differ in length,
            or target_unit_geometry is empty/invalid.

    Worked example (matches the scenario in the module docstring):
        Source village V = rectangle (0,0)-(2,1), area 2.0, value 1000.
        Population surface: left half (0,0)-(1,1) density 1000/km^2,
        right half (1,0)-(2,1) density 0/km^2.
        Target T = right half (1,0)-(2,1), area 1.0 (50% of V by area).

        >>> from shapely.geometry import Polygon
        >>> v = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
        >>> t = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        >>> left = PopulationCell(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), 1000.0)
        >>> right = PopulationCell(Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]), 0.0)
        >>> naive = interpolate_to_analysis_unit([v], [1000.0], t)
        >>> round(naive.value, 1)
        500.0
        >>> pop_weighted = interpolate_to_analysis_unit([v], [1000.0], t, [left, right])
        >>> round(pop_weighted.value, 1)
        0.0

        As shown above, naive area weighting assigns 500 people to an
        empty region; population weighting correctly assigns 0.
    """
    if len(source_polygons) != len(source_values):
        raise ValueError(
            f"source_polygons ({len(source_polygons)}) and source_values "
            f"({len(source_values)}) must have the same length."
        )
    if target_unit_geometry is None or target_unit_geometry.is_empty or not target_unit_geometry.is_valid:
        raise ValueError("target_unit_geometry must be a non-empty, valid polygon.")

    total_value = 0.0
    contributing_sources = 0
    # Track how much of the target's own area is actually covered by source
    # polygons, and how "fragmented" the contribution is, for the confidence
    # score.
    target_area = _safe_area(target_unit_geometry)
    covered_area = 0.0

    for source_geom, source_value in zip(source_polygons, source_values):
        if source_geom is None or source_geom.is_empty or not source_geom.intersects(target_unit_geometry):
            continue

        overlap = source_geom.intersection(target_unit_geometry)
        overlap_area = _safe_area(overlap)
        if overlap_area <= 0:
            continue

        if population_surface:
            pop_in_overlap = _population_in_region(overlap, population_surface)
            pop_in_source = _population_in_region(source_geom, population_surface)
            if pop_in_source > 0:
                weight = pop_in_overlap / pop_in_source
            else:
                # Population surface says the source has ~zero population
                # anywhere (e.g. an unpopulated rural polygon); fall back to
                # area weighting for this source only, since a population
                # weight would be an undefined 0/0.
                source_area = _safe_area(source_geom)
                weight = (overlap_area / source_area) if source_area > 0 else 0.0
        else:
            source_area = _safe_area(source_geom)
            weight = (overlap_area / source_area) if source_area > 0 else 0.0

        total_value += source_value * weight
        covered_area += overlap_area
        contributing_sources += 1

    confidence = _estimate_confidence(
        contributing_sources=contributing_sources,
        target_area=target_area,
        covered_area=covered_area,
        used_population_surface=bool(population_surface),
    )

    return InterpolationResult(value=total_value, confidence=confidence)


def _estimate_confidence(
    contributing_sources: int,
    target_area: float,
    covered_area: float,
    used_population_surface: bool,
) -> float:
    """Heuristic confidence score in [0, 1].

    Starts from full confidence and is discounted for:
      - No source overlap at all (0.0 -- no data to estimate from).
      - A single source polygon covering the full target unit with no
        splitting (stays at ~1.0: this is a direct match).
      - Multiple source polygons contributing (each additional split
        polygon adds interpolation uncertainty).
      - Incomplete coverage of the target unit by source geometry (the
        uncovered remainder is unaccounted for).
      - Falling back to naive area weighting instead of a population
        surface (weaker assumption, flat penalty).
    """
    if contributing_sources == 0 or target_area <= 0:
        return 0.0

    coverage_ratio = min(covered_area / target_area, 1.0)

    # Direct one-to-one match: exactly one contributing source and (near)
    # full coverage.
    if contributing_sources == 1 and coverage_ratio >= 0.999:
        base = 1.0
    else:
        # Penalize each extra contributing source polygon beyond the first.
        split_penalty = 0.08 * (contributing_sources - 1)
        base = max(0.5, 1.0 - split_penalty)
        # Penalize incomplete coverage of the target unit.
        base *= coverage_ratio

    if not used_population_surface:
        base *= 0.85  # flat penalty for the weaker area-weighting assumption

    return round(max(0.0, min(1.0, base)), 4)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
