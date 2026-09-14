"""
ingestion/cannibalization.py
=============================
Cannibalization scoring for candidate sites, built on top of the
Huff/gravity-model catchment calculator in File 19
(`ingestion/gravity_catchment.py`).

Why this exists
----------------
A candidate site can look excellent in isolation: strong demographics,
good attractiveness, an isochrone or gravity-model catchment full of
demand. But if most of that captured demand is not NEW demand being
pulled into the estate, and is instead demand that our OWN existing
stores would otherwise have captured, the "great" site is really just
moving revenue from one till to another (often at a net loss, once the
cost of opening/operating a second location is included). Isolated
demographic or catchment scores cannot see this, because they only look
at the candidate against the world in general -- they don't model the
candidate competing against our own estate for the same fixed pool of
origin demand.

`estimate_cannibalization` makes this visible by running the gravity
model (File 19's `compute_gravity_catchment`) twice over the same origin
demand:

  1. BASELINE  -- only the existing own-estate stores are in the
     competitive set. Each own-estate store captures some baseline
     amount of demand from each origin.
  2. SCENARIO  -- the candidate site is added to that same competitive
     set. Because gravity-model probabilities are a share of a fixed
     total (see File 19's "demand is conserved" sanity check), adding a
     new, closer-or-more-attractive destination can only ever pull
     probability mass away from the existing destinations (or from
     nothing, if the candidate is irrelevant to that origin) -- it can
     never increase what the existing stores capture.

The drop in own-estate captured demand between BASELINE and SCENARIO is
demand transferred away from the existing estate specifically because
the candidate opened. That transferred amount, expressed as a fraction
of the candidate's own total captured demand, is the cannibalization
score:

    cannibalization_score = (sum over own-estate stores of
                              baseline_captured - scenario_captured)
                             / candidate_captured_demand_in_scenario

    0.0  = candidate's demand is entirely net-new; no overlap with the
           existing estate.
    1.0  = candidate's entire captured demand is demand pulled straight
           off existing own-estate stores; effectively zero net-new
           demand once it opens.
    >1.0 is possible in degenerate cases (e.g. a tiny candidate that
    mostly siphons a small amount from many origins where it captures
    little demand itself) and should be treated as "very high"
    cannibalization rather than a literal percentage.

This function does not decide whether a site is good or bad -- a site
with moderate cannibalization can still be worth opening (e.g. to
defend market share ahead of a competitor). It exists purely to make
the cannibalization mechanism visible and quantified, alongside whatever
isolated demographic/attractiveness score a site already has, so that a
site that scores well in isolation but mostly eats an existing store is
not mistaken for a net-new-demand win.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .gravity_catchment import (
    DestinationSite,
    DistanceFn,
    GravityCatchmentError,
    OriginPoint,
    _coerce_destination,
    _coerce_origin,
    compute_gravity_catchment,
)

logger = logging.getLogger("cannibalization")


@dataclass
class CannibalizationResult:
    site_id: Any
    cannibalization_score: float
    candidate_captured_demand: float
    demand_transferred_from_estate: float
    per_store_transfer: Dict[str, float]  # own-estate store id -> demand lost to candidate

    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "cannibalization_score": self.cannibalization_score,
            "candidate_captured_demand": self.candidate_captured_demand,
            "demand_transferred_from_estate": self.demand_transferred_from_estate,
            "per_store_transfer": self.per_store_transfer,
        }


def estimate_cannibalization(
    new_site: Any,
    own_estate_sites: Sequence[Any],
    origin_points: Sequence[Any],
    alpha: float = 1.0,
    beta: float = 2.0,
    distance_fn: Optional[DistanceFn] = None,
    road_graph: Optional[Any] = None,
) -> CannibalizationResult:
    """Estimate what share of a candidate site's captured demand would be
    transferred FROM existing own-estate stores rather than representing
    net-new demand.

    Runs the File 19 gravity/Huff model twice over the same origin demand:
    once with only `own_estate_sites` competing (the baseline), and once
    with `new_site` added to that competitive set (the scenario). Because
    gravity-model shares are conserved (they sum to each origin's total
    demand -- see File 19's worked-example sanity check), any demand the
    existing stores lose between the two runs is demand the candidate site
    pulled directly off the estate, not demand it generated on its own.

    Args:
        new_site: the candidate DestinationSite (or dict with keys
            {id, lat, lon, attractiveness (or floor_area_sqm/floor_area)}).
        own_estate_sites: sequence of existing own-estate DestinationSite
            entries (or equivalent dicts) to test the candidate against.
        origin_points: sequence of OriginPoint (or dicts with keys
            {id, lat, lon, demand}) representing the demand surface shared
            by the candidate and the existing estate. This should be the
            same origin set used to score the candidate in isolation, so
            the cannibalization score is comparable to that isolated score.
        alpha: attractiveness sensitivity exponent, passed through to
            `compute_gravity_catchment` (default 1.0, matching File 19).
        beta: distance-decay sensitivity exponent, passed through to
            `compute_gravity_catchment` (default 2.0, matching File 19).
        distance_fn: optional callable(origin, dest) -> km, passed through
            to `compute_gravity_catchment`. If omitted, File 19's default
            road-network-with-haversine-fallback distance is used.
        road_graph: optional pre-built NetworkX graph, passed through to
            `compute_gravity_catchment` (ignored if `distance_fn` is given).

    Returns:
        A CannibalizationResult with:
            cannibalization_score: 0 = no overlap with own estate (all
                captured demand is net-new); higher = more demand
                transferred from existing stores (see module docstring
                for the >1.0 edge case).
            candidate_captured_demand: total demand the candidate site
                captures in the scenario run (own estate present).
            demand_transferred_from_estate: absolute demand lost by
                own-estate stores between baseline and scenario.
            per_store_transfer: same breakdown, per own-estate store id,
                so the specific store(s) most cannibalized can be
                identified rather than only an aggregate score.

    Raises:
        GravityCatchmentError: propagated from `compute_gravity_catchment`
            for invalid/missing input (empty origins, non-positive
            attractiveness, etc).
    """
    if not own_estate_sites:
        raise GravityCatchmentError(
            "own_estate_sites must not be empty; cannibalization is undefined "
            "with no existing estate to compare against."
        )

    candidate = _coerce_destination(new_site)
    estate = [_coerce_destination(s) for s in own_estate_sites]
    estate_ids = {str(s.id) for s in estate}

    if str(candidate.id) in estate_ids:
        raise GravityCatchmentError(
            f"new_site id {candidate.id!r} collides with an id in own_estate_sites; "
            "candidate and existing stores must have distinct ids."
        )

    # BASELINE: existing estate only, competing amongst itself.
    baseline = compute_gravity_catchment(
        origin_points=origin_points,
        destination_sites=estate,
        alpha=alpha,
        beta=beta,
        distance_fn=distance_fn,
        road_graph=road_graph,
    )

    # SCENARIO: candidate added to the same competitive set.
    scenario = compute_gravity_catchment(
        origin_points=origin_points,
        destination_sites=list(estate) + [candidate],
        alpha=alpha,
        beta=beta,
        distance_fn=distance_fn,
        road_graph=road_graph,
    )

    baseline_captured = baseline["captured_demand"]
    scenario_captured = scenario["captured_demand"]

    per_store_transfer: Dict[str, float] = {}
    total_transferred = 0.0
    for store_id in estate_ids:
        before = baseline_captured.get(store_id, 0.0)
        after = scenario_captured.get(store_id, 0.0)
        lost = before - after
        # Guard against floating-point noise producing a tiny negative
        # "gain" for a store the candidate has no real effect on.
        if lost < 0:
            lost = 0.0
        per_store_transfer[store_id] = lost
        total_transferred += lost

    candidate_captured = scenario_captured.get(str(candidate.id), 0.0)

    if candidate_captured <= 0:
        logger.warning(
            "Candidate site %s captures zero demand in the scenario run; "
            "cannibalization score is undefined and reported as 0.0.",
            candidate.id,
        )
        score = 0.0
    else:
        score = total_transferred / candidate_captured

    logger.info(
        "Cannibalization for candidate %s: score=%.4f, "
        "candidate_captured=%.2f, transferred_from_estate=%.2f",
        candidate.id, score, candidate_captured, total_transferred,
    )

    return CannibalizationResult(
        site_id=candidate.id,
        cannibalization_score=score,
        candidate_captured_demand=candidate_captured,
        demand_transferred_from_estate=total_transferred,
        per_store_transfer=per_store_transfer,
    )


def estimate_cannibalization_batch(
    candidate_sites: Sequence[Any],
    own_estate_sites: Sequence[Any],
    origin_points: Sequence[Any],
    alpha: float = 1.0,
    beta: float = 2.0,
    distance_fn: Optional[DistanceFn] = None,
    road_graph: Optional[Any] = None,
) -> List[CannibalizationResult]:
    """Convenience wrapper: run `estimate_cannibalization` for each of several
    candidate sites against the same own-estate/origin data, so multiple
    candidates can be ranked side by side by cannibalization score.

    Each candidate is scored independently against the existing estate
    (candidates are not treated as competing with each other).
    """
    results = []
    for site in candidate_sites:
        results.append(
            estimate_cannibalization(
                new_site=site,
                own_estate_sites=own_estate_sites,
                origin_points=origin_points,
                alpha=alpha,
                beta=beta,
                distance_fn=distance_fn,
                road_graph=road_graph,
            )
        )
    return results


__all__ = [
    "CannibalizationResult",
    "estimate_cannibalization",
    "estimate_cannibalization_batch",
]
