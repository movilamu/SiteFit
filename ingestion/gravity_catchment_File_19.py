"""
ingestion/gravity_catchment.py
===============================
Probabilistic (Huff/gravity-model) catchment calculator.

    P(i -> j) = (A_j^alpha / d_ij^beta) / sum_k (A_k^alpha / d_ik^beta)

    P(i -> j)   probability that demand originating at i is captured by
                destination j
    A_j         attractiveness of destination j (floor area, or any
                configurable proxy passed in on the destination record)
    d_ij        travel distance between origin i and destination j, in
                kilometers, measured over the routable ROAD NETWORK
                (File 12's `road_network` table / GeoJSON export) — not
                straight-line distance
    alpha, beta sensitivity parameters (default alpha=1, beta=2)

Why this is not just "run isochrones and count what falls inside"
-------------------------------------------------------------------
File 18's isochrone client (`ingestion/isochrones.py`) answers a binary,
single-destination question: "is this origin within N minutes of THIS
site?" Stacking isochrones from several competing sites still assigns
each origin's full demand exclusively to whichever polygon(s) contain it
(or, with overlaps, to all of them at 100% each — double-counting demand
rather than splitting it).

The gravity/Huff model instead treats every origin's demand as a fixed
mass that gets SPLIT probabilistically across every reachable destination,
in proportion to each destination's attractiveness and inverse penalized
distance. A site that is slightly farther but much larger can still win
the majority of an origin's demand; a site that is close but tiny loses
share to a bigger competitor a few minutes further away. This is what you
want for site-selection / cannibalization analysis: "of the demand at
origin i, what fraction do we expect to capture at site j given the
competitive set?" — a question isochrones alone cannot answer, because
they don't model destinations competing for the same demand.

Distance source
----------------
`compute_gravity_catchment` takes an optional `distance_fn(origin, dest)`
callable returning kilometers. If omitted, `network_distance_km` is used,
which:
  1. Tries to route over a NetworkX graph built from the `road_network`
     table (see File 12) via `build_road_network_graph` / `load_road_graph`.
  2. Falls back to great-circle (haversine) distance * DETOUR_FACTOR when
     no graph is available or no path exists between the snapped nodes
     (e.g. disconnected extract, missing edges) — this fallback is logged,
     never silent, since it changes the numbers.

Worked example (hand-verifiable, alpha=1, beta=2)
----------------------------------------------------
3 origins, 2 destinations. Distances are contrived round numbers so the
arithmetic can be checked by hand instead of depending on real road data.

    Destinations:
        D1: attractiveness A=1000   (a small retail unit)
        D2: attractiveness A=4000   (a large format store, 4x D1)

    Distances (km), road-network:
                    D1      D2
        O1 (mass=100):   2 km    4 km
        O2 (mass=50):    5 km    2 km
        O3 (mass=200):   1 km    8 km

    Score_ij = A_j^1 / d_ij^2

    O1: score(D1) = 1000 / 2^2  = 1000 / 4  = 250.0
        score(D2) = 4000 / 4^2  = 4000 / 16 = 250.0
        total = 500.0
        P(O1->D1) = 250/500 = 0.5000
        P(O1->D2) = 250/500 = 0.5000

    O2: score(D1) = 1000 / 5^2  = 1000 / 25 = 40.0
        score(D2) = 4000 / 2^2  = 4000 / 4  = 1000.0
        total = 1040.0
        P(O2->D1) = 40/1040   = 0.0385  (approx)
        P(O2->D2) = 1000/1040 = 0.9615  (approx)

    O3: score(D1) = 1000 / 1^2  = 1000.0
        score(D2) = 4000 / 8^2  = 4000 / 64 = 62.5
        total = 1062.5
        P(O3->D1) = 1000/1062.5 = 0.9412  (approx)
        P(O3->D2) = 62.5/1062.5 = 0.0588  (approx)

    Captured demand per destination (sum over origins of mass * P):
        D1 = 100*0.5000 + 50*0.0385 + 200*0.9412
           = 50.0 + 1.923 + 188.24
           = 240.16  (approx)
        D2 = 100*0.5000 + 50*0.9615 + 200*0.0588
           = 50.0 + 48.08 + 11.76
           = 109.84  (approx)

    Sanity check: 240.16 + 109.84 = 350.0 = 100 + 50 + 200 (total mass),
    confirming demand is conserved (split, not duplicated or lost).

Running `python -m ingestion.gravity_catchment --self-test` reproduces
these numbers to 2 decimal places using `distance_fn` overridden with the
fixed distance table above (no road network / network calls required).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import requests
except ImportError:
    requests = None  # network-graph loading from Supabase will be unavailable

logger = logging.getLogger("gravity_catchment")

EARTH_RADIUS_KM = 6371.0088
DETOUR_FACTOR = 1.30  # straight-line -> approximate road distance inflation
MIN_DISTANCE_KM = 0.001  # floor to avoid division by zero at d -> 0


class GravityCatchmentError(RuntimeError):
    """Raised for invalid input or distance-resolution failures."""


# ============================================================================
# Input records
# ============================================================================

@dataclass
class OriginPoint:
    id: Any
    lat: float
    lon: float
    demand: float = 1.0  # total demand mass at this origin (units are caller's choice)


@dataclass
class DestinationSite:
    id: Any
    lat: float
    lon: float
    attractiveness: float  # e.g. floor area (sqm), or any configurable proxy


def _coerce_origin(o: Any) -> OriginPoint:
    if isinstance(o, OriginPoint):
        return o
    if isinstance(o, dict):
        try:
            return OriginPoint(
                id=o["id"],
                lat=float(o["lat"]),
                lon=float(o["lon"]),
                demand=float(o.get("demand", 1.0)),
            )
        except KeyError as exc:
            raise GravityCatchmentError(f"origin_points entry missing required key: {exc}")
    raise GravityCatchmentError(f"Unsupported origin_points entry type: {type(o)!r}")


def _coerce_destination(d: Any) -> DestinationSite:
    if isinstance(d, DestinationSite):
        return d
    if isinstance(d, dict):
        try:
            attractiveness = float(
                d.get("attractiveness", d.get("floor_area_sqm", d.get("floor_area")))
            )
        except (TypeError, ValueError):
            raise GravityCatchmentError(
                f"destination_sites entry {d.get('id')!r} has no numeric "
                f"'attractiveness' (or 'floor_area_sqm' / 'floor_area') value."
            )
        try:
            return DestinationSite(
                id=d["id"],
                lat=float(d["lat"]),
                lon=float(d["lon"]),
                attractiveness=attractiveness,
            )
        except KeyError as exc:
            raise GravityCatchmentError(f"destination_sites entry missing required key: {exc}")
    raise GravityCatchmentError(f"Unsupported destination_sites entry type: {type(d)!r}")


# ============================================================================
# Distance: road network (preferred) with haversine fallback
# ============================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _supabase_headers() -> Dict[str, str]:
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    if not key:
        raise GravityCatchmentError(
            "SUPABASE_SERVICE_ROLE_KEY, SUPABASE_KEY, or SUPABASE_ANON_KEY is not set."
        )
    return {"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"}


def build_road_network_graph(
    rows: Sequence[Dict[str, Any]],
) -> "nx.Graph":
    """
    Build an undirected, distance-weighted NetworkX graph from `road_network`
    rows (see File 12's schema: osm_id, oneway, length_meters, geometry
    [GeoJSON LineString], highway_type). Node ids are (lon, lat) endpoint
    tuples rounded to 6 decimals (~0.1 m) so shared endpoints merge into a
    single routing node. Edge weight is `length_meters` (falls back to a
    haversine estimate from the LineString endpoints if missing).
    """
    if not NETWORKX_AVAILABLE:
        raise GravityCatchmentError(
            "networkx is not installed; install it or pass distance_fn explicitly."
        )
    graph = nx.Graph()
    for row in rows:
        geometry = row.get("geometry") or {}
        coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
        if not coords or len(coords) < 2:
            continue
        length_m = row.get("length_meters")
        if length_m is None:
            total = 0.0
            for a, b in zip(coords[:-1], coords[1:]):
                total += haversine_km(a[1], a[0], b[1], b[0]) * 1000.0
            length_m = total
        u = (round(coords[0][0], 6), round(coords[0][1], 6))
        v = (round(coords[-1][0], 6), round(coords[-1][1], 6))
        if u == v or length_m <= 0:
            continue
        # keep the shortest known edge if OSM ways were split/duplicated
        if graph.has_edge(u, v):
            if length_m < graph[u][v]["weight_m"]:
                graph[u][v]["weight_m"] = length_m
        else:
            graph.add_edge(u, v, weight_m=length_m)
    logger.info(
        "Built road network graph: %d nodes, %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def load_road_network_rows_from_supabase(bbox: Optional[Tuple[float, float, float, float]] = None) -> List[Dict[str, Any]]:
    """Fetch road_network rows (id, geometry, length_meters, ...) from Supabase."""
    if requests is None:
        raise GravityCatchmentError("requests is not installed; cannot query Supabase.")
    supabase_url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    if not supabase_url:
        raise GravityCatchmentError("SUPABASE_URL is not set.")
    params = {"select": "id,length_meters,geometry,highway_type,oneway"}
    resp = requests.get(
        f"{supabase_url}/rest/v1/road_network",
        headers=_supabase_headers(),
        params=params,
        timeout=60,
    )
    if resp.status_code != 200:
        raise GravityCatchmentError(
            f"Failed to read road_network ({resp.status_code}): {resp.text}"
        )
    return resp.json()


def _nearest_node(graph: "nx.Graph", lat: float, lon: float) -> Any:
    """Brute-force nearest-node snap by haversine distance. Fine at pilot-region
    scale (thousands of nodes); swap for a KD-tree/BallTree if the graph grows."""
    if graph.number_of_nodes() == 0:
        raise GravityCatchmentError("Road network graph has no nodes to snap to.")
    best_node, best_dist = None, math.inf
    for node in graph.nodes:
        node_lon, node_lat = node
        dist = haversine_km(lat, lon, node_lat, node_lon)
        if dist < best_dist:
            best_node, best_dist = node, dist
    return best_node


def network_distance_km(
    origin: OriginPoint,
    dest: DestinationSite,
    graph: Optional["nx.Graph"] = None,
) -> float:
    """
    Road-network shortest-path distance in km between an origin and a
    destination. Falls back to haversine * DETOUR_FACTOR (with a logged
    warning) when no graph is supplied, snapping fails, or no path exists
    between the snapped nodes.
    """
    if graph is not None and graph.number_of_nodes() > 0:
        try:
            u = _nearest_node(graph, origin.lat, origin.lon)
            v = _nearest_node(graph, dest.lat, dest.lon)
            length_m = nx.shortest_path_length(graph, u, v, weight="weight_m")
            return max(length_m / 1000.0, MIN_DISTANCE_KM)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            logger.warning(
                "No road-network path from origin %s to destination %s (%s); "
                "falling back to haversine*%.2f.",
                origin.id, dest.id, exc, DETOUR_FACTOR,
            )
    else:
        logger.warning(
            "No road network graph available for origin %s -> destination %s; "
            "falling back to haversine*%.2f.",
            origin.id, dest.id, DETOUR_FACTOR,
        )
    straight = haversine_km(origin.lat, origin.lon, dest.lat, dest.lon)
    return max(straight * DETOUR_FACTOR, MIN_DISTANCE_KM)


# ============================================================================
# Core gravity model
# ============================================================================

DistanceFn = Callable[[OriginPoint, DestinationSite], float]


def compute_gravity_catchment(
    origin_points: Sequence[Any],
    destination_sites: Sequence[Any],
    alpha: float = 1.0,
    beta: float = 2.0,
    distance_fn: Optional[DistanceFn] = None,
    road_graph: Optional["nx.Graph"] = None,
) -> Dict[str, Any]:
    """
    Compute the Huff/gravity-model probabilistic demand capture of each
    destination site from each origin point.

        P(i -> j) = (A_j^alpha / d_ij^beta) / sum_k (A_k^alpha / d_ik^beta)

    Args:
        origin_points: sequence of OriginPoint or dicts with keys
            {id, lat, lon, demand (optional, default 1.0)}.
        destination_sites: sequence of DestinationSite or dicts with keys
            {id, lat, lon, attractiveness (or floor_area_sqm/floor_area)}.
        alpha: attractiveness sensitivity exponent (default 1).
        beta: distance-decay sensitivity exponent (default 2).
        distance_fn: optional callable(origin, dest) -> km. If omitted,
            `network_distance_km` is used against `road_graph` (with
            haversine*DETOUR_FACTOR fallback).
        road_graph: optional pre-built NetworkX graph (see
            `build_road_network_graph`) used by the default distance_fn.
            Ignored if `distance_fn` is supplied.

    Returns:
        {
          "probabilities": {origin_id: {dest_id: P, ...}, ...},
          "captured_demand": {dest_id: total_demand_captured, ...},
          "total_demand": sum of all origin.demand,
        }

    Raises:
        GravityCatchmentError on invalid/missing input, non-positive
        distances/attractiveness, or an origin with zero total score
        across all destinations (e.g. every distance evaluated to
        infinity — should not happen with the haversine fallback, but is
        guarded against explicitly).
    """
    if not origin_points:
        raise GravityCatchmentError("origin_points must not be empty.")
    if not destination_sites:
        raise GravityCatchmentError("destination_sites must not be empty.")
    if beta <= 0:
        raise GravityCatchmentError("beta must be > 0 (distance must penalize, not reward).")

    origins = [_coerce_origin(o) for o in origin_points]
    destinations = [_coerce_destination(d) for d in destination_sites]

    for dest in destinations:
        if dest.attractiveness <= 0:
            raise GravityCatchmentError(
                f"destination {dest.id!r} has non-positive attractiveness "
                f"({dest.attractiveness}); every A_j must be > 0."
            )

    if distance_fn is None:
        distance_fn = lambda o, d: network_distance_km(o, d, graph=road_graph)  # noqa: E731

    probabilities: Dict[str, Dict[str, float]] = {}
    captured_demand: Dict[str, float] = {str(d.id): 0.0 for d in destinations}
    total_demand = 0.0

    for origin in origins:
        total_demand += origin.demand
        scores: Dict[str, float] = {}
        for dest in destinations:
            d_ij = distance_fn(origin, dest)
            if d_ij is None or d_ij <= 0:
                raise GravityCatchmentError(
                    f"distance_fn returned a non-positive distance for "
                    f"origin {origin.id!r} -> destination {dest.id!r}: {d_ij!r}"
                )
            scores[str(dest.id)] = (dest.attractiveness ** alpha) / (d_ij ** beta)

        total_score = sum(scores.values())
        if total_score <= 0:
            raise GravityCatchmentError(
                f"Total gravity score for origin {origin.id!r} is zero; "
                f"cannot compute probabilities (check distances/attractiveness)."
            )

        origin_probs = {dest_id: score / total_score for dest_id, score in scores.items()}
        probabilities[str(origin.id)] = origin_probs

        for dest_id, prob in origin_probs.items():
            captured_demand[dest_id] += origin.demand * prob

    return {
        "probabilities": probabilities,
        "captured_demand": captured_demand,
        "total_demand": total_demand,
    }


# ============================================================================
# Self-test reproducing the worked example in the module docstring
# ============================================================================

def _self_test() -> None:
    origins = [
        {"id": "O1", "lat": 13.05, "lon": 80.25, "demand": 100},
        {"id": "O2", "lat": 13.10, "lon": 80.20, "demand": 50},
        {"id": "O3", "lat": 13.00, "lon": 80.28, "demand": 200},
    ]
    destinations = [
        {"id": "D1", "lat": 13.04, "lon": 80.24, "attractiveness": 1000},
        {"id": "D2", "lat": 13.09, "lon": 80.30, "attractiveness": 4000},
    ]
    fixed_distances_km = {
        ("O1", "D1"): 2.0, ("O1", "D2"): 4.0,
        ("O2", "D1"): 5.0, ("O2", "D2"): 2.0,
        ("O3", "D1"): 1.0, ("O3", "D2"): 8.0,
    }

    def fixed_distance_fn(origin: OriginPoint, dest: DestinationSite) -> float:
        return fixed_distances_km[(str(origin.id), str(dest.id))]

    result = compute_gravity_catchment(origins, destinations, alpha=1, beta=2, distance_fn=fixed_distance_fn)

    expected_probs = {
        "O1": {"D1": 0.5000, "D2": 0.5000},
        "O2": {"D1": 0.0385, "D2": 0.9615},
        "O3": {"D1": 0.9412, "D2": 0.0588},
    }
    expected_captured = {"D1": 240.16, "D2": 109.84}

    print("probabilities:", result["probabilities"])
    print("captured_demand:", result["captured_demand"])
    print("total_demand:", result["total_demand"])

    for oid, dests in expected_probs.items():
        for did, expected in dests.items():
            actual = result["probabilities"][oid][did]
            assert abs(actual - expected) < 0.001, (
                f"P({oid}->{did}) = {actual:.4f}, expected ~{expected:.4f}"
            )
    for did, expected in expected_captured.items():
        actual = result["captured_demand"][did]
        assert abs(actual - expected) < 0.05, (
            f"captured_demand[{did}] = {actual:.2f}, expected ~{expected:.2f}"
        )
    assert abs(result["total_demand"] - 350.0) < 1e-9

    print("Self-test passed: matches the worked example in the module docstring.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gravity/Huff-model probabilistic catchment calculator.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the fixed-distance worked example and verify it against the docstring's expected values.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    args = parse_args()
    if args.self_test:
        _self_test()
    else:
        print("Nothing to do. Pass --self-test to run the worked example, or import compute_gravity_catchment().")


if __name__ == "__main__":
    main()


__all__ = [
    "DestinationSite",
    "GravityCatchmentError",
    "OriginPoint",
    "build_road_network_graph",
    "compute_gravity_catchment",
    "haversine_km",
    "load_road_network_rows_from_supabase",
    "network_distance_km",
]
