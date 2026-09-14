"""
ingestion/generate_catchments.py
================================
Build 10- and 15-minute drive-time catchments, a Huff/gravity catchment,
competitor/own-estate overlap, and a cannibalization score for every
candidate site, then persist the bundle to Supabase.

Candidate source (recommended: table, not a config file)
--------------------------------------------------------
Use a `candidate_sites` table as the source of truth:

  - Coordinates live next to attractiveness, name, and status.
  - Gravity + cannibalization need the same site identity as overlap.
  - Re-runs are keyed by `candidate_sites.id` (no duplicate rows).
  - Analysts can add/disable sites in SQL without editing a repo file.

A JSON config (`--sites-file`) is supported for local/dev seeding: those
rows are upserted into `candidate_sites` first, then processed. Do not
treat a checked-in JSON file as production inventory.

Run once in the Supabase SQL editor
-----------------------------------

    CREATE EXTENSION IF NOT EXISTS postgis;

    CREATE TABLE IF NOT EXISTS candidate_sites (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT,
        lat DOUBLE PRECISION NOT NULL,
        lon DOUBLE PRECISION NOT NULL,
        attractiveness DOUBLE PRECISION NOT NULL DEFAULT 1.0,
        floor_area_sqm DOUBLE PRECISION,
        status TEXT NOT NULL DEFAULT 'active'
            CHECK (status IN ('active', 'paused', 'rejected', 'opened')),
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (lat, lon)
    );

    CREATE INDEX IF NOT EXISTS candidate_sites_status_idx
        ON candidate_sites (status);

    CREATE TABLE IF NOT EXISTS candidate_catchments (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        candidate_site_id UUID NOT NULL
            REFERENCES candidate_sites(id) ON DELETE CASCADE,
        profile TEXT NOT NULL DEFAULT 'driving-car',
        range_minutes_short INTEGER NOT NULL DEFAULT 10,
        range_minutes_long INTEGER NOT NULL DEFAULT 15,
        lat DOUBLE PRECISION NOT NULL,
        lon DOUBLE PRECISION NOT NULL,
        isochrone_10min JSONB,
        isochrone_15min JSONB,
        gravity_catchment JSONB,
        competitor_overlap JSONB,
        cannibalization JSONB,
        cannibalization_score DOUBLE PRECISION,
        competitor_count_10min INTEGER,
        own_estate_count_10min INTEGER,
        competitor_count_15min INTEGER,
        own_estate_count_15min INTEGER,
        computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (
            candidate_site_id,
            profile,
            range_minutes_short,
            range_minutes_long
        )
    );

    CREATE INDEX IF NOT EXISTS candidate_catchments_score_idx
        ON candidate_catchments (cannibalization_score);

Demand origins for gravity / cannibalization are loaded from
`analysis_units` + `demographic_data.population`. Own-estate destinations
come from `own_estate_locations`; competitors from `competitor_locations`.

Usage
-----
    python -m ingestion.generate_catchments
    python -m ingestion.generate_catchments --site-id <uuid>
    python -m ingestion.generate_catchments --sites-file candidates.json
    python -m ingestion.generate_catchments --print-sql
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ingestion.cannibalization import estimate_cannibalization  # type: ignore
    from ingestion.competitor_overlap import compute_overlap  # type: ignore
    from ingestion.gravity_catchment import (  # type: ignore
        GravityCatchmentError,
        build_road_network_graph,
        compute_gravity_catchment,
        load_road_network_rows_from_supabase,
    )
    from ingestion.isochrones import (  # type: ignore
        IsochroneError,
        OrsDailyQuotaExceeded,
        get_isochrone,
        normalize_profile,
    )
else:
    from .cannibalization import estimate_cannibalization
    from .competitor_overlap import compute_overlap
    from .gravity_catchment import (
        GravityCatchmentError,
        build_road_network_graph,
        compute_gravity_catchment,
        load_road_network_rows_from_supabase,
    )
    from .isochrones import (
        IsochroneError,
        OrsDailyQuotaExceeded,
        get_isochrone,
        normalize_profile,
    )

logger = logging.getLogger("generate_catchments")

DEFAULT_PROFILE = "driving-car"
DEFAULT_SHORT_MINUTES = 10
DEFAULT_LONG_MINUTES = 15
PAGE_SIZE = 1000
DEFAULT_COMPETITOR_ATTRACTIVENESS = 1.0
DEFAULT_ESTATE_ATTRACTIVENESS = 1.0

CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS candidate_sites (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    attractiveness DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    floor_area_sqm DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'rejected', 'opened')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (lat, lon)
);

CREATE INDEX IF NOT EXISTS candidate_sites_status_idx
    ON candidate_sites (status);

CREATE TABLE IF NOT EXISTS candidate_catchments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_site_id UUID NOT NULL
        REFERENCES candidate_sites(id) ON DELETE CASCADE,
    profile TEXT NOT NULL DEFAULT 'driving-car',
    range_minutes_short INTEGER NOT NULL DEFAULT 10,
    range_minutes_long INTEGER NOT NULL DEFAULT 15,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    isochrone_10min JSONB,
    isochrone_15min JSONB,
    gravity_catchment JSONB,
    competitor_overlap JSONB,
    cannibalization JSONB,
    cannibalization_score DOUBLE PRECISION,
    competitor_count_10min INTEGER,
    own_estate_count_10min INTEGER,
    competitor_count_15min INTEGER,
    own_estate_count_15min INTEGER,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (
        candidate_site_id,
        profile,
        range_minutes_short,
        range_minutes_long
    )
);

CREATE INDEX IF NOT EXISTS candidate_catchments_score_idx
    ON candidate_catchments (cannibalization_score);
""".strip()


class CatchmentPipelineError(RuntimeError):
    """Raised for config, I/O, or persistence failures in this script."""


def _require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise CatchmentPipelineError(f"{name} is not set.")
    return value


def _supabase_url() -> str:
    return _require_env("SUPABASE_URL").rstrip("/")


def _supabase_key() -> str:
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    if not key:
        raise CatchmentPipelineError(
            "SUPABASE_SERVICE_ROLE_KEY, SUPABASE_KEY, or SUPABASE_ANON_KEY is not set."
        )
    return key


def _supabase_headers(*, prefer: Optional[str] = None) -> Dict[str, str]:
    key = _supabase_key()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _rest(path: str) -> str:
    return f"{_supabase_url()}/rest/v1/{path.lstrip('/')}"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _get_paginated(table: str, params: Dict[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        query = dict(params)
        query["limit"] = str(PAGE_SIZE)
        query["offset"] = str(offset)
        resp = requests.get(
            _rest(table),
            headers=_supabase_headers(),
            params=query,
            timeout=60,
        )
        if resp.status_code != 200:
            raise CatchmentPipelineError(
                f"Failed to read {table} ({resp.status_code}): {resp.text}"
            )
        batch = resp.json() or []
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def _extract_lat_lon(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for lat_key, lon_key in (
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("y", "x"),
    ):
        if row.get(lat_key) is not None and row.get(lon_key) is not None:
            return float(row[lat_key]), float(row[lon_key])

    geom = row.get("geometry") or row.get("centroid")
    if isinstance(geom, str):
        text = geom.strip()
        if text.startswith("{"):
            try:
                geom = json.loads(text)
            except json.JSONDecodeError:
                geom = None
        elif "POINT" in text.upper():
            inner = text[text.upper().find("POINT") :]
            inner = inner[inner.find("(") + 1 : inner.rfind(")")]
            parts = inner.replace(",", " ").split()
            if len(parts) >= 2:
                return float(parts[1]), float(parts[0])
            geom = None

    if isinstance(geom, dict):
        if geom.get("type") == "Feature":
            geom = geom.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Point" and coords and len(coords) >= 2:
            return float(coords[1]), float(coords[0])
        if gtype in ("Polygon", "MultiPolygon") and coords:
            # Fallback: first vertex. Prefer a dedicated centroid column.
            ring = coords[0][0] if gtype == "MultiPolygon" else coords[0]
            if ring:
                return float(ring[0][1]), float(ring[0][0])
    return None


def _attractiveness(row: Dict[str, Any], default: float) -> float:
    for key in (
        "attractiveness",
        "floor_area_sqm",
        "floor_area",
        "current_trading_performance",
    ):
        value = row.get(key)
        if value is not None:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    return default


def load_candidate_sites(site_id: Optional[str] = None) -> List[Dict[str, Any]]:
    params: Dict[str, str] = {
        "select": "id,name,lat,lon,attractiveness,floor_area_sqm,status,metadata",
        "status": "eq.active",
        "order": "id.asc",
    }
    if site_id:
        params["id"] = f"eq.{site_id}"
        params.pop("status", None)
    rows = _get_paginated("candidate_sites", params)
    sites = []
    for row in rows:
        coords = _extract_lat_lon(row)
        if coords is None:
            logger.warning("Skipping candidate_sites row %s: missing lat/lon", row.get("id"))
            continue
        lat, lon = coords
        sites.append(
            {
                "id": row["id"],
                "name": row.get("name"),
                "lat": lat,
                "lon": lon,
                "attractiveness": _attractiveness(row, 1.0),
            }
        )
    return sites


def upsert_candidate_sites_from_file(path: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("sites", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise CatchmentPipelineError(f"{path} contains no candidate sites.")

    now = datetime.now(timezone.utc).isoformat()
    body = []
    for item in records:
        lat = float(item["lat"])
        lon = float(item["lon"])
        rec: Dict[str, Any] = {
            "name": item.get("name"),
            "lat": lat,
            "lon": lon,
            "attractiveness": float(
                item.get("attractiveness")
                or item.get("floor_area_sqm")
                or 1.0
            ),
            "floor_area_sqm": item.get("floor_area_sqm"),
            "status": item.get("status") or "active",
            "metadata": item.get("metadata") or {},
            "updated_at": now,
        }
        if item.get("id"):
            rec["id"] = item["id"]
        body.append(rec)

    # Unique (lat, lon) makes file re-imports idempotent even without ids.
    resp = requests.post(
        _rest("candidate_sites"),
        headers=_supabase_headers(
            prefer="resolution=merge-duplicates,return=minimal"
        ),
        json=body,
        timeout=60,
    )
    if resp.status_code not in (200, 201, 204):
        raise CatchmentPipelineError(
            f"Failed to upsert candidate_sites from file ({resp.status_code}): {resp.text}"
        )


def load_origin_points() -> List[Dict[str, Any]]:
    units = _get_paginated(
        "analysis_units",
        {"select": "id,lat,lon,latitude,longitude,centroid,geometry", "order": "id.asc"},
    )
    demo = _get_paginated(
        "demographic_data",
        {"select": "unit_id,population", "order": "unit_id.asc"},
    )
    pop_by_unit = {
        str(row.get("unit_id")): float(row.get("population") or 0.0) for row in demo
    }
    origins: List[Dict[str, Any]] = []
    for unit in units:
        coords = _extract_lat_lon(unit)
        if coords is None:
            continue
        unit_id = str(unit["id"])
        demand = pop_by_unit.get(unit_id, 0.0)
        if demand <= 0:
            demand = 1.0
        origins.append(
            {"id": unit_id, "lat": coords[0], "lon": coords[1], "demand": demand}
        )
    return origins


def load_own_estate_destinations() -> List[Dict[str, Any]]:
    rows = _get_paginated(
        "own_estate_locations",
        {
            "select": "id,name,geometry,latitude,longitude,lat,lon,current_trading_performance,floor_area_sqm,attractiveness",
            "order": "id.asc",
        },
    )
    destinations = []
    for row in rows:
        coords = _extract_lat_lon(row)
        if coords is None:
            logger.warning("Skipping own_estate_locations %s: missing coordinates", row.get("id"))
            continue
        destinations.append(
            {
                "id": f"estate:{row['id']}",
                "lat": coords[0],
                "lon": coords[1],
                "attractiveness": _attractiveness(row, DEFAULT_ESTATE_ATTRACTIVENESS),
            }
        )
    return destinations


def load_competitor_destinations() -> List[Dict[str, Any]]:
    rows = _get_paginated(
        "competitor_locations",
        {
            "select": "id,name,category,geometry,latitude,longitude,lat,lon,attractiveness,floor_area_sqm",
            "order": "id.asc",
        },
    )
    destinations = []
    for row in rows:
        coords = _extract_lat_lon(row)
        if coords is None:
            continue
        destinations.append(
            {
                "id": f"competitor:{row['id']}",
                "lat": coords[0],
                "lon": coords[1],
                "attractiveness": _attractiveness(row, DEFAULT_COMPETITOR_ATTRACTIVENESS),
            }
        )
    return destinations


def _isochrone_geometry(result: Dict[str, Any]) -> Dict[str, Any]:
    geom = result.get("geometry")
    if isinstance(geom, dict) and geom.get("type") in ("Polygon", "MultiPolygon"):
        return geom
    raise CatchmentPipelineError(
        f"get_isochrone() did not return a polygon geometry: {type(geom)!r}"
    )


def _overlap_counts(overlap: Dict[str, Any]) -> Tuple[int, int]:
    summary = overlap.get("summary") or {}
    competitors = overlap.get("competitors") or {}
    estate = overlap.get("own_estate") or {}
    competitor_count = int(
        summary.get("competitor_count", competitors.get("count", 0)) or 0
    )
    own_count = int(
        summary.get("own_estate_count", estate.get("count", 0)) or 0
    )
    return competitor_count, own_count


def _load_road_graph() -> Any:
    try:
        rows = load_road_network_rows_from_supabase()
        if not rows:
            logger.warning("road_network is empty; gravity distances will use haversine fallback.")
            return None
        return build_road_network_graph(rows)
    except Exception as exc:
        logger.warning("Could not load road network graph (%s); using haversine fallback.", exc)
        return None


def process_site(
    site: Dict[str, Any],
    *,
    profile: str,
    short_minutes: int,
    long_minutes: int,
    origin_points: Sequence[Dict[str, Any]],
    own_estate: Sequence[Dict[str, Any]],
    competitors: Sequence[Dict[str, Any]],
    road_graph: Any,
    alpha: float,
    beta: float,
) -> Dict[str, Any]:
    site_id = site["id"]
    lat = float(site["lat"])
    lon = float(site["lon"])
    logger.info("Processing candidate %s (%.5f, %.5f)", site_id, lat, lon)

    iso_short = get_isochrone(lat, lon, profile, short_minutes)
    iso_long = get_isochrone(lat, lon, profile, long_minutes)
    poly_short = _isochrone_geometry(iso_short)
    poly_long = _isochrone_geometry(iso_long)

    overlap_short = compute_overlap(poly_short)
    overlap_long = compute_overlap(poly_long)
    c10, o10 = _overlap_counts(overlap_short)
    c15, o15 = _overlap_counts(overlap_long)

    gravity_payload: Optional[Dict[str, Any]] = None
    if origin_points:
        destinations: List[Dict[str, Any]] = [
            {
                "id": str(site_id),
                "lat": lat,
                "lon": lon,
                "attractiveness": float(site["attractiveness"]),
            }
        ]
        destinations.extend(own_estate)
        destinations.extend(competitors)
        try:
            gravity = compute_gravity_catchment(
                origin_points=origin_points,
                destination_sites=destinations,
                alpha=alpha,
                beta=beta,
                road_graph=road_graph,
            )
            captured = gravity.get("captured_demand") or {}
            gravity_payload = {
                "captured_demand": captured,
                "candidate_captured_demand": captured.get(str(site_id)),
                "total_demand": gravity.get("total_demand"),
                "origin_count": len(origin_points),
                "destination_count": len(destinations),
                "alpha": alpha,
                "beta": beta,
                # Full P(i->j) matrix can be large; keep per-origin share for this site only.
                "candidate_origin_probabilities": {
                    oid: probs.get(str(site_id))
                    for oid, probs in (gravity.get("probabilities") or {}).items()
                    if isinstance(probs, dict)
                },
            }
        except GravityCatchmentError as exc:
            logger.error("Gravity catchment failed for %s: %s", site_id, exc)
            gravity_payload = {"error": str(exc)}
    else:
        logger.warning("No origin points; skipping gravity catchment for %s", site_id)

    cannibalization_payload: Optional[Dict[str, Any]] = None
    cannibalization_score: Optional[float] = None
    if origin_points and own_estate:
        try:
            result = estimate_cannibalization(
                new_site={
                    "id": str(site_id),
                    "lat": lat,
                    "lon": lon,
                    "attractiveness": float(site["attractiveness"]),
                },
                own_estate_sites=own_estate,
                origin_points=origin_points,
                alpha=alpha,
                beta=beta,
                road_graph=road_graph,
            )
            cannibalization_payload = result.to_dict()
            cannibalization_score = float(result.cannibalization_score)
        except GravityCatchmentError as exc:
            logger.error("Cannibalization failed for %s: %s", site_id, exc)
            cannibalization_payload = {"error": str(exc)}
    else:
        logger.warning(
            "Skipping cannibalization for %s (need origin points and own-estate stores).",
            site_id,
        )

    return {
        "candidate_site_id": site_id,
        "profile": profile,
        "range_minutes_short": short_minutes,
        "range_minutes_long": long_minutes,
        "lat": lat,
        "lon": lon,
        "isochrone_10min": poly_short,
        "isochrone_15min": poly_long,
        "gravity_catchment": _jsonable(gravity_payload),
        "competitor_overlap": _jsonable(
            {
                f"{short_minutes}min": overlap_short,
                f"{long_minutes}min": overlap_long,
            }
        ),
        "cannibalization": _jsonable(cannibalization_payload),
        "cannibalization_score": cannibalization_score,
        "competitor_count_10min": c10,
        "own_estate_count_10min": o10,
        "competitor_count_15min": c15,
        "own_estate_count_15min": o15,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def upsert_catchment(record: Dict[str, Any]) -> None:
    resp = requests.post(
        _rest("candidate_catchments"),
        headers=_supabase_headers(
            prefer="resolution=merge-duplicates,return=minimal"
        ),
        json=record,
        timeout=60,
    )
    if resp.status_code not in (200, 201, 204):
        raise CatchmentPipelineError(
            f"Failed to upsert candidate_catchments ({resp.status_code}): {resp.text}"
        )


def run(
    *,
    site_id: Optional[str] = None,
    sites_file: Optional[str] = None,
    profile: str = DEFAULT_PROFILE,
    short_minutes: int = DEFAULT_SHORT_MINUTES,
    long_minutes: int = DEFAULT_LONG_MINUTES,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> int:
    ors_profile = normalize_profile(profile)
    if sites_file:
        upsert_candidate_sites_from_file(sites_file)

    sites = load_candidate_sites(site_id=site_id)
    if not sites:
        raise CatchmentPipelineError(
            "No candidate sites to process. Insert rows into candidate_sites "
            "(status='active') or pass --sites-file."
        )

    origin_points = load_origin_points()
    own_estate = load_own_estate_destinations()
    competitors = load_competitor_destinations()
    road_graph = _load_road_graph()

    logger.info(
        "Loaded %d candidate(s), %d origin(s), %d own-estate, %d competitor(s).",
        len(sites),
        len(origin_points),
        len(own_estate),
        len(competitors),
    )

    processed = 0
    for site in sites:
        try:
            record = process_site(
                site,
                profile=ors_profile,
                short_minutes=short_minutes,
                long_minutes=long_minutes,
                origin_points=origin_points,
                own_estate=own_estate,
                competitors=competitors,
                road_graph=road_graph,
                alpha=alpha,
                beta=beta,
            )
            upsert_catchment(record)
            processed += 1
        except OrsDailyQuotaExceeded:
            logger.error("ORS daily quota reached; stopping. Re-run next UTC day.")
            raise
        except (IsochroneError, CatchmentPipelineError):
            logger.exception("Failed processing candidate %s", site.get("id"))
            raise
    return processed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate drive-time, gravity, overlap, and cannibalization catchments."
    )
    parser.add_argument("--site-id", help="Process a single candidate_sites.id")
    parser.add_argument(
        "--sites-file",
        help="Optional JSON file of sites to upsert into candidate_sites before processing.",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--short-minutes", type=int, default=DEFAULT_SHORT_MINUTES)
    parser.add_argument("--long-minutes", type=int, default=DEFAULT_LONG_MINUTES)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument(
        "--print-sql",
        action="store_true",
        help="Print CREATE TABLE statements and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    if args.print_sql:
        print(CREATE_TABLE_SQL)
        return 0
    count = run(
        site_id=args.site_id,
        sites_file=args.sites_file,
        profile=args.profile,
        short_minutes=args.short_minutes,
        long_minutes=args.long_minutes,
        alpha=args.alpha,
        beta=args.beta,
    )
    logger.info("Upserted %d candidate_catchments row(s).", count)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CatchmentPipelineError, IsochroneError, OrsDailyQuotaExceeded) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc