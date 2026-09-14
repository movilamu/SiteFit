"""
ingestion/competitor_overlap.py
===============================
Spatial overlap analysis between a proposed site's catchment polygon
and existing locations (both competitors and own store estate).

Features:
1. PostGIS spatial querying using ST_Contains or ST_Intersects against
   `competitor_locations` and `own_estate_locations`.
2. Segregated return format containing counts and matched location records
   for both competitor and own estate locations.
3. Area-weighted trade-area overlap percentage: computes the proportion
   of a location's typical trade area (modeled as a circular buffer
   proxy when empirical trade areas are unavailable) that intersects
   the site catchment polygon:
       overlap_pct = (Area(Catchment ∩ TradeArea) / Area(TradeArea)) * 100
4. Comprehensive logging and custom error handling.
5. Flexible execution via direct PostGIS connection (psycopg2 / psycopg /
   DATABASE_URL) or Supabase PostgREST RPC (`compute_catchment_overlap`).

PostGIS RPC Migration (Optional for Supabase PostgREST execution):
------------------------------------------------------------------
To execute via Supabase client RPC instead of direct PostgreSQL connection,
run the SQL stored in `POSTGIS_RPC_MIGRATION_SQL` in your Supabase SQL editor.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Optional database driver imports
try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None

try:
    import psycopg
except ImportError:
    psycopg = None

try:
    import requests
except ImportError:
    requests = None

try:
    from supabase import Client, create_client
except ImportError:
    Client = None
    create_client = None

# Configure logger
logger = logging.getLogger("competitor_overlap")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


# --- Configuration & Constants ---

DEFAULT_TRADE_AREA_RADIUS_M: float = 1000.0  # 1 km default retail trade area proxy
ALLOWED_PREDICATES = frozenset({"ST_Contains", "ST_Intersects"})

POSTGIS_RPC_MIGRATION_SQL = """
-- Supabase / PostGIS RPC Migration Function
CREATE OR REPLACE FUNCTION compute_catchment_overlap(
    catchment_geojson JSONB,
    trade_area_radius_m DOUBLE PRECISION DEFAULT 1000.0,
    spatial_predicate TEXT DEFAULT 'ST_Intersects'
)
RETURNS TABLE (
    location_type TEXT,
    id TEXT,
    name TEXT,
    category TEXT,
    current_trading_performance NUMERIC,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    is_contained BOOLEAN,
    is_intersecting BOOLEAN,
    distance_to_centroid_m DOUBLE PRECISION,
    overlap_percentage NUMERIC
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    catchment_geom GEOMETRY;
BEGIN
    catchment_geom := ST_SetSRID(ST_GeomFromGeoJSON(catchment_geojson::text), 4326);

    IF catchment_geom IS NULL OR ST_IsEmpty(catchment_geom) THEN
        RAISE EXCEPTION 'Invalid or empty catchment geometry.';
    END IF;

    RETURN QUERY
    WITH site_catchment AS (
        SELECT catchment_geom AS geom
    ),
    competitors AS (
        SELECT
            'competitor'::TEXT AS location_type,
            c.id::TEXT AS id,
            c.name::TEXT AS name,
            c.category::TEXT AS category,
            NULL::NUMERIC AS current_trading_performance,
            ST_Y(c.geometry::geometry) AS latitude,
            ST_X(c.geometry::geometry) AS longitude,
            ST_Contains(sc.geom, c.geometry) AS is_contained,
            ST_Intersects(sc.geom, c.geometry) AS is_intersecting,
            ST_Distance(c.geometry::geography, ST_Centroid(sc.geom)::geography) AS distance_to_centroid_m,
            ROUND(
                (
                    ST_Area(
                        ST_Intersection(
                            ST_Buffer(c.geometry::geography, trade_area_radius_m)::geometry,
                            sc.geom
                        )::geography
                    ) / NULLIF(
                        ST_Area(ST_Buffer(c.geometry::geography, trade_area_radius_m)::geography),
                        0
                    ) * 100.0
                )::NUMERIC,
                2
            ) AS overlap_percentage
        FROM competitor_locations c, site_catchment sc
        WHERE (
            (spatial_predicate = 'ST_Contains' AND ST_Contains(sc.geom, c.geometry))
            OR
            (spatial_predicate = 'ST_Intersects' AND ST_Intersects(sc.geom, c.geometry))
        )
    ),
    own_estate AS (
        SELECT
            'own_estate'::TEXT AS location_type,
            o.id::TEXT AS id,
            o.name::TEXT AS name,
            NULL::TEXT AS category,
            o.current_trading_performance::NUMERIC AS current_trading_performance,
            ST_Y(o.geometry::geometry) AS latitude,
            ST_X(o.geometry::geometry) AS longitude,
            ST_Contains(sc.geom, o.geometry) AS is_contained,
            ST_Intersects(sc.geom, o.geometry) AS is_intersecting,
            ST_Distance(o.geometry::geography, ST_Centroid(sc.geom)::geography) AS distance_to_centroid_m,
            ROUND(
                (
                    ST_Area(
                        ST_Intersection(
                            ST_Buffer(o.geometry::geography, trade_area_radius_m)::geometry,
                            sc.geom
                        )::geography
                    ) / NULLIF(
                        ST_Area(ST_Buffer(o.geometry::geography, trade_area_radius_m)::geography),
                        0
                    ) * 100.0
                )::NUMERIC,
                2
            ) AS overlap_percentage
        FROM own_estate_locations o, site_catchment sc
        WHERE (
            (spatial_predicate = 'ST_Contains' AND ST_Contains(sc.geom, o.geometry))
            OR
            (spatial_predicate = 'ST_Intersects' AND ST_Intersects(sc.geom, o.geometry))
        )
    )
    SELECT * FROM competitors
    UNION ALL
    SELECT * FROM own_estate
    ORDER BY distance_to_centroid_m ASC;
END;
$$;
"""


# --- Custom Exceptions ---

class CompetitorOverlapError(RuntimeError):
    """Base exception for all competitor overlap computation errors."""


class InvalidCatchmentPolygonError(CompetitorOverlapError):
    """Raised when the input site catchment polygon is invalid or malformed."""


class DatabaseConnectionError(CompetitorOverlapError):
    """Raised when connecting to PostgreSQL / Supabase fails."""


class DatabaseQueryError(CompetitorOverlapError):
    """Raised when a spatial SQL query execution fails."""


# --- Geometry Validation & Normalization ---

def normalize_catchment_geometry(
    site_catchment_polygon: Union[Dict[str, Any], str, Any]
) -> Tuple[str, str]:
    """
    Validate and convert the input catchment polygon into GeoJSON and WKT representations.

    Supports:
    - GeoJSON Geometry dict: {"type": "Polygon"|"MultiPolygon", "coordinates": [...]}
    - GeoJSON Feature dict: {"type": "Feature", "geometry": {...}}
    - GeoJSON FeatureCollection dict: extracts first Polygon/MultiPolygon
    - JSON string representation of GeoJSON
    - WKT string: "POLYGON ((...))" or "SRID=4326;POLYGON ((...))"
    - Shapely geometry object (with __geo_interface__ or .wkt)

    Returns:
        Tuple of (geojson_str, wkt_str)
    """
    if site_catchment_polygon is None:
        raise InvalidCatchmentPolygonError("site_catchment_polygon cannot be None.")

    geom_dict: Optional[Dict[str, Any]] = None
    wkt_str: Optional[str] = None

    # Handle string input
    if isinstance(site_catchment_polygon, str):
        cleaned = site_catchment_polygon.strip()
        if not cleaned:
            raise InvalidCatchmentPolygonError("Catchment polygon string is empty.")

        if cleaned.startswith("{"):
            try:
                geom_dict = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise InvalidCatchmentPolygonError(
                    f"Malformed GeoJSON string: {exc}"
                ) from exc
        elif (
            cleaned.upper().startswith("POLYGON")
            or cleaned.upper().startswith("MULTIPOLYGON")
            or "POLYGON" in cleaned.upper()
        ):
            wkt_str = cleaned
            if not wkt_str.upper().startswith("SRID="):
                wkt_str = f"SRID=4326;{wkt_str}"

    # Handle dict input
    elif isinstance(site_catchment_polygon, dict):
        geom_dict = site_catchment_polygon

    # Handle Shapely or other geometry objects
    elif hasattr(site_catchment_polygon, "__geo_interface__"):
        geom_dict = getattr(site_catchment_polygon, "__geo_interface__")
    elif hasattr(site_catchment_polygon, "wkt"):
        wkt_str = f"SRID=4326;{getattr(site_catchment_polygon, 'wkt')}"
    else:
        raise InvalidCatchmentPolygonError(
            f"Unsupported polygon type: {type(site_catchment_polygon).__name__}. "
            "Expected GeoJSON dict, WKT string, or Shapely polygon."
        )

    # Unwrap Feature / FeatureCollection if dict
    if geom_dict is not None:
        if geom_dict.get("type") == "Feature":
            geom_dict = geom_dict.get("geometry") or {}
        elif geom_dict.get("type") == "FeatureCollection":
            features = geom_dict.get("features") or []
            if not features:
                raise InvalidCatchmentPolygonError("GeoJSON FeatureCollection contains no features.")
            geom_dict = features[0].get("geometry") or {}

        gtype = geom_dict.get("type")
        coords = geom_dict.get("coordinates")
        if gtype not in ("Polygon", "MultiPolygon") or not coords:
            raise InvalidCatchmentPolygonError(
                f"Catchment geometry must be 'Polygon' or 'MultiPolygon', got {gtype!r}."
            )

        geojson_str = json.dumps(geom_dict)
        if not wkt_str:
            wkt_str = _geojson_to_wkt(geom_dict)
        return geojson_str, wkt_str

    if wkt_str is not None:
        return _wkt_to_geojson_str(wkt_str), wkt_str

    raise InvalidCatchmentPolygonError("Failed to normalize catchment polygon geometry.")


def _geojson_to_wkt(geom: Dict[str, Any]) -> str:
    """Simple GeoJSON to EWKT serializer for Polygons."""
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])

    if gtype == "Polygon":
        rings = []
        for ring in coords:
            pts = ", ".join(f"{pt[0]} {pt[1]}" for pt in ring)
            rings.append(f"({pts})")
        return f"SRID=4326;POLYGON({', '.join(rings)})"

    if gtype == "MultiPolygon":
        polys = []
        for poly in coords:
            rings = []
            for ring in poly:
                pts = ", ".join(f"{pt[0]} {pt[1]}" for pt in ring)
                rings.append(f"({pts})")
            polys.append(f"({', '.join(rings)})")
        return f"SRID=4326;MULTIPOLYGON({', '.join(polys)})"

    raise InvalidCatchmentPolygonError(f"Unsupported geometry type for WKT conversion: {gtype}")


def _wkt_to_geojson_str(wkt: str) -> str:
    """Convert basic WKT to GeoJSON string."""
    cleaned = wkt.split(";", 1)[-1].strip()
    upper = cleaned.upper()

    if upper.startswith("POLYGON"):
        raw = cleaned[cleaned.find("(") :].strip()
        raw = raw.strip("()")
        rings_raw = [r.strip("()") for r in raw.split("),")]
        rings_coords = []
        for ring in rings_raw:
            pts = []
            for pt in ring.split(","):
                coords = [float(v) for v in pt.strip().split()]
                pts.append(coords[:2])
            rings_coords.append(pts)
        return json.dumps({"type": "Polygon", "coordinates": rings_coords})

    return json.dumps({"wkt": wkt})


# --- Database Execution Helpers ---

def _get_database_url() -> Optional[str]:
    return (
        os.environ.get("DATABASE_URL")
        or os.environ.get("POSTGRES_URL")
        or os.environ.get("SUPABASE_DB_URL")
    )


def _execute_direct_sql(
    geojson_str: str,
    trade_area_radius_m: float,
    predicate: str,
    db_url: str,
) -> List[Dict[str, Any]]:
    """Execute PostGIS spatial query directly via psycopg2 or psycopg connection."""
    where_predicate = (
        "ST_Contains(sc.geom, loc.geometry)"
        if predicate == "ST_Contains"
        else "ST_Intersects(sc.geom, loc.geometry)"
    )

    query = f"""
    WITH site_catchment AS (
        SELECT ST_SetSRID(ST_GeomFromGeoJSON(%(geojson)s), 4326) AS geom
    ),
    competitors AS (
        SELECT
            'competitor' AS location_type,
            loc.id::text AS id,
            loc.name,
            loc.category,
            NULL::numeric AS current_trading_performance,
            ST_Y(loc.geometry::geometry) AS latitude,
            ST_X(loc.geometry::geometry) AS longitude,
            ST_Contains(sc.geom, loc.geometry) AS is_contained,
            ST_Intersects(sc.geom, loc.geometry) AS is_intersecting,
            ST_Distance(loc.geometry::geography, ST_Centroid(sc.geom)::geography) AS distance_to_centroid_m,
            ROUND(
                (
                    ST_Area(
                        ST_Intersection(
                            ST_Buffer(loc.geometry::geography, %(radius)s)::geometry,
                            sc.geom
                        )::geography
                    ) / NULLIF(
                        ST_Area(ST_Buffer(loc.geometry::geography, %(radius)s)::geography),
                        0
                    ) * 100.0
                )::numeric,
                2
            ) AS overlap_percentage
        FROM competitor_locations loc, site_catchment sc
        WHERE {where_predicate}
    ),
    own_estate AS (
        SELECT
            'own_estate' AS location_type,
            loc.id::text AS id,
            loc.name,
            NULL::text AS category,
            loc.current_trading_performance,
            ST_Y(loc.geometry::geometry) AS latitude,
            ST_X(loc.geometry::geometry) AS longitude,
            ST_Contains(sc.geom, loc.geometry) AS is_contained,
            ST_Intersects(sc.geom, loc.geometry) AS is_intersecting,
            ST_Distance(loc.geometry::geography, ST_Centroid(sc.geom)::geography) AS distance_to_centroid_m,
            ROUND(
                (
                    ST_Area(
                        ST_Intersection(
                            ST_Buffer(loc.geometry::geography, %(radius)s)::geometry,
                            sc.geom
                        )::geography
                    ) / NULLIF(
                        ST_Area(ST_Buffer(loc.geometry::geography, %(radius)s)::geography),
                        0
                    ) * 100.0
                )::numeric,
                2
            ) AS overlap_percentage
        FROM own_estate_locations loc, site_catchment sc
        WHERE {where_predicate}
    )
    SELECT * FROM competitors
    UNION ALL
    SELECT * FROM own_estate
    ORDER BY distance_to_centroid_m ASC;
    """

    params = {"geojson": geojson_str, "radius": trade_area_radius_m}

    if psycopg2 is not None:
        try:
            with psycopg2.connect(db_url) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(query, params)
                    return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            raise DatabaseQueryError(f"psycopg2 query failed: {exc}") from exc

    if psycopg is not None:
        try:
            with psycopg.connect(db_url) as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute(query, params)
                    return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            raise DatabaseQueryError(f"psycopg query failed: {exc}") from exc

    raise DatabaseConnectionError(
        "Neither psycopg2 nor psycopg is installed to execute raw SQL against DATABASE_URL. "
        "Install via: pip install psycopg2-binary"
    )


def _execute_supabase_rpc(
    geojson_str: str,
    trade_area_radius_m: float,
    predicate: str,
) -> List[Dict[str, Any]]:
    """Execute PostGIS query via Supabase RPC (PostgREST API)."""
    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    supabase_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip()

    if not supabase_url or not supabase_key:
        raise DatabaseConnectionError(
            "Missing SUPABASE_URL and/or SUPABASE_KEY environment variables."
        )

    rpc_params = {
        "catchment_geojson": json.loads(geojson_str),
        "trade_area_radius_m": float(trade_area_radius_m),
        "spatial_predicate": predicate,
    }

    # Attempt via supabase-py client if installed
    if create_client is not None:
        try:
            client: Client = create_client(supabase_url, supabase_key)
            resp = client.rpc("compute_catchment_overlap", rpc_params).execute()
            if hasattr(resp, "data") and resp.data is not None:
                return resp.data
        except Exception as exc:
            logger.warning(
                "Supabase SDK RPC failed (%s); falling back to direct REST request.", exc
            )

    # Fallback to direct HTTP REST request via requests
    if requests is not None:
        url = f"{supabase_url}/rest/v1/rpc/compute_catchment_overlap"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            resp = requests.post(url, headers=headers, json=rpc_params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            raise DatabaseQueryError(
                f"Supabase REST RPC returned {resp.status_code}: {resp.text}"
            )
        except requests.RequestException as exc:
            raise DatabaseConnectionError(f"HTTP request to Supabase failed: {exc}") from exc

    raise DatabaseConnectionError(
        "Neither 'supabase' SDK nor 'requests' is installed. "
        "Install with: pip install requests supabase"
    )


# --- Core Analysis Function ---

def compute_overlap(
    site_catchment_polygon: Union[Dict[str, Any], str, Any],
    trade_area_radius_m: float = DEFAULT_TRADE_AREA_RADIUS_M,
    predicate: str = "ST_Intersects",
    connection: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compute competitor and own-estate overlap inside a site catchment polygon.

    1. Executes a PostGIS spatial query (`ST_Contains` or `ST_Intersects`)
       to find all `competitor_locations` and `own_estate_locations` falling
       inside the given catchment polygon.
    2. Returns counts and segregated lists of matched locations (competitors vs.
       own estate).
    3. Calculates an area-weighted overlap percentage for each matched location:
       models the location's typical trade area as a circular buffer
       (default 1,000m) and computes the proportion falling inside the catchment:
           (Area(Catchment ∩ TradeArea) / Area(TradeArea)) * 100
       (Note: standard simplification proxy when empirical store isochrones
       are unavailable).

    Args:
        site_catchment_polygon: GeoJSON dict, WKT string, or Shapely polygon.
        trade_area_radius_m: Radius in meters for the location's typical trade area
            buffer (default: 1000m). Must be positive.
        predicate: Spatial query predicate. Either 'ST_Intersects' (default)
            or 'ST_Contains'.
        connection: Optional existing psycopg2 / psycopg database connection.
            If None, connects using DATABASE_URL or Supabase RPC.

    Returns:
        A structured dictionary:
        {
            "competitors": {
                "count": int,
                "locations": [
                    {
                        "id": str,
                        "name": str,
                        "category": str,
                        "latitude": float,
                        "longitude": float,
                        "overlap_percentage": float,
                        "is_contained": bool,
                        "is_intersecting": bool,
                        "distance_to_centroid_m": float
                    },
                    ...
                ]
            },
            "own_estate": {
                "count": int,
                "locations": [
                    {
                        "id": str,
                        "name": str,
                        "current_trading_performance": Optional[float],
                        "latitude": float,
                        "longitude": float,
                        "overlap_percentage": float,
                        "is_contained": bool,
                        "is_intersecting": bool,
                        "distance_to_centroid_m": float
                    },
                    ...
                ]
            },
            "summary": {
                "competitor_count": int,
                "own_estate_count": int,
                "total_matched_locations": int,
                "avg_competitor_overlap_pct": float,
                "avg_own_estate_overlap_pct": float,
                "spatial_predicate": str,
                "trade_area_radius_m": float,
                "trade_area_simplification_note": str
            }
        }

    Raises:
        InvalidCatchmentPolygonError: If the input polygon cannot be parsed or validated.
        ValueError: If `trade_area_radius_m` <= 0 or `predicate` is unrecognized.
        DatabaseConnectionError: If no database credentials or connection can be established.
        DatabaseQueryError: If the PostGIS query execution fails.
    """
    logger.info("Initiating competitor and estate catchment overlap computation...")

    # Validate parameters
    if trade_area_radius_m <= 0:
        raise ValueError(
            f"trade_area_radius_m must be a positive number, got {trade_area_radius_m}"
        )

    norm_predicate = predicate.strip()
    if norm_predicate not in ALLOWED_PREDICATES:
        raise ValueError(
            f"Unsupported spatial predicate: {predicate!r}. "
            f"Allowed: {', '.join(sorted(ALLOWED_PREDICATES))}"
        )

    # Normalize catchment polygon
    try:
        geojson_str, wkt_str = normalize_catchment_geometry(site_catchment_polygon)
        logger.debug("Normalized catchment polygon to GeoJSON and WKT successfully.")
    except Exception as exc:
        logger.error("Failed to parse site catchment polygon: %s", exc)
        raise

    # Execute spatial query
    rows: List[Dict[str, Any]] = []
    db_url = _get_database_url()

    try:
        if connection is not None:
            logger.info("Executing PostGIS query via provided connection...")
            with connection.cursor() as cur:
                where_pred = (
                    "ST_Contains(sc.geom, loc.geometry)"
                    if norm_predicate == "ST_Contains"
                    else "ST_Intersects(sc.geom, loc.geometry)"
                )
                query = f"""
                WITH site_catchment AS (
                    SELECT ST_SetSRID(ST_GeomFromGeoJSON(%(geojson)s), 4326) AS geom
                ),
                competitors AS (
                    SELECT
                        'competitor' AS location_type,
                        loc.id::text AS id,
                        loc.name,
                        loc.category,
                        NULL::numeric AS current_trading_performance,
                        ST_Y(loc.geometry::geometry) AS latitude,
                        ST_X(loc.geometry::geometry) AS longitude,
                        ST_Contains(sc.geom, loc.geometry) AS is_contained,
                        ST_Intersects(sc.geom, loc.geometry) AS is_intersecting,
                        ST_Distance(loc.geometry::geography, ST_Centroid(sc.geom)::geography) AS distance_to_centroid_m,
                        ROUND(
                            (
                                ST_Area(
                                    ST_Intersection(
                                        ST_Buffer(loc.geometry::geography, %(radius)s)::geometry,
                                        sc.geom
                                    )::geography
                                ) / NULLIF(
                                    ST_Area(ST_Buffer(loc.geometry::geography, %(radius)s)::geography),
                                    0
                                ) * 100.0
                            )::numeric,
                            2
                        ) AS overlap_percentage
                    FROM competitor_locations loc, site_catchment sc
                    WHERE {where_pred}
                ),
                own_estate AS (
                    SELECT
                        'own_estate' AS location_type,
                        loc.id::text AS id,
                        loc.name,
                        NULL::text AS category,
                        loc.current_trading_performance,
                        ST_Y(loc.geometry::geometry) AS latitude,
                        ST_X(loc.geometry::geometry) AS longitude,
                        ST_Contains(sc.geom, loc.geometry) AS is_contained,
                        ST_Intersects(sc.geom, loc.geometry) AS is_intersecting,
                        ST_Distance(loc.geometry::geography, ST_Centroid(sc.geom)::geography) AS distance_to_centroid_m,
                        ROUND(
                            (
                                ST_Area(
                                    ST_Intersection(
                                        ST_Buffer(loc.geometry::geography, %(radius)s)::geometry,
                                        sc.geom
                                    )::geography
                                ) / NULLIF(
                                    ST_Area(ST_Buffer(loc.geometry::geography, %(radius)s)::geography),
                                    0
                                ) * 100.0
                            )::numeric,
                            2
                        ) AS overlap_percentage
                    FROM own_estate_locations loc, site_catchment sc
                    WHERE {where_pred}
                )
                SELECT * FROM competitors
                UNION ALL
                SELECT * FROM own_estate
                ORDER BY distance_to_centroid_m ASC;
                """
                cur.execute(query, {"geojson": geojson_str, "radius": trade_area_radius_m})
                cols = [desc[0] for desc in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        elif db_url:
            logger.info("Executing direct PostGIS query against DATABASE_URL...")
            rows = _execute_direct_sql(
                geojson_str=geojson_str,
                trade_area_radius_m=trade_area_radius_m,
                predicate=norm_predicate,
                db_url=db_url,
            )

        else:
            logger.info("Executing spatial query via Supabase RPC (compute_catchment_overlap)...")
            rows = _execute_supabase_rpc(
                geojson_str=geojson_str,
                trade_area_radius_m=trade_area_radius_m,
                predicate=norm_predicate,
            )

    except (DatabaseConnectionError, DatabaseQueryError):
        raise
    except Exception as exc:
        logger.error("Spatial overlap query execution encountered an error: %s", exc)
        raise DatabaseQueryError(f"Unexpected query failure: {exc}") from exc

    # Separate results
    competitor_list: List[Dict[str, Any]] = []
    own_estate_list: List[Dict[str, Any]] = []

    for row in rows:
        loc_type = row.get("location_type")
        raw_overlap = row.get("overlap_percentage")

        # Clamp overlap percentage strictly between 0.0% and 100.0%
        overlap_pct = (
            round(max(0.0, min(100.0, float(raw_overlap))), 2)
            if raw_overlap is not None
            else 0.0
        )

        item = {
            "id": row.get("id"),
            "name": row.get("name"),
            "latitude": float(row["latitude"]) if row.get("latitude") is not None else None,
            "longitude": float(row["longitude"]) if row.get("longitude") is not None else None,
            "overlap_percentage": overlap_pct,
            "is_contained": bool(row.get("is_contained")),
            "is_intersecting": bool(row.get("is_intersecting")),
            "distance_to_centroid_m": (
                round(float(row["distance_to_centroid_m"]), 1)
                if row.get("distance_to_centroid_m") is not None
                else None
            ),
        }

        if loc_type == "competitor":
            item["category"] = row.get("category")
            competitor_list.append(item)
        elif loc_type == "own_estate":
            perf = row.get("current_trading_performance")
            item["current_trading_performance"] = float(perf) if perf is not None else None
            own_estate_list.append(item)
        else:
            logger.warning("Unrecognized location_type '%s' in row, skipping.", loc_type)

    # Compute averages
    avg_comp_overlap = (
        round(
            sum(c["overlap_percentage"] for c in competitor_list) / len(competitor_list),
            2,
        )
        if competitor_list
        else 0.0
    )
    avg_own_overlap = (
        round(
            sum(o["overlap_percentage"] for o in own_estate_list) / len(own_estate_list),
            2,
        )
        if own_estate_list
        else 0.0
    )

    simplification_note = (
        "Empirical trade-area isochrones were unavailable for individual locations. "
        f"Each location's typical trade area was modeled as a standardized circular buffer "
        f"with a radius of {trade_area_radius_m:g} meters. The area-weighted overlap percentage "
        "represents the proportion of this modeled trade area that falls inside the site catchment: "
        "(Area(Catchment ∩ TradeArea) / Area(TradeArea)) * 100."
    )

    result = {
        "competitors": {
            "count": len(competitor_list),
            "locations": competitor_list,
        },
        "own_estate": {
            "count": len(own_estate_list),
            "locations": own_estate_list,
        },
        "summary": {
            "competitor_count": len(competitor_list),
            "own_estate_count": len(own_estate_list),
            "total_matched_locations": len(competitor_list) + len(own_estate_list),
            "avg_competitor_overlap_pct": avg_comp_overlap,
            "avg_own_estate_overlap_pct": avg_own_overlap,
            "spatial_predicate": norm_predicate,
            "trade_area_radius_m": float(trade_area_radius_m),
            "trade_area_simplification_note": simplification_note,
        },
    }

    logger.info(
        "Overlap calculation finished successfully: %d competitor(s), %d own estate location(s).",
        len(competitor_list),
        len(own_estate_list),
    )
    return result


__all__ = [
    "ALLOWED_PREDICATES",
    "CompetitorOverlapError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DEFAULT_TRADE_AREA_RADIUS_M",
    "InvalidCatchmentPolygonError",
    "POSTGIS_RPC_MIGRATION_SQL",
    "compute_overlap",
    "normalize_catchment_geometry",
]


if __name__ == "__main__":
    # Quick CLI test using a sample polygon around Central Chennai
    sample_catchment = {
        "type": "Polygon",
        "coordinates": [
            [
                [80.2600, 13.0700],
                [80.2900, 13.0700],
                [80.2900, 13.0950],
                [80.2600, 13.0950],
                [80.2600, 13.0700],
            ]
        ],
    }

    try:
        data = compute_overlap(sample_catchment, trade_area_radius_m=1000.0)
        print(json.dumps(data, indent=2))
    except CompetitorOverlapError as err:
        logger.error("Execution failed: %s", err)
        sys.exit(1)