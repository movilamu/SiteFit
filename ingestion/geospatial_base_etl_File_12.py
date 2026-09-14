"""
Geospatial Base ETL Pipeline (OpenStreetMap -> Supabase / PostGIS)
===================================================================
Pilot Sub-Region: Chennai District & Greater Chennai Corporation (GCC), Tamil Nadu, India.

Confirmed Open-Data Source:
    OpenStreetMap (OSM) via Overpass API / OSM Extracts.
    - High completeness in urban Tamil Nadu for roads, admin boundaries, and land use.
    - Licensing: Open Database License (ODbL) — "© OpenStreetMap contributors".

========================================================================================
DEPENDENCY NOTE: FILE 18 (ISOCHRONE GENERATION)
----------------------------------------------------------------------------------------
This module extracts, normalizes, and stores the road network data consumed by File 18
(isochrone generation and catchment calculation per ADR-002).

File 18 relies on the road network data produced here in two ways:
  1. PostGIS Database Table `road_network`:
     Stores routable road segments with LineString geometries, highway hierarchy classifications,
     geodesic segment lengths (meters), speed limits (km/h), and estimated travel times (seconds).
     File 18 queries this table to perform spatial distance buffering and network routing.
  2. Local Routable Graph / GeoJSON Artifact (`data/processed/chennai_road_network.geojson`):
     Exported with edge connectivity, lengths, and impedance weights (travel_time_sec).
     File 18 uses this representation to build local routing graphs (e.g. NetworkX / OSMnx / Pandana)
     for offline isochrone generation and as an automated fallback when external routing API
     quotas (e.g., OpenRouteService API) are exhausted, caching output polygons into `isochrone_cache`.
========================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# Attempt to load httpx (installed in environment), fallback to urllib if unavailable
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    import urllib.error
    import urllib.parse
    import urllib.request

# Optional dotenv loader
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format=LOG_FORMAT)
logger = logging.getLogger("geospatial_base_etl")


# ============================================================================
# COMPANION DATABASE SCHEMA / MIGRATION DDL
# ============================================================================
# This DDL creates the destination tables for the geospatial base layers.
# It is also documented as companion migration: /infra/migrations/002_geospatial_base.sql
CREATE_TABLES_SQL = """
-- ============================================================================
-- Migration: Geospatial Base Schema (Boundaries, Land Use, Routable Roads)
-- Target: PostGIS / Supabase
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- ----------------------------------------------------------------------------
-- Table 1: spatial_boundaries
-- Administrative boundaries (districts, taluks, zones, wards) for spatial
-- filtering and aggregation of demographic and economic indicators.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS spatial_boundaries (
    id BIGSERIAL PRIMARY KEY,
    osm_id BIGINT NOT NULL,
    name VARCHAR(255),
    admin_level INT,
    boundary_type VARCHAR(100) DEFAULT 'administrative',
    geometry GEOMETRY(Geometry, 4326) NOT NULL,
    area_sqkm DOUBLE PRECISION,
    properties JSONB DEFAULT '{}'::jsonb,
    source VARCHAR(100) DEFAULT 'OpenStreetMap',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_spatial_boundaries_osm UNIQUE (osm_id, boundary_type)
);

CREATE INDEX IF NOT EXISTS idx_spatial_boundaries_geometry 
    ON spatial_boundaries USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_spatial_boundaries_name 
    ON spatial_boundaries(name);
CREATE INDEX IF NOT EXISTS idx_spatial_boundaries_admin_level 
    ON spatial_boundaries(admin_level);

COMMENT ON TABLE spatial_boundaries IS 
    'Administrative boundary polygons (GCC zones, wards, taluks) ingested from OSM for spatial reconciliation.';

-- ----------------------------------------------------------------------------
-- Table 2: land_use_polygons
-- Zoned land use parcels (commercial, retail, residential, industrial, etc.)
-- used for retail zoning suitability scoring.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS land_use_polygons (
    id BIGSERIAL PRIMARY KEY,
    osm_id BIGINT NOT NULL,
    landuse_type VARCHAR(100) NOT NULL,
    name VARCHAR(255),
    geometry GEOMETRY(Geometry, 4326) NOT NULL,
    area_sqm DOUBLE PRECISION,
    properties JSONB DEFAULT '{}'::jsonb,
    source VARCHAR(100) DEFAULT 'OpenStreetMap',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_land_use_osm UNIQUE (osm_id, landuse_type)
);

CREATE INDEX IF NOT EXISTS idx_land_use_polygons_geometry 
    ON land_use_polygons USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_land_use_polygons_type 
    ON land_use_polygons(landuse_type);

COMMENT ON TABLE land_use_polygons IS 
    'Land use classification polygons (commercial, retail, residential, industrial) from OSM.';

-- ----------------------------------------------------------------------------
-- Table 3: road_network (Dependency for File 18: Isochrone Generation)
-- Routable highway segments with road classifications, speed limits, length,
-- and travel impedance for catchment area generation.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS road_network (
    id BIGSERIAL PRIMARY KEY,
    osm_id BIGINT NOT NULL,
    name VARCHAR(255),
    highway_type VARCHAR(100) NOT NULL,
    oneway BOOLEAN DEFAULT FALSE,
    maxspeed INT,
    length_meters DOUBLE PRECISION NOT NULL,
    estimated_travel_time_sec DOUBLE PRECISION NOT NULL,
    geometry GEOMETRY(LineString, 4326) NOT NULL,
    properties JSONB DEFAULT '{}'::jsonb,
    source VARCHAR(100) DEFAULT 'OpenStreetMap',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_road_network_osm UNIQUE (osm_id)
);

CREATE INDEX IF NOT EXISTS idx_road_network_geometry 
    ON road_network USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_road_network_highway_type 
    ON road_network(highway_type);

COMMENT ON TABLE road_network IS 
    'Routable road network segments used by File 18 for local isochrone catchment generation and travel impedance.';
"""


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Default Bounding Box for Chennai Pilot Sub-Region (Greater Chennai Corporation & surrounds)
# Format: (south_lat, west_lon, north_lat, east_lon)
CHENNAI_PILOT_BBOX: Tuple[float, float, float, float] = (12.85, 80.10, 13.25, 80.35)

# Public Overpass API endpoints with automatic failover rotation
OVERPASS_ENDPOINTS: List[str] = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Default vehicle speed assumptions in km/h by OSM highway tag for Chennai urban context
# Used to calculate estimated_travel_time_sec for File 18 isochrone generation
DEFAULT_HIGHWAY_SPEEDS_KMH: Dict[str, int] = {
    "motorway": 70,
    "motorway_link": 45,
    "trunk": 60,
    "trunk_link": 40,
    "primary": 45,
    "primary_link": 30,
    "secondary": 35,
    "secondary_link": 25,
    "tertiary": 25,
    "tertiary_link": 20,
    "residential": 20,
    "living_street": 15,
    "unclassified": 20,
    "service": 15,
}


@dataclass
class ETLConfig:
    """ETL runtime configuration parameters."""
    subregion_name: str = "Chennai"
    bbox: Tuple[float, float, float, float] = CHENNAI_PILOT_BBOX
    supabase_url: Optional[str] = field(default_factory=lambda: os.getenv("SUPABASE_URL"))
    supabase_key: Optional[str] = field(default_factory=lambda: os.getenv("SUPABASE_KEY"))
    database_url: Optional[str] = field(default_factory=lambda: os.getenv("DATABASE_URL"))
    output_dir: Path = field(default_factory=lambda: Path("data/processed"))
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    skip_db: bool = False
    dry_run: bool = False
    batch_size: int = 250
    overpass_timeout_sec: int = 180


# ============================================================================
# GEOMETRY & GEODESIC HELPERS (Pure Python / No mandatory external GIS libs)
# ============================================================================

def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth (WGS84).
    Returns distance in meters.
    """
    earth_radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_m * c


def calculate_linestring_length_meters(coords: List[List[float]]) -> float:
    """
    Calculate total length of a LineString given as [[lon, lat], ...].
    """
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        total += haversine_distance_meters(lat1, lon1, lat2, lon2)
    return round(total, 2)


def approximate_polygon_area_sqm(coords: List[List[float]]) -> float:
    """
    Approximate surface area in square meters for a planar/equirectangular projection of a polygon ring.
    Coordinates given as [[lon, lat], ...].
    """
    if len(coords) < 3:
        return 0.0
    # Mean latitude for scaling longitude degrees to meters
    mean_lat = math.radians(sum(p[1] for p in coords) / len(coords))
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(mean_lat)

    # Green's theorem / Shoelace formula in projected meters
    area = 0.0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        x_i = coords[i][0] * meters_per_deg_lon
        y_i = coords[i][1] * meters_per_deg_lat
        x_j = coords[j][0] * meters_per_deg_lon
        y_j = coords[j][1] * meters_per_deg_lat
        area += (x_i * y_j) - (x_j * y_i)

    return round(abs(area) / 2.0, 2)


def coords_to_linestring_wkt(coords: List[List[float]]) -> str:
    """Convert [[lon, lat], ...] to WKT 'LINESTRING(lon lat, ...)'."""
    coord_pairs = [f"{pt[0]} {pt[1]}" for pt in coords]
    return f"LINESTRING({', '.join(coord_pairs)})"


def coords_to_polygon_wkt(rings: List[List[List[float]]]) -> str:
    """Convert outer & inner rings to WKT 'POLYGON((lon lat, ...), ...)'."""
    formatted_rings = []
    for ring in rings:
        # Ensure polygon ring is closed
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        pairs = [f"{pt[0]} {pt[1]}" for pt in ring]
        formatted_rings.append(f"({', '.join(pairs)})")
    return f"POLYGON({', '.join(formatted_rings)})"


# ============================================================================
# OVERPASS API CLIENT & QUERY BUILDER
# ============================================================================

class OverpassClient:
    """Resilient client for querying OpenStreetMap data via Overpass API mirrors."""

    def __init__(self, endpoints: Optional[List[str]] = None, timeout_sec: int = 180):
        self.endpoints = endpoints or OVERPASS_ENDPOINTS
        self.timeout_sec = timeout_sec

    def query(self, ql_query: str, max_retries: int = 4) -> Dict[str, Any]:
        """
        Execute an Overpass QL query with automatic mirror failover and exponential backoff.
        """
        data_payload = {"data": ql_query}
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            endpoint = self.endpoints[attempt % len(self.endpoints)]
            logger.info(f"Querying Overpass mirror: {endpoint} (attempt {attempt + 1}/{max_retries})")

            try:
                if HTTPX_AVAILABLE:
                    with httpx.Client(timeout=self.timeout_sec) as client:
                        response = client.post(endpoint, data=data_payload)
                        if response.status_code == 200:
                            return response.json()
                        elif response.status_code in (429, 504):
                            logger.warning(f"Mirror {endpoint} returned status {response.status_code}. Backing off.")
                            time.sleep(3 * (attempt + 1))
                            continue
                        else:
                            response.raise_for_status()
                else:
                    encoded_data = urllib.parse.urlencode(data_payload).encode("utf-8")
                    req = urllib.request.Request(
                        endpoint,
                        data=encoded_data,
                        headers={"User-Agent": "RetailSiteSelectionPlatform/1.0 (Chennai Pilot)"},
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                        if resp.status == 200:
                            return json.loads(resp.read().decode("utf-8"))

            except Exception as ex:
                last_exception = ex
                logger.warning(f"Request to {endpoint} failed: {ex}. Retrying next mirror...")
                time.sleep(2 * (attempt + 1))

        raise RuntimeError(f"All Overpass API endpoints failed after {max_retries} attempts: {last_exception}")


def build_admin_boundaries_query(bbox: Tuple[float, float, float, float], timeout_sec: int = 180) -> str:
    """
    Overpass QL query for administrative boundaries (districts, zones, taluks) in bbox.
    Uses 'out geom;' to retrieve complete coordinate linestrings/polygons directly.
    """
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:{timeout_sec}];
    (
      relation["boundary"="administrative"]["admin_level"~"5|6|8"]({s},{w},{n},{e});
      relation["boundary"="administrative"]["name"~"Chennai|Greater Chennai",i]({s},{w},{n},{e});
    );
    out tags geom;
    """


def build_road_network_query(bbox: Tuple[float, float, float, float], timeout_sec: int = 180) -> str:
    """
    Overpass QL query for road network hierarchy.
    Filters to vehicular highways suitable for routing & isochrones.
    """
    s, w, n, e = bbox
    highway_filter = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street"
    return f"""
    [out:json][timeout:{timeout_sec}];
    (
      way["highway"~"^{highway_filter}$"]({s},{w},{n},{e});
    );
    out tags geom;
    """


def build_land_use_query(bbox: Tuple[float, float, float, float], timeout_sec: int = 180) -> str:
    """
    Overpass QL query for zoned land use parcels in bbox.
    Filters commercial, retail, residential, industrial, institutional, etc.
    """
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:{timeout_sec}];
    (
      way["landuse"]({s},{w},{n},{e});
      relation["landuse"]({s},{w},{n},{e});
    );
    out tags geom;
    """


# ============================================================================
# FEATURE PARSERS & NORMALIZERS
# ============================================================================

class OSMFeatureParser:
    """Parses raw Overpass JSON elements into normalized records and GeoJSON."""

    @staticmethod
    def parse_road_segment(element: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parses an OSM way element into a normalized road network segment.
        Prepares length and travel-time weights required by File 18 for isochrones.
        """
        geometry_pts = element.get("geometry")
        if not geometry_pts or len(geometry_pts) < 2:
            return None

        # Coordinates formatted as [longitude, latitude] for GeoJSON & PostGIS
        coords = [[pt["lon"], pt["lat"]] for pt in geometry_pts]
        tags = element.get("tags", {})
        highway_type = tags.get("highway", "unclassified")
        name = tags.get("name")
        osm_id = element["id"]

        # One-way parsing
        oneway_tag = tags.get("oneway", "no").lower()
        oneway = oneway_tag in ("yes", "true", "1")

        # Speed parsing (km/h)
        maxspeed_tag = tags.get("maxspeed")
        maxspeed: int = DEFAULT_HIGHWAY_SPEEDS_KMH.get(highway_type, 25)
        if maxspeed_tag:
            try:
                # Extract first numeric sequence (e.g., '40', '50 km/h')
                digits = "".join(filter(str.isdigit, maxspeed_tag))
                if digits:
                    parsed_speed = int(digits)
                    if 5 <= parsed_speed <= 140:
                        maxspeed = parsed_speed
            except Exception:
                pass

        # Length and impedance computation
        length_meters = calculate_linestring_length_meters(coords)
        speed_mps = (maxspeed * 1000.0) / 3600.0
        travel_time_sec = round(length_meters / speed_mps, 2) if speed_mps > 0 else 0.0

        geojson_geom = {
            "type": "LineString",
            "coordinates": coords
        }
        wkt_geom = coords_to_linestring_wkt(coords)

        return {
            "osm_id": osm_id,
            "name": name,
            "highway_type": highway_type,
            "oneway": oneway,
            "maxspeed": maxspeed,
            "length_meters": length_meters,
            "estimated_travel_time_sec": travel_time_sec,
            "geometry": geojson_geom,
            "wkt": wkt_geom,
            "properties": {
                "lanes": tags.get("lanes"),
                "surface": tags.get("surface"),
                "ref": tags.get("ref"),
                "bridge": tags.get("bridge"),
            },
            "source": "OpenStreetMap",
        }

    @staticmethod
    def parse_boundary(element: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parses an OSM relation or way into an administrative boundary polygon record.
        """
        tags = element.get("tags", {})
        osm_id = element["id"]
        name = tags.get("name") or tags.get("name:en") or f"Admin Boundary {osm_id}"
        admin_level_raw = tags.get("admin_level")
        admin_level = int(admin_level_raw) if (admin_level_raw and admin_level_raw.isdigit()) else None

        # Build geometry
        coords = []
        if "geometry" in element and element["geometry"]:
            coords = [[pt["lon"], pt["lat"]] for pt in element["geometry"]]
        elif "members" in element:
            # Overpass geom output for relation members
            for member in element.get("members", []):
                if member.get("role") in ("outer", "") and "geometry" in member:
                    coords.extend([[pt["lon"], pt["lat"]] for pt in member["geometry"]])

        if len(coords) < 3:
            return None

        # Close ring if necessary
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        area_sqm = approximate_polygon_area_sqm(coords)
        area_sqkm = round(area_sqm / 1_000_000.0, 4)

        geojson_geom = {
            "type": "Polygon",
            "coordinates": [coords]
        }
        wkt_geom = coords_to_polygon_wkt([coords])

        return {
            "osm_id": osm_id,
            "name": name,
            "admin_level": admin_level,
            "boundary_type": tags.get("boundary", "administrative"),
            "geometry": geojson_geom,
            "wkt": wkt_geom,
            "area_sqkm": area_sqkm,
            "properties": {
                "wikidata": tags.get("wikidata"),
                "wikipedia": tags.get("wikipedia"),
                "admin_level": admin_level,
            },
            "source": "OpenStreetMap",
        }

    @staticmethod
    def parse_land_use(element: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parses an OSM element into a land use parcel record.
        """
        tags = element.get("tags", {})
        landuse_type = tags.get("landuse")
        if not landuse_type:
            return None

        osm_id = element["id"]
        name = tags.get("name")

        coords = []
        if "geometry" in element and element["geometry"]:
            coords = [[pt["lon"], pt["lat"]] for pt in element["geometry"]]
        elif "members" in element:
            for member in element.get("members", []):
                if member.get("role") in ("outer", "") and "geometry" in member:
                    coords.extend([[pt["lon"], pt["lat"]] for pt in member["geometry"]])

        if len(coords) < 3:
            return None

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        area_sqm = approximate_polygon_area_sqm(coords)

        geojson_geom = {
            "type": "Polygon",
            "coordinates": [coords]
        }
        wkt_geom = coords_to_polygon_wkt([coords])

        return {
            "osm_id": osm_id,
            "landuse_type": landuse_type,
            "name": name,
            "geometry": geojson_geom,
            "wkt": wkt_geom,
            "area_sqm": area_sqm,
            "properties": {
                "operator": tags.get("operator"),
                "amenity": tags.get("amenity"),
            },
            "source": "OpenStreetMap",
        }


# ============================================================================
# PERSISTENCE & STORAGE MANAGER (SUPABASE / POSTGIS / LOCAL ARTIFACTS)
# ============================================================================

class SupabaseStorageManager:
    """
    Manages persistence to Supabase PostGIS tables and exports local routing artifacts
    for File 18 (isochrone generation).
    """

    def __init__(self, config: ETLConfig):
        self.config = config
        self._validate_credentials()

    def _validate_credentials(self) -> None:
        """Validate presence of database connection credentials."""
        if self.config.skip_db or self.config.dry_run:
            logger.info("Database storage skipped (dry-run or skip-db enabled).")
            return

        if not (self.config.supabase_url and self.config.supabase_key) and not self.config.database_url:
            logger.warning(
                "Neither SUPABASE_URL/SUPABASE_KEY nor DATABASE_URL are set in environment. "
                "The pipeline will output local artifacts and skip database upsert."
            )
            self.config.skip_db = True

    def initialize_schema(self) -> bool:
        """
        Execute companion DDL statements if direct PostgreSQL connection is available.
        """
        if self.config.skip_db or self.config.dry_run or not self.config.database_url:
            logger.info("Skipping automated schema execution (no direct DATABASE_URL provided).")
            return False

        try:
            import psycopg2  # type: ignore
            logger.info("Applying core schema migrations via direct PostgreSQL connection...")
            with psycopg2.connect(self.config.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_TABLES_SQL)
                conn.commit()
            logger.info("Geospatial base schema initialized successfully.")
            return True
        except ImportError:
            logger.warning("psycopg2 not installed; cannot execute DDL directly. Ensure migrations are applied.")
            return False
        except Exception as err:
            logger.error(f"Failed to execute schema DDL: {err}")
            return False

    def upsert_records_postgrest(self, table_name: str, records: List[Dict[str, Any]]) -> int:
        """
        Upsert records into Supabase via PostgREST HTTP endpoint using SUPABASE_URL and SUPABASE_KEY.
        Geometries are passed in GeoJSON format.
        """
        if self.config.skip_db or self.config.dry_run or not records:
            return 0

        endpoint = f"{self.config.supabase_url.rstrip('/')}/rest/v1/{table_name}"
        headers = {
            "apikey": self.config.supabase_key,
            "Authorization": f"Bearer {self.config.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }

        total_saved = 0
        batch_size = self.config.batch_size

        for i in range(0, len(records), batch_size):
            chunk = records[i: i + batch_size]
            payload = []
            for item in chunk:
                row = {k: v for k, v in item.items() if k != "wkt"}
                payload.append(row)

            try:
                if HTTPX_AVAILABLE:
                    with httpx.Client(timeout=60.0) as client:
                        resp = client.post(endpoint, json=payload, headers=headers)
                        if resp.status_code in (200, 201):
                            total_saved += len(chunk)
                        else:
                            logger.error(f"PostgREST error on table {table_name} [{resp.status_code}]: {resp.text}")
                else:
                    req_data = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(endpoint, data=req_data, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=60.0) as resp:
                        if resp.status in (200, 201):
                            total_saved += len(chunk)
            except Exception as ex:
                logger.error(f"Error persisting batch {i} to {table_name}: {ex}")

        logger.info(f"Persisted {total_saved}/{len(records)} records to {table_name}.")
        return total_saved

    def export_geojson(self, filename: str, features: List[Dict[str, Any]]) -> Path:
        """
        Export parsed features to a GeoJSON file in output_dir.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.config.output_dir / filename

        geojson_features = []
        for feat in features:
            props = {k: v for k, v in feat.items() if k not in ("geometry", "wkt")}
            geojson_features.append({
                "type": "Feature",
                "geometry": feat["geometry"],
                "properties": props
            })

        geojson_doc = {
            "type": "FeatureCollection",
            "name": filename.replace(".geojson", ""),
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": geojson_features
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(geojson_doc, f, indent=2)

        logger.info(f"Exported {len(features)} features to {out_path}")
        return out_path


# ============================================================================
# MAIN ETL ORCHESTRATOR
# ============================================================================

class GeospatialBaseETL:
    """Orchestrates extraction, transformation, and loading of the geospatial base layers."""

    def __init__(self, config: Optional[ETLConfig] = None):
        self.config = config or ETLConfig()
        self.client = OverpassClient(timeout_sec=self.config.overpass_timeout_sec)
        self.storage = SupabaseStorageManager(self.config)

    def run_boundaries_pipeline(self, raw_elements: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Extract, parse, and store administrative boundaries."""
        logger.info("--- Starting Administrative Boundaries Ingestion ---")
        if raw_elements is None:
            query = build_admin_boundaries_query(self.config.bbox, self.config.overpass_timeout_sec)
            resp = self.client.query(query)
            raw_elements = resp.get("elements", [])

        boundaries = []
        for elem in raw_elements:
            parsed = OSMFeatureParser.parse_boundary(elem)
            if parsed:
                boundaries.append(parsed)

        logger.info(f"Extracted {len(boundaries)} administrative boundary polygons for {self.config.subregion_name}.")
        self.storage.export_geojson(f"{self.config.subregion_name.lower()}_admin_boundaries.geojson", boundaries)
        self.storage.upsert_records_postgrest("spatial_boundaries", boundaries)
        return boundaries

    def run_roads_pipeline(self, raw_elements: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Extract, parse, and store routable road network data.
        Generates the spatial foundation and travel-time weights required by File 18.
        """
        logger.info("--- Starting Road Network Ingestion (File 18 Isochrone Dependency) ---")
        if raw_elements is None:
            query = build_road_network_query(self.config.bbox, self.config.overpass_timeout_sec)
            resp = self.client.query(query)
            raw_elements = resp.get("elements", [])

        roads = []
        for elem in raw_elements:
            parsed = OSMFeatureParser.parse_road_segment(elem)
            if parsed:
                roads.append(parsed)

        logger.info(f"Extracted {len(roads)} routable road segments for {self.config.subregion_name}.")
        # Export for File 18 local routing / offline isochrone catchment generator
        self.storage.export_geojson(f"{self.config.subregion_name.lower()}_road_network.geojson", roads)
        self.storage.upsert_records_postgrest("road_network", roads)
        return roads

    def run_landuse_pipeline(self, raw_elements: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Extract, parse, and store land use classification polygons."""
        logger.info("--- Starting Land Use Polygons Ingestion ---")
        if raw_elements is None:
            query = build_land_use_query(self.config.bbox, self.config.overpass_timeout_sec)
            resp = self.client.query(query)
            raw_elements = resp.get("elements", [])

        landuse = []
        for elem in raw_elements:
            parsed = OSMFeatureParser.parse_land_use(elem)
            if parsed:
                landuse.append(parsed)

        logger.info(f"Extracted {len(landuse)} land use polygons for {self.config.subregion_name}.")
        self.storage.export_geojson(f"{self.config.subregion_name.lower()}_land_use.geojson", landuse)
        self.storage.upsert_records_postgrest("land_use_polygons", landuse)
        return landuse

    def run_all(self, input_file: Optional[Path] = None) -> Dict[str, int]:
        """
        Run complete geospatial base ETL pipeline for pilot sub-region.
        """
        logger.info(f"Starting Geospatial Base ETL for Pilot Sub-Region: {self.config.subregion_name}")
        logger.info(f"Bounding Box (S, W, N, E): {self.config.bbox}")

        self.storage.initialize_schema()

        if input_file and input_file.exists():
            logger.info(f"Loading pre-downloaded OSM extract from: {input_file}")
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            elements = data.get("elements", [])
            # Route elements by tag
            boundary_elements = [e for e in elements if "boundary" in e.get("tags", {})]
            road_elements = [e for e in elements if "highway" in e.get("tags", {})]
            landuse_elements = [e for e in elements if "landuse" in e.get("tags", {})]

            boundaries = self.run_boundaries_pipeline(boundary_elements)
            roads = self.run_roads_pipeline(road_elements)
            landuse = self.run_landuse_pipeline(landuse_elements)
        else:
            boundaries = self.run_boundaries_pipeline()
            roads = self.run_roads_pipeline()
            landuse = self.run_landuse_pipeline()

        summary = {
            "administrative_boundaries_count": len(boundaries),
            "road_segments_count": len(roads),
            "land_use_polygons_count": len(landuse),
        }
        logger.info(f"Geospatial Base ETL Completed Successfully. Summary: {summary}")
        return summary


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest administrative boundaries, road network, and land use for Chennai pilot sub-region."
    )
    parser.add_argument("--subregion", default="Chennai", help="Sub-region name (default: Chennai)")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=list(CHENNAI_PILOT_BBOX),
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        help="Bounding box coordinates (default: 12.85 80.10 13.25 80.35)",
    )
    parser.add_argument("--input-file", type=Path, help="Path to local OSM Overpass JSON extract")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory for GeoJSON")
    parser.add_argument("--skip-db", action="store_true", help="Skip Supabase/PostGIS database insertion")
    parser.add_argument("--dry-run", action="store_true", help="Perform mock ETL run without remote network calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ETLConfig(
        subregion_name=args.subregion,
        bbox=tuple(args.bbox),  # type: ignore
        output_dir=args.output_dir,
        skip_db=args.skip_db,
        dry_run=args.dry_run,
    )

    etl = GeospatialBaseETL(config)

    if config.dry_run:
        logger.info("Executing dry-run mode with synthetic pilot dataset...")
        # Synthetic mock feature verification
        mock_road = {
            "id": 1001,
            "tags": {"highway": "primary", "name": "Anna Salai", "maxspeed": "50"},
            "geometry": [{"lat": 13.06, "lon": 80.25}, {"lat": 13.07, "lon": 80.26}],
        }
        mock_boundary = {
            "id": 2001,
            "tags": {"boundary": "administrative", "admin_level": "6", "name": "Chennai District"},
            "geometry": [
                {"lat": 12.90, "lon": 80.15},
                {"lat": 13.20, "lon": 80.15},
                {"lat": 13.20, "lon": 80.30},
                {"lat": 12.90, "lon": 80.30},
                {"lat": 12.90, "lon": 80.15},
            ],
        }
        mock_landuse = {
            "id": 3001,
            "tags": {"landuse": "commercial", "name": "T. Nagar Commercial Hub"},
            "geometry": [
                {"lat": 13.03, "lon": 80.22},
                {"lat": 13.04, "lon": 80.22},
                {"lat": 13.04, "lon": 80.24},
                {"lat": 13.03, "lon": 80.24},
                {"lat": 13.03, "lon": 80.22},
            ],
        }
        etl.run_boundaries_pipeline([mock_boundary])
        etl.run_roads_pipeline([mock_road])
        etl.run_landuse_pipeline([mock_landuse])
        logger.info("Dry run verified successfully.")
        return

    try:
        etl.run_all(input_file=args.input_file)
    except Exception as e:
        logger.critical(f"Geospatial Base ETL failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
