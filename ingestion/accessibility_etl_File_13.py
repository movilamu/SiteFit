"""
Accessibility ETL Pipeline: Chennai Pilot Sub-Region
=====================================================
Ingests micro-accessibility, public transit connectivity, and parking provision
for analysis units (H3 grid cells) in the Greater Chennai pilot region using
OpenStreetMap (OSM) via Overpass API and mobility proxies.

Schema Target: accessibility_data
- unit_id: VARCHAR(64) REFERENCES analysis_units(id)
- transit_score: DOUBLE PRECISION (0.0 to 100.0)
- parking_availability: DOUBLE PRECISION (0.0 to 100.0)
- footfall_estimate: INT4 (Nullable - left null where direct sensor/turnstile data unavailable)
- data_confidence: DOUBLE PRECISION (0.0 to 1.0, reflecting directness of measurement)

Sources:
- OpenStreetMap via Overpass API (Bus stops, Metro/MRTS stations, Suburban rail, Parking facilities)
- OpenCity / Vahan mobility indicators (proxy validation)
"""

import os
import sys
import time
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
import requests

# Try importing H3 (handles both v3 and v4 API)
try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False

# Try importing Supabase client
try:
    from supabase import create_client, Client
    HAS_SUPABASE_SDK = True
except ImportError:
    HAS_SUPABASE_SDK = False

# -----------------------------------------------------------------------------
# Configuration & Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("accessibility_etl")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()

# Overpass API endpoints with fallbacks
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]

# Chennai Pilot Bounding Box [south, west, north, east]
# Covering Greater Chennai Corporation (GCC) core + key suburban transit nodes
CHENNAI_BBOX = (12.85, 80.10, 13.25, 80.35)

# Transit node weightings for accessibility score
TRANSIT_WEIGHTS = {
    "metro_station": 5.0,    # Chennai Metro (CMRL)
    "railway_station": 4.5,  # Suburban / MRTS
    "bus_station": 3.0,      # Major MTC Bus Termini
    "bus_stop": 1.0,         # Standard MTC bus stop
    "tram_halt": 1.5,
    "platform": 1.0
}

# -----------------------------------------------------------------------------
# Helper Functions: H3 Spatial Utilities
# -----------------------------------------------------------------------------
def latlng_to_h3(lat: float, lng: float, resolution: int = 8) -> str:
    """Safely converts lat/lng to H3 index across v3 and v4 library versions."""
    if not HAS_H3:
        raise RuntimeError("h3 package is required. Run 'pip install h3'.")
    
    # H3 v4
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)
    # H3 v3
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lng, resolution)
    raise AttributeError("Compatible H3 conversion function not found.")

def h3_to_latlng(cell_id: str) -> Tuple[float, float]:
    """Safely converts H3 cell ID to centroid lat/lng."""
    if hasattr(h3, "cell_to_latlng"):
        return h3.cell_to_latlng(cell_id)
    if hasattr(h3, "h3_to_geo"):
        return h3.h3_to_geo(cell_id)
    raise AttributeError("Compatible H3 reverse function not found.")

# -----------------------------------------------------------------------------
# Overpass Data Extraction
# -----------------------------------------------------------------------------
def fetch_overpass_query(query: str, max_retries: int = 3, timeout_sec: int = 60) -> Dict[str, Any]:
    """Queries OSM Overpass API with retry and endpoint failover logic."""
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Executing Overpass query against %s (Attempt %d/%d)...", endpoint, attempt, max_retries)
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    headers={"User-Agent": "Antigravity-Retail-Accessibility-ETL/1.0"},
                    timeout=timeout_sec
                )
                if response.status_code == 200:
                    data = response.json()
                    elements = data.get("elements", [])
                    logger.info("Successfully fetched %d OSM elements from %s.", len(elements), endpoint)
                    return data
                elif response.status_code in (429, 504):
                    logger.warning("Overpass rate limited / timed out (%d). Backing off...", response.status_code)
                    time.sleep(attempt * 3)
                else:
                    logger.warning("Overpass returned status %d: %s", response.status_code, response.text[:100])
                    break
            except requests.RequestException as exc:
                logger.warning("Request failed to %s: %s", endpoint, exc)
                last_error = exc
                time.sleep(attempt * 2)

    raise RuntimeError(f"All Overpass API endpoints failed. Last error: {last_error}")

def extract_transit_nodes(bbox: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
    """Extracts bus stops, railway/metro stations, and transit platforms for Chennai."""
    s, w, n, e = bbox
    query = f"""
    [out:json][timeout:90];
    (
      // Metro & Railway Stations
      node["railway"="station"]({s},{w},{n},{e});
      node["station"="subway"]({s},{w},{n},{e});
      node["railway"="halt"]({s},{w},{n},{e});
      
      // Bus Stops & Terminals
      node["highway"="bus_stop"]({s},{w},{n},{e});
      node["public_transport"="platform"]["bus"="yes"]({s},{w},{n},{e});
      node["amenity"="bus_station"]({s},{w},{n},{e});
      
      // Ways with center points (e.g. large stations mapped as polygons)
      way["railway"="station"]({s},{w},{n},{e});
      way["amenity"="bus_station"]({s},{w},{n},{e});
    );
    out center;
    """
    data = fetch_overpass_query(query)
    results = []
    for el in data.get("elements", []):
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if not lat or not lon:
            continue
        
        tags = el.get("tags", {})
        transit_type = "bus_stop"
        if tags.get("railway") == "station" or tags.get("station") == "subway":
            if "metro" in tags.get("name", "").lower() or tags.get("subway") == "yes":
                transit_type = "metro_station"
            else:
                transit_type = "railway_station"
        elif tags.get("amenity") == "bus_station":
            transit_type = "bus_station"
        
        results.append({
            "id": el.get("id"),
            "lat": lat,
            "lon": lon,
            "type": transit_type,
            "weight": TRANSIT_WEIGHTS.get(transit_type, 1.0),
            "name": tags.get("name")
        })
    return results

def extract_parking_facilities(bbox: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
    """Extracts parking lots, multi-level parking, and designated street parking."""
    s, w, n, e = bbox
    query = f"""
    [out:json][timeout:90];
    (
      node["amenity"="parking"]({s},{w},{n},{e});
      way["amenity"="parking"]({s},{w},{n},{e});
      node["parking"="street_side"]({s},{w},{n},{e});
      node["parking"="multi-storey"]({s},{w},{n},{e});
      way["parking"="multi-storey"]({s},{w},{n},{e});
    );
    out center;
    """
    data = fetch_overpass_query(query)
    results = []
    for el in data.get("elements", []):
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if not lat or not lon:
            continue
        
        tags = el.get("tags", {})
        capacity_str = tags.get("capacity")
        capacity = None
        if capacity_str and capacity_str.isdigit():
            capacity = int(capacity_str)
        
        results.append({
            "id": el.get("id"),
            "lat": lat,
            "lon": lon,
            "parking_type": tags.get("parking", "surface"),
            "access": tags.get("access", "yes"),
            "capacity": capacity
        })
    return results

# -----------------------------------------------------------------------------
# Database Interaction (Supabase client or direct PostgREST REST API)
# -----------------------------------------------------------------------------
def get_existing_analysis_units(resolution: int = 8) -> List[str]:
    """
    Retrieves active analysis_unit IDs for Chennai from Supabase.
    If database is currently empty or inaccessible, falls back to generating
    canonical H3 cells across the Chennai bounding polygon.
    """
    unit_ids: List[str] = []
    
    if SUPABASE_URL and SUPABASE_KEY:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        try:
            url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/analysis_units?select=id&limit=10000"
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                rows = resp.json()
                unit_ids = [r["id"] for r in rows if "id" in r]
                logger.info("Found %d existing analysis_units in Supabase.", len(unit_ids))
            else:
                logger.warning("Could not fetch analysis_units (%d): %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.warning("Failed querying analysis_units table: %s", exc)

    # Fallback / Local Generation if units table not yet populated
    if not unit_ids:
        logger.info("Generating H3 cells (res=%d) across Chennai pilot bbox...", resolution)
        if not HAS_H3:
            raise RuntimeError("h3 library required to generate grid cells.")
        
        s, w, n, e = CHENNAI_BBOX
        lat_steps = int((n - s) / 0.008)
        lon_steps = int((e - w) / 0.008)
        cells = set()
        for i in range(lat_steps + 1):
            for j in range(lon_steps + 1):
                cur_lat = s + i * 0.008
                cur_lon = w + j * 0.008
                cell = latlng_to_h3(cur_lat, cur_lon, resolution)
                cells.add(cell)
        unit_ids = sorted(list(cells))
        logger.info("Generated %d synthetic H3 analysis units covering Chennai.", len(unit_ids))

    return unit_ids

def write_accessibility_records(records: List[Dict[str, Any]], batch_size: int = 500) -> int:
    """
    Writes or upserts computed records into the accessibility_data table.
    Uses PostgREST upsert with prefer=resolution=merge-duplicates.
    """
    if not records:
        logger.info("No accessibility records to persist.")
        return 0

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set. Operating in DRY-RUN mode.")
        logger.info("Sample record: %s", records[0])
        return len(records)

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/accessibility_data"
    
    total_written = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            resp = requests.post(url, headers=headers, json=batch, timeout=30)
            if resp.status_code in (200, 201):
                total_written += len(batch)
                logger.info("Successfully wrote batch %d-%d of %d records.", i + 1, i + len(batch), len(records))
            else:
                logger.error("Failed to insert batch: %d - %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.error("Exception writing batch to Supabase: %s", exc)

    return total_written

# -----------------------------------------------------------------------------
# Metric Calculation & Aggregation
# -----------------------------------------------------------------------------
def calculate_accessibility_metrics(
    unit_ids: List[str],
    transit_nodes: List[Dict[str, Any]],
    parking_facilities: List[Dict[str, Any]],
    resolution: int = 8
) -> List[Dict[str, Any]]:
    """
    Maps spatial transit and parking entities to analysis units and calculates:
    1. transit_score (0.0 to 100.0) based on weighted multi-modal density.
    2. parking_availability (0.0 to 100.0) based on capacity and facility density.
    3. footfall_estimate: Left NULL (None) per spec, since no free systematic
       footfall count dataset exists for Tamil Nadu.
    4. data_confidence (0.0 to 1.0): Reflects direct observation vs. sparsity.
    """
    logger.info("Mapping %d transit nodes and %d parking facilities to %d units...",
                len(transit_nodes), len(parking_facilities), len(unit_ids))

    transit_by_unit: Dict[str, List[Dict[str, Any]]] = {uid: [] for uid in unit_ids}
    for node in transit_nodes:
        try:
            cell = latlng_to_h3(node["lat"], node["lon"], resolution)
            if cell in transit_by_unit:
                transit_by_unit[cell].append(node)
        except Exception:
            continue

    parking_by_unit: Dict[str, List[Dict[str, Any]]] = {uid: [] for uid in unit_ids}
    for park in parking_facilities:
        try:
            cell = latlng_to_h3(park["lat"], park["lon"], resolution)
            if cell in parking_by_unit:
                parking_by_unit[cell].append(park)
        except Exception:
            continue

    output_records: List[Dict[str, Any]] = []
    PARKING_SATURATION_POINT = 150.0

    for uid in unit_ids:
        unit_transit = transit_by_unit.get(uid, [])
        unit_parking = parking_by_unit.get(uid, [])

        # 1. Calculate raw transit score
        weighted_transit_sum = sum(n.get("weight", 1.0) for n in unit_transit)
        if weighted_transit_sum > 0:
            transit_score = round(100.0 * (1.0 - math.exp(-weighted_transit_sum / 6.0)), 2)
        else:
            transit_score = 0.0

        # 2. Calculate parking availability
        parking_units_count = len(unit_parking)
        explicit_capacity_sum = sum(p["capacity"] for p in unit_parking if p.get("capacity"))
        effective_capacity = explicit_capacity_sum + (parking_units_count * 25)
        if effective_capacity > 0:
            parking_availability = round(min(100.0, 100.0 * (1.0 - math.exp(-effective_capacity / PARKING_SATURATION_POINT))), 2)
        else:
            parking_availability = 0.0

        # 3. Footfall estimate: left None (SQL NULL) per spec
        footfall_estimate = None

        # 4. Data Confidence scoring (0.0 to 1.0)
        if weighted_transit_sum > 0 and parking_units_count > 0:
            data_confidence = 0.85
        elif weighted_transit_sum > 0:
            data_confidence = 0.75
        elif parking_units_count > 0:
            data_confidence = 0.65
        else:
            data_confidence = 0.50

        record = {
            "unit_id": uid,
            "transit_score": transit_score,
            "parking_availability": parking_availability,
            "footfall_estimate": footfall_estimate,
            "data_confidence": data_confidence
        }
        output_records.append(record)

    logger.info("Successfully generated accessibility metrics for %d units.", len(output_records))
    return output_records

# -----------------------------------------------------------------------------
# Main Pipeline Runner
# -----------------------------------------------------------------------------
def run_accessibility_etl(dry_run: bool = False, resolution: int = 8):
    """Orchestrates the ingestion, metric calculation, and Supabase persistence."""
    logger.info("==========================================================")
    logger.info("Starting Accessibility ETL for Pilot Region: Chennai (GCC)")
    logger.info("==========================================================")
    
    start_time = time.time()

    # Step 1: Query transit infrastructure
    try:
        transit_nodes = extract_transit_nodes(CHENNAI_BBOX)
        logger.info("Extracted %d transit nodes.", len(transit_nodes))
    except Exception as e:
        logger.error("Failed to extract transit nodes: %s", e)
        transit_nodes = []

    # Step 2: Query parking infrastructure
    try:
        parking_facilities = extract_parking_facilities(CHENNAI_BBOX)
        logger.info("Extracted %d parking facilities.", len(parking_facilities))
    except Exception as e:
        logger.error("Failed to extract parking facilities: %s", e)
        parking_facilities = []

    if not transit_nodes and not parking_facilities:
        logger.critical("No spatial entities retrieved from OSM. Halting pipeline.")
        sys.exit(1)

    # Step 3: Fetch or generate analysis units
    units = get_existing_analysis_units(resolution=resolution)
    if not units:
        logger.critical("No analysis units available for mapping.")
        sys.exit(1)

    # Step 4: Calculate metrics
    records = calculate_accessibility_metrics(
        unit_ids=units,
        transit_nodes=transit_nodes,
        parking_facilities=parking_facilities,
        resolution=resolution
    )

    # Step 5: Persist to accessibility_data
    if dry_run:
        logger.info("[DRY RUN] Would write %d records to accessibility_data.", len(records))
        if records:
            logger.info("Sample record: %s", records[0])
    else:
        written = write_accessibility_records(records)
        logger.info("Successfully persisted %d accessibility records to Supabase.", written)

    elapsed = round(time.time() - start_time, 2)
    logger.info("Accessibility ETL completed in %s seconds.", elapsed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest accessibility data for Chennai pilot.")
    parser.add_argument("--dry-run", action="store_true", help="Process without writing to Supabase")
    parser.add_argument("--res", type=int, default=8, help="H3 resolution (default: 8)")
    args = parser.parse_args()

    run_accessibility_etl(dry_run=args.dry_run, resolution=args.res)
