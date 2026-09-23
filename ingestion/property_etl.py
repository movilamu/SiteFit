# /ingestion/property_etl.py
"""
Property listings ETL for the Chennai pilot.

SOURCE / COVERAGE LIMITATION
----------------------------
There is no comprehensive, free, machine-readable commercial-property
availability dataset for Tamil Nadu.

This ETL therefore uses the Tamil Nadu Government eProcurement / Tender Cum
Auction ecosystem as a best-effort free source for publicly advertised
commercial premises. The upstream portal is:
    https://tntenders.gov.in/

The portal provides tender schedules free of cost, but property availability,
rent, floor area and lease terms are not exposed as a consistent property
listing API. Consequently this ETL supports:
  1. A normalized CSV/JSON export produced from downloaded tender/auction
     notices; and
  2. HTML/text tender notices where fields can be extracted heuristically.

IMPORTANT:
    This is NOT a complete Chennai commercial-property inventory.
    Missing properties, rents, areas and lease terms are expected.

Expected normalized input fields:
    name
    address
    latitude / lat
    longitude / lon
    unit_size
    floor_area
    rent
    lease_term_years
    tenure_type
    source
    last_updated

The geometry is generated from latitude/longitude. If coordinates are absent,
the record is skipped rather than inserting a false location.

Environment:
    SUPABASE_URL
    SUPABASE_KEY

Optional:
    PROPERTY_SOURCE_URL
        URL to a CSV/JSON/HTML export prepared from the public TN tender source.

    PROPERTY_SOURCE_FILE
        Local CSV/JSON/HTML file. Useful for scheduled ETL after downloading
        tender schedules from the government portal.

    CHENNAI_BBOX
        Optional bounding box:
        min_lon,min_lat,max_lon,max_lat

Dependencies:
    pip install requests pandas supabase beautifulsoup4 python-dotenv
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import Client, create_client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SOURCE_URL = os.getenv("PROPERTY_SOURCE_URL")
SOURCE_FILE = os.getenv("PROPERTY_SOURCE_FILE")

# Chennai / Greater Chennai approximate bounding box.
# Can be overridden for the pilot or narrowed to a particular zone.
DEFAULT_BBOX = "80.10,12.80,80.40,13.25"
BBOX = os.getenv("CHENNAI_BBOX", DEFAULT_BBOX)

REQUEST_TIMEOUT = int(os.getenv("PROPERTY_REQUEST_TIMEOUT", "30"))

USER_AGENT = (
    "site-scoring-property-etl/1.0 "
    "(Tamil Nadu pilot; public-data ingestion)"
)

# Government tender/property data is inherently messy. Keep extraction
# conservative: false coordinates or false rents are worse than NULLs.
MIN_LAT = 8.0
MAX_LAT = 14.0
MIN_LON = 76.0
MAX_LON = 80.5


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("property_etl")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PropertyRecord:
    name: str | None
    geometry: str
    unit_size: float | None
    floor_area: float | None
    rent: float | None
    lease_term_years: float | None
    tenure_type: str | None
    source: str
    last_updated: str | None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def require_environment() -> None:
    """Fail early if required Supabase credentials are missing."""
    missing = []

    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")

    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )


def normalize_text(value: Any) -> str | None:
    """Normalize arbitrary input to a compact string."""
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return re.sub(r"\s+", " ", text)


def parse_number(value: Any) -> float | None:
    """
    Parse a numeric value from Indian-style tender text.

    Examples:
        '₹ 1,25,000' -> 125000
        '125000 per month' -> 125000
        '1,500 sq.ft.' -> 1500
    """
    if value is None:
        return None

    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)

    text = normalize_text(value)

    if not text:
        return None

    # Remove currency symbols, commas and surrounding text.
    match = re.search(
        r"[-+]?\d+(?:,\d{2,3})*(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    number = match.group(0).replace(",", "")

    try:
        return float(number)
    except ValueError:
        return None


def parse_lat_lon(row: dict[str, Any]) -> tuple[float, float] | None:
    """Read latitude/longitude from common input column names."""
    lat_keys = ("latitude", "lat", "y", "Latitude", "LATITUDE")
    lon_keys = ("longitude", "lon", "lng", "x", "Longitude", "LONGITUDE")

    lat = next((row.get(k) for k in lat_keys if row.get(k) is not None), None)
    lon = next((row.get(k) for k in lon_keys if row.get(k) is not None), None)

    lat_value = parse_number(lat)
    lon_value = parse_number(lon)

    if lat_value is None or lon_value is None:
        return None

    if not (MIN_LAT <= lat_value <= MAX_LAT):
        return None

    if not (MIN_LON <= lon_value <= MAX_LON):
        return None

    return lat_value, lon_value


def point_wkt(lat: float, lon: float) -> str:
    """Create PostGIS-compatible WKT."""
    return f"POINT({lon:.8f} {lat:.8f})"


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    """Parse min_lon,min_lat,max_lon,max_lat."""
    parts = [float(x.strip()) for x in value.split(",")]

    if len(parts) != 4:
        raise ValueError(
            "CHENNAI_BBOX must be "
            "min_lon,min_lat,max_lon,max_lat"
        )

    min_lon, min_lat, max_lon, max_lat = parts

    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Invalid bounding box")

    return min_lon, min_lat, max_lon, max_lat


def point_inside_bbox(
    lat: float,
    lon: float,
    bbox: tuple[float, float, float, float],
) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox

    return (
        min_lon <= lon <= max_lon
        and min_lat <= lat <= max_lat
    )


def deterministic_key(record: PropertyRecord) -> str:
    """
    Create a stable identity for a property listing.

    The database schema does not currently have a source-record ID, so this
    fingerprint prevents duplicate inserts when the same source is processed
    repeatedly.
    """
    raw = "|".join(
        [
            normalize_text(record.name) or "",
            record.geometry,
            str(record.unit_size or ""),
            str(record.floor_area or ""),
            str(record.rent or ""),
            str(record.lease_term_years or ""),
            normalize_text(record.source) or "",
        ]
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def download_source(url: str) -> bytes:
    """Download a public CSV/JSON/HTML source."""
    logger.info("Downloading property source: %s", url)

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    logger.info(
        "Downloaded %.1f KB",
        len(response.content) / 1024,
    )

    return response.content


def load_local_source(path: str) -> bytes:
    """Load a local source file."""
    source_path = Path(path)

    if not source_path.exists():
        raise FileNotFoundError(f"Property source not found: {path}")

    logger.info("Reading local property source: %s", source_path)

    return source_path.read_bytes()


def detect_source_type(
    content: bytes,
    source_name: str,
) -> str:
    """Determine whether content is JSON, CSV or HTML."""
    lowered = source_name.lower()

    if lowered.endswith(".json"):
        return "json"

    if lowered.endswith(".csv"):
        return "csv"

    if lowered.endswith((".html", ".htm")):
        return "html"

    stripped = content.lstrip()

    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return "json"

    if b"<html" in stripped[:1000].lower():
        return "html"

    return "csv"


def load_records_from_source(
    content: bytes,
    source_name: str,
) -> list[dict[str, Any]]:
    """Load raw records from CSV, JSON or HTML."""
    source_type = detect_source_type(content, source_name)

    logger.info("Detected source format: %s", source_type)

    if source_type == "json":
        data = json.loads(content.decode("utf-8-sig"))

        if isinstance(data, dict):
            # Support common wrappers such as {"data": [...]}.
            for key in ("data", "results", "properties", "listings"):
                if isinstance(data.get(key), list):
                    return data[key]

            return [data]

        if isinstance(data, list):
            return data

        raise ValueError("Unsupported JSON structure")

    if source_type == "csv":
        dataframe = pd.read_csv(io.BytesIO(content))
        dataframe = dataframe.where(pd.notnull(dataframe), None)

        return dataframe.to_dict(orient="records")

    # HTML fallback.
    # This is intentionally generic because the TN tender portal does not
    # expose a stable property-listing schema.
    soup = BeautifulSoup(content, "html.parser")

    records: list[dict[str, Any]] = []

    for table in soup.find_all("table"):
        try:
            tables = pd.read_html(io.StringIO(str(table)))

            if not tables:
                continue

            dataframe = tables[0]
            dataframe = dataframe.where(pd.notnull(dataframe), None)

            records.extend(dataframe.to_dict(orient="records"))
        except (ValueError, ImportError):
            continue

    if records:
        return records

    # Last-resort extraction from page text. This will normally only produce
    # title/address information and will leave numeric property fields NULL.
    text = soup.get_text(" ", strip=True)

    if text:
        records.append({"name": text[:500]})

    return records


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def first_value(
    row: dict[str, Any],
    keys: Iterable[str],
) -> Any:
    """Return the first non-empty value among aliases."""
    for key in keys:
        if key in row:
            value = row[key]

            if normalize_text(value) is not None:
                return value

    return None


def extract_unit_size(row: dict[str, Any]) -> float | None:
    return parse_number(
        first_value(
            row,
            (
                "unit_size",
                "unit size",
                "shop_size",
                "shop size",
                "plot_size",
                "plot size",
                "carpet_area",
                "carpet area",
            ),
        )
    )


def extract_floor_area(row: dict[str, Any]) -> float | None:
    return parse_number(
        first_value(
            row,
            (
                "floor_area",
                "floor area",
                "built_up_area",
                "built up area",
                "built-up area",
                "area",
                "super_built_up_area",
            ),
        )
    )


def extract_rent(row: dict[str, Any]) -> float | None:
    return parse_number(
        first_value(
            row,
            (
                "rent",
                "monthly_rent",
                "monthly rent",
                "lease_rent",
                "lease rent",
                "minimum_rent",
                "minimum rent",
                "reserve_price",
                "reserve price",
            ),
        )
    )


def extract_lease_term(row: dict[str, Any]) -> float | None:
    value = first_value(
        row,
        (
            "lease_term_years",
            "lease term years",
            "lease_term",
            "lease term",
            "lease_period",
            "lease period",
        ),
    )

    if value is None:
        return None

    text = normalize_text(value)

    if not text:
        return None

    number = parse_number(text)

    if number is None:
        return None

    lowered = text.lower()

    if "month" in lowered:
        return number / 12.0

    if "day" in lowered:
        return number / 365.0

    return number


def extract_tenure(row: dict[str, Any]) -> str | None:
    return normalize_text(
        first_value(
            row,
            (
                "tenure_type",
                "tenure",
                "lease_type",
                "lease type",
                "property_type",
                "property type",
            ),
        )
    )


def normalize_record(
    row: dict[str, Any],
    source_name: str,
    bbox: tuple[float, float, float, float],
) -> PropertyRecord | None:
    """Convert a raw source record into our DB model."""

    coordinates = parse_lat_lon(row)

    if coordinates is None:
        logger.debug(
            "Skipping record without valid coordinates: %s",
            row.get("name") or row.get("title"),
        )
        return None

    lat, lon = coordinates

    if not point_inside_bbox(lat, lon, bbox):
        logger.debug(
            "Skipping record outside pilot bbox: %s, %s",
            lat,
            lon,
        )
        return None

    name = normalize_text(
        first_value(
            row,
            (
                "name",
                "property_name",
                "property name",
                "title",
                "shop_name",
                "shop name",
                "description",
            ),
        )
    )

    address = normalize_text(
        first_value(
            row,
            (
                "address",
                "location",
                "locality",
                "site_address",
            ),
        )
    )

    # If no explicit property name exists, use address as a useful fallback.
    if not name:
        name = address

    last_updated = normalize_text(
        first_value(
            row,
            (
                "last_updated",
                "last updated",
                "updated_at",
                "date",
                "notice_date",
                "published_date",
            ),
        )
    )

    record = PropertyRecord(
        name=name,
        geometry=point_wkt(lat, lon),
        unit_size=extract_unit_size(row),
        floor_area=extract_floor_area(row),
        rent=extract_rent(row),
        lease_term_years=extract_lease_term(row),
        tenure_type=extract_tenure(row),
        source=source_name,
        last_updated=last_updated,
    )

    return record


# ---------------------------------------------------------------------------
# Supabase persistence
# ---------------------------------------------------------------------------

def create_supabase_client() -> Client:
    """Create Supabase client from environment credentials."""
    require_environment()

    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
    )


def build_insert_payload(record: PropertyRecord) -> dict[str, Any]:
    """
    Build the property_listings insert payload.

    geometry is sent as WKT. Supabase/PostgREST accepts this when the
    underlying PostGIS column is configured appropriately; if the project
    exposes geometry as a PostGIS type requiring a different representation,
    change this field to the project's RPC/stored-function interface.
    """
    return {
        "name": record.name,
        "geometry": record.geometry,
        "unit_size": record.unit_size,
        "floor_area": record.floor_area,
        "rent": record.rent,
        "lease_term_years": record.lease_term_years,
        "tenure_type": record.tenure_type,
        "source": record.source,
        "last_updated": record.last_updated,
    }


def fetch_existing_fingerprints(
    client: Client,
    source: str,
) -> set[str]:
    """
    Retrieve existing records for the source.

    Because the current schema has no unique source-record ID, the ETL
    reconstructs fingerprints from existing rows.
    """
    response = (
        client.table("property_listings")
        .select(
            "name,geometry,unit_size,floor_area,rent,"
            "lease_term_years,tenure_type,source"
        )
        .eq("source", source)
        .execute()
    )

    fingerprints: set[str] = set()

    for row in response.data or []:
        geometry = row.get("geometry")

        # Depending on the PostgREST representation, geometry may be returned
        # as WKT or JSON. Only use WKT-like strings for fingerprinting.
        if not isinstance(geometry, str):
            continue

        existing = PropertyRecord(
            name=row.get("name"),
            geometry=geometry,
            unit_size=row.get("unit_size"),
            floor_area=row.get("floor_area"),
            rent=row.get("rent"),
            lease_term_years=row.get("lease_term_years"),
            tenure_type=row.get("tenure_type"),
            source=row.get("source") or source,
            last_updated=None,
        )

        fingerprints.add(deterministic_key(existing))

    return fingerprints


def insert_records(
    client: Client,
    records: list[PropertyRecord],
) -> tuple[int, int]:
    """Insert only records that are not already present."""
    if not records:
        return 0, 0

    source = records[0].source

    existing = fetch_existing_fingerprints(
        client,
        source,
    )

    inserted = 0
    skipped = 0

    for record in records:
        fingerprint = deterministic_key(record)

        if fingerprint in existing:
            skipped += 1
            continue

        payload = build_insert_payload(record)

        try:
            client.table("property_listings").insert(
                payload
            ).execute()

            existing.add(fingerprint)
            inserted += 1

        except Exception:
            logger.exception(
                "Failed to insert property listing: %s",
                record.name,
            )

    return inserted, skipped


# ---------------------------------------------------------------------------
# Main ETL
# ---------------------------------------------------------------------------

def run() -> int:
    logger.info("Starting property ETL")

    try:
        bbox = parse_bbox(BBOX)

        logger.info(
            "Pilot bounding box: "
            "lon=%s..%s lat=%s..%s",
            bbox[0],
            bbox[2],
            bbox[1],
            bbox[3],
        )

        # ------------------------------------------------------------------
        # Source selection
        # ------------------------------------------------------------------
        #
        # The official TN eProcurement portal is the public upstream source,
        # but it is not a structured property API. PROPERTY_SOURCE_URL or
        # PROPERTY_SOURCE_FILE should therefore point to a downloaded/exported
        # tender/auction dataset after filtering for commercial premises.
        #
        # This avoids scraping an unstable government portal UI directly and
        # makes the ETL deterministic and testable.
        # ------------------------------------------------------------------

        if SOURCE_FILE:
            source_name = f"tn_tenders:{Path(SOURCE_FILE).name}"
            content = load_local_source(SOURCE_FILE)

        elif SOURCE_URL:
            source_name = f"tn_tenders:{SOURCE_URL}"
            content = download_source(SOURCE_URL)

        else:
            raise RuntimeError(
                "No property source configured. Set either "
                "PROPERTY_SOURCE_FILE or PROPERTY_SOURCE_URL.\n"
                "The official free upstream source is the Tamil Nadu "
                "Government eProcurement / Tender Cum Auction portal:\n"
                "https://tntenders.gov.in/\n"
                "Because it does not expose a complete property-listing API, "
                "provide a downloaded/exported CSV, JSON or HTML notice set "
                "via one of the variables above."
            )

        raw_records = load_records_from_source(
            content,
            source_name,
        )

        logger.info(
            "Loaded %d raw source records",
            len(raw_records),
        )

        normalized: list[PropertyRecord] = []

        for row in raw_records:
            try:
                record = normalize_record(
                    row,
                    source_name,
                    bbox,
                )

                if record:
                    normalized.append(record)

            except Exception:
                logger.exception(
                    "Failed to normalize property source record"
                )

        logger.info(
            "Normalized %d records inside pilot region",
            len(normalized),
        )

        if not normalized:
            logger.warning(
                "No usable property listings found. "
                "This can be expected because the free TN tender source "
                "is incomplete and often lacks coordinates."
            )
            return 0

        client = create_supabase_client()

        inserted, skipped = insert_records(
            client,
            normalized,
        )

        logger.info(
            "Property ETL complete: inserted=%d skipped=%d",
            inserted,
            skipped,
        )

        return 0

    except requests.RequestException:
        logger.exception("Property source download failed")
        return 1

    except Exception:
        logger.exception("Property ETL failed")
        return 1


if __name__ == "__main__":
    sys.exit(run())