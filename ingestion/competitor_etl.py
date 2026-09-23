#!/usr/bin/env python3
"""
Ingest retail/shop POIs from OpenStreetMap Overpass into Supabase/PostGIS.

Environment variables:
    SUPABASE_URL       Supabase project URL
    SUPABASE_KEY       Supabase API key (service-role key recommended for ETL)
    OVERPASS_URL       Optional Overpass endpoint
    PILOT_BBOX         Optional "south,west,north,east" bounding box

Example:
    PILOT_BBOX="12.80,80.05,13.25,80.35" python ingestion/competitor_etl.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from supabase import Client, create_client
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: supabase. Install with: "
        "pip install requests supabase"
    ) from exc


DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Chennai / Greater Chennai pilot fallback. Prefer setting PILOT_BBOX explicitly
# for the exact pilot sub-region.
DEFAULT_BBOX = "12.80,80.05,13.25,80.35"

# Retail/shop POI tags relevant to competitor/site-selection analysis.
SHOP_TYPES = (
    "supermarket",
    "convenience",
    "department_store",
    "general",
    "mall",
    "wholesale",
    "bakery",
    "butcher",
    "clothes",
    "electronics",
    "furniture",
    "hardware",
    "doityourself",
    "mobile_phone",
    "shoes",
    "sports",
    "books",
    "chemist",
    "beauty",
    "cosmetics",
    "gift",
    "jewelry",
    "optician",
    "stationery",
    "variety_store",
    "pet",
    "computer",
    "car",
    "car_parts",
)

LOGGER = logging.getLogger("competitor_etl")


@dataclass(frozen=True)
class Config:
    supabase_url: str
    supabase_key: str
    overpass_url: str
    bbox: str
    request_timeout: int = 90
    max_attempts: int = 5
    base_backoff_seconds: float = 3.0


def load_config() -> Config:
    missing = [
        name
        for name in ("SUPABASE_URL", "SUPABASE_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )

    return Config(
        supabase_url=os.environ["SUPABASE_URL"].rstrip("/"),
        supabase_key=os.environ["SUPABASE_KEY"],
        overpass_url=os.getenv("OVERPASS_URL", DEFAULT_OVERPASS_URL),
        bbox=os.getenv("PILOT_BBOX", DEFAULT_BBOX),
    )


def validate_bbox(bbox: str) -> str:
    try:
        values = [float(v.strip()) for v in bbox.split(",")]
    except ValueError as exc:
        raise ValueError(
            "PILOT_BBOX must be 'south,west,north,east'"
        ) from exc

    if len(values) != 4:
        raise ValueError("PILOT_BBOX must contain exactly four coordinates")

    south, west, north, east = values
    if not (-90 <= south < north <= 90):
        raise ValueError("Invalid latitude bounds in PILOT_BBOX")
    if not (-180 <= west < east <= 180):
        raise ValueError("Invalid longitude bounds in PILOT_BBOX")

    return ",".join(f"{v:g}" for v in values)


def build_overpass_query(bbox: str) -> str:
    shop_regex = "|".join(SHOP_TYPES)

    # Query nodes, ways and relations because OSM shops can be represented
    # by any of these element types. For ways/relations, Overpass derives a
    # representative center point.
    return f"""
[out:json][timeout:60];
(
  nwr["shop"~"^({shop_regex})$"]({bbox});
  nwr["shop"]["name"]({bbox});
);
out center tags;
""".strip()


def build_http_session() -> requests.Session:
    session = requests.Session()

    # Retry transient HTTP failures. 429 is handled explicitly below so that
    # we can respect Retry-After when Overpass provides it.
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "TamilNaduSiteSelection/1.0 "
            "(OpenStreetMap Overpass ETL)",
            "Accept": "application/json",
        }
    )
    return session


def query_overpass(
    session: requests.Session,
    config: Config,
    query: str,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            LOGGER.info(
                "Querying Overpass (attempt %d/%d)",
                attempt,
                config.max_attempts,
            )

            response = session.post(
                config.overpass_url,
                data={"data": query},
                timeout=config.request_timeout,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = max(
                        1.0, float(retry_after)
                    ) if retry_after else config.base_backoff_seconds * (2 ** (attempt - 1))
                except ValueError:
                    wait_seconds = config.base_backoff_seconds * (2 ** (attempt - 1))

                if attempt == config.max_attempts:
                    response.raise_for_status()

                LOGGER.warning(
                    "Overpass rate limited (429); sleeping %.1fs",
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code >= 500:
                response.raise_for_status()

            response.raise_for_status()
            payload = response.json()
            return payload.get("elements", [])

        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == config.max_attempts:
                break

            wait_seconds = config.base_backoff_seconds * (2 ** (attempt - 1))
            LOGGER.warning(
                "Overpass request failed: %s; retrying in %.1fs",
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Overpass query failed after {config.max_attempts} attempts"
    ) from last_error


def element_coordinates(element: dict[str, Any]) -> tuple[float, float] | None:
    if element.get("type") == "node":
        lat = element.get("lat")
        lon = element.get("lon")
    else:
        center = element.get("center") or {}
        lat = center.get("lat")
        lon = center.get("lon")

    if lat is None or lon is None:
        return None

    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def normalize_element(element: dict[str, Any]) -> dict[str, Any] | None:
    tags = element.get("tags") or {}
    name = (tags.get("name") or "").strip()

    # The table requires a name, so unnamed OSM shop features are skipped.
    if not name:
        return None

    coords = element_coordinates(element)
    if coords is None:
        return None

    lat, lon = coords
    shop_type = (tags.get("shop") or "retail").strip()

    return {
        "name": name[:255],
        "category": shop_type[:100],
        "geometry": f"SRID=4326;POINT({lon} {lat})",
        "source": "osm",
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate OSM records returned through overlapping tag clauses.

    The same place may be returned by both the explicit shop-type clause and
    the generic named-shop clause. Use name/category/geometry as the ETL key.
    """
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []

    for row in rows:
        key = (row["name"], row["category"], row["geometry"])
        if key not in seen:
            seen.add(key)
            result.append(row)

    return result


def write_to_supabase(
    client: Client,
    rows: list[dict[str, Any]],
    batch_size: int = 500,
) -> int:
    """
    Insert records using Supabase/PostgREST.

    geometry is supplied as EWKT and converted to PostGIS geometry by the
    database column type.
    """
    inserted = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]

        try:
            client.table("competitor_locations").insert(batch).execute()
            inserted += len(batch)
            LOGGER.info(
                "Inserted %d/%d competitor locations",
                inserted,
                len(rows),
            )
        except Exception:
            LOGGER.exception(
                "Failed inserting batch starting at row %d",
                start,
            )
            raise

    return inserted


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config()
        config = Config(
            **{
                **config.__dict__,
                "bbox": validate_bbox(config.bbox),
            }
        )

        LOGGER.info("Using pilot bounding box: %s", config.bbox)
        LOGGER.info("Overpass endpoint: %s", config.overpass_url)

        query = build_overpass_query(config.bbox)
        session = build_http_session()

        elements = query_overpass(session, config, query)
        LOGGER.info("Overpass returned %d elements", len(elements))

        rows = [
            row
            for element in elements
            if (row := normalize_element(element)) is not None
        ]
        rows = deduplicate(rows)

        LOGGER.info(
            "Prepared %d named retail/shop POIs after normalization/deduplication",
            len(rows),
        )

        if not rows:
            LOGGER.warning("No named retail/shop POIs found; nothing to insert")
            return 0

        client = create_client(config.supabase_url, config.supabase_key)
        inserted = write_to_supabase(client, rows)

        LOGGER.info("ETL completed successfully: %d rows inserted", inserted)
        return 0

    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return 130
    except Exception:
        LOGGER.exception("Competitor ETL failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
