"""
ingestion/isochrones.py
=======================
OpenRouteService (ORS) isochrone client with Supabase cache + daily quota
guard, per ADR-002.

Cache table: isochrone_cache
    UNIQUE (lat, lon, profile, range_value)
    range_value is stored as travel time in seconds (range_minutes * 60).

Quota table (create once in Supabase if it does not exist):

    CREATE TABLE IF NOT EXISTS ors_daily_usage (
        usage_date DATE PRIMARY KEY,          -- UTC calendar date
        request_count INTEGER NOT NULL DEFAULT 0,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


ORS_ISOCHRONES_URL = "https://api.openrouteservice.org/v2/isochrones/{profile}"
ORS_MAX_LOCATIONS_PER_REQUEST = 5
ORS_MAX_INTERVALS_PER_REQUEST = 10
ORS_MAX_DISTANCE_M = 120_000
ORS_MAX_DRIVING_RANGE_SECONDS = 3_600  # 1 hour (free-tier driving profiles)

DAILY_REQUEST_SOFT_CAP = 2_000  # buffer under the 2,500/day free-tier cap
COORD_DECIMALS = 5  # ~1 m; ADR-002 cache key precision

PROFILE_ALIASES = {
    "driving": "driving-car",
    "car": "driving-car",
    "driving-car": "driving-car",
    "driving-hgv": "driving-hgv",
    "hgv": "driving-hgv",
    "walking": "foot-walking",
    "foot": "foot-walking",
    "foot-walking": "foot-walking",
    "foot-hiking": "foot-hiking",
    "hiking": "foot-hiking",
    "cycling": "cycling-regular",
    "cycling-regular": "cycling-regular",
    "cycling-road": "cycling-road",
    "cycling-mountain": "cycling-mountain",
    "cycling-electric": "cycling-electric",
    "wheelchair": "wheelchair",
}

DRIVING_PROFILES = frozenset({"driving-car", "driving-hgv"})


class IsochroneError(RuntimeError):
    """Raised for cache, quota, validation, or ORS failures."""


class OrsDailyQuotaExceeded(IsochroneError):
    """Raised when today's persisted ORS request count reaches the soft cap."""


def _require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise IsochroneError(f"{name} is not set.")
    return value


def _ors_api_key() -> str:
    return _require_env("ORS_API_KEY")


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
        raise IsochroneError(
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


def normalize_profile(profile: str) -> str:
    if not profile or not str(profile).strip():
        raise IsochroneError("profile is required.")
    key = str(profile).strip().lower()
    if key == "transit":
        raise IsochroneError(
            "ORS has no transit profile. Use driving-car, foot-walking, or cycling-regular."
        )
    mapped = PROFILE_ALIASES.get(key)
    if mapped is None:
        raise IsochroneError(
            f"Unsupported profile {profile!r}. "
            f"Use one of: {', '.join(sorted(set(PROFILE_ALIASES.values())))}."
        )
    return mapped


def round_coord(value: float) -> float:
    return round(float(value), COORD_DECIMALS)


def minutes_to_range_value(range_minutes: float) -> int:
    if range_minutes is None or float(range_minutes) <= 0:
        raise IsochroneError("range_minutes must be a positive number.")
    return int(round(float(range_minutes) * 60.0))


def _validate_request(profile: str, range_seconds: int) -> None:
    if range_seconds > ORS_MAX_DRIVING_RANGE_SECONDS and profile in DRIVING_PROFILES:
        raise IsochroneError(
            f"Driving isochrones are capped at {ORS_MAX_DRIVING_RANGE_SECONDS}s "
            f"(1 hour) on the ORS free tier; got {range_seconds}s."
        )
    if range_seconds > ORS_MAX_DRIVING_RANGE_SECONDS:
        # Keep non-driving time ranges inside the same 1h product cap unless
        # explicitly using distance mode (not used by get_isochrone).
        raise IsochroneError(
            f"range_minutes exceeds the 1 hour maximum ({range_seconds}s)."
        )


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def get_daily_ors_request_count(usage_date: Optional[str] = None) -> int:
    """Return persisted ORS request count for a UTC date (default: today)."""
    day = usage_date or _utc_today()
    resp = requests.get(
        _rest("ors_daily_usage"),
        headers=_supabase_headers(),
        params={"usage_date": f"eq.{day}", "select": "request_count"},
        timeout=20,
    )
    if resp.status_code == 404 or (
        resp.status_code == 200 and resp.text in ("", "[]")
    ):
        return 0
    if resp.status_code != 200:
        raise IsochroneError(
            f"Failed to read ors_daily_usage ({resp.status_code}): {resp.text}"
        )
    rows = resp.json()
    if not rows:
        return 0
    return int(rows[0].get("request_count") or 0)


def _assert_quota_available() -> int:
    count = get_daily_ors_request_count()
    if count >= DAILY_REQUEST_SOFT_CAP:
        raise OrsDailyQuotaExceeded(
            f"ORS daily quota soft-cap reached: {count}/{DAILY_REQUEST_SOFT_CAP} "
            f"requests used on {_utc_today()} UTC. Remaining work should be queued "
            f"for the next UTC day (free tier is 2,500/day)."
        )
    return count


def increment_daily_ors_request_count(n: int = 1) -> int:
    """Persist +n against today's UTC usage row. Returns the new count."""
    if n < 1:
        raise IsochroneError("increment n must be >= 1.")
    day = _utc_today()
    current = get_daily_ors_request_count(day)
    new_count = current + n
    payload = {
        "usage_date": day,
        "request_count": new_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    resp = requests.post(
        _rest("ors_daily_usage"),
        headers=_supabase_headers(prefer="resolution=merge-duplicates,return=representation"),
        json=payload,
        timeout=20,
    )
    if resp.status_code not in (200, 201):
        raise IsochroneError(
            f"Failed to update ors_daily_usage ({resp.status_code}): {resp.text}"
        )
    body = resp.json()
    if isinstance(body, list) and body:
        return int(body[0]["request_count"])
    if isinstance(body, dict) and "request_count" in body:
        return int(body["request_count"])
    return new_count


def _lookup_cache(
    lat: float, lon: float, profile: str, range_value: int
) -> Optional[Dict[str, Any]]:
    resp = requests.get(
        _rest("isochrone_cache"),
        headers=_supabase_headers(),
        params={
            "select": "id,lat,lon,profile,range_value,geometry,created_at",
            "lat": f"eq.{lat}",
            "lon": f"eq.{lon}",
            "profile": f"eq.{profile}",
            "range_value": f"eq.{range_value}",
            "limit": "1",
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise IsochroneError(
            f"isochrone_cache lookup failed ({resp.status_code}): {resp.text}"
        )
    rows = resp.json()
    return rows[0] if rows else None


def _geojson_to_polygon(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ORS GeoJSON to a Polygon (schema is GEOMETRY(Polygon, 4326))."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon":
        return {"type": "Polygon", "coordinates": coords}
    if gtype == "MultiPolygon":
        if not coords:
            raise IsochroneError("ORS returned an empty MultiPolygon.")
        largest = max(coords, key=lambda poly: abs(_ring_area(poly[0])) if poly else 0.0)
        return {"type": "Polygon", "coordinates": largest}
    raise IsochroneError(f"Unsupported ORS geometry type: {gtype!r}")


def _ring_area(ring: List[List[float]]) -> float:
    area = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[i + 1][0], ring[i + 1][1]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _feature_for_range(
    fc: Dict[str, Any], range_seconds: int
) -> Dict[str, Any]:
    features = fc.get("features") or []
    if not features:
        raise IsochroneError("ORS isochrone response contained no features.")
    for feat in features:
        props = feat.get("properties") or {}
        value = props.get("value")
        if value is None:
            contour = (props.get("contour") or props.get("group_index"))
            if contour is not None:
                value = contour
        if value is not None and int(round(float(value))) == int(range_seconds):
            return feat
    return features[-1]


def _call_ors_isochrones(
    locations: List[Tuple[float, float]],
    profile: str,
    range_seconds_list: List[int],
) -> Dict[str, Any]:
    if not (1 <= len(locations) <= ORS_MAX_LOCATIONS_PER_REQUEST):
        raise IsochroneError(
            f"ORS allows 1–{ORS_MAX_LOCATIONS_PER_REQUEST} locations per request; "
            f"got {len(locations)}."
        )
    if not (1 <= len(range_seconds_list) <= ORS_MAX_INTERVALS_PER_REQUEST):
        raise IsochroneError(
            f"ORS allows 1–{ORS_MAX_INTERVALS_PER_REQUEST} intervals per request; "
            f"got {len(range_seconds_list)}."
        )
    for rs in range_seconds_list:
        _validate_request(profile, rs)

    body = {
        "locations": [[lon, lat] for lat, lon in locations],
        "range": range_seconds_list,
        "range_type": "time",
        "location_type": "start",
    }
    headers = {
        "Authorization": _ors_api_key(),
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json, application/geo+json",
    }
    url = ORS_ISOCHRONES_URL.format(profile=profile)
    resp = requests.post(url, json=body, headers=headers, timeout=60)
    if resp.status_code == 429:
        raise IsochroneError(
            f"ORS rate-limited the request (429): {resp.text}"
        )
    if resp.status_code >= 400:
        raise IsochroneError(
            f"ORS isochrone request failed ({resp.status_code}): {resp.text}"
        )
    return resp.json()


def _store_cache(
    lat: float,
    lon: float,
    profile: str,
    range_value: int,
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "lat": lat,
        "lon": lon,
        "profile": profile,
        "range_value": range_value,
        "geometry": geometry,
    }
    resp = requests.post(
        _rest("isochrone_cache"),
        headers=_supabase_headers(
            prefer="resolution=merge-duplicates,return=representation"
        ),
        json=payload,
        timeout=20,
    )
    if resp.status_code not in (200, 201):
        raise IsochroneError(
            f"Failed to write isochrone_cache ({resp.status_code}): {resp.text}"
        )
    body = resp.json()
    if isinstance(body, list) and body:
        return body[0]
    if isinstance(body, dict):
        return body
    return payload


def get_isochrone(
    lat: float,
    lon: float,
    profile: str,
    range_minutes: float,
) -> Dict[str, Any]:
    """
    Return a travel-time isochrone polygon for a single origin.

    Cache-first against isochrone_cache on
    (lat, lon, profile, range_value) where range_value is seconds.
    On miss, calls ORS (1 location, 1 interval) and persists the polygon.
    Cache hits do not consume daily ORS quota.
    """
    ors_profile = normalize_profile(profile)
    q_lat = round_coord(lat)
    q_lon = round_coord(lon)
    range_value = minutes_to_range_value(range_minutes)
    _validate_request(ors_profile, range_value)

    cached = _lookup_cache(q_lat, q_lon, ors_profile, range_value)
    if cached:
        cached["cache_hit"] = True
        cached["routing_source"] = "ors"
        return cached

    _assert_quota_available()
    fc = _call_ors_isochrones(
        locations=[(q_lat, q_lon)],
        profile=ors_profile,
        range_seconds_list=[range_value],
    )
    increment_daily_ors_request_count(1)

    feature = _feature_for_range(fc, range_value)
    polygon = _geojson_to_polygon(feature.get("geometry") or {})
    stored = _store_cache(q_lat, q_lon, ors_profile, range_value, polygon)
    stored["cache_hit"] = False
    stored["routing_source"] = "ors"
    return stored


__all__ = [
    "DAILY_REQUEST_SOFT_CAP",
    "IsochroneError",
    "OrsDailyQuotaExceeded",
    "get_daily_ors_request_count",
    "get_isochrone",
    "increment_daily_ors_request_count",
    "normalize_profile",
]