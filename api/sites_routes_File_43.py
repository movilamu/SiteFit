"""
api/sites_routes.py

FastAPI router exposing candidate site inventory:

1. GET /sites — all candidate sites with coordinates and basic info.
   Optional query params map to File 26 hard-constraint filters:
   min_unit_size, max_rent, required_tenure.

2. Query-parameter validation with explicit 400 error messages
   (invalid numbers, negatives, empty tenure lists).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

# =============================================================================
# File 26: scoring/constraints.py
# =============================================================================

try:
    from scoring.constraints import apply_hard_constraints
except ImportError:
    try:
        from constraints import apply_hard_constraints
    except ImportError:
        try:
            from File_26 import apply_hard_constraints
        except ImportError:
            apply_hard_constraints = None  # type: ignore[assignment]


# =============================================================================
# Response schemas
# =============================================================================

class SiteCoordinates(BaseModel):
    latitude: float
    longitude: float


class SiteBasicInfo(BaseModel):
    site_id: Any
    name: Optional[str] = None
    coordinates: SiteCoordinates
    unit_size: Optional[float] = None
    rent: Optional[float] = None
    tenure: Optional[str] = None
    status: Optional[str] = None
    attractiveness: Optional[float] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SitesListResponse(BaseModel):
    count: int
    filters_applied: Dict[str, Any]
    sites: List[SiteBasicInfo]
    excluded: List[Dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Query-parameter validation
# =============================================================================

def _parse_optional_nonnegative_float(raw: Optional[str], param_name: str) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid '{param_name}': expected a number, got {raw!r}. "
                "Example: ?min_unit_size=1500"
            ),
        )
    if not math.isfinite(value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid '{param_name}': value must be a finite number, got {raw!r}.",
        )
    if value < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid '{param_name}': must be greater than or equal to 0, got {value}.",
        )
    return value


def _parse_optional_tenure(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None

    values: List[str]
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid 'required_tenure': could not parse JSON list. "
                    "Use a comma-separated string (e.g. lease,freehold) "
                    f"or a JSON array (e.g. [\"lease\",\"freehold\"]). Parse error: {exc}."
                ),
            )
        if not isinstance(parsed, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid 'required_tenure': JSON value must be an array of strings.",
            )
        values = [str(item).strip() for item in parsed]
    else:
        values = [part.strip() for part in text.split(",")]

    values = [v for v in values if v]
    if not values:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid 'required_tenure': provide at least one tenure type. "
                "Example: ?required_tenure=lease,freehold"
            ),
        )
    return values


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return _json_safe(value.item())
        except (ValueError, AttributeError):
            pass
    return value if isinstance(value, (dict, list)) else str(value)


def _first_present(row: pd.Series, keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if key in row.index:
            val = _json_safe(row[key])
            if val is not None and val != "":
                return val
    return None


def _extract_lat_lon(row: pd.Series) -> Optional[Tuple[float, float]]:
    lat = _first_present(row, ("latitude", "lat", "y"))
    lon = _first_present(row, ("longitude", "lon", "lng", "x"))
    try:
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    except (TypeError, ValueError):
        pass

    geom = None
    for key in ("geometry", "centroid", "coordinates"):
        if key in row.index:
            geom = row[key]
            break
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
                try:
                    return float(parts[1]), float(parts[0])
                except ValueError:
                    return None

    if isinstance(geom, dict):
        if geom.get("type") == "Feature":
            geom = geom.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Point" and coords and len(coords) >= 2:
            return float(coords[1]), float(coords[0])
        if "latitude" in geom and "longitude" in geom:
            return float(geom["latitude"]), float(geom["longitude"])
        if "lat" in geom and "lon" in geom:
            return float(geom["lat"]), float(geom["lon"])
    return None


def _expand_metadata_column(df: pd.DataFrame) -> pd.DataFrame:
    if "metadata" not in df.columns:
        return df
    extra_rows: List[Dict[str, Any]] = []
    for value in df["metadata"]:
        parsed: Dict[str, Any] = {}
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = None
        if isinstance(value, dict):
            parsed = value
        extra_rows.append(parsed)
    extra = pd.DataFrame(extra_rows, index=df.index)
    for col in extra.columns:
        if col not in df.columns:
            df[col] = extra[col]
    return df


def _dataframe_from_records(records: Any) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, list):
        return pd.DataFrame(records)
    if isinstance(records, dict):
        if "sites" in records and isinstance(records["sites"], list):
            return pd.DataFrame(records["sites"])
        return pd.DataFrame([records])
    raise TypeError(f"Unsupported sites payload type: {type(records)!r}")


def _load_from_supabase() -> Optional[pd.DataFrame]:
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
    if not url or not key:
        return None

    try:
        import requests
    except ImportError:
        return None

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    endpoint = f"{url.rstrip('/')}/rest/v1/candidate_sites"
    params = {
        "select": "*",
        "order": "id.asc",
        "status": "eq.active",
    }
    try:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            resp = requests.get(
                endpoint,
                headers=headers,
                params={"select": "*", "order": "id.asc"},
                timeout=30,
            )
        if resp.status_code != 200:
            return None
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return None
        df = pd.DataFrame(rows)
        return _expand_metadata_column(df)
    except Exception:
        return None


def get_sites_dataframe(request: Request) -> pd.DataFrame:
    """Load candidate sites from app state, a provider module, or Supabase."""
    for attr in ("sites_df", "candidate_sites", "sites"):
        if hasattr(request.app.state, attr):
            val = getattr(request.app.state, attr)
            if val is not None:
                df = _dataframe_from_records(val)
                if not df.empty:
                    return _expand_metadata_column(df)

    for loader_path in (
        ("scoring.data", "load_candidate_sites"),
        ("api.data", "load_candidate_sites"),
        ("ingestion.generate_catchments", "load_candidate_sites"),
    ):
        module_name, func_name = loader_path
        try:
            module = __import__(module_name, fromlist=[func_name])
            loader = getattr(module, func_name)
            df = _dataframe_from_records(loader())
            if not df.empty:
                return _expand_metadata_column(df)
        except Exception:
            continue

    supabase_df = _load_from_supabase()
    if supabase_df is not None and not supabase_df.empty:
        return supabase_df

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            "Candidate site inventory is not available. "
            "Set request.app.state.sites_df (or candidate_sites) to a DataFrame / list of site records, "
            "or configure SUPABASE_URL and SUPABASE_KEY so candidate_sites can be loaded."
        ),
    )


def _site_to_payload(row: pd.Series) -> Optional[SiteBasicInfo]:
    coords = _extract_lat_lon(row)
    if coords is None:
        return None
    lat, lon = coords

    site_id = _first_present(row, ("site_id", "id", "site_identifier"))
    if site_id is None:
        site_id = row.name

    reserved = {
        "site_id",
        "id",
        "name",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "lng",
        "y",
        "x",
        "unit_size",
        "unit_size_sqft",
        "floor_area",
        "floor_area_sqm",
        "rent",
        "rent_per_sqft",
        "monthly_rent",
        "tenure_type",
        "tenure",
        "lease_type",
        "status",
        "attractiveness",
        "geometry",
        "centroid",
        "coordinates",
        "metadata",
    }
    attributes = {}
    for key, value in row.items():
        if key in reserved:
            continue
        safe = _json_safe(value)
        if safe is not None:
            attributes[str(key)] = safe

    unit_size = _first_present(row, ("unit_size", "unit_size_sqft", "floor_area", "floor_area_sqm"))
    rent = _first_present(row, ("rent", "monthly_rent", "rent_per_sqft"))
    tenure = _first_present(row, ("tenure_type", "tenure", "lease_type"))
    attractiveness = _first_present(row, ("attractiveness",))

    def _as_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            number = float(val)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    return SiteBasicInfo(
        site_id=site_id,
        name=_first_present(row, ("name", "site_name", "store_name")),
        coordinates=SiteCoordinates(latitude=float(lat), longitude=float(lon)),
        unit_size=_as_float(unit_size),
        rent=_as_float(rent),
        tenure=str(tenure) if tenure is not None else None,
        status=_first_present(row, ("status",)),
        attractiveness=_as_float(attractiveness),
        attributes=attributes,
    )


# =============================================================================
# Router
# =============================================================================

router = APIRouter(tags=["Sites"])


@router.get(
    "/sites",
    response_model=SitesListResponse,
    summary="List candidate sites",
    response_description="Candidate sites with coordinates and basic property info.",
)
def list_sites(
    request: Request,
    min_unit_size: Optional[str] = Query(
        None,
        description="File 26 hard constraint: minimum acceptable floor area / unit size.",
        examples=["1500"],
    ),
    max_rent: Optional[str] = Query(
        None,
        description="File 26 hard constraint: maximum acceptable rent cap.",
        examples=["250000"],
    ),
    required_tenure: Optional[str] = Query(
        None,
        description=(
            "File 26 hard constraint: allowed tenure type(s). "
            "Comma-separated (lease,freehold) or a JSON array."
        ),
        examples=["lease,freehold"],
    ),
) -> SitesListResponse:
    parsed_min_size = _parse_optional_nonnegative_float(min_unit_size, "min_unit_size")
    parsed_max_rent = _parse_optional_nonnegative_float(max_rent, "max_rent")
    parsed_tenures = _parse_optional_tenure(required_tenure)

    filters_applied: Dict[str, Any] = {}
    constraints: Dict[str, Any] = {}
    if parsed_min_size is not None:
        constraints["min_unit_size"] = parsed_min_size
        filters_applied["min_unit_size"] = parsed_min_size
    if parsed_max_rent is not None:
        constraints["max_rent"] = parsed_max_rent
        filters_applied["max_rent"] = parsed_max_rent
    if parsed_tenures is not None:
        constraints["required_tenure"] = parsed_tenures
        filters_applied["required_tenure"] = parsed_tenures

    sites_df = get_sites_dataframe(request)
    excluded: List[Dict[str, Any]] = []

    if constraints:
        if apply_hard_constraints is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Hard-constraint filtering is unavailable because scoring.constraints "
                    "(File 26) could not be imported."
                ),
            )
        id_column = "site_id" if "site_id" in sites_df.columns else (
            "id" if "id" in sites_df.columns else "site_id"
        )
        try:
            sites_df, excluded = apply_hard_constraints(
                sites_df,
                constraints,
                id_column=id_column,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to apply constraint filters: {exc}",
            )

    sites: List[SiteBasicInfo] = []
    skipped_without_coords = 0
    for _, row in sites_df.iterrows():
        payload = _site_to_payload(row)
        if payload is None:
            skipped_without_coords += 1
            continue
        sites.append(payload)

    if skipped_without_coords and not sites:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Candidate site records were found but none have usable coordinates "
                "(latitude/longitude, lat/lon, or Point geometry)."
            ),
        )

    return SitesListResponse(
        count=len(sites),
        filters_applied=filters_applied,
        sites=sites,
        excluded=excluded,
    )