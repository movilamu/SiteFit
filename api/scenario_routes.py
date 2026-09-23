"""FastAPI router for saved MCA scenarios and scenario comparisons.

File 46 /api/scenario_routes.py

Scenarios are persisted in the ``saved_scenarios`` table created by
``infra/migrations/003_scenario_schema.sql``.  Ranking is deliberately
recomputed through File 32's ``apply_weights``/``rank_sites`` functions over
the already-normalized matrix; File 32 explicitly defines this as the cheap
per-interaction aggregation step.

The normalized matrix is supplied by ``get_normalized_matrix()``.  In
production, applications can replace that function with their File 31
artifact loader.  The default implementation supports the common
``NORMALIZED_MATRIX_FILE`` environment variable (CSV, Parquet, or JSON), and
also permits tests/integrations to assign ``NORMALIZED_MATRIX`` directly.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from supabase import Client, create_client

try:
    from scoring.score import apply_weights, rank_sites
except ImportError:  # Running as a loose collection of project files.
    from File_32 import apply_weights, rank_sites


router = APIRouter(prefix="", tags=["scenarios"])

SUPPORTED_METHODS = {
    "weighted_sum",
    "weighted_product",
    "distance_to_ideal",
}

# Optional integration/test hook.  File 31's persisted artifact can be
# assigned here by an application startup routine instead of using a file.
NORMALIZED_MATRIX: pd.DataFrame | None = None


class ScenarioCreate(BaseModel):
    """Payload for creating a saved MCA scenario."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=255)
    weight_set: dict[str, float] = Field(..., min_length=1)
    aggregation_method: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("aggregation_method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        value = value.strip().lower().replace("-", "_").replace(" ", "_")
        if value not in SUPPORTED_METHODS:
            raise ValueError(
                "aggregation_method must be one of: "
                + ", ".join(sorted(SUPPORTED_METHODS))
            )
        return value

    @field_validator("weight_set")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("weight_set must contain at least one criterion")

        cleaned: dict[str, float] = {}
        for criterion, raw_weight in value.items():
            key = str(criterion).strip()
            if not key:
                raise ValueError("weight_set contains a blank criterion name")
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Weight for criterion {key!r} must be numeric"
                ) from exc
            if not math.isfinite(weight):
                raise ValueError(f"Weight for criterion {key!r} must be finite")
            if weight < 0:
                raise ValueError(f"Weight for criterion {key!r} must be non-negative")
            cleaned[key] = weight

        total = sum(cleaned.values())
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"weight_set weights must sum to 1.0; got {total:.12g}")

        return cleaned


class ScenarioResponse(BaseModel):
    """Serialized saved scenario metadata."""

    scenario_id: int
    name: str
    weight_set: dict[str, Any]
    aggregation_method: str
    created_at: Any
    updated_at: Any


class ScenarioWithRanking(ScenarioResponse):
    """Saved scenario plus its freshly recomputed site ranking."""

    ranking: list[dict[str, Any]]


class ScenarioComparison(BaseModel):
    """Side-by-side ranking rows for two or more saved scenarios."""

    scenarios: list[dict[str, Any]]
    ranking: list[dict[str, Any]]


def get_supabase_client() -> Client:
    """Create the Supabase client using the project's standard environment variables."""

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)


def get_normalized_matrix() -> pd.DataFrame:
    """Return the File 31 normalized decision matrix.

    File 32 expects an already-normalized, already-constraint-filtered matrix;
    this function intentionally does not normalize raw site data.

    The default file loader is useful for a simple deployment or tests:

    * ``NORMALIZED_MATRIX`` may be assigned directly to a DataFrame.
    * ``NORMALIZED_MATRIX_FILE`` may point to CSV, Parquet, or JSON.

    A File 31-specific persistence loader can replace this function without
    changing any route or scoring logic.
    """

    if NORMALIZED_MATRIX is not None:
        matrix = NORMALIZED_MATRIX.copy()
    else:
        configured_path = os.getenv("NORMALIZED_MATRIX_FILE")
        if not configured_path:
            from fastapi import HTTPException, status as _status

            raise HTTPException(
                status_code=_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "No normalized matrix is configured. Set NORMALIZED_MATRIX_FILE "
                    "or provide File 31's matrix through get_normalized_matrix()."
                ),
            )

        path = Path(configured_path)
        if not path.exists():
            raise RuntimeError(f"Normalized matrix file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            matrix = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            matrix = pd.read_parquet(path)
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            matrix = pd.DataFrame(payload)
        else:
            raise RuntimeError(
                "Unsupported NORMALIZED_MATRIX_FILE format. Use CSV, Parquet, or JSON."
            )

    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)

    if matrix.empty:
        raise RuntimeError("The normalized matrix contains no site rows")

    # If a persisted matrix contains a site_id column, make it the DataFrame
    # index so File 32 returns meaningful site IDs in rank_sites().
    if "site_id" in matrix.columns:
        matrix = matrix.set_index("site_id")

    return matrix


def _serialize_value(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-compatible values."""

    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def _ranking_for_scenario(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run File 32 scoring for one saved scenario and return JSON rows."""

    matrix = get_normalized_matrix()
    scores = apply_weights(
        matrix,
        scenario["weight_set"],
        method=scenario["aggregation_method"],
    )
    ranked = rank_sites(scores)

    records: list[dict[str, Any]] = []
    for row in ranked.to_dict(orient="records"):
        records.append({key: _serialize_value(value) for key, value in row.items()})
    return records


def _scenario_or_404(client: Client, scenario_id: int) -> dict[str, Any]:
    """Fetch a scenario or raise a FastAPI 404."""

    response = (
        client.table("saved_scenarios")
        .select("scenario_id,name,weight_set,aggregation_method,created_at,updated_at")
        .eq("scenario_id", scenario_id)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
    return rows[0]


@router.post("/scenarios", response_model=ScenarioResponse, status_code=201)
def create_scenario(payload: ScenarioCreate) -> dict[str, Any]:
    """Save a named MCA scenario."""

    try:
        client = get_supabase_client()
        response = (
            client.table("saved_scenarios")
            .insert(
                {
                    "name": payload.name,
                    "weight_set": payload.weight_set,
                    "aggregation_method": payload.aggregation_method,
                }
            )
            .execute()
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to save scenario") from exc

    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=500, detail="Scenario was not returned after insert")
    return rows[0]


@router.get("/scenarios", response_model=list[ScenarioResponse])
def list_scenarios() -> list[dict[str, Any]]:
    """List saved scenarios, newest first."""

    try:
        client = get_supabase_client()
        response = (
            client.table("saved_scenarios")
            .select("scenario_id,name,weight_set,aggregation_method,created_at,updated_at")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to list scenarios") from exc


@router.get("/scenarios/compare", response_model=ScenarioComparison)
def compare_scenarios(
    ids: str = Query(
        ...,
        description="Comma-separated scenario IDs; at least two are required.",
        min_length=3,
    ),
) -> dict[str, Any]:
    """Re-score two or more scenarios and return their rankings side by side."""

    raw_ids = [item.strip() for item in ids.split(",") if item.strip()]
    if len(raw_ids) < 2:
        raise HTTPException(status_code=400, detail="ids must contain at least two scenario IDs")

    try:
        scenario_ids = [int(item) for item in raw_ids]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="ids must be comma-separated integers") from exc

    if len(set(scenario_ids)) != len(scenario_ids):
        raise HTTPException(status_code=400, detail="ids must not contain duplicates")

    try:
        client = get_supabase_client()
        response = (
            client.table("saved_scenarios")
            .select("scenario_id,name,weight_set,aggregation_method,created_at,updated_at")
            .in_("scenario_id", scenario_ids)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to load scenarios") from exc

    found = {int(row["scenario_id"]): row for row in (response.data or [])}
    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in found]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario(s) not found: {', '.join(map(str, missing))}",
        )

    scenario_rows = [found[scenario_id] for scenario_id in scenario_ids]

    try:
        rankings_by_scenario: dict[int, list[dict[str, Any]]] = {
            int(scenario["scenario_id"]): _ranking_for_scenario(scenario)
            for scenario in scenario_rows
        }
    except (ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Side-by-side shape: one row per site, with rank/score columns for every
    # requested scenario. The same site IDs are aligned across scenarios.
    by_site: dict[str, dict[str, Any]] = {}
    for scenario in scenario_rows:
        scenario_id = int(scenario["scenario_id"])
        for row in rankings_by_scenario[scenario_id]:
            site_id = str(row["site_id"])
            output = by_site.setdefault(site_id, {"site_id": row["site_id"]})
            output[f"scenario_{scenario_id}_rank"] = row["rank"]
            output[f"scenario_{scenario_id}_score"] = row["score"]

    return {
        "scenarios": scenario_rows,
        "ranking": list(by_site.values()),
    }


@router.get("/scenarios/{scenario_id}", response_model=ScenarioWithRanking)
def get_scenario(scenario_id: int) -> dict[str, Any]:
    """Return one scenario and its ranking recomputed through File 32."""

    try:
        client = get_supabase_client()
        scenario = _scenario_or_404(client, scenario_id)
        ranking = _ranking_for_scenario(scenario)
    except HTTPException:
        raise
    except (ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to load scenario") from exc

    return {**scenario, "ranking": ranking}


__all__ = [
    "router",
    "ScenarioCreate",
    "ScenarioResponse",
    "ScenarioWithRanking",
    "ScenarioComparison",
    "get_supabase_client",
    "get_normalized_matrix",
]
