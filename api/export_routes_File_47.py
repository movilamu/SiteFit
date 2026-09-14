"""FastAPI router for reproducible ranked-shortlist exports.

File 47 /api/export_routes.py

Exports are JSON rather than CSV because JSON can carry the ranked shortlist,
the exact global weight mapping, aggregation method, scenario metadata, and
export metadata in one self-describing artifact.  This makes an exported
recommendation reproducible instead of preserving only the final ranking.

The router reuses File 46's scenario lookup / normalized-matrix loading and
File 32's apply_weights/rank_sites implementation, so exports use exactly the
same scoring path as the scenario API.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from supabase import Client, create_client

try:
    from api.scenario_routes import (
        SUPPORTED_METHODS,
        _ranking_for_scenario,
        get_supabase_client,
    )
except ImportError:
    try:
        from scenario_routes import (
            SUPPORTED_METHODS,
            _ranking_for_scenario,
            get_supabase_client,
        )
    except ImportError:
        # Loose-file execution fallback.  File 46 is expected to be available
        # alongside this module in a normal project checkout.
        from File_46 import (
            SUPPORTED_METHODS,
            _ranking_for_scenario,
            get_supabase_client,
        )


router = APIRouter(prefix="", tags=["exports"])

# JSON is preferred over CSV here because the weights and ranking metadata can
# be represented exactly and atomically in one portable, human-readable file.
DEFAULT_EXPORT_BUCKET = os.getenv("SUPABASE_EXPORT_BUCKET", "exports")


class ExportRequest(BaseModel):
    """Request for a reproducible shortlist export.

    Exactly one source of weights is required:
      * scenario_id loads the persisted scenario from ``saved_scenarios``; or
      * weight_set supplies an inline weight mapping.

    Inline weights also need an aggregation method because a weight set alone
    does not uniquely determine the scoring operation.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: int | None = Field(default=None, ge=1)
    weight_set: dict[str, float] | None = Field(default=None, min_length=1)
    aggregation_method: str = "weighted_sum"

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
    def validate_weights(cls, value: dict[str, float] | None):
        if value is None:
            return value

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

            if not (weight >= 0.0) or weight != weight or weight in (
                float("inf"),
                float("-inf"),
            ):
                raise ValueError(
                    f"Weight for criterion {key!r} must be finite and non-negative"
                )
            cleaned[key] = weight

        total = sum(cleaned.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"weight_set weights must sum to 1.0; got {total:.12g}"
            )
        return cleaned

    @field_validator("weight_set", mode="after")
    @classmethod
    def require_exactly_one_source(
        cls, value: dict[str, float] | None, info
    ) -> dict[str, float] | None:
        scenario_id = info.data.get("scenario_id")
        if (scenario_id is None) == (value is None):
            raise ValueError(
                "Provide exactly one of scenario_id or weight_set"
            )
        return value


class ExportResponse(BaseModel):
    """Reference to the generated export artifact."""

    export_id: Any
    scenario_id: int | None
    format: str
    storage_path: str
    download_url: str
    weight_set: dict[str, float]
    aggregation_method: str
    row_count: int


def _inline_scenario(payload: ExportRequest) -> dict[str, Any]:
    """Build a File-46-compatible scenario object for inline weights."""
    assert payload.weight_set is not None
    return {
        "scenario_id": None,
        "name": "inline-export",
        "weight_set": payload.weight_set,
        "aggregation_method": payload.aggregation_method,
        "created_at": None,
        "updated_at": None,
    }


def _load_scenario(client: Client, payload: ExportRequest) -> dict[str, Any]:
    if payload.scenario_id is None:
        return _inline_scenario(payload)

    response = (
        client.table("saved_scenarios")
        .select(
            "scenario_id,name,weight_set,aggregation_method,created_at,updated_at"
        )
        .eq("scenario_id", payload.scenario_id)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario {payload.scenario_id} not found",
        )
    return rows[0]


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar values without leaking non-JSON values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _make_export_document(
    scenario: dict[str, Any],
    ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the self-contained, reproducible JSON export payload."""
    return {
        "export_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "scenario": {
            "scenario_id": scenario.get("scenario_id"),
            "name": scenario.get("name"),
            "aggregation_method": scenario["aggregation_method"],
            # Critical reproducibility field: never export a ranking without it.
            "weight_set": _json_safe(scenario["weight_set"]),
        },
        "ranking": _json_safe(ranking),
    }


def _upload_and_get_url(
    client: Client,
    storage_path: str,
    content: bytes,
) -> str:
    bucket = os.getenv("SUPABASE_EXPORT_BUCKET", DEFAULT_EXPORT_BUCKET)

    # Supabase Storage upload is intentionally performed before the database
    # insert: export_records should never point at an artifact that failed to
    # upload.
    client.storage.from_(bucket).upload(
        storage_path,
        content,
        {
            "content-type": "application/json",
            "upsert": "false",
        },
    )

    # Prefer a signed URL so the export bucket can remain private.  The
    # lifetime is configurable; one hour is a sensible default for downloads.
    expires_in = int(os.getenv("SUPABASE_EXPORT_URL_TTL_SECONDS", "3600"))
    signed = client.storage.from_(bucket).create_signed_url(
        storage_path, expires_in
    )

    if isinstance(signed, dict):
        url = signed.get("signedURL") or signed.get("signedUrl")
        if url:
            return str(url)

    # Some supabase-py versions expose a public URL method instead.  This is
    # useful when the configured bucket is public.
    try:
        public = client.storage.from_(bucket).get_public_url(storage_path)
        if isinstance(public, str):
            return public
        if isinstance(public, dict):
            url = public.get("publicURL") or public.get("publicUrl")
            if url:
                return str(url)
    except Exception:
        pass

    raise RuntimeError("Supabase Storage did not return a download URL")


def _record_export(
    client: Client,
    *,
    scenario_id: int | None,
    storage_path: str,
    download_url: str,
) -> dict[str, Any]:
    """Persist the export reference in File 8's export_records table.

    File 8's export record is deliberately kept separate from the JSON
    artifact: Storage contains the immutable payload while this table provides
    an application-level audit/reference record.
    """
    response = (
        client.table("export_records")
        .insert(
            {
                "scenario_id": scenario_id,
                "storage_path": storage_path,
                "file_format": "json",
                "download_url": download_url,
            }
        )
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise RuntimeError("Export record was not returned after insert")
    return rows[0]


@router.post("/export", response_model=ExportResponse, status_code=201)
def export_ranked_shortlist(payload: ExportRequest) -> dict[str, Any]:
    """Generate, store, and register a reproducible ranked shortlist export.

    The exported JSON contains:
      * the ranked ``site_id`` / ``rank`` / ``score`` rows;
      * the exact global ``weight_set`` used to generate them;
      * the aggregation method; and
      * scenario metadata.

    ``scenario_id`` exports reuse the saved scenario.  Inline exports are
    scored directly from the supplied weight set.
    """
    client: Client | None = None
    storage_path: str | None = None

    try:
        client = get_supabase_client()
        scenario = _load_scenario(client, payload)

        # File 46 delegates ranking to File 32's apply_weights/rank_sites, so
        # this endpoint cannot silently use a different scoring implementation.
        ranking = _ranking_for_scenario(scenario)
        document = _make_export_document(scenario, ranking)

        content = (
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        scenario_part = (
            f"scenario-{scenario['scenario_id']}"
            if scenario.get("scenario_id") is not None
            else "inline"
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        storage_path = f"{scenario_part}/{timestamp}.json"

        download_url = _upload_and_get_url(
            client,
            storage_path,
            content,
        )

        try:
            record = _record_export(
                client,
                scenario_id=scenario.get("scenario_id"),
                storage_path=storage_path,
                download_url=download_url,
            )
        except Exception:
            # Avoid leaving an orphaned Storage object when the audit record
            # cannot be written.
            try:
                client.storage.from_(
                    os.getenv("SUPABASE_EXPORT_BUCKET", DEFAULT_EXPORT_BUCKET)
                ).remove([storage_path])
            except Exception:
                pass
            raise

        export_id = record.get("export_id", record.get("id"))

        return {
            "export_id": export_id,
            "scenario_id": scenario.get("scenario_id"),
            "format": "json",
            "storage_path": storage_path,
            "download_url": download_url,
            "weight_set": scenario["weight_set"],
            "aggregation_method": scenario["aggregation_method"],
            "row_count": len(ranking),
        }

    except HTTPException:
        raise
    except (ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to create export") from exc


__all__ = [
    "router",
    "ExportRequest",
    "ExportResponse",
    "export_ranked_shortlist",
]
