"""FastAPI router for site score decomposition endpoints."""

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from scoring.score import METHODS, _align_weights
except ImportError:
    METHODS = {"weighted_sum": None, "weighted_product": None, "distance_to_ideal": None}

router = APIRouter(tags=["decomposition"])


class CriterionDecomposition(BaseModel):
    """Breakdown for an individual criterion for a specific site."""

    criterion: str = Field(..., description="Criterion identifier or column name.")
    normalized_value: float = Field(..., description="Normalized performance value in [0, 1].")
    weight: float = Field(..., description="Assigned weight for this criterion.")
    contribution: float = Field(..., description="Calculated contribution to total score.")


class SiteDecompositionResponse(BaseModel):
    """Response model containing site score decomposition details."""

    site_id: str = Field(..., description="Unique identifier of the site.")
    method: str = Field(..., description="Aggregation method used.")
    total_score: float = Field(..., description="Aggregated total score for the site.")
    criteria: List[CriterionDecomposition] = Field(
        ..., description="Per-criterion decomposition items."
    )


def _load_normalized_matrix():
    """Load the File 31 normalized matrix used by File 44/46 scoring."""
    try:
        from api.scenario_routes import get_normalized_matrix
    except ImportError:
        try:
            from scenario_routes import get_normalized_matrix
        except ImportError:
            from File_46 import get_normalized_matrix
    return get_normalized_matrix()


def _load_site_data(site_id: str) -> Optional[Dict[str, float]]:
    """Look up a site's precomputed normalized criterion values from File 31."""
    try:
        matrix = _load_normalized_matrix()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Normalized matrix is unavailable: {exc}",
        ) from exc

    row = None
    if site_id in matrix.index:
        row = matrix.loc[site_id]
    else:
        for idx in matrix.index:
            if str(idx) == str(site_id):
                row = matrix.loc[idx]
                break
    if row is None:
        return None
    values: Dict[str, float] = {}
    for criterion, value in row.items():
        try:
            values[str(criterion)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _parse_weights(weights_param: str) -> Dict[str, float]:
    """Parse JSON string weights parameter into standard dictionary."""
    try:
        parsed = json.loads(weights_param)
        if not isinstance(parsed, dict):
            raise ValueError("Weights JSON must represent an object/mapping.")
        return {str(k): float(v) for k, v in parsed.items()}
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid weights query parameter. Must be valid JSON mapping: {err}",
        )


@router.get(
    "/sites/{site_id}/decomposition",
    response_model=SiteDecompositionResponse,
    summary="Get per-criterion score breakdown for a specific site",
)
def get_site_decomposition(
    site_id: str,
    weights: str = Query(
        ...,
        description='JSON string of leaf criterion weights, e.g. {"accessibility":0.4,"demographics":0.6}',
    ),
    method: str = Query(
        "weighted_sum",
        description="Aggregation method (weighted_sum, weighted_product, distance_to_ideal)",
    ),
) -> SiteDecompositionResponse:
    """
    Calculate per-criterion breakdown (normalized value, weight, contribution)
    for a given site based on specified leaf weights and aggregation method.
    """
    normalized_key = method.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_key not in METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown aggregation method '{method}'. Choose from: {sorted(METHODS.keys())}",
        )

    site_values = _load_site_data(site_id)
    if site_values is None:
        raise HTTPException(status_code=404, detail=f"Site with id '{site_id}' not found.")

    weights_dict = _parse_weights(weights)

    criteria_list: List[CriterionDecomposition] = []
    total_score = 0.0

    for criterion, norm_val in site_values.items():
        weight = float(weights_dict.get(criterion, 0.0))

        if normalized_key == "weighted_sum":
            contribution = norm_val * weight
        elif normalized_key == "weighted_product":
            contribution = (norm_val ** weight) if norm_val > 0 else 0.0
        elif normalized_key == "distance_to_ideal":
            contribution = weight * ((1.0 - norm_val) ** 2)
        else:
            contribution = norm_val * weight

        total_score += contribution
        criteria_list.append(
            CriterionDecomposition(
                criterion=criterion,
                normalized_value=norm_val,
                weight=weight,
                contribution=contribution,
            )
        )

    return SiteDecompositionResponse(
        site_id=site_id,
        method=normalized_key,
        total_score=total_score,
        criteria=criteria_list,
    )