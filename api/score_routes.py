"""
api/score_routes.py

FastAPI router for multi-criteria site scoring.

Request/Response Schema Convention:
----------------------------------
This router establishes the standardized request payload structure for passing criterion weights
and selecting aggregation methods across the platform.

Consumed by:
- File 54 (Comparison View): Calls POST /score with custom/adjusted weights to compare side-by-side rankings.
- File 56 (Sensitivity View): Repeatedly calls POST /score while perturbing individual criteria weights to generate sensitivity curves.

Request Body (POST /score):
{
    "weights": {
        "resident_density": 0.4,
        "rent": 0.3,
        "competitor_density": 0.3
    },
    "method": "weighted_sum"  # Options: "weighted_sum" | "weighted_product" | "distance_to_ideal"
}

Response Body (200 OK):
{
    "method": "weighted_sum",
    "sites": [
        {"rank": 1, "site_id": "site_101", "score": 0.842},
        {"rank": 2, "site_id": "site_102", "score": 0.715},
        ...
    ]
}
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Sequence, Union

from fastapi import APIRouter, HTTPException, status
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Fallback imports to support module-style package import or direct script execution
try:
    from scoring.score import apply_weights, rank_sites
except ImportError:
    from File_32 import apply_weights, rank_sites


router = APIRouter(prefix="", tags=["scoring"])


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    """
    Payload for scoring sites using normalized global weights.
    
    This structure is shared by File 54 (comparison) and File 56 (sensitivity).
    """
    weights: Dict[str, float] = Field(
        ...,
        description="Map of criterion IDs to non-negative global weights. Weights must sum to ~1.0.",
        example={
            "resident_density": 0.4,
            "rent": 0.3,
            "competitor_density": 0.3,
        },
    )
    method: Literal["weighted_sum", "weighted_product", "distance_to_ideal"] = Field(
        default="weighted_sum",
        description="Aggregation method algorithm from File 32.",
    )

    @field_validator("weights")
    @classmethod
    def validate_weights_sum(cls, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate that weights are non-negative and sum to approximately 1.0 (tol=0.01)."""
        if not weights:
            raise ValueError("The 'weights' mapping cannot be empty.")

        for k, v in weights.items():
            if v < 0.0:
                raise ValueError(f"Weight for criterion '{k}' must be non-negative (got {v}).")

        total = sum(weights.values())
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(
                f"Weights must sum to 1.0 (±0.01). Current total sum is {total:.4f}."
            )

        return weights


class RankedSiteResult(BaseModel):
    rank: int = Field(..., description="1-based ranking position (1 is best).")
    site_id: Union[str, int] = Field(..., description="Unique site identifier.")
    score: float = Field(..., description="Computed aggregation score.")


class ScoreResponse(BaseModel):
    method: str = Field(..., description="The aggregation method used for scoring.")
    sites: List[RankedSiteResult] = Field(..., description="Ranked list of sites ordered best-first.")


# ---------------------------------------------------------------------------
# File 31 normalized matrix — same loader as File 46 (no random mock data)
# ---------------------------------------------------------------------------

def _get_precomputed_matrix(criteria_keys: Sequence[str]) -> pd.DataFrame:
    """Load the File 31 precomputed, normalized matrix used by File 46 scoring."""
    try:
        from api.scenario_routes import get_normalized_matrix
    except ImportError:
        try:
            from scenario_routes import get_normalized_matrix
        except ImportError:
            from File_46 import get_normalized_matrix

    matrix = get_normalized_matrix()
    missing = [key for key in criteria_keys if key not in matrix.columns]
    if missing:
        raise ValueError(
            "Requested criteria are not in the normalized matrix: "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(map(str, matrix.columns))
        )
    return matrix.loc[:, list(criteria_keys)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute site scores and rankings",
    description=(
        "Accepts criterion weights and an aggregation method. "
        "Validates that weights sum to ~1.0, applies the aggregation algorithm via File 32, "
        "and returns ranked sites sorted descending by score."
    ),
)
def compute_scores(request: ScoreRequest) -> ScoreResponse:
    try:
        # 1. Fetch File 31 precomputed matrix matching requested criteria
        criteria_list = list(request.weights.keys())
        normalized_matrix = _get_precomputed_matrix(criteria_list)

        # 2. Apply weights via File 32 aggregation engine
        scores_series = apply_weights(
            normalized_matrix=normalized_matrix,
            global_weights=request.weights,
            method=request.method,
        )

        # 3. Sort and rank sites descending by score
        ranked_df = rank_sites(scores_series)

        # 4. Transform output into standard JSON schema response
        sites_list = [
            RankedSiteResult(
                rank=int(row["rank"]),
                site_id=str(row["site_id"]),
                score=round(float(row["score"]), 6),
            )
            for _, row in ranked_df.iterrows()
        ]

        return ScoreResponse(method=request.method, sites=sites_list)

    except ValueError as err:
        # Catch unexpected mathematical or alignment errors and present as 422
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(err),
        ) from err
    except RuntimeError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while scoring: {str(err)}",
        ) from err