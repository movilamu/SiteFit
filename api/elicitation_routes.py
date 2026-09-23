from typing import Dict, List, Tuple
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from elicitation.consistency_ratio import (
    compute_consistency_ratio,
    is_acceptable,
    most_inconsistent_pairs,
    principal_eigenvalue,
)
from elicitation.pairwise_ahp import build_comparison_matrix, compute_weights_dict
from elicitation.swing_weighting import swing_weight

router = APIRouter(prefix="/elicitation", tags=["Elicitation"])


# =============================================================================
# Pydantic Schemas - AHP
# =============================================================================

class PairwiseJudgmentItem(BaseModel):
    criterion_a: str = Field(..., description="Name of the first criterion")
    criterion_b: str = Field(..., description="Name of the second criterion")
    ratio: float = Field(
        ...,
        gt=0,
        description="Ratio specifying how many times more important criterion_a is than criterion_b",
    )


class AHPRequest(BaseModel):
    criteria: List[str] = Field(
        ..., min_length=2, description="List of all criterion names being evaluated"
    )
    judgments: List[PairwiseJudgmentItem] = Field(
        ..., description="List of pairwise comparison judgments"
    )

    @field_validator("criteria")
    @classmethod
    def validate_unique_criteria(cls, criteria: List[str]) -> List[str]:
        if len(criteria) != len(set(criteria)):
            raise ValueError("Criteria names must be unique.")
        return criteria


class AHPResponse(BaseModel):
    weights: Dict[str, float]
    consistency_ratio: float
    consistency_index: float


# =============================================================================
# Pydantic Schemas - Swing Weighting
# =============================================================================

class SwingRequest(BaseModel):
    criteria_ranges: Dict[str, Tuple[float, float]] = Field(
        ..., description="Mapping of criterion name to (min_val, max_val) observed range"
    )
    swing_order: List[str] = Field(
        ...,
        min_length=1,
        description="List of criterion keys ordered from highest to lowest preference swing",
    )
    swing_ratings: Dict[str, float] = Field(
        ...,
        description="Map of criterion key to assigned swing rating (0 to 100 relative to top rank)",
    )


class SwingResponse(BaseModel):
    weights: Dict[str, float]


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/ahp", response_model=AHPResponse, status_code=status.HTTP_200_OK)
def calculate_ahp(payload: AHPRequest):
    pairwise_dict = {
        (item.criterion_a, item.criterion_b): item.ratio
        for item in payload.judgments
    }

    try:
        matrix = build_comparison_matrix(pairwise_dict, payload.criteria)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    try:
        lambda_max = principal_eigenvalue(matrix)
        ci, cr = compute_consistency_ratio(matrix, lambda_max)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    if not is_acceptable(cr):
        offending_pairs = most_inconsistent_pairs(matrix, payload.criteria, top_n=3)
        revisions = [
            {
                "criterion_a": a,
                "criterion_b": b,
                "current_ratio": round(
                    float(matrix[payload.criteria.index(a), payload.criteria.index(b)]), 4
                ),
                "log_deviation": round(dev, 4),
                "guidance": f"Revisit judgment between '{a}' and '{b}' to improve consistency.",
            }
            for a, b, dev in offending_pairs
        ]

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": (
                    f"Consistency ratio (CR = {cr:.4f}) is greater than or equal to 0.10. "
                    "Pairwise judgments are inconsistent and must be revised."
                ),
                "consistency_ratio": round(cr, 4),
                "consistency_index": round(ci, 4),
                "judgments_to_revise": revisions,
            },
        )

    weights = compute_weights_dict(matrix, payload.criteria)
    return AHPResponse(
        weights=weights,
        consistency_ratio=round(cr, 6),
        consistency_index=round(ci, 6),
    )


@router.post("/swing", response_model=SwingResponse, status_code=status.HTTP_200_OK)
def calculate_swing(payload: SwingRequest):
    try:
        weights = swing_weight(
            criteria_ranges=payload.criteria_ranges,
            swing_order=payload.swing_order,
            swing_ratings=payload.swing_ratings,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    return SwingResponse(weights=weights)