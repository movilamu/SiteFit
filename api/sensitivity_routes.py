"""
api/sensitivity_routes.py

FastAPI router exposing sensitivity analysis endpoints for retail site selection:
1. GET /sensitivity/thresholds   - File 38 critical threshold report
2. GET /sensitivity/stability    - File 39 rank stability report
3. GET /sensitivity/near-ties    - File 40 near-tie groups
4. GET /sensitivity/cross-method - File 30 cross-method agreement summary

All endpoints consistently accept the current weight vector as input via:
- JSON request body: e.g. {"weights": {"accessibility": 0.4, ...}} or {"accessibility": 0.4, ...}
- Query parameter: e.g. ?weights={"accessibility":0.4,...} or ?accessibility=0.4&demographics=0.6
- App state fallback: request.app.state.current_weights / request.app.state.weights
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

# =============================================================================
# Robust Dependency Imports (with internal fallbacks matching Files 30,32,38,39,40)
# =============================================================================

# --- File 32: Scoring Base ---
try:
    from scoring.score import apply_weights, rank_sites, _unwrap_precomputed_matrix
except ImportError:
    try:
        from score import apply_weights, rank_sites, _unwrap_precomputed_matrix
    except ImportError:
        try:
            from File_32 import apply_weights, rank_sites, _unwrap_precomputed_matrix
        except ImportError:
            def _unwrap_precomputed_matrix(matrix: Any):
                if hasattr(matrix, "matrix") and hasattr(matrix, "columns"):
                    arr = np.asarray(getattr(matrix, "matrix"), dtype=float)
                    cols = list(getattr(matrix, "columns"))
                    idx = list(getattr(matrix, "index", [f"site_{i}" for i in range(len(arr))]))
                    return arr, idx, cols
                if isinstance(matrix, pd.DataFrame):
                    return matrix.to_numpy(dtype=float), list(matrix.index), list(matrix.columns)
                arr = np.asarray(matrix, dtype=float)
                return arr, [f"site_{i}" for i in range(arr.shape[0])], [f"c{j}" for j in range(arr.shape[1])]

            def apply_weights(normalized_matrix: Any, weights: Any, method: str = "weighted_sum") -> pd.Series:
                arr, index, columns = _unwrap_precomputed_matrix(normalized_matrix)
                if isinstance(weights, Mapping):
                    w = np.asarray([weights.get(col, 0.0) for col in columns], dtype=float)
                else:
                    w = np.asarray(weights, dtype=float).reshape(-1)
                total = w.sum()
                if total > 0:
                    w = w / total

                if method == "weighted_sum":
                    scores = arr @ w
                elif method == "weighted_product":
                    eps = 1e-12
                    clipped = np.clip(arr, eps, None)
                    scores = np.exp(np.log(clipped) @ w)
                elif method == "distance_to_ideal":
                    ideal = arr.max(axis=0)
                    nadir = arr.min(axis=0)
                    d_pos = np.sqrt(np.sum(w * (arr - ideal) ** 2, axis=1))
                    d_neg = np.sqrt(np.sum(w * (arr - nadir) ** 2, axis=1))
                    denom = d_pos + d_neg
                    scores = np.where(denom > 0, d_neg / denom, 0.0)
                else:
                    scores = arr @ w
                return pd.Series(scores, index=index, name="score")

            def rank_sites(scores: Any) -> pd.DataFrame:
                if isinstance(scores, pd.DataFrame) and "score" in scores.columns:
                    df = scores.copy()
                    if "site_id" not in df.columns:
                        df["site_id"] = df.index
                elif isinstance(scores, pd.Series):
                    df = pd.DataFrame({"site_id": scores.index, "score": scores.values})
                else:
                    arr = np.asarray(scores, dtype=float).reshape(-1)
                    df = pd.DataFrame({"site_id": [f"site_{i}" for i in range(len(arr))], "score": arr})
                df = df.sort_values(by="score", ascending=False, kind="mergesort").reset_index(drop=True)
                df["rank"] = np.arange(1, len(df) + 1)
                return df[["rank", "site_id", "score"]]


# --- File 38: Critical Thresholds ---
try:
    from sensitivity.critical_thresholds import compute_all_thresholds, find_critical_threshold
except ImportError:
    try:
        from critical_thresholds import compute_all_thresholds, find_critical_threshold
    except ImportError:
        try:
            from File_38 import compute_all_thresholds, find_critical_threshold
        except ImportError:
            def _top_site(normalized_matrix, weights, method: str):
                scores = apply_weights(normalized_matrix, weights, method=method)
                ranked = rank_sites(scores)
                top_row = ranked.iloc[0]
                return top_row["site_id"], float(top_row["score"])

            def _as_weight_array(current_weights, columns: Sequence[str]) -> np.ndarray:
                if isinstance(current_weights, Mapping):
                    if all(col in current_weights for col in columns):
                        return np.asarray([current_weights[col] for col in columns], dtype=float)
                    return np.asarray(list(current_weights.values()), dtype=float)
                return np.asarray(current_weights, dtype=float).reshape(-1)

            def _rescale_weights(weights: np.ndarray, idx: int, new_value: float) -> np.ndarray:
                weights = np.asarray(weights, dtype=float)
                total = weights.sum()
                other_sum = total - weights[idx]
                new_weights = weights.copy()
                remaining = total - new_value
                if other_sum <= 0:
                    new_weights[idx] = new_value
                    return new_weights
                factor = remaining / other_sum
                new_weights = weights * factor
                new_weights[idx] = new_value
                return new_weights

            def find_critical_threshold(normalized_matrix, current_weights, criterion_to_vary, method="weighted_sum", coarse_steps=40):
                _, _, columns = _unwrap_precomputed_matrix(normalized_matrix)
                weights = _as_weight_array(current_weights, columns)
                if isinstance(criterion_to_vary, str):
                    if criterion_to_vary not in columns:
                        raise ValueError(f"Unknown criterion {criterion_to_vary!r}")
                    idx = columns.index(criterion_to_vary)
                    crit_name = criterion_to_vary
                else:
                    idx = int(criterion_to_vary)
                    crit_name = columns[idx] if idx < len(columns) else f"c{idx}"

                total_mass = float(weights.sum())
                current_value = float(weights[idx])
                baseline_top, _ = _top_site(normalized_matrix, weights, method)
                best = None

                for direction, bound in (("increase", total_mass), ("decrease", 0.0)):
                    if np.isclose(current_value, bound):
                        continue
                    grid = np.linspace(current_value, bound, coarse_steps + 1)[1:]
                    prev_v, prev_t = current_value, baseline_top
                    cand = None
                    for val in grid:
                        tw = _rescale_weights(weights, idx, val)
                        ts, _ = _top_site(normalized_matrix, tw, method)
                        if ts != prev_t:
                            lo, hi = prev_v, val
                            lo_t, hi_t = prev_t, ts
                            for _ in range(40):
                                if abs(hi - lo) < 1e-6:
                                    break
                                mid = (lo + hi) / 2.0
                                mw = _rescale_weights(weights, idx, mid)
                                mt, _ = _top_site(normalized_matrix, mw, method)
                                if mt == lo_t:
                                    lo = mid
                                else:
                                    hi = mid
                                    hi_t = mt
                            cand = (hi, hi_t)
                            break
                        prev_v, prev_t = val, ts

                    if cand is not None:
                        tw, nt = cand
                        delta = tw - current_value
                        res = {
                            "criterion": crit_name, "current_weight": current_value,
                            "current_top_site": baseline_top, "direction": direction,
                            "threshold_weight": tw, "delta": delta,
                            "new_top_site": nt, "method": method,
                        }
                        if best is None or abs(delta) < abs(best["delta"]):
                            best = res

                if best is not None:
                    return best
                return {
                    "criterion": crit_name, "current_weight": current_value,
                    "current_top_site": baseline_top, "direction": None,
                    "threshold_weight": None, "delta": None,
                    "new_top_site": None, "method": method,
                }

            def compute_all_thresholds(normalized_matrix, current_weights, method="weighted_sum", coarse_steps=40) -> pd.DataFrame:
                _, _, columns = _unwrap_precomputed_matrix(normalized_matrix)
                rows = []
                for col in columns:
                    res = find_critical_threshold(normalized_matrix, current_weights, criterion_to_vary=col, method=method, coarse_steps=coarse_steps)
                    rows.append(res if isinstance(res, dict) else res.to_dict())
                report = pd.DataFrame(rows)
                report["_abs_delta"] = report["delta"].abs()
                report = report.sort_values(by="_abs_delta", ascending=True, na_position="last", kind="mergesort").drop(columns="_abs_delta")
                return report.reset_index(drop=True)


# --- File 39: Rank Stability ---
try:
    from sensitivity.stability import rank_stability_report, sample_weight_perturbations
except ImportError:
    try:
        from stability import rank_stability_report, sample_weight_perturbations
    except ImportError:
        try:
            from File_39 import rank_stability_report, sample_weight_perturbations
        except ImportError:
            def sample_weight_perturbations(current_weights, n_samples: int = 200, perturbation_size: float = 0.1):
                if n_samples < 0 or not 0 <= perturbation_size <= 1:
                    raise ValueError("Invalid n_samples or perturbation_size")
                if isinstance(current_weights, Mapping):
                    weights = np.asarray(list(current_weights.values()), dtype=float)
                else:
                    weights = np.asarray(current_weights, dtype=float).reshape(-1)
                total = weights.sum()
                if total <= 0:
                    raise ValueError("Total weights must be positive")
                weights = weights / total
                if n_samples == 0:
                    return np.empty((0, weights.size), dtype=float)
                factors = np.random.uniform(1.0 - perturbation_size, 1.0 + perturbation_size, size=(n_samples, weights.size))
                samples = weights[None, :] * factors
                samples /= samples.sum(axis=1, keepdims=True)
                return samples

            def rank_stability_report(normalized_matrix, current_weights, n_samples: int = 200, top_n: Optional[int] = None) -> pd.DataFrame:
                baseline_scores = apply_weights(normalized_matrix, current_weights)
                baseline_ranked = rank_sites(baseline_scores)
                if top_n is not None:
                    top_n = min(int(top_n), len(baseline_ranked))
                    target = set(baseline_ranked.iloc[:top_n]["site_id"])
                    crit = f"top_{top_n}"
                else:
                    target = dict(zip(baseline_ranked["site_id"], baseline_ranked["rank"]))
                    crit = "exact_rank"

                w = list(current_weights.values()) if isinstance(current_weights, Mapping) else current_weights
                perturbations = sample_weight_perturbations(w, n_samples=n_samples)
                hit_counts = pd.Series(0, index=baseline_ranked["site_id"], dtype=int)

                for pw in perturbations:
                    sc = apply_weights(normalized_matrix, pw)
                    rk = rank_sites(sc)
                    if top_n is not None:
                        ptop = set(rk.iloc[:top_n]["site_id"])
                        for sid in target:
                            if sid in ptop:
                                hit_counts.loc[sid] += 1
                    else:
                        pranks = dict(zip(rk["site_id"], rk["rank"]))
                        for sid, brank in target.items():
                            if pranks.get(sid) == brank:
                                hit_counts.loc[sid] += 1

                pct = (hit_counts.to_numpy(dtype=float) / n_samples * 100.0) if n_samples > 0 else np.zeros(len(hit_counts))
                rep = baseline_ranked[["site_id", "rank"]].rename(columns={"rank": "current_rank"}).copy()
                rep["stability_percentage"] = pct
                rep["stability_criterion"] = crit
                if top_n is not None:
                    rep["top_n"] = top_n
                return rep.reset_index(drop=True)


# --- File 40: Near-Ties ---
try:
    from sensitivity.near_ties import identify_near_ties, derive_uncertainty_threshold
except ImportError:
    try:
        from near_ties import identify_near_ties, derive_uncertainty_threshold
    except ImportError:
        try:
            from File_40 import identify_near_ties, derive_uncertainty_threshold
        except ImportError:
            def identify_near_ties(ranked_scores: pd.DataFrame, uncertainty_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
                if not isinstance(ranked_scores, pd.DataFrame):
                    raise TypeError("ranked_scores must be a DataFrame")
                ranked = ranked_scores.sort_values("rank").reset_index(drop=True)
                scores = ranked["score"].to_numpy(dtype=float)
                if uncertainty_threshold is None:
                    score_range = float(scores.max() - scores.min()) if len(scores) > 0 else 0.0
                    uncertainty_threshold = max(score_range * 0.005, 1e-6)
                if uncertainty_threshold < 0:
                    raise ValueError("uncertainty_threshold must be non-negative")

                clusters = []
                cur_sites = [ranked.iloc[0]["site_id"]]
                cur_scores = [float(ranked.iloc[0]["score"])]
                c_rank = int(ranked.iloc[0]["rank"])

                for i in range(1, len(ranked)):
                    s = float(ranked.iloc[i]["score"])
                    n_min = min(min(cur_scores), s)
                    n_max = max(max(cur_scores), s)
                    if (n_max - n_min) <= uncertainty_threshold:
                        cur_sites.append(ranked.iloc[i]["site_id"])
                        cur_scores.append(s)
                    else:
                        clusters.append({
                            "cluster_rank": c_rank,
                            "site_ids": cur_sites,
                            "scores": cur_scores,
                            "score_range": max(cur_scores) - min(cur_scores),
                        })
                        cur_sites = [ranked.iloc[i]["site_id"]]
                        cur_scores = [s]
                        c_rank = int(ranked.iloc[i]["rank"])
                clusters.append({
                    "cluster_rank": c_rank,
                    "site_ids": cur_sites,
                    "scores": cur_scores,
                    "score_range": max(cur_scores) - min(cur_scores),
                })
                return clusters

            def derive_uncertainty_threshold(score_std: Sequence[float], percentile: float = 50.0) -> float:
                arr = np.asarray(score_std, dtype=float)
                if arr.size == 0:
                    raise ValueError("score_std cannot be empty")
                return float(np.percentile(arr, percentile))


# --- File 30: Cross-Method Agreement ---
try:
    from scoring.cross_method_agreement import compare_rankings
except ImportError:
    try:
        from cross_method_agreement import compare_rankings
    except ImportError:
        try:
            from File_30 import compare_rankings
        except ImportError:
            def _to_ranks(scores: np.ndarray) -> np.ndarray:
                order = np.argsort(-scores, kind="mergesort")
                ranks = np.empty(len(scores), dtype=float)
                sorted_scores = scores[order]
                i, n = 0, len(scores)
                while i < n:
                    j = i
                    while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
                        j += 1
                    avg_rank = (i + 1 + j + 1) / 2.0
                    for k in range(i, j + 1):
                        ranks[order[k]] = avg_rank
                    i = j + 1
                return ranks

            def _spearman(ra: np.ndarray, rb: np.ndarray) -> float:
                if len(ra) < 2:
                    return 1.0
                a = ra - ra.mean()
                b = rb - rb.mean()
                denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
                return float(np.sum(a * b) / denom) if denom > 0 else 1.0

            def compare_rankings(weighted_sum_result, weighted_product_result, distance_to_ideal_result, site_ids: Sequence = None, rank_diff_threshold: int = 3) -> Dict[str, Any]:
                ws = np.asarray(weighted_sum_result, dtype=float).reshape(-1)
                wp = np.asarray(weighted_product_result, dtype=float).reshape(-1)
                dti = np.asarray(distance_to_ideal_result, dtype=float).reshape(-1)
                n = ws.size
                if wp.size != n or dti.size != n:
                    raise ValueError("Score arrays must have equal length")
                if site_ids is None:
                    site_ids = list(range(n))
                else:
                    site_ids = list(site_ids)

                method_scores = {"weighted_sum": ws, "weighted_product": wp, "distance_to_ideal": dti}
                method_ranks = {k: _to_ranks(v) for k, v in method_scores.items()}
                names = list(method_scores.keys())
                pairwise = {}
                corrs = []
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        c = _spearman(method_ranks[names[i]], method_ranks[names[j]])
                        pairwise[f"{names[i]}__vs__{names[j]}"] = c
                        corrs.append(c)

                flagged = []
                for idx in range(n):
                    r_site = {name: method_ranks[name][idx] for name in names}
                    spread = max(r_site.values()) - min(r_site.values())
                    if spread >= rank_diff_threshold:
                        flagged.append({"site_id": site_ids[idx], "ranks": r_site, "rank_spread": spread})
                flagged.sort(key=lambda x: x["rank_spread"], reverse=True)

                return {
                    "ranks": {k: list(v) for k, v in method_ranks.items()},
                    "pairwise_spearman": pairwise,
                    "overall_agreement": float(np.mean(corrs)) if corrs else 1.0,
                    "flagged_sites": flagged,
                }


# =============================================================================
# Request Schemas & Helper Utilities
# =============================================================================

class WeightPayload(BaseModel):
    """Payload schema allowing flexible input of weight vectors."""
    weights: Optional[Union[Dict[str, float], List[float]]] = Field(
        default=None,
        description="Dictionary mapping criterion name to float weight, or a list of float weights."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "weights": {
                    "foot_traffic": 0.35,
                    "demographics": 0.25,
                    "accessibility": 0.20,
                    "rental_cost": 0.20
                }
            }
        }
    }


def _clean_json_record(obj: Any) -> Any:
    """Ensure all numpy/pandas types and NaNs are converted to native Python types/None."""
    if isinstance(obj, dict):
        return {k: _clean_json_record(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_record(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


async def resolve_weight_vector(
    request: Request,
    payload: Optional[WeightPayload] = None,
    weights_query: Optional[str] = None,
) -> Union[Dict[str, float], List[float]]:
    """
    Extract the weight vector consistently from:
    1. Pydantic request body `payload.weights`
    2. Raw request body (JSON dict/list directly e.g. `{"c1": 0.5}`)
    3. `weights` query parameter (JSON string or comma-separated pairs)
    4. General query parameters matching criterion names
    5. Application state: `request.app.state.current_weights` or `.weights`
    """
    # 1. Pydantic Payload
    if payload is not None and payload.weights is not None:
        return payload.weights

    # 2. Query Parameter 'weights'
    if weights_query:
        weights_str = weights_query.strip()
        if weights_str.startswith("{") and weights_str.endswith("}"):
            try:
                parsed = json.loads(weights_str)
                if isinstance(parsed, dict):
                    return {str(k): float(v) for k, v in parsed.items()}
            except (json.JSONDecodeError, ValueError):
                pass
        elif weights_str.startswith("[") and weights_str.endswith("]"):
            try:
                parsed = json.loads(weights_str)
                if isinstance(parsed, list):
                    return [float(v) for v in parsed]
            except (json.JSONDecodeError, ValueError):
                pass
        elif "," in weights_str:
            parts = [p.strip() for p in weights_str.split(",") if p.strip()]
            if all(":" in p or "=" in p for p in parts):
                res = {}
                for p in parts:
                    delim = ":" if ":" in p else "="
                    k, v = p.split(delim, 1)
                    res[k.strip()] = float(v.strip())
                return res
            else:
                try:
                    return [float(p) for p in parts]
                except ValueError:
                    pass

    # 3. Raw Body Inspection (for direct GET/POST bodies without wrapping 'weights')
    try:
        raw_body = await request.body()
        if raw_body:
            parsed_json = json.loads(raw_body.decode("utf-8"))
            if isinstance(parsed_json, dict):
                if "weights" in parsed_json and parsed_json["weights"] is not None:
                    return parsed_json["weights"]
                # Direct dictionary of {criterion: weight}
                try:
                    return {str(k): float(v) for k, v in parsed_json.items() if not str(k).startswith("_")}
                except (ValueError, TypeError):
                    pass
            elif isinstance(parsed_json, list):
                return [float(v) for v in parsed_json]
    except Exception:
        pass

    # 4. Check arbitrary query params (e.g. ?c1=0.4&c2=0.6)
    reserved_params = {
        "weights", "n_samples", "top_n", "method", "coarse_steps",
        "uncertainty_threshold", "rank_diff_threshold", "percentile",
        "derive_from_perturbations"
    }
    extracted_params = {}
    for key, val in request.query_params.items():
        if key not in reserved_params:
            try:
                extracted_params[key] = float(val)
            except ValueError:
                pass
    if extracted_params:
        return extracted_params

    # 5. App State Fallback
    for attr in ("current_weights", "weights", "default_weights"):
        if hasattr(request.app.state, attr):
            state_weights = getattr(request.app.state, attr)
            if state_weights is not None:
                return state_weights

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            "Weight vector is required. Provide it via request body (e.g. {'weights': {...}}), "
            "the 'weights' query parameter (e.g. ?weights={'c1':0.5}), or set "
            "request.app.state.current_weights."
        ),
    )


def get_normalized_matrix(request: Request) -> Any:
    """
    Retrieve the normalized decision matrix from FastAPI application state or provider.
    """
    for attr in ("normalized_matrix", "matrix", "precomputed_matrix", "matrix_artifact"):
        if hasattr(request.app.state, attr):
            val = getattr(request.app.state, attr)
            if val is not None:
                return val

    # Attempt to load from matrix module if available
    try:
        from scoring.matrix import get_normalized_matrix as _load_matrix
        return _load_matrix()
    except (ImportError, AttributeError):
        pass

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Normalized decision matrix is not initialized in app.state. Please precompute or initialize the matrix."
    )


# =============================================================================
# Router Definition
# =============================================================================

router = APIRouter(tags=["Sensitivity Analysis"])


# -----------------------------------------------------------------------------
# 1. GET /sensitivity/thresholds
# -----------------------------------------------------------------------------
@router.get(
    "/sensitivity/thresholds",
    summary="Critical Threshold Report",
    response_description="File 38 critical threshold report for each criterion.",
)
@router.get("/thresholds", include_in_schema=False)
async def get_critical_thresholds(
    request: Request,
    payload: Optional[WeightPayload] = Body(None),
    weights: Optional[str] = Query(None, description="Current weight vector as JSON string or comma-separated pairs."),
    method: str = Query("weighted_sum", description="Scoring aggregation method ('weighted_sum', 'weighted_product', 'distance_to_ideal')."),
    coarse_steps: int = Query(40, ge=2, description="Coarse grid points per direction before bisection."),
    matrix: Any = Depends(get_normalized_matrix),
):
    """
    Returns File 38's critical threshold report for the current weights.

    Finds, for each criterion, how far its weight can move (holding other relative
    proportions fixed) before the top-ranked site changes.
    """
    current_weights = await resolve_weight_vector(request, payload=payload, weights_query=weights)

    try:
        report_df = compute_all_thresholds(
            normalized_matrix=matrix,
            current_weights=current_weights,
            method=method,
            coarse_steps=coarse_steps,
        )
        report_clean = report_df.replace({np.nan: None}).to_dict(orient="records")
        return _clean_json_record(report_clean)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Threshold calculation failed: {e}")


# -----------------------------------------------------------------------------
# 2. GET /sensitivity/stability
# -----------------------------------------------------------------------------
@router.get(
    "/sensitivity/stability",
    summary="Rank Stability Report",
    response_description="File 39 rank stability percentages under random weight perturbations.",
)
@router.get("/stability", include_in_schema=False)
async def get_rank_stability(
    request: Request,
    payload: Optional[WeightPayload] = Body(None),
    weights: Optional[str] = Query(None, description="Current weight vector as JSON string or comma-separated pairs."),
    n_samples: int = Query(200, ge=0, description="Number of perturbed weight vectors to evaluate."),
    top_n: Optional[int] = Query(None, gt=0, description="Optional top-N subset stability criterion. If None, checks exact baseline rank."),
    matrix: Any = Depends(get_normalized_matrix),
):
    """
    Returns File 39's rank stability report.

    Evaluates how often each site retains its baseline rank (or stays in top-N)
    under uniform random perturbations of the current criterion weights.
    """
    current_weights = await resolve_weight_vector(request, payload=payload, weights_query=weights)

    try:
        report_df = rank_stability_report(
            normalized_matrix=matrix,
            current_weights=current_weights,
            n_samples=n_samples,
            top_n=top_n,
        )
        report_clean = report_df.replace({np.nan: None}).to_dict(orient="records")
        return _clean_json_record(report_clean)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Stability calculation failed: {e}")


# -----------------------------------------------------------------------------
# 3. GET /sensitivity/near-ties
# -----------------------------------------------------------------------------
@router.get(
    "/sensitivity/near-ties",
    summary="Near-Tie Groups",
    response_description="File 40 ordered clusters of effectively tied sites.",
)
@router.get("/near-ties", include_in_schema=False)
async def get_near_ties(
    request: Request,
    payload: Optional[WeightPayload] = Body(None),
    weights: Optional[str] = Query(None, description="Current weight vector as JSON string or comma-separated pairs."),
    uncertainty_threshold: Optional[float] = Query(None, ge=0.0, description="Max score difference to treat as tied. If omitted, uses 0.5% of score spread."),
    method: str = Query("weighted_sum", description="Scoring aggregation method."),
    derive_from_perturbations: bool = Query(False, description="Whether to derive uncertainty threshold from File 39 score std across perturbations."),
    percentile: float = Query(50.0, ge=0.0, le=100.0, description="Percentile of score std if derive_from_perturbations is True."),
    matrix: Any = Depends(get_normalized_matrix),
):
    """
    Returns File 40's near-tie groups.

    Groups adjacent ranked sites whose score difference is smaller than an
    uncertainty threshold, returning them as rank clusters for UI display.
    """
    current_weights = await resolve_weight_vector(request, payload=payload, weights_query=weights)

    try:
        # Score sites with the current weights
        scores = apply_weights(matrix, current_weights, method=method)
        ranked_scores = rank_sites(scores)

        # Derive uncertainty threshold from perturbations if requested
        if uncertainty_threshold is None and derive_from_perturbations:
            w = list(current_weights.values()) if isinstance(current_weights, Mapping) else current_weights
            samples = sample_weight_perturbations(w, n_samples=100)
            sampled_scores = [apply_weights(matrix, sw, method=method).to_numpy() for sw in samples]
            score_std = np.std(sampled_scores, axis=0)
            uncertainty_threshold = derive_uncertainty_threshold(score_std, percentile=percentile)

        clusters = identify_near_ties(ranked_scores, uncertainty_threshold=uncertainty_threshold)
        return _clean_json_record(clusters)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Near-tie clustering failed: {e}")


# -----------------------------------------------------------------------------
# 4. GET /sensitivity/cross-method
# -----------------------------------------------------------------------------
@router.get(
    "/sensitivity/cross-method",
    summary="Cross-Method Agreement Summary",
    response_description="File 30 cross-method rank comparisons, pairwise Spearman correlations, and flagged sensitive sites.",
)
@router.get("/cross-method", include_in_schema=False)
async def get_cross_method_agreement(
    request: Request,
    payload: Optional[WeightPayload] = Body(None),
    weights: Optional[str] = Query(None, description="Current weight vector as JSON string or comma-separated pairs."),
    rank_diff_threshold: int = Query(3, ge=0, description="Minimum rank spread across methods to flag a site as ranking-sensitive."),
    matrix: Any = Depends(get_normalized_matrix),
):
    """
    Returns File 30's cross-method agreement summary.

    Compares rankings across:
    - Weighted Sum (File 27)
    - Weighted Product (File 28)
    - Distance to Ideal / TOPSIS (File 29)
    Quantifies overall agreement via Spearman correlation and flags unstable sites.
    """
    current_weights = await resolve_weight_vector(request, payload=payload, weights_query=weights)

    try:
        ws_scores = apply_weights(matrix, current_weights, method="weighted_sum")
        wp_scores = apply_weights(matrix, current_weights, method="weighted_product")
        dti_scores = apply_weights(matrix, current_weights, method="distance_to_ideal")

        def _extract_values(s):
            if isinstance(s, (pd.Series, pd.DataFrame)):
                return s.to_numpy().reshape(-1)
            return np.asarray(s, dtype=float).reshape(-1)

        _, site_ids, _ = _unwrap_precomputed_matrix(matrix)

        agreement = compare_rankings(
            weighted_sum_result=_extract_values(ws_scores),
            weighted_product_result=_extract_values(wp_scores),
            distance_to_ideal_result=_extract_values(dti_scores),
            site_ids=site_ids,
            rank_diff_threshold=rank_diff_threshold,
        )
        return _clean_json_record(agreement)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Cross-method agreement failed: {e}")


__all__ = ["router"]