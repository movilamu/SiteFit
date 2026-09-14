"""
/scoring/constraints.py

Implements hard-constraint pre-scoring filtering for candidate site evaluation.

Why Pre-Scoring Filtering vs. Scored Criterion:
----------------------------------------------
Hard constraints represent non-compensatory business requirements and absolute deal-breakers
(e.g., minimum physical floor area, maximum budget cap, permitted tenure types, or vehicle access).
In multi-criteria compensatory scoring models (such as Weighted Sum or AHP synthesis), high scores
in one criterion (e.g., exceptional demographics or pedestrian footfall) can mathematically compensate
for low scores in another.

However, a site lacking critical physical, legal, or financial viability cannot be rescued by high
demographic scores because it simply cannot operate. Filtering invalid candidates before scoring:
1. Prevents non-viable sites from appearing in decision shortlists through trade-off compensation.
2. Avoids skewing min-max normalization ranges for legitimate candidates in down-stream scoring.
"""

from typing import Any, Callable, Dict, List, Tuple
import pandas as pd


def apply_hard_constraints(
    sites_df: pd.DataFrame,
    constraints_dict: Dict[str, Any],
    id_column: str = "site_id",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Filters out candidate sites failing hard constraints BEFORE scoring begins.

    Parameters
    ----------
    sites_df : pd.DataFrame
        DataFrame containing candidate site attributes.
    constraints_dict : Dict[str, Any]
        Dictionary of hard constraints to apply. Supported parameters:
        - `min_unit_size` (float | int): Minimum acceptable floor area/unit size.
        - `max_rent` (float | int): Maximum acceptable rent cap.
        - `required_tenure` (list[str] | set[str] | str): Allowed tenure types.
        - `min_<column_name>` / `max_<column_name>`: Extensible min/max bounds on any numeric column.
        - Custom callable function: Extensible key mapping to a predicate function `fn(row) -> bool`.
    id_column : str, default "site_id"
        Name of the unique site identifier column for exclusion tracing.
        If not present in `sites_df`, falls back to index values.

    Returns
    -------
    Tuple[pd.DataFrame, List[Dict[str, Any]]]
        - `filtered_df`: DataFrame containing only sites passing all hard constraints.
        - `excluded_sites`: List of dicts documenting excluded site IDs and specific failure reasons.
    """
    if sites_df.empty:
        return sites_df.copy(), []

    has_id_col = id_column in sites_df.columns
    excluded_sites: List[Dict[str, Any]] = []
    excluded_indices: set = set()

    # Pre-determine column aliases for standard constraints
    unit_size_col = next(
        (c for c in ["unit_size", "unit_size_sqft", "floor_area"] if c in sites_df.columns),
        None,
    )
    rent_col = next(
        (c for c in ["rent", "rent_per_sqft", "monthly_rent"] if c in sites_df.columns),
        None,
    )
    tenure_col = next(
        (c for c in ["tenure_type", "tenure", "lease_type"] if c in sites_df.columns),
        None,
    )

    for idx, row in sites_df.iterrows():
        site_identifier = row[id_column] if has_id_col else idx
        reasons: List[str] = []

        # 1. Minimum Unit Size
        if "min_unit_size" in constraints_dict and constraints_dict["min_unit_size"] is not None:
            min_size = constraints_dict["min_unit_size"]
            if unit_size_col is None:
                reasons.append("Missing required unit size data column.")
            else:
                val = row[unit_size_col]
                if pd.isna(val) or val < min_size:
                    reasons.append(
                        f"Unit size ({val}) is below minimum requirement of {min_size}."
                    )

        # 2. Maximum Rent Cap
        if "max_rent" in constraints_dict and constraints_dict["max_rent"] is not None:
            max_rent = constraints_dict["max_rent"]
            if rent_col is None:
                reasons.append("Missing required rent data column.")
            else:
                val = row[rent_col]
                if pd.isna(val) or val > max_rent:
                    reasons.append(
                        f"Rent ({val}) exceeds maximum cap of {max_rent}."
                    )

        # 3. Required Tenure Type
        if "required_tenure" in constraints_dict and constraints_dict["required_tenure"]:
            raw_tenures = constraints_dict["required_tenure"]
            if isinstance(raw_tenures, (list, tuple, set)):
                allowed_tenures = {str(t).lower().strip() for t in raw_tenures}
            else:
                allowed_tenures = {str(raw_tenures).lower().strip()}

            if tenure_col is None:
                reasons.append("Missing required tenure type data column.")
            else:
                raw_val = row[tenure_col]
                val = str(raw_val).lower().strip() if pd.notna(raw_val) else None
                if val is None or val not in allowed_tenures:
                    reasons.append(
                        f"Tenure type '{raw_val}' is not in allowed list: {list(raw_tenures)}."
                    )

        # 4. Extensible Custom & Dynamic Constraints
        for key, rule in constraints_dict.items():
            if key in {"min_unit_size", "max_rent", "required_tenure"}:
                continue

            # Case A: Dynamic min/max bounds on custom columns (e.g. min_frontage, max_fit_out_cost)
            if key.startswith("min_") and key[4:] in sites_df.columns:
                col_name = key[4:]
                val = row[col_name]
                if pd.isna(val) or val < rule:
                    reasons.append(
                        f"Column '{col_name}' value ({val}) is below minimum limit of {rule}."
                    )
            elif key.startswith("max_") and key[4:] in sites_df.columns:
                col_name = key[4:]
                val = row[col_name]
                if pd.isna(val) or val > rule:
                    reasons.append(
                        f"Column '{col_name}' value ({val}) exceeds maximum limit of {rule}."
                    )

            # Case B: Custom predicate function fn(row: pd.Series) -> bool
            elif callable(rule):
                try:
                    passed = rule(row)
                    if not passed:
                        reasons.append(f"Failed custom constraint predicate '{key}'.")
                except Exception as err:
                    reasons.append(f"Error executing constraint predicate '{key}': {err}")

        if reasons:
            excluded_indices.add(idx)
            excluded_sites.append(
                {
                    "site_identifier": site_identifier,
                    "reasons": reasons,
                }
            )

    filtered_df = sites_df.drop(index=list(excluded_indices)).copy()
    return filtered_df, excluded_sites