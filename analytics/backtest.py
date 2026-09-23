"""
analytics/backtest.py

Primary empirical validation check for the site-selection scoring platform.

Per the original spec, this is the platform's ground-truth sanity check:
the client's own store estate is the one set of "candidate sites" for
which we already know the real-world outcome (current_trading_performance).
`backtest_against_estate` scores those stores through the exact same
pipeline used for genuine candidate-site evaluation --
  File 26 hard constraints -> File 24 normalization -> File 31
  (`build_normalized_matrix`) -> File 25 (`compute_global_weights`) ->
  File 32 (`apply_weights`)
-- and correlates the resulting model score against actual trading
performance.

This is a validation check, not a feature. A weak or non-significant
correlation means the criteria set, normalization, or weighting scheme
does not track real trading outcomes, and that is a modeling problem
that must be investigated and resolved BEFORE the platform's scores are
allowed to inform any real capital-allocation decision. Do not treat a
weak correlation result as acceptable noise to route around.

IMPORTANT -- inputs this module deliberately does NOT fabricate
--------------------------------------------------------------
`own_estate_locations` (File 15) only carries store_name / geometry /
current_trading_performance -- it does not carry the raw scorable
criterion columns (footfall, demographics, drive-time reach, etc.) that
File 31's `build_normalized_matrix` needs. Wherever those criteria are
sourced for genuine candidate sites, the own-estate stores must be run
through that same feature pipeline (e.g. joined by location against
whatever table/service supplies criteria for candidate sites) before
they reach this module. That join is out of scope here because none of
Files 15/25/31/32 define it -- `backtest_against_estate` takes the
already-assembled feature frame (`sites_df`) as input rather than
guessing at a join that could silently score stores on the wrong data.

Likewise, File 25 has no notion of a stored "default" weight set --
`compute_global_weights` only turns an explicit `level_weights` mapping
into leaf weights. The platform's actual default/baseline sliders live
wherever the UI defines them (not in Files 15/25/31/32), so
`criteria_hierarchy_config` and `level_weights` are required parameters
here, not something this module invents on the caller's behalf.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for backtest_against_estate (pip install scipy "
        "--break-system-packages)."
    ) from exc

try:
    from supabase import create_client, Client
except ImportError:  # pragma: no cover
    create_client = None
    Client = None

try:
    from scoring.score import apply_weights
except ImportError:  # running as a loose script alongside File 32
    try:
        from score import apply_weights
    except ImportError:
        from File_32 import apply_weights

try:
    # File 31: builds the normalized, constraint-filtered decision matrix
    # apply_weights expects. Backtesting goes through the real offline
    # precompute step candidate sites go through, not a shortcut.
    from scoring.precompute_matrix import build_normalized_matrix, NormalizedMatrixArtifact
except ImportError:
    try:
        from precompute_matrix import build_normalized_matrix, NormalizedMatrixArtifact
    except ImportError:
        from File_31 import build_normalized_matrix, NormalizedMatrixArtifact

try:
    # File 25: rolls explicit level weights up into global leaf weights.
    from scoring.hierarchical_rollup import compute_global_weights
except ImportError:
    try:
        from hierarchical_rollup import compute_global_weights
    except ImportError:
        from File_25 import compute_global_weights


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backtest")

DEFAULT_METHOD = "weighted_sum"
DEFAULT_PERFORMANCE_COLUMN = "current_trading_performance"
DEFAULT_ID_COLUMN = "site_id"


@dataclass
class BacktestResult:
    """Result of backtest_against_estate.

    Attributes
    ----------
    correlation : float
        Spearman's rank correlation coefficient between model score and
        actual current_trading_performance. NaN if fewer than 2 stores
        had usable data.
    p_value : Optional[float]
        Two-sided p-value for the null hypothesis of no monotonic
        association. None if not computable (e.g. < 2 data points).
    n : int
        Number of own-estate stores included in the correlation (i.e.
        passed File 26's hard constraints AND had a valid model score
        AND a non-null actual performance figure).
    scatter_data : pandas.DataFrame
        Columns: site_id, predicted_score, actual_performance. Ready to
        hand straight to a scatter-plot call.
    method : str
        Aggregation method used to compute predicted_score.
    excluded_by_constraints : pandas.DataFrame
        Own-estate stores that File 26's hard constraints filtered out
        before scoring, with their actual performance attached where
        available. A real, currently-trading store landing here is
        itself a validation signal -- it means the hard-constraint
        rules would reject a location that demonstrably works, which is
        as serious a modeling problem as a weak correlation and should
        be investigated, not waved through.
    warnings : List[str]
        Non-fatal notes, e.g. stores dropped for unparseable geometry
        or missing from the scored matrix.
    """

    correlation: float
    p_value: Optional[float]
    n: int
    scatter_data: pd.DataFrame
    method: str
    excluded_by_constraints: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: List[str] = field(default_factory=list)

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value is not None and self.p_value < alpha

    def summary(self) -> str:
        if np.isnan(self.correlation):
            return "Backtest inconclusive: fewer than 2 own-estate stores with usable data."
        sig = "significant" if self.is_significant() else "NOT significant"
        verdict = (
            "model score tracks real trading performance"
            if (self.correlation > 0 and self.is_significant())
            else "model score does NOT reliably track real trading performance "
                 "-- criteria/weighting review required before this platform "
                 "informs any real capital decision"
        )
        return (
            f"Spearman rho={self.correlation:.3f}, p={self.p_value:.4g} "
            f"(n={self.n}, {sig} at alpha=0.05) -> {verdict}"
        )


def _get_supabase_client() -> "Client":
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise EnvironmentError(
            "Missing SUPABASE_URL and/or SUPABASE_KEY environment variables. "
            "Both must be set before running this script."
        )
    if create_client is None:
        raise ImportError(
            "The 'supabase' package is not installed. "
            "Install it with: pip install supabase --break-system-packages"
        )
    return create_client(url, key)


def load_own_estate_performance(
    client: Optional["Client"] = None,
    *,
    performance_column: str = DEFAULT_PERFORMANCE_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
) -> pd.DataFrame:
    """Convenience loader: own_estate_locations id/name/lat/lon/performance.

    This does NOT return anything scoreable on its own -- it has no
    criterion columns. It exists so a caller can join it (by id or by
    location) against whichever table/service supplies the raw scorable
    criteria for candidate sites, to build the `sites_df` that
    `backtest_against_estate` actually needs. Rows with a null
    performance figure or unparseable geometry are dropped.
    """
    supabase = client or _get_supabase_client()
    response = (
        supabase.table("own_estate_locations")
        .select(f"id, name, geometry, {performance_column}")
        .not_.is_(performance_column, "null")
        .execute()
    )
    rows = response.data or []
    if not rows:
        logger.warning("No own_estate_locations rows with non-null %s found.", performance_column)
        return pd.DataFrame(columns=[id_column, "name", "latitude", "longitude", performance_column])

    records = []
    for row in rows:
        lon, lat = _parse_point_geometry(row.get("geometry"))
        if lon is None or lat is None:
            logger.warning(
                "Skipping own_estate_locations id=%s ('%s'): unparseable geometry.",
                row.get("id"), row.get("name"),
            )
            continue
        records.append(
            {
                id_column: row.get("id"),
                "name": row.get("name"),
                "latitude": lat,
                "longitude": lon,
                performance_column: float(row[performance_column]),
            }
        )
    return pd.DataFrame.from_records(records)


def _parse_point_geometry(geometry: Any):
    """Best-effort extraction of (lon, lat) from a PostGIS geometry value.

    Supabase/PostgREST typically returns geometry as GeoJSON (when the
    column is exposed that way) or as an EWKT/EWKB string. Handles the
    common cases; returns (None, None) if the shape is unrecognized so
    the caller can skip the row rather than fail the whole backtest.
    """
    if geometry is None:
        return None, None

    if isinstance(geometry, dict):
        coords = geometry.get("coordinates")
        if coords and len(coords) >= 2:
            return float(coords[0]), float(coords[1])
        return None, None

    if isinstance(geometry, str):
        text = geometry.strip()
        if text.upper().startswith("SRID=") and "POINT(" in text.upper():
            try:
                inner = text[text.index("(") + 1 : text.index(")")]
                lon_str, lat_str = inner.split()
                return float(lon_str), float(lat_str)
            except (ValueError, IndexError):
                return None, None
        # EWKB hex string: not decoded here to avoid a hard dependency on
        # shapely/geoalchemy2; if this path is hit in practice, decode
        # upstream (e.g. via a Postgres view that exposes GeoJSON instead).
        return None, None

    return None, None


def backtest_against_estate(
    sites_df: pd.DataFrame,
    criteria_hierarchy_config: Any,
    level_weights: Dict[str, Any],
    *,
    performance_column: str = DEFAULT_PERFORMANCE_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    constraints_dict: Optional[Dict[str, Any]] = None,
    method: str = DEFAULT_METHOD,
) -> BacktestResult:
    """Validate model scores against the client's own store estate.

    This is the platform's primary empirical validation check per the
    original spec: own-estate stores are the only "candidate sites" for
    which real trading outcomes are already known, so scoring them with
    the current baseline weight set and correlating against actual
    performance is the ground-truth test of whether the criteria/
    weighting scheme means anything. A weak or non-significant
    correlation indicates a criteria or weighting problem that must be
    resolved before this platform is allowed to inform a real capital
    decision -- it is not an acceptable result to shrug off or tune away
    without investigation. The same applies if genuinely trading stores
    get filtered out by hard constraints before scoring even happens
    (see `BacktestResult.excluded_by_constraints`).

    Parameters
    ----------
    sites_df : DataFrame
        One row per own-estate store, carrying the SAME raw criterion
        columns and constraint fields that candidate sites are scored
        from (File 31 requires this to build the normalized matrix),
        plus `performance_column` and `id_column`. This module does not
        assemble this frame itself -- see the module docstring for why.
    criteria_hierarchy_config : File 23 tree / dict tree
        The exact hierarchy candidate sites are scored against.
    level_weights : dict
        The baseline/default weight set to backtest, in the nested or
        flat form File 25's `compute_global_weights` expects. There is
        no implicit default -- pass whatever the platform currently
        treats as its baseline sliders.
    performance_column : str
        Column in `sites_df` holding actual trading performance.
    id_column : str
        Site identity column, forwarded to File 31 and used to align
        scores back to `sites_df`.
    constraints_dict : optional
        Hard constraints forwarded to File 31 / File 26. If omitted,
        constraints bundled on `criteria_hierarchy_config` are used (or
        none, if there are none bundled either).
    method : str
        Aggregation method forwarded to File 32 `apply_weights`
        ('weighted_sum', 'weighted_product', or 'distance_to_ideal').

    Returns
    -------
    BacktestResult
        correlation (Spearman's rho), p_value, n, scatter_data,
        method, excluded_by_constraints, warnings.

    Raises
    ------
    ValueError
        If `sites_df` lacks `performance_column` or `id_column`, or if
        `level_weights` don't sum to 1.0 at every level (File 25 raises
        this from `compute_global_weights`).
    """
    if id_column not in sites_df.columns:
        raise ValueError(f"sites_df is missing required id_column '{id_column}'.")
    if performance_column not in sites_df.columns:
        raise ValueError(f"sites_df is missing required performance_column '{performance_column}'.")

    warns: List[str] = []

    # Keep performance aside -- it must never be treated as a scorable
    # criterion or it would leak the label into the normalization step.
    performance_by_id = (
        sites_df.set_index(id_column)[performance_column].astype(float)
    )
    feature_df = sites_df.drop(columns=[performance_column])

    artifact: NormalizedMatrixArtifact = build_normalized_matrix(
        feature_df,
        criteria_hierarchy_config,
        constraints_dict=constraints_dict,
        id_column=id_column,
    )

    # Stores File 26 filtered out before scoring even started. A real,
    # currently-trading store showing up here is a validation signal in
    # its own right -- surface it rather than silently dropping it.
    excluded_records = []
    for excl in artifact.excluded_sites:
        excl_id = excl.get(id_column) if isinstance(excl, dict) else None
        actual_perf = performance_by_id.get(excl_id) if excl_id in performance_by_id.index else None
        record = dict(excl) if isinstance(excl, dict) else {"raw": excl}
        record["actual_performance"] = actual_perf
        excluded_records.append(record)
    excluded_df = pd.DataFrame.from_records(excluded_records)
    if len(excluded_df):
        trading_and_excluded = excluded_df["actual_performance"].notna().sum()
        if trading_and_excluded:
            msg = (
                f"{trading_and_excluded} currently-trading own-estate store(s) were "
                "excluded by hard constraints before scoring. That's a signal the "
                "constraint rules may be too strict, independent of the correlation "
                "result below -- see excluded_by_constraints."
            )
            logger.warning(msg)
            warns.append(msg)

    global_weights = compute_global_weights(criteria_hierarchy_config, level_weights)
    scores = apply_weights(artifact, global_weights, method=method)

    scores_df = scores.rename("predicted_score").to_frame()
    scores_df.index.name = id_column

    merged = scores_df.join(performance_by_id, how="inner")
    dropped = len(scores_df) - len(merged)
    if dropped > 0:
        msg = (
            f"{dropped} scored site(s) had no matching performance figure in "
            "sites_df and were excluded from the correlation."
        )
        logger.warning(msg)
        warns.append(msg)

    merged = merged.dropna(subset=["predicted_score", performance_column])
    n = len(merged)

    if n < 2:
        logger.warning("Fewer than 2 usable own-estate stores (n=%d); correlation undefined.", n)
        scatter = merged.reset_index().rename(columns={performance_column: "actual_performance"})
        return BacktestResult(
            correlation=float("nan"),
            p_value=None,
            n=n,
            scatter_data=scatter,
            method=method,
            excluded_by_constraints=excluded_df,
            warnings=warns + ["Fewer than 2 usable data points; correlation is undefined."],
        )

    rho, p_value = spearmanr(
        merged["predicted_score"].to_numpy(dtype=float),
        merged[performance_column].to_numpy(dtype=float),
    )

    scatter_data = merged.reset_index().rename(columns={performance_column: "actual_performance"})

    result = BacktestResult(
        correlation=float(rho),
        p_value=float(p_value),
        n=n,
        scatter_data=scatter_data,
        method=method,
        excluded_by_constraints=excluded_df,
        warnings=warns,
    )

    logger.info(result.summary())
    return result


def main() -> None:
    raise SystemExit(
        "backtest_against_estate() requires sites_df (own-estate stores joined "
        "to the same raw criteria candidate sites are scored from), "
        "criteria_hierarchy_config, and level_weights -- there is no default "
        "data source wired up here. Call it directly from a script or notebook "
        "that assembles those inputs, e.g.:\n\n"
        "    from analytics.backtest import backtest_against_estate, "
        "load_own_estate_performance\n"
        "    perf = load_own_estate_performance()  # id/name/lat/lon/performance only\n"
        "    # ... join `perf` against your criteria feature source to build sites_df ...\n"
        "    result = backtest_against_estate(sites_df, CRITERIA_TREE, level_weights)\n"
        "    print(result.summary())"
    )


if __name__ == "__main__":
    main()
