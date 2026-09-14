"""
analytics/backtest_report.py

Turns a `BacktestResult` from File 61 (`backtest_against_estate`) into a
readable Markdown report for File 63 (Metabase notes) and File 74 (the
stakeholder document).

Design note -- why this doesn't "just run" on its own
-------------------------------------------------------
File 61 deliberately does not assemble `sites_df` / `criteria_hierarchy_config`
/ `level_weights` itself (see its module docstring): the own-estate stores
need to be joined against whatever table/service supplies raw scorable
criteria for candidate sites, and that join isn't defined by Files 15/25/31/32.
This module doesn't invent that join either -- inventing it would silently
score stores on made-up or wrong data, which is exactly what File 61 is
designed to avoid.

So `main()` below has the same shape as File 61's own `main()`: it wires
`load_own_estate_performance()` up to `backtest_against_estate()`, but the
`build_sites_df(...)` step in between is a clearly-marked hook you fill in
with your actual criteria-source join, criteria hierarchy config, and
baseline `level_weights`. Once you supply those three things, everything
downstream (running the backtest, interpreting the correlation, writing the
report) is fully automatic.

Usage
-----
    from analytics.backtest_report import run_and_report

    path = run_and_report(
        sites_df=sites_df,                       # see build_sites_df() below
        criteria_hierarchy_config=CRITERIA_TREE,
        level_weights=BASELINE_WEIGHTS,
        output_path="analytics/output/backtest_report.md",
    )

Or from the command line, after filling in `build_sites_df()`:

    python -m analytics.backtest_report
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict, Optional

import pandas as pd

try:
    from analytics.backtest import (
        BacktestResult,
        backtest_against_estate,
        load_own_estate_performance,
    )
except ImportError:
    try:
        from backtest import (
            BacktestResult,
            backtest_against_estate,
            load_own_estate_performance,
        )
    except ImportError:
        from File_61 import (
            BacktestResult,
            backtest_against_estate,
            load_own_estate_performance,
        )


DEFAULT_OUTPUT_PATH = "analytics/output/backtest_report.md"

# How many of the largest predicted-vs-actual rank divergences to surface
# for manual investigation.
DEFAULT_TOP_DIVERGENT_N = 10


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def _interpret_correlation(rho: float) -> str:
    """Plain-language read of a Spearman rho magnitude.

    Thresholds follow the common Cohen-style convention used for
    correlation coefficients (rules of thumb, not statistical law):
      |rho| < 0.10        -> no meaningful relationship
      0.10 <= |rho| < 0.30 -> weak
      0.30 <= |rho| < 0.50 -> moderate
      0.50 <= |rho| < 0.70 -> strong
      |rho| >= 0.70        -> very strong
    """
    if pd.isna(rho):
        return "undefined (insufficient data)"

    direction = "positive" if rho > 0 else "negative" if rho < 0 else "no"
    magnitude = abs(rho)

    if direction == "no":
        return "no relationship"

    if magnitude < 0.10:
        strength = "no meaningful"
    elif magnitude < 0.30:
        strength = "weak"
    elif magnitude < 0.50:
        strength = "moderate"
    elif magnitude < 0.70:
        strength = "strong"
    else:
        strength = "very strong"

    if strength == "no meaningful":
        return "no meaningful relationship"
    return f"{strength} {direction}"


def _significance_text(result: BacktestResult, alpha: float = 0.05) -> str:
    if result.p_value is None:
        return "not computable (fewer than 2 usable data points)"
    sig = "statistically significant" if result.is_significant(alpha) else "not statistically significant"
    return f"{sig} (p={result.p_value:.4g}, alpha={alpha})"


def _rank_divergence(scatter_data: pd.DataFrame) -> pd.DataFrame:
    """Rank-based divergence between predicted_score and actual_performance.

    Both columns are converted to percentile ranks (0-1, higher = better)
    so the divergence is comparable regardless of each column's raw scale.
    `rank_gap` = predicted_rank - actual_rank:
      positive -> model ranked the store higher than it actually performed
                  (model over-optimistic about this site)
      negative -> model ranked the store lower than it actually performed
                  (model missed a genuinely strong site)
    """
    if scatter_data.empty:
        return scatter_data.assign(
            predicted_rank_pct=[], actual_rank_pct=[], rank_gap=[]
        )

    df = scatter_data.copy()
    df["predicted_rank_pct"] = df["predicted_score"].rank(pct=True)
    df["actual_rank_pct"] = df["actual_performance"].rank(pct=True)
    df["rank_gap"] = df["predicted_rank_pct"] - df["actual_rank_pct"]
    df["abs_rank_gap"] = df["rank_gap"].abs()
    return df.sort_values("abs_rank_gap", ascending=False)


def _divergence_direction_note(row: pd.Series) -> str:
    if row["rank_gap"] > 0:
        return "model ranked this site higher than its actual performance justifies"
    if row["rank_gap"] < 0:
        return "model ranked this site lower than its actual performance -- a genuinely strong site the model missed"
    return "no divergence"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    result: BacktestResult,
    *,
    id_column: str = "site_id",
    top_n_divergent: int = DEFAULT_TOP_DIVERGENT_N,
    generated_at: Optional[_dt.datetime] = None,
) -> str:
    """Build a Markdown report from a `BacktestResult`.

    Includes:
      - headline correlation coefficient + plain-language interpretation
      - significance
      - the scatter-plot data (predicted score vs actual performance)
      - stores where model ranking and actual performance diverged most
        sharply, for manual investigation
      - any warnings the backtest raised (e.g. trading stores excluded by
        hard constraints), since those are validation signals in their
        own right
    """
    generated_at = generated_at or _dt.datetime.now(_dt.timezone.utc)
    lines: list[str] = []

    lines.append("# Site-Selection Model Backtest Report")
    lines.append("")
    lines.append(f"_Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")
    lines.append(
        "This report validates the site-selection model's scores against "
        "the client's own store estate -- the only \"candidate sites\" for "
        "which real trading outcomes are already known. See `analytics/backtest.py` "
        "for the full methodology."
    )
    lines.append("")

    # --- Headline result -----------------------------------------------
    lines.append("## Headline Result")
    lines.append("")
    if pd.isna(result.correlation):
        lines.append(
            "**Inconclusive.** Fewer than 2 own-estate stores had usable "
            "data (a valid model score and a non-null actual performance "
            "figure), so no correlation could be computed."
        )
    else:
        interpretation = _interpret_correlation(result.correlation)
        sig_text = _significance_text(result)
        lines.append(f"- **Spearman correlation (rho):** {result.correlation:.3f}")
        lines.append(f"- **Plain-language interpretation:** {interpretation} relationship")
        lines.append(f"- **Significance:** {sig_text}")
        lines.append(f"- **n (stores included):** {result.n}")
        lines.append(f"- **Aggregation method:** {result.method}")
        lines.append("")
        if result.correlation > 0 and result.is_significant():
            lines.append(
                "**Verdict:** the model's scores track real trading performance."
            )
        else:
            lines.append(
                "**Verdict:** the model's scores do NOT reliably track real "
                "trading performance. Per the original spec, this is a "
                "modeling problem (criteria set, normalization, and/or "
                "weighting scheme) that must be investigated and resolved "
                "before this platform's scores are used to inform any real "
                "capital-allocation decision. This is not noise to route "
                "around."
            )
    lines.append("")

    # --- Warnings --------------------------------------------------------
    if result.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    # --- Excluded-by-constraints -----------------------------------------
    if result.excluded_by_constraints is not None and len(result.excluded_by_constraints):
        trading_excluded = result.excluded_by_constraints[
            result.excluded_by_constraints.get("actual_performance").notna()
        ] if "actual_performance" in result.excluded_by_constraints.columns else pd.DataFrame()
        lines.append("## Stores Excluded by Hard Constraints")
        lines.append("")
        lines.append(
            f"{len(result.excluded_by_constraints)} own-estate store(s) were filtered "
            "out before scoring. A currently-trading store landing here is itself "
            "a validation signal -- it means the hard-constraint rules would reject "
            "a location that demonstrably works."
        )
        if len(trading_excluded):
            lines.append("")
            lines.append(f"**{len(trading_excluded)} of these are currently trading** and warrant review:")
            lines.append("")
            lines.append(trading_excluded.to_markdown(index=False))
        lines.append("")

    # --- Scatter plot data -------------------------------------------------
    lines.append("## Scatter Plot Data")
    lines.append("")
    lines.append(
        "Predicted model score vs. actual trading performance, one row per "
        "own-estate store included in the correlation. Feed directly into a "
        "scatter-plot call (e.g. Metabase, matplotlib, plotly)."
    )
    lines.append("")
    if result.scatter_data is not None and len(result.scatter_data):
        lines.append(result.scatter_data.to_markdown(index=False))
    else:
        lines.append("_No scatter data available (0 usable stores)._")
    lines.append("")

    # --- Divergent stores ----------------------------------------------
    lines.append("## Stores With Largest Model/Actual Divergence")
    lines.append("")
    lines.append(
        "Stores where the model's rank (by predicted score) diverges most "
        "sharply from their actual-performance rank, expressed as a "
        "percentile-rank gap. Flagged here for manual investigation, not "
        "automatically explained."
    )
    lines.append("")
    if result.scatter_data is not None and len(result.scatter_data) >= 2:
        ranked = _rank_divergence(result.scatter_data)
        top_divergent = ranked.head(top_n_divergent).copy()
        top_divergent["note"] = top_divergent.apply(_divergence_direction_note, axis=1)
        display_cols = [
            c for c in [id_column, "predicted_score", "actual_performance",
                        "predicted_rank_pct", "actual_rank_pct", "rank_gap", "note"]
            if c in top_divergent.columns
        ]
        lines.append(top_divergent[display_cols].to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("_Not enough data points to compute rank divergence._")
    lines.append("")

    return "\n".join(lines)


def write_report(report_markdown: str, output_path: str = DEFAULT_OUTPUT_PATH) -> str:
    """Save the report to disk, creating parent directories as needed."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)
    return output_path


def run_and_report(
    sites_df: pd.DataFrame,
    criteria_hierarchy_config: Any,
    level_weights: Dict[str, Any],
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    id_column: str = "site_id",
    top_n_divergent: int = DEFAULT_TOP_DIVERGENT_N,
    **backtest_kwargs: Any,
) -> str:
    """Run `backtest_against_estate` and write the resulting report to disk.

    Returns the path the report was written to.
    """
    result = backtest_against_estate(
        sites_df,
        criteria_hierarchy_config,
        level_weights,
        id_column=id_column,
        **backtest_kwargs,
    )
    report_md = generate_report(result, id_column=id_column, top_n_divergent=top_n_divergent)
    return write_report(report_md, output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_sites_df(perf_df: pd.DataFrame) -> pd.DataFrame:
    """*** FILL THIS IN FOR YOUR ENVIRONMENT ***

    File 61 deliberately does not define how own-estate stores get joined
    against the raw scorable criteria (footfall, demographics, drive-time
    reach, etc.) that candidate sites are scored from -- that join lives
    wherever your platform actually sources those criteria (a table, a
    service call, a precomputed feature store, ...).

    `perf_df` is the output of `load_own_estate_performance()`: id / name /
    latitude / longitude / current_trading_performance only. Join it here
    against your real criteria source (e.g. by id, or by lat/lon proximity)
    and return a single DataFrame with:
      - the same raw criterion columns candidate sites are scored from
      - the same hard-constraint fields File 26 checks
      - `current_trading_performance` (carried through from `perf_df`)
      - `site_id` (or whatever `id_column` you pass to `backtest_against_estate`)

    This function intentionally raises until you wire that join up -- a
    fabricated join would silently score stores on the wrong data, which
    is exactly the failure mode File 61 was written to avoid.
    """
    raise NotImplementedError(
        "build_sites_df() is a hook, not an implementation. Join `perf_df` "
        "against your actual criteria-source table/service to build the "
        "sites_df backtest_against_estate() needs, per the module docstring "
        "of analytics/backtest.py."
    )


def main() -> None:
    """CLI entry point.

    Mirrors File 61's own `main()`: there is no default criteria hierarchy
    or baseline weight set wired up here (those live in the platform's UI
    config, not in Files 15/25/31/32), so you must fill in
    `CRITERIA_HIERARCHY_CONFIG`, `LEVEL_WEIGHTS`, and `build_sites_df()`
    below before running this as a script.
    """
    # *** FILL THESE IN FOR YOUR ENVIRONMENT ***
    CRITERIA_HIERARCHY_CONFIG: Any = None
    LEVEL_WEIGHTS: Optional[Dict[str, Any]] = None

    if CRITERIA_HIERARCHY_CONFIG is None or LEVEL_WEIGHTS is None:
        raise SystemExit(
            "backtest_report.main() needs CRITERIA_HIERARCHY_CONFIG and "
            "LEVEL_WEIGHTS filled in (the platform's actual criteria tree "
            "and baseline slider weights), plus a working build_sites_df() "
            "that joins own-estate stores against your real criteria "
            "source. There is no default wired up here -- see the "
            "docstrings in this file and in analytics/backtest.py.\n\n"
            "Once filled in:\n"
            "    perf = load_own_estate_performance()\n"
            "    sites_df = build_sites_df(perf)\n"
            "    path = run_and_report(sites_df, CRITERIA_HIERARCHY_CONFIG, LEVEL_WEIGHTS)\n"
            "    print(f'Report written to {path}')"
        )

    perf = load_own_estate_performance()
    sites_df = build_sites_df(perf)
    path = run_and_report(sites_df, CRITERIA_HIERARCHY_CONFIG, LEVEL_WEIGHTS)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
