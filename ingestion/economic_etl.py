#!/usr/bin/env python3
"""
Economic ETL for the Tamil Nadu site-scoring pilot.

Source:
    MoSPI / NSS Household Consumption Expenditure Survey (HCES) 2023-24
    (HCES 2022-23 is also accepted via --source-year).

The File 1 source assessment says HCES provides detailed expenditure data at
state and rural/urban levels, while fine district-level disposable income and
spending-by-category data are not uniformly available. Therefore this ETL
supports both:
  1. district-level input, when present; and
  2. state-level HCES input applied uniformly to the pilot's analysis units.

When state-level values are propagated to smaller H3 cells, data_confidence is
intentionally reduced and the condition is logged explicitly.

Expected input:
    CSV, JSON/JSONL, or Parquet containing at least a geography level and
    expenditure/income fields. The exact public HCES microdata layout varies
    by release/file, so column aliases are configurable below and can also be
    supplied through CLI options.

Examples:
    python ingestion/economic_etl.py \
        --input data/hces_2023_24.csv \
        --pilot-district "Chennai" \
        --source-year 2024

    python ingestion/economic_etl.py \
        --input data/hces_2023_24.csv \
        --pilot-district "Chennai" \
        --source-year 2024 \
        --state-code 33 \
        --upsert-batch-size 250

Environment:
    SUPABASE_URL
    SUPABASE_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from supabase import Client, create_client


LOG = logging.getLogger("economic_etl")

# HCES-derived state-level application is materially less precise than a
# district/unit-specific estimate. Keep this visibly conservative.
CONFIDENCE_STATE_LEVEL = 0.35
CONFIDENCE_DISTRICT_LEVEL = 0.70
CONFIDENCE_UNIT_LEVEL = 0.90

DEFAULT_PILOT_DISTRICT = "Chennai"
DEFAULT_SOURCE_YEAR = 2024  # HCES 2023-24
DEFAULT_STATE_NAME = "Tamil Nadu"

# Common aliases seen in HCES extracts or normalized staging files.
ALIAS_SETS = {
    "state": [
        "state",
        "state_name",
        "state_ut",
        "state_ut_name",
        "state_code",
        "state_ut_code",
    ],
    "district": [
        "district",
        "district_name",
        "district_code",
    ],
    "geography_level": [
        "geography_level",
        "geo_level",
        "level",
        "geography",
    ],
    "mpce": [
        "mpce",
        "monthly_per_capita_consumption_expenditure",
        "monthly_per_capita_expenditure",
        "mpce_rs",
        "mpce_inr",
    ],
    "disposable_income": [
        "disposable_income_estimate",
        "disposable_income",
        "income_estimate",
        "estimated_disposable_income",
    ],
    "expenditure_index": [
        "expenditure_index",
        "consumption_expenditure_index",
        "expenditure_idx",
    ],
    "spending_by_category": [
        "spending_by_category",
        "spend_by_category",
        "expenditure_by_category",
        "category_spend",
    ],
}


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def get_supabase() -> Client:
    return create_client(required_env("SUPABASE_URL"), required_env("SUPABASE_KEY"))


def read_input(path: Path) -> pd.DataFrame:
    LOG.info("Reading economic source file: %s", path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported input format '{suffix}'. Use CSV, JSON/JSONL, or Parquet."
    )


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in out.columns
    ]
    return out


def find_column(df: pd.DataFrame, logical_name: str) -> str | None:
    aliases = ALIAS_SETS[logical_name]
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def first_present(row: pd.Series, candidates: Iterable[str]) -> Any:
    for col in candidates:
        if col in row.index:
            value = row[col]
            if pd.notna(value) and value != "":
                return value
    return None


def parse_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def parse_json_object(value: Any) -> dict[str, Any] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            # Permit simple "food=1000;housing=2000" staging files.
            result: dict[str, float] = {}
            for token in text.split(";"):
                if "=" not in token:
                    continue
                key, raw = token.split("=", 1)
                number = parse_number(raw)
                if number is not None:
                    result[key.strip()] = number
            return result or None

    return None


def canonicalize_category_values(value: Any) -> dict[str, float] | None:
    parsed = parse_json_object(value)
    if not parsed:
        return None

    result: dict[str, float] = {}
    for key, raw in parsed.items():
        number = parse_number(raw)
        if number is not None:
            result[str(key)] = number
    return result or None


def infer_geography_level(row: pd.Series, state_col: str | None,
                          district_col: str | None,
                          geo_col: str | None) -> str:
    if geo_col:
        raw = str(row.get(geo_col, "")).strip().lower()
        if raw in {"unit", "h3", "analysis_unit", "cell"}:
            return "unit"
        if raw in {"district", "district_level"}:
            return "district"
        if raw in {"state", "state_level"}:
            return "state"

    if district_col and pd.notna(row.get(district_col)):
        district = str(row[district_col]).strip()
        if district and district.lower() not in {"nan", "none", "all"}:
            return "district"

    return "state"


def normalize_state(value: Any) -> str:
    return str(value).strip().lower()


def district_matches(value: Any, pilot_district: str) -> bool:
    if value is None or pd.isna(value):
        return False
    a = str(value).strip().lower()
    b = pilot_district.strip().lower()
    return a == b or a.replace(" district", "") == b.replace(" district", "")


def load_analysis_units(supabase: Client) -> list[dict[str, Any]]:
    """
    Load the canonical analysis units.

    The ETL only needs IDs for this state/district-level pilot because HCES
    values are propagated uniformly. A geometry is not required unless a
    future unit-level HCES/model source is supplied.
    """
    LOG.info("Loading analysis_units from Supabase")
    response = supabase.table("analysis_units").select("id").execute()
    rows = response.data or []

    if not rows:
        raise RuntimeError("analysis_units returned no rows")

    return rows


def extract_hces_metrics(
    df: pd.DataFrame,
    pilot_district: str,
    state_name: str,
    source_year: int,
) -> tuple[dict[str, Any], str]:
    """
    Select the best available HCES aggregate for the pilot.

    Priority:
      unit -> district -> state

    The normal free HCES path for this pilot is state-level. If a district
    record exists in the supplied staging file, it is preferred.
    """
    state_col = find_column(df, "state")
    district_col = find_column(df, "district")
    geo_col = find_column(df, "geography_level")
    mpce_col = find_column(df, "mpce")
    income_col = find_column(df, "disposable_income")
    index_col = find_column(df, "expenditure_index")
    category_col = find_column(df, "spending_by_category")

    if not any([mpce_col, income_col, index_col, category_col]):
        raise ValueError(
            "Input has no recognized economic metric columns. "
            f"Expected aliases for: {', '.join(ALIAS_SETS)}"
        )

    if state_col:
        state_mask = df[state_col].map(normalize_state) == normalize_state(state_name)
        state_df = df[state_mask].copy()
    else:
        LOG.warning(
            "No state column found; treating the input as already restricted "
            "to %s.",
            state_name,
        )
        state_df = df.copy()

    if state_df.empty:
        raise ValueError(f"No rows found for state '{state_name}'")

    state_df["_geo_level"] = state_df.apply(
        lambda row: infer_geography_level(row, state_col, district_col, geo_col),
        axis=1,
    )

    # Prefer an explicit district record for Chennai if the source contains it.
    district_df = state_df[
        (state_df["_geo_level"] == "district")
        & state_df[district_col].map(lambda x: district_matches(x, pilot_district))
    ] if district_col else state_df.iloc[0:0]

    if not district_df.empty:
        chosen = district_df
        level = "district"
    else:
        unit_df = state_df[state_df["_geo_level"] == "unit"]
        if not unit_df.empty:
            chosen = unit_df
            level = "unit"
        else:
            chosen = state_df[state_df["_geo_level"] == "state"]
            if chosen.empty:
                # Some normalized files omit geography_level and contain only
                # one state aggregate. Use the whole filtered state frame.
                chosen = state_df
            level = "state"

    def numeric_aggregate(column: str | None) -> float | None:
        if not column:
            return None
        values = pd.to_numeric(chosen[column], errors="coerce").dropna()
        if values.empty:
            return None
        return float(values.mean())

    category_values: dict[str, float] | None = None
    if category_col:
        category_objects = [
            canonicalize_category_values(v)
            for v in chosen[category_col].tolist()
        ]
        category_objects = [x for x in category_objects if x]
        if category_objects:
            keys = sorted({k for obj in category_objects for k in obj})
            category_values = {
                key: float(
                    sum(obj.get(key, 0.0) for obj in category_objects)
                    / len(category_objects)
                )
                for key in keys
            }

    mpce = numeric_aggregate(mpce_col)
    disposable_income = numeric_aggregate(income_col)

    # Disposable income is not directly published at district level in the
    # confirmed source. If an input has only MPCE, keep the field null rather
    # than pretending MPCE equals disposable income.
    expenditure_index = numeric_aggregate(index_col)

    metrics = {
        "spending_by_category": category_values,
        "disposable_income_estimate": disposable_income,
        "expenditure_index": expenditure_index,
        "mpce": mpce,
        "geography_level": level,
        "source_year": source_year,
    }

    return metrics, level


def confidence_for_level(level: str) -> float:
    return {
        "unit": CONFIDENCE_UNIT_LEVEL,
        "district": CONFIDENCE_DISTRICT_LEVEL,
        "state": CONFIDENCE_STATE_LEVEL,
    }[level]


def build_rows(
    analysis_units: list[dict[str, Any]],
    metrics: dict[str, Any],
    pilot_district: str,
    state_name: str,
) -> list[dict[str, Any]]:
    level = metrics["geography_level"]
    confidence = confidence_for_level(level)

    if level == "state":
        LOG.warning(
            "LOW CONFIDENCE ECONOMIC DATA: HCES source is state-level and will "
            "be applied uniformly to every analysis_unit in pilot district '%s'. "
            "This does NOT represent cell-level disposable income.",
            pilot_district,
        )
    elif level == "district":
        LOG.warning(
            "MODERATE CONFIDENCE ECONOMIC DATA: district-level values for '%s' "
            "will be applied uniformly to smaller analysis units.",
            pilot_district,
        )

    rows: list[dict[str, Any]] = []
    for unit in analysis_units:
        row = {
            "unit_id": unit["id"],
            "spending_by_category": metrics["spending_by_category"],
            "disposable_income_estimate": metrics["disposable_income_estimate"],
            "expenditure_index": metrics["expenditure_index"],
            "data_confidence": confidence,
            "source_year": metrics["source_year"],
        }

        # Keep provenance in logs rather than inventing a schema column that
        # does not exist in migration 002.
        if level == "state":
            LOG.debug(
                "unit_id=%s: applying %s state-level HCES metrics; "
                "confidence=%.2f; state=%s",
                unit["id"],
                metrics["source_year"],
                confidence,
                state_name,
            )

        rows.append(row)

    return rows


def upsert_batches(
    supabase: Client,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> int:
    if not rows:
        return 0

    total = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        try:
            # UNIQUE(unit_id, source_year) makes this idempotent for reruns.
            response = (
                supabase.table("economic_data")
                .upsert(batch, on_conflict="unit_id,source_year")
                .execute()
            )
            count = len(response.data or batch)
            total += count
            LOG.info(
                "Upserted economic_data batch %d-%d (%d rows)",
                start + 1,
                start + len(batch),
                count,
            )
        except Exception:
            LOG.exception(
                "Failed to upsert economic_data batch beginning at row %d",
                start + 1,
            )
            raise

    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest HCES economic data into Supabase economic_data."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="HCES-derived CSV, JSON/JSONL, or Parquet staging file.",
    )
    parser.add_argument(
        "--pilot-district",
        default=DEFAULT_PILOT_DISTRICT,
        help="Pilot district to ingest (default: Chennai).",
    )
    parser.add_argument(
        "--state-name",
        default=DEFAULT_STATE_NAME,
        help="State name used to filter the source (default: Tamil Nadu).",
    )
    parser.add_argument(
        "--source-year",
        type=int,
        default=DEFAULT_SOURCE_YEAR,
        help="HCES source year. 2024 denotes HCES 2023-24.",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=250,
        help="Number of economic_data rows per Supabase upsert.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    try:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")

        if args.upsert_batch_size < 1:
            raise ValueError("--upsert-batch-size must be >= 1")

        supabase = get_supabase()
        df = normalize_column_names(read_input(args.input))

        LOG.info(
            "HCES economic ETL starting: state=%s district=%s source_year=%s",
            args.state_name,
            args.pilot_district,
            args.source_year,
        )

        metrics, level = extract_hces_metrics(
            df=df,
            pilot_district=args.pilot_district,
            state_name=args.state_name,
            source_year=args.source_year,
        )

        LOG.info(
            "Selected %s-level source metrics: mpce=%s, disposable_income=%s, "
            "expenditure_index=%s, categories=%s",
            level,
            metrics["mpce"],
            metrics["disposable_income_estimate"],
            metrics["expenditure_index"],
            bool(metrics["spending_by_category"]),
        )

        analysis_units = load_analysis_units(supabase)
        rows = build_rows(
            analysis_units=analysis_units,
            metrics=metrics,
            pilot_district=args.pilot_district,
            state_name=args.state_name,
        )

        written = upsert_batches(
            supabase=supabase,
            rows=rows,
            batch_size=args.upsert_batch_size,
        )

        LOG.info(
            "Economic ETL completed successfully: %d analysis units written; "
            "source geography=%s; confidence=%.2f",
            written,
            level,
            confidence_for_level(level),
        )
        return 0

    except Exception as exc:
        LOG.exception("Economic ETL failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
