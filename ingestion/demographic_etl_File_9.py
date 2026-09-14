#!/usr/bin/env python3
"""
Demographic ETL Ingestion Script for Tamil Nadu Pilot Sub-Region (Chennai District).

Source: Census of India 2011 - Primary Census Abstract (PCA) / District Census Handbook (DCHB)
Target: Supabase Postgres tables (`analysis_units` and `demographic_data`)

Execution flow:
1. Load configuration and initialize Supabase client from environment variables.
2. Acquire Census 2011 PCA dataset (from direct file path, URL, or local fallback).
3. Filter records for the Chennai pilot sub-region (District 603 / State 33 - Tamil Nadu).
4. Parse demographic indicators: total population, age bands (e.g., 0-6 years),
   household count, literacy rate, and workforce participation rate.
5. Map administrative unit records to canonical H3 grid cells in `analysis_units`.
6. Batch upsert processed records into `demographic_data` table.
"""

import argparse
import json
import logging
import os
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from supabase import Client, create_client

# ============================================================================
# Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("demographic_etl")

# ============================================================================
# Constants & Defaults
# ============================================================================
CENSUS_SOURCE_YEAR = 2011
CHENNAI_DISTRICT_CODE = "603"  # 2011 Census District Code for Chennai, TN
DEFAULT_BATCH_SIZE = 100

# Default Census PCA Column Mappings (standard Census 2011 PCA layout)
PCA_COL_MAP = {
    "STATE": ["State", "STATE", "ST_CODE"],
    "DISTRICT": ["District", "DISTRICT", "DIST_CODE"],
    "SUB_DISTRICT": ["Subdistt", "SUB_DISTRICT", "SUBDIST_CODE"],
    "TOWN_VILLAGE": ["Town/Village", "TOWN_VILLAGE", "VILLAGE_CODE"],
    "NAME": ["Name", "NAME", "MDDS_NAME"],
    "LEVEL": ["Level", "LEVEL"],
    "TRU": ["TRU", "TRU_TYPE"],  # Total / Rural / Urban
    "HOUSEHOLDS": ["No_HH", "HOUSEHOLDS", "NO_HH"],
    "TOT_POP": ["TOT_P", "TOT_POP", "TOTAL_POPULATION"],
    "POP_0_6": ["P_06", "POP_0_6", "P_0_6"],
    "LITERATE_POP": ["P_LIT", "LITERATE_POP", "P_LIT_TOT"],
    "WORKER_POP": ["TOT_WORK_P", "TOTAL_WORKERS", "TOT_WORK"],
}


def get_supabase_client() -> Client:
    """Read credentials from environment and initialize Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        logger.critical(
            "Missing required environment variables: SUPABASE_URL and SUPABASE_KEY must be set."
        )
        sys.exit(1)

    try:
        return create_client(url, key)
    except Exception as err:
        logger.critical(f"Failed to initialize Supabase client: {err}")
        sys.exit(1)


def download_or_load_census_data(file_path: Optional[str], download_url: Optional[str]) -> pd.DataFrame:
    """Load Census Excel/CSV data from local path or URL."""
    target_path = file_path

    if not target_path and download_url:
        logger.info(f"Downloading Census dataset from: {download_url}")
        target_path = "/tmp/census_chennai_2011.xlsx"
        try:
            urllib.request.urlretrieve(download_url, target_path)
            logger.info(f"Downloaded source file to {target_path}")
        except Exception as err:
            logger.error(f"Failed to download dataset from URL: {err}")
            raise

    if not target_path:
        # Fallback to local default file path if present
        target_path = "data/DDW_PCA3302_2011_MDDS_with_UI.xlsx"
        if not os.path.exists(target_path):
            target_path = "data/census_chennai_2011.csv"

    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Census data source file not found at '{target_path}'. "
            "Please provide a valid --file-path or --download-url."
        )

    logger.info(f"Reading Census data from: {target_path}")
    if target_path.endswith(".csv"):
        df = pd.read_csv(target_path, dtype=str)
    else:
        df = pd.read_excel(target_path, dtype=str)

    logger.info(f"Successfully loaded {len(df)} rows from raw source.")
    return df


def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Map dynamic DataFrame columns to standardized canonical Census names."""
    df.columns = [str(c).strip() for c in df.columns]
    canonical_map = {}

    for canonical_key, candidate_names in PCA_COL_MAP.items():
        found = None
        for candidate in candidate_names:
            matches = [c for c in df.columns if c.upper() == candidate.upper()]
            if matches:
                found = matches[0]
                break
        if found:
            canonical_map[canonical_key] = found
        else:
            logger.warning(f"Could not automatically map column for '{canonical_key}'. Candidates tested: {candidate_names}")

    return df, canonical_map


def filter_chennai_pilot_data(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    """Filter records specifically for the Chennai pilot sub-region and remove macro-aggregate rows."""
    dist_col = col_map.get("DISTRICT")
    level_col = col_map.get("LEVEL")
    tru_col = col_map.get("TRU")

    filtered_df = df.copy()

    # Filter by District code if present
    if dist_col and dist_col in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df[dist_col].astype(str).str.strip().str.zfill(3) == CHENNAI_DISTRICT_CODE.zfill(3)
        ]
        logger.info(f"Filtered for Chennai District ({CHENNAI_DISTRICT_CODE}): {len(filtered_df)} records remaining.")

    # Exclude State/District summary rows (keep Ward/Village/Town granular records)
    if level_col and level_col in filtered_df.columns:
        filtered_df = filtered_df[
            ~filtered_df[level_col].astype(str).str.upper().isin(["STATE", "DISTRICT"])
        ]

    # Prefer 'Total' or 'Urban' rows for Chennai Corporation
    if tru_col and tru_col in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df[tru_col].astype(str).str.upper().isin(["TOTAL", "URBAN"])
        ]

    logger.info(f"Final scoped records for processing: {len(filtered_df)}")
    return filtered_df


def parse_demographic_record(row: pd.Series, col_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Safe extraction and calculation of demographic attributes from a Census row."""
    def _safe_int(key: str) -> int:
        col = col_map.get(key)
        if not col or col not in row or pd.isna(row[col]):
            return 0
        try:
            return int(float(str(row[col]).replace(",", "").strip()))
        except (ValueError, TypeError):
            return 0

    tot_pop = _safe_int("TOT_POP")
    if tot_pop <= 0:
        return None  # Skip zero-population or invalid rows

    households = _safe_int("HOUSEHOLDS")
    pop_0_6 = _safe_int("POP_0_6")
    literates = _safe_int("LITERATE_POP")
    workers = _safe_int("WORKER_POP")

    # Calculated rates
    literacy_rate = round((literates / tot_pop) * 100.0, 4) if tot_pop > 0 else 0.0
    workforce_participation = round((workers / tot_pop) * 100.0, 4) if tot_pop > 0 else 0.0

    # Structuring age bands (Census PCA provides 0-6, remaining pop as 7+)
    age_bands = {
        "0_6": pop_0_6,
        "7_plus": max(0, tot_pop - pop_0_6),
    }

    raw_id = str(row.get(col_map.get("TOWN_VILLAGE", ""), "")).strip() or str(row.get(col_map.get("NAME", ""), "")).strip()

    return {
        "raw_geo_id": raw_id,
        "population": tot_pop,
        "household_count": households,
        "literacy_rate": literacy_rate,
        "workforce_participation": workforce_participation,
        "age_bands": age_bands,
    }


def map_record_to_analysis_unit(
    record: Dict[str, Any],
    existing_units: Dict[str, str]
) -> Tuple[Optional[str], float]:
    """
    Map Census administrative unit to an analysis_units H3 cell ID.

    TODO(File 16):
    -------------------------------------------------------------------------
    This module currently performs direct 1:1 key/polygon matching.
    File 16 (areal interpolation) must hook in here to handle spatial crosswalks
    where administrative boundary polygons (e.g., Census 2011 wards/villages)
    overlap multiple H3 resolution-8 cells (`analysis_units`).

    When File 16 is integrated:
      1. Intersect Ward/Village geometry with H3 grid geometry.
      2. Compute area-weighted or population-weighted proportions for each cell.
      3. Emit multiple demographic records per unit, adjusting counts proportionally.
      4. Lower `data_confidence` (e.g., to 0.70 - 0.85 depending on overlap fit).
    -------------------------------------------------------------------------
    """
    raw_id = record["raw_geo_id"]

    # Direct lookup attempt against fetched analysis units
    if raw_id in existing_units:
        # Direct match found
        unit_id = existing_units[raw_id]
        data_confidence = 1.0  # Direct 1:1 match starts at full confidence
        return unit_id, data_confidence

    # Fallback for pilot stage: If raw_id is already a valid H3 index
    if len(raw_id) == 15 and raw_id.isalnum():
        return raw_id, 1.0

    # No direct match found without spatial interpolation
    logger.debug(
        f"Unit '{raw_id}' does not directly match any analysis_unit ID. "
        "Areal interpolation (File 16) required for allocation."
    )
    return None, 0.0


def fetch_existing_analysis_units(supabase: Client) -> Dict[str, str]:
    """Fetch existing analysis_units from Supabase to validate mapping target."""
    logger.info("Fetching existing analysis_units grid cells from Supabase...")
    try:
        response = supabase.table("analysis_units").select("id, name").execute()
        units_map = {}
        for row in response.data:
            units_map[row["id"]] = row["id"]
            if row.get("name"):
                units_map[row["name"]] = row["id"]
        logger.info(f"Retrieved {len(response.data)} active analysis_units for reconciliation.")
        return units_map
    except Exception as err:
        logger.error(f"Failed to fetch analysis_units from database: {err}")
        return {}


def upsert_demographic_batch(supabase: Client, records: List[Dict[str, Any]], dry_run: bool = False) -> int:
    """Batch upsert processed demographic records into Supabase `demographic_data` table."""
    if not records:
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Would upsert {len(records)} records into demographic_data.")
        return len(records)

    try:
        response = supabase.table("demographic_data").upsert(
            records,
            on_conflict="unit_id,source_year"
        ).execute()

        inserted_count = len(response.data) if response.data else len(records)
        logger.info(f"Successfully upserted {inserted_count} records into demographic_data.")
        return inserted_count
    except Exception as err:
        logger.error(f"Failed to write batch to Supabase: {err}")
        raise


def run_etl(
    file_path: Optional[str] = None,
    download_url: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> None:
    """Main ETL orchestration pipeline."""
    logger.info("=== Starting Demographic ETL Ingestion Pipeline ===")
    supabase = get_supabase_client()

    # Step 1: Load Raw Dataset
    try:
        raw_df = download_or_load_census_data(file_path, download_url)
    except Exception as err:
        logger.critical(f"ETL pipeline halted during data acquisition: {err}")
        sys.exit(1)

    # Step 2: Normalize and Filter
    df_norm, col_map = normalize_column_names(raw_df)
    pilot_df = filter_chennai_pilot_data(df_norm, col_map)

    if pilot_df.empty:
        logger.error("No valid Census records found matching Chennai pilot scope. Exiting.")
        sys.exit(1)

    # Step 3: Fetch Target Analysis Units for Matching
    existing_units = fetch_existing_analysis_units(supabase)

    # Step 4: Parse & Transform Records
    upsert_buffer = []
    total_processed = 0
    total_matched = 0
    total_unmatched = 0

    for idx, row in pilot_df.iterrows():
        parsed = parse_demographic_record(row, col_map)
        if not parsed:
            continue

        total_processed += 1
        unit_id, confidence = map_record_to_analysis_unit(parsed, existing_units)

        if not unit_id:
            total_unmatched += 1
            continue

        total_matched += 1
        db_record = {
            "unit_id": unit_id,
            "population": parsed["population"],
            "age_bands": json.dumps(parsed["age_bands"]),
            "household_count": parsed["household_count"],
            "literacy_rate": parsed["literacy_rate"],
            "workforce_participation": parsed["workforce_participation"],
            "data_confidence": confidence,  # 1.0 for direct match; lowering once File 16 interpolation is applied
            "source_year": CENSUS_SOURCE_YEAR,
        }
        upsert_buffer.append(db_record)

        # Flush batch
        if len(upsert_buffer) >= batch_size:
            upsert_demographic_batch(supabase, upsert_buffer, dry_run=dry_run)
            upsert_buffer.clear()

    # Flush remaining buffer
    if upsert_buffer:
        upsert_demographic_batch(supabase, upsert_buffer, dry_run=dry_run)
        upsert_buffer.clear()

    # Step 5: Final Execution Summary & Diagnostics
    logger.info("=== Demographic ETL Pipeline Execution Summary ===")
    logger.info(f"Total Census records processed : {total_processed}")
    logger.info(f"Successfully matched & written  : {total_matched}")
    logger.info(f"Unmatched (Pending File 16)     : {total_unmatched}")

    if total_unmatched > 0:
        logger.warning(
            f"{total_unmatched} records could not be mapped directly to analysis_units. "
            "Integrate File 16 (areal interpolation) to resolve non-matching boundary geometries."
        )

    if total_matched == 0 and not dry_run:
        logger.error("Zero records were written to Supabase. Marking execution as failed.")
        sys.exit(1)

    logger.info("ETL Ingestion completed successfully.")


# ============================================================================
# CLI Entrypoint
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="ETL Pipeline: Ingest Census Demographic Data for Tamil Nadu Pilot into Supabase."
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Path to local Census 2011 Excel/CSV data file.",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=None,
        help="Direct URL to download Census PCA/DCHB file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of records per database upsert batch (default: 100).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform parsing and matching validation without writing to Supabase.",
    )

    args = parser.parse_args()

    run_etl(
        file_path=args.file_path,
        download_url=args.download_url,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()