#!/usr/bin/env python3
"""
Diagnostic & Data Verification Script for Tamil Nadu Site-Scoring Pilot.
========================================================================

This script is an inspection and health-check tool (not part of the pipeline)
that connects directly to Supabase to verify ingestion completeness and data
sanity across all pilot tables:

1. Row Counts:
   - analysis_units
   - demographic_data
   - economic_data
   - competitor_locations
   - accessibility_data
   - property_listings

2. Data Completeness & Reconciliation:
   - Identifies analysis_units missing demographic or economic records.
   - Detects null values in critical indicators (population, income, expenditure).

3. Competitor Distribution:
   - Counts competitor locations grouped by commercial category.

4. Data Confidence Distribution:
   - Summarizes confidence levels across all source tables (e.g., % below 0.70 threshold).

5. Spot-Check Inspection:
   - Prints 5 random (or specified) analysis units with their full joined records
     across all tables, formatted cleanly for human eyeballing.

Usage:
    python ingestion/verify_data.py
    python ingestion/verify_data.py --threshold 0.75 --samples 5
    python ingestion/verify_data.py --unit-id "88618925d3fffff"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Supabase Client
try:
    from supabase import Client, create_client
except ImportError:
    print("ERROR: 'supabase' package is required. Install it using: pip install supabase")
    sys.exit(1)


# ============================================================================
# Formatting & Terminal Helpers
# ============================================================================

DIVIDER_HEAVY = "=" * 80
DIVIDER_LIGHT = "-" * 80
DIVIDER_SUB   = "·" * 80


def print_banner(title: str) -> None:
    print(f"\n{DIVIDER_HEAVY}")
    print(f" {title.upper()}")
    print(DIVIDER_HEAVY)


def print_section(title: str) -> None:
    print(f"\n{DIVIDER_LIGHT}")
    print(f"  {title}")
    print(DIVIDER_LIGHT)


def format_number(val: Any) -> str:
    if val is None:
        return "NULL (Missing)"
    if isinstance(val, (int, float)):
        if isinstance(val, float):
            return f"{val:,.2f}"
        return f"{val:,}"
    return str(val)


# ============================================================================
# Supabase Connection
# ============================================================================

def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        print("\n[!] CRITICAL: SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) are not set.")
        print("    Please configure your environment variables or provide a .env file.")
        sys.exit(1)

    try:
        return create_client(url.strip(), key.strip())
    except Exception as err:
        print(f"\n[!] Failed to connect to Supabase: {err}")
        sys.exit(1)


# ============================================================================
# Check 1: Table Row Counts
# ============================================================================

TARGET_TABLES = [
    "analysis_units",
    "demographic_data",
    "economic_data",
    "competitor_locations",
    "accessibility_data",
    "property_listings",
]


def check_table_counts(client: Client) -> Dict[str, Optional[int]]:
    print_section("1. TABLE ROW COUNTS")
    counts: Dict[str, Optional[int]] = {}

    for table in TARGET_TABLES:
        try:
            # PostgREST exact row count
            response = client.table(table).select("*", count="exact").limit(0).execute()
            count = response.count if response.count is not None else 0
            counts[table] = count
            status = "✓ OK" if count > 0 else "⚠ EMPTY"
            print(f"  • {table:<25} : {count:>7,} rows   [{status}]")
        except Exception as e:
            counts[table] = None
            print(f"  • {table:<25} : ERROR ({e})")

    return counts


# ============================================================================
# Check 2: Missing / Null Demographic & Economic Fields
# ============================================================================

def check_missing_or_null_fields(client: Client) -> None:
    print_section("2. ANALYSIS UNITS INTEGRITY & NULL CHECKS")

    try:
        units_resp = client.table("analysis_units").select("id, name").execute()
        units = units_resp.data or []
    except Exception as err:
        print(f"  [!] Failed to query analysis_units: {err}")
        return

    total_units = len(units)
    if total_units == 0:
        print("  [!] analysis_units is empty. Cannot check data coverage.")
        return

    unit_ids = [u["id"] for u in units]

    # Demographic check
    try:
        demo_resp = client.table("demographic_data").select("unit_id, population, household_count, literacy_rate").execute()
        demo_rows = demo_resp.data or []
    except Exception as err:
        demo_rows = []
        print(f"  [!] Failed to query demographic_data: {err}")

    demo_by_unit = {r["unit_id"]: r for r in demo_rows}
    missing_demo_units = [uid for uid in unit_ids if uid not in demo_by_unit]
    null_population_units = [
        uid for uid, r in demo_by_unit.items()
        if r.get("population") is None or r.get("population") == 0
    ]

    # Economic check
    try:
        econ_resp = client.table("economic_data").select(
            "unit_id, disposable_income_estimate, expenditure_index, spending_by_category"
        ).execute()
        econ_rows = econ_resp.data or []
    except Exception as err:
        econ_rows = []
        print(f"  [!] Failed to query economic_data: {err}")

    econ_by_unit = {r["unit_id"]: r for r in econ_rows}
    missing_econ_units = [uid for uid in unit_ids if uid not in econ_by_unit]
    null_income_units = [
        uid for uid, r in econ_by_unit.items()
        if r.get("disposable_income_estimate") is None
    ]
    null_expenditure_units = [
        uid for uid, r in econ_by_unit.items()
        if r.get("expenditure_index") is None
    ]

    print(f"  Total Analysis Units Ingested     : {total_units:,}")
    print()
    print("  Demographic Data Coverage:")
    print(f"    - Units completely missing demographics : {len(missing_demo_units):>6,} / {total_units:,}")
    if missing_demo_units:
        sample_missing = missing_demo_units[:5]
        print(f"      (Sample missing unit IDs: {', '.join(sample_missing)}{'...' if len(missing_demo_units) > 5 else ''})")
    print(f"    - Units with NULL/0 population           : {len(null_population_units):>6,} / {len(demo_rows):,}")

    print()
    print("  Economic Data Coverage:")
    print(f"    - Units completely missing economics    : {len(missing_econ_units):>6,} / {total_units:,}")
    if missing_econ_units:
        sample_missing = missing_econ_units[:5]
        print(f"      (Sample missing unit IDs: {', '.join(sample_missing)}{'...' if len(missing_econ_units) > 5 else ''})")
    print(f"    - Units with NULL disposable income     : {len(null_income_units):>6,} / {len(econ_rows):,}")
    print(f"      (Note: HCES 2024 published MPCE directly; disposable income may be NULL by design)")
    print(f"    - Units with NULL expenditure index     : {len(null_expenditure_units):>6,} / {len(econ_rows):,}")


# ============================================================================
# Check 3: Competitor Count Grouped by Category
# ============================================================================

def check_competitors_by_category(client: Client) -> None:
    print_section("3. COMPETITOR LOCATIONS GROUPED BY CATEGORY")

    try:
        response = client.table("competitor_locations").select("id, category").execute()
        rows = response.data or []
    except Exception as err:
        print(f"  [!] Failed to query competitor_locations: {err}")
        return

    if not rows:
        print("  [!] No competitor records found.")
        return

    category_counts = Counter()
    for row in rows:
        cat = row.get("category")
        if not cat:
            cat = "(Uncategorized / Null)"
        category_counts[cat.strip()] += 1

    total_competitors = sum(category_counts.values())
    print(f"  Total Competitor Locations Ingested: {total_competitors:,}\n")
    print(f"  {'Category':<35} {'Count':>8} {'Share (%)':>12}")
    print(f"  {'-'*35} {'-'*8} {'-'*12}")

    for cat, count in category_counts.most_common():
        pct = (count / total_competitors) * 100.0
        print(f"  {cat:<35} {count:>8,} {pct:>11.1f}%")


# ============================================================================
# Check 4: Data Confidence Distribution
# ============================================================================

CONFIDENCE_TABLES = [
    "demographic_data",
    "economic_data",
    "accessibility_data",
]


def check_confidence_distribution(client: Client, threshold: float = 0.70) -> None:
    print_section(f"4. DATA CONFIDENCE DISTRIBUTION (Threshold: < {threshold:.2f})")

    print(f"  {'Table Name':<25} {'Total':>7} {'< Threshold':>12} {'% Below':>10} {'Min':>6} {'Avg':>6} {'Max':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*12} {'-'*10} {'-'*6} {'-'*6} {'-'*6}")

    for table in CONFIDENCE_TABLES:
        try:
            resp = client.table(table).select("data_confidence").execute()
            rows = resp.data or []
        except Exception as err:
            print(f"  {table:<25} ERROR: {err}")
            continue

        if not rows:
            print(f"  {table:<25} {'0':>7} {'0':>12} {'0.0%':>10} {'-':>6} {'-':>6} {'-':>6}")
            continue

        vals = [
            float(r["data_confidence"])
            for r in rows
            if r.get("data_confidence") is not None
        ]

        if not vals:
            print(f"  {table:<25} {len(rows):>7} {'All Null':>12} {'100.0%':>10} {'-':>6} {'-':>6} {'-':>6}")
            continue

        total = len(vals)
        below = sum(1 for v in vals if v < threshold)
        pct_below = (below / total) * 100.0
        min_v = min(vals)
        max_v = max(vals)
        avg_v = sum(vals) / total

        flag = "⚠" if pct_below > 50.0 else " "
        print(
            f"{flag} {table:<25} {total:>7,} {below:>12,} {pct_below:>9.1f}% "
            f"{min_v:>6.2f} {avg_v:>6.2f} {max_v:>6.2f}"
        )

    print("\n  Note on confidence values:")
    print("    - demographic_data  : 1.0 (direct Census unit match), 0.70-0.85 (areal interpolation)")
    print("    - economic_data     : 0.35 (state-level uniform propagation), 0.70 (district aggregate)")
    print("    - accessibility_data: 0.50-0.85 (based on direct multi-modal transit & parking coverage)")


# ============================================================================
# Check 5: Spot Check Analysis Units (Joined Records Eyeball)
# ============================================================================

def spot_check_records(client: Client, sample_size: int = 5, specific_unit_id: Optional[str] = None) -> None:
    print_section(f"5. SPOT-CHECK INSPECTION ({sample_size} FULL JOINED RECORDS)")

    # 1. Fetch analysis units
    try:
        if specific_unit_id:
            units_resp = client.table("analysis_units").select("*").eq("id", specific_unit_id).execute()
        else:
            units_resp = client.table("analysis_units").select("*").limit(200).execute()
        units = units_resp.data or []
    except Exception as err:
        print(f"  [!] Failed to fetch analysis_units: {err}")
        return

    if not units:
        print("  [!] No analysis units found to inspect.")
        return

    if specific_unit_id:
        selected_units = units
    else:
        selected_units = random.sample(units, min(sample_size, len(units)))

    unit_ids = [u["id"] for u in selected_units]

    # 2. Fetch demographic records for sample
    try:
        demo_resp = client.table("demographic_data").select("*").in_("unit_id", unit_ids).execute()
        demo_map = {r["unit_id"]: r for r in demo_resp.data or []}
    except Exception:
        demo_map = {}

    # 3. Fetch economic records for sample
    try:
        econ_resp = client.table("economic_data").select("*").in_("unit_id", unit_ids).execute()
        econ_map = {r["unit_id"]: r for r in econ_resp.data or []}
    except Exception:
        econ_map = {}

    # 4. Fetch accessibility records for sample
    try:
        access_resp = client.table("accessibility_data").select("*").in_("unit_id", unit_ids).execute()
        access_map = {r["unit_id"]: r for r in access_resp.data or []}
    except Exception:
        access_map = {}

    # 5. Print spot-check cards
    for idx, unit in enumerate(selected_units, 1):
        uid = unit.get("id", "Unknown")
        name = unit.get("name") or "(Unnamed Cell)"
        demo = demo_map.get(uid)
        econ = econ_map.get(uid)
        access = access_map.get(uid)

        print(f"\n{DIVIDER_SUB}")
        print(f" [SPOT-CHECK {idx}/{len(selected_units)}] Unit ID: {uid} | Name: {name}")
        print(DIVIDER_SUB)

        # Demographics
        print("  • DEMOGRAPHICS (Census):")
        if demo:
            pop = format_number(demo.get("population"))
            hh = format_number(demo.get("household_count"))
            lit = f"{demo.get('literacy_rate'):.2f}%" if demo.get("literacy_rate") is not None else "NULL"
            work = f"{demo.get('workforce_participation'):.2f}%" if demo.get("workforce_participation") is not None else "NULL"
            conf = format_number(demo.get("data_confidence"))
            year = demo.get("source_year", "N/A")

            # Parse age bands
            age_bands_raw = demo.get("age_bands")
            if isinstance(age_bands_raw, str):
                try:
                    age_bands_raw = json.loads(age_bands_raw)
                except Exception:
                    pass

            print(f"      Total Population         : {pop}")
            print(f"      Household Count          : {hh}")
            print(f"      Literacy Rate            : {lit}")
            print(f"      Workforce Participation  : {work}")
            print(f"      Age Bands Breakdown      : {age_bands_raw}")
            print(f"      Data Confidence          : {conf} (Source Year: {year})")
        else:
            print("      [!] No demographic_data record found for this unit.")

        # Economics
        print("  • ECONOMICS (HCES):")
        if econ:
            mpce = f"₹ {econ.get('mpce'):,.2f}" if econ.get("mpce") is not None else "NULL"
            income = f"₹ {econ.get('disposable_income_estimate'):,.2f}" if econ.get("disposable_income_estimate") is not None else "NULL"
            exp_idx = format_number(econ.get("expenditure_index"))
            conf = format_number(econ.get("data_confidence"))
            year = econ.get("source_year", "N/A")
            cat_spend = econ.get("spending_by_category")

            print(f"      MPCE (Per Capita Spend)  : {mpce}")
            print(f"      Disposable Income Est.   : {income}")
            print(f"      Expenditure Index        : {exp_idx}")
            print(f"      Spending by Category     : {cat_spend if cat_spend else 'None'}")
            print(f"      Data Confidence          : {conf} (Source Year: {year})")
        else:
            print("      [!] No economic_data record found for this unit.")

        # Accessibility
        print("  • ACCESSIBILITY (OSM & Transit):")
        if access:
            t_score = f"{access.get('transit_score'):.2f} / 100.0" if access.get("transit_score") is not None else "NULL"
            p_score = f"{access.get('parking_availability'):.2f} / 100.0" if access.get("parking_availability") is not None else "NULL"
            footfall = format_number(access.get("footfall_estimate"))
            conf = format_number(access.get("data_confidence"))

            print(f"      Transit Score            : {t_score}")
            print(f"      Parking Availability     : {p_score}")
            print(f"      Footfall Estimate        : {footfall} (Null expected if no sensors)")
            print(f"      Data Confidence          : {conf}")
        else:
            print("      [!] No accessibility_data record found for this unit.")


# ============================================================================
# Main Orchestrator
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Supabase data ingestion integrity for Tamil Nadu site-scoring pilot."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for distribution warning (default: 0.70).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of random analysis units to spot check (default: 5).",
    )
    parser.add_argument(
        "--unit-id",
        type=str,
        default=None,
        help="Specific analysis unit ID to inspect instead of random sampling.",
    )
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Run only table count checks.",
    )

    args = parser.parse_args()

    print_banner("Tamil Nadu Pilot: Supabase Data Verification & Diagnostics")

    client = get_supabase_client()

    # 1. Row counts
    check_table_counts(client)
    if args.counts_only:
        return

    # 2. Missing & Null checks
    check_missing_or_null_fields(client)

    # 3. Competitors by category
    check_competitors_by_category(client)

    # 4. Confidence distribution
    check_confidence_distribution(client, threshold=args.threshold)

    # 5. Spot-check printout
    spot_check_records(client, sample_size=args.samples, specific_unit_id=args.unit_id)

    print(f"\n{DIVIDER_HEAVY}")
    print(" Verification run completed.")
    print(DIVIDER_HEAVY)


if __name__ == "__main__":
    main()
