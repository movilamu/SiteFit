"""
own_estate_etl.py

Structural ETL template for importing client-owned store location data
into the `own_estate_locations` table.

NOTE: This is a template only. The client's own store data (name,
coordinates, and current trading performance) must be supplied later
as a CSV file matching the format below. No real data pull happens here.

--------------------------------------------------------------------------
Expected CSV format (own_estate_locations_import.csv):

    store_name,latitude,longitude,current_trading_performance
    Riverside Store,51.5074,-0.1278,1250000.00
    Northgate Retail Park,53.4808,-2.2426,980000.50
    Old Town Branch,52.4862,-1.8904,

Notes on columns:
    store_name                  - text, required, non-empty
    latitude                    - numeric, required, range -90 to 90
    longitude                   - numeric, required, range -180 to 180
    current_trading_performance - numeric, optional (may be blank/null)
--------------------------------------------------------------------------

Environment variables required:
    SUPABASE_URL - Supabase project URL
    SUPABASE_KEY - Supabase service role or anon key with insert privileges

Usage:
    python own_estate_etl.py /path/to/own_estate_locations_import.csv
"""

import csv
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

try:
    from supabase import create_client, Client
except ImportError:  # pragma: no cover
    create_client = None
    Client = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("own_estate_etl")


REQUIRED_COLUMNS = [
    "store_name",
    "latitude",
    "longitude",
    "current_trading_performance",
]


class RowValidationError(Exception):
    """Raised when a single CSV row fails validation."""


@dataclass
class OwnEstateRow:
    name: str
    latitude: float
    longitude: float
    current_trading_performance: Optional[float]

    def to_record(self) -> dict:
        """Convert to a Supabase-insertable record.

        geometry is expressed as WKT (EWKT with SRID 4326) so Supabase/
        PostGIS can cast it directly to the GEOMETRY(Point, 4326) column.
        """
        return {
            "name": self.name,
            "geometry": f"SRID=4326;POINT({self.longitude} {self.latitude})",
            "current_trading_performance": self.current_trading_performance,
        }


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


def _validate_header(fieldnames: Optional[List[str]]) -> None:
    if fieldnames is None:
        raise ValueError("CSV file appears to be empty (no header row found).")

    missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )


def _parse_float(value: str, field_name: str, row_number: int, required: bool = True) -> Optional[float]:
    value = (value or "").strip()

    if value == "":
        if required:
            raise RowValidationError(
                f"Row {row_number}: '{field_name}' is missing but required."
            )
        return None

    try:
        return float(value)
    except ValueError:
        raise RowValidationError(
            f"Row {row_number}: '{field_name}' value '{value}' is not a valid number."
        )


def _validate_row(row: dict, row_number: int) -> OwnEstateRow:
    """Validate a single CSV row and return a structured OwnEstateRow.

    Raises RowValidationError with a clear message on any problem.
    """
    store_name = (row.get("store_name") or "").strip()
    if not store_name:
        raise RowValidationError(f"Row {row_number}: 'store_name' is missing or empty.")

    latitude = _parse_float(row.get("latitude", ""), "latitude", row_number, required=True)
    longitude = _parse_float(row.get("longitude", ""), "longitude", row_number, required=True)

    if latitude is not None and not (-90.0 <= latitude <= 90.0):
        raise RowValidationError(
            f"Row {row_number}: 'latitude' value {latitude} is out of range (-90 to 90)."
        )
    if longitude is not None and not (-180.0 <= longitude <= 180.0):
        raise RowValidationError(
            f"Row {row_number}: 'longitude' value {longitude} is out of range (-180 to 180)."
        )

    # current_trading_performance is optional -- nullable per schema
    performance = _parse_float(
        row.get("current_trading_performance", ""),
        "current_trading_performance",
        row_number,
        required=False,
    )

    return OwnEstateRow(
        name=store_name,
        latitude=latitude,
        longitude=longitude,
        current_trading_performance=performance,
    )


def import_own_estate_csv(filepath: str, batch_size: int = 500) -> dict:
    """Load a client-provided CSV of store locations into own_estate_locations.

    Args:
        filepath: Path to the CSV file. Must contain columns:
            store_name, latitude, longitude, current_trading_performance.
        batch_size: Number of rows to insert per Supabase batch call.

    Returns:
        A summary dict: {"inserted": int, "skipped": int, "errors": list}

    Raises:
        FileNotFoundError: if filepath does not exist.
        ValueError: if the CSV header is missing required columns.
        EnvironmentError: if Supabase credentials are not configured.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    logger.info("Starting import from '%s'", filepath)

    valid_rows: List[OwnEstateRow] = []
    errors: List[str] = []

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        _validate_header(reader.fieldnames)

        for i, raw_row in enumerate(reader, start=2):  # row 1 is the header
            try:
                valid_rows.append(_validate_row(raw_row, i))
            except RowValidationError as exc:
                logger.warning(str(exc))
                errors.append(str(exc))

    logger.info(
        "Validation complete: %d valid row(s), %d invalid row(s).",
        len(valid_rows),
        len(errors),
    )

    if not valid_rows:
        logger.warning("No valid rows to import. Aborting insert.")
        return {"inserted": 0, "skipped": len(errors), "errors": errors}

    supabase = _get_supabase_client()

    inserted = 0
    try:
        for start in range(0, len(valid_rows), batch_size):
            batch = valid_rows[start : start + batch_size]
            records = [row.to_record() for row in batch]

            logger.info(
                "Inserting batch of %d record(s) into own_estate_locations "
                "(rows %d-%d)...",
                len(records),
                start + 1,
                start + len(records),
            )

            response = supabase.table("own_estate_locations").insert(records).execute()

            batch_inserted = len(response.data) if getattr(response, "data", None) else 0
            inserted += batch_inserted

    except Exception as exc:  # noqa: BLE001 - surface any Supabase/network error
        logger.error("Insert failed: %s", exc)
        raise

    logger.info(
        "Import finished. Inserted: %d, Skipped (invalid): %d",
        inserted,
        len(errors),
    )

    return {"inserted": inserted, "skipped": len(errors), "errors": errors}


def main() -> None:
    if len(sys.argv) != 2:
        logger.error("Usage: python own_estate_etl.py <path_to_csv>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        summary = import_own_estate_csv(filepath)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error handler
        logger.error("Import failed: %s", exc)
        sys.exit(1)

    if summary["errors"]:
        logger.warning(
            "Import completed with %d skipped row(s). See warnings above for details.",
            summary["skipped"],
        )

    logger.info("Done. %d row(s) inserted.", summary["inserted"])


if __name__ == "__main__":
    main()
