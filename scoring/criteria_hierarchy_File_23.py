"""
criteria_hierarchy.py

Defines the full criteria tree used for site scoring (AHP synthesis,
weighted-sum, and TOPSIS aggregation) as a typed, importable Python
structure. Consumed by:

    - File 24/25/26 (elicitation/aggregation modules: ahp_pairwise.py,
      weighted_sum.py, topsis.py, ahp_synthesis.py) for pairwise
      comparisons and score aggregation
    - scoring/normalization/* for per-criterion normalization, which
      branches on `Criterion.direction` and `Criterion.scoring_mode`

Tree shape: Category -> Subcategory -> Criterion (leaf). Only leaves carry
scoring metadata; categories and subcategories exist purely for grouping
and (later) for hierarchical AHP weight elicitation.

Every leaf's `data_field` is a dotted path into the data model produced by
Phase 1/2 ingestion and schema work:

    - "demographic_data.<column>"        -> infra/migrations/001_core_schema.sql
    - "analysis_units.<column>"          -> infra/migrations/001_core_schema.sql
    - "competitor_locations.<column>"    -> infra/migrations/001_core_schema.sql (aggregated per unit)
    - "isochrone_cache.<column>"         -> infra/migrations/001_core_schema.sql (derived accessibility metrics)
    - "economic.<field>" / "property.<field>" / "client_estate.<field>" / "geospatial_base.<field>"
      -> ingestion/sources/{economic,property,client_estate,geospatial_base}.py
      (fields not yet materialized as DB columns as of Migration 1; expected
      to land in Migration 2/3 tables of the same source name)

Fields that don't yet have a concrete backing column (because their table
lands in Migration 2 or 3) are marked with `pending=True` so downstream
consumers can skip or stub them without breaking the tree structure.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(str, Enum):
    """Whether higher raw values are better (benefit) or worse (cost)."""
    BENEFIT = "benefit"
    COST = "cost"


class ScoringMode(str, Enum):
    """
    How a leaf's raw value is converted to a normalized score.

    MONOTONIC:   score increases (benefit) or decreases (cost) monotonically
                 with the raw value; direction alone determines shape.
    TARGET_BAND: score peaks within an ideal range and falls off on both
                 sides (non-monotonic) — direction is not meaningful and
                 should be ignored by normalization logic when this mode
                 is set.
    """
    MONOTONIC = "monotonic"
    TARGET_BAND = "target_band"


@dataclass(frozen=True)
class Criterion:
    """A single leaf (scorable) criterion."""
    key: str                       # stable machine key, used in weight/config lookups
    label: str                     # human-readable name
    data_field: str                # dotted path into Phase 1/2 data model (see module docstring)
    direction: Optional[Direction] # None when scoring_mode is TARGET_BAND
    scoring_mode: ScoringMode = ScoringMode.MONOTONIC
    target_band: Optional[tuple] = None  # (low, high) ideal range, required when scoring_mode == TARGET_BAND
    pending: bool = False           # True if data_field's backing table/column doesn't exist yet (later migration)
    notes: str = ""

    def __post_init__(self):
        if self.scoring_mode == ScoringMode.TARGET_BAND and self.target_band is None:
            raise ValueError(f"Criterion '{self.key}' is target_band scored but has no target_band range set.")
        if self.scoring_mode == ScoringMode.MONOTONIC and self.direction is None:
            raise ValueError(f"Criterion '{self.key}' is monotonic scored but has no direction set.")


@dataclass(frozen=True)
class Subcategory:
    key: str
    label: str
    criteria: tuple  # tuple[Criterion, ...]


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    subcategories: tuple  # tuple[Subcategory, ...]


# =============================================================================
# The tree
# =============================================================================

CRITERIA_TREE: tuple = (

    Category(
        key="population",
        label="Population",
        subcategories=(
            Subcategory(
                key="population_core",
                label="Population",
                criteria=(
                    Criterion(
                        key="resident_density",
                        label="Resident density",
                        data_field="demographic_data.population",
                        direction=Direction.BENEFIT,
                        notes="Normalized per unit area (H3 res-8 cell area is constant, so raw population "
                              "is already a valid density proxy within the platform's canonical grid).",
                    ),
                    Criterion(
                        key="daytime_population",
                        label="Daytime population",
                        data_field="geospatial_base.daytime_population_estimate",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Derived metric (workplace/footfall-based); not a direct Census field. "
                              "Backing table lands in geospatial_base ingestion, Migration 2/3.",
                    ),
                    Criterion(
                        key="projected_growth",
                        label="Projected growth",
                        data_field="demographic_data.population",  # trend computed across source_year rows
                        direction=Direction.BENEFIT,
                        notes="Computed as growth rate across multiple demographic_data.source_year snapshots "
                              "for the same unit_id, not a single stored column.",
                    ),
                    Criterion(
                        key="household_formation_rate",
                        label="Household formation rate",
                        data_field="demographic_data.household_count",  # trend across source_year
                        direction=Direction.BENEFIT,
                        notes="Computed as household_count growth across source_year snapshots.",
                    ),
                ),
            ),
        ),
    ),

    Category(
        key="demographics",
        label="Demographics",
        subcategories=(
            Subcategory(
                key="demographics_core",
                label="Demographics",
                criteria=(
                    Criterion(
                        key="age_band_representation",
                        label="Age band representation",
                        data_field="demographic_data.age_bands",
                        direction=Direction.BENEFIT,
                        notes="Score derived from alignment between demographic_data.age_bands (JSONB) "
                              "and the client's target age-band profile.",
                    ),
                    Criterion(
                        key="household_size_composition",
                        label="Household size / composition",
                        data_field="demographic_data.household_count",  # combined with population for avg size
                        direction=Direction.BENEFIT,
                        notes="Average household size derived as population / household_count.",
                    ),
                    Criterion(
                        key="life_stage_segments",
                        label="Life-stage segments",
                        data_field="economic.life_stage_segment_shares",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Segment shares (e.g. young family, retiree) from HCES/economic ingestion; "
                              "backing table lands in Migration 2/3.",
                    ),
                ),
            ),
        ),
    ),

    Category(
        key="spending",
        label="Spending",
        subcategories=(
            Subcategory(
                key="spending_core",
                label="Spending",
                criteria=(
                    Criterion(
                        key="category_relevant_expenditure",
                        label="Category-relevant expenditure",
                        data_field="economic.category_expenditure",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="HCES-derived category spend estimate, interpolated onto the hex grid; "
                              "backing table lands in Migration 2/3.",
                    ),
                    Criterion(
                        key="disposable_income",
                        label="Disposable income",
                        data_field="economic.disposable_income_estimate",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Backing table lands in Migration 2/3.",
                    ),
                    Criterion(
                        key="spend_index_vs_national",
                        label="Spend index against national baseline",
                        data_field="economic.spend_index",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Ratio metric (local / national baseline); backing table lands in Migration 2/3.",
                    ),
                ),
            ),
        ),
    ),

    Category(
        key="accessibility",
        label="Accessibility",
        subcategories=(
            Subcategory(
                key="accessibility_core",
                label="Accessibility",
                criteria=(
                    Criterion(
                        key="drive_time_reach",
                        label="Drive-time reach",
                        data_field="isochrone_cache.geom",  # area/population reachable within drive isochrone
                        direction=Direction.BENEFIT,
                        notes="Derived as population or area covered by the isochrone_cache polygon "
                              "for profile='drive' at the relevant range_value.",
                    ),
                    Criterion(
                        key="public_transport_proximity",
                        label="Public transport proximity",
                        data_field="geospatial_base.transit_stop_distance",
                        direction=Direction.COST,
                        pending=True,
                        notes="Distance to nearest transit stop (OSM-derived); smaller is better, "
                              "hence cost direction. Backing table lands in Migration 2/3.",
                    ),
                    Criterion(
                        key="parking",
                        label="Parking",
                        data_field="property.parking_capacity",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Backing table lands in Migration 2/3 (property ingestion).",
                    ),
                    Criterion(
                        key="pedestrian_footfall",
                        label="Pedestrian footfall",
                        data_field="geospatial_base.pedestrian_footfall_index",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Backing table lands in Migration 2/3.",
                    ),
                ),
            ),
        ),
    ),

    Category(
        key="competition",
        label="Competition",
        subcategories=(
            Subcategory(
                key="competition_core",
                label="Competition",
                criteria=(
                    Criterion(
                        key="competitor_density_in_catchment",
                        label="Competitor density within catchment",
                        data_field="competitor_locations.geom",  # count aggregated per unit/catchment
                        direction=None,
                        scoring_mode=ScoringMode.TARGET_BAND,
                        target_band=(1, 3),  # placeholder ideal count-per-catchment band; tune per category/format
                        notes="Explicitly target-band (non-monotonic) per spec: too few competitors can "
                              "signal an unproven/undersized market, too many signals saturation. "
                              "Count of competitor_locations rows falling within the unit's catchment "
                              "(k-ring of analysis_units), filtered by relevant category.",
                    ),
                    Criterion(
                        key="format_overlap",
                        label="Format overlap",
                        data_field="competitor_locations.category",
                        direction=Direction.COST,
                        notes="Share of nearby competitors matching the same retail format/category "
                              "as the candidate site; higher overlap is worse (cost).",
                    ),
                    Criterion(
                        key="distance_to_nearest_direct_competitor",
                        label="Distance to nearest direct competitor",
                        data_field="competitor_locations.geom",
                        direction=Direction.BENEFIT,
                        notes="Distance to nearest same-format competitor; farther is better (benefit), "
                              "in contrast to competitor_density_in_catchment which is target-band.",
                    ),
                ),
            ),
        ),
    ),

    Category(
        key="cannibalisation",
        label="Cannibalisation",
        subcategories=(
            Subcategory(
                key="cannibalisation_core",
                label="Cannibalisation",
                criteria=(
                    Criterion(
                        key="own_estate_catchment_overlap",
                        label="Overlap with existing own-estate catchments",
                        data_field="client_estate.catchment_geom",
                        direction=Direction.COST,
                        pending=True,
                        notes="Overlap area/proportion between candidate site's catchment and existing "
                              "client_estate store catchments; backing table lands in Migration 2/3.",
                    ),
                    Criterion(
                        key="expected_transfer_share",
                        label="Expected transfer share",
                        data_field="client_estate.expected_transfer_share",
                        direction=Direction.COST,
                        pending=True,
                        notes="Modeled % of new-site revenue expected to be transferred from existing "
                              "own-estate stores rather than net-new; backing table lands in Migration 2/3.",
                    ),
                ),
            ),
        ),
    ),

    Category(
        key="site_and_property",
        label="Site and property",
        subcategories=(
            Subcategory(
                key="site_and_property_core",
                label="Site and property",
                criteria=(
                    Criterion(
                        key="unit_size",
                        label="Unit size",
                        data_field="property.unit_size_sqft",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Direction is benefit as a default; if the client's format has a strict "
                              "ideal size range, switch to TARGET_BAND. Backing table lands in Migration 2/3.",
                    ),
                    Criterion(
                        key="frontage",
                        label="Frontage",
                        data_field="property.frontage_meters",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Backing table lands in Migration 2/3.",
                    ),
                    Criterion(
                        key="visibility",
                        label="Visibility",
                        data_field="property.visibility_score",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Backing table lands in Migration 2/3.",
                    ),
                    Criterion(
                        key="rent",
                        label="Rent",
                        data_field="property.rent_per_sqft",
                        direction=Direction.COST,
                        pending=True,
                        notes="Backing table lands in Migration 2/3.",
                    ),
                    Criterion(
                        key="lease_terms",
                        label="Lease terms",
                        data_field="property.lease_term_score",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Composite score of lease flexibility/duration favorability; "
                              "backing table lands in Migration 2/3.",
                    ),
                    Criterion(
                        key="fit_out_cost",
                        label="Fit-out cost",
                        data_field="property.fit_out_cost_estimate",
                        direction=Direction.COST,
                        pending=True,
                        notes="Backing table lands in Migration 2/3.",
                    ),
                ),
            ),
        ),
    ),

    Category(
        key="risk",
        label="Risk",
        subcategories=(
            Subcategory(
                key="risk_core",
                label="Risk",
                criteria=(
                    Criterion(
                        key="planning_constraints",
                        label="Planning constraints",
                        data_field="geospatial_base.planning_constraint_score",
                        direction=Direction.COST,
                        pending=True,
                        notes="Composite score of zoning/land-use restrictions; backing table lands "
                              "in Migration 2/3.",
                    ),
                    Criterion(
                        key="redevelopment_exposure",
                        label="Redevelopment exposure",
                        data_field="geospatial_base.redevelopment_risk_score",
                        direction=Direction.COST,
                        pending=True,
                        notes="Likelihood of area redevelopment disrupting the site/catchment; "
                              "backing table lands in Migration 2/3.",
                    ),
                    Criterion(
                        key="tenure_security",
                        label="Tenure security",
                        data_field="property.tenure_security_score",
                        direction=Direction.BENEFIT,
                        pending=True,
                        notes="Backing table lands in Migration 2/3.",
                    ),
                ),
            ),
        ),
    ),
)


def iter_leaves(tree: tuple = CRITERIA_TREE):
    """Yield (category, subcategory, criterion) for every leaf in the tree."""
    for category in tree:
        for subcategory in category.subcategories:
            for criterion in subcategory.criteria:
                yield category, subcategory, criterion


def get_criterion(key: str, tree: tuple = CRITERIA_TREE) -> Criterion:
    """Look up a single leaf Criterion by its stable key."""
    for _, _, criterion in iter_leaves(tree):
        if criterion.key == key:
            return criterion
    raise KeyError(f"No criterion found with key '{key}'")


def target_band_keys(tree: tuple = CRITERIA_TREE) -> tuple:
    """Convenience accessor: keys of all criteria requiring target-band (non-monotonic) scoring."""
    return tuple(
        c.key for _, _, c in iter_leaves(tree) if c.scoring_mode == ScoringMode.TARGET_BAND
    )


def pending_keys(tree: tuple = CRITERIA_TREE) -> tuple:
    """Convenience accessor: keys of all criteria whose backing data isn't materialized yet."""
    return tuple(c.key for _, _, c in iter_leaves(tree) if c.pending)
