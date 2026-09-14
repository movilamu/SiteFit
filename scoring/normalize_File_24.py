"""
scoring/normalize.py

Per-criterion normalization for the site-scoring pipeline.

Consumes the criteria tree defined in `criteria_hierarchy.py` (File 23):
each leaf `Criterion` carries a `direction` (BENEFIT / COST, meaningful only
for `ScoringMode.MONOTONIC`) and a `scoring_mode` (MONOTONIC or
TARGET_BAND). `normalize_all` walks a dataframe of raw per-site values and
produces a fully normalized [0, 1] matrix, dispatching to the right
normalization function per criterion:

    - MONOTONIC + BENEFIT -> normalize_benefit
    - MONOTONIC + COST    -> normalize_cost
    - TARGET_BAND         -> normalize_target_band (direction is ignored,
                              per the ScoringMode docstring in File 23)

All raw values are first robustly clipped at the 5th/95th percentile of
the *observed* data for that criterion, so a handful of extreme outliers
can't compress the rest of the site's scores toward the middle of the
[0, 1] range.

`pending=True` criteria (per File 23, their backing tables haven't landed
yet) and criteria with no matching column in `sites_df` are skipped rather
than raising, since the site-scoring dataframe legitimately won't have
every column until later migrations land.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Robust percentile clipping
# =============================================================================

def clip_percentiles(x, low_pct: float = 5, high_pct: float = 95):
    """
    Clip values of `x` below the `low_pct` percentile and above the
    `high_pct` percentile of `x` itself (i.e. clipping bounds are derived
    from the observed data, not externally supplied).

    Returns a numpy array of the same shape as `x`. NaNs are ignored when
    computing the percentile bounds and are passed through unchanged.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()

    finite = x[~np.isnan(x)]
    if finite.size == 0:
        return x.copy()

    lo = np.nanpercentile(finite, low_pct)
    hi = np.nanpercentile(finite, high_pct)
    if lo > hi:
        lo, hi = hi, lo

    return np.clip(x, lo, hi)


# =============================================================================
# Monotonic normalizers
# =============================================================================

def normalize_benefit(x, min_val: float, max_val: float):
    """
    x_tilde = (x - min) / (max - min)

    Higher raw values -> higher score. `x` is assumed to already be
    clipped/range-bounded by the caller (see `normalize_all`); this
    function just applies the linear rescale.
    """
    x = np.asarray(x, dtype=float)
    denom = max_val - min_val
    if denom == 0:
        # Degenerate: every observed value is identical -> no discriminating
        # signal, treat everyone as at the top of the band.
        return np.ones_like(x)
    return np.clip((x - min_val) / denom, 0.0, 1.0)


def normalize_cost(x, min_val: float, max_val: float):
    """
    x_tilde = (max - x) / (max - min)

    Lower raw values -> higher score (e.g. rent, distance-to-transit).
    """
    x = np.asarray(x, dtype=float)
    denom = max_val - min_val
    if denom == 0:
        return np.ones_like(x)
    return np.clip((max_val - x) / denom, 0.0, 1.0)


# =============================================================================
# Non-monotonic (target-band) normalizer
# =============================================================================

def normalize_target_band(
    x,
    target_min: float,
    target_max: float,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
):
    """
    Score is 1.0 for any value inside [target_min, target_max], and falls
    off smoothly (linearly) toward 0 the further a value sits outside the
    band, reaching 0 at the observed extreme on that side.

    `x_min` / `x_max` set the observed range used to scale the falloff on
    the low / high side respectively. If omitted, they default to the
    min/max of `x` itself. If the observed range on a side collapses to
    (i.e. equals) the band edge, that side is treated as an immediate
    cliff to 0 just outside the band.

    Examples (target_band = (1, 3)):
        x=1   -> 1.0   (inside band, lower edge)
        x=3   -> 1.0   (inside band, upper edge)
        x=2   -> 1.0   (inside band)
        x=0, x_min=-2 -> 0.5  (halfway from band edge to observed min)
        x=5, x_max=7  -> 0.5  (halfway from band edge to observed max)
    """
    x = np.asarray(x, dtype=float)

    if x_min is None:
        x_min = np.nanmin(x) if x.size else target_min
    if x_max is None:
        x_max = np.nanmax(x) if x.size else target_max

    # Falloff shouldn't extend *into* the band even if x_min/x_max are
    # inconsistent with the band bounds (e.g. supplied externally).
    x_min = min(x_min, target_min)
    x_max = max(x_max, target_max)

    score = np.ones_like(x)

    below = x < target_min
    above = x > target_max

    left_range = target_min - x_min
    right_range = x_max - target_max

    if np.any(below):
        if left_range <= 0:
            score[below] = 0.0
        else:
            score[below] = np.clip(1.0 - (target_min - x[below]) / left_range, 0.0, 1.0)

    if np.any(above):
        if right_range <= 0:
            score[above] = 0.0
        else:
            score[above] = np.clip(1.0 - (x[above] - target_max) / right_range, 0.0, 1.0)

    return score


# =============================================================================
# Full-matrix normalization driven by the File 23 criteria tree
# =============================================================================

def normalize_all(
    sites_df: pd.DataFrame,
    criteria_config: Iterable,
    low_pct: float = 5,
    high_pct: float = 95,
) -> pd.DataFrame:
    """
    Apply the correct normalization function per criterion and return a
    full normalized [0, 1] matrix, indexed the same as `sites_df`.

    Parameters
    ----------
    sites_df : DataFrame
        One row per site, one column per criterion `key` (raw values).
        Columns for criteria not present are simply skipped.
    criteria_config : iterable of `Criterion` (or any object exposing
        `.key`, `.direction`, `.scoring_mode`, `.target_band`, `.pending`)
        Typically `criteria_hierarchy.iter_leaves(...)` mapped down to the
        `Criterion` objects, or a flat list of `Criterion`s.
    low_pct, high_pct : float
        Percentile bounds used for robust clipping (default 5th/95th).

    Returns
    -------
    DataFrame of normalized scores in [0, 1], same index as `sites_df`,
    with one column per criterion that was actually scored.
    """
    from criteria_hierarchy import ScoringMode, Direction  # local import: avoid hard dep at module load

    out = {}

    for criterion in criteria_config:
        key = criterion.key

        if getattr(criterion, "pending", False):
            continue
        if key not in sites_df.columns:
            continue

        raw = sites_df[key].to_numpy(dtype=float)
        clipped = clip_percentiles(raw, low_pct, high_pct)

        if criterion.scoring_mode == ScoringMode.TARGET_BAND:
            target_min, target_max = criterion.target_band
            scores = normalize_target_band(
                clipped,
                target_min=target_min,
                target_max=target_max,
                x_min=np.nanmin(clipped),
                x_max=np.nanmax(clipped),
            )
        else:
            min_val = np.nanmin(clipped)
            max_val = np.nanmax(clipped)
            if criterion.direction == Direction.BENEFIT:
                scores = normalize_benefit(clipped, min_val, max_val)
            elif criterion.direction == Direction.COST:
                scores = normalize_cost(clipped, min_val, max_val)
            else:
                # Should be unreachable given Criterion.__post_init__ validation.
                raise ValueError(f"Criterion '{key}' has no usable direction for MONOTONIC scoring.")

        out[key] = scores

    return pd.DataFrame(out, index=sites_df.index)


# =============================================================================
# Unit tests
# =============================================================================

if __name__ == "__main__":
    import unittest

    class TestClipping(unittest.TestCase):
        def test_clip_extremes(self):
            # 5th/95th percentile of 0..99 (100 points) clips the tails.
            x = np.arange(100, dtype=float)
            clipped = clip_percentiles(x, 5, 95)
            self.assertAlmostEqual(clipped.min(), np.nanpercentile(x, 5))
            self.assertAlmostEqual(clipped.max(), np.nanpercentile(x, 95))
            # Interior values pass through unchanged.
            self.assertEqual(clipped[50], 50)

        def test_clip_no_variation(self):
            x = np.array([5.0, 5.0, 5.0])
            clipped = clip_percentiles(x)
            np.testing.assert_array_equal(clipped, x)

    class TestBenefitCost(unittest.TestCase):
        def test_benefit_by_hand(self):
            # min=10, max=30 -> x=10 -> 0.0, x=20 -> 0.5, x=30 -> 1.0
            x = np.array([10, 20, 30])
            got = normalize_benefit(x, 10, 30)
            np.testing.assert_allclose(got, [0.0, 0.5, 1.0])

        def test_cost_by_hand(self):
            # min=10, max=30 -> x=10 -> 1.0, x=20 -> 0.5, x=30 -> 0.0
            x = np.array([10, 20, 30])
            got = normalize_cost(x, 10, 30)
            np.testing.assert_allclose(got, [1.0, 0.5, 0.0])

        def test_degenerate_range(self):
            x = np.array([7, 7, 7])
            np.testing.assert_allclose(normalize_benefit(x, 7, 7), [1.0, 1.0, 1.0])
            np.testing.assert_allclose(normalize_cost(x, 7, 7), [1.0, 1.0, 1.0])

    class TestTargetBand(unittest.TestCase):
        def test_inside_band_is_one(self):
            x = np.array([1, 2, 3])  # band = (1, 3)
            got = normalize_target_band(x, target_min=1, target_max=3, x_min=-2, x_max=7)
            np.testing.assert_allclose(got, [1.0, 1.0, 1.0])

        def test_below_band_by_hand(self):
            # band (1, 3), x_min = -2 -> left_range = 1 - (-2) = 3
            # x=0 -> 1 - (1-0)/3 = 1 - 1/3 = 0.6667
            # x=-2 (observed min) -> 0.0
            got = normalize_target_band(
                np.array([0, -2]), target_min=1, target_max=3, x_min=-2, x_max=7
            )
            np.testing.assert_allclose(got, [2 / 3, 0.0], atol=1e-9)

        def test_above_band_by_hand(self):
            # band (1, 3), x_max = 7 -> right_range = 7 - 3 = 4
            # x=5 -> 1 - (5-3)/4 = 1 - 0.5 = 0.5
            # x=7 (observed max) -> 0.0
            got = normalize_target_band(
                np.array([5, 7]), target_min=1, target_max=3, x_min=-2, x_max=7
            )
            np.testing.assert_allclose(got, [0.5, 0.0], atol=1e-9)

        def test_cliff_when_no_observed_range_beyond_band(self):
            # x_max equals target_max -> no room to fall off -> immediate 0.
            got = normalize_target_band(
                np.array([3.5]), target_min=1, target_max=3, x_min=1, x_max=3
            )
            np.testing.assert_allclose(got, [0.0])

    class TestNormalizeAll(unittest.TestCase):
        def setUp(self):
            # Minimal stand-in for File 23's Criterion/ScoringMode/Direction,
            # imported lazily inside normalize_all so tests don't need the
            # real criteria_hierarchy module on the path.
            import types
            import sys

            mod = types.ModuleType("criteria_hierarchy")

            class Direction:
                BENEFIT = "benefit"
                COST = "cost"

            class ScoringMode:
                MONOTONIC = "monotonic"
                TARGET_BAND = "target_band"

            mod.Direction = Direction
            mod.ScoringMode = ScoringMode
            sys.modules["criteria_hierarchy"] = mod
            self.Direction = Direction
            self.ScoringMode = ScoringMode

            class Criterion:
                def __init__(self, key, direction=None, scoring_mode=ScoringMode.MONOTONIC,
                             target_band=None, pending=False):
                    self.key = key
                    self.direction = direction
                    self.scoring_mode = scoring_mode
                    self.target_band = target_band
                    self.pending = pending

            self.Criterion = Criterion

        def test_hand_verifiable_matrix(self):
            # 3 sites, 3 criteria: one benefit, one cost, one target-band.
            df = pd.DataFrame({
                "resident_density": [10, 20, 30],           # benefit, min=10 max=30
                "rent": [10, 20, 30],                        # cost, min=10 max=30
                "competitor_density_in_catchment": [1, 0, 5],  # target band (1,3)
            })

            criteria = [
                self.Criterion("resident_density", direction=self.Direction.BENEFIT),
                self.Criterion("rent", direction=self.Direction.COST),
                self.Criterion(
                    "competitor_density_in_catchment",
                    scoring_mode=self.ScoringMode.TARGET_BAND,
                    target_band=(1, 3),
                ),
            ]

            # No clipping distortion expected here: with only 3 points,
            # 5th/95th percentile clipping barely touches the extremes,
            # but to keep this hand-verifiable we disable it.
            result = normalize_all(df, criteria, low_pct=0, high_pct=100)

            np.testing.assert_allclose(result["resident_density"], [0.0, 0.5, 1.0])
            np.testing.assert_allclose(result["rent"], [1.0, 0.5, 0.0])

            # target band (1,3): observed range is [0, 5]
            # x=1 (in band) -> 1.0
            # x=0 -> left_range = 1-0=1 -> 1 - (1-0)/1 = 0.0
            # x=5 -> right_range = 5-3=2 -> 1 - (5-3)/2 = 0.0
            np.testing.assert_allclose(
                result["competitor_density_in_catchment"], [1.0, 0.0, 0.0]
            )

        def test_pending_and_missing_columns_skipped(self):
            df = pd.DataFrame({"resident_density": [1, 2, 3]})
            criteria = [
                self.Criterion("resident_density", direction=self.Direction.BENEFIT),
                self.Criterion("daytime_population", direction=self.Direction.BENEFIT, pending=True),
                self.Criterion("frontage", direction=self.Direction.BENEFIT),  # not in df, no column
            ]
            result = normalize_all(df, criteria)
            self.assertEqual(list(result.columns), ["resident_density"])

    unittest.main(verbosity=2)
