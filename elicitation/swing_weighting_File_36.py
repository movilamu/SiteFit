"""
elicitation/swing_weighting.py

Implements Swing Weighting for Multi-Criteria Decision Analysis (MCDA).

Why Swing Weighting Anchors Weights to Data Ranges:
---------------------------------------------------
Traditional weighting methods (such as direct point allocation or standard AHP)
often suffer from the "abstract importance fallibility." Decision-makers, when
asked in abstract terms, will state that criteria like "Rent" or "Safety" are
extremely important. However, in a specific candidate selection scenario, if
all candidate sites have nearly identical rent (e.g., varying only between
$48/sqft and $50/sqft), rent has minimal ability to differentiate between options.
Assigning a large weight to rent in this context distorts decision outcomes by
letting a static factor consume proportion in the weighting space.

Swing weighting solves this by anchoring subjective preference directly to the
actual observed data range [min_val, max_val] across candidate sites.

Instead of asking "How important is Criterion A in general?", swing weighting
presents a thought experiment:
    1. Imagine a hypothetical baseline candidate site that sits at the worst
       observed value across ALL criteria.
    2. If you could "swing" exactly ONE criterion from its worst observed level
       to its best observed level across candidates, which one provides the
       highest overall increase in utility?

The criterion offering the most valuable swing is assigned top rank and a score
of 100. Remaining criteria are then rated from 0 to 100 relative to that top
swing score.

As a result:
- Criteria with wide ranges or high contextual impact receive higher weights.
- Criteria that barely vary across candidate sites yield negligible swing
  value and naturally receive near-zero weights, preventing uninformative
  data fields from influencing site rankings.

Worked Example:
---------------
Criteria evaluated across candidate sites:
1. 'pedestrian_footfall'    Range: [1,000, 10,000 visitors/day]
2. 'rent_per_sqft'          Range: [$20, $50 / sqft]
3. 'competitor_density'     Range: [2, 3 competitors in catchment]

Swing Order:
    ['pedestrian_footfall', 'rent_per_sqft', 'competitor_density']

Swing Ratings:
    'pedestrian_footfall': 100  (Moving 1,000 -> 10,000 gives greatest value)
    'rent_per_sqft':        70  (Moving $50 -> $20 rent is 70% as valuable)
    'competitor_density':   10  (Moving 3 -> 2 competitors has minor impact due to narrow range)

Sum of Ratings = 100 + 70 + 10 = 180

Expected Normalized Weights:
    'pedestrian_footfall': 100 / 180 = 0.555556 (55.56%)
    'rent_per_sqft':        70 / 180 = 0.388889 (38.89%)
    'competitor_density':   10 / 180 = 0.055556 (5.56%)
"""

from typing import Dict, List, Tuple


def swing_weight(
    criteria_ranges: Dict[str, Tuple[float, float]],
    swing_order: List[str],
    swing_ratings: Dict[str, float],
) -> Dict[str, float]:
    """
    Computes normalized weights using the Swing Weighting method.

    Parameters
    ----------
    criteria_ranges : Dict[str, Tuple[float, float]]
        Map of criterion key to observed tuple (min_val, max_val) across candidates.
    swing_order : List[str]
        List of criterion keys ordered from highest preference swing to lowest.
    swing_ratings : Dict[str, float]
        Map of criterion key to assigned swing rating relative to top criterion (0 to 100).

    Returns
    -------
    Dict[str, float]
        Map of criterion key to normalized weight (summing to 1.0).

    Raises
    ------
    ValueError
        If inputs are empty, keys mismatch across inputs, range bounds are invalid,
        or rating sum is non-positive.
    """
    if not criteria_ranges or not swing_order or not swing_ratings:
        raise ValueError("Inputs 'criteria_ranges', 'swing_order', and 'swing_ratings' must not be empty.")

    keys_set = set(criteria_ranges.keys())
    if set(swing_order) != keys_set or set(swing_ratings.keys()) != keys_set:
        raise ValueError("Criteria keys must match across criteria_ranges, swing_order, and swing_ratings.")

    adjusted_ratings: Dict[str, float] = {}

    for key, (min_val, max_val) in criteria_ranges.items():
        if min_val > max_val:
            raise ValueError(f"Invalid range for criterion '{key}': min_val ({min_val}) > max_val ({max_val}).")

        rating = swing_ratings[key]
        if rating < 0:
            raise ValueError(f"Rating for criterion '{key}' cannot be negative (got {rating}).")

        # Zero variance across candidate sites means no swing utility can be realized
        if min_val == max_val:
            adjusted_ratings[key] = 0.0
        else:
            adjusted_ratings[key] = rating

    total_rating = sum(adjusted_ratings.values())
    if total_rating <= 0:
        raise ValueError("Sum of swing ratings must be greater than zero.")

    # Compute normalized weights rounded to 6 decimal places
    weights = {key: round(adjusted_ratings[key] / total_rating, 6) for key in swing_order}

    # Adjust rounding discrepancy on top criterion so sum equals exactly 1.0
    weight_sum = sum(weights.values())
    if weight_sum != 1.0 and weights:
        top_key = swing_order[0]
        weights[top_key] = round(weights[top_key] + (1.0 - weight_sum), 6)

    return weights


if __name__ == "__main__":
    # Worked Example
    sample_ranges = {
        "pedestrian_footfall": (1000.0, 10000.0),
        "rent_per_sqft": (20.0, 50.0),
        "competitor_density": (2.0, 3.0),
    }

    sample_order = ["pedestrian_footfall", "rent_per_sqft", "competitor_density"]

    sample_ratings = {
        "pedestrian_footfall": 100.0,
        "rent_per_sqft": 70.0,
        "competitor_density": 10.0,
    }

    computed_weights = swing_weight(sample_ranges, sample_order, sample_ratings)

    print("--- Swing Weighting Worked Example ---")
    for criterion, weight in computed_weights.items():
        print(f"Criterion: {criterion:<20} | Weight: {weight:.6f} ({weight * 100:.2f}%)")