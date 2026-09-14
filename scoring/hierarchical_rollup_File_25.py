"""
scoring/hierarchical_rollup.py

Calculates global leaf weights for hierarchical criteria trees by multiplying
local weights down each branch of the hierarchy:

    w(global, leaf) = w(level_1) * w(level_2) * ... * w(leaf)

Validates that:
  1. Local weights at each level sum to 1.0 (within tolerance).
  2. Aggregated global leaf weights sum to 1.0 across all leaves.
"""

import math
from typing import Any, Dict, List, Set, Tuple


def _extract_weight(parent_key: str, child_key: str, level_weights: Dict[str, Any]) -> float:
    """Extracts local weight for a child node under a parent node from level_weights."""
    # Grouped format: level_weights[parent_key][child_key]
    if parent_key in level_weights and isinstance(level_weights[parent_key], dict):
        if child_key in level_weights[parent_key]:
            return float(level_weights[parent_key][child_key])

    # Flat format: level_weights[child_key]
    if child_key in level_weights and isinstance(level_weights[child_key], (int, float)):
        return float(level_weights[child_key])

    raise KeyError(
        f"Missing weight assignment for node '{child_key}' under parent '{parent_key}' in level_weights."
    )


def _build_hierarchy_map(hierarchy_config: Any) -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Parses hierarchy configuration representations (Category dataclass tuples, dicts,
    or nested tree structures) into a canonical parent-to-children mapping and leaf key set.
    """
    parent_to_children: Dict[str, List[str]] = {}
    all_nodes: Set[str] = set()
    parent_nodes: Set[str] = set()

    # Case 1: Tuple/List of Category objects (e.g. CRITERIA_TREE)
    if isinstance(hierarchy_config, (tuple, list)):
        root_children = []
        for cat in hierarchy_config:
            cat_key = getattr(cat, "key", str(cat))
            root_children.append(cat_key)
            all_nodes.add(cat_key)

            if hasattr(cat, "subcategories"):
                sub_keys = []
                parent_nodes.add(cat_key)
                for sub in cat.subcategories:
                    sub_key = getattr(sub, "key", str(sub))
                    sub_keys.append(sub_key)
                    all_nodes.add(sub_key)

                    if hasattr(sub, "criteria"):
                        crit_keys = []
                        parent_nodes.add(sub_key)
                        for crit in sub.criteria:
                            c_key = getattr(crit, "key", str(crit))
                            crit_keys.append(c_key)
                            all_nodes.add(c_key)
                        parent_to_children[sub_key] = crit_keys
                    parent_to_children[cat_key] = sub_keys
            elif hasattr(cat, "criteria"):
                crit_keys = []
                parent_nodes.add(cat_key)
                for crit in cat.criteria:
                    c_key = getattr(crit, "key", str(crit))
                    crit_keys.append(c_key)
                    all_nodes.add(c_key)
                parent_to_children[cat_key] = crit_keys

        parent_to_children["root"] = root_children
        parent_nodes.add("root")
        leaf_keys = all_nodes - parent_nodes
        return parent_to_children, leaf_keys

    # Case 2: Dictionary representation
    if isinstance(hierarchy_config, dict):
        if "root" in hierarchy_config:
            for parent, children in hierarchy_config.items():
                parent_nodes.add(parent)
                all_nodes.add(parent)
                child_keys = []
                for child in children:
                    c_key = getattr(child, "key", str(child)) if not isinstance(child, str) else child
                    child_keys.append(c_key)
                    all_nodes.add(c_key)
                parent_to_children[parent] = child_keys
        else:
            top_level = list(hierarchy_config.keys())
            parent_to_children["root"] = top_level
            parent_nodes.add("root")

            def _walk_dict(node_key: str, children_structure: Any):
                all_nodes.add(node_key)
                if isinstance(children_structure, (list, tuple)):
                    parent_nodes.add(node_key)
                    child_keys = []
                    for child in children_structure:
                        if isinstance(child, dict):
                            for sub_k, sub_v in child.items():
                                child_keys.append(sub_k)
                                _walk_dict(sub_k, sub_v)
                        else:
                            c_key = getattr(child, "key", str(child))
                            child_keys.append(c_key)
                            all_nodes.add(c_key)
                    parent_to_children[node_key] = child_keys
                elif isinstance(children_structure, dict):
                    parent_nodes.add(node_key)
                    child_keys = list(children_structure.keys())
                    parent_to_children[node_key] = child_keys
                    for sub_k, sub_v in children_structure.items():
                        _walk_dict(sub_k, sub_v)

            for top_k, top_v in hierarchy_config.items():
                _walk_dict(top_k, top_v)

        all_nodes.discard("root")
        leaf_keys = all_nodes - parent_nodes
        return parent_to_children, leaf_keys

    raise ValueError(f"Unsupported hierarchy_config format: {type(hierarchy_config)}")


def validate_level_weights(
    parent_to_children: Dict[str, List[str]],
    level_weights: Dict[str, Any],
    tolerance: float = 1e-5,
) -> None:
    """
    Validates that weights at each level of the hierarchy sum to 1.0 (within tolerance).
    Raises ValueError if any level fails validation.
    """
    for parent_key, children in parent_to_children.items():
        if not children:
            continue

        level_sum = 0.0
        weight_breakdown = {}
        for child_key in children:
            w = _extract_weight(parent_key, child_key, level_weights)
            weight_breakdown[child_key] = w
            level_sum += w

        if not math.isclose(level_sum, 1.0, abs_tol=tolerance):
            raise ValueError(
                f"Level weight validation failed for parent '{parent_key}': "
                f"weights sum to {level_sum:.6f}, expected 1.0 (±{tolerance}). "
                f"Assigned weights: {weight_breakdown}"
            )


def validate_global_leaf_weights(
    global_leaf_weights: Dict[str, float],
    tolerance: float = 1e-5,
) -> None:
    """
    Validates that global leaf weights sum to 1.0 across all leaf nodes.
    Raises ValueError if validation fails.
    """
    if not global_leaf_weights:
        raise ValueError("Global leaf weights dictionary is empty.")

    total_sum = sum(global_leaf_weights.values())
    if not math.isclose(total_sum, 1.0, abs_tol=tolerance):
        raise ValueError(
            f"Global leaf weight validation failed: total leaf weights sum to {total_sum:.6f}, "
            f"expected 1.0 (±{tolerance})."
        )


def compute_global_weights(
    hierarchy_config: Any,
    level_weights: Dict[str, Any],
    tolerance: float = 1e-5,
) -> Dict[str, float]:
    """
    Computes global leaf weights from per-level relative weight assignments:
        w(global, leaf) = w(criterion) * w(sub-criterion) * ...

    Args:
        hierarchy_config: Tree structure defining the hierarchy (Category tuples or dicts).
        level_weights: Relative weights per level (grouped dict or flat node-weight dict).
        tolerance: Absolute numerical tolerance for weight sum validation (default: 1e-5).

    Returns:
        Dict mapping leaf node keys to their computed global weights.

    Raises:
        ValueError: If weights at any level do not sum to 1.0, or if the global leaf weights
                    sum does not equal 1.0 across all leaves.
        KeyError: If a required weight is missing from level_weights.
    """
    parent_to_children, leaf_keys = _build_hierarchy_map(hierarchy_config)

    # 1. Validate per-level weight sums
    validate_level_weights(parent_to_children, level_weights, tolerance=tolerance)

    # 2. Compute global leaf weights recursively
    global_leaf_weights: Dict[str, float] = {}

    def _traverse(node_key: str, current_weight: float):
        children = parent_to_children.get(node_key, [])
        if node_key in leaf_keys or not children:
            global_leaf_weights[node_key] = current_weight
            return

        for child_key in children:
            local_w = _extract_weight(node_key, child_key, level_weights)
            _traverse(child_key, current_weight * local_w)

    root_children = parent_to_children.get("root", [])
    for top_node in root_children:
        top_w = _extract_weight("root", top_node, level_weights)
        _traverse(top_node, top_w)

    # 3. Validate overall global leaf weight sum
    validate_global_leaf_weights(global_leaf_weights, tolerance=tolerance)

    return global_leaf_weights


def run_worked_example() -> Dict[str, float]:
    """
    Worked Example with 2 top-level criteria and 2 sub-criteria each.

    Hierarchy:
      - Population (weight: 0.60)
          |- resident_density           (local weight: 0.70)
          |- daytime_population         (local weight: 0.30)
      - Accessibility (weight: 0.40)
          |- drive_time_reach           (local weight: 0.80)
          |- public_transport_proximity  (local weight: 0.20)

    Hand Calculations:
      - Level 1 sum: 0.60 + 0.40 = 1.00
      - Population sub-criteria sum: 0.70 + 0.30 = 1.00
      - Accessibility sub-criteria sum: 0.80 + 0.20 = 1.00

      Global Leaf Weights:
        w(resident_density)           = 0.60 * 0.70 = 0.42
        w(daytime_population)         = 0.60 * 0.30 = 0.18
        w(drive_time_reach)           = 0.40 * 0.80 = 0.32
        w(public_transport_proximity) = 0.40 * 0.20 = 0.08

      Leaf Sum Verification:
        0.42 + 0.18 + 0.32 + 0.08 = 1.00
    """
    hierarchy_config = {
        "Population": ["resident_density", "daytime_population"],
        "Accessibility": ["drive_time_reach", "public_transport_proximity"],
    }

    level_weights = {
        "root": {
            "Population": 0.60,
            "Accessibility": 0.40,
        },
        "Population": {
            "resident_density": 0.70,
            "daytime_population": 0.30,
        },
        "Accessibility": {
            "drive_time_reach": 0.80,
            "public_transport_proximity": 0.20,
        },
    }

    return compute_global_weights(hierarchy_config, level_weights)


if __name__ == "__main__":
    result = run_worked_example()
    print("Worked Example Execution Results:")
    print("---------------------------------")
    for leaf, global_weight in result.items():
        print(f"  {leaf:28s} -> {global_weight:.4f}")
    print("---------------------------------")
    print(f"Total Leaf Weight Sum: {sum(result.values()):.4f}")