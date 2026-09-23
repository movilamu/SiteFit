"""Thin bridge so `from scoring.matrix import get_normalized_matrix` (used as a
fallback by api/sensitivity_routes.py) resolves to the same loader the rest
of the API already uses.
"""

from __future__ import annotations

from api.scenario_routes import get_normalized_matrix  # noqa: F401
