"""
scoring/precompute_matrix.py

EXPENSIVE, offline, weight-independent precomputation.

This is the batch step that must run against a full sites dataframe whenever
the underlying data snapshot changes. It:

  1. Applies hard constraints (File 26 / scoring/constraints.py) so
     non-viable sites never enter the decision matrix and cannot distort
     percentile clipping / min-max ranges.
  2. Normalizes every scorable leaf criterion onto [0, 1]
     (File 24 / scoring/normalize.py).
  3. Prepares the hierarchical weight *structure* from File 25
     (scoring/hierarchical_rollup.py): parent→children map and ordered
     leaf keys. Global weights themselves are NOT computed here — they
     arrive later from user input. This module only makes the tree
     File 32 will multiply against.
  4. Versions the resulting matrix and writes it to Supabase (table row
     plus a Storage object) tagged with a data-release timestamp / version
     ID, so any later score can be traced to the snapshot it used.

Contrast with File 32 (the cheap, per-interaction step):
    File 32 loads one already-normalized, already-versioned matrix and
    applies the user's current level weights → File 25 global rollup →
    Files 27–29 aggregation. That path is O(sites × criteria) arithmetic
    with no clipping, no constraint scan, and no I/O of the raw snapshot.
    Re-running THIS module on every slider movement would be the wrong
    split; re-running File 32 is the right one.

Typical flow
------------
    artifact = build_normalized_matrix(sites_df, CRITERIA_TREE,
                                       constraints_dict={...})
    persist_normalized_matrix(artifact)   # offline / CI / data-release job
    # File 32 later: load_normalized_matrix(version_id) + user weights
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


try:
    from scoring.constraints import apply_hard_constraints
    from scoring.normalize import normalize_all
    from scoring.hierarchical_rollup import _build_hierarchy_map
except ImportError:  # running as a loose script alongside Files 24–26
    from constraints import apply_hard_constraints
    from normalize import normalize_all
    from hierarchical_rollup import _build_hierarchy_map


# Default Postgres table + Storage bucket used to version snapshots.
MATRIX_VERSIONS_TABLE = os.environ.get(
    "SCORING_MATRIX_VERSIONS_TABLE", "normalized_matrix_versions"
)
MATRIX_STORAGE_BUCKET = os.environ.get(
    "SCORING_MATRIX_STORAGE_BUCKET", "scoring-matrices"
)
DEFAULT_ID_COLUMN = "site_id"


# =============================================================================
# Artifact
# =============================================================================

@dataclass
class NormalizedMatrixArtifact:
    """Versioned, weight-independent output of the offline precompute step."""

    matrix: pd.DataFrame
    version_id: str
    data_release_at: str
    parent_to_children: Dict[str, List[str]]
    leaf_keys: List[str]
    scored_criterion_keys: List[str]
    excluded_sites: List[Dict[str, Any]] = field(default_factory=list)
    storage_path: Optional[str] = None
    n_sites_in: int = 0
    n_sites_out: int = 0
    persisted: bool = False

    def to_metadata(self) -> Dict[str, Any]:
        """JSON-serialisable header stored alongside the matrix blob."""
        return {
            "version_id": self.version_id,
            "data_release_at": self.data_release_at,
            "parent_to_children": self.parent_to_children,
            "leaf_keys": self.leaf_keys,
            "scored_criterion_keys": self.scored_criterion_keys,
            "n_sites_in": self.n_sites_in,
            "n_sites_out": self.n_sites_out,
            "n_excluded": len(self.excluded_sites),
            "storage_path": self.storage_path,
            "weight_independent": True,
            "global_weights": None,
            "notes": (
                "Global weights are not stored on this snapshot; File 32 "
                "applies user-supplied level weights at interaction time."
            ),
        }


# =============================================================================
# Hierarchy helpers (File 25 structure only — no numeric weights)
# =============================================================================

def prepare_weight_structure(
    criteria_hierarchy_config: Any,
) -> Tuple[Dict[str, List[str]], List[str], Dict[str, Dict[str, None]]]:
    """
    Build the File 25 hierarchy map without assigning weights.

    Returns
    -------
    parent_to_children : dict
        Canonical parent → ordered child keys, including the synthetic
        ``"root"`` node used by ``compute_global_weights``.
    leaf_keys : list
        Ordered leaf criterion keys (the columns File 32 will weight).
    level_weight_template : dict
        Nested dict with the same shape ``compute_global_weights`` expects
        for ``level_weights``, values left as ``None`` for the UI / File 32
        to fill in.
    """
    parent_to_children, leaf_key_set = _build_hierarchy_map(criteria_hierarchy_config)
    leaf_keys = _ordered_leaves(parent_to_children, leaf_key_set)

    template: Dict[str, Dict[str, None]] = {}
    for parent, children in parent_to_children.items():
        template[parent] = {child: None for child in children}

    return parent_to_children, leaf_keys, template


def _ordered_leaves(
    parent_to_children: Dict[str, List[str]],
    leaf_key_set: Set[str],
) -> List[str]:
    """Depth-first leaf order matching File 25's traversal from ``root``."""
    ordered: List[str] = []

    def _walk(node: str) -> None:
        children = parent_to_children.get(node, [])
        if node in leaf_key_set or not children:
            if node != "root":
                ordered.append(node)
            return
        for child in children:
            _walk(child)

    for top in parent_to_children.get("root", []):
        _walk(top)
    return ordered


def _collect_leaf_criteria(criteria_hierarchy_config: Any) -> List[Any]:
    """
    Flatten the hierarchy into the Criterion objects File 24's
    ``normalize_all`` expects (``.key``, ``.direction``, ``.scoring_mode``, …).
    """
    tree = _unwrap_tree(criteria_hierarchy_config)

    # File 23 Category / Subcategory / Criterion tree
    if isinstance(tree, (tuple, list)) and tree and hasattr(tree[0], "subcategories"):
        leaves = []
        for cat in tree:
            for sub in getattr(cat, "subcategories", ()):
                leaves.extend(list(getattr(sub, "criteria", ())))
            if hasattr(cat, "criteria") and not hasattr(cat, "subcategories"):
                leaves.extend(list(cat.criteria))
        return leaves

    # Already a flat sequence of Criterion-like objects
    if isinstance(tree, (tuple, list)) and tree and hasattr(tree[0], "key"):
        if not hasattr(tree[0], "subcategories"):
            return list(tree)

    # Dict / key-only config: resolve leaves against File 23 if available
    _, leaf_keys, _ = prepare_weight_structure(tree)
    try:
        from criteria_hierarchy import get_criterion
    except ImportError:
        try:
            from scoring.criteria_hierarchy import get_criterion
        except ImportError:
            raise TypeError(
                "criteria_hierarchy_config did not contain Criterion objects "
                "and criteria_hierarchy.get_criterion is not importable. "
                "Pass the File 23 CRITERIA_TREE (or a list of Criterion)."
            )
    return [get_criterion(k) for k in leaf_keys]


def _unwrap_tree(criteria_hierarchy_config: Any) -> Any:
    if isinstance(criteria_hierarchy_config, dict):
        for key in ("tree", "hierarchy", "criteria_hierarchy_config", "CRITERIA_TREE"):
            if key in criteria_hierarchy_config:
                return criteria_hierarchy_config[key]
    return criteria_hierarchy_config


def _unwrap_constraints(
    criteria_hierarchy_config: Any,
    constraints_dict: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if constraints_dict:
        return constraints_dict
    if isinstance(criteria_hierarchy_config, dict):
        bundled = criteria_hierarchy_config.get("constraints") or criteria_hierarchy_config.get(
            "constraints_dict"
        )
        if bundled:
            return bundled
    return {}


# =============================================================================
# Version IDs
# =============================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _make_version_id(matrix: pd.DataFrame, data_release_at: str) -> str:
    """Stable-ish id: release timestamp + content hash of the matrix."""
    payload = matrix.to_csv(index=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    stamp = data_release_at.replace(":", "").replace("-", "")
    return f"{stamp}-{digest}"


# =============================================================================
# Core: expensive offline build
# =============================================================================

def build_normalized_matrix(
    sites_df: pd.DataFrame,
    criteria_hierarchy_config: Any,
    *,
    constraints_dict: Optional[Dict[str, Any]] = None,
    id_column: str = DEFAULT_ID_COLUMN,
    data_release_at: Optional[str] = None,
    persist: bool = False,
    supabase_client: Any = None,
) -> NormalizedMatrixArtifact:
    """
    Filter, normalize, and package a weight-independent decision matrix.

    Parameters
    ----------
    sites_df : DataFrame
        One row per candidate site, raw criterion columns plus any constraint
        fields (unit size, rent, tenure, …).
    criteria_hierarchy_config : File 23 tree, dict tree, or dict wrapping
        ``{"tree": ..., "constraints": {...}}``.
    constraints_dict : optional
        Hard constraints for File 26. If omitted, constraints bundled on
        ``criteria_hierarchy_config`` are used; if those are also absent,
        no sites are filtered.
    id_column : str
        Site identity column forwarded to ``apply_hard_constraints``.
    data_release_at : optional ISO-8601 UTC timestamp for this snapshot.
        Generated now if omitted.
    persist : bool
        If True, immediately version the artifact into Supabase.
    supabase_client : optional
        Pre-built client; otherwise env vars are used when ``persist=True``.

    Returns
    -------
    NormalizedMatrixArtifact
        ``.matrix`` is the full normalized [0, 1] DataFrame (same index as
        the *filtered* sites). ``.parent_to_children`` / ``.leaf_keys`` are
        the File 25 structures File 32 needs to accept user weights later.
    """
    if sites_df is None or not isinstance(sites_df, pd.DataFrame):
        raise TypeError("sites_df must be a pandas DataFrame.")

    tree = _unwrap_tree(criteria_hierarchy_config)
    constraints = _unwrap_constraints(criteria_hierarchy_config, constraints_dict)

    n_in = len(sites_df)
    filtered_df, excluded_sites = apply_hard_constraints(
        sites_df,
        constraints,
        id_column=id_column,
    )

    leaf_criteria = _collect_leaf_criteria(tree)
    normalized = normalize_all(filtered_df, leaf_criteria)

    parent_to_children, leaf_keys, _template = prepare_weight_structure(tree)

    # Keep matrix columns in hierarchy leaf order where those columns exist.
    ordered_cols = [k for k in leaf_keys if k in normalized.columns]
    extra_cols = [c for c in normalized.columns if c not in ordered_cols]
    if ordered_cols or extra_cols:
        normalized = normalized.loc[:, ordered_cols + extra_cols]

    released = data_release_at or _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    version_id = _make_version_id(normalized, released)
    storage_path = f"{version_id}/normalized_matrix.parquet"

    artifact = NormalizedMatrixArtifact(
        matrix=normalized,
        version_id=version_id,
        data_release_at=released,
        parent_to_children=parent_to_children,
        leaf_keys=leaf_keys,
        scored_criterion_keys=list(normalized.columns),
        excluded_sites=excluded_sites,
        storage_path=storage_path,
        n_sites_in=n_in,
        n_sites_out=len(filtered_df),
    )

    if persist:
        persist_normalized_matrix(artifact, supabase_client=supabase_client)

    return artifact


# =============================================================================
# Supabase versioning
# =============================================================================

def _get_supabase_client(client: Any = None) -> Any:
    if client is not None:
        return client
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "The 'supabase' package is required to version the matrix. "
            "Install with: pip install supabase"
        ) from exc

    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) "
            "must be set to persist a normalized-matrix version."
        )
    return create_client(url, key)


def _matrix_bytes(matrix: pd.DataFrame) -> Tuple[bytes, str]:
    """Prefer parquet; fall back to CSV if pyarrow/fastparquet is absent."""
    buffer = io.BytesIO()
    try:
        matrix.to_parquet(buffer, index=True)
        return buffer.getvalue(), "application/octet-stream"
    except Exception:
        csv_bytes = matrix.to_csv(index=True).encode("utf-8")
        return csv_bytes, "text/csv"


def persist_normalized_matrix(
    artifact: NormalizedMatrixArtifact,
    *,
    supabase_client: Any = None,
    bucket: str = MATRIX_STORAGE_BUCKET,
    table: str = MATRIX_VERSIONS_TABLE,
) -> NormalizedMatrixArtifact:
    """
    Write the matrix blob to Supabase Storage and a version row to Postgres.

    Storage object:
        {bucket}/{version_id}/normalized_matrix.parquet  (or .csv)

    Table row (``normalized_matrix_versions``) is tagged with
    ``version_id`` and ``data_release_at`` so File 32 / any saved score
    can cite the snapshot it was computed from.

    Expected table (create once in a migration)::

        create table if not exists normalized_matrix_versions (
            version_id text primary key,
            data_release_at timestamptz not null,
            storage_path text not null,
            n_sites_in int,
            n_sites_out int,
            n_excluded int,
            scored_criterion_keys jsonb,
            leaf_keys jsonb,
            parent_to_children jsonb,
            excluded_sites jsonb,
            metadata jsonb,
            created_at timestamptz default now()
        );
    """
    client = _get_supabase_client(supabase_client)
    body, content_type = _matrix_bytes(artifact.matrix)

    ext = "parquet" if content_type == "application/octet-stream" else "csv"
    storage_path = f"{artifact.version_id}/normalized_matrix.{ext}"
    artifact.storage_path = storage_path

    meta_path = f"{artifact.version_id}/metadata.json"
    meta_bytes = json.dumps(artifact.to_metadata(), default=str).encode("utf-8")

    storage = client.storage.from_(bucket)
    _storage_upsert(storage, storage_path, body, content_type)
    _storage_upsert(storage, meta_path, meta_bytes, "application/json")

    row = {
        "version_id": artifact.version_id,
        "data_release_at": artifact.data_release_at,
        "storage_path": storage_path,
        "n_sites_in": artifact.n_sites_in,
        "n_sites_out": artifact.n_sites_out,
        "n_excluded": len(artifact.excluded_sites),
        "scored_criterion_keys": artifact.scored_criterion_keys,
        "leaf_keys": artifact.leaf_keys,
        "parent_to_children": artifact.parent_to_children,
        "excluded_sites": artifact.excluded_sites,
        "metadata": artifact.to_metadata(),
    }
    client.table(table).upsert(row, on_conflict="version_id").execute()
    artifact.persisted = True
    return artifact


def _storage_upsert(storage: Any, path: str, body: bytes, content_type: str) -> None:
    file_options = {"content-type": content_type, "upsert": "true"}
    try:
        storage.upload(path, body, file_options=file_options)
    except Exception:
        # Some client versions reject overwrite on upload; replace instead.
        storage.update(path, body, file_options=file_options)


def load_normalized_matrix(
    version_id: str,
    *,
    supabase_client: Any = None,
    bucket: str = MATRIX_STORAGE_BUCKET,
    table: str = MATRIX_VERSIONS_TABLE,
) -> NormalizedMatrixArtifact:
    """
    Reload a previously versioned snapshot for File 32.

    Looks up the version row, downloads the Storage object, and rebuilds
    the artifact (still weight-independent).
    """
    client = _get_supabase_client(supabase_client)
    resp = (
        client.table(table)
        .select("*")
        .eq("version_id", version_id)
        .limit(1)
        .execute()
    )
    rows: Sequence[Dict[str, Any]] = getattr(resp, "data", None) or []
    if not rows:
        raise KeyError(f"No normalized matrix version '{version_id}' in {table}.")
    row = rows[0]
    storage_path = row["storage_path"]
    blob = client.storage.from_(bucket).download(storage_path)

    buffer = io.BytesIO(blob)
    if storage_path.endswith(".csv"):
        matrix = pd.read_csv(buffer, index_col=0)
    else:
        matrix = pd.read_parquet(buffer)

    return NormalizedMatrixArtifact(
        matrix=matrix,
        version_id=row["version_id"],
        data_release_at=str(row["data_release_at"]),
        parent_to_children=row.get("parent_to_children") or {},
        leaf_keys=list(row.get("leaf_keys") or []),
        scored_criterion_keys=list(row.get("scored_criterion_keys") or list(matrix.columns)),
        excluded_sites=list(row.get("excluded_sites") or []),
        storage_path=storage_path,
        n_sites_in=int(row.get("n_sites_in") or 0),
        n_sites_out=int(row.get("n_sites_out") or len(matrix)),
        persisted=True,
    )


__all__ = [
    "NormalizedMatrixArtifact",
    "build_normalized_matrix",
    "prepare_weight_structure",
    "persist_normalized_matrix",
    "load_normalized_matrix",
    "MATRIX_VERSIONS_TABLE",
    "MATRIX_STORAGE_BUCKET",
]