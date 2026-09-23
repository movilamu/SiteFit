# scoring/precision_bands.py
"""
Convert raw numeric site scores into coarse presentation bands.

This module sits downstream of File 32's ``rank_sites`` and File 40's
``identify_near_ties``. Its purpose is to stop the front end from ever
displaying scores to 2-3 decimal places (spurious precision the original
spec explicitly warns against) and instead show a small set of qualitative
bands such as "Excellent / Strong / Good / Marginal".

Band edges are computed from the *cluster representative scores* produced
by File 40, not from raw per-site scores. That guarantees the one property
that actually matters here: two sites File 40 says are statistically
indistinguishable can never be split across two different bands. A band
boundary is only ever placed between clusters, never inside one.

Example
-------
>>> ranked = rank_sites(scores)
>>> clusters = identify_near_ties(ranked)
>>> assign_score_bands(ranked, clusters)
  rank site_id  score  cluster_rank      band
0    1       A  0.812             1  Excellent
1    2       B  0.809             1  Excellent
2    3       C  0.781             3    Strong
...
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_BAND_LABELS = ("Excellent", "Strong", "Good", "Marginal")


def assign_score_bands(
    ranked_scores,
    near_tie_groups,
    band_labels: Sequence[str] = DEFAULT_BAND_LABELS,
):
    """
    Assign presentation bands to sites, respecting near-tie clusters.

    Parameters
    ----------
    ranked_scores : pandas.DataFrame
        Output from File 32's ``rank_sites()``. Must contain columns
        ``rank``, ``site_id``, ``score``, sorted best-first.

    near_tie_groups : list of dict
        Output from File 40's ``identify_near_ties()``. Each element must
        contain ``cluster_rank``, ``site_ids``, ``scores``. Clusters must
        be given best-first (matching ``ranked_scores`` order) and must
        partition all site ids present in ``ranked_scores``.

    band_labels : sequence of str, default=("Excellent", "Strong", "Good",
        "Marginal")
        Labels to use for the bands, best-first. The number of labels
        determines the (maximum) number of bands. If there are fewer
        clusters than labels, only the first N labels are used, one per
        cluster, so no band is ever left empty.

    Returns
    -------
    pandas.DataFrame
        Columns:
            rank : int
                Rank from ``ranked_scores``.
            site_id
                Site identifier.
            score : float
                Raw underlying score (kept for internal/debug use; the
                front end should display ``band`` instead of this value).
            cluster_rank : int
                The ``cluster_rank`` of the near-tie cluster this site
                belongs to, from File 40.
            band : str
                Presentation band. Sites in the same cluster always get
                the same band.

        Row order matches ``ranked_scores`` (best-first).
    """
    if not isinstance(ranked_scores, pd.DataFrame):
        raise TypeError("ranked_scores must be a pandas DataFrame.")

    required = {"rank", "site_id", "score"}
    missing = required - set(ranked_scores.columns)
    if missing:
        raise ValueError(
            f"ranked_scores missing required columns: {sorted(missing)}"
        )

    if not near_tie_groups:
        raise ValueError("near_tie_groups cannot be empty.")

    if len(band_labels) == 0:
        raise ValueError("band_labels cannot be empty.")

    ranked = ranked_scores.sort_values("rank").reset_index(drop=True)

    # --- Validate that clusters partition ranked_scores' site ids ----
    site_to_cluster_rank = {}
    cluster_rep_score = {}  # cluster_rank -> representative score (max within cluster)
    cluster_order = []      # cluster_rank values in the order clusters were given

    for cluster in near_tie_groups:
        for key in ("cluster_rank", "site_ids", "scores"):
            if key not in cluster:
                raise ValueError(
                    f"near_tie_groups entries must contain {key!r}."
                )

        c_rank = cluster["cluster_rank"]
        cluster_order.append(c_rank)

        if not cluster["site_ids"]:
            raise ValueError(
                f"Cluster with cluster_rank={c_rank!r} has no site_ids."
            )

        for site_id in cluster["site_ids"]:
            if site_id in site_to_cluster_rank:
                raise ValueError(
                    f"site_id {site_id!r} appears in more than one cluster; "
                    "near_tie_groups must partition the sites."
                )
            site_to_cluster_rank[site_id] = c_rank

        # Use the best (max) score in the cluster as its representative for
        # placing band edges. Ties are grouped, so within-cluster spread is
        # already known to be small (bounded by File 40's threshold).
        cluster_rep_score[c_rank] = float(max(cluster["scores"]))

    ranked_site_ids = set(ranked["site_id"])
    clustered_site_ids = set(site_to_cluster_rank)
    if ranked_site_ids != clustered_site_ids:
        missing_from_clusters = ranked_site_ids - clustered_site_ids
        extra_in_clusters = clustered_site_ids - ranked_site_ids
        details = []
        if missing_from_clusters:
            details.append(
                f"in ranked_scores but not in near_tie_groups: "
                f"{sorted(map(str, missing_from_clusters))}"
            )
        if extra_in_clusters:
            details.append(
                f"in near_tie_groups but not in ranked_scores: "
                f"{sorted(map(str, extra_in_clusters))}"
            )
        raise ValueError(
            "ranked_scores and near_tie_groups must cover the same sites. "
            + "; ".join(details)
        )

    # --- Decide how many bands we actually need -----------------------
    n_clusters = len(cluster_order)
    n_bands = min(len(band_labels), n_clusters)

    # --- Distribute clusters across bands, best-first ------------------
    # Clusters are already ordered best-first (same order as ranked_scores,
    # by construction of File 40's identify_near_ties). We split that
    # ordered list of clusters into n_bands contiguous groups, as evenly
    # as possible, using the representative scores only to decide *how
    # many* clusters land in each band when cluster sizes are uneven --
    # in practice this just means: give every band roughly the same
    # number of *sites* rather than the same number of *clusters*.
    cluster_sizes = [
        len(next(c for c in near_tie_groups if c["cluster_rank"] == cr)["site_ids"])
        for cr in cluster_order
    ]
    total_sites = sum(cluster_sizes)

    band_assignment = _split_clusters_into_bands(
        cluster_order, cluster_sizes, total_sites, n_bands
    )

    labels_to_use = list(band_labels[:n_bands])
    cluster_rank_to_band = {
        c_rank: labels_to_use[band_idx]
        for band_idx, cluster_group in enumerate(band_assignment)
        for c_rank in cluster_group
    }

    ranked = ranked.copy()
    ranked["cluster_rank"] = ranked["site_id"].map(site_to_cluster_rank)
    ranked["band"] = ranked["cluster_rank"].map(cluster_rank_to_band)

    return ranked[["rank", "site_id", "score", "cluster_rank", "band"]]


def _split_clusters_into_bands(
    cluster_order: Sequence,
    cluster_sizes: Sequence[int],
    total_sites: int,
    n_bands: int,
):
    """
    Split an ordered (best-first) list of clusters into ``n_bands``
    contiguous groups, aiming for roughly equal total *site* counts per
    band without ever breaking a cluster apart.

    Returns
    -------
    list of list
        ``n_bands`` groups of cluster_rank values, best-first, covering
        every cluster in ``cluster_order`` exactly once.
    """
    n_clusters = len(cluster_order)
    if n_bands <= 1:
        return [list(cluster_order)]

    target_per_band = total_sites / n_bands

    bands = []
    current_group = []
    current_count = 0
    remaining_bands = n_bands

    for i, (c_rank, size) in enumerate(zip(cluster_order, cluster_sizes)):
        remaining_clusters = n_clusters - i
        current_group.append(c_rank)
        current_count += size

        # Never leave fewer clusters than remaining bands need, and never
        # close the final band early.
        clusters_left_after_this = remaining_clusters - 1
        bands_left_after_this = remaining_bands - 1

        must_close = (
            bands_left_after_this > 0
            and clusters_left_after_this >= bands_left_after_this
            and current_count >= target_per_band
        )

        if must_close and remaining_bands > 1:
            bands.append(current_group)
            current_group = []
            current_count = 0
            remaining_bands -= 1

    bands.append(current_group)  # last (possibly only) band gets the rest

    # Safety net: if rounding left an empty band or extra bands, collapse.
    bands = [b for b in bands if b]
    while len(bands) > n_bands:
        bands[-2].extend(bands[-1])
        bands.pop()

    return bands


__all__ = ["assign_score_bands", "DEFAULT_BAND_LABELS"]
