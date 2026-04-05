"""Primary dense query-time grounding pipeline.

Dense semantic retrieval is the primary path. Query-time clusters become
highlightable regions immediately and can optionally be promoted into cached
instances / an instance graph after grounding succeeds.
"""

from __future__ import annotations

import logging

import numpy as np

from server.services.instance_cache import (
    build_instance_graph,
    ensure_instance_store,
    upsert_query_nodes,
)
from server.services.query_node_builder import build_query_nodes

logger = logging.getLogger(__name__)


def build_highlight_regions(
    nodes: list[dict],
    positions,
    *,
    max_regions: int = 16,
) -> list[dict]:
    """Build tight spherical highlight regions from grounded nodes."""
    if positions is None:
        return []

    regions = []
    for node in nodes:
        indices = np.asarray(node.get("gaussian_indices", []), dtype=np.int64)
        if len(indices) == 0:
            continue
        valid = indices[(indices >= 0) & (indices < len(positions))]
        if len(valid) == 0:
            continue
        cluster_pos = positions[valid]
        centroid = cluster_pos.mean(axis=0)
        dists = np.linalg.norm(cluster_pos - centroid, axis=1)
        # Use 95th percentile distance + padding for radius — cover the full cluster
        fallback_radius = float(np.percentile(dists, 95)) + 0.1 if len(dists) > 0 else 0.3
        radius = max(fallback_radius, float(node.get("radius", 0)))
        regions.append(
            {
                "node_id": node.get("id", ""),
                "label": node.get("label", "region"),
                "centroid": centroid.tolist(),
                "radius": max(0.1, min(radius, 3.0)),
                "count": int(len(valid)),
                "mean_score": float(node.get("match_score", node.get("confidence", 0.0))),
                "semantic_level": int(node.get("semantic_level", 0)),
            }
        )
    regions.sort(key=lambda item: -item["mean_score"])
    return regions[:max_regions]


def ground_query(
    *,
    text: str,
    state: dict,
    persist_instances: bool = True,
    progress_cb=None,
    preferred_levels: tuple[int, ...] = (0, 1, 2),
) -> dict:
    """Ground a free-form text query directly against the dense 3D feature field."""
    gaussian_store = state.get("gaussian_store")
    clip_encoder = state.get("clip_encoder")
    negative_embeddings = state.get("negative_embeddings")

    query_nodes, match_data = build_query_nodes(
        text=text,
        gaussian_store=gaussian_store,
        clip_encoder=clip_encoder,
        negative_embeddings=negative_embeddings,
        progress_cb=progress_cb,
        preferred_levels=preferred_levels,
        return_match_data=True,
    )

    if not query_nodes:
        if progress_cb is not None:
            progress_cb("prepare_response", "active")
            progress_cb("prepare_response", "completed", detail="no grounded regions")
        return {
            "query": text,
            "nodes": [],
            "highlight_regions": [],
            "highlight_indices": [],
            "answer": f"No grounded region matched '{text}'.",
            "reasoning": "Dense sem2 retrieval found no clusters above threshold.",
            "semantic_levels": [],
            "semantic_fallback": False,
            "highlight_match": {"indices": [], "scores": [], "level": 2},
        }

    materialized_nodes = query_nodes
    if persist_instances:
        if progress_cb is not None:
            progress_cb("materialize_instances", "active")
        instance_store = ensure_instance_store(state)
        persisted = upsert_query_nodes(
            instance_store,
            query_nodes,
            gaussian_store,
            query_text=text,
        )
        if persisted:
            materialized_nodes = persisted
        if progress_cb is not None:
            progress_cb(
                "materialize_instances",
                "completed",
                detail=f"{len(materialized_nodes)} reusable instance(s)",
            )

        if progress_cb is not None:
            progress_cb("build_graph", "active")
        instance_graph = build_instance_graph(instance_store, gaussian_store)
        state["instance_graph"] = instance_graph
        # Backward-compatible alias: graph now means grounded instances, not startup labels.
        state["scene_graph"] = instance_graph
        if progress_cb is not None:
            progress_cb(
                "build_graph",
                "completed",
                detail=f"{len(instance_graph.get('edges', []))} edge(s)",
            )

    if progress_cb is not None:
        progress_cb("prepare_response", "active")
    # Use query_nodes (pre-materialization) for highlights — they always
    # have fresh gaussian_indices from the current DBSCAN clustering.
    # materialized_nodes may have empty indices on repeat queries.
    highlight_regions = build_highlight_regions(
        query_nodes,
        getattr(gaussian_store, "positions", None),
    )
    clustered_indices = sorted(
        {
            int(idx)
            for node in query_nodes
            for idx in node.get("gaussian_indices", [])
        }
    )
    highlight_indices = clustered_indices
    semantic_levels = sorted(
        {int(node.get("semantic_level", 0)) for node in query_nodes}
    )

    labels = [node.get("label", "region") for node in materialized_nodes]
    answer = (
        f"Found {len(highlight_regions)} grounded region(s) for '{text}': "
        + ", ".join(labels[:6])
    )

    reasoning = (
        f"Dense sem2 retrieval produced {len(query_nodes)} query-time cluster(s); "
        f"{len(materialized_nodes)} instance(s) were materialized and cached."
        if persist_instances
        else f"Dense sem2 retrieval produced {len(query_nodes)} query-time cluster(s)."
    )
    if progress_cb is not None:
        progress_cb(
            "prepare_response",
            "completed",
            detail=f"{len(highlight_regions)} highlight region(s)",
        )

    return {
        "query": text,
        "nodes": materialized_nodes,
        "highlight_regions": highlight_regions,
        "highlight_indices": highlight_indices,
        "highlight_match": {
            "indices": clustered_indices,
            "scores": [1.0] * len(clustered_indices),
            "level": 2,
            "used_threshold": match_data.get("used_threshold"),
        },
        "answer": answer,
        "reasoning": reasoning,
        "semantic_levels": semantic_levels,
        "semantic_fallback": False,
    }
