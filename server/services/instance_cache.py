"""Persistent instance cache built from dense query-time grounding.

This module treats query-grounded clusters as the primary source of objects.
Instances are matched and merged by spatial overlap / Gaussian overlap, then
optionally lifted into a lightweight scene graph for downstream UX.
"""

from __future__ import annotations

import logging

import numpy as np

from server.services.scene_matching import tokenize
from server.services.spatial_relations import build_hierarchy, extract_spatial_relations

logger = logging.getLogger(__name__)


def empty_instance_store() -> dict:
    """Create an empty instance store."""
    return {
        "instances": [],
        "next_id": 0,
        "query_history": [],
    }


def ensure_instance_store(state: dict) -> dict:
    """Ensure app state has an instance store."""
    store = state.get("instance_store")
    if store is None:
        store = empty_instance_store()
        state["instance_store"] = store
    return store


def _focus_label(text: str) -> str:
    tokens = tokenize(text)
    if not tokens:
        return text.strip().lower() or "region"
    return " ".join(tokens)


def _bbox_iou_3d(a: list[float], b: list[float]) -> float:
    ax1, ay1, az1, ax2, ay2, az2 = a
    bx1, by1, bz1, bx2, by2, bz2 = b

    ix1, iy1, iz1 = max(ax1, bx1), max(ay1, by1), max(az1, bz1)
    ix2, iy2, iz2 = min(ax2, bx2), min(ay2, by2), min(az2, bz2)

    inter_dx = max(0.0, ix2 - ix1)
    inter_dy = max(0.0, iy2 - iy1)
    inter_dz = max(0.0, iz2 - iz1)
    inter = inter_dx * inter_dy * inter_dz
    if inter <= 0:
        return 0.0

    vol_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) * max(0.0, az2 - az1)
    vol_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1) * max(0.0, bz2 - bz1)
    union = vol_a + vol_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _gaussian_jaccard(a: list[int], b: list[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    if inter == 0:
        return 0.0
    union = len(set_a | set_b)
    return inter / union


def _distance_score(a: dict, b: dict) -> float:
    ca = np.asarray(a.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32)
    cb = np.asarray(b.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32)
    dist = float(np.linalg.norm(ca - cb))
    radius_a = float(a.get("radius", 0.2))
    radius_b = float(b.get("radius", 0.2))
    scale = max(radius_a + radius_b, 1e-4)
    return max(0.0, 1.0 - (dist / (1.5 * scale)))


def _instance_label(instance: dict) -> str:
    counts = instance.get("alias_counts", {})
    if counts:
        return max(counts.items(), key=lambda item: item[1])[0]
    aliases = instance.get("aliases", [])
    if aliases:
        return aliases[0]
    return instance.get("label", "region")


def _recompute_geometry(indices: list[int], gaussian_store) -> tuple[list[float], list[float], float]:
    if gaussian_store is None or getattr(gaussian_store, "positions", None) is None or not indices:
        return [0.0, 0.0, 0.0], [0.0] * 6, 0.15

    valid = np.asarray(indices, dtype=np.int64)
    valid = valid[(valid >= 0) & (valid < len(gaussian_store.positions))]
    if len(valid) == 0:
        return [0.0, 0.0, 0.0], [0.0] * 6, 0.15

    pos = gaussian_store.positions[valid]
    centroid = pos.mean(axis=0)
    bbox = [
        float(pos[:, 0].min()),
        float(pos[:, 1].min()),
        float(pos[:, 2].min()),
        float(pos[:, 0].max()),
        float(pos[:, 1].max()),
        float(pos[:, 2].max()),
    ]
    dists = np.linalg.norm(pos - centroid, axis=1)
    radius = float(np.percentile(dists, 75)) + 0.03 if len(dists) else 0.15
    return centroid.tolist(), bbox, max(0.08, min(radius, 0.8))


def _mean_embedding(indices: list[int], gaussian_store) -> np.ndarray | None:
    decoded = getattr(gaussian_store, "decoded_embeddings", None)
    if decoded is None or not indices:
        return None
    valid = np.asarray(indices, dtype=np.int64)
    valid = valid[(valid >= 0) & (valid < len(decoded))]
    if len(valid) == 0:
        return None
    emb = decoded[valid].mean(axis=0).astype(np.float32)
    norm = float(np.linalg.norm(emb))
    if norm <= 1e-8:
        return None
    return emb / norm


def _match_instance(query_node: dict, instances: list[dict]) -> dict | None:
    best = None
    best_score = 0.0
    for instance in instances:
        overlap_score = _gaussian_jaccard(
            query_node.get("gaussian_indices", []),
            instance.get("gaussian_indices", []),
        )
        bbox_score = _bbox_iou_3d(
            query_node.get("bbox", [0.0] * 6),
            instance.get("bbox", [0.0] * 6),
        )
        dist_score = _distance_score(query_node, instance)
        score = max(overlap_score, bbox_score, dist_score)
        if score > best_score:
            best = instance
            best_score = score
    if best_score >= 0.35:
        return best
    return None


def upsert_query_nodes(
    instance_store: dict,
    query_nodes: list[dict],
    gaussian_store,
    *,
    query_text: str,
) -> list[dict]:
    """Merge query-time nodes into the persistent instance cache."""
    if not query_nodes:
        return []

    persisted = []
    query_label = _focus_label(query_text)
    instance_store.setdefault("query_history", []).append(query_text)

    for query_node in query_nodes:
        match = _match_instance(query_node, instance_store["instances"])
        if match is None:
            instance_id = f"inst_{instance_store['next_id']:03d}"
            instance_store["next_id"] += 1
            aliases = [query_label]
            alias_counts = {query_label: 1}
            gaussian_indices = sorted(set(query_node.get("gaussian_indices", [])))
            centroid, bbox, radius = _recompute_geometry(gaussian_indices, gaussian_store)
            instance = {
                "id": instance_id,
                "label": query_label,
                "aliases": aliases,
                "alias_counts": alias_counts,
                "source_queries": [query_text],
                "gaussian_indices": gaussian_indices,
                "centroid": centroid,
                "bbox": bbox,
                "radius": radius,
                "confidence": float(query_node.get("confidence", 0.0)),
                "match_score": float(query_node.get("match_score", query_node.get("confidence", 0.0))),
                "semantic_level": int(query_node.get("semantic_level", 0)),
                "query_count": 1,
            }
            instance_store["instances"].append(instance)
            persisted.append(instance)
            continue

        merged_indices = sorted(
            set(match.get("gaussian_indices", [])) | set(query_node.get("gaussian_indices", []))
        )
        match["gaussian_indices"] = merged_indices
        match["source_queries"] = sorted(set(match.get("source_queries", [])) | {query_text})
        match["query_count"] = int(match.get("query_count", 0)) + 1
        alias_counts = dict(match.get("alias_counts", {}))
        alias_counts[query_label] = alias_counts.get(query_label, 0) + 1
        match["alias_counts"] = alias_counts
        match["aliases"] = sorted(alias_counts)
        match["label"] = _instance_label(match)
        centroid, bbox, radius = _recompute_geometry(merged_indices, gaussian_store)
        match["centroid"] = centroid
        match["bbox"] = bbox
        match["radius"] = radius
        match["confidence"] = max(
            float(match.get("confidence", 0.0)),
            float(query_node.get("confidence", 0.0)),
        )
        match["match_score"] = max(
            float(match.get("match_score", 0.0)),
            float(query_node.get("match_score", 0.0)),
        )
        match["semantic_level"] = min(
            int(match.get("semantic_level", 0)),
            int(query_node.get("semantic_level", 0)),
        )
        persisted.append(match)

    logger.info(
        "Instance cache: %d total instance(s) after query '%s'",
        len(instance_store["instances"]),
        query_text,
    )
    return persisted


def build_instance_graph(instance_store: dict, gaussian_store) -> dict:
    """Build a lightweight graph from cached instances."""
    instances = instance_store.get("instances", [])
    nodes: list[dict] = []
    clusters_for_edges: list[dict] = []

    for idx, instance in enumerate(instances):
        gaussian_indices = sorted(set(instance.get("gaussian_indices", [])))
        centroid, bbox, radius = _recompute_geometry(gaussian_indices, gaussian_store)
        mean_emb = _mean_embedding(gaussian_indices, gaussian_store)
        node = {
            "id": instance["id"],
            "label": _instance_label(instance),
            "centroid": centroid,
            "bbox": bbox,
            "gaussian_indices": gaussian_indices,
            "confidence": float(instance.get("confidence", 0.0)),
            "match_score": float(instance.get("match_score", instance.get("confidence", 0.0))),
            "radius": radius,
            "aliases": instance.get("aliases", []),
            "query_count": int(instance.get("query_count", 0)),
        }
        nodes.append(node)
        clusters_for_edges.append(
            {
                "label": idx,
                "centroid": centroid,
                "bbox": bbox,
                "mean_embedding": mean_emb if mean_emb is not None else np.zeros(512, dtype=np.float32),
            }
        )

    raw_edges = extract_spatial_relations(clusters_for_edges)
    raw_hierarchy = build_hierarchy(
        [c for c in clusters_for_edges if np.any(c["mean_embedding"])],
        similarity_threshold=0.72,
    )

    edges = []
    for edge in raw_edges:
        src = nodes[edge["from_label"]]["id"]
        dst = nodes[edge["to_label"]]["id"]
        edges.append(
            {
                "from": src,
                "to": dst,
                "relation": edge["relation"],
                "confidence": edge["confidence"],
                "distance": edge["distance"],
            }
        )

    hierarchy = []
    for edge in raw_hierarchy:
        src = nodes[edge["from_label"]]["id"]
        dst = nodes[edge["to_label"]]["id"]
        hierarchy.append(
            {
                "from": src,
                "to": dst,
                "relation": edge["relation"],
                "similarity": edge["similarity"],
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "hierarchy": hierarchy,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "hierarchy_count": len(hierarchy),
            "source": "instance-cache",
        },
    }
