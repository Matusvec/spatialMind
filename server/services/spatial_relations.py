"""Spatial relationship extraction and CLIP hierarchy computation.

Derives on_top_of and next_to edges from cluster centroids and bounding
boxes, using scene-relative thresholds. Builds emergent hierarchy from
CLIP embedding cosine similarity between clusters.
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def _compute_scene_scale(clusters: list[dict]) -> float:
    """Compute scene scale from the maximum extent of all cluster centroids.

    Args:
        clusters: List of cluster dicts with 'centroid' key.

    Returns:
        Positive float representing the max extent across all axes.
        Returns 1.0 as minimum to avoid division by zero.
    """
    if len(clusters) < 2:
        return 1.0

    centroids = np.array([c["centroid"] for c in clusters])
    extents = centroids.max(axis=0) - centroids.min(axis=0)
    scale = float(np.max(extents))
    return max(scale, 1e-6)


def _bbox_y_overlap(bbox_a: list, bbox_b: list) -> float:
    """Compute Y-axis overlap fraction between two bounding boxes.

    Args:
        bbox_a: [x_min, y_min, z_min, x_max, y_max, z_max] for cluster A.
        bbox_b: Same format for cluster B.

    Returns:
        Overlap fraction in [0, 1]. 0 means no overlap, 1 means full overlap.
    """
    a_y_min, a_y_max = bbox_a[1], bbox_a[4]
    b_y_min, b_y_max = bbox_b[1], bbox_b[4]

    overlap_min = max(a_y_min, b_y_min)
    overlap_max = min(a_y_max, b_y_max)
    overlap = max(0.0, overlap_max - overlap_min)

    # Normalize by the smaller bbox extent
    extent_a = a_y_max - a_y_min
    extent_b = b_y_max - b_y_min
    min_extent = min(extent_a, extent_b)

    if min_extent <= 0:
        return 0.0

    return min(overlap / min_extent, 1.0)


def extract_spatial_relations(clusters: list[dict]) -> list[dict]:
    """Extract on_top_of and next_to spatial relationships between clusters.

    Uses scene-relative thresholds so it works for any coordinate space.

    Args:
        clusters: List of cluster dicts from cluster_gaussians(). Each must
            have 'label', 'centroid' (3-list), and 'bbox' (6-list).

    Returns:
        List of edge dicts sorted by confidence descending. Each dict:
            from_label: int source cluster label
            to_label: int target cluster label
            relation: "on_top_of" or "next_to"
            confidence: float in (0, 1]
            distance: float Euclidean distance between centroids
    """
    if len(clusters) < 2:
        return []

    scene_scale = _compute_scene_scale(clusters)
    y_threshold = 0.3 * scene_scale
    xz_proximity_threshold = 0.5 * scene_scale

    edges = []
    seen_next_to = set()

    for i, ci in enumerate(clusters):
        for j, cj in enumerate(clusters):
            if i == j:
                continue

            cx_i, cy_i, cz_i = ci["centroid"]
            cx_j, cy_j, cz_j = cj["centroid"]

            y_offset = cy_i - cy_j
            xz_distance = math.sqrt((cx_i - cx_j) ** 2 + (cz_i - cz_j) ** 2)
            euclidean_dist = math.sqrt(
                (cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2 + (cz_i - cz_j) ** 2
            )

            # on_top_of: i is above j
            if y_offset > y_threshold and xz_distance < xz_proximity_threshold:
                overlap_ratio = _bbox_y_overlap(ci["bbox"], cj["bbox"])
                confidence = max(1.0 - overlap_ratio, 0.1)
                edges.append(
                    {
                        "from_label": ci["label"],
                        "to_label": cj["label"],
                        "relation": "on_top_of",
                        "confidence": float(confidence),
                        "distance": float(euclidean_dist),
                    }
                )

            # next_to: close in XZ, roughly same height
            if (
                xz_distance < xz_proximity_threshold
                and abs(y_offset) < y_threshold
            ):
                # Deduplicate symmetric pairs
                pair = (
                    min(ci["label"], cj["label"]),
                    max(ci["label"], cj["label"]),
                )
                if pair not in seen_next_to:
                    seen_next_to.add(pair)
                    confidence = max(
                        1.0 - (xz_distance / xz_proximity_threshold), 0.1
                    )
                    edges.append(
                        {
                            "from_label": pair[0],
                            "to_label": pair[1],
                            "relation": "next_to",
                            "confidence": float(confidence),
                            "distance": float(euclidean_dist),
                        }
                    )

    # Sort by confidence descending
    edges.sort(key=lambda e: e["confidence"], reverse=True)

    logger.info(
        "Extracted %d spatial edges (scene_scale=%.3f)", len(edges), scene_scale
    )
    return edges


def build_hierarchy(
    clusters: list[dict], similarity_threshold: float = 0.7
) -> list[dict]:
    """Build emergent hierarchy from CLIP embedding cosine similarity.

    Groups clusters whose mean CLIP embeddings are above the similarity
    threshold, creating "similar_to" edges for the knowledge graph.

    Args:
        clusters: List of cluster dicts with 'mean_embedding' (512-dim ndarray).
        similarity_threshold: Minimum cosine similarity for a hierarchy edge.

    Returns:
        List of hierarchy edge dicts sorted by similarity descending. Each dict:
            from_label: int source cluster label
            to_label: int target cluster label
            relation: "similar_to"
            similarity: float cosine similarity score
    """
    if len(clusters) < 2:
        return []

    edges = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            emb_i = clusters[i]["mean_embedding"]
            emb_j = clusters[j]["mean_embedding"]
            similarity = float(np.dot(emb_i, emb_j))

            if similarity > similarity_threshold:
                edges.append(
                    {
                        "from_label": clusters[i]["label"],
                        "to_label": clusters[j]["label"],
                        "relation": "similar_to",
                        "similarity": similarity,
                    }
                )

    edges.sort(key=lambda e: e["similarity"], reverse=True)

    logger.info(
        "Built %d hierarchy edges (threshold=%.2f)",
        len(edges),
        similarity_threshold,
    )
    return edges
