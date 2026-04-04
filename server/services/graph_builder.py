"""Scene graph builder service.

Orchestrates CLIP-based Gaussian selection, DBSCAN clustering, spatial
relationship extraction, and hierarchy computation into a single scene
graph payload. Called once at scene load (not per-query).
"""

import logging

import numpy as np

from server.services.clustering import cluster_gaussians
from server.services.similarity import top_k
from server.services.spatial_relations import build_hierarchy, extract_spatial_relations

logger = logging.getLogger(__name__)


def build_scene_graph(
    gaussian_store,
    clip_encoder,
    query: str = "objects",
    k: int = 5000,
    min_samples: int = 6,
    hierarchy_threshold: float = 0.7,
) -> dict:
    """Build a complete scene graph from Gaussian data.

    Orchestration pipeline:
    1. Encode query text via CLIP
    2. Select top-K Gaussians by similarity
    3. Cluster selected Gaussians with DBSCAN
    4. Extract spatial relationships between clusters
    5. Build hierarchy from CLIP embedding similarity

    Args:
        gaussian_store: GaussianStore with loaded positions and decoded_embeddings.
        clip_encoder: CLIPEncoder with encode_text() method.
        query: Text query for Gaussian selection. Default "objects" captures everything.
        k: Maximum number of Gaussians to select for clustering.
        min_samples: DBSCAN min_samples parameter.
        hierarchy_threshold: Minimum cosine similarity for hierarchy edges.

    Returns:
        Dict with keys: nodes, edges, hierarchy, metadata.
    """
    # 1. Encode query via CLIP
    text_embedding = clip_encoder.encode_text(query)

    # 2. Select top-K Gaussians by similarity
    indices, scores = top_k(
        text_embedding, gaussian_store.decoded_embeddings, k=k
    )

    # 3. Extract positions and embeddings for selected Gaussians
    selected_positions = gaussian_store.positions[indices]
    selected_embeddings = gaussian_store.decoded_embeddings[indices]

    # 4. Cluster selected Gaussians
    clusters = cluster_gaussians(
        selected_positions, selected_embeddings, scores, indices, min_samples
    )

    # 5. Extract spatial relationships
    spatial_edges = extract_spatial_relations(clusters)

    # 6. Build hierarchy from CLIP similarity
    hierarchy_edges = build_hierarchy(clusters, hierarchy_threshold)

    # 7. Build response dict
    result = {
        "nodes": [
            {
                "id": f"obj_{i:03d}",
                "label": f"object_{i}",
                "centroid": cluster["centroid"],
                "bbox": cluster["bbox"],
                "gaussian_indices": cluster["gaussian_indices"],
                "confidence": cluster["confidence"],
            }
            for i, cluster in enumerate(clusters)
        ],
        "edges": [
            {
                "from": f"obj_{edge['from_label']:03d}",
                "to": f"obj_{edge['to_label']:03d}",
                "relation": edge["relation"],
                "confidence": edge["confidence"],
                "distance": edge["distance"],
            }
            for edge in spatial_edges
        ],
        "hierarchy": [
            {
                "from": f"obj_{h['from_label']:03d}",
                "to": f"obj_{h['to_label']:03d}",
                "relation": h["relation"],
                "similarity": h["similarity"],
            }
            for h in hierarchy_edges
        ],
        "metadata": {
            "gaussian_count": gaussian_store.count,
            "selected_count": len(indices),
            "cluster_count": len(clusters),
            "edge_count": len(spatial_edges),
            "hierarchy_count": len(hierarchy_edges),
            "query": query,
            "k": k,
        },
    }

    logger.info(
        "Scene graph built: %d clusters, %d edges, %d hierarchy edges",
        len(clusters),
        len(spatial_edges),
        len(hierarchy_edges),
    )

    return result
