"""DBSCAN clustering service with dynamic epsilon.

Groups semantically similar Gaussians into object-level clusters using
DBSCAN with a data-driven epsilon computed from the k-distance graph.
Each cluster includes centroid, bounding box, member indices, confidence,
and mean CLIP embedding.
"""

import logging

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def compute_dynamic_epsilon(
    positions: np.ndarray, min_samples: int = 6
) -> float:
    """Compute DBSCAN epsilon from the k-distance graph elbow.

    Builds a KDTree, queries k-th nearest neighbor distance for each point,
    sorts ascending, and finds the elbow via maximum second derivative.
    Falls back to median k-distance if elbow detection fails.

    Args:
        positions: (M, 3) float32 array of 3D positions.
        min_samples: DBSCAN min_samples parameter; also used as k for
            the k-distance graph.

    Returns:
        Positive float epsilon value for DBSCAN.
    """
    k = min(min_samples, len(positions) - 1)
    if k < 1:
        return 1.0

    tree = KDTree(positions)
    # Query k+1 neighbors (includes self at distance 0)
    distances, _ = tree.query(positions, k=k + 1)
    # k-th neighbor distance is at index k (0-indexed, index 0 is self)
    k_distances = np.sort(distances[:, k])

    # Find elbow using maximum curvature (second derivative)
    if len(k_distances) > 2:
        second_deriv = np.diff(k_distances, n=2)
        if len(second_deriv) > 0 and np.any(second_deriv != 0):
            elbow_idx = int(np.argmax(np.abs(second_deriv)))
            epsilon = float(k_distances[elbow_idx])
            if epsilon > 0:
                return epsilon

    # Fallback: median k-distance
    epsilon = float(np.median(k_distances))
    return epsilon if epsilon > 0 else 1.0


def cluster_gaussians(
    positions: np.ndarray,
    embeddings: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    min_samples: int = 6,
) -> list[dict]:
    """Cluster Gaussian positions using DBSCAN with dynamic epsilon.

    Args:
        positions: (M, 3) float32 -- XYZ of the M selected Gaussians.
        embeddings: (M, 512) float32 -- CLIP embeddings of selected Gaussians.
        scores: (M,) float32 -- similarity scores from top_k.
        indices: (M,) int -- original Gaussian indices (into the full store).
        min_samples: DBSCAN min_samples parameter.

    Returns:
        List of cluster dicts sorted by confidence descending. Each dict:
            label: int cluster ID
            centroid: [x, y, z] list
            bbox: [x_min, y_min, z_min, x_max, y_max, z_max] list
            gaussian_indices: list of original Gaussian indices
            confidence: mean similarity score of cluster members
            mean_embedding: (512,) L2-normalized ndarray
    """
    epsilon = compute_dynamic_epsilon(positions, min_samples)
    logger.info("Dynamic epsilon: %.6f", epsilon)

    db = DBSCAN(
        eps=epsilon, min_samples=min_samples, algorithm="ball_tree"
    ).fit(positions)

    labels = db.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)

    noise_count = int(np.sum(labels == -1))
    logger.info(
        "Clusters found: %d, noise points: %d, epsilon: %.6f",
        len(unique_labels),
        noise_count,
        epsilon,
    )

    clusters = []
    for label in sorted(unique_labels):
        mask = labels == label
        cluster_positions = positions[mask]
        cluster_embeddings = embeddings[mask]
        cluster_scores = scores[mask]
        cluster_indices = indices[mask]

        centroid = cluster_positions.mean(axis=0).tolist()

        bbox = [
            float(cluster_positions[:, 0].min()),
            float(cluster_positions[:, 1].min()),
            float(cluster_positions[:, 2].min()),
            float(cluster_positions[:, 0].max()),
            float(cluster_positions[:, 1].max()),
            float(cluster_positions[:, 2].max()),
        ]

        mean_emb = cluster_embeddings.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        clusters.append(
            {
                "label": int(label),
                "centroid": centroid,
                "bbox": bbox,
                "gaussian_indices": [int(i) for i in cluster_indices],
                "confidence": float(cluster_scores.mean()),
                "mean_embedding": mean_emb.astype(np.float32),
            }
        )

    # Sort by confidence descending
    clusters.sort(key=lambda c: c["confidence"], reverse=True)
    return clusters
