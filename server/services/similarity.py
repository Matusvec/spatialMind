"""Cosine similarity computation and top-K retrieval.

Both text embeddings and Gaussian embeddings are assumed to be L2-normalized,
so cosine similarity reduces to a dot product.
"""

import numpy as np


def top_k(
    text_embedding: np.ndarray,
    gaussian_embeddings: np.ndarray,
    k: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the top-K most similar Gaussians to a text query.

    Uses np.argpartition for O(N) partial sort, then sorts only the
    top-K results by descending similarity.

    Args:
        text_embedding: (512,) normalized text CLIP vector.
        gaussian_embeddings: (N, 512) normalized Gaussian CLIP vectors.
        k: Number of top results to return.

    Returns:
        Tuple of (indices, scores) as 1-D arrays, sorted by descending score.
    """
    # Cosine similarity = dot product (both L2-normalized)
    similarities = gaussian_embeddings @ text_embedding

    # Clamp k to available Gaussians
    k = min(k, len(similarities))

    # O(N) partial sort to find top-k indices
    top_k_unsorted = np.argpartition(-similarities, k)[:k]

    # Sort the top-k by descending similarity
    top_k_scores = similarities[top_k_unsorted]
    sorted_order = np.argsort(-top_k_scores)

    indices = top_k_unsorted[sorted_order]
    scores = top_k_scores[sorted_order]

    return indices, scores


def soft_scores(
    text_embedding: np.ndarray,
    gaussian_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity for ALL Gaussians.

    Returns the raw similarity scores as a continuous field,
    used by the probability cloud endpoint.

    Args:
        text_embedding: (512,) normalized text CLIP vector.
        gaussian_embeddings: (N, 512) normalized Gaussian CLIP vectors.

    Returns:
        (N,) float32 array of cosine similarity scores.
    """
    return (gaussian_embeddings @ text_embedding).astype(np.float32)
