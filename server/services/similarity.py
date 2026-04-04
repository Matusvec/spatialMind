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


def highlight_mask(
    text_embedding: np.ndarray,
    gaussian_embeddings: np.ndarray,
    k: int = 100,
    mode: str = "soft",
) -> np.ndarray:
    """Generate per-Gaussian opacity values for highlighting.

    Args:
        text_embedding: (512,) normalized CLIP text vector.
        gaussian_embeddings: (N, 512) normalized Gaussian embeddings.
        k: Number of top Gaussians to highlight.
        mode: "hard" for binary 0/1, "soft" for gradient opacity.

    Returns:
        (N,) float32 array with values in [0.0, 1.0].
    """
    similarities = gaussian_embeddings @ text_embedding  # (N,)
    mask = np.zeros(len(similarities), dtype=np.float32)

    # Clamp k to available Gaussians
    k = min(k, len(similarities))

    if mode == "hard":
        # Binary: top-k get 1.0, rest get 0.0
        top_indices = np.argpartition(-similarities, k)[:k]
        mask[top_indices] = 1.0
    else:
        # Soft: top-k get opacity proportional to their similarity score
        top_indices = np.argpartition(-similarities, k)[:k]
        top_scores = similarities[top_indices]
        # Normalize scores to [0, 1] range within the top-k
        score_min = top_scores.min()
        score_max = top_scores.max()
        if score_max > score_min:
            normalized = (top_scores - score_min) / (score_max - score_min)
            # Apply minimum opacity of 0.3 for visibility
            mask[top_indices] = 0.3 + 0.7 * normalized
        else:
            mask[top_indices] = 1.0

    return mask


def probability_cloud(
    text_embedding: np.ndarray,
    gaussian_embeddings: np.ndarray,
    temperature: float = 0.1,
) -> np.ndarray:
    """Generate per-Gaussian soft probability assignments.

    Applies softmax with temperature scaling to cosine similarities,
    producing a probability distribution over ALL Gaussians. This is
    the continuous semantic field -- every Gaussian gets a score, not
    just the top-k.

    Args:
        text_embedding: (512,) normalized CLIP text vector.
        gaussian_embeddings: (N, 512) normalized Gaussian embeddings.
        temperature: Softmax temperature. Lower = sharper peaks. Default 0.1.

    Returns:
        (N,) float32 array of soft assignment probabilities (sums to ~1.0).
    """
    similarities = gaussian_embeddings @ text_embedding  # (N,)
    # Temperature-scaled softmax for soft assignments
    # Subtract max for numerical stability
    scaled = similarities / temperature
    scaled = scaled - scaled.max()
    exp_scores = np.exp(scaled)
    probabilities = exp_scores / exp_scores.sum()
    return probabilities.astype(np.float32)
