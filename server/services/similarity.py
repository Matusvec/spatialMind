"""LERF-style relevancy scoring for LangSplat Gaussians.

The core insight: raw CLIP cosine similarity is relative, not absolute.
A "couch" query scores ~0.28 on the couch and ~0.24 on the carpet.
Thresholding raw similarity lights up everything.

LERF fixes this with canonical phrase comparison:
  For each Gaussian φ and query q, against canonical phrases c_i:
    pairwise_i = exp(φ·q) / (exp(φ·q) + exp(φ·c_i))
    relevancy = min(pairwise_1, ..., pairwise_M)

A Gaussian only scores high if it beats EVERY canonical phrase.
Threshold at 0.5 = "more relevant than generic stuff."
"""

import numpy as np


CANONICAL_NEGATIVES = ["background", "wall", "floor", "ceiling", "nothing", "empty space"]


def relevancy_scores(
    text_embedding: np.ndarray,
    negative_embeddings: np.ndarray,
    gaussian_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute LERF relevancy scores for all Gaussians.

    For each Gaussian φ:
      For each canonical phrase c_i:
        pairwise_i = exp(φ·q) / (exp(φ·q) + exp(φ·c_i))
      relevancy = min(pairwise_1, ..., pairwise_M)

    Args:
        text_embedding: (512,) L2-normalized query vector.
        negative_embeddings: (M, 512) canonical phrase vectors.
        gaussian_embeddings: (N, 512) decoded Gaussian CLIP vectors.

    Returns:
        (N,) float32 relevancy scores in [0, 1].
    """
    # (N,) query similarity
    sim_q = gaussian_embeddings @ text_embedding
    # (N, M) canonical similarities
    sim_c = gaussian_embeddings @ negative_embeddings.T

    # Pairwise softmax: exp(sim_q) / (exp(sim_q) + exp(sim_c_i))
    # For numerical stability, compute in log space:
    #   sigmoid(sim_q - sim_c_i) = 1 / (1 + exp(sim_c_i - sim_q))
    # This is equivalent to exp(sim_q) / (exp(sim_q) + exp(sim_c_i))
    diff = sim_q[:, None] - sim_c  # (N, M)
    pairwise = 1.0 / (1.0 + np.exp(-diff))  # sigmoid, (N, M)

    # Relevancy = worst case (min across all canonical phrases)
    relevancy = pairwise.min(axis=1)  # (N,)

    return relevancy.astype(np.float32)


def relevancy_top_k(
    text_embedding: np.ndarray,
    negative_embeddings: np.ndarray,
    gaussian_embeddings: np.ndarray,
    threshold: float = 0.5,
    max_k: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Find Gaussians with relevancy above threshold.

    Args:
        text_embedding: (512,) query vector.
        negative_embeddings: (M, 512) canonical phrase vectors.
        gaussian_embeddings: (N, 512) Gaussian CLIP vectors.
        threshold: Minimum relevancy (0.5 = beats all negatives).
        max_k: Cap on returned indices.

    Returns:
        (indices, scores) sorted by descending relevancy.
    """
    scores = relevancy_scores(
        text_embedding, negative_embeddings, gaussian_embeddings
    )
    mask = scores >= threshold
    indices = np.where(mask)[0]
    matched_scores = scores[indices]

    # Sort descending
    order = np.argsort(-matched_scores)
    indices = indices[order][:max_k]
    matched_scores = matched_scores[order][:max_k]

    return indices, matched_scores


def multi_level_relevancy(
    text_embedding: np.ndarray,
    negative_embeddings: np.ndarray,
    level_embeddings: list[np.ndarray],
    threshold: float = 0.5,
    max_k: int = 50000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute relevancy across multiple semantic levels, per-Gaussian max.

    For each Gaussian, the best-matching level wins. A Gaussian that
    scores 0.3 as "chair" (level 0) but 0.7 as "armrest" (level 1)
    gets 0.7.

    Args:
        text_embedding: (512,) query vector.
        negative_embeddings: (M, 512) canonical phrase vectors.
        level_embeddings: List of (N, 512) arrays, one per level.
        threshold: Minimum relevancy.
        max_k: Cap on returned indices.

    Returns:
        (indices, scores, best_levels) sorted by descending relevancy.
    """
    N = level_embeddings[0].shape[0]
    best_scores = np.zeros(N, dtype=np.float32)
    best_levels = np.zeros(N, dtype=np.int32)

    for level_idx, embeddings in enumerate(level_embeddings):
        scores = relevancy_scores(text_embedding, negative_embeddings, embeddings)
        better = scores > best_scores
        best_scores[better] = scores[better]
        best_levels[better] = level_idx

    # Threshold
    mask = best_scores >= threshold
    indices = np.where(mask)[0]
    matched_scores = best_scores[indices]
    matched_levels = best_levels[indices]

    # Sort descending, cap
    order = np.argsort(-matched_scores)
    indices = indices[order][:max_k]
    matched_scores = matched_scores[order][:max_k]
    matched_levels = matched_levels[order][:max_k]

    return indices, matched_scores, matched_levels


def top_k(
    text_embedding: np.ndarray,
    gaussian_embeddings: np.ndarray,
    k: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the top-K most similar Gaussians by raw cosine similarity.

    Kept for backward compat (scene graph builder). Prefer relevancy_top_k.
    """
    similarities = gaussian_embeddings @ text_embedding
    k = min(k, len(similarities))

    if k >= len(similarities):
        sorted_order = np.argsort(-similarities)
        return sorted_order, similarities[sorted_order]

    top_k_unsorted = np.argpartition(-similarities, k)[:k]
    top_k_scores = similarities[top_k_unsorted]
    sorted_order = np.argsort(-top_k_scores)

    return top_k_unsorted[sorted_order], top_k_scores[sorted_order]


def highlight_mask(
    text_embedding: np.ndarray,
    gaussian_embeddings: np.ndarray,
    k: int = 100,
    mode: str = "soft",
) -> np.ndarray:
    """Generate per-Gaussian opacity values for highlighting. Kept for compat."""
    similarities = gaussian_embeddings @ text_embedding
    mask = np.zeros(len(similarities), dtype=np.float32)
    k = min(k, len(similarities))
    top_indices = np.argpartition(-similarities, k)[:k]
    top_scores = similarities[top_indices]
    score_min = top_scores.min()
    score_max = top_scores.max()
    if score_max > score_min:
        normalized = (top_scores - score_min) / (score_max - score_min)
        mask[top_indices] = 0.3 + 0.7 * normalized
    else:
        mask[top_indices] = 1.0
    return mask


def probability_cloud(
    text_embedding: np.ndarray,
    gaussian_embeddings: np.ndarray,
    temperature: float = 0.1,
) -> np.ndarray:
    """Per-Gaussian soft probability (softmax). Kept for heatmap endpoint."""
    similarities = gaussian_embeddings @ text_embedding
    scaled = similarities / temperature
    scaled = scaled - scaled.max()
    exp_scores = np.exp(scaled)
    return (exp_scores / exp_scores.sum()).astype(np.float32)
