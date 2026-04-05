"""Query-time node builder from dense semantic Gaussian retrieval."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import DBSCAN

from scipy.spatial import KDTree

from server.services.clustering import compute_dynamic_epsilon
from server.services.scene_matching import tokenize
from server.services.similarity import multi_level_relevancy, relevancy_scores, relevancy_top_k

logger = logging.getLogger(__name__)

LEVEL_CONFIG = {
    0: {"name": "objects", "min_samples": 10, "eps_min": 0.10, "eps_max": 0.80},
    1: {"name": "parts", "min_samples": 8, "eps_min": 0.06, "eps_max": 0.45},
    2: {"name": "subparts", "min_samples": 6, "eps_min": 0.04, "eps_max": 0.30},
}


def _focus_query(text: str) -> str:
    """Reduce a user query to its object-bearing content words."""
    tokens = tokenize(text)
    return " ".join(tokens) if tokens else text.strip().lower()


def _cluster_level(
    *,
    level: int,
    positions: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    query_label: str,
) -> list[dict]:
    """Cluster one semantic level into transient query-time nodes."""
    if len(positions) == 0:
        return []

    cfg = LEVEL_CONFIG[level]
    # Adapt min_samples to actual point count — aggressive clustering for sparse matches
    min_samples = min(cfg["min_samples"], max(2, len(positions) // 4))
    eps = compute_dynamic_epsilon(positions, min_samples=min_samples)
    eps = max(cfg["eps_min"], min(cfg["eps_max"], float(eps)))

    logger.info(
        "Query nodes level %d (%s): %d points, eps=%.3f, min_samples=%d",
        level,
        cfg["name"],
        len(positions),
        eps,
        min_samples,
    )

    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree").fit(positions)
    labels = db.labels_
    unique_labels = sorted(set(labels) - {-1})

    nodes = []
    for order, lbl in enumerate(unique_labels):
        mask = labels == lbl
        cluster_pos = positions[mask]
        cluster_scores = scores[mask]
        cluster_indices = indices[mask]
        centroid = cluster_pos.mean(axis=0)
        dists = np.linalg.norm(cluster_pos - centroid, axis=1)
        radius = float(np.percentile(dists, 75)) + 0.03 if len(dists) > 0 else 0.12
        bbox = [
            float(cluster_pos[:, 0].min()),
            float(cluster_pos[:, 1].min()),
            float(cluster_pos[:, 2].min()),
            float(cluster_pos[:, 0].max()),
            float(cluster_pos[:, 1].max()),
            float(cluster_pos[:, 2].max()),
        ]
        label = query_label if order == 0 else f"{query_label} {order + 1}"
        nodes.append(
            {
                "id": f"query_{level}_{order:02d}",
                "label": label,
                "centroid": centroid.tolist(),
                "bbox": bbox,
                "gaussian_indices": [int(i) for i in cluster_indices],
                "confidence": float(cluster_scores.mean()),
                "match_score": float(cluster_scores.mean()),
                "semantic_level": level,
                "semantic_level_name": cfg["name"],
                "radius": max(cfg["eps_min"], min(radius, cfg["eps_max"])),
            }
        )

    return nodes


def _expand_clusters_spatially(
    nodes: list[dict],
    all_positions: np.ndarray,
    all_scores: np.ndarray,
    min_score: float,
) -> list[dict]:
    """Iteratively flood-fill expand clusters to cover the full object shape.

    For each cluster, iteratively expands outward from the seed, pulling in
    nearby Gaussians with scores above `min_score`. Each round recomputes
    cluster geometry so it naturally walks outward to cover the full object.

    Args:
        nodes: Seed clusters from DBSCAN.
        all_positions: (N, 3) all Gaussian positions.
        all_scores: (N,) relevancy scores for all Gaussians.
        min_score: Minimum score for expansion candidates.
    """
    if len(nodes) == 0 or all_positions is None:
        return nodes

    MIN_RADIUS = {0: 0.30, 1: 0.15, 2: 0.08}

    # Pre-filter: only positions above min_score are candidates
    candidate_mask = all_scores >= min_score
    candidate_indices = np.where(candidate_mask)[0]
    if len(candidate_indices) == 0:
        return nodes
    candidate_positions = all_positions[candidate_indices]
    candidate_tree = KDTree(candidate_positions)

    # Three expansion rounds with increasing reach
    radius_multipliers = [2.0, 2.5, 3.0]

    for node in nodes:
        level = node.get("semantic_level", 0)
        min_r = MIN_RADIUS.get(level, 0.15)
        initial_count = len(node["gaussian_indices"])

        for radius_mult in radius_multipliers:
            current_indices = np.asarray(node["gaussian_indices"], dtype=np.int64)
            current_pos = all_positions[current_indices]
            centroid = current_pos.mean(axis=0)

            dists_from_centroid = np.linalg.norm(current_pos - centroid, axis=1)
            cluster_extent = float(np.percentile(dists_from_centroid, 95)) if len(dists_from_centroid) > 0 else min_r
            search_radius = max(min_r, cluster_extent) * radius_mult

            nearby = candidate_tree.query_ball_point(centroid, search_radius)
            if not nearby:
                continue

            nearby_global = candidate_indices[nearby]
            existing = set(node["gaussian_indices"])
            new_indices = [int(i) for i in nearby_global if int(i) not in existing]
            if new_indices:
                node["gaussian_indices"] = sorted(existing | set(new_indices))

        # Recompute geometry after all expansion rounds
        final_indices = np.asarray(node["gaussian_indices"], dtype=np.int64)
        if len(final_indices) > initial_count:
            pos = all_positions[final_indices]
            node["centroid"] = pos.mean(axis=0).tolist()
            dists = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
            node["bbox"] = [
                float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 2].min()),
                float(pos[:, 0].max()), float(pos[:, 1].max()), float(pos[:, 2].max()),
            ]
            cfg = LEVEL_CONFIG.get(level, LEVEL_CONFIG[0])
            node["radius"] = max(cfg["eps_min"], min(
                float(np.percentile(dists, 90)) + 0.05,
                cfg["eps_max"] * 2,
            ))
            logger.info(
                "Expanded cluster '%s': %d -> %d splats (%.1fx), radius=%.3f",
                node.get("label", "?"),
                initial_count,
                len(node["gaussian_indices"]),
                len(node["gaussian_indices"]) / max(1, initial_count),
                node["radius"],
            )

    return nodes


def build_query_nodes(
    *,
    text: str,
    gaussian_store,
    clip_encoder,
    negative_embeddings,
    threshold_schedule: tuple[float, ...] = (0.52, 0.50, 0.48, 0.45, 0.42, 0.40),
    max_k: int = 50000,
    max_nodes: int = 8,
    progress_cb=None,
    preferred_levels: tuple[int, ...] = (0, 1, 2),
    return_match_data: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Build transient query-specific nodes from the preferred semantic levels."""
    if progress_cb is not None:
        progress_cb("normalize_query", "active")
    if gaussian_store is None or clip_encoder is None or negative_embeddings is None:
        return ([], {}) if return_match_data else []

    focus_text = _focus_query(text)
    if progress_cb is not None:
        progress_cb("normalize_query", "completed", detail=focus_text or text.strip().lower())
    if not focus_text:
        return ([], {}) if return_match_data else []

    available_levels = []
    for level in range(3):
        emb = gaussian_store.get_embeddings(level)
        if emb is not None:
            available_levels.append(level)
    if not available_levels:
        return ([], {}) if return_match_data else []

    selected_levels = [level for level in preferred_levels if level in available_levels]
    if not selected_levels:
        selected_levels = available_levels

    level_embeddings = []
    level_map = []
    for level in selected_levels:
        emb = gaussian_store.get_embeddings(level)
        if emb is not None:
            level_embeddings.append(emb)
            level_map.append(level)
    if not level_embeddings:
        return ([], {}) if return_match_data else []

    if progress_cb is not None:
        progress_cb("encode_query", "active")
    query_embedding = clip_encoder.encode_text(focus_text)
    if progress_cb is not None:
        progress_cb("encode_query", "completed")

    best_indices = np.array([], dtype=np.int64)
    best_scores = np.array([], dtype=np.float32)
    best_levels = np.array([], dtype=np.int32)
    used_threshold = None
    scoring_mode = "lerf"

    if progress_cb is not None:
        progress_cb("score_semantics", "active")

    # Step 1: Try LERF relevancy scoring first
    N = level_embeddings[0].shape[0]
    lerf_scores = np.zeros(N, dtype=np.float32)
    all_levels = np.zeros(N, dtype=np.int32)
    for level_idx, embeddings in enumerate(level_embeddings):
        scores = relevancy_scores(query_embedding, negative_embeddings, embeddings)
        better = scores > lerf_scores
        lerf_scores[better] = scores[better]
        all_levels[better] = level_idx

    lerf_spread = float(lerf_scores.max()) - float(np.median(lerf_scores))

    if lerf_spread >= 0.02:
        # Normal regime: LERF has good discrimination, use absolute thresholds
        all_scores = lerf_scores
        for threshold in threshold_schedule:
            mask = all_scores >= threshold
            indices = np.where(mask)[0]
            if len(indices) > 0:
                best_indices = indices
                best_scores = all_scores[indices]
                best_levels = all_levels[indices]
                order = np.argsort(-best_scores)
                best_indices = best_indices[order][:max_k]
                best_scores = best_scores[order][:max_k]
                best_levels = best_levels[order][:max_k]
                used_threshold = threshold
                break
        logger.info(
            "Query nodes '%s': LERF mode (spread=%.5f), threshold=%.4f, %d matches",
            focus_text, lerf_spread, used_threshold if used_threshold else -1.0, len(best_indices),
        )
    else:
        # Compressed regime: LERF is degenerate (all scores ≈ 0.50).
        # Fall back to raw cosine similarity with percentile-based top-K.
        # This happens when the autoencoder produces collapsed embeddings.
        scoring_mode = "cosine"
        cosine_scores = np.zeros(N, dtype=np.float32)
        for level_idx, embeddings in enumerate(level_embeddings):
            cos = (embeddings @ query_embedding).astype(np.float32)
            better = cos > cosine_scores
            cosine_scores[better] = cos[better]
            all_levels[better] = level_idx

        all_scores = cosine_scores
        cos_spread = float(cosine_scores.max()) - float(np.median(cosine_scores))

        # Take top-K by percentile: top 0.1% of Gaussians, min 200, max max_k
        target_count = min(max_k, max(200, int(N * 0.001)))
        sorted_scores = np.sort(cosine_scores)[::-1]
        adaptive_threshold = float(sorted_scores[min(target_count - 1, len(sorted_scores) - 1)])

        mask = cosine_scores >= adaptive_threshold
        best_indices = np.where(mask)[0]
        best_scores = cosine_scores[best_indices]
        best_levels = all_levels[best_indices]

        order = np.argsort(-best_scores)
        best_indices = best_indices[order][:max_k]
        best_scores = best_scores[order][:max_k]
        best_levels = best_levels[order][:max_k]
        used_threshold = adaptive_threshold

        logger.info(
            "Query nodes '%s': COSINE FALLBACK (lerf_spread=%.5f, cos_spread=%.5f), "
            "threshold=%.6f, %d matches (target=%d)",
            focus_text, lerf_spread, cos_spread, adaptive_threshold,
            len(best_indices), target_count,
        )

    if len(best_indices) == 0:
        logger.info("Query nodes '%s': no gaussians above any threshold", focus_text)
        if progress_cb is not None:
            progress_cb("score_semantics", "completed", detail="0 matches")
        return ([], {}) if return_match_data else []
    if progress_cb is not None:
        progress_cb(
            "score_semantics",
            "completed",
            detail=f"{len(best_indices)} matches across {len(level_embeddings)} level(s)",
        )

    nodes = []
    if progress_cb is not None:
        progress_cb("cluster_regions", "active")
    for local_level_idx, real_level in enumerate(level_map):
        mask = best_levels == local_level_idx
        if not np.any(mask):
            continue

        level_indices = best_indices[mask]
        level_scores = best_scores[mask]
        level_positions = gaussian_store.positions[level_indices]
        level_nodes = _cluster_level(
            level=real_level,
            positions=level_positions,
            scores=level_scores,
            indices=level_indices,
            query_label=focus_text,
        )
        nodes.extend(level_nodes)

    # Spatial expansion: pull in nearby Gaussians just below threshold.
    # Use the already-computed all_scores (full scene, best level per Gaussian).
    if nodes and used_threshold is not None and gaussian_store.positions is not None:
        # Expansion threshold: grab nearby Gaussians scoring slightly below the cutoff.
        # In cosine mode, use 85% of the adaptive threshold.
        # In LERF mode, use 65% of the absolute threshold.
        if scoring_mode == "cosine":
            expansion_threshold = used_threshold * 0.85
        else:
            expansion_threshold = used_threshold * 0.65
        nodes = _expand_clusters_spatially(
            nodes,
            gaussian_store.positions,
            all_scores,
            expansion_threshold,
        )

    nodes.sort(key=lambda node: (-node["match_score"], node["semantic_level"]))
    if progress_cb is not None:
        progress_cb("cluster_regions", "completed", detail=f"{len(nodes[:max_nodes])} cluster(s)")
    match_data = {
        "focus_text": focus_text,
        "indices": [int(i) for i in best_indices.tolist()],
        "scores": [float(s) for s in best_scores.tolist()],
        "levels": [int(level_map[int(local_idx)]) for local_idx in best_levels.tolist()],
        "used_threshold": float(used_threshold if used_threshold is not None else -1.0),
    }
    if return_match_data:
        return nodes[:max_nodes], match_data
    return nodes[:max_nodes]
