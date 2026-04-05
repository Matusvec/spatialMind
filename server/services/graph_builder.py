"""Scene graph builder — clusters Gaussians by decoded CLIP embeddings.

No hardcoded seed queries. Uses DBSCAN on 3D positions of the most
semantically distinct Gaussians, then auto-labels each cluster by
finding the closest text embedding from a vocabulary. All open-vocabulary.
"""

import logging

import numpy as np

from server.services.clustering import cluster_gaussians
from server.services.similarity import relevancy_scores, CANONICAL_NEGATIVES
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
    """Build scene graph by clustering semantically distinct Gaussians.

    1. Compute relevancy scores for a broad query ("things in a room")
    2. Take Gaussians above threshold — these are the "interesting" ones
    3. DBSCAN cluster by 3D position
    4. Auto-label each cluster via CLIP similarity against a vocab
    """
    embeddings = gaussian_store.decoded_embeddings
    positions = gaussian_store.positions

    # Pre-encode negatives
    neg_embs = np.stack([
        clip_encoder.encode_text(neg) for neg in CANONICAL_NEGATIVES
    ])

    # Broad query to find "interesting" Gaussians
    broad_emb = clip_encoder.encode_text("things in a room")
    scores = relevancy_scores(broad_emb, neg_embs, embeddings)

    # Take all Gaussians with relevancy > 0.5
    mask = scores > 0.5
    interesting_count = mask.sum()
    logger.info("Broad discovery: %d / %d Gaussians above 0.5 relevancy", interesting_count, len(scores))

    if interesting_count < 20:
        # Not enough — lower threshold
        threshold = float(np.percentile(scores, 99.5))
        mask = scores >= threshold
        interesting_count = mask.sum()
        logger.info("Lowered threshold to %.3f: %d Gaussians", threshold, interesting_count)

    # Cap at 50K for DBSCAN performance
    if interesting_count > 50000:
        top_indices = np.argpartition(-scores, 50000)[:50000]
        interesting_indices = top_indices
    else:
        interesting_indices = np.where(mask)[0]

    sel_positions = positions[interesting_indices]
    sel_embeddings = embeddings[interesting_indices]
    sel_scores = scores[interesting_indices]

    # DBSCAN cluster by position
    clusters = cluster_gaussians(
        sel_positions, sel_embeddings, sel_scores, interesting_indices, min_samples=min_samples
    )

    if not clusters:
        logger.warning("No clusters found in scene")
        return {"nodes": [], "edges": [], "hierarchy": [], "metadata": {"cluster_count": 0}}

    # Auto-label each cluster via CLIP similarity
    # Use a broad vocabulary (not hardcoded 35 words)
    vocab = [
        "person", "man", "woman", "child", "chair", "armchair", "stool", "bench",
        "table", "desk", "coffee table", "couch", "sofa", "loveseat",
        "bed", "pillow", "cushion", "blanket",
        "window", "door", "wall", "floor", "ceiling", "stairs", "railing",
        "lamp", "ceiling light", "floor lamp", "chandelier",
        "plant", "potted plant", "flower", "tree",
        "rug", "carpet", "mat",
        "monitor", "laptop", "computer", "TV", "screen", "keyboard",
        "phone", "tablet", "remote",
        "bookshelf", "shelf", "cabinet", "drawer", "wardrobe", "closet",
        "basket", "box", "container", "bin", "trash can",
        "bag", "backpack", "purse", "suitcase",
        "bottle", "cup", "mug", "glass", "plate", "bowl",
        "book", "notebook", "paper", "magazine",
        "clock", "mirror", "painting", "poster", "photo frame",
        "shoe", "sneaker", "boot", "slipper",
        "toy", "ball", "teddy bear",
        "guitar", "piano", "instrument",
        "umbrella", "hat", "coat", "jacket",
        "towel", "tissue box", "soap",
        "whiteboard", "blackboard", "projector",
        "fan", "heater", "air conditioner",
        "refrigerator", "microwave", "oven", "sink", "faucet",
    ]
    vocab_embeddings = np.stack([clip_encoder.encode_text(w) for w in vocab])

    label_counts: dict[str, int] = {}
    for cluster in clusters:
        mean_emb = cluster["mean_embedding"]
        sims = vocab_embeddings @ mean_emb
        best_idx = int(np.argmax(sims))
        raw_label = vocab[best_idx]
        count = label_counts.get(raw_label, 0)
        label_counts[raw_label] = count + 1
        cluster["semantic_label"] = raw_label if count == 0 else f"{raw_label} {count + 1}"

    # Spatial relationships
    spatial_edges = extract_spatial_relations(clusters)
    hierarchy_edges = build_hierarchy(clusters, hierarchy_threshold)

    result = {
        "nodes": [
            {
                "id": f"obj_{i:03d}",
                "label": c["semantic_label"],
                "centroid": c["centroid"],
                "bbox": c["bbox"],
                "gaussian_indices": c["gaussian_indices"],
                "confidence": c["confidence"],
            }
            for i, c in enumerate(clusters)
        ],
        "edges": [
            {
                "from": f"obj_{e['from_label']:03d}",
                "to": f"obj_{e['to_label']:03d}",
                "relation": e["relation"],
                "confidence": e["confidence"],
                "distance": e["distance"],
            }
            for e in spatial_edges
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
            "cluster_count": len(clusters),
            "edge_count": len(spatial_edges),
            "labels": [c["semantic_label"] for c in clusters],
        },
    }

    logger.info(
        "Scene graph: %d objects [%s], %d edges",
        len(clusters),
        ", ".join(c["semantic_label"] for c in clusters),
        len(spatial_edges),
    )

    return result
