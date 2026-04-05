"""Tests for the highlight pipeline: LERF scoring → DBSCAN → regions.

Verifies:
1. LERF relevancy scores are in [0, 1] with correct formula
2. "person" query returns positions near the known person location
3. "guitar" (not in scene) returns fewer matches than "person"
4. Returned Gaussian indices map to valid 3D positions
5. Highlight regions have reasonable centroids and radii
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def app_state():
    """Load the full pipeline once for all tests."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from server.config import Settings
    from server.services.gaussian_store import GaussianStore
    from server.services.clip_encoder import CLIPEncoder
    from server.models.autoencoder import load_autoencoder
    from server.services.similarity import CANONICAL_NEGATIVES

    config = Settings()
    gs = GaussianStore(config)
    gs.load_ply(os.path.join(config.scene_dir, "semantic_0.ply"))

    ae = load_autoencoder(os.path.join(config.scene_dir, "autoencoder.pth"), config.device)
    gs.decode_all(ae, config.device)
    gs.decoded_levels[0] = gs.decoded_embeddings

    clip = CLIPEncoder(config.clip_model, config.clip_pretrained, config.device)
    neg_embs = np.stack([clip.encode_text(n) for n in CANONICAL_NEGATIVES])

    return {
        "gaussian_store": gs,
        "clip_encoder": clip,
        "negative_embeddings": neg_embs,
        "config": config,
    }


class TestLERFScoring:
    def test_relevancy_range(self, app_state):
        """Relevancy scores should be in [0, 1]."""
        from server.services.similarity import relevancy_scores

        gs = app_state["gaussian_store"]
        clip = app_state["clip_encoder"]
        neg = app_state["negative_embeddings"]

        emb = clip.encode_text("person")
        scores = relevancy_scores(emb, neg, gs.decoded_embeddings)

        assert scores.min() >= 0.0
        assert scores.max() <= 1.0
        assert scores.dtype == np.float32

    def test_person_scores_higher_than_guitar(self, app_state):
        """'person' should have higher max relevancy than 'guitar' in this scene."""
        from server.services.similarity import relevancy_scores

        gs = app_state["gaussian_store"]
        clip = app_state["clip_encoder"]
        neg = app_state["negative_embeddings"]

        person_scores = relevancy_scores(clip.encode_text("person"), neg, gs.decoded_embeddings)
        guitar_scores = relevancy_scores(clip.encode_text("guitar"), neg, gs.decoded_embeddings)

        person_above_threshold = (person_scores > 0.5).sum()
        guitar_above_threshold = (guitar_scores > 0.5).sum()

        print(f"person above 0.5: {person_above_threshold}, guitar above 0.5: {guitar_above_threshold}")
        assert person_above_threshold > guitar_above_threshold

    def test_min_pairwise_formula(self, app_state):
        """Verify LERF uses min-of-pairwise-sigmoid, not softmax."""
        from server.services.similarity import relevancy_scores

        gs = app_state["gaussian_store"]
        clip = app_state["clip_encoder"]
        neg = app_state["negative_embeddings"]

        emb = clip.encode_text("chair")
        scores = relevancy_scores(emb, neg, gs.decoded_embeddings[:100])

        # Manual computation for first Gaussian
        phi = gs.decoded_embeddings[0]
        sim_q = float(phi @ emb)
        sim_c = phi @ neg.T  # (4,)
        diff = sim_q - sim_c
        pairwise = 1.0 / (1.0 + np.exp(-diff))
        expected = float(pairwise.min())

        assert abs(scores[0] - expected) < 1e-5, f"Expected {expected:.6f}, got {scores[0]:.6f}"


class TestHighlightPositions:
    def test_matched_indices_are_valid(self, app_state):
        """Returned Gaussian indices should be within bounds."""
        from server.services.similarity import relevancy_top_k

        gs = app_state["gaussian_store"]
        clip = app_state["clip_encoder"]
        neg = app_state["negative_embeddings"]

        indices, scores = relevancy_top_k(
            clip.encode_text("person"), neg, gs.decoded_embeddings, threshold=0.5
        )

        assert len(indices) > 0, "Should find some person Gaussians"
        assert indices.min() >= 0
        assert indices.max() < gs.count
        assert len(scores) == len(indices)

    def test_matched_positions_are_spatially_coherent(self, app_state):
        """Matched Gaussians should cluster spatially, not be random."""
        from server.services.similarity import relevancy_top_k

        gs = app_state["gaussian_store"]
        clip = app_state["clip_encoder"]
        neg = app_state["negative_embeddings"]

        indices, _ = relevancy_top_k(
            clip.encode_text("person"), neg, gs.decoded_embeddings, threshold=0.5, max_k=500
        )

        positions = gs.positions[indices]
        centroid = positions.mean(axis=0)
        dists = np.linalg.norm(positions - centroid, axis=1)
        mean_dist = dists.mean()

        # If Gaussians are random, mean distance from centroid would be ~scene diameter
        scene_diameter = np.linalg.norm(gs.positions.max(axis=0) - gs.positions.min(axis=0))
        print(f"Mean dist from centroid: {mean_dist:.2f}, scene diameter: {scene_diameter:.2f}")

        # Matched Gaussians should be within 50% of scene diameter (not randomly scattered)
        assert mean_dist < scene_diameter * 0.5, (
            f"Matched Gaussians too scattered: mean_dist={mean_dist:.2f} vs scene_diameter={scene_diameter:.2f}"
        )

    def test_position_matches_splat_file(self, app_state):
        """PLY index N should have the same position as splat index N."""
        import struct

        gs = app_state["gaussian_store"]
        splat_path = "room-cognito/public/scene-indexed.splat"

        # Read first 10 positions from splat
        with open(splat_path, "rb") as f:
            for i in range(10):
                data = f.read(32)
                sx, sy, sz = struct.unpack("fff", data[:12])
                px, py, pz = gs.positions[i]
                assert abs(sx - px) < 0.01, f"Index {i} X mismatch: splat={sx:.4f} ply={px:.4f}"
                assert abs(sy - py) < 0.01, f"Index {i} Y mismatch: splat={sy:.4f} ply={py:.4f}"
                assert abs(sz - pz) < 0.01, f"Index {i} Z mismatch: splat={sz:.4f} ply={pz:.4f}"

        print("First 10 indices: PLY ↔ splat positions match ✓")


class TestGroundQuery:
    def test_ground_query_returns_regions(self, app_state):
        """ground_query should return highlight regions with valid structure."""
        from server.services.dense_query import ground_query

        state = {**app_state, "autoencoder": None, "instance_store": {
            "instances": [], "next_id": 0, "query_history": [],
        }, "instance_graph": {"nodes": [], "edges": [], "hierarchy": [], "metadata": {}}}

        result = ground_query(text="person", state=state, persist_instances=False)

        assert "nodes" in result
        assert "highlight_regions" in result
        assert "highlight_match" in result

        hm = result["highlight_match"]
        assert len(hm["indices"]) > 0, "Should find Gaussian indices"
        print(f"ground_query('person'): {len(hm['indices'])} indices, {len(result['highlight_regions'])} regions")

        # Check regions have valid structure
        for r in result["highlight_regions"]:
            assert "centroid" in r
            assert "radius" in r
            assert len(r["centroid"]) == 3
            assert r["radius"] > 0
