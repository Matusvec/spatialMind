"""Tests for spatial relationship extractor and hierarchy builder."""

import numpy as np
import pytest

from server.services.spatial_relations import build_hierarchy, extract_spatial_relations


def _make_cluster(label, centroid, bbox=None, embedding=None):
    """Helper to create a cluster dict for testing."""
    cx, cy, cz = centroid
    if bbox is None:
        # Default tight bbox around centroid
        bbox = [cx - 0.5, cy - 0.5, cz - 0.5, cx + 0.5, cy + 0.5, cz + 0.5]
    if embedding is None:
        rng = np.random.default_rng(label)
        embedding = rng.standard_normal(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
    return {
        "label": label,
        "centroid": list(centroid),
        "bbox": list(bbox),
        "gaussian_indices": list(range(label * 10, label * 10 + 10)),
        "confidence": 0.9,
        "mean_embedding": embedding,
    }


class TestExtractSpatialRelations:
    """Tests for extract_spatial_relations."""

    def test_on_top_of_from_y_offset(self):
        """Cluster A above B produces on_top_of edge from A to B."""
        cluster_a = _make_cluster(0, [0, 5, 0])
        cluster_b = _make_cluster(1, [0, 0, 0])
        edges = extract_spatial_relations([cluster_a, cluster_b])
        on_top = [e for e in edges if e["relation"] == "on_top_of"]
        assert len(on_top) >= 1
        # A is on top of B (A has higher Y)
        edge = on_top[0]
        assert edge["from_label"] == 0
        assert edge["to_label"] == 1

    def test_next_to_from_xz_proximity(self):
        """Clusters close in XZ plane produce next_to edge."""
        cluster_a = _make_cluster(0, [1, 0, 0])
        cluster_b = _make_cluster(1, [3, 0, 0])
        edges = extract_spatial_relations([cluster_a, cluster_b])
        next_to = [e for e in edges if e["relation"] == "next_to"]
        assert len(next_to) >= 1

    def test_no_on_top_of_at_same_height(self):
        """No on_top_of edge when both clusters are at same height."""
        cluster_a = _make_cluster(0, [0, 0, 0])
        cluster_b = _make_cluster(1, [2, 0, 0])
        edges = extract_spatial_relations([cluster_a, cluster_b])
        on_top = [e for e in edges if e["relation"] == "on_top_of"]
        assert len(on_top) == 0

    def test_no_next_to_when_far(self):
        """No next_to edge when XZ distance is very large."""
        cluster_a = _make_cluster(0, [0, 0, 0])
        cluster_b = _make_cluster(1, [1000, 0, 1000])
        edges = extract_spatial_relations([cluster_a, cluster_b])
        next_to = [e for e in edges if e["relation"] == "next_to"]
        assert len(next_to) == 0

    def test_on_top_of_confidence_clear_vs_ambiguous(self):
        """on_top_of confidence is higher for clear gap than overlapping bboxes."""
        # Clear gap: A is well above B
        cluster_a_clear = _make_cluster(
            0, [0, 5, 0], bbox=[-0.5, 4.5, -0.5, 0.5, 5.5, 0.5]
        )
        cluster_b_clear = _make_cluster(
            1, [0, 0, 0], bbox=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
        )
        edges_clear = extract_spatial_relations([cluster_a_clear, cluster_b_clear])
        on_top_clear = [e for e in edges_clear if e["relation"] == "on_top_of"]

        # Ambiguous: overlapping bboxes in Y
        cluster_a_ambig = _make_cluster(
            0, [0, 2, 0], bbox=[-0.5, 0.5, -0.5, 0.5, 3.5, 0.5]
        )
        cluster_b_ambig = _make_cluster(
            1, [0, 0, 0], bbox=[-0.5, -0.5, -0.5, 0.5, 1.5, 0.5]
        )
        edges_ambig = extract_spatial_relations([cluster_a_ambig, cluster_b_ambig])
        on_top_ambig = [e for e in edges_ambig if e["relation"] == "on_top_of"]

        assert len(on_top_clear) >= 1
        assert len(on_top_ambig) >= 1
        assert on_top_clear[0]["confidence"] > on_top_ambig[0]["confidence"]

    def test_next_to_confidence_inversely_proportional(self):
        """next_to confidence is higher for closer clusters."""
        # Close pair
        cluster_a = _make_cluster(0, [0, 0, 0])
        cluster_b = _make_cluster(1, [1, 0, 0])
        # Farther pair
        cluster_c = _make_cluster(2, [0, 0, 0])
        cluster_d = _make_cluster(3, [4, 0, 0])

        edges_close = extract_spatial_relations([cluster_a, cluster_b])
        edges_far = extract_spatial_relations([cluster_c, cluster_d])

        next_to_close = [e for e in edges_close if e["relation"] == "next_to"]
        next_to_far = [e for e in edges_far if e["relation"] == "next_to"]

        assert len(next_to_close) >= 1
        assert len(next_to_far) >= 1
        assert next_to_close[0]["confidence"] > next_to_far[0]["confidence"]

    def test_edge_dict_keys(self):
        """Each edge dict has required keys."""
        cluster_a = _make_cluster(0, [0, 5, 0])
        cluster_b = _make_cluster(1, [0, 0, 0])
        edges = extract_spatial_relations([cluster_a, cluster_b])
        assert len(edges) >= 1
        for e in edges:
            assert "from_label" in e
            assert "to_label" in e
            assert "relation" in e
            assert "confidence" in e
            assert "distance" in e


class TestBuildHierarchy:
    """Tests for build_hierarchy."""

    def test_similar_clusters_grouped(self):
        """Two clusters with high cosine similarity produce a hierarchy edge."""
        rng = np.random.default_rng(42)
        base_emb = rng.standard_normal(512).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        # Make two very similar embeddings
        noise = rng.standard_normal(512).astype(np.float32) * 0.01
        similar_emb = base_emb + noise
        similar_emb = similar_emb / np.linalg.norm(similar_emb)

        # And one very different
        diff_emb = rng.standard_normal(512).astype(np.float32)
        diff_emb = diff_emb / np.linalg.norm(diff_emb)

        cluster_a = _make_cluster(0, [0, 0, 0], embedding=base_emb)
        cluster_b = _make_cluster(1, [5, 0, 0], embedding=similar_emb)
        cluster_c = _make_cluster(2, [10, 0, 0], embedding=diff_emb)

        hierarchy = build_hierarchy([cluster_a, cluster_b, cluster_c])
        # A and B should be linked
        ab_edges = [
            e for e in hierarchy
            if {e["from_label"], e["to_label"]} == {0, 1}
        ]
        assert len(ab_edges) >= 1

    def test_hierarchy_edge_format(self):
        """Hierarchy edges have relation 'similar_to' and similarity score."""
        rng = np.random.default_rng(42)
        base_emb = rng.standard_normal(512).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        noise = rng.standard_normal(512).astype(np.float32) * 0.01
        similar_emb = base_emb + noise
        similar_emb = similar_emb / np.linalg.norm(similar_emb)

        cluster_a = _make_cluster(0, [0, 0, 0], embedding=base_emb)
        cluster_b = _make_cluster(1, [5, 0, 0], embedding=similar_emb)

        hierarchy = build_hierarchy([cluster_a, cluster_b])
        assert len(hierarchy) >= 1
        edge = hierarchy[0]
        assert edge["relation"] == "similar_to"
        assert "similarity" in edge
        assert isinstance(edge["similarity"], float)
        assert 0.0 <= edge["similarity"] <= 1.0
