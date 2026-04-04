"""Tests for the graph builder service.

Verifies that build_scene_graph orchestrates clustering, spatial relations,
and hierarchy into a complete scene graph payload.
"""

import numpy as np
import pytest

from server.services.graph_builder import build_scene_graph


# --- Test fixtures ---


class MockCLIPEncoder:
    """Mock CLIP encoder that returns a fixed unit vector."""

    def encode_text(self, text: str) -> np.ndarray:
        emb = np.zeros(512, dtype=np.float32)
        emb[0] = 1.0
        return emb


class MockGaussianStore:
    """Mock Gaussian store with 3 well-separated spatial clusters.

    Cluster layout:
    - Cluster A: 40 Gaussians near (0, 0, 0)
    - Cluster B: 30 Gaussians near (10, 0, 0)
    - Cluster C: 30 Gaussians near (0, 5, 0) — above cluster A
    """

    def __init__(self, n: int = 100):
        rng = np.random.RandomState(42)
        positions = []
        embeddings = []

        # Cluster A: around origin
        for _ in range(40):
            positions.append(rng.randn(3).astype(np.float32) * 0.3)
            emb = rng.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        # Cluster B: around (10, 0, 0)
        for _ in range(30):
            pos = rng.randn(3).astype(np.float32) * 0.3
            pos[0] += 10.0
            positions.append(pos)
            emb = rng.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        # Cluster C: around (0, 5, 0) — above cluster A
        for _ in range(30):
            pos = rng.randn(3).astype(np.float32) * 0.3
            pos[1] += 5.0
            positions.append(pos)
            emb = rng.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        self.positions = np.array(positions, dtype=np.float32)
        self.decoded_embeddings = np.array(embeddings, dtype=np.float32)
        self.count = n

    @property
    def is_loaded(self) -> bool:
        return True


@pytest.fixture
def mock_store():
    return MockGaussianStore()


@pytest.fixture
def mock_encoder():
    return MockCLIPEncoder()


# --- Tests ---


def test_build_scene_graph_returns_required_keys(mock_store, mock_encoder):
    """build_scene_graph returns dict with nodes, edges, hierarchy, metadata."""
    result = build_scene_graph(mock_store, mock_encoder)
    assert "nodes" in result
    assert "edges" in result
    assert "hierarchy" in result
    assert "metadata" in result


def test_nodes_length_matches_cluster_count(mock_store, mock_encoder):
    """Nodes list length matches the number of DBSCAN clusters (not noise)."""
    result = build_scene_graph(mock_store, mock_encoder)
    assert len(result["nodes"]) == result["metadata"]["cluster_count"]
    assert len(result["nodes"]) > 0


def test_node_has_required_keys(mock_store, mock_encoder):
    """Each node has id, label, centroid, bbox, gaussian_indices, confidence."""
    result = build_scene_graph(mock_store, mock_encoder)
    required_keys = {"id", "label", "centroid", "bbox", "gaussian_indices", "confidence"}
    for node in result["nodes"]:
        assert required_keys.issubset(node.keys()), f"Missing keys in node: {required_keys - node.keys()}"


def test_node_id_format(mock_store, mock_encoder):
    """Node id is in 'obj_NNN' format (zero-padded 3 digits)."""
    result = build_scene_graph(mock_store, mock_encoder)
    for i, node in enumerate(result["nodes"]):
        assert node["id"] == f"obj_{i:03d}", f"Expected obj_{i:03d}, got {node['id']}"


def test_node_label_default_format(mock_store, mock_encoder):
    """Node label defaults to 'object_N'."""
    result = build_scene_graph(mock_store, mock_encoder)
    for i, node in enumerate(result["nodes"]):
        assert node["label"] == f"object_{i}", f"Expected object_{i}, got {node['label']}"


def test_edges_contain_spatial_relationships(mock_store, mock_encoder):
    """Edges list contains spatial relationships between nodes."""
    result = build_scene_graph(mock_store, mock_encoder)
    # With 3 clusters at different positions, there should be some spatial edges
    assert isinstance(result["edges"], list)
    # Each edge should have the required structure
    for edge in result["edges"]:
        assert "from" in edge
        assert "to" in edge
        assert "relation" in edge
        assert "confidence" in edge
        assert "distance" in edge


def test_hierarchy_contains_similarity_edges(mock_store, mock_encoder):
    """Hierarchy list contains similarity edges between semantically similar nodes."""
    result = build_scene_graph(mock_store, mock_encoder)
    assert isinstance(result["hierarchy"], list)
    for h in result["hierarchy"]:
        assert "from" in h
        assert "to" in h
        assert "relation" in h
        assert "similarity" in h
        assert h["relation"] == "similar_to"


def test_k_parameter_passed_to_top_k(mock_store, mock_encoder):
    """build_scene_graph with custom k passes it to top_k for Gaussian selection."""
    # With k=50, fewer Gaussians are selected, potentially fewer clusters
    result_small = build_scene_graph(mock_store, mock_encoder, k=50)
    assert result_small["metadata"]["k"] == 50
    assert result_small["metadata"]["selected_count"] <= 50


def test_metadata_fields(mock_store, mock_encoder):
    """Metadata contains all required fields."""
    result = build_scene_graph(mock_store, mock_encoder)
    meta = result["metadata"]
    required = {"gaussian_count", "selected_count", "cluster_count", "edge_count", "hierarchy_count", "query", "k"}
    assert required.issubset(meta.keys())
    assert meta["gaussian_count"] == mock_store.count
    assert meta["query"] == "objects"
