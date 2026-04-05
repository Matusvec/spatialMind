"""Tests for query-time semantic node building."""

import numpy as np

from server.services.query_node_builder import build_query_nodes


class MockCLIPEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        if text == "person":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


class MockGaussianStore:
    def __init__(self):
        self.positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.1, 0.02, 0.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )
        level0 = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.decoded_levels = {0: level0}

    def get_embeddings(self, level: int = 0):
        return self.decoded_levels.get(level)


def test_build_query_nodes_clusters_semantic_matches():
    store = MockGaussianStore()
    encoder = MockCLIPEncoder()
    negative_embeddings = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    nodes = build_query_nodes(
        text="person",
        gaussian_store=store,
        clip_encoder=encoder,
        negative_embeddings=negative_embeddings,
        threshold_schedule=(0.5,),
        max_k=100,
    )

    assert len(nodes) == 1
    assert nodes[0]["label"] == "person"
    assert len(nodes[0]["gaussian_indices"]) == 3
