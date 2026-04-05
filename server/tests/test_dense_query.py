"""Tests for the dense query-time grounding pipeline."""

import numpy as np

from server.services.dense_query import ground_query


class MockCLIPEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        if text in {"person", "people"}:
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
        self.decoded_embeddings = level0
        self.count = len(self.positions)

    def get_embeddings(self, level: int = 0):
        return self.decoded_levels.get(level)


def test_ground_query_materializes_cached_instances():
    state = {
        "gaussian_store": MockGaussianStore(),
        "clip_encoder": MockCLIPEncoder(),
        "negative_embeddings": np.array(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        "instance_store": {"instances": [], "next_id": 0, "query_history": []},
        "instance_graph": None,
        "scene_graph": None,
    }

    result = ground_query(text="people", state=state, persist_instances=True)

    assert result["nodes"]
    assert result["highlight_regions"]
    assert state["instance_store"]["instances"]
    assert state["instance_graph"]["nodes"]
    assert state["scene_graph"]["nodes"]
    assert state["instance_store"]["instances"][0]["label"] == "person"
