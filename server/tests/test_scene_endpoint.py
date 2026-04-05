"""Tests for the POST /scene/build endpoint.

Integration tests verifying the endpoint materializes an instance graph from
grounded instances and caches the result.
"""

from unittest.mock import patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from server.main import app, get_app_state


# --- Test fixtures ---


class MockCLIPEncoder:
    """Mock CLIP encoder for endpoint tests."""

    def encode_text(self, text: str) -> np.ndarray:
        emb = np.zeros(512, dtype=np.float32)
        emb[0] = 1.0
        return emb


class MockGaussianStore:
    """Mock store with 3 well-separated clusters (100 Gaussians)."""

    def __init__(self):
        rng = np.random.RandomState(42)
        positions = []
        embeddings = []

        for _ in range(40):
            positions.append(rng.randn(3).astype(np.float32) * 0.3)
            emb = rng.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        for _ in range(30):
            pos = rng.randn(3).astype(np.float32) * 0.3
            pos[0] += 10.0
            positions.append(pos)
            emb = rng.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        for _ in range(30):
            pos = rng.randn(3).astype(np.float32) * 0.3
            pos[1] += 5.0
            positions.append(pos)
            emb = rng.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        self.positions = np.array(positions, dtype=np.float32)
        self.decoded_embeddings = np.array(embeddings, dtype=np.float32)
        self.count = 100

    @property
    def is_loaded(self) -> bool:
        return True


class UnloadedGaussianStore:
    """Mock store that is not loaded."""

    positions = None
    decoded_embeddings = None
    count = 0

    @property
    def is_loaded(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def reset_app_state():
    """Reset app state before each test."""
    state = get_app_state()
    state["gaussian_store"] = MockGaussianStore()
    state["clip_encoder"] = MockCLIPEncoder()
    state["scene_graph"] = None
    state["instance_graph"] = None
    state["instance_store"] = {
        "instances": [
            {
                "id": "inst_000",
                "label": "chair",
                "aliases": ["chair"],
                "alias_counts": {"chair": 1},
                "source_queries": ["chair"],
                "gaussian_indices": list(range(40)),
                "centroid": [0.0, 0.0, 0.0],
                "bbox": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                "radius": 0.5,
                "confidence": 0.9,
                "match_score": 0.9,
                "semantic_level": 0,
                "query_count": 1,
            }
        ],
        "next_id": 1,
        "query_history": ["chair"],
    }
    yield
    state["scene_graph"] = None
    state["instance_graph"] = None


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --- Tests ---


@pytest.mark.asyncio
async def test_scene_build_returns_200_with_required_keys(client):
    """POST /scene/build returns 200 with nodes, edges, hierarchy, metadata."""
    resp = await client.post("/scene/build", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert "hierarchy" in data
    assert "metadata" in data


@pytest.mark.asyncio
async def test_scene_build_custom_query_and_k(client):
    """POST /scene/build keeps metadata from request even though instances are cached."""
    resp = await client.post("/scene/build", json={"query": "chairs", "k": 50})
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["query"] == "chairs"
    assert data["metadata"]["k"] == 50
    assert data["metadata"]["selected_count"] > 0


@pytest.mark.asyncio
async def test_scene_build_503_when_store_not_loaded(client):
    """POST /scene/build returns 503 when gaussian_store is not loaded."""
    state = get_app_state()
    state["gaussian_store"] = UnloadedGaussianStore()
    resp = await client.post("/scene/build", json={})
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_metadata_cluster_count_matches_nodes(client):
    """Response metadata.cluster_count matches length of nodes list."""
    resp = await client.post("/scene/build", json={})
    data = resp.json()
    assert data["metadata"]["cluster_count"] == len(data["nodes"])


@pytest.mark.asyncio
async def test_scene_graph_cached_in_app_state(client):
    """Scene graph result is cached in app_state after first build."""
    state = get_app_state()
    assert state["instance_graph"] is None
    resp = await client.post("/scene/build", json={})
    assert resp.status_code == 200
    assert state["instance_graph"] is not None
    assert state["scene_graph"] is not None


@pytest.mark.asyncio
async def test_cached_result_returned_without_rebuild(client):
    """Second POST with force=false returns cached result."""
    # First build
    resp1 = await client.post("/scene/build", json={})
    data1 = resp1.json()

    with patch("server.services.instance_cache.build_instance_graph") as mock_build:
        resp2 = await client.post("/scene/build", json={"force": False})
        data2 = resp2.json()
        mock_build.assert_not_called()

    assert data1 == data2


@pytest.mark.asyncio
async def test_force_true_rebuilds(client):
    """POST /scene/build with force=true rebuilds even if cached."""
    # First build
    await client.post("/scene/build", json={})

    state = get_app_state()
    assert state["instance_graph"] is not None

    # Force rebuild
    with patch("server.services.instance_cache.build_instance_graph") as mock_build:
        mock_build.return_value = {
            "nodes": [], "edges": [], "hierarchy": [],
            "metadata": {
                "gaussian_count": 0, "selected_count": 0,
                "cluster_count": 0, "edge_count": 0,
                "hierarchy_count": 0, "query": "objects", "k": 5000,
            },
        }
        resp = await client.post("/scene/build", json={"force": True})
        mock_build.assert_called_once()
    assert resp.status_code == 200
