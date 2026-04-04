"""Tests for ExplorationWalker — catalog generation and Backboard persistence.

Tests cover: full traversal, catalog formatting, spatial relationships,
empty graph handling, disconnected components, and Backboard write via store_catalog.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from server.services.exploration_walker import ExplorationWalker


# --- Test fixtures ---


def _make_graph(nodes: list[dict], edges: list[dict]) -> dict:
    """Build a scene graph dict from node and edge lists."""
    return {"nodes": nodes, "edges": edges}


def _node(id: str, label: str, centroid: list[float], confidence: float) -> dict:
    return {
        "id": id,
        "label": label,
        "centroid": centroid,
        "bbox": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "gaussian_indices": [0],
        "confidence": confidence,
    }


FIVE_NODE_GRAPH = _make_graph(
    nodes=[
        _node("obj_000", "object_0", [1.2, 0.5, -0.8], 0.85),
        _node("obj_001", "object_1", [2.0, 0.3, 1.1], 0.72),
        _node("obj_002", "object_2", [0.5, 1.0, 0.0], 0.90),
        _node("obj_003", "object_3", [5.0, 0.0, 5.0], 0.60),
        _node("obj_004", "object_4", [5.5, 0.2, 4.8], 0.65),
    ],
    edges=[
        {"from": "obj_000", "to": "obj_001", "relation": "next_to", "confidence": 0.75, "distance": 1.2},
        {"from": "obj_000", "to": "obj_002", "relation": "on_top_of", "confidence": 0.90, "distance": 0.8},
        {"from": "obj_003", "to": "obj_004", "relation": "next_to", "confidence": 0.80, "distance": 0.5},
    ],
)
"""5-node graph with 2 disconnected components: {obj_000, obj_001, obj_002} and {obj_003, obj_004}."""

THREE_NODE_GRAPH = _make_graph(
    nodes=[
        _node("obj_000", "object_0", [1.0, 2.0, 3.0], 0.80),
        _node("obj_001", "object_1", [4.0, 5.0, 6.0], 0.70),
        _node("obj_002", "object_2", [7.0, 8.0, 9.0], 0.60),
    ],
    edges=[
        {"from": "obj_000", "to": "obj_001", "relation": "next_to", "confidence": 0.75, "distance": 1.0},
        {"from": "obj_001", "to": "obj_002", "relation": "on_top_of", "confidence": 0.85, "distance": 0.5},
    ],
)

EMPTY_GRAPH = _make_graph(nodes=[], edges=[])


# --- Tests ---


def test_traverse_all_visits_every_node_in_5_node_graph():
    """Test 1: ExplorationWalker.traverse_all() visits every node in a 5-node graph."""
    walker = ExplorationWalker(FIVE_NODE_GRAPH)
    walker.traverse_all()
    assert len(walker.catalog_entries) == 5
    visited_ids = {entry["id"] for entry in walker.catalog_entries}
    assert visited_ids == {"obj_000", "obj_001", "obj_002", "obj_003", "obj_004"}


def test_format_catalog_produces_multiline_with_positions_and_relationships():
    """Test 2: format_catalog() produces multi-line string with object names, positions, relationships."""
    walker = ExplorationWalker(THREE_NODE_GRAPH)
    walker.traverse_all()
    catalog = walker.format_catalog()

    # Header with count
    assert "3 objects" in catalog

    # Each object appears with label, id, and position
    assert "object_0 (obj_000)" in catalog
    assert "object_1 (obj_001)" in catalog
    assert "object_2 (obj_002)" in catalog

    # Positions formatted as (x, y, z) with 1 decimal
    assert "(1.0, 2.0, 3.0)" in catalog
    assert "(4.0, 5.0, 6.0)" in catalog

    # Confidence appears
    assert "0.80" in catalog or "0.8" in catalog


def test_catalog_includes_spatial_relationships():
    """Test 3: Catalog includes spatial relationships between objects."""
    walker = ExplorationWalker(THREE_NODE_GRAPH)
    walker.traverse_all()
    catalog = walker.format_catalog()

    # Relationships should reference target labels
    assert "next_to" in catalog
    assert "on_top_of" in catalog
    # Should reference object labels, not just IDs
    assert "object_1" in catalog


def test_empty_graph_returns_empty_catalog():
    """Test 4: ExplorationWalker handles empty graph (0 nodes) gracefully."""
    walker = ExplorationWalker(EMPTY_GRAPH)
    walker.traverse_all()
    catalog = walker.format_catalog()
    assert "0 objects" in catalog
    assert len(walker.catalog_entries) == 0


def test_disconnected_components_all_visited():
    """Test 5: ExplorationWalker handles disconnected components — visits all."""
    walker = ExplorationWalker(FIVE_NODE_GRAPH)
    walker.traverse_all()

    # Both components visited
    visited_ids = {entry["id"] for entry in walker.catalog_entries}
    assert "obj_000" in visited_ids  # Component 1
    assert "obj_003" in visited_ids  # Component 2
    assert "obj_004" in visited_ids  # Component 2
    assert len(walker.catalog_entries) == 5


@pytest.mark.asyncio
async def test_run_calls_store_catalog_on_backboard():
    """Test 6: run() calls memory_service.store_catalog() with the generated catalog."""
    mock_memory = MagicMock()
    mock_memory.store_catalog = AsyncMock()

    walker = ExplorationWalker(
        FIVE_NODE_GRAPH,
        memory_service=mock_memory,
        scene_id="test_scene",
    )
    result = await walker.run()

    # store_catalog called once
    mock_memory.store_catalog.assert_called_once()
    call_args = mock_memory.store_catalog.call_args
    assert call_args.args[0] == "test_scene"
    catalog_text = call_args.args[1]
    assert "5 objects" in catalog_text

    # Return value structure
    assert result["object_count"] == 5
    assert result["scene_id"] == "test_scene"
    assert "catalog" in result
    assert len(result["catalog"]) > 0


@pytest.mark.asyncio
async def test_run_without_memory_service_succeeds():
    """run() works without memory_service (None) — no store_catalog call."""
    walker = ExplorationWalker(FIVE_NODE_GRAPH, memory_service=None)
    result = await walker.run()
    assert result["object_count"] == 5
    assert result["catalog"] != ""
