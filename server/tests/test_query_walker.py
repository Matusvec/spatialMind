"""Tests for QueryWalker scene graph traversal and NL answer generation.

Tests keyword matching, spatial context building, Backboard LLM routing,
fallback without Backboard, highlight indices, and multi-turn context.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from server.services.query_walker import QueryWalker


# --- Test fixtures ---


def _build_test_graph():
    """Build a test scene graph with 5 objects and 3 edges."""
    nodes = [
        {
            "id": "obj_000",
            "label": "table",
            "centroid": [1.2, 0.5, -0.8],
            "bbox": [0.0, 0.0, 0.0, 2.0, 1.0, 1.0],
            "gaussian_indices": [10, 11, 12],
            "confidence": 0.90,
        },
        {
            "id": "obj_001",
            "label": "chair",
            "centroid": [2.0, 0.4, -0.5],
            "bbox": [1.5, 0.0, -1.0, 2.5, 0.8, 0.0],
            "gaussian_indices": [20, 21],
            "confidence": 0.85,
        },
        {
            "id": "obj_002",
            "label": "lamp",
            "centroid": [5.0, 1.5, 3.0],
            "bbox": [4.5, 1.0, 2.5, 5.5, 2.0, 3.5],
            "gaussian_indices": [30, 31, 32, 33],
            "confidence": 0.80,
        },
        {
            "id": "obj_003",
            "label": "book",
            "centroid": [1.3, 1.0, -0.7],
            "bbox": [1.1, 0.9, -0.9, 1.5, 1.1, -0.5],
            "gaussian_indices": [40],
            "confidence": 0.75,
        },
        {
            "id": "obj_004",
            "label": "window",
            "centroid": [5.5, 2.0, 3.5],
            "bbox": [5.0, 1.5, 3.0, 6.0, 2.5, 4.0],
            "gaussian_indices": [50, 51],
            "confidence": 0.70,
        },
    ]
    edges = [
        {
            "from": "obj_000",
            "to": "obj_001",
            "relation": "next_to",
            "confidence": 0.75,
            "distance": 1.2,
        },
        {
            "from": "obj_003",
            "to": "obj_000",
            "relation": "on_top_of",
            "confidence": 0.88,
            "distance": 0.5,
        },
        {
            "from": "obj_002",
            "to": "obj_004",
            "relation": "next_to",
            "confidence": 0.65,
            "distance": 0.8,
        },
    ]
    return {"nodes": nodes, "edges": edges}


# --- Test 1: Keyword relevance matching ---


def test_query_walker_marks_relevant_node():
    """QueryWalker with query 'table' marks table node as relevant."""
    graph = _build_test_graph()
    walker = QueryWalker(scene_graph=graph, query="table")
    walker.traverse_all()

    matched_labels = [n["label"] for n in walker.matched_nodes]
    assert "table" in matched_labels
    assert "chair" not in matched_labels
    assert "lamp" not in matched_labels


# --- Test 2: Spatial context for relevant nodes ---


def test_query_walker_collects_spatial_context():
    """QueryWalker collects spatial context for relevant nodes with relationships."""
    graph = _build_test_graph()
    walker = QueryWalker(scene_graph=graph, query="table")
    walker.traverse_all()
    context = walker.build_spatial_context()

    assert "table" in context
    assert "obj_000" in context
    assert "1.2" in context  # centroid x
    assert "next_to" in context  # relationship to chair
    assert "chair" in context  # neighbor label


# --- Test 3: build_spatial_context returns multi-line summary ---


def test_build_spatial_context_format():
    """build_spatial_context returns multi-line string with match count header."""
    graph = _build_test_graph()
    walker = QueryWalker(scene_graph=graph, query="table")
    walker.traverse_all()
    context = walker.build_spatial_context()

    lines = context.strip().split("\n")
    assert len(lines) >= 2  # at least header + one node line
    assert "Found" in lines[0]
    assert "1" in lines[0]  # 1 matched object
    assert "table" in lines[0]


# --- Test 4: run() with memory_service calls query_with_context ---


@pytest.mark.asyncio
async def test_run_with_memory_service():
    """QueryWalker.run() with memory_service calls query_with_context."""
    graph = _build_test_graph()
    mock_memory = AsyncMock()
    mock_memory.query_with_context.return_value = (
        "The table is next to the chair with a book on top."
    )

    walker = QueryWalker(
        scene_graph=graph,
        query="table",
        memory_service=mock_memory,
        scene_id="test_scene",
    )
    result = await walker.run()

    mock_memory.query_with_context.assert_called_once()
    call_args = mock_memory.query_with_context.call_args
    # Called with positional args: (scene_id, query, spatial_context)
    assert call_args[0][0] == "test_scene"
    assert result["answer"] == "The table is next to the chair with a book on top."


# --- Test 5: run() without memory_service returns fallback ---


@pytest.mark.asyncio
async def test_run_without_memory_service():
    """QueryWalker.run() without memory_service returns spatial context as answer."""
    graph = _build_test_graph()
    walker = QueryWalker(scene_graph=graph, query="table")
    result = await walker.run()

    assert "table" in result["answer"]
    assert "next_to" in result["answer"]
    assert result["query"] == "table"


# --- Test 6: Returns matched node ids and gaussian_indices ---


@pytest.mark.asyncio
async def test_run_returns_highlight_data():
    """QueryWalker returns matched node ids and gaussian_indices for highlight mask."""
    graph = _build_test_graph()
    walker = QueryWalker(scene_graph=graph, query="table")
    result = await walker.run()

    assert len(result["matched_nodes"]) == 1
    assert result["matched_nodes"][0]["id"] == "obj_000"
    assert result["matched_nodes"][0]["label"] == "table"
    assert result["matched_nodes"][0]["centroid"] == [1.2, 0.5, -0.8]
    # highlight_indices should contain table's gaussian_indices
    assert set([10, 11, 12]).issubset(set(result["highlight_indices"]))


# --- Test 7: Multi-turn reuses same scene_id for Backboard thread ---


@pytest.mark.asyncio
async def test_multi_turn_same_scene_id():
    """Two QueryWalkers with same scene_id pass same scene_id to memory_service (BB-05)."""
    graph = _build_test_graph()
    mock_memory = AsyncMock()
    mock_memory.query_with_context.return_value = "Answer from Backboard."

    # First query
    walker1 = QueryWalker(
        scene_graph=graph,
        query="table",
        memory_service=mock_memory,
        scene_id="room_1",
    )
    await walker1.run()

    # Second query (follow-up) with same scene_id
    walker2 = QueryWalker(
        scene_graph=graph,
        query="chair",
        memory_service=mock_memory,
        scene_id="room_1",
    )
    await walker2.run()

    # Both calls should use the same scene_id
    calls = mock_memory.query_with_context.call_args_list
    assert len(calls) == 2
    # Verify both calls pass "room_1" as scene_id (first positional arg)
    assert calls[0][0][0] == "room_1"
    assert calls[1][0][0] == "room_1"


# --- Test 8: No matches returns appropriate message ---


@pytest.mark.asyncio
async def test_no_matches_returns_empty():
    """QueryWalker with unmatched query returns no-match message."""
    graph = _build_test_graph()
    walker = QueryWalker(scene_graph=graph, query="refrigerator")
    result = await walker.run()

    assert len(result["matched_nodes"]) == 0
    assert len(result["highlight_indices"]) == 0
    assert "No objects matching" in result["answer"]
