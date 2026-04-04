"""Tests for WalkerBase — BFS traversal with visited-set cycle prevention.

Uses plain dict scene graphs matching the Phase 3 schema.
"""

import pytest

from server.services.walker_base import WalkerBase


def _make_node(node_id: str, label: str = "") -> dict:
    """Create a minimal scene graph node."""
    return {
        "id": node_id,
        "label": label or f"object_{node_id}",
        "centroid": [0.0, 0.0, 0.0],
        "bbox": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "gaussian_indices": [0],
        "confidence": 0.9,
    }


def _make_edge(from_id: str, to_id: str) -> dict:
    """Create a minimal scene graph edge."""
    return {
        "from": from_id,
        "to": to_id,
        "relation": "next_to",
        "confidence": 0.8,
        "distance": 1.0,
    }


LINEAR_GRAPH = {
    "nodes": [_make_node("A"), _make_node("B"), _make_node("C")],
    "edges": [_make_edge("A", "B"), _make_edge("B", "C")],
    "hierarchy": [],
    "metadata": {"cluster_count": 3, "edge_count": 2},
}

CYCLIC_GRAPH = {
    "nodes": [_make_node("A"), _make_node("B"), _make_node("C")],
    "edges": [
        _make_edge("A", "B"),
        _make_edge("B", "C"),
        _make_edge("C", "A"),
    ],
    "hierarchy": [],
    "metadata": {"cluster_count": 3, "edge_count": 3},
}


class CollectorWalker(WalkerBase):
    """Test walker that collects visited node IDs."""

    def on_node(self, node: dict, depth: int) -> bool:
        """Append node id to results."""
        self.results.append(node["id"])
        return True


def test_linear_graph_visits_all_nodes():
    """Test 1: Traverses A->B->C visiting all 3 nodes."""
    walker = CollectorWalker(LINEAR_GRAPH)
    result = walker.traverse("A")

    assert set(result) == {"A", "B", "C"}
    assert len(result) == 3


def test_cyclic_graph_visits_each_once():
    """Test 2: Cyclic graph (A->B->C->A) visits each node exactly once."""
    walker = CollectorWalker(CYCLIC_GRAPH)
    result = walker.traverse("A")

    assert len(result) == 3
    assert set(result) == {"A", "B", "C"}
    # No duplicates
    assert len(result) == len(set(result))


def test_max_depth_limits_traversal():
    """Test 3: max_depth=1 stops after immediate neighbors only."""
    walker = CollectorWalker(LINEAR_GRAPH, max_depth=1)
    result = walker.traverse("A")

    # A (depth 0) and B (depth 1) — C is depth 2, excluded
    assert "A" in result
    assert "B" in result
    assert "C" not in result


def test_get_neighbors_returns_connected():
    """Test 4: get_neighbors returns connected node IDs."""
    walker = WalkerBase(LINEAR_GRAPH)

    neighbors_a = walker.get_neighbors("A")
    assert "B" in neighbors_a

    neighbors_b = walker.get_neighbors("B")
    assert "A" in neighbors_b
    assert "C" in neighbors_b


def test_get_node_returns_dict():
    """Test 5: get_node returns the node dict by ID."""
    walker = WalkerBase(LINEAR_GRAPH)

    node_a = walker.get_node("A")
    assert node_a is not None
    assert node_a["id"] == "A"

    missing = walker.get_node("Z")
    assert missing is None


def test_subclass_on_node_collects_results():
    """Test 6: Subclass overriding on_node collects custom results."""

    class LabelCollector(WalkerBase):
        def on_node(self, node: dict, depth: int) -> bool:
            self.results.append(node["label"])
            return True

    walker = LabelCollector(LINEAR_GRAPH)
    result = walker.traverse("A")

    assert len(result) == 3
    assert all(label.startswith("object_") for label in result)


def test_traverse_all_handles_disconnected():
    """Bonus: traverse_all visits disconnected components."""
    disconnected = {
        "nodes": [
            _make_node("A"),
            _make_node("B"),
            _make_node("C"),
            _make_node("D"),
        ],
        "edges": [_make_edge("A", "B")],  # C and D are disconnected
        "hierarchy": [],
        "metadata": {"cluster_count": 4, "edge_count": 1},
    }
    walker = CollectorWalker(disconnected)
    result = walker.traverse_all()

    assert set(result) == {"A", "B", "C", "D"}


def test_disengage_stops_traversal():
    """Bonus: on_node returning False stops traversal (disengage)."""

    class StopAtB(WalkerBase):
        def on_node(self, node: dict, depth: int) -> bool:
            self.results.append(node["id"])
            return node["id"] != "B"  # disengage when hitting B

    walker = StopAtB(LINEAR_GRAPH)
    result = walker.traverse("A")

    assert "A" in result
    assert "B" in result
    assert "C" not in result
