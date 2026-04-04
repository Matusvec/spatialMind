"""Base walker class for scene graph traversal.

Implements BFS traversal with visited-set cycle prevention, max_depth safety,
and disengage pattern. Subclass and override on_node() to collect results.
"""

import logging
from collections import deque

logger = logging.getLogger(__name__)


class WalkerBase:
    """Base class for walkers that traverse the Phase 3 scene graph dict.

    Provides BFS traversal with:
    - Visited-set to prevent infinite loops on cyclic graphs
    - max_depth to limit traversal depth
    - on_node() hook for subclasses to collect results
    - Disengage pattern (on_node returns False to stop)
    """

    def __init__(self, scene_graph: dict, max_depth: int = 100):
        """Initialize walker with a scene graph.

        Args:
            scene_graph: Dict with 'nodes' and 'edges' from build_scene_graph().
            max_depth: Maximum BFS depth. Default 100 as safety net.
        """
        self.nodes: dict[str, dict] = {
            n["id"]: n for n in scene_graph["nodes"]
        }
        self.edges: list[dict] = scene_graph["edges"]
        self.adjacency: dict[str, list[str]] = {}
        self.visited: set[str] = set()
        self.max_depth: int = max_depth
        self.results: list = []

        # Build undirected adjacency list from edges
        for edge in self.edges:
            src, dst = edge["from"], edge["to"]
            self.adjacency.setdefault(src, []).append(dst)
            self.adjacency.setdefault(dst, []).append(src)

    def get_node(self, node_id: str) -> dict | None:
        """Get node dict by ID.

        Args:
            node_id: Node identifier.

        Returns:
            Node dict or None if not found.
        """
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get connected node IDs.

        Args:
            node_id: Node identifier.

        Returns:
            List of neighbor node IDs.
        """
        return self.adjacency.get(node_id, [])

    def get_edge(self, from_id: str, to_id: str) -> dict | None:
        """Find edge between two nodes.

        Args:
            from_id: Source node ID.
            to_id: Target node ID.

        Returns:
            Edge dict or None if not found.
        """
        for edge in self.edges:
            if (edge["from"] == from_id and edge["to"] == to_id) or (
                edge["from"] == to_id and edge["to"] == from_id
            ):
                return edge
        return None

    def on_node(self, node: dict, depth: int) -> bool:
        """Called when visiting a node. Override in subclasses.

        Args:
            node: The node dict being visited.
            depth: Current BFS depth.

        Returns:
            True to continue traversal, False to disengage.
        """
        return True

    def traverse(self, start_id: str | None = None) -> list:
        """BFS traversal from start node.

        Args:
            start_id: Node ID to start from. Defaults to first node.

        Returns:
            self.results list populated by on_node() calls.
        """
        if start_id is None:
            if not self.nodes:
                return self.results
            start_id = next(iter(self.nodes))

        queue: deque[tuple[str, int]] = deque([(start_id, 0)])

        while queue:
            node_id, depth = queue.popleft()

            if node_id in self.visited or depth > self.max_depth:
                continue

            self.visited.add(node_id)
            node = self.get_node(node_id)
            if node is None:
                continue

            should_continue = self.on_node(node, depth)
            if not should_continue:
                break

            for neighbor_id in self.get_neighbors(node_id):
                if neighbor_id not in self.visited:
                    queue.append((neighbor_id, depth + 1))

        return self.results

    def traverse_all(self) -> list:
        """Traverse all connected components.

        Visits every node in the graph, handling disconnected components
        by starting new traversals from unvisited nodes.

        Returns:
            self.results list populated by on_node() calls.
        """
        for node_id in self.nodes:
            if node_id not in self.visited:
                self.traverse(node_id)
        return self.results
