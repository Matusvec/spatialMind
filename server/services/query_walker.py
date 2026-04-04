"""QueryWalker for natural language scene graph queries.

Traverses the spatial graph to find nodes relevant to a natural language query,
builds spatial context from matched nodes and their relationships, and routes
through Backboard for LLM-powered answer generation with multi-turn context.
"""

import logging

from server.services.walker_base import WalkerBase

logger = logging.getLogger(__name__)


class QueryWalker(WalkerBase):
    """Walker that answers natural language questions about the scene graph.

    Traverses all nodes, identifies relevant ones via keyword matching,
    builds spatial context, and optionally routes through Backboard for
    LLM-generated natural language answers (BB-04). Falls back to raw
    spatial facts when Backboard is unavailable.
    """

    def __init__(
        self,
        scene_graph: dict,
        query: str,
        memory_service=None,
        scene_id: str = "default",
    ):
        """Initialize QueryWalker with query and optional Backboard service.

        Args:
            scene_graph: Dict with 'nodes' and 'edges' from build_scene_graph().
            query: Natural language question to answer.
            memory_service: Optional SpatialMemoryService for LLM routing.
            scene_id: Scene identifier for Backboard thread reuse (BB-05).
        """
        super().__init__(scene_graph)
        self.query = query.lower()
        self.memory_service = memory_service
        self.scene_id = scene_id
        self.matched_nodes: list[dict] = []
        self.spatial_context_parts: list[str] = []

    def is_relevant(self, node: dict) -> bool:
        """Check if a node is relevant to the query via keyword matching.

        Simple keyword matching: checks if any word in the query appears
        in the node label, or if the query is a substring of the label.

        Args:
            node: Node dict with at least a 'label' key.

        Returns:
            True if the node is relevant to the query.
        """
        label = node.get("label", "").lower()
        # Check if query is a substring of label
        if self.query in label:
            return True
        # Check if any query word appears in label
        query_words = self.query.split()
        for word in query_words:
            if word in label:
                return True
        return False

    def on_node(self, node: dict, depth: int) -> bool:
        """Visit a node: if relevant, collect it and build spatial context.

        Args:
            node: The node dict being visited.
            depth: Current BFS depth.

        Returns:
            True always (continue traversal to find ALL relevant nodes).
        """
        if self.is_relevant(node):
            self.matched_nodes.append(node)

            # Build context line for this node
            cx, cy, cz = node.get("centroid", [0, 0, 0])
            label = node.get("label", "unknown")
            node_id = node.get("id", "?")
            context_line = f"{label} ({node_id}) at ({cx:.1f}, {cy:.1f}, {cz:.1f})"
            self.spatial_context_parts.append(context_line)

            # Add relationship lines for edges connected to this node
            for edge in self.edges:
                if edge["from"] == node_id or edge["to"] == node_id:
                    neighbor_id = (
                        edge["to"] if edge["from"] == node_id else edge["from"]
                    )
                    neighbor = self.get_node(neighbor_id)
                    neighbor_label = (
                        neighbor.get("label", neighbor_id)
                        if neighbor
                        else neighbor_id
                    )
                    conf = edge.get("confidence", 0)
                    rel = edge.get("relation", "related_to")
                    self.spatial_context_parts.append(
                        f"  - {rel} {neighbor_label} (confidence: {conf:.2f})"
                    )

        return True

    def build_spatial_context(self) -> str:
        """Build a multi-line spatial context string from matched nodes.

        Returns:
            Human-readable summary of matched nodes and their relationships,
            or a no-match message if nothing was found.
        """
        if not self.matched_nodes:
            return "No objects matching the query were found in the scene."

        header = (
            f"Found {len(self.matched_nodes)} objects matching "
            f"'{self.query}':"
        )
        return header + "\n" + "\n".join(self.spatial_context_parts)

    async def run(self) -> dict:
        """Execute the query: traverse, build context, generate answer.

        Traverses all nodes, builds spatial context for matched nodes,
        and either routes through Backboard for an LLM answer (BB-04)
        or returns raw spatial facts as a fallback.

        Returns:
            Dict with answer, query, matched_nodes, highlight_indices, scene_id.
        """
        self.traverse_all()
        spatial_context = self.build_spatial_context()

        # Collect gaussian indices for highlight mask
        gaussian_indices: list[int] = []
        for node in self.matched_nodes:
            gaussian_indices.extend(node.get("gaussian_indices", []))

        # Generate answer via Backboard or fallback
        if self.memory_service is not None:
            try:
                answer = await self.memory_service.query_with_context(
                    self.scene_id, self.query, spatial_context
                )
            except Exception as exc:
                logger.warning(
                    "Backboard query failed, falling back to spatial context: %s",
                    exc,
                )
                answer = spatial_context
        else:
            answer = spatial_context

        return {
            "answer": answer,
            "query": self.query,
            "matched_nodes": [
                {
                    "id": n["id"],
                    "label": n["label"],
                    "centroid": n["centroid"],
                }
                for n in self.matched_nodes
            ],
            "highlight_indices": gaussian_indices,
            "scene_id": self.scene_id,
        }
