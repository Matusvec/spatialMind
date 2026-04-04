"""ExplorationWalker — autonomous scene cataloging and Backboard persistence.

Traverses all nodes in the scene graph (including disconnected components),
generates a human-readable catalog with positions and spatial relationships,
and writes the catalog to Backboard via SpatialMemoryService.store_catalog().
"""

import logging

from server.services.walker_base import WalkerBase

logger = logging.getLogger(__name__)


class ExplorationWalker(WalkerBase):
    """Walker that catalogs every object in the scene graph.

    Extends WalkerBase to visit all nodes via traverse_all(), collecting
    node info and spatial relationships into a natural language catalog.
    Optionally persists the catalog to Backboard memory.
    """

    def __init__(
        self,
        scene_graph: dict,
        memory_service=None,
        scene_id: str = "default",
    ):
        """Initialize ExplorationWalker.

        Args:
            scene_graph: Dict with 'nodes' and 'edges' from build_scene_graph().
            memory_service: Optional SpatialMemoryService for Backboard persistence.
            scene_id: Identifier for the scene being explored.
        """
        super().__init__(scene_graph)
        self.memory_service = memory_service
        self.scene_id = scene_id
        self.catalog_entries: list[dict] = []

    def on_node(self, node: dict, depth: int) -> bool:
        """Collect node info and relationships for catalog.

        Args:
            node: The node dict being visited.
            depth: Current BFS depth.

        Returns:
            True always — explore everything, never disengage early.
        """
        node_id = node["id"]

        # Find edges connected to this node and build relationships
        relationships = []
        for edge in self.edges:
            if edge["from"] == node_id:
                target = self.get_node(edge["to"])
                target_label = target["label"] if target else edge["to"]
                relationships.append(
                    f"{edge['relation']} {target_label} "
                    f"(conf: {edge['confidence']:.2f})"
                )
            elif edge["to"] == node_id:
                target = self.get_node(edge["from"])
                target_label = target["label"] if target else edge["from"]
                relationships.append(
                    f"{edge['relation']} {target_label} "
                    f"(conf: {edge['confidence']:.2f})"
                )

        self.catalog_entries.append(
            {
                "id": node_id,
                "label": node["label"],
                "centroid": node["centroid"],
                "confidence": node["confidence"],
                "relationships": relationships,
            }
        )

        return True

    def format_catalog(self) -> str:
        """Build natural language catalog from collected entries.

        Returns:
            Multi-line string with object names, positions, and relationships.
        """
        count = len(self.catalog_entries)
        lines = [f"Scene Object Catalog ({count} objects):", ""]

        for entry in self.catalog_entries:
            cx, cy, cz = entry["centroid"]
            lines.append(
                f"- {entry['label']} ({entry['id']}): "
                f"located at ({cx:.1f}, {cy:.1f}, {cz:.1f}), "
                f"confidence {entry['confidence']:.2f}"
            )
            if entry["relationships"]:
                rels = ", ".join(entry["relationships"])
                lines.append(f"  Relationships: {rels}")

        return "\n".join(lines)

    async def run(self) -> dict:
        """Execute full exploration: traverse, catalog, and persist.

        Returns:
            Dict with keys: catalog (str), object_count (int), scene_id (str).
        """
        self.traverse_all()
        catalog = self.format_catalog()

        if self.memory_service is not None:
            try:
                await self.memory_service.store_catalog(
                    self.scene_id, catalog
                )
                logger.info(
                    "Catalog stored to Backboard for scene %s",
                    self.scene_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to store catalog to Backboard: %s", exc
                )

        return {
            "catalog": catalog,
            "object_count": len(self.catalog_entries),
            "scene_id": self.scene_id,
        }
