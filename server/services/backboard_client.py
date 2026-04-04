"""Backboard memory service for persistent spatial reasoning.

Wraps BackboardClient with assistant/thread/message lifecycle management.
Each scene gets its own Backboard thread for persistent spatial memory.
"""

import logging

logger = logging.getLogger(__name__)

# Graceful import — server doesn't crash if backboard-sdk not installed
try:
    from backboard import BackboardClient

    BACKBOARD_AVAILABLE = True
except ImportError:
    BackboardClient = None  # type: ignore[assignment, misc]
    BACKBOARD_AVAILABLE = False
    logger.warning(
        "backboard-sdk not installed. SpatialMemoryService will raise "
        "RuntimeError on method calls. Install with: pip install backboard-sdk"
    )

SPATIAL_SYSTEM_PROMPT = """You are SpatialMind, an AI that understands 3D spaces.
You have access to a spatial knowledge graph of objects in a scanned room.
When answering questions:
- Reference specific objects by name and their spatial relationships
- Use directional language (left of, above, near, facing)
- If you remember objects from previous conversations, reference that memory
- Be concise but spatially precise"""


class SpatialMemoryService:
    """Manages Backboard assistant and threads for spatial memory.

    Provides persistent conversational memory per scene via Backboard threads.
    Each scene_id maps to a unique thread, enabling cross-session recall.
    """

    def __init__(self, api_key: str):
        """Initialize with Backboard API key.

        Args:
            api_key: Backboard API key for authentication.
        """
        if not BACKBOARD_AVAILABLE:
            raise RuntimeError(
                "backboard-sdk is not installed. "
                "Install with: pip install backboard-sdk"
            )
        self.client = BackboardClient(api_key=api_key)
        self.assistant_id: str | None = None
        self.scene_threads: dict[str, str] = {}

    async def initialize(self, system_prompt: str | None = None) -> None:
        """Create a Backboard assistant with spatial reasoning prompt.

        Args:
            system_prompt: Custom system prompt. Uses SPATIAL_SYSTEM_PROMPT default.
        """
        prompt = system_prompt or SPATIAL_SYSTEM_PROMPT
        assistant = await self.client.create_assistant(
            name="SpatialMind",
            system_prompt=prompt,
        )
        self.assistant_id = assistant.assistant_id
        logger.info("Backboard assistant created: %s", self.assistant_id)

    async def get_or_create_thread(self, scene_id: str) -> str:
        """Get existing thread for scene or create a new one.

        Args:
            scene_id: Unique identifier for the scene.

        Returns:
            Thread ID for the scene.
        """
        if scene_id in self.scene_threads:
            return self.scene_threads[scene_id]

        thread = await self.client.create_thread(self.assistant_id)
        self.scene_threads[scene_id] = thread.thread_id
        logger.info(
            "Backboard thread created for scene %s: %s",
            scene_id,
            thread.thread_id,
        )
        return thread.thread_id

    async def store_catalog(self, scene_id: str, catalog: str) -> None:
        """Store object catalog in scene's Backboard thread.

        Args:
            scene_id: Scene to store catalog for.
            catalog: Text catalog of scene objects.
        """
        thread_id = await self.get_or_create_thread(scene_id)
        await self.client.add_message(
            thread_id=thread_id,
            content=catalog,
            stream=False,
        )
        logger.info("Catalog stored for scene %s", scene_id)

    async def query_with_context(
        self, scene_id: str, query: str, spatial_context: str
    ) -> str:
        """Send enriched spatial query and return LLM response.

        Args:
            scene_id: Scene to query.
            query: User's natural language question.
            spatial_context: Spatial graph context to enrich the query.

        Returns:
            LLM response content string.
        """
        thread_id = await self.get_or_create_thread(scene_id)
        enriched = (
            f"Spatial context: {spatial_context}\n\nUser question: {query}"
        )
        response = await self.client.add_message(
            thread_id=thread_id,
            content=enriched,
            stream=False,
        )
        return response.content
