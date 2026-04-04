"""Walker endpoints for scene graph traversal agents.

Provides POST /walker/explore for autonomous scene cataloging via ExplorationWalker
and POST /walker/query for natural language scene queries via QueryWalker.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.services.exploration_walker import ExplorationWalker
from server.services.query_walker import QueryWalker

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request/Response models ---


class ExploreRequest(BaseModel):
    """Request body for POST /walker/explore."""

    scene_id: str = "default"
    force: bool = False


class ExploreResponse(BaseModel):
    """Response from POST /walker/explore."""

    catalog: str
    object_count: int
    scene_id: str


# --- Endpoint ---


@router.post("/walker/explore", response_model=ExploreResponse)
async def explore_scene(request: ExploreRequest):
    """Explore the scene graph and generate an object catalog.

    Creates an ExplorationWalker that traverses all nodes, generates a
    human-readable catalog with positions and relationships, and optionally
    writes to Backboard memory. Result is cached; use force=true to re-explore.
    """
    from server.main import get_app_state

    state = get_app_state()

    # Guard: scene graph must exist
    if state.get("scene_graph") is None:
        raise HTTPException(
            status_code=503,
            detail="Scene graph not built. Call POST /scene/build first.",
        )

    # Return cached result if available and not forced
    if not request.force and state.get("exploration_catalog") is not None:
        logger.info("Returning cached exploration catalog")
        return ExploreResponse(**state["exploration_catalog"])

    # Create and run walker
    walker = ExplorationWalker(
        scene_graph=state["scene_graph"],
        memory_service=state.get("memory_service"),
        scene_id=request.scene_id,
    )
    result = await walker.run()

    # Cache result
    state["exploration_catalog"] = result

    return ExploreResponse(**result)


# --- Query models ---


class QueryRequest(BaseModel):
    """Request body for POST /walker/query."""

    query: str
    scene_id: str = "default"


class MatchedNode(BaseModel):
    """A scene graph node matched by a query."""

    id: str
    label: str
    centroid: list[float]


class QueryResponse(BaseModel):
    """Response from POST /walker/query."""

    answer: str
    query: str
    matched_nodes: list[MatchedNode]
    highlight_indices: list[int]
    scene_id: str


# --- Query endpoint ---


@router.post("/walker/query", response_model=QueryResponse)
async def query_scene(request: QueryRequest):
    """Query the scene graph with natural language.

    Creates a QueryWalker that traverses all nodes, finds relevant objects
    by keyword matching, builds spatial context, and routes through Backboard
    for LLM-powered natural language answers. Multi-turn context is maintained
    via scene_id (same scene_id = same Backboard thread, BB-05).
    """
    from server.main import get_app_state

    state = get_app_state()

    # Guard: scene graph must exist
    if state.get("scene_graph") is None:
        raise HTTPException(
            status_code=503,
            detail="Scene graph not built. Call POST /scene/build first.",
        )

    # Guard: query must be non-empty
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query must be non-empty.",
        )

    # Create and run walker
    walker = QueryWalker(
        scene_graph=state["scene_graph"],
        query=request.query,
        memory_service=state.get("memory_service"),
        scene_id=request.scene_id,
    )
    result = await walker.run()

    return QueryResponse(**result)
