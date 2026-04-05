"""Walker endpoints for cached-instance graph exploration and dense querying."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.services.exploration_walker import ExplorationWalker

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
    """Explore the cached instance graph and generate an object catalog.

    Creates an ExplorationWalker that traverses all grounded instances,
    generates a human-readable catalog with positions and relationships,
    and optionally writes to Backboard memory.
    """
    from server.main import get_app_state

    state = get_app_state()
    scene_graph = state.get("instance_graph") or state.get("scene_graph")

    if scene_graph is None:
        raise HTTPException(
            status_code=503,
            detail="No grounded instances exist yet. Query the scene first.",
        )

    # Return cached result if available and not forced
    if not request.force and state.get("exploration_catalog") is not None:
        logger.info("Returning cached exploration catalog")
        return ExploreResponse(**state["exploration_catalog"])

    # Create and run walker
    walker = ExplorationWalker(
        scene_graph=scene_graph,
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
    """Ground a query directly against the dense feature field."""
    from server.main import get_app_state
    from server.services.dense_query import ground_query

    state = get_app_state()

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query must be non-empty.",
        )

    result = ground_query(text=request.query, state=state, persist_instances=True)
    return QueryResponse(
        answer=result.get("answer", ""),
        query=request.query,
        matched_nodes=[
            MatchedNode(
                id=node.get("id", ""),
                label=node.get("label", ""),
                centroid=node.get("centroid", [0.0, 0.0, 0.0]),
            )
            for node in result.get("nodes", [])
        ],
        highlight_indices=result.get("highlight_indices", []),
        scene_id=request.scene_id,
    )
