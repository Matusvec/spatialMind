"""Scene graph build endpoint.

Triggers scene graph construction by orchestrating DBSCAN clustering,
spatial relationship extraction, and hierarchy computation. The result
is cached in app_state so it's built once at scene load, not per-query.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from server.services.graph_builder import build_scene_graph

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request/Response models ---


class SceneBuildRequest(BaseModel):
    """Request body for POST /scene/build."""

    query: str = "objects"
    k: int = 5000
    min_samples: int = 6
    hierarchy_threshold: float = 0.7
    force: bool = False


class SceneNodeResponse(BaseModel):
    """A single object node in the scene graph."""

    id: str
    label: str
    centroid: list[float]
    bbox: list[float]
    gaussian_indices: list[int]
    confidence: float


class SceneEdgeResponse(BaseModel):
    """A spatial relationship edge between two nodes."""

    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    relation: str
    confidence: float
    distance: float

    model_config = {"populate_by_name": True}


class SceneHierarchyResponse(BaseModel):
    """A hierarchy edge showing CLIP similarity between nodes."""

    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    relation: str
    similarity: float

    model_config = {"populate_by_name": True}


class SceneMetadataResponse(BaseModel):
    """Metadata about the scene graph build."""

    gaussian_count: int
    selected_count: int
    cluster_count: int
    edge_count: int
    hierarchy_count: int
    query: str
    k: int


class SceneBuildResponse(BaseModel):
    """Full scene graph response."""

    nodes: list[SceneNodeResponse]
    edges: list[SceneEdgeResponse]
    hierarchy: list[SceneHierarchyResponse]
    metadata: SceneMetadataResponse


# --- Endpoint ---


@router.post("/scene/build", response_model=SceneBuildResponse)
async def build_scene(request: SceneBuildRequest):
    """Build the scene graph from loaded Gaussian data.

    Orchestrates CLIP selection, DBSCAN clustering, spatial relations,
    and hierarchy computation. Result is cached in app_state; subsequent
    calls return the cached result unless force=true.
    """
    from server.main import get_app_state

    state = get_app_state()

    # Guard: Gaussian store must be loaded
    if not state.get("gaussian_store") or not state["gaussian_store"].is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Gaussian store not loaded. Check /health for status.",
        )

    # Guard: CLIP encoder must be initialized
    if not state.get("clip_encoder"):
        raise HTTPException(
            status_code=503,
            detail="CLIP encoder not initialized. Check /health for status.",
        )

    # Return cached result if available and not forced
    if not request.force and state.get("scene_graph") is not None:
        logger.info("Returning cached scene graph")
        return state["scene_graph"]

    # Build scene graph
    result = build_scene_graph(
        gaussian_store=state["gaussian_store"],
        clip_encoder=state["clip_encoder"],
        query=request.query,
        k=request.k,
        min_samples=request.min_samples,
        hierarchy_threshold=request.hierarchy_threshold,
    )

    # Cache result
    state["scene_graph"] = result

    return result
