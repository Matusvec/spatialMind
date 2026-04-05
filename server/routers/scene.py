"""Instance-graph build endpoint.

Materializes a lightweight scene graph from grounded query instances.
Dense query-time grounding is primary; this endpoint only lifts the
cached instances into graph form.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request/Response models ---


class SceneBuildRequest(BaseModel):
    """Request body for POST /scene/build."""

    query: str = "instances"
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
    """Metadata about the instance graph build."""

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
    """Build the instance graph from cached grounded instances."""
    from server.main import get_app_state
    from server.services.instance_cache import build_instance_graph, ensure_instance_store

    state = get_app_state()

    if not state.get("gaussian_store") or not state["gaussian_store"].is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Gaussian store not loaded. Check /health for status.",
        )

    if not request.force and state.get("instance_graph") is not None:
        logger.info("Returning cached instance graph")
        return state["instance_graph"]

    store = ensure_instance_store(state)
    graph = build_instance_graph(store, state["gaussian_store"])
    metadata = graph.setdefault("metadata", {})
    metadata.update(
        {
            "gaussian_count": int(state["gaussian_store"].count),
            "selected_count": sum(len(node.get("gaussian_indices", [])) for node in graph.get("nodes", [])),
            "cluster_count": len(graph.get("nodes", [])),
            "edge_count": len(graph.get("edges", [])),
            "hierarchy_count": len(graph.get("hierarchy", [])),
            "query": request.query,
            "k": request.k,
        }
    )
    state["instance_graph"] = graph
    state["scene_graph"] = graph
    return graph
