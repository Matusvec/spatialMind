"""Health check endpoint for the SpatialMind server."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Return server health with loaded artifact status."""
    from server.main import get_app_state

    state = get_app_state()
    gaussian_store = state["gaussian_store"]
    ply_loaded = gaussian_store.is_loaded if gaussian_store else False
    clip_ready = state["clip_encoder"] is not None
    instance_graph = state.get("instance_graph")
    instance_graph_nodes = (
        len(instance_graph["nodes"])
        if instance_graph and isinstance(instance_graph, dict) and "nodes" in instance_graph
        else 0
    )
    instance_graph_ready = instance_graph_nodes > 0
    pipeline_ready = bool(ply_loaded and clip_ready)

    return {
        "status": "ok",
        "ply_loaded": ply_loaded,
        "autoencoder_initialized": state["autoencoder"] is not None,
        "gaussian_count": (
            gaussian_store.count
            if gaussian_store and gaussian_store.is_loaded
            else 0
        ),
        "clip_model": state["config"].clip_model,
        "clip_ready": clip_ready,
        "scene_graph_ready": instance_graph_ready,
        "scene_graph_nodes": instance_graph_nodes,
        "instance_graph_ready": instance_graph_ready,
        "instance_graph_nodes": instance_graph_nodes,
        "pipeline_ready": pipeline_ready,
        "scene_source": state.get("scene_source", "none"),
    }
