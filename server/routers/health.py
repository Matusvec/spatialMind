"""Health check endpoint for the SpatialMind server."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Return server health with loaded artifact status."""
    from server.main import get_app_state

    state = get_app_state()
    gaussian_store = state["gaussian_store"]
    return {
        "status": "ok",
        "ply_loaded": gaussian_store.is_loaded if gaussian_store else False,
        "autoencoder_initialized": state["autoencoder"] is not None,
        "gaussian_count": (
            gaussian_store.count
            if gaussian_store and gaussian_store.is_loaded
            else 0
        ),
        "clip_model": state["config"].clip_model,
    }
