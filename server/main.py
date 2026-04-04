"""SpatialMind Semantic Query Server.

FastAPI application that loads LangSplat artifacts (PLY + autoencoder),
encodes text queries via CLIP, and returns top-K matching Gaussian indices
with cosine similarity scores.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import Settings
from server.models.autoencoder import load_autoencoder
from server.routers import clip, health, scene, walker
from server.services.clip_encoder import CLIPEncoder
from server.services.gaussian_store import GaussianStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Application state — populated during startup
_app_state: dict = {
    "config": None,
    "autoencoder": None,
    "gaussian_store": None,
    "clip_encoder": None,
    "scene_graph": None,
    "memory_service": None,
    "exploration_catalog": None,
}


def get_app_state() -> dict:
    """Get the application state dict. Used by routers."""
    return _app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load artifacts at startup, cleanup on shutdown."""
    start_time = time.time()
    config = Settings()
    _app_state["config"] = config

    device = config.device
    logger.info("Device: %s", device)
    logger.info("Scene directory: %s", config.scene_dir)

    # --- Load autoencoder ---
    autoencoder = None
    ae_candidates = [
        os.path.join(config.scene_dir, "artifacts", "autoencoder.pth"),
        os.path.join(config.scene_dir, "ae_ckpt", "best_ckpt.pth"),
    ]
    for ae_path in ae_candidates:
        if os.path.exists(ae_path):
            try:
                autoencoder = load_autoencoder(ae_path, device)
                logger.info("Autoencoder loaded from: %s", ae_path)
                break
            except Exception as exc:
                logger.warning(
                    "Failed to load autoencoder from %s: %s", ae_path, exc
                )
    if autoencoder is None:
        logger.warning(
            "Autoencoder not found. Checked: %s", ae_candidates
        )
    _app_state["autoencoder"] = autoencoder

    # --- Load Gaussian store ---
    gaussian_store = GaussianStore(config)
    ply_path = os.path.join(
        config.scene_dir, "artifacts", "point_cloud.ply"
    )
    if os.path.exists(ply_path):
        try:
            gaussian_store.load_ply(ply_path)
            logger.info(
                "PLY loaded: %d Gaussians", gaussian_store.count
            )
        except Exception as exc:
            logger.warning("Failed to load PLY from %s: %s", ply_path, exc)
    else:
        logger.warning("PLY file not found: %s", ply_path)
    _app_state["gaussian_store"] = gaussian_store

    # --- Decode all Gaussians to 512-dim ---
    if autoencoder and gaussian_store.latent_features is not None:
        try:
            gaussian_store.decode_all(autoencoder, device)
            logger.info(
                "Decoded %d Gaussians to 512-dim", gaussian_store.count
            )
        except Exception as exc:
            logger.warning("Failed to decode Gaussians: %s", exc)
    else:
        logger.warning(
            "Skipping Gaussian decoding: "
            "autoencoder=%s, latent_features=%s",
            autoencoder is not None,
            gaussian_store.latent_features is not None,
        )

    # --- Load CLIP encoder ---
    try:
        clip_encoder = CLIPEncoder(
            config.clip_model, config.clip_pretrained, device
        )
        _app_state["clip_encoder"] = clip_encoder
        logger.info("CLIP encoder ready: %s", config.clip_model)
    except Exception as exc:
        logger.warning("Failed to load CLIP encoder: %s", exc)
        _app_state["clip_encoder"] = None

    # --- Initialize Backboard memory service ---
    backboard_key = config.backboard_api_key
    if backboard_key:
        try:
            from server.services.backboard_client import SpatialMemoryService

            memory_service = SpatialMemoryService(backboard_key)
            await memory_service.initialize()
            _app_state["memory_service"] = memory_service
            logger.info("Backboard memory service initialized")
        except Exception as exc:
            logger.warning("Failed to initialize Backboard: %s", exc)
    else:
        logger.info("BACKBOARD_API_KEY not set — memory service disabled")

    elapsed = time.time() - start_time
    logger.info(
        "Startup complete in %.1fs | Gaussians: %d | Device: %s | CLIP: %s",
        elapsed,
        gaussian_store.count,
        device,
        config.clip_model,
    )

    yield

    # Shutdown cleanup
    logger.info("Server shutting down")


app = FastAPI(
    title="SpatialMind Semantic Query Server",
    description="Semantic query pipeline for LangSplat 3D scenes",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware — add first (per PITFALLS.md Gotcha 2)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(clip.router)
app.include_router(scene.router)
app.include_router(walker.router)


if __name__ == "__main__":
    import uvicorn

    config = Settings()
    uvicorn.run(
        "server.main:app",
        host=config.host,
        port=config.port,
        reload=True,
    )
