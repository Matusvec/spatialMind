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

import torch

from server.config import Settings
from server.models.autoencoder import load_autoencoder
from server.routers import adapter, clip, health, scene, walker
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
    "instance_graph": None,
    "instance_store": None,
    "memory_service": None,
    "exploration_catalog": None,
    "scene_source": "none",
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
        os.path.join(config.scene_dir, "autoencoder.pth"),
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

    # --- Load Gaussian store with LangSplat semantic levels ---
    gaussian_store = GaussianStore(config)

    # Try semantic PLY files first (LangSplat output: semantic_0/1/2.ply)
    sem0_path = os.path.join(config.scene_dir, "semantic_0.ply")
    if os.path.exists(sem0_path):
        try:
            # Load sem0 as the primary (has positions + language features for whole objects)
            gaussian_store.load_ply(sem0_path)
            _app_state["scene_source"] = "live"
            logger.info(
                "Loaded semantic_0.ply (whole objects): %d Gaussians",
                gaussian_store.count,
            )

            # Load levels 1/2 latents (62MB each — just the 3-dim compressed features)
            for level in [1, 2]:
                sem_path = os.path.join(config.scene_dir, f"semantic_{level}.ply")
                if os.path.exists(sem_path):
                    try:
                        gaussian_store.load_semantic_level(sem_path, level)
                    except Exception as exc:
                        logger.warning("Failed to load semantic_%d.ply: %s", level, exc)
        except Exception as exc:
            logger.warning("Failed to load semantic_0.ply: %s", exc)

    # Fallback: try point_cloud.ply in artifacts or fallback dir
    if not gaussian_store.is_loaded:
        for search_dir, source_label in [
            (os.path.join(config.scene_dir, "artifacts"), "live"),
            (os.path.join(config.fallback_scene_dir, "artifacts"), "fallback"),
        ]:
            ply_path = os.path.join(search_dir, "point_cloud.ply")
            if os.path.exists(ply_path):
                try:
                    gaussian_store.load_ply(ply_path)
                    _app_state["scene_source"] = source_label
                    logger.info(
                        "Using %s scene from %s (%d Gaussians)",
                        source_label, search_dir, gaussian_store.count,
                    )
                    break
                except Exception as exc:
                    logger.warning("Failed to load PLY from %s: %s", ply_path, exc)

    if not gaussian_store.is_loaded:
        logger.warning("No PLY with language features found in any location")

    _app_state["gaussian_store"] = gaussian_store

    # --- Decode all 3 levels to float16 (5.3GB each = ~16GB total) ---
    if autoencoder and gaussian_store.latent_features is not None:
        try:
            # Level 0 (whole objects)
            gaussian_store.decode_all(autoencoder, device)
            gaussian_store.decoded_levels[0] = gaussian_store.decoded_embeddings

            # Levels 1/2 (parts, subparts)
            for level in [1, 2]:
                if level in gaussian_store.semantic_levels:
                    gaussian_store.get_embeddings(level, autoencoder=autoencoder, device=device)

            total_gb = sum(
                arr.nbytes for arr in gaussian_store.decoded_levels.values()
            ) / 1e9
            logger.info(
                "All levels decoded to float16: %.1f GB total (%d levels)",
                total_gb, len(gaussian_store.decoded_levels),
            )

            # --- Sanity-check decoded embeddings vs the CLIP text encoder ---
            # Compute pairwise centroid similarity: if the decoded Gaussian embeddings
            # are all pointing in nearly the same direction (centroid norm >> 0),
            # the autoencoder checkpoint does not match the PLY's latent space.
            # This indicates the autoencoder.pth was from a different training run.
            _sample = gaussian_store.decoded_levels.get(0)
            if _sample is not None and len(_sample) > 0:
                import numpy as _np
                _sub = _sample[:min(1000, len(_sample))]
                _centroid = _sub.mean(axis=0)
                _centroid_norm = float(_np.linalg.norm(_centroid))
                # For uniformly distributed unit vectors the centroid norm ≈ 0.
                # Values > 0.5 indicate near-degenerate / collapsed embeddings.
                if _centroid_norm > 0.5:
                    logger.error(
                        "AUTOENCODER MISMATCH DETECTED: decoded Gaussian embeddings "
                        "are collapsed (centroid_norm=%.3f >> 0). "
                        "The autoencoder.pth checkpoint was not trained on the same "
                        "scene as the semantic PLY files. "
                        "Fix: download the matching autoencoder from the Colab training run "
                        "(ae_ckpt/best_ckpt.pth) and place it at data/scene/autoencoder.pth. "
                        "Queries will return near-zero matches until this is fixed.",
                        _centroid_norm,
                    )
                else:
                    logger.info(
                        "Autoencoder sanity check passed (centroid_norm=%.3f)",
                        _centroid_norm,
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

        # Pre-encode canonical negatives for relevancy scoring
        import numpy as np
        from server.services.similarity import CANONICAL_NEGATIVES
        neg_embs = np.stack([
            clip_encoder.encode_text(neg) for neg in CANONICAL_NEGATIVES
        ])
        _app_state["negative_embeddings"] = neg_embs
        logger.info("Encoded %d canonical negatives for relevancy scoring", len(CANONICAL_NEGATIVES))
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

    # Dense query-time retrieval is now the primary path.
    _app_state["scene_graph"] = None
    _app_state["instance_graph"] = {
        "nodes": [],
        "edges": [],
        "hierarchy": [],
        "metadata": {"node_count": 0, "edge_count": 0, "hierarchy_count": 0, "source": "instance-cache"},
    }
    _app_state["instance_store"] = {
        "instances": [],
        "next_id": 0,
        "query_history": [],
    }
    logger.info(
        "Dense query-time grounding ready; no startup scene graph will be built."
    )

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
app.include_router(adapter.router)


if __name__ == "__main__":
    import uvicorn

    config = Settings()
    uvicorn.run(
        "server.main:app",
        host=config.host,
        port=config.port,
        reload=False,
    )
