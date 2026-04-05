"""Adapter router mapping frontend /api/* endpoints to real backend endpoints.

Bridges the frontend's expected API contract (URL prefix, response shape,
polling pattern) with ~100 lines of Python rather than modifying dozens of
frontend API calls. This is the single highest-leverage integration task.
"""

import asyncio
import glob
import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# --- Module-level exploration state ---
_explore_state: dict = {"status": "idle", "result": None, "scene_graph": None}

# --- Category heuristic ---
CATEGORY_MAP = {
    "chair": "Furniture",
    "table": "Furniture",
    "desk": "Furniture",
    "shelf": "Furniture",
    "couch": "Furniture",
    "sofa": "Furniture",
    "bookshelf": "Furniture",
    "cabinet": "Furniture",
    "armchair": "Furniture",
    "monitor": "Electronics",
    "laptop": "Electronics",
    "screen": "Electronics",
    "lamp": "Lighting",
    "light": "Lighting",
    "plant": "Decor",
    "rug": "Decor",
    "vase": "Decor",
    "painting": "Decor",
    "window": "Structure",
    "wall": "Structure",
    "door": "Structure",
    "floor": "Structure",
}


def categorize(label: str) -> str:
    """Derive a display category from an object label."""
    lower = label.lower()
    for keyword, category in CATEGORY_MAP.items():
        if keyword in lower:
            return category
    return "Objects"


def _nodes_to_objects(nodes: list[dict]) -> list[dict]:
    """Map scene graph nodes to frontend DetectedObject format."""
    return [
        {
            "id": n.get("id", f"obj_{i:03d}"),
            "label": n.get("label", "Unknown"),
            "confidence": n.get("confidence", 0.9),
            "position": n.get("centroid", [0, 0, 0]),
            "category": categorize(n.get("label", "")),
        }
        for i, n in enumerate(nodes)
    ]


def _edges_to_graph_edges(edges: list[dict]) -> list[dict]:
    """Map scene graph edges to frontend GraphEdge format."""
    return [
        {
            "from": e.get("from", e.get("from_id", "")),
            "to": e.get("to", e.get("to_id", "")),
            "relation": e.get("relation", "related"),
        }
        for e in edges
    ]


# --- 1. Health endpoint ---


@router.get("/health")
async def api_health():
    """Proxy to internal health endpoint. Frontend checks data.status === 'ok'."""
    from server.main import get_app_state

    state = get_app_state()
    gaussian_store = state.get("gaussian_store")
    return {
        "status": "ok",
        "ply_loaded": (
            gaussian_store.is_loaded if gaussian_store else False
        ),
        "autoencoder_initialized": state.get("autoencoder") is not None,
        "gaussian_count": (
            gaussian_store.count
            if gaussian_store and gaussian_store.is_loaded
            else 0
        ),
    }


# --- 2. Explore start ---


async def _run_exploration(scene_id: str = "default") -> None:
    """Background task: build scene graph then run exploration walker."""
    from server.main import get_app_state

    state = get_app_state()

    try:
        # Step 1: Build scene graph if not already built
        if state.get("scene_graph") is None:
            logger.info("Adapter: building scene graph before exploration")
            try:
                from server.services.graph_builder import build_scene_graph

                result = build_scene_graph(
                    gaussian_store=state["gaussian_store"],
                    clip_encoder=state["clip_encoder"],
                    query="objects",
                    k=5000,
                    min_samples=6,
                    hierarchy_threshold=0.7,
                )
                state["scene_graph"] = result
                # Make scene graph available for polling
                _explore_state["scene_graph"] = result
                _explore_state["status"] = "exploring"
                logger.info("Adapter: scene graph built")
            except Exception as exc:
                logger.error("Adapter: scene graph build failed: %s", exc)
                _explore_state["status"] = "complete"
                _explore_state["result"] = None
                return

        # Step 2: Run exploration walker
        logger.info("Adapter: starting exploration walker")
        from server.services.exploration_walker import ExplorationWalker

        walker = ExplorationWalker(
            scene_graph=state["scene_graph"],
            memory_service=state.get("memory_service"),
            scene_id=scene_id,
        )
        result = await walker.run()
        state["exploration_catalog"] = result
        _explore_state["result"] = result
        _explore_state["status"] = "complete"
        logger.info("Adapter: exploration complete")

    except Exception as exc:
        logger.error("Adapter: exploration failed: %s", exc)
        _explore_state["status"] = "complete"
        _explore_state["result"] = None


@router.post("/explore/start")
async def api_explore_start():
    """Kick off exploration in the background. Returns immediately."""
    _explore_state["status"] = "exploring"
    _explore_state["result"] = None
    _explore_state["scene_graph"] = None
    asyncio.create_task(_run_exploration())
    return {"status": "started"}


# --- 3. Explore status (polled by frontend) ---


@router.get("/explore/status")
async def api_explore_status():
    """Return exploration status shaped for the frontend polling contract."""
    from server.main import get_app_state

    state = get_app_state()
    scene_graph = _explore_state.get("scene_graph") or state.get("scene_graph")

    # Default empty response
    objects: list[dict] = []
    graph_nodes: list[dict] = []
    graph_edges: list[dict] = []
    progress = 0.0

    if scene_graph is not None:
        # Extract nodes and edges from scene graph (Pydantic model or dict)
        raw_nodes = (
            scene_graph.get("nodes", [])
            if isinstance(scene_graph, dict)
            else [n.model_dump(by_alias=True) for n in scene_graph.nodes]
            if hasattr(scene_graph, "nodes")
            else []
        )
        raw_edges = (
            scene_graph.get("edges", [])
            if isinstance(scene_graph, dict)
            else [e.model_dump(by_alias=True) for e in scene_graph.edges]
            if hasattr(scene_graph, "edges")
            else []
        )

        objects = _nodes_to_objects(raw_nodes)
        graph_nodes = [
            {"id": n.get("id", ""), "label": n.get("label", ""), "type": "object"}
            for n in raw_nodes
        ]
        graph_edges = _edges_to_graph_edges(raw_edges)
        progress = 0.5 if _explore_state["status"] == "exploring" else 1.0

    status = _explore_state["status"]
    if status == "idle":
        status = "exploring"

    return {
        "status": status,
        "objects": objects,
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "progress": progress,
    }


# --- 4. Query endpoint ---


@router.post("/query")
async def api_query(body: dict):
    """Accept a chat query and shape the walker response for the frontend."""
    from server.main import get_app_state

    state = get_app_state()
    query = body.get("query", "")
    session_id = body.get("session_id", "default")

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must be non-empty.")

    if state.get("scene_graph") is None:
        raise HTTPException(
            status_code=503,
            detail="Scene graph not built. Run exploration first.",
        )

    from server.services.query_walker import QueryWalker

    walker = QueryWalker(
        scene_graph=state["scene_graph"],
        query=query,
        memory_service=state.get("memory_service"),
        scene_id=session_id,
    )
    result = await walker.run()

    matched = result.get("matched_nodes", [])
    labels = [n.get("label", "") for n in matched]

    return {
        "description": result.get("answer", ""),
        "answer": result.get("answer", ""),
        "nodes": [
            {"label": n.get("label", ""), "confidence": 0.9}
            for n in matched
        ],
        "reasoning": (
            f"QueryWalker traversed {len(matched)} nodes, "
            f"matched: {', '.join(labels) if labels else 'none'}"
        ),
    }


# --- 5. Scenes list ---


@router.get("/scenes")
async def api_scenes():
    """Return available scenes (hardcoded for hackathon, INS-02)."""
    return [
        {
            "id": "default",
            "name": "Modern Living Room",
            "status": "ready",
            "object_count": 12,
            "splat_url": "/api/scene/splat",
        }
    ]


# --- 6. Splat file serving ---


@router.get("/scene/splat")
async def api_scene_splat():
    """Serve the .splat file from the scene artifacts directory."""
    from server.main import get_app_state

    state = get_app_state()
    config = state.get("config")
    if config is None:
        raise HTTPException(status_code=503, detail="Server not initialized.")

    artifacts_dir = os.path.join(config.scene_dir, "artifacts")
    splat_files = glob.glob(os.path.join(artifacts_dir, "*.splat"))

    if not splat_files:
        raise HTTPException(status_code=404, detail="No .splat file found.")

    return FileResponse(
        splat_files[0],
        media_type="application/octet-stream",
        filename=os.path.basename(splat_files[0]),
    )


# --- 7. CLIP proxy endpoints (for SparkRenderer highlight/heatmap) ---


@router.post("/clip/highlight")
async def api_clip_highlight(body: dict):
    """Proxy to /clip/highlight for Spark.js semantic highlighting."""
    from server.main import get_app_state

    state = get_app_state()
    gaussian_store = state.get("gaussian_store")
    clip_encoder = state.get("clip_encoder")

    if not gaussian_store or not clip_encoder or gaussian_store.decoded_embeddings is None:
        raise HTTPException(status_code=503, detail="CLIP pipeline not ready.")

    from server.services.similarity import top_k, highlight_mask

    text_embedding = clip_encoder.encode_text(body.get("text", ""))
    indices, scores = top_k(text_embedding, gaussian_store.decoded_embeddings, k=body.get("k", 100))
    mask = highlight_mask(scores, indices, gaussian_store.count)
    return {"mask": mask.tolist(), "indices": indices.tolist(), "scores": scores.tolist()}


@router.post("/clip/probability")
async def api_clip_probability(body: dict):
    """Proxy to /clip/probability for probability cloud heatmap."""
    from server.main import get_app_state

    state = get_app_state()
    gaussian_store = state.get("gaussian_store")
    clip_encoder = state.get("clip_encoder")

    if not gaussian_store or not clip_encoder or gaussian_store.decoded_embeddings is None:
        raise HTTPException(status_code=503, detail="CLIP pipeline not ready.")

    from server.services.similarity import probability_cloud

    text_embedding = clip_encoder.encode_text(body.get("text", ""))
    scores = probability_cloud(text_embedding, gaussian_store.decoded_embeddings, temperature=body.get("temperature", 0.07))
    return {"scores": scores.tolist()}


# --- 8. Memory stub ---


@router.get("/memory/{session_id}")
async def api_memory(session_id: str):
    """Stub for MemoryPanel (BottomPanels.tsx). Will be backed by Backboard."""
    return {
        "room_summary": "Scene explored",
        "changes_detected": 0,
    }


# --- 9. Edit endpoint (scene editing: recolor objects) ---

# Common color name → hex mapping
COLOR_MAP = {
    "red": "#ff0000",
    "blue": "#0000ff",
    "green": "#00ff00",
    "yellow": "#ffff00",
    "orange": "#ff8800",
    "purple": "#8800ff",
    "pink": "#ff69b4",
    "white": "#ffffff",
    "black": "#000000",
    "brown": "#8b4513",
    "gray": "#808080",
    "grey": "#808080",
    "gold": "#ffd700",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "teal": "#008080",
}


def parse_color_from_query(query: str) -> str:
    """Extract a color name from a natural-language edit query.

    Searches for known color names in the query text and returns the
    corresponding hex value. Falls back to red if no color is detected.
    """
    lower = query.lower()
    for name, hex_val in COLOR_MAP.items():
        if name in lower:
            return hex_val
    return "#ff0000"


@router.post("/edit")
async def api_edit(body: dict):
    """Handle scene editing queries like 'change the couch to red'.

    Uses QueryWalker to find the target object, then returns recolor
    instructions with matched Gaussian indices and target color.
    """
    from server.main import get_app_state

    state = get_app_state()

    if state.get("scene_graph") is None:
        raise HTTPException(
            status_code=503,
            detail="Scene graph not built. Run exploration first.",
        )

    query = body.get("query", "")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must be non-empty.")

    from server.services.query_walker import QueryWalker

    walker = QueryWalker(
        scene_graph=state["scene_graph"],
        query=query,
        memory_service=state.get("memory_service"),
        scene_id=body.get("session_id", "default"),
    )
    result = await walker.run()

    color = parse_color_from_query(query)
    matched = result.get("matched_nodes", [])
    target_label = matched[0].get("label", "unknown") if matched else "unknown"

    return {
        "action": "recolor",
        "target": target_label,
        "color": color,
        "matched_gaussians": result.get("highlight_indices", []),
        "matched_nodes": [
            {"label": n.get("label", ""), "confidence": 0.9}
            for n in matched
        ],
        "answer": f"Done! I've changed the {target_label} to {color}.",
    }
