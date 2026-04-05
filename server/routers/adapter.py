"""Adapter router mapping frontend /api/* endpoints to real backend endpoints.

Bridges the frontend's expected API contract (URL prefix, response shape,
polling pattern) with ~100 lines of Python rather than modifying dozens of
frontend API calls. This is the single highest-leverage integration task.
"""

import asyncio
import glob
import logging
import os
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# --- Module-level exploration state ---
_explore_state: dict = {"status": "idle", "result": None, "scene_graph": None}
_query_jobs: dict[str, dict] = {}

QUERY_STEP_ORDER = [
    ("normalize_query", "Normalize query"),
    ("encode_query", "Encode text with CLIP"),
    ("score_semantics", "Score sem2"),
    ("cluster_regions", "Group grounded splats in 3D"),
    ("materialize_instances", "Materialize reusable instances"),
    ("build_graph", "Build instance graph"),
    ("prepare_response", "Prepare highlights and response"),
]


def _init_query_steps() -> list[dict]:
    return [
        {"key": key, "label": label, "status": "pending", "detail": None}
        for key, label in QUERY_STEP_ORDER
    ]


def _using_agent(state: dict) -> bool:
    from server.services.gemini_agent import gemini_enabled

    return gemini_enabled(state)


def _set_query_step(job: dict, key: str, status: str, detail: str | None = None) -> None:
    for step in job["steps"]:
        if step["key"] == key:
            step["status"] = status
            if detail is not None:
                step["detail"] = detail
            break


async def _run_query_job(job_id: str, body: dict) -> None:
    from server.main import get_app_state
    from server.services.dense_query import ground_query
    from server.services.gemini_agent import GeminiSceneAgent, format_dense_payload

    state = get_app_state()
    job = _query_jobs[job_id]
    query = body.get("query", "")
    mode = body.get("mode", "query")

    def progress_cb(step_key: str, status: str, detail: str | None = None) -> None:
        _set_query_step(job, step_key, status, detail)

    try:
        if _using_agent(state):
            try:
                agent = GeminiSceneAgent(state=state, progress_cb=progress_cb)
                payload = await asyncio.to_thread(agent.run, query=query, mode=mode)
            except Exception as exc:
                logger.exception("Gemini agent failed; falling back to dense query: %s", exc)
                result = await asyncio.to_thread(
                    ground_query,
                    text=query,
                    state=state,
                    persist_instances=True,
                    progress_cb=progress_cb,
                )
                payload = format_dense_payload(
                    result,
                    mode=mode,
                    query=query,
                    answer=result.get("answer", ""),
                )
                payload["agent_fallback"] = True
        else:
            result = await asyncio.to_thread(
                ground_query,
                text=query,
                state=state,
                persist_instances=True,
                progress_cb=progress_cb,
            )
            payload = format_dense_payload(
                result,
                mode=mode,
                query=query,
                answer=result.get("answer", ""),
            )
            if mode == "edit":
                color = parse_color_from_query(query)
                matched = result.get("nodes", [])
                target_label = matched[0].get("label", "unknown") if matched else "unknown"
                payload.update(
                    {
                        "action": "recolor",
                        "target": target_label,
                        "color": color,
                        "matched_gaussians": result.get("highlight_indices", []),
                        "matched_nodes": [
                            {"label": n.get("label", ""), "confidence": 0.9}
                            for n in matched
                        ],
                        "answer": f"Done! I've changed the {target_label} to {color}.",
                        "description": f"Done! I've changed the {target_label} to {color}.",
                    }
                )
        job["result"] = payload
        job["status"] = "complete"
    except Exception as exc:
        logger.exception("Query job failed: %s", exc)
        job["status"] = "error"
        job["error"] = str(exc)

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


def _build_highlight_regions(
    matched_nodes: list[dict],
    positions,
    *,
    max_regions: int = 8,
) -> list[dict]:
    """Build tight spherical highlight regions directly from matched nodes."""
    import numpy as np

    if positions is None:
        return []

    regions = []
    for node in matched_nodes:
        indices = np.asarray(node.get("gaussian_indices", []), dtype=np.int64)
        if len(indices) == 0:
            continue

        valid = indices[(indices >= 0) & (indices < len(positions))]
        if len(valid) == 0:
            continue

        cluster_pos = positions[valid]
        centroid = cluster_pos.mean(axis=0)
        dists = np.linalg.norm(cluster_pos - centroid, axis=1)
        radius = float(np.percentile(dists, 75)) + 0.05 if len(dists) > 0 else 0.15
        regions.append(
            {
                "node_id": node.get("id", ""),
                "label": node.get("label", "Unknown"),
                "centroid": centroid.tolist(),
                "radius": max(0.08, min(radius, 0.6)),
                "count": int(len(valid)),
                "mean_score": float(node.get("match_score", node.get("confidence", 0.0))),
            }
        )

    regions.sort(key=lambda r: -r["mean_score"])
    return regions[:max_regions]


# --- 1. Health endpoint ---


@router.get("/health")
async def api_health():
    """Proxy to internal health endpoint. Frontend checks data.status === 'ok'."""
    from server.main import get_app_state

    state = get_app_state()
    gaussian_store = state.get("gaussian_store")
    clip_encoder = state.get("clip_encoder")
    instance_graph = state.get("instance_graph")
    ply_loaded = gaussian_store.is_loaded if gaussian_store else False
    clip_ready = clip_encoder is not None
    graph_nodes = len(instance_graph.get("nodes", [])) if isinstance(instance_graph, dict) else 0
    graph_ready = graph_nodes > 0
    pipeline_ready = ply_loaded and clip_ready
    agent_ready = _using_agent(state)
    return {
        "status": "ok",
        "ply_loaded": ply_loaded,
        "autoencoder_initialized": state.get("autoencoder") is not None,
        "gaussian_count": gaussian_store.count if gaussian_store and ply_loaded else 0,
        "clip_ready": clip_ready,
        "scene_graph_ready": graph_ready,
        "scene_graph_nodes": graph_nodes,
        "instance_graph_ready": graph_ready,
        "instance_graph_nodes": graph_nodes,
        "pipeline_ready": pipeline_ready,
        "agent_ready": agent_ready,
        "agent_provider": "gemini" if agent_ready else None,
        "scene_source": state.get("scene_source", "none"),
    }


# --- 2. Explore start ---


async def _run_exploration(scene_id: str = "default") -> None:
    """Background task: catalog the currently grounded instance graph."""
    from server.main import get_app_state

    state = get_app_state()

    try:
        instance_graph = state.get("instance_graph") or {
            "nodes": [],
            "edges": [],
            "hierarchy": [],
            "metadata": {"node_count": 0, "edge_count": 0, "hierarchy_count": 0, "source": "instance-cache"},
        }
        _explore_state["scene_graph"] = instance_graph
        _explore_state["status"] = "exploring"

        logger.info("Adapter: starting exploration walker")
        from server.services.exploration_walker import ExplorationWalker

        walker = ExplorationWalker(
            scene_graph=instance_graph,
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
    scene_graph = _explore_state.get("scene_graph") or state.get("instance_graph")

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
        status = "complete"
        progress = 1.0 if scene_graph is not None else 0.0

    return {
        "status": status,
        "objects": objects,
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "progress": progress,
    }


# --- 4. Query endpoint ---


@router.post("/query/start")
async def api_query_start(body: dict):
    """Start a query job and return a polling handle for real progress updates."""
    from server.main import get_app_state
    from server.services.gemini_agent import init_agent_steps

    state = get_app_state()
    query = body.get("query", "")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must be non-empty.")

    steps = init_agent_steps() if _using_agent(state) else _init_query_steps()
    job_id = uuid.uuid4().hex
    _query_jobs[job_id] = {
        "id": job_id,
        "status": "running",
        "query": query,
        "mode": body.get("mode", "query"),
        "steps": steps,
        "result": None,
        "error": None,
    }
    asyncio.create_task(_run_query_job(job_id, body))
    return {"job_id": job_id, "status": "running", "steps": _query_jobs[job_id]["steps"]}


@router.get("/query/status/{job_id}")
async def api_query_status(job_id: str):
    """Poll a running query job for real stage progress and final result."""
    job = _query_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Query job not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "query": job["query"],
        "mode": job["mode"],
        "steps": job["steps"],
        "result": job["result"],
        "error": job["error"],
    }


@router.post("/query")
async def api_query(body: dict):
    """Ground a chat query directly against the dense 3D semantic field."""
    from server.main import get_app_state
    from server.services.dense_query import ground_query
    from server.services.gemini_agent import GeminiSceneAgent, format_dense_payload

    state = get_app_state()
    query = body.get("query", "")
    mode = body.get("mode", "query")

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must be non-empty.")

    if _using_agent(state):
        try:
            agent = GeminiSceneAgent(state=state)
            return await asyncio.to_thread(agent.run, query=query, mode=mode)
        except Exception as exc:
            logger.exception("Gemini agent failed on direct query; falling back: %s", exc)

    result = ground_query(text=query, state=state, persist_instances=True)
    return format_dense_payload(
        result,
        mode=mode,
        query=query,
        answer=result.get("answer", ""),
    )


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


@router.get("/scene/scene.ply")
async def api_scene_ply_ext():
    """Serve the scene PLY with .ply extension in URL for Spark.js format detection."""
    return await api_scene_splat()


@router.get("/scene/splat")
async def api_scene_splat():
    """Serve the scene file (.splat or .ply) from the artifacts directory."""
    from server.main import get_app_state

    state = get_app_state()
    config = state.get("config")
    if config is None:
        raise HTTPException(status_code=503, detail="Server not initialized.")

    artifacts_dir = os.path.join(config.scene_dir, "artifacts")

    # Try .splat first, then .ply (Spark.js supports both)
    for ext in ["*.splat", "*.ply", "*.spz", "*.ksplat"]:
        files = glob.glob(os.path.join(artifacts_dir, ext))
        if files:
            # Resolve symlinks to get real path
            real_path = os.path.realpath(files[0])
            logger.info("Serving scene file: %s (%s)", real_path, ext)
            return FileResponse(
                real_path,
                media_type="application/octet-stream",
                filename=os.path.basename(files[0]),
            )

    raise HTTPException(
        status_code=404,
        detail="No scene file found. Checked for .splat, .ply, .spz, .ksplat",
    )


# --- 7. CLIP proxy endpoints (for SparkRenderer highlight/heatmap) ---


@router.post("/clip/highlight")
async def api_clip_highlight(body: dict):
    """Return exact grounded highlight regions for a text query."""
    from server.main import get_app_state
    from server.services.dense_query import ground_query

    state = get_app_state()
    gaussian_store = state.get("gaussian_store")
    clip_encoder = state.get("clip_encoder")

    if not gaussian_store or not clip_encoder:
        raise HTTPException(status_code=503, detail="CLIP pipeline not ready.")

    text = body.get("text", "")
    result = ground_query(text=text, state=state, persist_instances=True)

    return {
        "regions": result.get("highlight_regions", []),
        "indices": result.get("highlight_match", {}).get("indices", []),
        "scores": result.get("highlight_match", {}).get("scores", []),
        "total_matched": len(result.get("highlight_indices", [])),
        "cluster_count": len(result.get("highlight_regions", [])),
        "level": 2,
        "level_name": "sem2",
        "query": text,
        "threshold": result.get("highlight_match", {}).get("used_threshold"),
        "semantic_fallback": result.get("semantic_fallback", False),
    }


@router.post("/clip/probability")
async def api_clip_probability(body: dict):
    """Probability cloud heatmap with multi-level support."""
    from server.main import get_app_state

    state = get_app_state()
    gaussian_store = state.get("gaussian_store")
    clip_encoder = state.get("clip_encoder")

    if not gaussian_store or not clip_encoder:
        raise HTTPException(status_code=503, detail="CLIP pipeline not ready.")

    level = body.get("level", 0)
    embeddings = gaussian_store.get_embeddings(level)
    if embeddings is None:
        raise HTTPException(status_code=503, detail=f"Semantic level {level} not loaded.")

    from server.services.similarity import probability_cloud

    text_embedding = clip_encoder.encode_text(body.get("text", ""))
    scores = probability_cloud(text_embedding, embeddings, temperature=body.get("temperature", 0.07))
    return {"scores": scores.tolist(), "level": level}


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

    Uses dense query grounding to find the target object, then returns recolor
    instructions with matched Gaussian indices and target color.
    """
    from server.main import get_app_state
    from server.services.dense_query import ground_query

    state = get_app_state()

    query = body.get("query", "")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must be non-empty.")
    result = ground_query(text=query, state=state, persist_instances=True)

    color = parse_color_from_query(query)
    matched = result.get("nodes", [])
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
        "highlight_regions": result.get("highlight_regions", []),
        "highlight_match": result.get("highlight_match", {"indices": [], "scores": [], "level": 2}),
        "answer": f"Done! I've changed the {target_label} to {color}.",
    }
