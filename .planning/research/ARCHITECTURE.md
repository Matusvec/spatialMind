# Architecture Patterns

**Project:** SpatialMind
**Domain:** Spatial AI — LangSplat + JAC + Spark.js
**Researched:** 2026-04-04
**Confidence:** MEDIUM (JAC/jac-cloud patterns HIGH via official docs; LangSplat output structure MEDIUM via GitHub README; Backboard API LOW via docs only; InsForge JS-only SDK LOW for Python use)

---

## Recommended Architecture

### System Overview

SpatialMind has three distinct layers, each with a hard boundary:

```
+---------------------------+
|      TRAINING LAYER       |
|  (Colab A100 — one-shot)  |
|  COLMAP + SAM + LangSplat |
|  Output: PLY + .npy files |
+------------|--+------------+
             | Files uploaded/served
+------------|--+--------------------------------------------------+
|            v     PYTHON BACKEND SERVER (FastAPI)                 |
|                                                                   |
|  [Gaussian Store]  <-->  [CLIP Query Engine]                     |
|       numpy/PLY             openai/clip PyTorch                  |
|           |                        |                              |
|           v                        v                              |
|  [DBSCAN Clustering]   [Spatial Relationship Extractor]          |
|   sklearn DBSCAN         Y-offset, XZ-proximity, covariance      |
|           |                        |                              |
|           +----------+-------------+                              |
|                      |                                            |
|                      v                                            |
|              [JAC Cloud Server] (:8000)                          |
|         Spatial Knowledge Graph (nodes + edges)                  |
|         Walker agents: Query, Navigation, Exploration, Memory    |
|         Persistence: MongoDB (jac-scale)                         |
|                      |                                            |
|              [Backboard.io]                                       |
|         Persistent memory per session (thread_id)               |
+----------------------|--------------------------------------------+
                       | REST (JSON over HTTP)
+----------------------|--------------------------------------------+
|                      v     BROWSER FRONTEND (React)              |
|                                                                   |
|  [Spark.js SplatMesh]    [Graph Overlay]     [Chat Interface]    |
|  Three.js renderer       React canvas/SVG    Text input          |
|  .PLY/.SPZ file          node/edge viz       session state       |
|           ^                    ^                   |              |
|           |                    |                   v              |
|     highlight masks        graph JSON         POST /walker/*     |
|     (Gaussian indices)   (nodes + edges)                         |
|                      ^                                            |
|                      |  SSE or WebSocket (stretch goal)          |
+----------------------|--------------------------------------------+
                       | via InsForge (auth + storage + edge fns)
+----------------------|--------------------------------------------+
|              [InsForge]                                           |
|  Auth (JWT/OAuth)  |  File storage (PLY serving)  |  Postgres    |
+-------------------+-------------------------------------------+--+
```

---

## Component Responsibilities

### Training Layer (Colab A100 — offline, blocking)

| Component | Responsibility |
|-----------|---------------|
| Scaniverse capture | iOS app to capture scene as photos before/during hackathon |
| COLMAP | Structure-from-motion: computes camera poses from photos |
| SAM (Segment Anything) | Per-image semantic masks for hierarchical CLIP feature generation |
| LangSplat autoencoder | Compresses 512-dim CLIP embeddings to 3-dim latent per Gaussian |
| LangSplat trainer | Trains Gaussians with color + opacity + latent language feature |
| Output artifacts | `point_cloud.ply`, `language_feature/` (full .npy), `language_feature_dim3/` (compressed .npy), `autoencoder.pth` |

**Output directory structure (confirmed from GitHub README):**
```
<scene_name>/
  images/                        # Input photos
  sparse/0/                      # COLMAP camera poses
  language_feature/              # Per-image 512-dim CLIP features (.npy)
  language_feature_dim3/         # Compressed 3-dim features (.npy)
  output/<scene_name>/
    point_cloud/iteration_30000/
      point_cloud.ply            # Gaussians: XYZ + color + opacity + covariance
    chkpnt30000.pth              # Model checkpoint
    autoencoder.pth              # Decoder: 3-dim latent → 512-dim CLIP
```

**Critical constraint:** The 3-dim latent features stored per-Gaussian must be decoded back to 512-dim CLIP space at query time using the trained autoencoder. The autoencoder is scene-specific and must travel with the PLY file.

---

### Python Backend Server (FastAPI, port 8001)

This is the **semantic compute layer** — all heavy computation lives here.

| Component | Responsibility | Key Libraries |
|-----------|---------------|---------------|
| Gaussian Store | Load PLY + npy files into memory as numpy arrays; hold Gaussian positions + latent features | `numpy`, `plyfile` |
| CLIP Text Encoder | Encode incoming query strings to 512-dim CLIP text vectors | `torch`, `clip` (openai/CLIP) |
| Autoencoder Decoder | Decode stored 3-dim Gaussian latents back to 512-dim for cosine similarity | `torch` (loaded from `autoencoder.pth`) |
| Cosine Similarity Ranker | Compare CLIP text vector against all decoded Gaussian embeddings; return top-K indices + scores | `numpy`, `torch` |
| DBSCAN Clusterer | Group semantically similar Gaussians by 3D XYZ proximity into object-level clusters | `sklearn.cluster.DBSCAN` |
| Spatial Relationship Extractor | Given cluster centroids + bounding boxes, derive on_top_of, next_to, inside, facing edges | `numpy` (custom logic) |
| Graph Builder | POST cluster+edge data to JAC server to populate the spatial knowledge graph | `httpx` (async REST calls to JAC) |
| Highlight Mask API | Accept query result (Gaussian indices + confidence scores) and return as JSON mask | FastAPI endpoint |
| PLY File Server | Serve the visual-only (no language feature) PLY for Spark.js to load | FastAPI static route or InsForge storage |

**Key design decision:** The Python server calls JAC over HTTP to write graph nodes/edges. It does NOT embed JAC — JAC runs as a separate process. This keeps the compute-heavy Python side independent of the JAC graph runtime.

---

### JAC Cloud Server (jac-scale, port 8000)

The **spatial reasoning and agent layer**. JAC natively turns walkers into REST endpoints.

| Walker | Responsibility | Endpoint |
|--------|---------------|----------|
| `QueryWalker` | Accept natural language query + session ID; call Python CLIP server; traverse graph to find matching nodes; return node IDs + highlight mask + spatial description | `POST /walker/query` |
| `NavigationWalker` | Given a target node ID, compute traversal path through the spatial graph; return ordered node list with directions | `POST /walker/navigate` |
| `ExplorationWalker` | Triggered on scene load; traverse all unvisited nodes; summarize undiscovered objects; call `by llm()` for descriptions | `POST /walker/explore` |
| `MemoryWalker` | Write session facts to Backboard; retrieve prior session context; inject into current walker state | `POST /walker/memory` |
| `SceneWalker` | Accept Python-built graph payload (nodes + edges JSON); upsert into JAC graph store | `POST /walker/build_scene` (called by Python, not browser) |

**Node schema:**
```jac
node SpatialObject {
  has id: str;
  has label: str;           // "chair", "table", "lamp"
  has centroid: list[float]; // [x, y, z]
  has bbox: list[float];    // [x_min, y_min, z_min, x_max, y_max, z_max]
  has gaussian_indices: list[int]; // Indices into Gaussian array
  has confidence: float;
  has description: str;     // LLM-generated
  has visited: bool = False;
}

edge SpatialRelation {
  has relation: str;        // "on_top_of" | "next_to" | "inside" | "facing"
  has confidence: float;
  has distance: float;
}
```

**Persistence:** jac-scale uses MongoDB automatically. Every node and edge persists across server restarts with no manual save code.

**Authentication:** `:pub` modifier on browser-facing walkers. `SceneWalker` is `:priv` — only callable by the Python server with a server-side token.

---

### Browser Frontend (React + TypeScript, Vite)

| Component | Responsibility | Technology |
|-----------|---------------|-----------|
| Spark.js Viewer | Load + render visual PLY as 3DGS; handle camera orbit; receive highlight mask and apply per-Gaussian color tinting | `@sparkjsdev/spark`, Three.js |
| Graph Overlay | Fetch graph JSON from JAC; render nodes as 3D-projected markers and edges as lines in CSS/SVG overlay; click-to-select | React + Three.js projection math |
| Chat Interface | Text input for spatial queries; stream responses from JAC QueryWalker; display spatial descriptions | React, `fetch` with streaming |
| Highlight Controller | Receive Gaussian indices from query result; use Spark.js Dyno shader graph to tint matched Gaussians highlight color | Spark.js Dyno API |
| Scene Loader | On app init: request PLY URL from backend; load into Spark.js `SplatMesh`; trigger ExplorationWalker | React `useEffect` |
| Auth Gate | InsForge JWT auth; get user session; pass token to JAC walker calls | `@insforge/sdk` |

**Spark.js integration note (MEDIUM confidence):** The Dyno shader graph system allows per-Gaussian GPU-side manipulation. For highlighting, the pattern is: build a `SplatMesh`, then compose a Dyno that checks if a Gaussian's index is in a provided buffer of "highlighted indices" and returns a different RGBA. The exact Dyno API requires consulting `sparkjs.dev/docs/dyno-overview/` — this is a risk item for Phase 1 frontend work.

---

### Backboard (Persistent Memory)

| Function | How Used |
|----------|----------|
| Session memory | Each chat session gets a `thread_id`; the MemoryWalker stores spatial facts ("user asked about the chair near the window twice") |
| Cross-session recall | On new session start, MemoryWalker retrieves prior thread facts via Backboard RAG; injects into ExplorationWalker context |
| Fact storage | Structured facts from QueryWalker results ("table is next_to lamp") stored as memory entries |

**API pattern (confirmed):**
```
POST https://app.backboard.io/api/threads/{thread_id}/messages
{
  "content": "<spatial fact or query>",
  "memory": "Auto",
  "stream": false
}
```

**Constraint:** Backboard's memory API is thread-scoped, not graph-scoped. Spatial facts must be serialized to natural language before storage (Backboard does not store structured JSON natively). The MemoryWalker handles this translation.

---

### InsForge (Backend Infrastructure)

| Function | How Used |
|----------|----------|
| Auth | JWT-based user auth for frontend; InsForge SDK handles signup/login/sessions |
| PLY File Storage | S3-compatible bucket stores the trained PLY scene file; URL served to Spark.js |
| Edge Functions | Optional: run scene-processing webhooks (e.g., trigger Python pipeline on file upload) |
| Postgres | Store scene metadata (scene name, upload time, status); not the graph — JAC owns the graph |

**SDK:** JavaScript/TypeScript only (`@insforge/sdk`). No Python SDK found. The Python FastAPI server interacts with InsForge via REST calls (standard HTTP to InsForge endpoints) or not at all — the Python server reads PLY from file system or a direct S3 URL.

---

## Data Flow

### Flow 1: Initial Scene Build (one-time, sequential)

```
1. Scaniverse (iOS)      → photos
2. COLMAP (Colab)        → camera_poses.bin (sparse/ directory)
3. LangSplat training    → point_cloud.ply + language_feature_dim3/*.npy + autoencoder.pth
4. Upload to InsForge    → storage bucket: scene.ply, features.npy, autoencoder.pth
5. Python server start   → load PLY + npy into memory (Gaussian Store)
6. Python: DBSCAN run    → cluster objects → centroid + bbox per cluster
7. Python: spatial rels  → compute on_top_of / next_to edges
8. Python: POST to JAC   → POST /walker/build_scene (graph construction)
9. JAC: SceneWalker      → upsert nodes + edges into MongoDB
10. Python: ExploreWalker trigger → POST /walker/explore → LLM descriptions per node
```

**This flow is blocking.** The graph cannot be queried until steps 5-9 complete. Target: < 5 minutes for a room-sized scene.

### Flow 2: Natural Language Query (real-time, per query)

```
Browser: user types "where is the lamp?"
   |
   v
Browser: POST /walker/query { query: "lamp", session_id: "abc" }
   |
   v
JAC QueryWalker:
   1. Forward query text to Python server: POST :8001/clip/encode { text: "lamp" }
   2. Python: CLIP encode → 512-dim text vector
   3. Python: cosine similarity against all Gaussian embeddings
   4. Python: return { gaussian_indices: [...], scores: [...] }
   5. JAC: lookup which graph nodes contain those Gaussian indices
   6. JAC: traverse graph edges to find spatial context ("next_to table")
   7. JAC: call MemoryWalker to check/store session memory
   8. JAC: return { nodes: [...], highlight_mask: { indices: [...], scores: [...] }, description: "The lamp is on the table near the window" }
   |
   v
Browser: render highlight mask via Spark.js Dyno
Browser: overlay node markers on matched nodes
Browser: display description in chat panel
```

**Latency targets (rough):**
- CLIP encode: ~50ms (GPU, server-side)
- Cosine similarity (50K Gaussians): ~20ms (numpy vectorized)
- JAC graph traversal: ~5ms (in-memory MongoDB query)
- Backboard memory call: ~200ms (external API, non-blocking)
- Total perceived latency: < 500ms for non-memory path

### Flow 3: Autonomous Exploration (background, on scene load)

```
JAC ExplorationWalker spawned on scene load:
   1. Traverse all SpatialObject nodes where visited = False
   2. For each node: call `by llm()` with centroid + neighbors → generate description
   3. Store description on node
   4. Mark visited = True
   5. (Optional) POST brief summary to Backboard as session memory seed
```

This runs asynchronously. Frontend shows a "discovering scene..." indicator.

### Flow 4: Navigation Request

```
Browser: user asks "how do I get from the chair to the bookshelf?"
   |
   v
JAC NavigationWalker:
   1. Find node IDs for "chair" and "bookshelf" via CLIP query
   2. BFS/DFS over SpatialRelation edges in JAC graph
   3. Return ordered path: [chair_node → next_to → table_node → next_to → bookshelf_node]
   4. Compute direction vectors from centroid deltas
   5. Return { path: [...], directions: ["turn left", "move forward 2m"] }
   |
   v
Browser: animate camera path through Spark.js scene
```

---

## Integration Points

### Python FastAPI ↔ JAC (Internal, server-to-server)

```
Direction: Python → JAC (scene build, explore trigger)
           JAC → Python (CLIP encode during query)

Python calls JAC:
POST http://localhost:8000/walker/build_scene
Authorization: Bearer <server_token>
Content-Type: application/json
{
  "nodes": [
    { "id": "obj_001", "label": "chair", "centroid": [1.2, 0.5, -0.8], "gaussian_indices": [1024, 1025, ...] }
  ],
  "edges": [
    { "from": "obj_001", "to": "obj_002", "relation": "next_to", "distance": 0.6 }
  ]
}

JAC calls Python (from within QueryWalker):
POST http://localhost:8001/clip/encode
{ "text": "lamp near the window" }
Response: { "embedding": [0.12, -0.34, ...] }  // 512-dim float array
```

### Browser ↔ JAC (Public REST)

```
Direction: Browser → JAC (queries, navigation, memory)

POST http://localhost:8000/walker/query
Content-Type: application/json
{ "query": "where is the lamp?", "session_id": "user_abc_session_1" }

Response:
{
  "nodes": [{ "id": "obj_004", "label": "lamp", "centroid": [2.1, 1.2, 0.3] }],
  "highlight_mask": { "indices": [4096, 4097, ...], "scores": [0.92, 0.88, ...] },
  "description": "The lamp is on the side table next to the armchair."
}
```

### Browser ↔ Spark.js (In-process)

```
SplatMesh loads PLY URL from InsForge storage bucket.
Highlight mask applied via Dyno shader:
  - Dyno reads "highlighted_indices" uniform buffer
  - Per-Gaussian: if index in buffer → RGBA tint applied (yellow/orange)
  - Dyno compiles to GLSL, runs on GPU
```

### Browser ↔ InsForge (Auth + Storage)

```
import { createClient } from '@insforge/sdk'
const insforge = createClient({ url: INSFORGE_URL, key: INSFORGE_ANON_KEY })

// Auth
const { user } = await insforge.auth.signIn({ email, password })

// PLY URL
const { data: { publicUrl } } = insforge.storage.from('scenes').getPublicUrl('scene_name/scene.ply')
```

### JAC ↔ Backboard (Memory Persistence)

```
Called from within MemoryWalker using JAC's HTTP client or `by llm()` bridge:
POST https://app.backboard.io/api/threads/{session_id}/messages
Authorization: Bearer <backboard_api_key>
{ "content": "User asked about lamp (obj_004). Found next_to armchair.", "memory": "Auto" }
```

---

## Suggested Build Order

Dependencies are strict — each phase unblocks the next.

### Phase 0: Training (Blocking Dependency — Start Immediately)
**Do before anything else.** Colab A100 training takes 1-3 hours.
1. Capture scene with Scaniverse → export photos
2. Run COLMAP on photos
3. Run LangSplat training (full pipeline)
4. Download artifacts: `point_cloud.ply`, `language_feature_dim3/`, `autoencoder.pth`

Everything below can be prototyped in parallel with training using a dummy scene (pre-existing LangSplat demo data).

### Phase 1: Python Server Foundation
1. FastAPI skeleton with health check
2. PLY loader + numpy array in memory (Gaussian positions + latent features)
3. Load autoencoder.pth; implement 3-dim → 512-dim decode
4. `POST /clip/encode` endpoint — CLIP text encoder
5. Cosine similarity against decoded Gaussian embeddings
6. Return top-K Gaussian indices + scores

**Deliverable:** `curl -X POST /clip/encode -d '{"text":"chair"}' ` returns matching Gaussian indices.

### Phase 2: Clustering + Graph Construction
1. DBSCAN on XYZ positions of semantically matched Gaussians
2. Compute cluster centroids, bounding boxes
3. Spatial relationship extraction (Y-offset → on_top_of, XZ proximity → next_to)
4. JAC node/edge schema definitions
5. `SceneWalker` in JAC to accept and upsert graph payload
6. Python calls JAC to build graph from DBSCAN output

**Deliverable:** `curl /walker/build_scene` populates a queryable JAC graph.

### Phase 3: JAC Query + Navigation Walkers
1. `QueryWalker` — calls Python CLIP encoder, looks up graph nodes, returns highlight mask + description
2. `NavigationWalker` — BFS over graph edges, returns path
3. `ExplorationWalker` — traverse unvisited nodes, generate descriptions via `by llm()`
4. `MemoryWalker` — Backboard read/write for session context
5. jac-scale persistence configuration (MongoDB)

**Deliverable:** `curl /walker/query -d '{"query":"lamp"}' ` returns nodes + highlight mask + description.

### Phase 4: Browser Frontend
1. Vite + React + TypeScript skeleton
2. Spark.js `SplatMesh` loading PLY from URL (use demo splat first)
3. InsForge auth + PLY URL resolution
4. Chat interface wired to JAC `QueryWalker`
5. Highlight mask application via Spark.js Dyno shader
6. Graph overlay: project node centroids to screen, render markers + edges

**Deliverable:** Working browser showing splat + chat query + highlight response.

### Phase 5: Polish + Integration
1. Connect live trained PLY to frontend
2. ExplorationWalker auto-trigger on scene load
3. MemoryWalker integration (Backboard session persistence)
4. Probability cloud visualization (soft assignments per Gaussian)
5. Navigation path camera animation

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Calling CLIP in the Browser
**What:** Attempting to run CLIP text encoding in the browser via ONNX or Transformers.js
**Why bad:** CLIP ViT-B/32 is 350MB+. Load time alone kills the demo. No GPU acceleration in browser context.
**Instead:** All CLIP computation stays in Python FastAPI server. Browser sends text strings only.

### Anti-Pattern 2: Storing Full 512-dim Embeddings Per Gaussian
**What:** Skipping the LangSplat autoencoder and storing raw CLIP embeddings in the PLY
**Why bad:** 50,000 Gaussians x 512 dims x 4 bytes = 100MB per scene, in-memory. Loading and cosine-similarity over this is slow.
**Instead:** Store the 3-dim latents (LangSplat default). Decode to 512-dim at query time for only the matched subset.

### Anti-Pattern 3: Rebuilding the Graph on Every Query
**What:** Running DBSCAN + spatial relationship extraction on every query call
**Why bad:** DBSCAN on 50K points takes 2-5 seconds. Ruins interactivity.
**Instead:** Build the graph once on scene load (scene build flow). Queries only traverse the pre-built JAC graph.

### Anti-Pattern 4: Tight Coupling Between Python Server and JAC
**What:** Embedding JAC logic inside Python or calling Python functions directly from JAC via subprocess
**Why bad:** Makes both services harder to develop in parallel; debugging cross-language calls via subprocess is painful under time pressure.
**Instead:** Strict REST boundary. Python → JAC over HTTP. JAC → Python over HTTP. No shared memory or subprocess calls.

### Anti-Pattern 5: Synchronous Backboard Calls in Query Hot Path
**What:** Awaiting Backboard memory retrieval before returning query results to browser
**Why bad:** Backboard external API call adds ~200-500ms per query. Unacceptable latency for interactive use.
**Instead:** Return query results immediately. Fire Backboard memory write as fire-and-forget. Pre-fetch memory context at session start, not per-query.

### Anti-Pattern 6: Using JAC Graph for Gaussian-Level Data
**What:** Storing individual Gaussian embeddings as JAC nodes (50K+ nodes)
**Why bad:** JAC's graph is for object-level semantic nodes, not point-level data. MongoDB with 50K+ nodes per scene would be slow and wasteful.
**Instead:** Python numpy arrays own Gaussian-level data. JAC owns object-level (cluster) nodes only. Gaussian indices are stored as a list property on SpatialObject nodes.

---

## Scalability Considerations

This is a hackathon — single-scene, single-user. These notes are for awareness, not action.

| Concern | At 1 scene (now) | At 10 scenes | At 100+ scenes |
|---------|-----------------|-------------|----------------|
| Gaussian storage | In-memory numpy (fast) | Per-scene load on demand | Mmap or vector DB (Qdrant) |
| Graph persistence | MongoDB via jac-scale (auto) | Same, scene-scoped roots | Sharding by scene_id |
| CLIP compute | Single GPU on server | Queue-based async | Batch inference server |
| PLY serving | InsForge storage | CDN-fronted | Streaming LoD (Spark 2.0) |

---

## Sources

- LangSplat GitHub README (output structure, training pipeline): [minghanqin/LangSplat](https://github.com/minghanqin/LangSplat)
- LangSplat CVPR 2024 paper (autoencoder compression, query approach): [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.pdf)
- JAC walker architecture: [docs.jaseci.org walkers](https://docs.jaseci.org/learn/data_spatial/walkers/)
- JAC walkers as REST endpoints: [chapter 12](https://docs.jaseci.org/jac_book/chapter_12/)
- JAC local API server / jac-start: [production local guide](https://docs.jaseci.org/tutorials/production/local/)
- JAC node persistence (jac-scale, MongoDB): [jac-scale release notes](https://docs.jaseci.org/community/release_notes/jac-scale/)
- Spark.js overview + file formats: [sparkjs.dev/docs/overview](https://sparkjs.dev/docs/overview/)
- Spark.js GitHub (SplatMesh API, Dyno system): [sparkjsdev/spark](https://github.com/sparkjsdev/spark)
- Backboard.io quickstart (thread_id, memory API): [docs.backboard.io/quickstart](https://docs.backboard.io/quickstart)
- InsForge introduction (Postgres, auth, storage, edge functions): [docs.insforge.dev/introduction](https://docs.insforge.dev/introduction)
- InsForge JS SDK: [InsForge/InsForge-sdk-js](https://github.com/InsForge/InsForge-sdk-js)
