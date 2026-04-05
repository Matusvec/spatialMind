# SpatialMind

**An AI that sees, navigates, and reasons inside photorealistic 3D worlds.**

SpatialMind turns 3D Gaussian splat scenes into queryable semantic fields. Point at any object with natural language — "show me the reading chair" — and the system grounds your query directly in the 3D geometry, highlighting the relevant Gaussians in real time. Every Gaussian in the scene carries a CLIP semantic embedding, enabling continuous, multi-resolution spatial understanding without render-to-2D hacks.

Built at [JacHacks](https://jachacks.com) (April 2026).

---

## Demo

| Natural Language Query | Scene Editing | Spatial Knowledge Graph |
|:---:|:---:|:---:|
| Ask "where is the lamp?" and matching Gaussians light up in 3D | Say "turn the couch red" and colors change in real time | Objects and spatial relationships auto-discovered |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (React + Vite)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Spark.js     │  │   Sidebar    │  │    Chat Interface      │ │
│  │  3D Viewport  │  │  Knowledge   │  │    Query + Edit +      │ │
│  │  5.2M Gauss.  │  │  Graph Viz   │  │    Explore Controls    │ │
│  └──────┬───────┘  └──────────────┘  └───────────┬────────────┘ │
└─────────┼────────────────────────────────────────┼──────────────┘
          │  Highlight indices + regions            │  Queries
          ▼                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Server (Python 3.11)                   │
│                                                                  │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ CLIP        │  │ Dense Query  │  │ Gemini Agent             │ │
│  │ Encoder     │──│ Pipeline     │──│ (tool-calling, optional) │ │
│  │ (ViT-B/32) │  │ DBSCAN + K-D │  │ find / highlight /       │ │
│  └────────────┘  │ Tree cluster │  │ recolor / move           │ │
│                  └──────┬───────┘  └──────────────────────────┘ │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────────┐ │
│  │              Gaussian Store (in-memory)                      │ │
│  │  semantic_0.ply — whole objects    (5.2M × 512-dim CLIP)    │ │
│  │  semantic_1.ply — object parts     (decoded via VAE)        │ │
│  │  semantic_2.ply — fine subparts                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
          ▲
          │  LangSplat Training (Colab A100)
          │  COLMAP → 3DGS → SAM masks → CLIP embeddings
          │  Outputs: semantic_*.ply + autoencoder.pth
```

### Query Pipeline

1. **Encode** — User text is encoded to a 512-dim CLIP vector via OpenCLIP ViT-B/32
2. **Score** — LERF-style relevancy scoring against all Gaussians, with canonical negatives ("wall", "floor", "ceiling") to suppress background
3. **Cluster** — DBSCAN groups spatially coherent high-scoring Gaussians into object instances
4. **Ground** — Instances are materialized with bounding spheres, centroids, and confidence scores
5. **Highlight** — Per-Gaussian indices and colors are sent to the browser; Spark.js paints them in the 3D viewport
6. **Reason** — Optional Gemini agent performs multi-step tool-calling for complex spatial queries

### Three Semantic Levels

LangSplat embeds CLIP features at three hierarchical scales via SAM-generated masks:

| Level | Granularity | Example |
|-------|-------------|---------|
| **0** | Whole objects | chair, table, couch |
| **1** | Object parts | armrest, table leg, cushion |
| **2** | Fine subparts | button, zipper, book spine |

The system dynamically selects the best level per query, or can be explicitly controlled.

---

## Tech Stack

### Frontend — `room-cognito/`

| Technology | Version | Purpose |
|------------|---------|---------|
| [React](https://react.dev) | 18.3 | UI framework (functional components + hooks) |
| [Vite](https://vite.dev) | 5.4 | Build tool + dev server (port 8080) |
| [TypeScript](https://typescriptlang.org) | 5.8 | Type safety |
| [Three.js](https://threejs.org) | 0.178 | 3D scene graph, camera, WebGL renderer |
| [Spark.js](https://sparkjs.dev) | 0.1.10 | Gaussian splat rendering (by World Labs) |
| [Tailwind CSS](https://tailwindcss.com) | 3.4 | Utility-first styling (dark theme) |
| [shadcn/ui](https://ui.shadcn.com) | — | Component library (Radix UI primitives) |
| [TanStack Query](https://tanstack.com/query) | 5.83 | Async state management + polling |
| [Recharts](https://recharts.org) | 2.15 | Data visualization |
| [React Router](https://reactrouter.com) | 6.30 | Client-side routing |

### Backend — `server/`

| Technology | Version | Purpose |
|------------|---------|---------|
| [FastAPI](https://fastapi.tiangolo.com) | 0.135 | REST API server (async, OpenAPI docs) |
| [Uvicorn](https://uvicorn.org) | latest | ASGI server |
| [PyTorch](https://pytorch.org) | ≥2.0 | Tensor ops, autoencoder inference |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | 3.3.0 | CLIP text encoding (ViT-B/32) |
| [scikit-learn](https://scikit-learn.org) | ≥1.8 | DBSCAN spatial clustering |
| [SciPy](https://scipy.org) | ≥1.17 | KDTree, convex hull, spatial utilities |
| [NumPy](https://numpy.org) | ≥2.0 | Vectorized cosine similarity, array ops |
| [plyfile](https://github.com/dranjan/python-plyfile) | ≥0.9 | Gaussian PLY file parsing |
| [Pydantic](https://docs.pydantic.dev) | ≥2.0 | Settings + request/response validation |
| [Google Gemini](https://ai.google.dev) | 2.5 Flash | Tool-calling agent for complex queries |

### Training Pipeline — `colab/`

| Technology | Purpose |
|------------|---------|
| [LangSplat](https://github.com/minghanqin/LangSplat) | Hierarchical CLIP embedding into Gaussians |
| [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Base RGB scene reconstruction |
| [SAM](https://github.com/facebookresearch/segment-anything) (v1) | Multi-scale mask generation |
| [COLMAP](https://colmap.github.io) | Structure-from-Motion camera poses |
| [Scaniverse](https://scaniverse.com) | Scene capture (iOS/Android) |
| Google Colab A100 | GPU compute for training |

---

## Features

### Natural Language Scene Querying
Type any query — "find the bookshelf", "where are the plants?" — and matching Gaussians illuminate in the 3D viewport with bounding regions and confidence scores.

### Real-Time Scene Editing
Say "turn the couch blue" or "make the lamp red" and the system parses the color, grounds the object, and recolors the Gaussians instantly via Spark.js's `setSplat()` API.

### Autonomous Exploration
An exploration walker traverses the scene, discovers objects via multi-level CLIP queries, builds a catalog, and populates a spatial knowledge graph — all shown progressively in the UI.

### Spatial Knowledge Graph
Discovered objects are connected with spatial relationships (left_of, above, behind, near) computed from Gaussian cluster centroids. Visualized as a force-directed graph in the sidebar and as 3D sphere overlays in the viewport.

### Multi-Level Semantic Control
Query at whole-object, part, or subpart granularity. Ask for "the chair" (level 0), "the armrest" (level 1), or "the button on the cushion" (level 2).

### Gemini Agent Integration
Complex queries ("what furniture is near the window?") route to a Gemini tool-calling agent that composes multiple scene lookups, spatial reasoning, and natural language answers.

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `R` | Reset camera to default view |
| `S` | Save current camera view |
| `1`–`9` | Jump to saved camera views |

---

## Project Structure

```
spatialMind/
├── room-cognito/                  # Frontend application
│   ├── src/
│   │   ├── components/
│   │   │   ├── SparkRenderer.tsx   # Spark.js + Three.js 3D viewport
│   │   │   ├── Sidebar.tsx         # Object list + knowledge graph
│   │   │   ├── BottomPanels.tsx    # Chat interface + memory
│   │   │   ├── GraphOverlay3D.tsx  # 3D graph node rendering
│   │   │   ├── StatusBar.tsx       # Connection + Gaussian count
│   │   │   └── TopStatusBar.tsx    # Query status indicator
│   │   ├── pages/
│   │   │   ├── Landing.tsx         # Hero / marketing page
│   │   │   └── Index.tsx           # Main dashboard (3D + chat)
│   │   ├── data/mockData.ts        # TypeScript interfaces
│   │   └── lib/config.ts           # API URL configuration
│   ├── public/
│   │   └── scene-indexed.splat     # Pre-processed splat for rendering
│   └── package.json
│
├── server/                         # Python backend
│   ├── main.py                     # FastAPI app + lifespan startup
│   ├── config.py                   # Pydantic settings
│   ├── routers/
│   │   ├── adapter.py              # Frontend-facing API (/api/*)
│   │   ├── clip.py                 # CLIP encode/highlight/probability
│   │   ├── scene.py                # Scene graph building
│   │   ├── walker.py               # Explore + query walkers
│   │   └── health.py               # Health check
│   ├── services/
│   │   ├── gaussian_store.py       # PLY loading + VAE decoding
│   │   ├── clip_encoder.py         # OpenCLIP text encoder
│   │   ├── similarity.py           # LERF relevancy scoring
│   │   ├── dense_query.py          # Primary query pipeline
│   │   ├── query_node_builder.py   # Multi-level DBSCAN grounding
│   │   ├── instance_cache.py       # Cross-query instance memory
│   │   ├── clustering.py           # DBSCAN wrapper
│   │   ├── spatial_relations.py    # Spatial relationship computation
│   │   ├── graph_builder.py        # Scene graph construction
│   │   ├── gemini_agent.py         # Gemini tool-calling agent
│   │   ├── exploration_walker.py   # Autonomous scene explorer
│   │   └── backboard_client.py     # Persistent memory service
│   ├── models/
│   │   └── autoencoder.py          # VAE decoder (3-dim → 512-dim)
│   └── requirements.txt
│
├── data/scene/                     # LangSplat artifacts (~4GB)
│   ├── semantic_0.ply              # Whole-object embeddings (1.3GB)
│   ├── semantic_1.ply              # Part-level embeddings (1.3GB)
│   ├── semantic_2.ply              # Subpart-level embeddings (1.3GB)
│   └── autoencoder.pth             # VAE decoder checkpoint (1.7MB)
│
├── colab/
│   └── spatialMind_training.ipynb  # LangSplat training pipeline
│
└── scripts/
    ├── demo_launch.py              # Local demo startup
    ├── demo_e2e_test.py            # End-to-end integration tests
    ├── ply_to_splat.py             # PLY → SPZ conversion
    └── verify_artifacts.py         # Artifact validation
```

---

## Getting Started

### Prerequisites

- **Python** 3.11+
- **Node.js** 18+ (with pnpm)
- **GPU** recommended for server (CUDA-capable; falls back to CPU)
- **~4GB disk** for scene artifacts

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/spatialMind.git
cd spatialMind
```

### 2. Set Up the Backend

```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key    # Optional — enables Gemini agent
BACKBOARD_API_KEY=your_key            # Optional — enables persistent memory
```

### 3. Prepare Scene Data

Place LangSplat-trained artifacts in `data/scene/`:

```
data/scene/
├── semantic_0.ply
├── semantic_1.ply
├── semantic_2.ply
└── autoencoder.pth
```

> To train your own scene, see the [Training Pipeline](#training-your-own-scene) section below.

### 4. Set Up the Frontend

```bash
cd room-cognito
pnpm install
```

### 5. Run

**Terminal 1 — Backend** (from project root):

```bash
cd server
uvicorn server.main:app --host 0.0.0.0 --port 8001
```

Startup loads all PLY files and decodes embeddings (~30s on GPU, longer on CPU).

**Terminal 2 — Frontend** (from project root):

```bash
cd room-cognito
pnpm dev
```

Open **http://localhost:8080** in your browser.

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status, Gaussian count, pipeline readiness |
| `POST` | `/clip/encode` | Encode text → top-K matching Gaussian indices |
| `POST` | `/clip/highlight` | Per-Gaussian opacity mask (soft/hard modes) |
| `POST` | `/clip/probability` | Probability heatmap over all Gaussians |
| `POST` | `/scene/build` | Materialize instance graph with spatial relations |
| `POST` | `/walker/explore` | Run autonomous exploration walker |
| `POST` | `/walker/query` | Ground a query and return answer + highlights |

### Frontend Adapter (`/api/*`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health proxy |
| `POST` | `/api/explore/start` | Start async exploration |
| `GET` | `/api/explore/status` | Poll exploration progress |
| `POST` | `/api/query/start` | Start async query (returns job ID) |
| `GET` | `/api/query/status/{job_id}` | Poll query progress with step tracking |
| `POST` | `/api/query` | Synchronous query (Gemini agent if enabled) |
| `GET` | `/api/scenes` | List available scenes |
| `GET` | `/api/scene/splat` | Serve splat/PLY file for Spark.js |
| `POST` | `/api/clip/highlight` | Highlight regions from query |
| `POST` | `/api/edit` | Scene editing (recolor objects) |

Full OpenAPI docs available at `http://localhost:8001/docs` when the server is running.

---

## Training Your Own Scene

### 1. Capture

Use [Scaniverse](https://scaniverse.com) (iOS/Android) to capture a room. Export the images and camera calibration data.

### 2. Structure from Motion

Run COLMAP to compute camera poses:

```bash
colmap automatic_reconstructor \
  --workspace_path ./scene \
  --image_path ./scene/images
```

### 3. Train Base Gaussians

Train a standard 3D Gaussian Splatting model using the [graphdeco-inria implementation](https://github.com/graphdeco-inria/gaussian-splatting).

### 4. Train LangSplat

Use the Colab notebook at `colab/spatialMind_training.ipynb` on an A100 GPU:

1. Generate SAM masks at three scales
2. Extract CLIP features per mask
3. Train the autoencoder (3-dim latent → 512-dim CLIP)
4. Produce `semantic_0.ply`, `semantic_1.ply`, `semantic_2.ply`, and `autoencoder.pth`

### 5. Deploy

Copy the four output files to `data/scene/` and restart the server.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **All CLIP computation server-side** | ViT-B/32 is ~350MB; running cosine similarity over 5.2M embeddings is infeasible in-browser |
| **LERF-style relevancy scoring** | Canonical negatives ("wall", "floor", "ceiling") dramatically improve grounding precision by suppressing background |
| **DBSCAN over K-Means** | No need to specify cluster count; handles arbitrary object shapes; built-in noise rejection |
| **3-dim VAE latents in PLY** | Compresses 512-dim CLIP vectors to 3 floats per Gaussian in storage; decoded at server startup |
| **Spark.js `maxSh = 0`** | Disabling Spherical Harmonics is required to make `setSplat()` color overrides visible |
| **Multi-level semantic hierarchy** | Different queries need different granularity; "chair" vs "armrest" vs "button" |

---

## Acknowledgments

- [LangSplat](https://github.com/minghanqin/LangSplat) — Hierarchical language-embedded Gaussian splatting
- [Spark.js](https://sparkjs.dev) by World Labs — Browser-based Gaussian splat renderer
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — Open-source CLIP implementation
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Original 3DGS paper (SIGGRAPH 2023)
- [LERF](https://www.lerf.io/) — Language Embedded Radiance Fields (relevancy scoring approach)

---

## License

This project was built at JacHacks 2026. See [LICENSE](LICENSE) for details.
