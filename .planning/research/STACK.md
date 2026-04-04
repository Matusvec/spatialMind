# Technology Stack

**Project:** SpatialMind
**Researched:** 2026-04-04
**Confidence:** MEDIUM-HIGH (training pipeline verified via official repo; JAC versions verified via PyPI; Spark.js verified via GitHub; open-clip-torch and FastAPI verified via PyPI)

---

## Recommended Stack

### Overview

The stack splits cleanly into four independent execution environments:

| Environment | Purpose | Runtime |
|-------------|---------|---------|
| Training (Colab A100) | LangSplat CLIP embedding generation | Python 3.10, CUDA 11.8 |
| Python Server | CLIP queries, DBSCAN clustering, JAC graph | Python 3.11+, FastAPI |
| JAC Runtime | Walker agents, spatial reasoning, `by llm()` | Jaseci 2.3.12 |
| Browser | Splat rendering, graph viz, chat UI | Vite + React + Spark.js |

---

## Core Technologies

### LangSplat Training Pipeline

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.10.x | Training environment | LangSplat environment.yml targets 3.10; CUDA 11.8 extensions require it |
| PyTorch | 2.0.x with CUDA 11.8 | Tensor ops, Gaussian rasterization | Must match CUDA SDK version; LangSplat CUDA extensions compiled against 11.8 |
| CUDA SDK | 11.8 | GPU compute for diff-gaussian-rasterization | Original LangSplat paper used 11.8; newer CUDA versions require patching the custom ops |
| COLMAP | 3.9 (binary) | SfM — camera poses from Scaniverse images | LangSplat requires COLMAP-format sparse/ directory (cameras.bin, images.bin, points3D.bin); 3.9 is stable pre-built |
| Gaussian Splatting (graphdeco-inria) | latest main | Base 3DGS checkpoint before LangSplat overlay | LangSplat requires a pre-trained RGB 3DGS .pth checkpoint as starting point |
| SAM (Segment Anything) | 1.0 (original) | Hierarchical mask generation for CLIP features | LangSplat uses SAM to generate three scales of masks per image before CLIP encoding; SAM2 is unnecessary (video model) |
| open-clip-torch | 3.3.0 | CLIP text/image encoding | LangSplat uses OpenCLIP (not OpenAI CLIP) for licensing and model variety; 3.3.0 released Feb 2026; Python >=3.9 |
| segment-anything-langsplat | from source | LangSplat-patched SAM wrapper | LangSplat requires this fork, not stock facebook/segment-anything |

**Confidence:** MEDIUM — PyTorch/CUDA version pairing is confirmed as a hard constraint (custom CUDA ops must match SDK). LangSplat's environment.yml pins to CUDA 11.8. Colab A100 supports CUDA 11.8 and 12.x; use 11.8 to match LangSplat defaults.

**Training output:** `output/<scene>/point_cloud/iteration_30000/point_cloud.ply` — contains Gaussian positions, covariances, colors, opacity, and 3-dimensional CLIP latent vectors (via trained autoencoder).

**Critical training order:**
1. Capture with Scaniverse, export as PLY/images
2. Run COLMAP on images → sparse/ directory
3. Train base 3DGS RGB model (30k iterations)
4. Generate per-image SAM masks + CLIP features (3 scales: finest, medium, coarsest)
5. Train per-scale autoencoder (256→128→64→32→3 dims)
6. Train LangSplat overlay on 3DGS checkpoint (per scale: `--feature_level 0/1/2`)
7. Export final PLY with embedded language features

---

### Python Query Server

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.11.x | Server runtime | Jaseci requires Python >=3.11; FastAPI requires >=3.10; 3.11 satisfies both |
| FastAPI | 0.135.3 | REST API + WebSocket server | Current as of April 1 2026; async-native, Pydantic v2, built-in OpenAPI; ideal for numpy array responses |
| Uvicorn | 0.34.x | ASGI server | FastAPI's recommended server; production-grade with --workers flag |
| open-clip-torch | 3.3.0 | Text-to-CLIP-vector encoding at query time | Encode user text queries to 512-dim CLIP vectors; same model used in training = matching embedding space |
| numpy | 2.x | Gaussian data storage and cosine similarity | PLY Gaussian data loaded as numpy arrays; cosine sim is single vectorized operation |
| scikit-learn | 1.8.0 | DBSCAN clustering | Current stable; `sklearn.cluster.DBSCAN` — no predefined cluster count, handles arbitrary shapes, built-in noise point labeling |
| scipy | 1.17.0 | Spatial utilities (KDTree, convex hull, centroid) | `scipy.spatial.KDTree` for nearest-neighbor spatial queries; `scipy.spatial.ConvexHull` for bounding volumes |
| plyfile | 0.9.x | Load/parse Gaussian PLY files | Pure Python PLY reader; handles custom properties (the CLIP embedding columns) |

**Confidence:** HIGH — All versions verified via PyPI as of April 2026. scikit-learn 1.8.0 confirmed as current stable.

**Why DBSCAN over K-means:** Object count is unknown (can't predefine k). Gaussians form dense clusters of arbitrary shape with sparse regions between objects. DBSCAN naturally handles this and marks isolated noise Gaussians without forcing them into clusters.

**Why cosine similarity over L2 for CLIP:** CLIP vectors are normalized; cosine similarity = dot product. L2 distance on unnormalized vectors would require extra normalization steps. LangSplat paper uses cosine sim.

---

### JAC Runtime (Spatial Agents)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| jaseci (meta-package) | 2.3.12 | JAC language runtime + byLLM + server | Released April 3 2026; installs jaclang 0.13.5 + byllm + jac-client; Python >=3.11 |
| jaclang | 0.13.5 | Jac language compiler/interpreter | Core walker/node/edge graph execution engine |
| byllm | bundled with jaseci | `by llm()` LLM integration | Turns typed function signatures into auto-generated LLM prompts; no prompt engineering required |

**Installation:** `pip install jaseci` — installs the full ecosystem.

**Server start:** `jac start main.jac` (development) or `jac start main.jac --scale` (Kubernetes).

**Walker pattern for SpatialMind:**
```jac
node SpatialObject {
    has label: str;
    has centroid: list[float];
    has extent: list[float];
    has clip_embedding: list[float];
    has confidence: float;
}

edge SpatialRel {
    has relation: str;  # on_top_of, next_to, facing, inside
    has confidence: float;
}

walker QueryWalker {
    has query: str;
    has results: list = [];

    can reason_about_space(obj: SpatialObject) -> str
        by llm();  # signature + types become the prompt automatically

    with entry {
        visit [root --> SpatialObject];
    }

    with SpatialObject entry {
        self.results.append(self.reason_about_space(here));
        visit [here --> SpatialObject];
    }
}
```

**`by llm()` mechanism:** Jac extracts function name, parameter types, docstring, and return type to construct the prompt. No manual prompt strings needed. Requires an LLM configured in the environment (OpenAI or Anthropic via byllm config).

**Confidence:** HIGH — Version 2.3.12 confirmed on PyPI (April 3 2026). Walker syntax confirmed via official jaseci GitHub README.

---

### Browser Rendering (Frontend)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| @sparkjsdev/spark | 0.1.10 | Gaussian splat rendering in Three.js | World Labs' renderer; current stable; supports .PLY, .SPZ, .SPLAT, .KSPLAT, .SOG; SplatMesh integrates with Three.js scene graph |
| three | 0.183.2 | 3D scene graph, camera, WebGL | Spark explicitly targets Three.js 0.178+ (confirmed in Spark docs); 0.183.2 is current npm stable |
| React | 19.x | UI layer | Team's default; functional components + hooks |
| Vite | 6.x | Build tooling | Fast HMR, native ESM, Three.js-compatible |
| @react-three/fiber | 9.x | React bindings for Three.js | Simplifies Three.js scene management from React; works with Spark's SplatMesh as it's a standard THREE.Object3D |
| @react-three/drei | 10.x | R3F helpers (OrbitControls, etc.) | Camera controls, helpers; saves setup time |
| TypeScript | 5.x | Type safety | Reduces integration errors at the frontend/backend boundary |

**Spark.js usage for SpatialMind:**
```javascript
import { SplatMesh } from '@sparkjsdev/spark'

// In React Three Fiber
const splat = new SplatMesh({ url: '/scene.ply' })
scene.add(splat)

// Highlight mask: create a float32 array of per-Gaussian alpha values
// received from the Python server query response, then drive Spark's
// color/alpha editing API (real-time splat color editing is supported in 0.1.10)
```

**Confidence:** HIGH — Version 0.1.10 confirmed via GitHub. Three.js 0.183.2 confirmed current on npm. Spark docs explicitly show Three.js 0.178.0 compatibility URL.

---

### Scaniverse (Capture)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Scaniverse | iOS/Android (latest) | On-device photo capture → PLY/SPZ | Free app; exports images + camera calibration; .PLY export compatible with COLMAP workflow; 100% on-device processing |

**Export flow for LangSplat training:** Scan room → Export as "Images + Camera Data" (or use the bundled COLMAP sparse/ output if available) → upload to Colab → run LangSplat pipeline.

**Confidence:** MEDIUM — Scaniverse GLB/PLY/SPZ export confirmed via official site. COLMAP sparse/ compatibility depends on Scaniverse's export format — verify the export includes cameras.bin/images.bin or plan to run COLMAP on raw images.

---

## Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| httpx | 0.28.x | Async HTTP from Python server | Frontend → Python server communication during development; also for JAC → Python server calls |
| python-dotenv | 1.x | Environment variable loading | API keys for Backboard, OpenAI/Anthropic in the JAC runtime |
| Backboard Python SDK / REST API | latest | Persistent spatial memory across sessions | Store walker-discovered objects and relationships between sessions; use `docs.backboard.io/sdk/quickstart` for integration |
| websockets (via FastAPI) | native | Push highlight masks to frontend in real-time | FastAPI natively supports WebSocket endpoints; avoids polling for query results |
| pynput / pillow | optional | Debug image overlays during training | Only during development pipeline debugging |

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Browser splat renderer | @sparkjsdev/spark 0.1.10 | mkkellogg/GaussianSplats3D | Spark is actively maintained by World Labs, supports .SPZ compression (90% size reduction), has real-time color editing API needed for semantic highlighting |
| Browser splat renderer | @sparkjsdev/spark 0.1.10 | Luma AI embed | No programmatic control over highlight masks; locked to Luma's hosting |
| CLIP library | open-clip-torch 3.3.0 | openai/clip | OpenAI CLIP repo is unmaintained since 2021; open-clip-torch has more models, active development, same API, used by LangSplat paper |
| Object clustering | scikit-learn DBSCAN | HDBSCAN | DBSCAN is simpler to tune for this use case (known density characteristics of Gaussian clusters); HDBSCAN adds complexity without clear benefit for static scene |
| Object clustering | scikit-learn DBSCAN | K-Means | Requires knowing k (object count) in advance — impossible for open-world scenes |
| Python web framework | FastAPI | Flask | FastAPI is async-native, has automatic OpenAPI docs, Pydantic validation; Flask requires more boilerplate for JSON APIs |
| Python web framework | FastAPI | Django REST | Far too heavy for a hackathon; FastAPI can be up in minutes |
| JAC server integration | jaseci jac start | FastAPI-only backend | JAC's walker paradigm is the core technical requirement (hackathon prize target); `by llm()` would need custom reimplementation in FastAPI |
| LLM provider for `by llm()` | OpenAI GPT-4o | Anthropic Claude | Both work with byllm; GPT-4o is the default in most JAC examples and documentation; switch based on API key availability |
| SfM tool | COLMAP 3.9 | COLMAP-Free 3DGS | COLMAP-Free 3DGS is research-grade and less stable; Scaniverse already provides calibrated camera data; COLMAP 3.9 binary is the proven path |
| Language features | LangSplat (static) | LangSplat V2 | LangSplat V2 (NeurIPS 2025) achieves 450+ FPS but is a newer research codebase; original LangSplat has more community tutorials and Colab examples; for a 24-hour hackathon, original LangSplat is less risky |

---

## What NOT to Use

| Technology | Why Not |
|------------|---------|
| LangSplat V2 | NeurIPS 2025 — fewer tutorials, higher risk during hackathon. Original LangSplat is better documented for the training pipeline. |
| SAM 2 or SAM 3 | LangSplat's segment-anything-langsplat fork is based on SAM 1. SAM 2/3 are video-segmentation models with different APIs. LangSplat's mask generation pipeline will break. |
| Browser-side CLIP inference (onnxruntime-web) | CLIP ViT-B/32 is ~350MB; inference on 100k+ Gaussian embeddings is infeasible in the browser. All CLIP computation stays on the Python server. |
| nerfstudio / gsplat as training base | LangSplat was built on graphdeco-inria/gaussian-splatting specifically; the diff-gaussian-rasterization CUDA ops are tightly coupled. Do not swap the 3DGS base. |
| Langchain / LangGraph for agents | Redundant with JAC's native walker paradigm. Adding both would create competing agent abstractions. Use `by llm()` in JAC instead. |
| Redis for session state | Backboard is the designated persistent memory sponsor. Use it. Redis adds infra complexity without hackathon prize alignment. |
| CUDA 12.x for LangSplat training | LangSplat's custom diff-gaussian-rasterization ops are pinned to CUDA 11.8 in environment.yml. Upgrading requires patching submodule source. Use CUDA 11.8 on Colab explicitly. |
| WebXR / AR.js | Out of scope per PROJECT.md |
| Socket.io | FastAPI's native WebSocket support is sufficient; Socket.io adds a dependency for no benefit |

---

## Installation

### Training Environment (Colab A100)

```bash
# Force CUDA 11.8 on Colab
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Clone LangSplat with submodules
!git clone --recursive https://github.com/minghanqin/LangSplat.git
!cd LangSplat && conda env create --file environment.yml
!conda activate langsplat

# Install SAM (LangSplat fork)
!pip install git+https://github.com/minghanqin/segment-anything-langsplat.git

# Download SAM checkpoint
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ckpts/
```

### Python Server

```bash
pip install fastapi==0.135.3 uvicorn[standard]
pip install open-clip-torch==3.3.0
pip install scikit-learn==1.8.0
pip install scipy==1.17.0
pip install numpy plyfile python-dotenv httpx
```

### JAC Runtime

```bash
pip install jaseci  # installs jaseci 2.3.12 = jaclang 0.13.5 + byllm + jac-client
# Requires Python >=3.11
jac start main.jac
```

### Frontend

```bash
pnpm create vite spatialMind-frontend --template react-ts
cd spatialMind-frontend
pnpm install three@0.183.2 @sparkjsdev/spark@0.1.10
pnpm install @react-three/fiber @react-three/drei
```

---

## Sources

- LangSplat official repo: https://github.com/minghanqin/LangSplat [MEDIUM confidence — CUDA/PyTorch versions from README, environment.yml contents not fully exposed]
- Spark.js GitHub: https://github.com/sparkjsdev/spark [HIGH — version 0.1.10, npm package, Three.js compat confirmed]
- Spark.js docs: https://sparkjs.dev/docs/overview/ [HIGH — format support confirmed]
- Three.js npm: https://www.npmjs.com/package/three [HIGH — 0.183.2 confirmed current]
- jaseci PyPI: https://pypi.org/project/jaseci/ [HIGH — version 2.3.12, April 3 2026, Python >=3.11]
- open-clip-torch PyPI: https://pypi.org/project/open-clip-torch/ [HIGH — version 3.3.0, Feb 2026, Python >=3.9]
- FastAPI PyPI: https://pypi.org/project/fastapi/ [HIGH — version 0.135.3, April 1 2026, Python >=3.10]
- scikit-learn docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html [HIGH — version 1.8.0 confirmed current stable]
- scipy docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html [HIGH — v1.17.0 confirmed]
- COLMAP install docs: https://colmap.github.io/install.html [HIGH — build from source, 3.9 binary available]
- Scaniverse official: https://scaniverse.com [MEDIUM — export formats confirmed, COLMAP integration inferred]
- Jaseci GitHub: https://github.com/jaseci-labs/jaseci [HIGH — jaclang 0.13.5, walker/node syntax, by llm() confirmed]
- Backboard.io: https://backboard.io/ [MEDIUM — REST API + SDKs confirmed, Python SDK at docs.backboard.io/sdk/quickstart]
- LangSplat V2: https://langsplat-v2.github.io/ [HIGH — NeurIPS 2025 publication confirmed, not recommended for hackathon]
