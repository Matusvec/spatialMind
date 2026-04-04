<!-- GSD:project-start source:PROJECT.md -->
## Project

**SpatialMind**

An autonomous AI agent system that sees, navigates, and reasons inside photorealistic 3D worlds. Built on JAC with LangSplat-powered Gaussian splatting, SpatialMind turns photorealistic 3D scenes into queryable, structured spatial knowledge graphs — where every Gaussian carries a semantic embedding and the entire scene is a continuous, multi-resolution semantic field. Built for JacHacks hackathon (April 4-5, 2026).

**Core Value:** **An AI that understands 3D space as a structured graph, queryable with natural language, powered by semantically-embedded Gaussians — not hacky render-to-2D-and-back pipelines.**

If everything else fails, a user must be able to load a LangSplat scene, type a natural language query, and see the relevant objects light up in the 3D view with spatial relationships explained.

### Constraints

- **Timeline**: 24 hours (hackathon)
- **Compute**: LangSplat training requires Colab A100 — must kick off early, blocking dependency
- **JAC familiarity**: Learning on the fly — favor simple patterns, lean on `by llm()` for complex logic
- **Team size**: 2 people — backend/frontend split, minimize integration surface
- **Scenes**: Need to capture with Scaniverse before/during hackathon, then train with LangSplat
- **Browser rendering**: Spark.js handles visual splat only — all semantic computation is server-side
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
### Overview
| Environment | Purpose | Runtime |
|-------------|---------|---------|
| Training (Colab A100) | LangSplat CLIP embedding generation | Python 3.10, CUDA 11.8 |
| Python Server | CLIP queries, DBSCAN clustering, JAC graph | Python 3.11+, FastAPI |
| JAC Runtime | Walker agents, spatial reasoning, `by llm()` | Jaseci 2.3.12 |
| Browser | Splat rendering, graph viz, chat UI | Vite + React + Spark.js |
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
### JAC Runtime (Spatial Agents)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| jaseci (meta-package) | 2.3.12 | JAC language runtime + byLLM + server | Released April 3 2026; installs jaclang 0.13.5 + byllm + jac-client; Python >=3.11 |
| jaclang | 0.13.5 | Jac language compiler/interpreter | Core walker/node/edge graph execution engine |
| byllm | bundled with jaseci | `by llm()` LLM integration | Turns typed function signatures into auto-generated LLM prompts; no prompt engineering required |
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
### Scaniverse (Capture)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Scaniverse | iOS/Android (latest) | On-device photo capture → PLY/SPZ | Free app; exports images + camera calibration; .PLY export compatible with COLMAP workflow; 100% on-device processing |
## Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| httpx | 0.28.x | Async HTTP from Python server | Frontend → Python server communication during development; also for JAC → Python server calls |
| python-dotenv | 1.x | Environment variable loading | API keys for Backboard, OpenAI/Anthropic in the JAC runtime |
| Backboard Python SDK / REST API | latest | Persistent spatial memory across sessions | Store walker-discovered objects and relationships between sessions; use `docs.backboard.io/sdk/quickstart` for integration |
| websockets (via FastAPI) | native | Push highlight masks to frontend in real-time | FastAPI natively supports WebSocket endpoints; avoids polling for query results |
| pynput / pillow | optional | Debug image overlays during training | Only during development pipeline debugging |
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
## Installation
### Training Environment (Colab A100)
# Force CUDA 11.8 on Colab
# Clone LangSplat with submodules
# Install SAM (LangSplat fork)
# Download SAM checkpoint
### Python Server
### JAC Runtime
# Requires Python >=3.11
### Frontend
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
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
