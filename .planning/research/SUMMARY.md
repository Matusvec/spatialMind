# Project Research Summary

**Project:** SpatialMind
**Domain:** Spatial AI — 3D scene understanding with language-embedded Gaussian splatting and JAC graph-walker agents
**Researched:** 2026-04-04
**Confidence:** MEDIUM-HIGH

## Executive Summary

SpatialMind is a technically novel spatial AI system that combines LangSplat (CVPR 2024) with JAC's walker agent paradigm to let users query a photorealistic 3D scene in natural language. The architecture is fundamentally a four-layer pipeline: an offline training layer (LangSplat on Colab A100), a Python semantic compute server (FastAPI + CLIP + DBSCAN), a JAC graph reasoning layer (walker agents + `by llm()`), and a browser rendering layer (Spark.js + React). Each layer has a hard boundary and communicates strictly over REST — this separation is both an architectural virtue and a practical requirement for parallel development under hackathon time pressure.

The recommended approach is to accept the training pipeline as a blocking dependency and start it within the first hour of the hackathon, then build all other layers in parallel against pre-existing LangSplat demo scenes. The critical path runs: scene capture → COLMAP → 3DGS RGB training → LangSplat autoencoder + overlay → PLY artifact. This blocking step takes 3–6 hours on Colab A100. The graph layer (DBSCAN → JAC nodes/edges) is built once at scene load time, not per query, which is the key performance insight that makes interactive querying viable. All CLIP computation stays server-side; the browser receives only highlight masks and natural language descriptions.

The primary risks are infrastructure failures, not algorithmic ones: Colab session timeouts losing training progress, COLMAP failing on low-texture rooms, DBSCAN epsilon miscalibration producing meaningless object clusters, and browser memory exhaustion from uncompressed PLY files. Every one of these risks has a known mitigation strategy and a known fallback. The secondary risk is building features in the wrong order — the feature dependency tiers are strict, and attempting agentic features before the CLIP query pipeline is verified will waste several hours debugging the wrong layer. Build in tier order and ship incrementally.

---

## Key Findings

### Recommended Stack

The stack separates into four execution environments with no shared runtime. Training runs on Python 3.10 + CUDA 11.8 (hard constraint — LangSplat's custom CUDA ops are pinned to 11.8). The Python server runs on 3.11+ (required by jaseci). The JAC runtime installs as a single `pip install jaseci` meta-package (v2.3.12, April 2026). The browser uses Vite + React 19 + Three.js 0.183.2 + Spark.js 0.1.10. All versions are confirmed current as of April 2026.

**Core technologies:**
- **Python 3.10 / CUDA 11.8 (training):** LangSplat's CUDA extensions are compiled against 11.8 — using any other version requires patching submodule source code.
- **Python 3.11 / FastAPI 0.135.3 (server):** Satisfies both jaseci (>=3.11) and FastAPI (>=3.10); async-native, Pydantic v2, automatic OpenAPI.
- **jaseci 2.3.12 (JAC runtime):** One-command install; includes jaclang 0.13.5 + byllm + jac-client; walkers become REST endpoints automatically via `jac start`.
- **open-clip-torch 3.3.0:** Same model used in LangSplat training — critical for matching embedding spaces between training and query time.
- **scikit-learn 1.8.0 DBSCAN:** Object count is unknown, shapes are arbitrary — DBSCAN is the correct algorithm; K-means requires knowing k.
- **@sparkjsdev/spark 0.1.10:** World Labs' renderer; supports .PLY/.SPZ/.KSPLAT; real-time per-Gaussian color editing via Dyno shader graph — required for highlight masks.
- **Three.js 0.183.2:** Spark.js peer dependency (requires 0.178+); 0.183.2 is current npm stable.

Do NOT use: SAM2/SAM3 (breaks LangSplat's fork), CUDA 12.x for training, LangSplat V2 (NeurIPS 2025 — too new, fewer tutorials), browser-side CLIP inference (350MB model, no GPU), LangChain/LangGraph (redundant with JAC walkers), or Redis (Backboard is the designated persistence sponsor).

### Expected Features

Research identifies a strict feature dependency hierarchy across five tiers. All features within a tier depend on the tier below being complete and verified.

**Must have (table stakes — Tier 0-3):**
- Natural language object query via CLIP text-to-embedding cosine similarity
- Semantic Gaussian highlighting in Spark.js viewer (per-splat color/opacity via Dyno)
- 3D scene load and render from .PLY/.SPZ
- Object clustering via DBSCAN producing centroids and bounding boxes
- JAC spatial knowledge graph with SpatialObject nodes and SpatialRelation edges
- Spatial relationship detection (on_top_of, next_to, inside, facing) from cluster geometry
- Chat interface wired to JAC QueryWalker
- Graph visualization overlay (nodes as spheres, edges as lines in Three.js)
- Natural language description via `by llm()` on the matched sub-graph

**Should have (differentiators — Tier 4):**
- Autonomous ExplorationWalker that traverses all unvisited nodes and catalogs objects — critical for Agentic AI Track prize
- Persistent spatial memory via Backboard thread per session — critical for Best Use of Backboard prize
- Probability cloud layer (soft per-Gaussian cosine scores as continuous opacity gradient) — single most impressive visual differentiator

**Defer (stretch / v2):**
- Multi-hop relational queries ("what is on top of the desk near the window") — technically impressive but high implementation risk in 24 hours
- Multi-resolution semantic queries — demo with 2-3 curated examples rather than a full system
- Navigation path camera animation — implement only if Tiers 1-4 are complete
- Base44 integration — no prize upside, add only if 4+ hours remain after everything else works

### Architecture Approach

The system has three runtime layers separated by strict REST boundaries, plus an offline training layer. The Python server is the semantic compute layer (loads PLY, encodes text queries, runs DBSCAN, computes spatial edges), but it does not own any agent logic. The JAC server is the reasoning layer (walker agents, graph persistence via MongoDB through jac-scale, `by llm()` calls). The browser is a pure rendering and input layer. There is a critical but non-obvious bidirectional dependency: the Python server calls JAC to build the graph (via `POST /walker/build_scene`), but the JAC QueryWalker calls Python to encode text (via `POST /cli/encode`). Both services must be up and mutually reachable for queries to function.

**Major components:**
1. **Training Layer (Colab A100, offline)** — produces point_cloud.ply, language_feature_dim3/*.npy, autoencoder.pth; runs once; blocks everything downstream
2. **Python FastAPI Server (port 8001)** — Gaussian Store (numpy), CLIP encoder, autoencoder decoder, DBSCAN clusterer, spatial relationship extractor, highlight mask API; stateless between requests
3. **JAC Cloud Server (port 8000, jac-scale)** — SpatialObject/SpatialRelation graph persisted in MongoDB; QueryWalker, ExplorationWalker, NavigationWalker, MemoryWalker; walkers are REST endpoints
4. **Backboard** — thread-scoped persistent memory per session; MemoryWalker writes spatial facts in natural language on query, reads on session start
5. **Browser (React + Spark.js)** — SplatMesh renders PLY from InsForge storage; Dyno shader applies highlight masks; chat input POSTs to JAC; graph overlay projects node centroids via Three.js
6. **InsForge** — JWT auth (browser-side only, JS SDK), PLY file storage (S3-compatible), scene metadata in Postgres; the Python server bypasses InsForge entirely

### Critical Pitfalls

1. **LangSplat training is a 5-stage sequential pipeline — each stage can fail silently.** Stage 3 requires a pre-trained 3DGS RGB checkpoint; if it is missing or wrong, LangSplat trains from random initialization. Symptom: cosine similarity scores are uniformly 0.3–0.5 for every query. Prevention: verify the 5-stage order, run vanilla 3DGS to 30K iterations first, smoke-test output at each stage before proceeding.

2. **Colab session timeouts kill unfinished training.** The 90-minute idle disconnect triggers when switching browser tabs — which is constant during a hackathon. Prevention: mount Google Drive in cell 1 before anything else, save all checkpoints to Drive with 5K-iteration saves, kick off training as the very first hackathon action.

3. **DBSCAN epsilon miscalibration produces one giant cluster or all noise.** Gaussian world-space coordinates are not standardized across scenes. `epsilon=0.1` may mean 10 cm in one scene and 10 meters in another. Prevention: normalize coordinates to [0,1] or compute epsilon dynamically from the k-distance graph knee. Never hardcode epsilon.

4. **CLIP thresholds are not universal — fixed cutoffs break on novel queries.** CLIP similarity is relative, not absolute. Use top-K retrieval (K=500–1000 Gaussians as DBSCAN input) as the primary mechanism, not a fixed similarity cutoff.

5. **JAC walker infinite traversal on cyclic spatial graphs.** Spatial graphs have cycles (Room contains objects that are next_to other objects that are all inside Room). JAC walkers without explicit visited-set tracking will loop forever, hanging all queries. Prevention: every walker must carry a `has visited: set[str]` and check it at entry; use `disengage` (not `return`) when the goal is met.

---

## Implications for Roadmap

Based on the feature dependency tiers, architecture layer boundaries, and critical pitfall timing, the following phase structure is recommended.

### Phase 0: Training + Fallback Scene Preparation (Pre-hackathon or Hour 0)
**Rationale:** The entire project is blocked on LangSplat output artifacts. This must be started before any code is written. No other development can be validated against real data until this is done.
**Delivers:** point_cloud.ply + language_feature_dim3/ + autoencoder.pth for a real indoor scene; plus a pre-trained fallback scene downloaded to Drive.
**Addresses:** FEATURES.md Tier 0 ("blocking dependency — must complete first")
**Avoids:** Colab session timeout (Drive mount), silent training failure (stage verification), COLMAP failure on featureless rooms (pre-tested scene capture)
**Research flag:** No additional research needed — LangSplat pipeline is documented. Execution risk, not knowledge risk.

### Phase 1: Python Server Foundation (Semantic Query Core)
**Rationale:** The CLIP query pipeline is the single piece all other features depend on. DBSCAN, JAC graph construction, and the browser highlight mask all require verified CLIP query output. Build and validate this in isolation first.
**Delivers:** `POST /clip/encode` that returns top-K Gaussian indices + similarity scores for any text query. `GET /health` confirms PLY and autoencoder are loaded.
**Uses:** FastAPI 0.135.3, open-clip-torch 3.3.0, plyfile, numpy, scipy
**Implements:** Gaussian Store + CLIP Text Encoder + Autoencoder Decoder + Cosine Similarity Ranker
**Avoids:** CLIP threshold pitfall (implement top-K from the start, never a fixed cutoff)
**Research flag:** Standard FastAPI patterns — no research phase needed.

### Phase 2: Clustering + Graph Construction
**Rationale:** Object-level abstraction is the conceptual bridge between raw Gaussian data and the JAC knowledge graph. DBSCAN clustering must be calibrated against the actual demo scene before JAC nodes are created. Cannot be done against synthetic data.
**Delivers:** JAC graph populated with SpatialObject nodes and SpatialRelation edges; `POST /walker/build_scene` callable from Python; queryable graph via jac-scale MongoDB persistence.
**Uses:** scikit-learn DBSCAN, scipy KDTree/ConvexHull, jaseci 2.3.12, httpx (Python-to-JAC calls)
**Implements:** DBSCAN Clusterer, Spatial Relationship Extractor, Graph Builder, JAC SceneWalker node/edge schema
**Avoids:** DBSCAN epsilon miscalibration (normalize coordinates or use k-distance knee); rebuilding graph per query (build once on scene load)
**Research flag:** JAC walker REST endpoint setup may need verification against jac-scale docs — low risk but confirm `jac start` command and MongoDB config before implementation.

### Phase 3: JAC Query + Agent Walkers
**Rationale:** With the graph populated, walker logic can be developed and tested. QueryWalker is prerequisite to the browser demo. ExplorationWalker is prerequisite for the Agentic AI Track prize. MemoryWalker is prerequisite for the Backboard prize.
**Delivers:** `POST /walker/query` returns highlight mask + node data + natural language description; `POST /walker/explore` autonomously catalogs all scene objects with `by llm()` descriptions; `POST /walker/memory` reads/writes Backboard session context.
**Uses:** jaseci 2.3.12, byllm, Backboard thread API
**Implements:** QueryWalker, ExplorationWalker, MemoryWalker (NavigationWalker deferred to Phase 5)
**Avoids:** Walker infinite traversal (visited-set pattern in every walker), synchronous Backboard in query hot path (fire-and-forget memory writes)
**Research flag:** `by llm()` function signature → prompt generation behavior should be validated with a simple smoke test before relying on it for ExplorationWalker descriptions.

### Phase 4: Browser Frontend
**Rationale:** Once JAC walkers are returning correct JSON responses, the frontend can be wired end-to-end. Spark.js Dyno highlight mask integration is a risk item — the Dyno shader API is moderately complex and may require iteration.
**Delivers:** Working browser showing Gaussian splat scene + chat input + semantic highlighting response + graph overlay. Full end-to-end demo path functional.
**Uses:** Spark.js 0.1.10, Three.js 0.183.2, React 19, @react-three/fiber, @react-three/drei, InsForge JS SDK
**Implements:** SplatMesh loader, Highlight Controller (Dyno shader), Chat Interface wired to QueryWalker, Graph Overlay (node spheres + edge lines)
**Avoids:** CORS blocking (add FastAPI CORS middleware on day one of backend work), browser PLY memory exhaustion (convert to .spz or .ksplat before loading in browser), Three.js version mismatch (pin to 0.183.2 immediately)
**Research flag:** Spark.js Dyno API for per-Gaussian highlighting — consult `sparkjs.dev/docs/dyno-overview/` during implementation. This is the highest-risk frontend item per ARCHITECTURE.md.

### Phase 5: Polish, Integration + Differentiators
**Rationale:** Connect all layers using the live trained PLY. Add the probability cloud differentiator, which is the single most visually impressive feature and distinguishes SpatialMind from every other spatial AI demo. Add remaining agentic behaviors if time allows.
**Delivers:** Live scene (not demo scene) running end-to-end; probability cloud overlay showing continuous semantic field; ExplorationWalker auto-triggered on scene load; Backboard persistent memory recall working across sessions.
**Implements:** Probability cloud Dyno shader (soft cosine score → opacity gradient), scene version management (PLY content hash), Backboard cross-session recall in MemoryWalker
**Avoids:** Probability cloud as a placeholder (verify it is wired to actual CLIP scores, not uniform values); coordinate system mismatch between LangSplat world space and Three.js scene space (requires explicit transform)
**Research flag:** No additional research needed — all patterns are documented. Execution and integration risk only.

### Phase Ordering Rationale

- Phase 0 is non-negotiable as the first action — every other phase produces untestable output without real PLY data.
- Phases 1-2 build bottom-up along the data flow: raw Gaussians → object clusters → graph. This order ensures each layer is independently verifiable before adding the next.
- Phase 3 (JAC walkers) comes after the Python server and graph are verified so that walker debugging isolates JAC logic from CLIP/DBSCAN bugs.
- Phase 4 (frontend) comes last among the core phases because it depends on JAC returning well-formed JSON — implementing the frontend before the backend API is stable wastes time on integration churn.
- Phase 5 is explicitly reserved for differentiators to prevent scope creep into early phases.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 0 (Training):** COLMAP + Scaniverse export compatibility needs validation before the hackathon. Verify that Scaniverse exports include cameras.bin/images.bin or that raw images + COLMAP can reconstruct correctly from Scaniverse captures.
- **Phase 3 (JAC walkers):** `by llm()` behavior with typed function signatures needs a smoke test early. The mechanism is documented but behavior at edge cases (complex return types, nullable fields) is not fully characterized.
- **Phase 4 (Spark.js Dyno):** Dyno shader API for per-Gaussian index-based highlighting is the highest-uncertainty implementation item. Consult Spark.js docs on the Dyno system before committing to this approach.

Phases with standard patterns (skip research-phase):
- **Phase 1 (FastAPI server):** Well-documented patterns. FastAPI + numpy + scikit-learn is a standard ML serving stack.
- **Phase 2 (DBSCAN + graph construction):** DBSCAN is a known algorithm. JAC node/edge schema is clearly specified. The epsilon calibration approach is documented.
- **Phase 5 (polish):** All integration patterns are known by this point. No new unknowns.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | All library versions confirmed via PyPI/npm as of April 2026. CUDA 11.8 / PyTorch 2.0 pairing for LangSplat is MEDIUM — confirmed from README/environment.yml but not independently executed. All other versions HIGH. |
| Features | HIGH | Feature tiers are directly derived from LangSplat CVPR 2024 paper capabilities + JAC walker documentation + Backboard API. Anti-features are clearly justified. Priority order is well-reasoned for prize targets. |
| Architecture | MEDIUM | JAC walker REST patterns are HIGH confidence via official docs. LangSplat output file structure is MEDIUM (from GitHub README, not independently run). Backboard API pattern is LOW-MEDIUM (from docs only, not tested). InsForge Python server interaction is LOW (JS SDK only — Python server must use REST directly). |
| Pitfalls | HIGH | Pitfalls are grounded in concrete failure modes: Colab limits documented by Google, DBSCAN epsilon sensitivity documented with k-distance knee solution, CLIP threshold instability documented in literature, JAC cycle traversal documented in official walker docs. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Scaniverse → COLMAP compatibility:** Research notes this as inferred, not confirmed. Before the hackathon, do a test capture with Scaniverse and verify the export produces a valid COLMAP sparse/ directory (cameras.bin, images.bin, points3D.bin) or that running COLMAP on exported images produces a complete reconstruction.
- **Spark.js Dyno API for index-based highlighting:** ARCHITECTURE.md explicitly flags this as a risk item requiring direct doc consultation. The pattern described (build a SplatMesh, compose a Dyno that checks Gaussian index against a uniform buffer) is directionally correct but exact API is unconfirmed.
- **Backboard memory read on session start:** ARCHITECTURE.md documents the write path clearly; the read path for cross-session recall is low-confidence and needs implementation validation.
- **InsForge — Python server file access:** InsForge has a JS SDK only. The Python FastAPI server must read PLY files either from the local filesystem or via direct S3 URL (if InsForge exposes standard S3-compatible endpoints). This integration path should be confirmed before building the Scene Loader in Phase 4.
- **JAC `by llm()` LLM provider configuration:** Both OpenAI and Anthropic are supported. API key availability during the hackathon should be confirmed and the byllm config should be set up in Phase 3, not discovered at the last minute.

---

## Sources

### Primary (HIGH confidence)
- Spark.js GitHub (sparkjsdev/spark) — version 0.1.10, Three.js 0.178+ compatibility, SplatMesh API, Dyno system overview
- Three.js npm (npmjs.com/package/three) — version 0.183.2 confirmed current
- jaseci PyPI (pypi.org/project/jaseci/) — version 2.3.12, April 3 2026, Python >=3.11, walker syntax
- Jaseci GitHub (jaseci-labs/jaseci) — jaclang 0.13.5, walker/node/edge syntax, `by llm()` mechanism, `visit` and `disengage` semantics
- open-clip-torch PyPI — version 3.3.0, Feb 2026, Python >=3.9
- FastAPI PyPI — version 0.135.3, April 1 2026, Python >=3.10
- scikit-learn docs — version 1.8.0 DBSCAN confirmed current stable
- scipy docs — version 1.17.0 KDTree/ConvexHull
- COLMAP install docs (colmap.github.io) — version 3.9 binary, ALIKED extractor option, SIFT failure modes
- JAC walker docs (docs.jaseci.org) — REST endpoint creation, `by llm()` agentic AI patterns, jac-scale MongoDB persistence
- DBSCAN Parameter Selection Guide (sefidian.com) — k-distance knee method for epsilon
- Google Colab session limits — 12-hour hard cap, 90-minute idle disconnect
- Spark.js docs (sparkjs.dev) — WebGL2 requirement, Three.js r126 async readback requirement
- Making Gaussian Splats Smaller (Aras Pranckevičius) — .spz/.ksplat compression formats

### Secondary (MEDIUM confidence)
- LangSplat GitHub README (minghanqin/LangSplat) — output directory structure, training pipeline stages, CUDA/PyTorch version requirements
- LangSplat CVPR 2024 paper — autoencoder compression (512→3 dim), cosine similarity as query mechanism, three-scale SAM segmentation
- Backboard.io quickstart (docs.backboard.io) — thread_id, memory API, POST /api/threads/{thread_id}/messages
- InsForge introduction (docs.insforge.dev) — Postgres, auth, storage, edge functions; JS SDK confirmed, Python SDK not found
- WebSplatter (arxiv.org/html/2602.03207) — browser WebGL buffer limits, memory constraints for Gaussian rendering
- Semantic Consistent Language Gaussian Splatting (arxiv.org/html/2503.21767) — CLIP threshold instability in 3D querying
- Scaniverse official (scaniverse.com) — PLY/SPZ export confirmed; COLMAP sparse/ output compatibility inferred

### Tertiary (LOW confidence)
- Backboard cross-session read path — documented in concept but not independently tested
- InsForge S3-compatible storage URL format for Python server access — inferred from Supabase-like architecture, not confirmed
- JAC `by llm()` behavior with complex return types — documented in simple examples, edge cases not characterized

---
*Research completed: 2026-04-04*
*Ready for roadmap: yes*
