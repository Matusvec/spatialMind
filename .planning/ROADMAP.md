# Roadmap: SpatialMind

## Overview

SpatialMind builds bottom-up along its data pipeline: training produces the artifacts, the Python server loads and queries them, the JAC graph reasons over them, the walkers traverse and narrate them, and the browser renders them. Each phase delivers one independently verifiable layer. The final phase wires all layers into the three-beat hackathon demo. Phases follow the strict feature dependency tiers identified in research — no layer can be validated without the one below it working.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Training Pipeline** - Kick off LangSplat on Colab A100 and produce PLY artifacts — blocking dependency for everything else
- [ ] **Phase 2: Semantic Query Server** - Python FastAPI server with CLIP encoding, autoencoder decoding, cosine similarity, and probability clouds
- [ ] **Phase 3: Clustering + JAC Graph** - DBSCAN object clustering, spatial relationship extraction, and JAC knowledge graph construction
- [ ] **Phase 4: Agent Walkers + Backboard** - QueryWalker and ExplorationWalker with natural language reasoning; Backboard persistent memory integration
- [ ] **Phase 5: Frontend + InsForge** - Spark.js splat viewer, semantic highlighting, graph overlay, chat interface, and InsForge storage/auth
- [ ] **Phase 6: Demo Integration** - End-to-end three-beat demo sequence running on live trained scene

## Phase Details

### Phase 1: Training Pipeline
**Goal**: LangSplat artifacts are on disk and verified — the three files (PLY, .npy embeddings, autoencoder.pth) are co-located and smoke-tested so every downstream phase has real data
**Depends on**: Nothing (first phase)
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04
**Success Criteria** (what must be TRUE):
  1. LangSplat 5-stage pipeline completes on Colab A100 producing point_cloud.ply, language_feature_dim3/*.npy, and autoencoder.pth as co-located artifacts
  2. Cosine similarity smoke test on the output returns non-uniform scores (scores vary meaningfully across a "chair" vs "table" query pair — not uniformly 0.3-0.5)
  3. Pre-trained fallback scene is downloaded to Google Drive and loadable as a drop-in replacement if live training fails
  4. JacHacks venue captured with Scaniverse and training kicked off within the first hour of the hackathon
**Plans:** 2 plans
Plans:
- [x] 01-01-PLAN.md — Create LangSplat training Colab notebook and verification scripts
- [ ] 01-02-PLAN.md — Hackathon execution runbook and notebook verification checkpoint

### Phase 2: Semantic Query Server
**Goal**: A running Python server that accepts a text query and returns the top-K matching Gaussian indices with similarity scores, plus soft probability assignments for all Gaussians
**Depends on**: Phase 1
**Requirements**: QUERY-01, QUERY-02, QUERY-03, QUERY-04, QUERY-05, QUERY-06
**Success Criteria** (what must be TRUE):
  1. `GET /health` confirms PLY is loaded, autoencoder is initialized, and Gaussian count is non-zero
  2. `POST /clip/encode` with "chair" returns top-K Gaussian indices with cosine similarity scores that visually correspond to chair geometry when rendered
  3. `POST /clip/highlight` returns a highlight mask (per-Gaussian opacity values) usable by the browser renderer
  4. Probability cloud endpoint returns per-Gaussian soft assignment scores across all discovered clusters, not a binary match
  5. Multi-resolution query ("furniture" vs "seating" vs "Herman Miller") returns different Gaussian sets at each granularity level
**Plans**: TBD
**UI hint**: yes

### Phase 3: Clustering + JAC Graph
**Goal**: DBSCAN clusters the top-K Gaussians into named object nodes and JAC graph is populated with SpatialObject nodes connected by SpatialRelation edges derived from 3D geometry
**Depends on**: Phase 2
**Requirements**: GRAPH-01, GRAPH-02, GRAPH-03, GRAPH-04
**Success Criteria** (what must be TRUE):
  1. DBSCAN produces distinct object clusters (not one giant cluster, not all noise) with centroid positions and spatial extents that match visual object locations in the scene
  2. JAC graph is queryable via `jac start` with SpatialObject nodes and SpatialRelation edges present after `POST /walker/build_scene`
  3. Spatial relationship edges are correctly derived from 3D coordinates: on_top_of fires for objects with clear Y-axis offset, next_to fires for XZ-proximate objects
  4. Edge confidence scores distinguish clear relationships (book on desk: high confidence) from ambiguous ones (couch near table: medium confidence)
  5. CLIP embedding proximity produces an emergent object hierarchy — querying at "furniture" level returns all furniture nodes, not just exact-match objects
**Plans**: TBD

### Phase 4: Agent Walkers + Backboard
**Goal**: QueryWalker traverses the spatial graph and returns natural language answers; ExplorationWalker autonomously catalogs the scene and writes it to Backboard; multi-turn conversational context persists across follow-up queries
**Depends on**: Phase 3
**Requirements**: GRAPH-05, GRAPH-06, BB-01, BB-02, BB-03, BB-04, BB-05, BB-06
**Success Criteria** (what must be TRUE):
  1. `POST /walker/query` with "what's on the tables?" returns a natural language answer that correctly names objects and their spatial relationships, generated via `by llm()` over the matched sub-graph
  2. `POST /walker/explore` autonomously traverses all unvisited nodes and writes a complete object catalog to a Backboard thread with `memory="Auto"`
  3. Follow-up query "is any of that near the window?" correctly uses context from the previous query turn without re-stating all prior information
  4. Revisiting a previously explored scene reads spatial memory from Backboard instead of re-running DBSCAN clustering
  5. Walker traversal terminates correctly on cyclic graphs — no infinite loops, uses visited-set pattern with `disengage` when goal is met
**Plans**: TBD

### Phase 5: Frontend + InsForge
**Goal**: Browser renders the photorealistic splat scene at interactive framerates, semantic highlighting responds to query results in real-time, graph overlay floats in 3D space, and chat interface is wired to QueryWalker
**Depends on**: Phase 3 (can start against Phase 2 mock data in parallel)
**Requirements**: UI-01, UI-02, UI-03, UI-04, UI-05, INS-01, INS-02, INS-03, INS-04
**Success Criteria** (what must be TRUE):
  1. Spark.js renders the .PLY scene at interactive framerates in Chrome without exceeding browser memory limits
  2. Typing a query in the chat interface sends it to QueryWalker and matching Gaussians visibly change color/opacity in the 3D scene within a few seconds
  3. 3D graph overlay shows object nodes as spheres and spatial edges as lines floating at correct 3D positions in the scene
  4. Probability cloud heatmap visualization renders as a continuous opacity gradient across Gaussians (not binary on/off)
  5. Scene list loads from InsForge DB and splat files stream from InsForge storage; user can log in and see their scenes
**Plans**: TBD
**UI hint**: yes

### Phase 6: Demo Integration
**Goal**: The three-beat demo sequence runs live end-to-end on the trained JacHacks venue scene — basic highlighting, spatial reasoning, and an unrehearsed query all work without rehearsed inputs
**Depends on**: Phase 4, Phase 5
**Requirements**: DEMO-01, DEMO-02, DEMO-03
**Success Criteria** (what must be TRUE):
  1. Beat 1: typing "chairs" causes chair Gaussians to light up in the scene with no errors in under 5 seconds
  2. Beat 2: typing "what's on the tables?" returns a correct natural language answer listing real objects visible in the scene
  3. Beat 3: an unrehearsed, novel query (chosen during demo) returns a spatially correct response — demonstrating real scene understanding not memorized outputs
  4. Full pipeline runs on live-trained JacHacks venue PLY, with automatic fallback to pre-trained scene if the live scene is unavailable
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Training Pipeline | 0/2 | Planning complete | - |
| 2. Semantic Query Server | 0/TBD | Not started | - |
| 3. Clustering + JAC Graph | 0/TBD | Not started | - |
| 4. Agent Walkers + Backboard | 0/TBD | Not started | - |
| 5. Frontend + InsForge | 0/TBD | Not started | - |
| 6. Demo Integration | 0/TBD | Not started | - |
