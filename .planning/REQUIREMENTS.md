# Requirements: SpatialMind

**Defined:** 2026-04-04
**Core Value:** An AI that understands 3D space as a structured graph, queryable with natural language, powered by semantically-embedded Gaussians — giving physical spaces a mind.

## v1 Requirements

Requirements for hackathon demo. Each maps to roadmap phases.

### Training Pipeline

- [ ] **TRAIN-01**: LangSplat 5-stage pipeline runs on Colab A100 (COLMAP → SAM → autoencoder → 3DGS RGB → LangSplat training)
- [ ] **TRAIN-02**: Training output produces PLY + .npy latent features + autoencoder.pth as three co-located artifacts
- [ ] **TRAIN-03**: Pre-trained fallback scene stored on Google Drive as insurance against training failures
- [ ] **TRAIN-04**: Live scan of JacHacks venue at UMich captured and trained during the hackathon

### Semantic Query Engine

- [ ] **QUERY-01**: Text input is converted to CLIP vector via open-clip-torch on Python server
- [ ] **QUERY-02**: Cosine similarity computed between CLIP text vector and all Gaussian embeddings (autoencoder decodes 3-dim latents → 512-dim CLIP space)
- [ ] **QUERY-03**: Top-K retrieval returns matching Gaussians (not threshold-based — CLIP scores aren't calibrated)
- [ ] **QUERY-04**: Matching Gaussians are highlighted in the browser 3D scene via highlight mask sent from server
- [ ] **QUERY-05**: Probability cloud layer — each Gaussian carries soft assignment scores across all discovered object nodes, queryable as a continuous semantic field
- [ ] **QUERY-06**: Multi-resolution queries work at any granularity ("furniture" → couch+table, "seating" → just couch, "Herman Miller Aeron" → specific chair)

### Spatial Graph (JAC)

- [ ] **GRAPH-01**: DBSCAN clusters semantically similar Gaussians into object nodes with centroid positions and spatial extents
- [ ] **GRAPH-02**: Emergent object hierarchy from CLIP embedding proximity — no hardcoded taxonomy, organic tree at any specificity level
- [ ] **GRAPH-03**: Spatial relationship edges derived from 3D coordinates: on_top_of (Y-axis offset), next_to (XZ proximity), facing (covariance orientation), inside (containment)
- [ ] **GRAPH-04**: Edge confidence scores based on how clearly the spatial relationship holds
- [ ] **GRAPH-05**: QueryWalker traverses the spatial graph and synthesizes natural language answers via Backboard LLM routing + `by llm()`
- [ ] **GRAPH-06**: ExplorationWalker autonomously catalogs all objects in the scene and writes object catalog to Backboard memory

### Frontend (Partner)

- [ ] **UI-01**: Spark.js renders photorealistic Gaussian splat scene in browser at interactive framerates
- [ ] **UI-02**: Chat interface accepts natural language queries and displays agent responses
- [ ] **UI-03**: Semantic highlight overlays — matching Gaussians change color/opacity in the 3D scene based on query results
- [ ] **UI-04**: 3D graph overlay — object nodes and spatial edges rendered floating in the scene at their 3D positions
- [ ] **UI-05**: Probability cloud heatmap visualization showing continuous semantic field intensity

### Backboard Integration

- [ ] **BB-01**: Backboard assistant created with system prompt describing spatial reasoning capabilities
- [ ] **BB-02**: Each scanned scene gets its own Backboard thread for persistent spatial memory
- [ ] **BB-03**: After clustering, ExplorationWalker writes full object catalog to Backboard thread via `add_message()` with `memory="Auto"`
- [ ] **BB-04**: QueryWalker routes spatial questions through Backboard for LLM reasoning, enriched with spatial graph context
- [ ] **BB-05**: Multi-turn conversational context — follow-up queries retain context from previous questions in the thread
- [ ] **BB-06**: Cross-session recall — revisiting a scene reads spatial memory from Backboard instead of re-clustering

### InsForge Integration

- [ ] **INS-01**: S3-compatible storage hosts PLY files, .npy embeddings, and autoencoder weights
- [ ] **INS-02**: PostgreSQL stores scene metadata (scene ID, training status, object count, bounding box, thumbnails)
- [ ] **INS-03**: User authentication via InsForge auth for multi-user scene libraries
- [ ] **INS-04**: Frontend fetches scene list from InsForge DB and loads splat files from InsForge storage

### Demo

- [ ] **DEMO-01**: Three-beat demo sequence works live — basic object detection ("chairs" light up) → spatial reasoning ("what's on the tables") → unrehearsed query proving real understanding
- [ ] **DEMO-02**: Demo uses live-trained JacHacks venue scene (with fallback to pre-trained scene if training fails)
- [ ] **DEMO-03**: Full pipeline runs end-to-end: query → CLIP → similarity → highlight + graph traversal → natural language answer

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Navigation

- **NAV-01**: NavigationWalker plans camera paths through the scene via graph traversal
- **NAV-02**: Animated camera flight along planned path with narration
- **NAV-03**: Obstacle avoidance using spatial graph clearance data

### Memory & Change Detection

- **MEM-01**: MemoryWalker generates high-level room summaries
- **MEM-02**: Change detection between sessions ("the plant moved from the table to the windowsill")
- **MEM-03**: Cross-scene knowledge queries ("Did we see a printer in any room?")

### Advanced Features

- **ADV-01**: InsForge edge function triggered when Colab training completes to update scene status
- **ADV-02**: Base44 demo landing page with pre-loaded example scenes
- **ADV-03**: WebXR support (Quest 3, Vision Pro)
- **ADV-04**: Multi-room scene linking with inter-room navigation

## Out of Scope

| Feature | Reason |
|---------|--------|
| Custom SAM/CLIP model training | Use existing pretrained models — no time for custom training |
| Dynamic/real-time scene updates | Static scenes only — LangSplat requires offline training |
| Mobile app | Web-first for hackathon demo |
| Video input pipeline | Photos-to-COLMAP only |
| Performance optimization (LOD, streaming) | Premature for 24-hour hackathon |
| Multi-user real-time collaboration | Single-user for v1 |
| Base44 integration | Low priority, only if significant time remains |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRAIN-01 | Phase 1 | Pending |
| TRAIN-02 | Phase 1 | Pending |
| TRAIN-03 | Phase 1 | Pending |
| TRAIN-04 | Phase 1 | Pending |
| QUERY-01 | Phase 2 | Pending |
| QUERY-02 | Phase 2 | Pending |
| QUERY-03 | Phase 2 | Pending |
| QUERY-04 | Phase 2 | Pending |
| QUERY-05 | Phase 2 | Pending |
| QUERY-06 | Phase 2 | Pending |
| GRAPH-01 | Phase 3 | Pending |
| GRAPH-02 | Phase 3 | Pending |
| GRAPH-03 | Phase 3 | Pending |
| GRAPH-04 | Phase 3 | Pending |
| GRAPH-05 | Phase 4 | Pending |
| GRAPH-06 | Phase 4 | Pending |
| UI-01 | Phase 5 | Pending |
| UI-02 | Phase 5 | Pending |
| UI-03 | Phase 5 | Pending |
| UI-04 | Phase 5 | Pending |
| UI-05 | Phase 5 | Pending |
| BB-01 | Phase 4 | Pending |
| BB-02 | Phase 4 | Pending |
| BB-03 | Phase 4 | Pending |
| BB-04 | Phase 4 | Pending |
| BB-05 | Phase 4 | Pending |
| BB-06 | Phase 4 | Pending |
| INS-01 | Phase 5 | Pending |
| INS-02 | Phase 5 | Pending |
| INS-03 | Phase 5 | Pending |
| INS-04 | Phase 5 | Pending |
| DEMO-01 | Phase 6 | Pending |
| DEMO-02 | Phase 6 | Pending |
| DEMO-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 34 total
- Mapped to phases: 34
- Unmapped: 0

---
*Requirements defined: 2026-04-04*
*Last updated: 2026-04-04 after roadmap creation — all 34 requirements mapped*
