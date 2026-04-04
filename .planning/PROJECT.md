# SpatialMind

## What This Is

An autonomous AI agent system that sees, navigates, and reasons inside photorealistic 3D worlds. Built on JAC with LangSplat-powered Gaussian splatting, SpatialMind turns photorealistic 3D scenes into queryable, structured spatial knowledge graphs — where every Gaussian carries a semantic embedding and the entire scene is a continuous, multi-resolution semantic field. Built for JacHacks hackathon (April 4-5, 2026).

## Core Value

**An AI that understands 3D space as a structured graph, queryable with natural language, powered by semantically-embedded Gaussians — not hacky render-to-2D-and-back pipelines.**

If everything else fails, a user must be able to load a LangSplat scene, type a natural language query, and see the relevant objects light up in the 3D view with spatial relationships explained.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] LangSplat training pipeline produces Gaussians with CLIP semantic embeddings
- [ ] Natural language queries resolve to matching Gaussians via CLIP cosine similarity
- [ ] DBSCAN clustering groups semantically similar Gaussians into object nodes
- [ ] JAC spatial knowledge graph with nodes (objects, rooms) and edges (spatial relationships)
- [ ] Spatial relationships derived from 3D coordinates: on_top_of, next_to, facing, inside
- [ ] Probability cloud layer with soft assignments — continuous semantic field at any resolution
- [ ] Walker agents traverse and reason over the spatial graph (Query, Navigation, Exploration, Memory)
- [ ] Spark.js renders photorealistic splat scene in browser
- [ ] Server sends highlight masks to frontend for semantic overlays on 3D scene
- [ ] Graph visualization overlay showing nodes and edges in 3D space
- [ ] Chat interface for spatial queries and navigation commands
- [ ] Backboard integration for persistent spatial memory across sessions
- [ ] InsForge backend infrastructure (auth, storage, edge functions)
- [ ] Autonomous exploration discovers and catalogs objects without manual prompting

### Out of Scope

- Base44 integration — low priority, only if time permits
- Mobile/WebXR support — web-first for hackathon
- Multi-room scene linking — single scene per demo
- Real-time collaborative editing — single-user for v1
- Custom SAM/CLIP model training — use existing pretrained models
- Video input pipeline — photos-to-COLMAP only

## Context

### Technical Architecture

**The core insight:** Instead of the vision doc's Gemini-based perception pipeline (render viewpoints → detect objects in 2D → back-project to 3D), SpatialMind uses LangSplat to embed semantic CLIP vectors directly into every Gaussian during training. This eliminates viewpoint sampling, 2D detection, back-projection, and cross-view merging entirely. Semantic understanding is native to the 3D representation.

**Pipeline:**
1. **Training (Colab A100):** Photos → COLMAP poses → LangSplat training → Gaussians with position + color + opacity + CLIP embedding
2. **Querying (Python server):** Text → CLIP text vector → cosine similarity against all Gaussian embeddings → matching Gaussians
3. **Clustering (Python server):** DBSCAN on semantically similar Gaussians → object nodes with centroid positions and spatial extents
4. **Graph (JAC):** Nodes from clusters, edges from 3D spatial relationships (Y-offset → on_top_of, XZ proximity → next_to, covariance orientation → facing, containment → inside)
5. **Probability Clouds:** Soft assignments per Gaussian across all nodes — continuous semantic field queryable at any granularity ("furniture" → couch + table, "seating" → just couch)
6. **Rendering (browser):** Spark.js renders visual splat, server sends highlight masks for semantic overlays

**Team split:**
- Backend (owner): Gaussian splatting pipeline, LangSplat training, Python server with CLIP, JAC graph + walkers, Backboard, spatial reasoning
- Frontend (partner): React UI, Spark.js viewer, graph visualization, chat interface, highlighting

### Integration Architecture

**Backboard (backboard.io):** Unified AI infrastructure API — persistent memory, stateful conversations, RAG, 17,000+ LLM routing.
- **Spatial Memory Layer:** ExplorationWalker writes full object catalog to a Backboard thread after clustering. On revisit, reads from Backboard instead of re-clustering. Scene "remembers" across sessions.
- **Conversational Spatial Queries:** QueryWalker uses Backboard threads for multi-turn context. "What's on the table?" → follow-up "is any of that near the window?" retains context.
- **Cross-Scene Knowledge:** Multiple scanned rooms share Backboard memory. "Did we see a printer in any room?" searches across all scene memories.
- **LLM Routing:** Use Backboard's built-in LLM routing instead of managing separate API keys.
- Python SDK: `from backboard import BackboardClient`

**InsForge (insforge.dev):** YC-backed backend platform for AI agents — PostgreSQL, auth, S3 storage, edge functions, MCP-controllable.
- **Splat File Storage:** S3-compatible storage for PLY, .npy embeddings, autoencoder weights.
- **Scene Metadata DB:** PostgreSQL stores scene ID, training status, object count, bounding box, thumbnails.
- **User Auth:** InsForge auth for multi-user scene libraries.
- **Edge Functions:** Webhook when Colab training completes → update scene status in DB.

**Priority:** Core demo must work without integrations (hardcoded scene, in-memory graph). Then layer in InsForge for storage/auth and Backboard for memory/LLM.

### Domain Context

- Spatial AI is a $45B+ investment category (2024-2026)
- Y Combinator Spring 2026 RFS explicitly calls out "Large Spatial Models"
- LLMs achieve ~48% on spatial reasoning benchmarks vs 98% human — externalizing to a graph is the key innovation
- LangSplat (2024) demonstrated language-embedded Gaussians for open-vocabulary 3D queries
- JAC's walker paradigm maps naturally to spatial agent traversal

### Hackathon Context

- JacHacks — 24-hour hackathon, April 4-5, 2026
- Sponsors: Backboard (persistent memory), InsForge (infra), Lovable (frontend), Base44 (stretch)
- Prize targets: Agentic AI Track ($750), Best JAC Builder ($400), Best Use of Backboard, SH1P Interview
- JAC is new to the builder but strong Python/full-stack background

## Constraints

- **Timeline**: 24 hours (hackathon)
- **Compute**: LangSplat training requires Colab A100 — must kick off early, blocking dependency
- **JAC familiarity**: Learning on the fly — favor simple patterns, lean on `by llm()` for complex logic
- **Team size**: 2 people — backend/frontend split, minimize integration surface
- **Scenes**: Need to capture with Scaniverse before/during hackathon, then train with LangSplat
- **Browser rendering**: Spark.js handles visual splat only — all semantic computation is server-side

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| LangSplat over Gemini perception pipeline | Eliminates viewpoint sampling, 2D detection, back-projection, and cross-view merging. Semantics are native to 3D. | — Pending |
| Server-side CLIP queries (not browser) | CLIP is too heavy for browser; server holds full Gaussian data with embeddings | — Pending |
| Probability clouds as core v1 (not stretch) | Soft assignments and continuous semantic fields are the technical differentiator — what makes this publishable-quality | — Pending |
| DBSCAN for object clustering | Well-understood, handles arbitrary cluster shapes, no need to predefine number of objects | — Pending |
| Spark.js for rendering | State-of-the-art browser splat renderer by World Labs, trivial setup, Three.js integration | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-04 after initialization*
