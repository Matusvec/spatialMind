# Feature Landscape

**Domain:** Spatial AI — 3D scene understanding with language-embedded Gaussian splatting and autonomous graph-walker agents
**Researched:** 2026-04-04
**Context:** JacHacks 24-hour hackathon. Prize targets: Agentic AI Track, Best JAC Builder, Best Use of Backboard.

---

## Framing: What Are Judges Evaluating?

Research across spatial AI hackathon judging criteria reveals a clear pattern: judges are not impressed by chat interfaces wrapping existing models. They want to see:

1. **Working demo that runs** — over polish and over-engineering
2. **Novel technical architecture** — not "system prompt in a basic UI"
3. **Rich visual output** — spatial understanding SHOWN, not described
4. **Genuine agentic behavior** — agents that plan, reason, and adapt — not hardcoded scripts

The SpatialMind core value statement ("AI that understands 3D space as a structured graph, queryable with natural language, powered by semantically-embedded Gaussians — not hacky render-to-2D-and-back pipelines") is directly in line with what wins.

---

## Table Stakes

Features where absence causes the demo to fail outright. Missing any of these and the project has no story to tell.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Natural language object query ("show me the chair") | LangSplat's core demonstrated capability; the entire value prop requires this | Medium | CLIP text → cosine similarity → matching Gaussians. Well-understood pipeline. This IS the minimum viable feature. |
| Semantic highlighting in 3D viewer | Without visual feedback the query has no output — judges see nothing | Medium | Spark.js supports per-splat color/opacity override. Server sends highlight mask. Frontend applies it. |
| 3D Gaussian splat scene load and render | The photorealistic foundation — if the scene doesn't render, nothing else matters | Low | Spark.js handles this. Blocking dependency: LangSplat training on Colab A100 must finish early. |
| Spatial relationship detection on query result | "Where is the chair?" requires more than "here are some Gaussians." Judges expect "the chair is next to the table, on the floor." | Medium | Derivable from 3D coordinates after DBSCAN clustering produces object nodes with centroids. |
| Chat interface for spatial queries | Expected UX for any AI product in 2026 — text in, result out | Low | Simple React chat component. Partner-side work. Should not be overengineered. |
| Object clustering (DBSCAN) | Without clustering, you have Gaussian clouds not objects — can't build a graph without nodes | Medium | DBSCAN on semantically similar Gaussians → object centroids + extents. Well-understood algorithm. |
| JAC spatial knowledge graph | Project is built on JAC — graph with nodes (objects) and edges (relationships) is the structural claim | Medium | This is what differentiates from "just doing CLIP queries." Nodes from DBSCAN, edges from geometric tests. |

---

## Differentiators

Features that make judges say "wow." Not baseline expected, but achievable in 24 hours and technically impressive.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Probability cloud layer — continuous semantic field | This is the publishable-quality differentiator. Ask "show me furniture" and get a soft heatmap over ALL semantically related Gaussians, not just a hard selection. Demonstrates that semantics are a continuous field in 3D, not discrete labels. No competitor demo shows this. | High | Soft cosine similarity scores retained per-Gaussian (not thresholded). Rendered as opacity/color gradient overlay. Core differentiator — worth the complexity. |
| Graph visualization overlay in 3D | Seeing the knowledge graph nodes and edges floating in the 3D space is visually arresting. Judges understand the architecture instantly. | Medium | Three.js geometry atop Spark.js scene. Spheres at centroids, lines for edges, edge labels (on_top_of, next_to). Frontend partner work. |
| Multi-resolution semantic query ("furniture" vs "couch") | LangSplat's three-level hierarchy (whole/part/subpart) combined with probability clouds enables "furniture" returning couch+table+chair at high opacity, "seating" returning only chair+couch. Shows the system understands semantic granularity. | Medium | Natural extension of probability cloud layer. Use CLIP similarity at multiple abstraction levels. LangSplat already has this in its architecture. |
| Autonomous exploration walker | A JAC walker that traverses the spatial graph without prompting — discovering all objects, building a catalog — is a genuine agentic behavior demo. Shows JAC's walker paradigm at work, not just a Python function. | High | Exploration Walker visits every node, records findings in Backboard. The "agent thinks for itself" narrative. Critical for Agentic AI Track prize. |
| Persistent spatial memory via Backboard | Asking "what did we find last time?" and getting accurate recall of previously explored scene contents. Backboard's thread-level persistent memory maps directly to this. Scores the Best Use of Backboard prize. | Low-Medium | One Backboard thread per scene. Write object catalog on exploration. Read on future queries. Simple API — the impressive part is the USE CASE not the integration complexity. |
| Reasoning over spatial relationships — multi-hop ("what is on top of the desk near the window?") | Queries that chain spatial relationships. Demonstrates the GRAPH is doing real work — this is not achievable with a flat CLIP query. Multi-hop traversal = why the graph exists. | High | Requires Query Walker to traverse edges: find "desk" node → follow "next_to(window)" edge → follow "on_top_of" edges from desk. JAC's visit syntax handles this elegantly. |
| Live natural language explanation of spatial relationships | After a query resolves, the system narrates: "The red mug is on the desk, which is next to the window. The desk has a lamp on top of it." Turns graph traversal into language. | Medium | `by llm()` over the sub-graph. Small LLM call. High demo impact for low implementation cost. |

---

## Anti-Features

Things to deliberately NOT build. Each one is a trap that consumes time without moving the demo forward.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Real-time scene update / dynamic Gaussians | 4D LangSplat (dynamic scenes) is a research paper, not a 24-hour feature. Attempting dynamic updates blows up the training pipeline and rendering complexity. | Use a single static scene. The static scene is already impressive enough. |
| Custom CLIP / SAM model training | The PROJECT.md explicitly rules this out. Pretrained ViT-H SAM and ViT-L CLIP are sufficient. Fine-tuning on a hackathon timeline is a trap. | Use pretrained weights from HuggingFace directly. |
| WebXR / AR / mobile support | Not in scope. Multi-platform support in 24 hours fragments effort without adding demo value. | Web-first, desktop browser only. |
| Multi-scene linking / cross-room navigation | Single scene is the demo. Multi-room requires a scene stitching pipeline that doesn't exist yet. | Focus all effort on one well-chosen scene (living room or office with clear objects). |
| User authentication / accounts / multi-user | Not a product, it's a hackathon demo. Auth adds zero demo value and costs 2-3 hours. | Single hardcoded demo mode. Backboard thread ID from env var. |
| Video input pipeline | COLMAP from video requires significant preprocessing. Scaniverse produces COLMAP-compatible output directly from photos. | Scan the room with Scaniverse before/during hackathon, export COLMAP format. |
| Editable / annotatable scenes | Scene editing is a separate product category. Out of scope. | Read-only semantic queries only. |
| Fine-grained permission / RBAC | Zero value in a demo context. | Skip entirely. |
| Base44 integration | PROJECT.md marks this as stretch / low priority. It adds a frontend complexity layer with no prize upside. | Only add if everything else is done and 4+ hours remain. |
| Polished onboarding / tutorial UX | Judges evaluate the demo, not the product. Onboarding is irrelevant. | Give judges a guided script at submission time. |
| Performance optimization / production scaling | Premature optimization in a hackathon. Python server with NumPy + sklearn is sufficient for demo queries. | Ship the naive implementation. Optimize nothing unless something is broken. |

---

## Feature Dependencies

The build order matters. Each tier must complete before the next is viable.

```
Tier 0: Blocking dependencies (must complete first, before any coding)
  LangSplat training (Colab A100) → produces trained .ply with CLIP embeddings
  Scaniverse scene capture → feeds into COLMAP → feeds into LangSplat training

Tier 1: Core pipeline (no demo without these)
  Load .ply Gaussian data → Python server
  CLIP text encoder → query vector
  Cosine similarity against Gaussian embeddings → matching Gaussians
  Spark.js renders scene → browser
  Server sends highlight mask → frontend applies semantic overlay
  Result: "query → scene lights up"

Tier 2: Graph layer (enables spatial reasoning)
  DBSCAN clustering on matching Gaussians → object nodes with centroids
  Geometric relationship tests (Y-offset, XZ proximity) → edges
  JAC graph construction → nodes + edges
  Result: "query → object graph returned"

Tier 3: Communication layer (makes demo legible)
  Chat interface → accepts queries, shows results
  Graph visualization overlay → nodes + edges visible in 3D scene
  Natural language explanation via by llm() → narrates spatial relationships
  Result: "humans understand what the system found"

Tier 4: Agentic layer (prize-winning features)
  Exploration Walker → autonomous traversal, catalogs all objects
  Query Walker → multi-hop graph traversal for complex queries
  Memory Walker → reads/writes to Backboard thread
  Result: "agents think and remember"

Tier 5: Differentiators (wow factor)
  Probability cloud layer → continuous semantic field as opacity gradient
  Multi-resolution queries → semantic granularity demo
  Persistent memory recall → "what did we find last time?"
  Result: "this is research-quality, not a toy"
```

**Critical path:** Tier 0 blocks everything. LangSplat training is the single biggest risk. Must kick off the A100 job within the first 1-2 hours of the hackathon and have a pre-scanned scene ready.

**Fallback:** If Tier 0 fails (training doesn't finish in time), use a pre-existing LangSplat scene from the public LangSplat dataset (their paper provides trained scenes for lerf_figurines, teatime, ramen). This removes the "we scanned our own room" story but preserves all technical features.

---

## MVP Definition

**Minimum that wins prizes:**

Build Tier 1 + Tier 2 + Tier 3 + the Exploration Walker from Tier 4. This gives:

1. A photorealistic 3D scene in the browser
2. Natural language queries that highlight relevant objects
3. A knowledge graph with spatial relationships shown as a 3D overlay
4. An autonomous agent that explores and catalogs objects
5. Natural language explanation of what was found

Add probability cloud layer (Tier 5) if Tier 1-3 are complete by hour 16. It's the single most impressive differentiator and differentiates from every other spatial AI project.

**Defer without regret:**
- Multi-hop reasoning (complex JAC traversal patterns) — impressive but risky to implement correctly in 24 hours
- Persistent memory recall — implement last, after everything else works
- Multi-resolution queries — demo with 2-3 example queries instead of a full system

**Priority order when time is short:**
1. Scene renders in browser (Tier 1 visual)
2. Query highlights objects (Tier 1 semantic)
3. Spatial relationship text output (Tier 2 + `by llm()`)
4. Graph overlay visible in 3D (Tier 3 visual)
5. Exploration Walker runs autonomously (Tier 4 agentic)
6. Probability cloud heatmap (Tier 5 wow)
7. Backboard persistent memory (prize eligibility)
8. Multi-hop query (Tier 4 stretch)

---

## Sources

- [LangSplat: 3D Language Gaussian Splatting (CVPR 2024)](https://langsplat.github.io/)
- [LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS](https://langsplat-v2.github.io/)
- [Semantic Consistent Language Gaussian Splatting for Point-Level Open-vocabulary Querying](https://arxiv.org/html/2503.21767v1)
- [Spark.js: Advanced 3D Gaussian Splatting renderer for THREE.js](https://sparkjs.dev/)
- [JAC Walker Documentation — Jaseci](https://docs.jaseci.org/learn/jac_ref/walker_visit_and_disengage_(osp)/)
- [byLLM for Agentic AI — Jaseci](https://docs.jaseci.org/learn/jac-byllm/agentic_ai/)
- [Backboard.io — Persistent AI Memory](https://backboard.io/)
- [3D Scene Graphs for Spatial AI with NetworkX](https://learngeodata.eu/3d-scene-graphs-for-spatial-ai-with-networkx-and-openusd/)
- [3D Dynamic Scene Graphs: Actionable Spatial Perception (RSS 2020)](https://arxiv.org/abs/2002.06289)
- [Beyond Bare Queries: Open-Vocabulary Object Grounding with 3D Scene Graph](https://arxiv.org/html/2406.07113v3)
- [Spatial Intelligence: The Future of AI — UVA School of Data Science](https://datascience.virginia.edu/news/spatial-intelligence-future-ai)
- [Niantic Spatial & The Large Geospatial Model — 2026 World Models](https://www.nianticspatial.com/blog/world-models-2026)
