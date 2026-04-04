# Domain Pitfalls

**Domain:** Spatial AI — LangSplat + Gaussian Splatting + JAC agents
**Researched:** 2026-04-04
**Hackathon:** JacHacks, April 4-5, 2026 (24-hour window)

---

## Critical Pitfalls

Mistakes that cause a rewrite, a catastrophic demo failure, or a total pipeline stall.

---

### Pitfall 1: LangSplat Training Is a 5-Stage Sequential Pipeline — Each Stage Can Fail Silently

**What goes wrong:**
The LangSplat README documents five mandatory sequential stages: (1) SAM-based language feature generation via `preprocess.py`, (2) autoencoder training + 3D feature compression, (3) LangSplat training on top of a pre-trained 3DGS RGB checkpoint, (4) rendering, (5) evaluation. Many builders assume it is a single `train.py` call. Skipping or botching any stage produces outputs that look plausible — the scene renders — but the Gaussian embeddings are garbage. CLIP queries will return wrong or empty results with no error messages.

**Why it happens:**
Stage 3 requires an already-trained 3DGS RGB checkpoint as input (the `--start_checkpoint` flag). If you run LangSplat training without a valid RGB checkpoint, it either crashes with a non-obvious error or silently trains from random initialization, producing low-quality embeddings. Stage 2 (autoencoder) trains a scene-specific compression of CLIP features down to 3 dimensions — if this stage is skipped or the checkpoint path is wrong, all embeddings are random noise.

**Consequences:**
CLIP cosine similarity scores will be uniformly random (0.3–0.5 range for everything). Every query returns the same set of Gaussians regardless of the text. DBSCAN will cluster on noise and produce one giant blob or hundreds of micro-clusters. The demo shows "highlighting" but it highlights random geometry.

**Prevention:**
- Map the full 5-stage pipeline before starting Colab training.
- First train vanilla 3DGS RGB to completion (checkpoint at 30K iterations) BEFORE starting LangSplat.
- Verify stage 2 autoencoder checkpoint exists and is non-empty before running stage 3.
- After stage 3, smoke-test CLIP query with a known object (e.g., "chair") against a known scene — expect top matches to cluster visually.

**Detection:**
- Cosine similarity scores are all in the 0.3–0.5 range regardless of query → embeddings are noise.
- All queries return the same Gaussian indices → embedding space is collapsed.
- Stage 3 starts immediately without loading a checkpoint → missing RGB model dependency.

**Phase:** Training setup (first 4 hours of hackathon). Must be unblocked before any other work.

**Recovery if it happens:**
Re-run from the failing stage. LangSplat stages are idempotent — re-running stage 2 or 3 from scratch on Colab costs 30–90 minutes but does not corrupt earlier stages. Keep stage checkpoints in Google Drive, not Colab's ephemeral `/content/`.

---

### Pitfall 2: Colab Session Timeout Kills Unfinished Training, Losing All Progress

**What goes wrong:**
Google Colab has a hard 12-hour session limit (free tier) and a 90-minute idle disconnect. The full LangSplat pipeline (COLMAP preprocessing, 3DGS RGB training to 30K iterations, autoencoder training, LangSplat training) can easily exceed 3–6 hours on A100 for a custom indoor scene. If the session drops mid-training, all in-memory state is lost. Colab's `/content/` filesystem is ephemeral — files not explicitly saved to Drive are gone on reconnect.

**Why it happens:**
The 90-minute idle timer triggers if the browser tab loses focus — common during a hackathon when context-switching between frontend and backend tasks.

**Consequences:**
Total training restart during the critical first 6 hours. A100 availability is not guaranteed on reconnect (you may get L4 or T4, which are 2–4x slower). Lost 3DGS RGB checkpoint means both stage 1 and stage 3 of LangSplat must be redone.

**Prevention:**
- Mount Google Drive as the first cell: `from google.colab import drive; drive.mount('/content/drive')`. Save ALL checkpoints to Drive, not `/content/`.
- Kick off Colab training before any other hackathon task — it is the single longest blocking dependency.
- Add checkpoint saving every 5K iterations in the 3DGS training config.
- Keep browser focused on Colab tab or use a simple keep-alive JavaScript snippet in console.
- Pre-download the SAM checkpoint (`vit_h`) to Drive so it does not need to re-download after disconnect.

**Detection:**
- Colab runtime shows "Disconnected" badge — check immediately.
- Drive checkpoint folder timestamp shows last save time > 2 hours ago.

**Phase:** Training kickoff (hour 0–1 of hackathon). Mount Drive before doing anything else.

**Recovery if it happens:**
Reconnect and load from the last saved checkpoint. If no checkpoint exists, restart training. Budget 90 minutes for worst-case A100 restart. Have a fallback pre-trained scene ready (the LangSplat authors provide example scenes — download to Drive as insurance before the hackathon).

---

### Pitfall 3: COLMAP Fails on Low-Texture Indoor Scenes

**What goes wrong:**
COLMAP's default SIFT feature extractor struggles with scenes that have large uniform surfaces: white walls, ceilings, plain floors. It fails to find enough keypoint correspondences, producing a sparse or completely failed reconstruction. LangSplat requires COLMAP-derived camera poses — if COLMAP outputs fewer than ~50% of images as registered, the downstream 3DGS training will produce a visually broken scene with missing geometry and floating Gaussians.

**Why it happens:**
SIFT is a local feature detector that requires textural variation. Indoor scenes captured for a spatial AI demo (living room, office) frequently have large featureless surfaces. Scaniverse exports work well for Gaussian splatting's own pipeline but may not export COLMAP-compatible formats (binary sparse reconstruction) directly — the conversion step is non-trivial and sparsely documented.

**Consequences:**
No COLMAP output = no LangSplat training = no demo. Even partial failures (30% images unregistered) cause 3DGS to produce large holes in the scene, making the spatial graph inaccurate.

**Prevention:**
- Capture scene with maximum overlap: move slowly, take overlapping images from multiple angles, never rotate in place.
- Add textured objects to the scene before capture (books, rugs, posters) if shooting a featureless room.
- Verify COLMAP registration success: check that `images.bin` registers 80%+ of input images before starting 3DGS.
- If COLMAP fails, switch to the ALIKED feature extractor (added in COLMAP 3.8+) via `--ImageReader.single_camera 1 --SiftExtraction.use_gpu 1`.
- Alternatively: capture with Scaniverse which produces its own reconstruction, then export in a compatible format. Validate the export workflow BEFORE the hackathon.

**Detection:**
- COLMAP `model_info` shows fewer than 80% of images registered.
- `sparse/0/points3D.bin` is empty or has fewer than 1000 points.
- 3DGS training loss does not decrease after 1K iterations.

**Phase:** Scene capture (pre-hackathon or hour 0). Test capture workflow before the event.

**Recovery if it happens:**
Use a pre-captured fallback scene. The LangSplat repo links to pre-processed scenes (lerf_ovs dataset). Download one as a Colab-ready backup before the hackathon. Never be blocked on a single live capture.

---

### Pitfall 4: Browser Memory Exhaustion From Large PLY / Splat Files

**What goes wrong:**
A LangSplat-trained scene from 100–200 photos produces a `.ply` file ranging from 200 MB to 2+ GB (3–6 million Gaussians). Browsers have implicit WebGL buffer limits (typically 1–2 GB of GPU VRAM addressable from JavaScript) and explicit JavaScript heap limits (~4 GB on Chrome). Loading a raw `.ply` file of this size will either cause a hard out-of-memory crash or sustained jank (>5 second load freeze) that destroys the live demo impression.

**Why it happens:**
Each Gaussian has 59 floats of data in full SH (spherical harmonic) representation. 3 million Gaussians × 59 × 4 bytes = ~700 MB of raw GPU buffer. This is before the JavaScript decode pass or the sorting buffer Spark.js maintains for depth-correct rendering.

**Consequences:**
Browser tab crashes mid-demo. Or a 30-second loading spinner while judges watch — which reads as "it doesn't work."

**Prevention:**
- After training, convert `.ply` to `.spz` or `.ksplat` format (5–10x compression, Spark.js supports both).
- Cap the training at 7K iterations for the demo scene if file size is a concern — fewer Gaussians, smaller file, nearly indistinguishable to a non-technical judge.
- Reduce SH degree from 3 to 0 (color only, no view-dependent appearance) during export — cuts per-Gaussian data by ~70%.
- Serve the splat file from a CDN (Cloudflare R2 / Vercel Edge) with gzip transfer encoding, not a local Python dev server.
- Test the full browser load cycle on demo hardware before presentation.

**Detection:**
- Chrome DevTools Memory tab shows heap climbing past 1 GB while loading.
- Spark.js load callback never fires within 10 seconds.
- GPU process memory (chrome://gpu-internals) exceeds 1.5 GB.

**Phase:** Frontend rendering integration. Test during the first frontend milestone, not at presentation time.

**Recovery if it happens:**
Export a reduced splat (lower iteration count or pruned Gaussians). Spark.js supports streaming load — implement it. If catastrophic, pre-render a video flythrough as demo fallback.

---

### Pitfall 5: CLIP Threshold Is Not Universal — Fixed Thresholds Will Return Garbage on Novel Queries

**What goes wrong:**
A common implementation approach is to compute cosine similarity between a text query embedding and all Gaussian embeddings, then apply a fixed threshold (e.g., `sim > 0.5`) to find matches. This fails in production: CLIP similarity scores are not calibrated — a similarity of 0.5 for "chair" may mean a good match while 0.5 for "window" may be noise. The score distribution shifts based on query specificity and scene content.

**Why it happens:**
CLIP is trained to maximize similarity between matched image-text pairs but does not guarantee an absolute numeric scale. The embedding space is relative, not absolute. LangSplat compounds this by compressing CLIP embeddings through a scene-specific autoencoder (to 3 dims), which further warps the similarity space.

**Consequences:**
Queries for common objects ("chair", "table") return good results in testing but fail on less common objects ("plant", "lamp") because the threshold is tuned to the training queries. Live demo shows "no results found" or highlights half the scene indiscriminately.

**Prevention:**
- Use relative thresholding: take the top-N Gaussians by similarity score rather than a fixed cutoff.
- Alternatively, use softmax / temperature scaling on similarities so that scores sum to 1 — then threshold on the normalized distribution.
- Test queries beyond the obvious ones (chair, table) in smoke-testing: test "cushion", "plant", "switch" to see where the score distribution breaks.
- Pre-compute similarity score distributions for the demo scene and choose thresholds from the 90th percentile of the positive class, not a guess.

**Detection:**
- Query for a known object returns 0 results with threshold 0.5, but returns results with threshold 0.3.
- Query for "room" highlights individual Gaussians rather than all geometry.
- Top-5 results for clearly different queries (e.g., "floor" and "ceiling") are the same Gaussians.

**Phase:** CLIP query server. Must be tuned before DBSCAN integration.

**Recovery if it happens:**
Switch to top-K retrieval immediately. Use K=500 or K=1000 Gaussians as input to DBSCAN rather than a similarity cutoff. This bypasses threshold sensitivity entirely.

---

### Pitfall 6: DBSCAN epsilon Parameter Produces One Giant Cluster or All Noise

**What goes wrong:**
DBSCAN's two parameters — `epsilon` (neighborhood radius) and `min_samples` — are highly sensitive to the scale and density of the input data. Gaussian splatting coordinates are in scene-space units (typically meters or a normalized range), but LangSplat Gaussian positions are not standardized across scenes. An `epsilon` value that works for a bedroom scene (coordinates spanning 0–5 meters) will produce either one giant merged cluster or zero clusters when applied to a different scene with a different coordinate scale.

**Why it happens:**
The 3D coordinates of Gaussians are in whatever world-space scale COLMAP computed. Different scenes, different photo distances, and different COLMAP configurations produce wildly different coordinate ranges. Hardcoding `epsilon=0.1` may mean 10 cm in one scene and 10 meters in another.

**Consequences:**
Object nodes in the JAC graph are wrong (one "room" node instead of individual objects, or 5000 singleton "objects"). Spatial relationships (on_top_of, next_to) computed from cluster centroids are meaningless. The entire graph abstraction fails.

**Prevention:**
- Never hardcode `epsilon`. Compute it dynamically: after filtering Gaussians by CLIP similarity, compute the k-distance graph (sorted pairwise distances for k = min_samples) and use the "knee" as epsilon.
- Alternatively, normalize Gaussian coordinates to [0,1] range before DBSCAN, then apply a standardized epsilon of ~0.03–0.05.
- Test DBSCAN on the actual demo scene, not a different dataset.
- Use `sklearn.cluster.DBSCAN` with `algorithm='ball_tree'` for faster computation on 3D data.
- Rule of thumb: `min_samples = 2 * num_dimensions = 6` for 3D spatial coordinates.

**Detection:**
- DBSCAN returns a single cluster label (all points labeled 0) → epsilon too large.
- DBSCAN returns N clusters where N ≈ number of input points (all labeled -1 or unique) → epsilon too small.
- Cluster centroids are all within 0.1 scene units of each other → scale problem.

**Phase:** DBSCAN clustering (Phase 3). Requires the actual demo scene to tune against.

**Recovery if it happens:**
Implement dynamic epsilon using the k-distance elbow method. The full computation on 10K–100K points takes < 5 seconds in Python. This is a 30-minute fix if caught early, a 4-hour rewrite if caught at demo time.

---

### Pitfall 7: JAC Walker Infinite Traversal on Cyclic Spatial Graph

**What goes wrong:**
Spatial graphs have cycles: Room → Table → Chair → Room (Chair is inside Room). A JAC walker traversing edges without explicit loop prevention will revisit nodes indefinitely, consuming all memory and timing out the HTTP request without returning results.

**Why it happens:**
JAC's `visit` statement continues traversal to adjacent nodes. In a spatial knowledge graph where every node is connected to the enclosing Room node and Room connects back to everything, walkers without `disengage` or visited-set tracking will loop forever. Beginners confuse `return` (exits current function, walker continues) with `disengage` (terminates the entire walker).

**Consequences:**
Walker endpoints hang indefinitely. JAC server appears unresponsive from the frontend. All spatial queries time out during the demo.

**Prevention:**
- Always include a visited-node set in walker `has` fields: `has visited: set[str] = set();`
- At the top of every `with entry { ... }` block, check `if here.id in self.visited { disengage; }` then `self.visited.add(here.id);`
- Use `disengage` (not `return`) when a search goal is met.
- For NavigationWalker and ExplorationWalker, add a max_depth counter in walker state.
- Test each walker with a minimal graph containing a cycle before attaching it to the full spatial graph.

**Detection:**
- Walker endpoint does not return within 5 seconds for a small graph.
- JAC server CPU pegged at 100%.
- Graph has fewer than 20 nodes but walker visits > 1000 nodes (shows in debug logging).

**Phase:** JAC walker implementation (Phase 4). Unit-test walkers on synthetic graphs first.

**Recovery if it happens:**
Add an emergency max_visits counter. Any walker exceeding 500 node visits should `disengage` and return partial results. Implement this as a defensive default in a base walker class.

---

## Technical Debt Patterns

Patterns that do not break the hackathon build immediately but create compounding problems in the second half of the 24 hours.

---

### Debt 1: Training With Provided Sample Scenes, Demoing With Custom Scene

**What goes wrong:**
The entire pipeline is developed and debugged against LangSplat's provided example scenes (lerf_ovs/figurines, lerf_ovs/ramen). CLIP queries work great on these — they were used in the paper. At demo time, a live-captured room scene is substituted. Everything breaks because the new scene's coordinate scale, Gaussian density, embedding distribution, and SAM segmentation quality are all different.

**Prevention:**
Capture the demo scene during the first 4 hours. Run the full pipeline against it within the first 8 hours. Never demo on untested data.

---

### Debt 2: Hardcoded Scene Paths and Gaussian Data in Python Server

**What goes wrong:**
Python backend hardcodes the path to the PLY file and the numpy array of embeddings as module-level globals. Works fine until the scene changes, a second scene is needed, or the Colab output path changes after a session restart.

**Prevention:**
Load scene data from a configurable path at startup. Accept scene name as a server argument. Store the loaded numpy arrays in a singleton `scene_context` dict, not bare globals.

---

### Debt 3: React Frontend Polls HTTP Instead of Using WebSockets

**What goes wrong:**
The frontend sends a GET request to `/query?text=chair`, the backend runs CLIP similarity + DBSCAN + JAC walker, and returns highlight masks as JSON. This works for a single query. Under demo conditions (multiple rapid queries, concurrent users), HTTP polling adds visible latency and drops requests when the server is busy with a previous query.

**Prevention:**
Use a single WebSocket connection from the frontend to the Python server. Queue queries server-side. Stream partial results back (send Gaussian indices first, then graph nodes, then spatial relationships) so the frontend starts rendering immediately.

---

### Debt 4: Spark.js Scene Loaded on Component Mount, Never Unloaded

**What goes wrong:**
`useEffect(() => { sparky.load(...) }, [])` is the natural React pattern. If the user navigates away and back, a second Spark.js renderer is created on the same canvas, causing WebGL context conflicts. The old renderer's GPU buffers are never freed.

**Prevention:**
Return a cleanup function from the effect that calls `sparky.dispose()`. Ensure canvas ref is stable across renders.

---

## Integration Gotchas

Problems that only appear at the seam between two subsystems.

---

### Gotcha 1: Gaussian Indices Are Not Stable Across Renders

**What goes wrong:**
The Python server returns a list of Gaussian indices (e.g., `[102, 4503, 8821]`) to the frontend for highlight rendering. The frontend passes these to Spark.js as a highlight mask. If the PLY file was re-exported (different training run, different pruning threshold), the Gaussian indices shift. The frontend highlights wrong geometry.

**Prevention:**
Tie the server-side numpy array and the frontend's Spark.js load call to the same PLY file version. Use a content hash or training run ID as a versioning mechanism. Never serve highlight indices from a different model version than the loaded splat.

---

### Gotcha 2: CORS Blocks WebSocket Connection From Vite Dev Server

**What goes wrong:**
Frontend runs on `localhost:5173` (Vite default). Python FastAPI server runs on `localhost:8000`. WebSocket handshake is blocked by browser CORS policy if the FastAPI app does not explicitly allow `http://localhost:5173` as an allowed origin.

**Prevention:**
Add CORS middleware to FastAPI on startup:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://your-vercel-domain.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
Add this in the first 30 minutes of backend work — it is always needed and wastes 45 minutes when discovered at integration time.

---

### Gotcha 3: JAC Walker Returns Node Data as Nested Dicts, Not Typed Objects

**What goes wrong:**
When a JAC walker reports node data back through a walker result or HTTP endpoint, the data arrives as untyped nested dictionaries in Python. The frontend expects a typed JSON schema (e.g., `{ id, label, centroid: [x,y,z], edges: [...] }`). Without an explicit serialization layer, the JAC output schema drifts silently and the frontend JSON parsing fails with unhelpful errors.

**Prevention:**
Define Pydantic models for all data crossing the JAC-to-Python-to-frontend boundary. Validate walker output against the schema at the FastAPI layer before sending to the browser.

---

### Gotcha 4: Spark.js Requires Three.js r126+ for Async GPU Readback

**What goes wrong:**
Spark.js uses `readRenderTargetPixelsAsync` (added in Three.js r126) for non-blocking depth sorting. If the project was initialized with an older Three.js version (e.g., from a template that pins r122), Spark.js crashes immediately on initialization with a cryptic undefined function error, not a version error.

**Prevention:**
Check `import * as THREE from 'three'` and `THREE.REVISION` against Spark.js minimum requirement on installation. Pin Three.js at the version Spark.js specifies in its package.json `peerDependencies`, not "latest".

---

## Performance Traps

Issues that work fine in development but visibly degrade under demo conditions.

---

### Trap 1: CLIP Text Encoding Called on Every Query, Not Cached

**What goes wrong:**
`clip.encode_text(query)` runs a transformer forward pass on every HTTP request. On CPU, this takes 200–500 ms per call. Under the demo's "live query" UX, every keystroke potentially triggers an encode call, stacking latency and making the system feel sluggish.

**Prevention:**
Cache CLIP text encodings with a simple `lru_cache` on the normalized query string. Pre-encode a dictionary of likely demo queries at server startup. Run CLIP encoding on GPU at startup and warm the cache.

---

### Trap 2: DBSCAN Runs on All Matching Gaussians — 100K+ Points Takes Seconds

**What goes wrong:**
A broad query like "furniture" may match 100K+ Gaussians. Running DBSCAN on 100K 3D points with a naive implementation takes 10–30 seconds per query — blocking the response. The demo gives the impression the system is broken.

**Prevention:**
Cap the CLIP similarity retrieval at the top-N before passing to DBSCAN (N=5000 is sufficient for most queries). Use `sklearn.cluster.DBSCAN` with `algorithm='ball_tree'` and `leaf_size=30`. Pre-compute DBSCAN clusters for the 20 most likely demo queries and cache results.

---

### Trap 3: Graph Visualization Renders All Nodes and Edges — Large Graphs Kill FPS

**What goes wrong:**
The JAC spatial graph for a moderately complex scene may have 100+ nodes and 500+ edges. Rendering all of them as 3D lines + labels in Three.js, on top of the Gaussian splat renderer, drops FPS below 30 on non-gaming hardware.

**Prevention:**
Render only the subgraph relevant to the current query (queried node + 1-hop neighborhood). Use LOD: hide edge labels below a zoom threshold. Limit simultaneous rendered edges to 50. The Three.js `LineSegments` batch draw call is far more efficient than individual `Line` objects — batch all edges into a single geometry.

---

## "Looks Done But Isn't" Checklist

Signs the demo appears to work but will fail under inspection or judges' own queries.

- [ ] CLIP queries return results but the highlighted Gaussians don't visually correspond to the named object (embeddings from wrong training stage).
- [ ] Spatial graph shows "on_top_of" edges between objects that are not actually stacked (Y-axis threshold miscalibrated for scene scale).
- [ ] DBSCAN produces clusters but each "object" cluster contains geometry from multiple real objects (epsilon too large).
- [ ] Walker traversal completes but "visited" nodes appear twice in the response (walker revisiting without deduplication).
- [ ] Backboard memory stores spatial query history but does not load it on next session (persistence write succeeds but read path is wrong).
- [ ] Probability clouds appear on screen but all Gaussians have uniform soft-assignment probability regardless of query (cloud layer is a placeholder, not wired to CLIP similarity scores).
- [ ] Spark.js renders the scene but highlight masks are not visible (z-fighting between highlight shader and splat renderer — needs depth offset).
- [ ] Graph visualization nodes render at correct 3D positions in isolated test but appear in wrong location when overlaid on live splat (coordinate systems not aligned — LangSplat world space vs Three.js scene space require explicit transform).
- [ ] Chat interface accepts input and shows a response but the response is from a generic LLM prompt, not from the spatial graph walker (JAC walker is not called, just a raw LLM call).
- [ ] Demo works on the developer's machine but not on the projection screen (WebGL context lost due to driver difference — always test on the presentation machine before the demo window).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Scene capture (pre-hackathon) | COLMAP fails on featureless walls | Capture texture-rich scene; test COLMAP reconstruction on a test capture before hackathon day |
| Colab training kickoff | Session timeout without Drive mount | Mount Drive in cell 1; kick off before anything else |
| LangSplat pipeline staging | Missing RGB checkpoint for stage 3 | Document and verify the 5 pipeline stages; smoke-test output at each stage |
| CLIP query server | Fixed similarity threshold breaks novel queries | Use top-K retrieval as primary mechanism; threshold as secondary filter |
| DBSCAN clustering | Wrong epsilon for scene coordinate scale | Normalize coords or compute epsilon dynamically from k-distance graph |
| JAC walker implementation | Infinite traversal on cyclic graph | Visited-set pattern in all walkers; disengage vs return awareness |
| Spark.js integration | Three.js version mismatch or WebGL2 missing | Pin Three.js version per Spark.js peer deps; test in target browser on day 1 |
| Frontend-backend seam | CORS blocks WebSocket, Gaussian index drift | Add CORS middleware first; version-lock PLY file to server model |
| Live demo | Browser crash from large PLY file | Convert to .spz or .ksplat; cap Gaussians at demo resolution; test on presentation hardware |
| Judge-driven queries | CLIP fails on unexpected query terms | Pre-test 30+ diverse queries; have a scripted demo path as fallback |

---

## Sources

- [LangSplat GitHub — Issues and README](https://github.com/minghanqin/LangSplat) — MEDIUM confidence (training stages, CUDA memory, custom dataset issues)
- [COLMAP FAQ](https://colmap.github.io/faq.html) — HIGH confidence (feature extraction failures, ALIKED extractor option)
- [Spark.js Documentation](https://sparkjs.dev/docs/overview/) — HIGH confidence (WebGL2 requirement, Three.js r126 async readback)
- [WebSplatter — WebGPU vs WebGL for Gaussian Splatting](https://arxiv.org/html/2602.03207) — MEDIUM confidence (memory limits, compute limits in browser)
- [Making Gaussian Splats Smaller — Aras Pranckevičius](https://aras-p.info/blog/2023/09/13/Making-Gaussian-Splats-smaller/) — HIGH confidence (file size, compression formats)
- [Spark.js Hacker News Discussion](https://news.ycombinator.com/item?id=44249565) — MEDIUM confidence (Three.js version requirement, WebGL2 extension conflicts)
- [JAC Walker visit and disengage](https://docs.jaseci.org/learn/jac_ref/walker_visit_and_disengage_(osp)/) — HIGH confidence (disengage vs return, infinite loop prevention)
- [DBSCAN Parameter Selection Guide](https://sefidian.com/2022/12/18/how-to-determine-epsilon-and-minpts-parameters-of-dbscan-clustering/) — HIGH confidence (k-distance knee method, epsilon sensitivity)
- [CLIP embedding limitations in 3D retrieval](https://arxiv.org/html/2411.05195v1) — HIGH confidence (scores not calibrated, threshold instability across queries)
- [Google Colab session limits](https://github.com/googlecolab/colabtools/issues/1053) — HIGH confidence (12-hour hard cap, 90-minute idle disconnect)
- [Semantic Consistent Language Gaussian Splatting](https://arxiv.org/html/2503.21767v1) — MEDIUM confidence (CLIP threshold issues in open-vocabulary 3D querying)
- [3DGS textureless scene reconstruction failures](https://arxiv.org/html/2402.14650) — MEDIUM confidence (SfM failure modes in low-texture regions)
