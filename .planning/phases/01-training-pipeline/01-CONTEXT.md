# Phase 1: Training Pipeline - Context

**Gathered:** 2026-04-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce LangSplat-trained Gaussian splat artifacts (PLY + .npy latent features + autoencoder.pth) from a real scene, verified and ready for all downstream phases. This phase covers scene capture, Colab training, fallback preparation, and smoke-testing the output. It does NOT cover the Python query server, DBSCAN clustering, or any JAC graph work.

</domain>

<decisions>
## Implementation Decisions

### Scene Capture
- **D-01:** Scan the actual hackathon room at UMich — judges will be in that space, recognition adds demo impact
- **D-02:** Capture 50-80 photos with Scaniverse — good reconstruction quality while keeping training under 3 hours
- **D-03:** Use Scaniverse on iPhone, export photos for COLMAP processing

### Fallback Plan
- **D-04:** Fallback scene is `lerf_figurines` from the public LangSplat dataset — tabletop scene with varied objects, good for spatial queries
- **D-05:** Run fallback and live training in parallel from the start — download and pre-process figurines immediately at minute 0, start venue training after scan at ~30 min. Use whichever produces valid output first.

### Timing & Risk
- **D-06:** Staggered start — kick off figurines (fallback) training at minute 0 of hackathon, start venue training after Scaniverse scan completes (~30 min in)
- **D-07:** Checkpoint every 5K iterations to Google Drive — if Colab session dies, resume from last checkpoint on a new session
- **D-08:** Mount Google Drive as the very first Colab cell — all artifacts and checkpoints survive session timeouts

### Colab Notebook
- **D-09:** All-in-one notebook with sequential cells: setup → COLMAP → SAM → autoencoder → 3DGS RGB (30K iter) → LangSplat training. Single notebook, run top-to-bottom.
- **D-10:** Final artifacts auto-upload to InsForge S3 after training completes (Drive + InsForge dual storage)
- **D-11:** Monitor via Colab output tab — just check the notebook occasionally for training loss prints. No elaborate monitoring system.

### Claude's Discretion
- Training hyperparameters (iterations, learning rate, etc.) — use LangSplat defaults unless research suggests otherwise
- COLMAP configuration — standard defaults for indoor scenes
- CLIP backbone choice — use whatever LangSplat's default is (likely ViT-L/14)
- Exact Scaniverse export settings — standard photo export

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### LangSplat Training
- `.planning/research/STACK.md` — CUDA 11.8 pinning, open-clip-torch version, two separate Python envs
- `.planning/research/PITFALLS.md` — 5-stage sequential pipeline, Colab session timeout risks, SAM preprocessing
- `.planning/research/ARCHITECTURE.md` §Training Layer — Output directory structure, artifact co-location requirements

### Project Context
- `.planning/PROJECT.md` §Technical Architecture — Full pipeline description, LangSplat over Gemini rationale
- `.planning/REQUIREMENTS.md` §Training Pipeline — TRAIN-01 through TRAIN-04 acceptance criteria
- `SpatialMind.md` §Technical Architecture — Original vision doc for background context (note: architecture has pivoted to LangSplat)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- No existing code — this is a greenfield project. Phase 1 creates the first artifacts.

### Established Patterns
- None yet — Phase 1 establishes the training pipeline pattern.

### Integration Points
- Output artifacts (PLY + .npy + autoencoder.pth) feed directly into Phase 2 (Python server loads them into memory)
- InsForge S3 storage receives the trained files — Phase 5 frontend fetches PLY from there
- The Colab notebook is a standalone artifact, not part of the main codebase

</code_context>

<specifics>
## Specific Ideas

- The venue scan should prioritize a section of the hackathon room with clearly distinct objects — tables, chairs, monitors, backpacks, water bottles. Avoid featureless walls.
- The fallback (figurines) should be downloaded and pre-processed BEFORE the hackathon if possible — or at minimum within the first 15 minutes.
- The three-beat demo requires objects with clear spatial relationships — "on top of", "next to". The scan area should have objects at different heights and positions.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-training-pipeline*
*Context gathered: 2026-04-04*
