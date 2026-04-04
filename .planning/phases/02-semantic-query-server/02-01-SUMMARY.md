---
phase: 02-semantic-query-server
plan: 01
subsystem: api
tags: [fastapi, clip, openclip, pytorch, autoencoder, plyfile, cosine-similarity, gaussian-splatting]

# Dependency graph
requires:
  - phase: 01-training-pipeline
    provides: LangSplat training artifacts (PLY + autoencoder.pth)
provides:
  - FastAPI server skeleton on port 8001
  - Autoencoder model matching LangSplat architecture (3-dim to 512-dim decoder)
  - GaussianStore PLY loader with batched decode to CLIP space
  - CLIPEncoder with OpenCLIP and LRU caching
  - top_k similarity retrieval with O(N) argpartition
  - soft_scores for probability cloud computation
  - GET /health endpoint with artifact status reporting
  - POST /clip/encode endpoint returning top-K Gaussian indices with scores
affects: [02-02, 03-spatial-graph, 05-frontend]

# Tech tracking
tech-stack:
  added: [fastapi, uvicorn, open-clip-torch, plyfile, pydantic-settings, scikit-learn, scipy]
  patterns: [lifespan context manager, Pydantic BaseSettings, LRU-cached CLIP encoding, argpartition top-K]

key-files:
  created:
    - server/main.py
    - server/config.py
    - server/models/autoencoder.py
    - server/services/gaussian_store.py
    - server/services/clip_encoder.py
    - server/services/similarity.py
    - server/routers/health.py
    - server/routers/clip.py
    - server/tests/test_query_pipeline.py
    - server/requirements.txt
  modified: []

key-decisions:
  - "Lifespan context manager over deprecated on_event for FastAPI startup"
  - "LRU cache (256 entries) on CLIP encode_text to avoid repeated 200-500ms inference"
  - "argpartition-based top-K for O(N) retrieval instead of full sort"
  - "Graceful artifact loading: server starts without PLY/autoencoder, reports missing via /health"
  - "Batch decode in chunks of 10000 for GPU memory safety"

patterns-established:
  - "Service pattern: thin routers import from services, services are stateless with injected dependencies"
  - "App state dict pattern: global _app_state populated during lifespan, accessed via get_app_state()"
  - "PLY property naming fallback: try language_feature_0 then f_language_feature_0"
  - "Autoencoder dual-load: handle both torch.save(model) and state_dict patterns"

requirements-completed: [QUERY-01, QUERY-02, QUERY-03]

# Metrics
duration: 10min
completed: 2026-04-04
---

# Phase 02 Plan 01: Core Query Pipeline Summary

**FastAPI server with CLIP text-to-Gaussian similarity pipeline: autoencoder decode, OpenCLIP encoding with LRU cache, O(N) top-K retrieval, and /health + /clip/encode endpoints**

## Performance

- **Duration:** 10 min
- **Started:** 2026-04-04T19:41:38Z
- **Completed:** 2026-04-04T19:51:30Z
- **Tasks:** 2
- **Files modified:** 14

## Accomplishments
- Autoencoder model exactly matching LangSplat architecture (encoder: 512->256->128->64->32->3, decoder: 3->16->32->64->128->256->256->512)
- Complete query pipeline: text -> CLIP encode -> cosine similarity -> top-K indices with scores
- 14 unit tests passing with synthetic data (no real PLY or GPU required)
- Server handles missing artifacts gracefully -- starts and reports status via /health

## Task Commits

Each task was committed atomically:

1. **Task 1: Create server skeleton, data models, and core services**
   - `0f35503` (test) - Failing tests for query pipeline (TDD RED)
   - `55b4465` (feat) - Implement all core services (TDD GREEN)
2. **Task 2: Wire FastAPI endpoints and startup lifecycle** - `5ec7d79` (feat)
3. **Housekeeping: .gitignore** - `5a6e839` (chore)

## Files Created/Modified
- `server/main.py` - FastAPI app with lifespan startup, CORS, router includes
- `server/config.py` - Pydantic BaseSettings with .env support and CUDA auto-detect
- `server/models/autoencoder.py` - LangSplat autoencoder (nn.Module) with dual-load utility
- `server/services/gaussian_store.py` - PLY loader, batched decode to 512-dim, L2 normalization
- `server/services/clip_encoder.py` - OpenCLIP text encoder with functools.lru_cache
- `server/services/similarity.py` - top_k (argpartition) and soft_scores functions
- `server/routers/health.py` - GET /health with artifact status
- `server/routers/clip.py` - POST /clip/encode with Pydantic request/response models
- `server/tests/test_query_pipeline.py` - 14 tests covering autoencoder, store, similarity, CLIP
- `server/requirements.txt` - All Python dependencies pinned
- `server/__init__.py` - Package init
- `server/models/__init__.py` - Package init
- `server/services/__init__.py` - Package init
- `server/routers/__init__.py` - Package init

## Decisions Made
- Used lifespan context manager (not deprecated `on_event`) for FastAPI startup/shutdown
- LRU cache with 256 entries on CLIP encode_text to avoid repeated ~200-500ms inference on CPU
- np.argpartition for O(N) top-K instead of full O(N log N) sort
- Server starts gracefully without artifacts, reporting missing items via /health
- Batch size of 10000 for Gaussian decoding to prevent GPU OOM on large scenes
- Created .venv for Arch Linux (PEP 668 compliance)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created virtual environment for Arch Linux**
- **Found during:** Task 1 (dependency installation)
- **Issue:** Arch Linux enforces PEP 668, preventing pip install to system Python
- **Fix:** Created .venv with `python -m venv .venv`, installed all dependencies there
- **Files modified:** .venv/ (not tracked), .gitignore
- **Verification:** All imports succeed, tests pass
- **Committed in:** 5a6e839 (chore commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary for Arch Linux compatibility. No scope creep.

## Issues Encountered
None beyond the venv requirement.

## Known Stubs
None -- all endpoints are fully wired to real service implementations.

## User Setup Required
None - no external service configuration required. Server loads artifacts from SCENE_DIR environment variable (defaults to ./data/scene).

## Next Phase Readiness
- Query pipeline is complete and tested, ready for Plan 02 (clustering + probability clouds)
- POST /clip/encode returns indices + scores that DBSCAN clustering will consume
- soft_scores() function ready for probability cloud endpoint
- Server needs actual LangSplat artifacts (PLY + autoencoder.pth) in SCENE_DIR to serve real queries

---
*Phase: 02-semantic-query-server*
*Completed: 2026-04-04*
