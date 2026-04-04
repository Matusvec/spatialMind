---
phase: 02-semantic-query-server
plan: 02
subsystem: api
tags: [fastapi, clip, numpy, softmax, highlight-mask, probability-cloud]

# Dependency graph
requires:
  - phase: 02-semantic-query-server/01
    provides: "top_k, soft_scores, CLIP encoder, Gaussian store, /clip/encode endpoint"
provides:
  - "POST /clip/highlight endpoint with hard/soft opacity mask modes"
  - "POST /clip/probability endpoint with temperature-scaled softmax"
  - "highlight_mask() and probability_cloud() similarity functions"
  - "10 integration tests for new functions and endpoints"
affects: [05-frontend-viewer, 03-spatial-clustering]

# Tech tracking
tech-stack:
  added: []
  patterns: [argpartition-based top-K mask, temperature-scaled softmax, min-opacity soft highlighting]

key-files:
  created:
    - server/tests/test_endpoints.py
  modified:
    - server/services/similarity.py
    - server/routers/clip.py

key-decisions:
  - "Soft mode uses 0.3 minimum opacity for visibility of weakest top-k matches"
  - "Temperature default 0.1 for sharp softmax peaks around matching Gaussians"

patterns-established:
  - "Service function pattern: pure numpy functions in similarity.py, router wires to FastAPI"
  - "503 guard pattern: check gaussian_store.is_loaded and clip_encoder before processing"

requirements-completed: [QUERY-04, QUERY-05, QUERY-06]

# Metrics
duration: 4min
completed: 2026-04-04
---

# Phase 02 Plan 02: Highlight Mask and Probability Cloud Summary

**Per-Gaussian highlight mask with hard/soft modes and temperature-scaled softmax probability cloud over all Gaussians, completing the Phase 2 API surface**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-04T19:54:19Z
- **Completed:** 2026-04-04T19:58:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- highlight_mask() computes per-Gaussian opacity values with hard (binary top-k) and soft (gradient with 0.3 min opacity) modes
- probability_cloud() applies temperature-scaled softmax to produce a full probability distribution over all N Gaussians
- Full Phase 2 API surface complete: /health, /clip/encode, /clip/highlight, /clip/probability
- 24 total tests passing (10 new + 14 existing), including multi-resolution behavior verification

## Task Commits

Each task was committed atomically:

1. **Task 1: Add highlight mask and probability cloud functions** - `df4f32e` (test) + `c11fe49` (feat) [TDD]
2. **Task 2: Wire highlight and probability endpoints into FastAPI router** - `df61fe1` (feat)

## Files Created/Modified
- `server/services/similarity.py` - Added highlight_mask() and probability_cloud() functions
- `server/routers/clip.py` - Added POST /clip/highlight and POST /clip/probability endpoints with Pydantic models
- `server/tests/test_endpoints.py` - 10 tests covering mask length/range, hard/soft modes, probability distribution, cross-function consistency, multi-resolution

## Decisions Made
- Soft mode minimum opacity of 0.3 ensures weakest top-k matches remain visible in the 3D viewer
- Temperature default of 0.1 produces sharp peaks that clearly distinguish matching Gaussians from background

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## Known Stubs
None - all functions are fully implemented with real computation.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 2 API surface is complete: all four endpoints registered and tested
- Ready for Phase 3 (spatial clustering) which will consume these similarity functions
- Ready for Phase 5 (frontend) which will call /clip/highlight and /clip/probability

---
*Phase: 02-semantic-query-server*
*Completed: 2026-04-04*
