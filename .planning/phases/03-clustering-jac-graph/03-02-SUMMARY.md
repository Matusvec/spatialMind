---
phase: 03-clustering-jac-graph
plan: 02
subsystem: api
tags: [fastapi, pydantic, graph-builder, scene-graph, caching, orchestration]

# Dependency graph
requires:
  - phase: 03-clustering-jac-graph
    plan: 01
    provides: "cluster_gaussians, extract_spatial_relations, build_hierarchy services"
  - phase: 02-query-server
    provides: "GaussianStore, CLIPEncoder, similarity.top_k, FastAPI app structure"
provides:
  - "build_scene_graph orchestration function combining top_k + DBSCAN + spatial relations + hierarchy"
  - "POST /scene/build endpoint with Pydantic request/response models and caching"
  - "Scene graph payload schema: nodes (obj_NNN), edges, hierarchy, metadata"
affects: [04-walker-agents, 05-frontend]

# Tech tracking
tech-stack:
  added: []
  patterns: [scene graph orchestration pipeline, endpoint result caching in app_state, TDD red-green]

key-files:
  created:
    - server/services/graph_builder.py
    - server/routers/scene.py
    - server/tests/test_graph_builder.py
    - server/tests/test_scene_endpoint.py
  modified:
    - server/main.py
    - server/services/similarity.py

key-decisions:
  - "Scene graph cached in app_state['scene_graph'] after first build; force=true to rebuild"
  - "Node IDs use obj_NNN zero-padded format; labels are placeholder object_N (Phase 4 adds LLM descriptions)"
  - "Pydantic Field(alias='from') for edge serialization since 'from' is a Python keyword"

patterns-established:
  - "Scene graph payload: {nodes, edges, hierarchy, metadata} consumed by frontend and JAC walker agents"
  - "Endpoint caching pattern: check app_state cache before expensive computation"

requirements-completed: [GRAPH-01, GRAPH-02, GRAPH-03, GRAPH-04]

# Metrics
duration: 4min
completed: 2026-04-04
---

# Phase 3 Plan 2: Graph Builder and Scene Build Endpoint Summary

**Scene graph builder orchestrating DBSCAN clustering + spatial relations + CLIP hierarchy via POST /scene/build with Pydantic validation and app_state caching**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-04T21:19:43Z
- **Completed:** 2026-04-04T21:23:54Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Graph builder service orchestrating top_k -> cluster_gaussians -> extract_spatial_relations -> build_hierarchy into a single scene graph payload
- POST /scene/build endpoint with full Pydantic request/response models, 503 guards, and result caching
- 17 new tests (9 graph builder + 8 endpoint), 60 total tests passing with zero regressions
- Fixed edge case bug in top_k where k >= array length caused argpartition crash

## Task Commits

Each task was committed atomically:

1. **Task 1: Graph builder service** - `a28d407` (test: RED), `63beeb6` (feat: GREEN)
2. **Task 2: POST /scene/build endpoint** - `8643fce` (test: RED), `d14149f` (feat: GREEN)

_TDD tasks have separate test and implementation commits._

## Files Created/Modified
- `server/services/graph_builder.py` - Orchestrates clustering pipeline into scene graph payload
- `server/routers/scene.py` - POST /scene/build endpoint with Pydantic models and caching
- `server/main.py` - Scene router registration and scene_graph app_state key
- `server/services/similarity.py` - Fixed top_k edge case when k >= array length
- `server/tests/test_graph_builder.py` - 9 unit tests for graph builder
- `server/tests/test_scene_endpoint.py` - 8 integration tests for scene endpoint

## Decisions Made
- Scene graph cached in app_state after first build; force=true parameter to rebuild (avoids per-query re-clustering per ARCHITECTURE.md anti-pattern 3)
- Node IDs use zero-padded obj_NNN format for consistent sorting; labels are placeholder object_N pending Phase 4 LLM descriptions
- Pydantic Field(alias="from") for edge from/to serialization since "from" is a Python reserved keyword

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed top_k edge case when k >= array length**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** np.argpartition crashes with ValueError when kth equals array length (100 Gaussians, k clamped to 100)
- **Fix:** Added early return path when k >= len(similarities) that uses full argsort instead of argpartition
- **Files modified:** server/services/similarity.py
- **Verification:** All 60 tests pass including existing similarity tests
- **Committed in:** 63beeb6 (part of Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Edge case fix in pre-existing code. No scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functions are fully implemented with real logic. Node labels are placeholder "object_N" by design; Phase 4 adds LLM-generated descriptions.

## Next Phase Readiness
- Scene graph builder and endpoint are ready for consumption by JAC walker agents (Phase 4)
- Scene graph payload schema (nodes, edges, hierarchy, metadata) is the contract for frontend visualization
- All 60 tests pass across all phases

## Self-Check: PASSED

All 4 created files verified on disk. All 4 commit hashes verified in git log.

---
*Phase: 03-clustering-jac-graph*
*Completed: 2026-04-04*
