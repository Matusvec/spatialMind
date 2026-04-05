---
phase: 05-frontend-insforge
plan: 02
subsystem: ui
tags: [spark-js, gaussian-splatting, clip, highlighting, heatmap, three-js, orbit-controls]

# Dependency graph
requires:
  - phase: 05-frontend-insforge
    plan: 01
    provides: "/api adapter router with /api/clip/highlight, /api/clip/probability, /api/scene/splat endpoints"
  - phase: 02-clip-server
    provides: "CLIP cosine similarity, highlight mask, probability cloud endpoints"
provides:
  - "SparkRenderer with real splat loading from backend /api/scene/splat"
  - "Per-Gaussian semantic highlighting via Spark.js Dyno API on CLIP highlight masks"
  - "Probability cloud heatmap with blue-to-red color ramp via Dyno API"
  - "OrbitControls for camera interaction in 3D scene"
  - "Query-driven highlight/heatmap pipeline from chat input to 3D visualization"
affects: [05-03, 06-demo]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Batched Dyno API calls (10K/frame) to avoid blocking main thread on 100K+ Gaussians"
    - "AbortController for canceling in-flight highlight/heatmap fetches on prop change"
    - "Blue-to-red color ramp for probability visualization (score < 0.2 blue, 0.2-0.5 cyan, 0.5-0.7 yellow, > 0.7 red)"

key-files:
  created: []
  modified:
    - room-cognito/src/config.ts
    - room-cognito/src/components/SparkRenderer.tsx
    - room-cognito/src/components/Viewport3D.tsx
    - room-cognito/src/pages/Index.tsx

key-decisions:
  - "Batch size 10K per animation frame for Dyno color updates -- balances responsiveness vs throughput on 100K+ Gaussian scenes"
  - "AbortController on highlight/heatmap fetches -- prevents race conditions when queries change rapidly"
  - "shamefully-hoist=true in .npmrc -- resolves @tanstack/query-core pnpm hoisting issue for build"

patterns-established:
  - "SparkRenderer highlight/heatmap props driven by parent query state"
  - "Prop flow: Index.tsx lastQuery -> Viewport3D -> SparkRenderer highlightQuery/heatmapQuery"

requirements-completed: [UI-01, UI-03, UI-05]

# Metrics
duration: 5min
completed: 2026-04-05
---

# Phase 05 Plan 02: Spark.js Semantic Highlighting Summary

**Spark.js renderer wired to real backend splat loading with per-Gaussian CLIP highlighting (golden mask) and probability cloud heatmap (blue-to-red gradient) via Dyno API**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-05T00:58:38Z
- **Completed:** 2026-04-05T01:04:02Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- SparkRenderer loads real splat files from backend /api/scene/splat instead of hardcoded localhost
- Typing a query highlights matching Gaussians in golden yellow via CLIP highlight mask endpoint
- Heatmap toggle renders continuous probability cloud with blue-to-red color ramp across all Gaussians
- OrbitControls enable camera interaction (pan, zoom, rotate) in the 3D scene
- Batched color application (10K Gaussians per frame) prevents UI freezing on large scenes

## Task Commits

Each task was committed atomically:

1. **Task 1: Update config.ts and SparkRenderer to load splat from backend and support semantic highlighting** - `5d38f06` (feat)
2. **Task 2: Wire Viewport3D to pass highlight and heatmap queries to SparkRenderer** - `16bd8ae` (feat)

## Files Created/Modified
- `room-cognito/src/config.ts` - API_URL changed from localhost:3001 to localhost:8001
- `room-cognito/src/components/SparkRenderer.tsx` - Full rewrite with highlight mask, probability cloud, OrbitControls, batch Dyno API
- `room-cognito/src/components/Viewport3D.tsx` - Added lastQuery prop, passes highlightQuery/heatmapQuery/heatmapEnabled to SparkRenderer
- `room-cognito/src/pages/Index.tsx` - Added lastQuery state tracking, passes to Viewport3D on answered status
- `room-cognito/.npmrc` - Added shamefully-hoist=true for pnpm dependency resolution

## Decisions Made
- Batch size 10K per animation frame for Dyno color updates -- balances responsiveness vs throughput
- AbortController on highlight/heatmap fetches -- prevents race conditions on rapid query changes
- shamefully-hoist=true needed to resolve @tanstack/query-core pnpm hoisting issue (pre-existing)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added .npmrc with shamefully-hoist for pnpm build**
- **Found during:** Task 1 (build verification)
- **Issue:** pnpm strict hoisting prevented @tanstack/query-core resolution, causing build failure
- **Fix:** Created .npmrc with shamefully-hoist=true, reinstalled dependencies
- **Files modified:** room-cognito/.npmrc
- **Verification:** Build succeeds after fix
- **Committed in:** 5d38f06 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Pre-existing pnpm hoisting issue, not caused by plan changes. Fix is standard pnpm configuration.

## Issues Encountered

- room-cognito is a separate git repo (has its own .git directory), so commits go to room-cognito repo, not the parent spatialMind repo

## Known Stubs

None -- all data flows are wired to real backend endpoints.

## Next Phase Readiness
- Semantic highlighting pipeline fully wired: chat input -> CLIP highlight mask -> per-Gaussian golden overlay
- Heatmap pipeline fully wired: chat input + toggle -> CLIP probability cloud -> blue-to-red gradient
- Ready for 05-03 (graph visualization overlay) and 06 (demo polish)

## Self-Check: PASSED

- All 4 modified files exist on disk
- Both task commits verified in room-cognito git log (5d38f06, 16bd8ae)
- SUMMARY.md created at expected path

---
*Phase: 05-frontend-insforge*
*Completed: 2026-04-05*
