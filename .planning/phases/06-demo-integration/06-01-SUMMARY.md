---
phase: 06-demo-integration
plan: 01
subsystem: infra
tags: [demo, launcher, health-check, fallback, scene-graph, startup]

# Dependency graph
requires:
  - phase: 02-semantic-query
    provides: "CLIP encoder, Gaussian store, /health endpoint"
  - phase: 03-scene-graph
    provides: "build_scene_graph service"
provides:
  - "One-command demo launcher (scripts/demo_launch.py)"
  - "Auto scene-graph build on server startup"
  - "PLY fallback scene detection"
  - "Full pipeline readiness via /api/health"
affects: [06-demo-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: ["fallback scene detection", "auto scene-graph on startup", "pipeline_ready health field"]

key-files:
  created: [scripts/demo_launch.py]
  modified: [server/main.py, server/config.py, server/routers/health.py]

key-decisions:
  - "Scene source tracked as app_state field (live/fallback/none) for health endpoint"
  - "Auto scene-graph build uses try/except — server starts even if graph build fails"

patterns-established:
  - "Fallback pattern: try primary path, then fallback, log which was used"
  - "Pipeline readiness: pipeline_ready = ply_loaded AND clip_ready AND scene_graph_ready"

requirements-completed: [DEMO-02, DEMO-03]

# Metrics
duration: 3min
completed: 2026-04-05
---

# Phase 06 Plan 01: Demo Launch and Readiness Summary

**One-command demo launcher with PLY fallback detection, auto scene-graph build on startup, and full pipeline readiness health endpoint**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-05T01:27:42Z
- **Completed:** 2026-04-05T01:30:07Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Server startup auto-detects live vs fallback scene PLY and logs which was used
- Scene graph auto-builds during startup when all artifacts present (no manual trigger)
- /api/health returns pipeline_ready, clip_ready, scene_graph_ready, scene_source
- Single-command launcher checks scene availability, starts server, polls health, prints demo instructions

## Task Commits

Each task was committed atomically:

1. **Task 1: Add scene fallback logic and auto scene-graph build to server startup** - `6c1d514` (feat)
2. **Task 2: Create one-command demo launcher script** - `3ea78c1` (feat)

## Files Created/Modified
- `server/config.py` - Added fallback_scene_dir setting
- `server/main.py` - Fallback PLY loading, scene_source tracking, auto scene-graph build in lifespan
- `server/routers/health.py` - Extended with clip_ready, scene_graph_ready, pipeline_ready, scene_source fields
- `scripts/demo_launch.py` - One-command demo launcher with scene detection, server launch, health polling, demo instructions

## Decisions Made
- Scene source tracked as `_app_state["scene_source"]` (live/fallback/none) to expose via health endpoint
- Auto scene-graph build wrapped in try/except so server starts even if graph build fails
- Demo launcher uses only stdlib (subprocess, urllib, json, argparse) for zero extra dependencies

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Demo launcher ready for use: `python scripts/demo_launch.py`
- Pipeline readiness endpoint can be polled by any monitoring tool
- Phase 06 Plan 02 can build on this for final demo polish

---
*Phase: 06-demo-integration*
*Completed: 2026-04-05*

## Self-Check: PASSED

All files exist, all commits verified.
