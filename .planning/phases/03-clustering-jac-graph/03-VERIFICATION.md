---
phase: 03-clustering-jac-graph
verified: 2026-04-04T22:00:00Z
status: gaps_found
score: 6/9 must-haves verified
re_verification: false
gaps:
  - truth: "JAC graph is queryable via `jac start` with SpatialObject nodes and SpatialRelation edges present after POST /walker/build_scene"
    status: failed
    reason: "No JAC files exist in the project. No .jac source code, no SpatialObject or SpatialRelation node/edge type definitions, no /walker/build_scene endpoint. The 03-02 plan objective explicitly deferred JAC integration to Phase 4, but ROADMAP Success Criterion #2 for Phase 3 explicitly requires this. The Python server owns a scene graph data structure that COULD populate a JAC graph, but the JAC graph itself does not exist."
    artifacts:
      - path: "jac/spatial_graph.jac"
        issue: "File does not exist. No JAC source directory exists at all."
      - path: "server/routers/walker.py"
        issue: "File does not exist. POST /walker/build_scene is not implemented anywhere."
    missing:
      - "JAC source file defining SpatialObject node type and SpatialRelation edge type"
      - "POST /walker/build_scene endpoint or equivalent JAC walker that reads from build_scene_graph() output and populates JAC graph nodes/edges"
      - "jac start compatibility — JAC runtime integration"

  - truth: "Spatial relationship edges include facing (covariance orientation) and inside (containment) per GRAPH-03"
    status: failed
    reason: "GRAPH-03 in REQUIREMENTS.md defines four edge types: on_top_of, next_to, facing (covariance orientation), and inside (containment). Only on_top_of and next_to are implemented. The 03-01 plan scoped to only on_top_of and next_to, but REQUIREMENTS.md marks GRAPH-03 as complete, which is an overclaim. facing and inside are absent from spatial_relations.py."
    artifacts:
      - path: "server/services/spatial_relations.py"
        issue: "Implements on_top_of and next_to only. No facing or inside edge types. Lines 63-149 define the complete extract_spatial_relations function with no mention of facing or inside."
    missing:
      - "facing edge type derived from Gaussian covariance orientation"
      - "inside edge type derived from containment (3D bounding box overlap)"

  - truth: "REQUIREMENTS.md marks GRAPH-03 as complete — GRAPH-03 full scope is satisfied"
    status: failed
    reason: "REQUIREMENTS.md traceability table marks GRAPH-03 as Phase 3 Complete. However, facing and inside edge types specified in GRAPH-03 are not implemented. The requirement checkbox [x] and traceability status 'Complete' are overclaims."
    artifacts:
      - path: ".planning/REQUIREMENTS.md"
        issue: "Line 30: [x] GRAPH-03 marked complete. Lines 118: traceability shows Complete. Both are incorrect for the facing and inside sub-requirements."
    missing:
      - "Update REQUIREMENTS.md to reflect partial GRAPH-03 implementation, or implement facing and inside"
human_verification: []
---

# Phase 3: Clustering + JAC Graph Verification Report

**Phase Goal:** DBSCAN clusters the top-K Gaussians into named object nodes and JAC graph is populated with SpatialObject nodes connected by SpatialRelation edges derived from 3D geometry
**Verified:** 2026-04-04T22:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                           | Status      | Evidence                                                                                      |
|----|------------------------------------------------------------------------------------------------|-------------|-----------------------------------------------------------------------------------------------|
| 1  | DBSCAN produces distinct object clusters with centroid, bbox, and Gaussian index list          | VERIFIED    | `cluster_gaussians()` in clustering.py lines 59-140; 10 tests pass including 3-cloud test    |
| 2  | Dynamic epsilon derived from k-distance graph (never hardcoded)                                | VERIFIED    | `compute_dynamic_epsilon()` lines 18-56; KDTree + second derivative elbow; median fallback    |
| 3  | Each cluster has centroid, bbox, gaussian_indices, confidence, mean_embedding                  | VERIFIED    | All 5 fields built in cluster dict lines 111-135; test_cluster_dict_keys asserts all          |
| 4  | Spatial relations: on_top_of and next_to derived from 3D coordinates with scene-relative thresholds | VERIFIED | spatial_relations.py lines 84-149; y_threshold=0.3*scale, xz_threshold=0.5*scale            |
| 5  | Edge confidence scores distinguish clear from ambiguous relationships                          | VERIFIED    | on_top_of confidence = 1.0 - bbox_y_overlap; next_to confidence = 1.0 - xz_dist/threshold   |
| 6  | CLIP embedding proximity produces emergent hierarchy (similar_to edges)                        | VERIFIED    | `build_hierarchy()` lines 152-198; cosine similarity above threshold => similar_to edge      |
| 7  | POST /scene/build returns scene graph with nodes, edges, hierarchy, metadata                   | VERIFIED    | scene.py lines 91-135; all 8 endpoint tests pass; caching and force=true work                |
| 8  | JAC graph queryable via jac start with SpatialObject/SpatialRelation types                     | FAILED      | No JAC files exist anywhere in the project. No .jac source, no walker endpoint.              |
| 9  | Spatial relations include facing (covariance) and inside (containment) per GRAPH-03            | FAILED      | Only on_top_of and next_to implemented. facing and inside absent from spatial_relations.py.   |

**Score:** 7/9 truths verified (7 automated pass, 2 failed)

Note: The score is 7/9 for individual truths, but 6/9 for must-haves because Truth #9 is a sub-gap of GRAPH-03 which is the same requirement violation as Gap #2 above. The header shows 6/9 reflecting distinct must-have goals.

### Required Artifacts

| Artifact                                        | Expected                                          | Status      | Details                                                    |
|-------------------------------------------------|---------------------------------------------------|-------------|------------------------------------------------------------|
| `server/services/clustering.py`                 | DBSCAN clustering with dynamic epsilon            | VERIFIED    | 141 lines; exports cluster_gaussians, compute_dynamic_epsilon |
| `server/services/spatial_relations.py`          | Spatial relationship extraction and hierarchy     | PARTIAL     | on_top_of + next_to + similar_to present; facing + inside absent |
| `server/services/graph_builder.py`              | Orchestrates full scene graph pipeline            | VERIFIED    | 121 lines; exports build_scene_graph; all wiring present   |
| `server/routers/scene.py`                       | POST /scene/build endpoint                        | VERIFIED    | 136 lines; Pydantic models, 503 guards, caching            |
| `server/tests/test_clustering.py`               | Unit tests for clustering service                 | VERIFIED    | 10 tests, all pass                                         |
| `server/tests/test_spatial_relations.py`        | Unit tests for spatial relations                  | VERIFIED    | 9 tests, all pass                                          |
| `server/tests/test_graph_builder.py`            | Unit tests for graph builder                      | VERIFIED    | 9 tests, all pass                                          |
| `server/tests/test_scene_endpoint.py`           | Integration tests for /scene/build                | VERIFIED    | 8 tests, all pass                                          |
| `jac/spatial_graph.jac` (or equivalent)         | JAC SpatialObject node / SpatialRelation edge types | MISSING  | No JAC directory or .jac files exist anywhere              |
| `POST /walker/build_scene` (or equivalent)      | Endpoint that populates JAC graph                 | MISSING     | Not implemented; deferred to Phase 4 in plan objective     |

### Key Link Verification

| From                              | To                                    | Via                            | Status  | Details                                                        |
|-----------------------------------|---------------------------------------|--------------------------------|---------|----------------------------------------------------------------|
| `server/services/graph_builder.py` | `server/services/clustering.py`      | calls cluster_gaussians()      | WIRED   | Import line 12; called line 60                                 |
| `server/services/graph_builder.py` | `server/services/spatial_relations.py` | calls extract_spatial_relations + build_hierarchy | WIRED | Import line 14; called lines 65, 68        |
| `server/routers/scene.py`         | `server/services/graph_builder.py`    | calls build_scene_graph()      | WIRED   | Import line 13; called line 123                                |
| `server/main.py`                  | `server/routers/scene.py`             | app.include_router(scene.router) | WIRED | Line 18: `from server.routers import clip, health, scene`; line 159: `app.include_router(scene.router)` |
| `server/services/clustering.py`   | `server/services/gaussian_store.py`  | Uses positions[indices]        | WIRED   | graph_builder.py line 56: `gaussian_store.positions[indices]`  |
| `server/services/graph_builder.py` | `server/services/similarity.py`      | calls top_k()                  | WIRED   | Import line 13; called line 51                                 |
| JAC graph                         | `server/services/graph_builder.py`   | POST /walker/build_scene       | NOT_WIRED | JAC runtime not present; no bridge to Python scene graph      |

### Data-Flow Trace (Level 4)

| Artifact                     | Data Variable   | Source                      | Produces Real Data  | Status      |
|------------------------------|-----------------|-----------------------------|---------------------|-------------|
| `server/routers/scene.py`    | `result`        | `build_scene_graph()` call  | Yes — DBSCAN output | FLOWING     |
| `server/services/graph_builder.py` | `clusters`  | `cluster_gaussians()` on real Gaussian positions | Yes | FLOWING |
| JAC graph (absent)           | SpatialObject nodes | POST /walker/build_scene | N/A                | DISCONNECTED |

### Behavioral Spot-Checks

| Behavior                                      | Command                                                           | Result         | Status  |
|-----------------------------------------------|-------------------------------------------------------------------|----------------|---------|
| /scene/build registered in app routes         | `.venv/bin/python -c "from server.main import app; assert '/scene/build' in [r.path for r in app.routes]"` | exit 0 | PASS |
| 36 Phase 3 tests pass                         | `.venv/bin/python -m pytest server/tests/test_clustering.py server/tests/test_spatial_relations.py server/tests/test_graph_builder.py server/tests/test_scene_endpoint.py` | 36 passed in 2.17s | PASS |
| 60 total tests pass (no regressions)          | `.venv/bin/python -m pytest server/tests/`                        | 60 passed in 5.11s | PASS |
| JAC runtime present and queryable             | `find . -name "*.jac"`                                            | No files found | FAIL   |

### Requirements Coverage

| Requirement | Source Plan     | Description                                                                                      | Status   | Evidence                                                                        |
|-------------|-----------------|--------------------------------------------------------------------------------------------------|----------|---------------------------------------------------------------------------------|
| GRAPH-01    | 03-01, 03-02    | DBSCAN clusters semantically similar Gaussians into object nodes with centroid and spatial extents | SATISFIED | cluster_gaussians() with dynamic epsilon; 10 tests; graph_builder orchestrates into nodes |
| GRAPH-02    | 03-01           | Emergent hierarchy from CLIP embedding proximity — no hardcoded taxonomy                          | SATISFIED | build_hierarchy() with cosine similarity threshold; similar_to edges; 2 tests  |
| GRAPH-03    | 03-01           | Spatial edges: on_top_of, next_to, facing, inside                                                 | PARTIAL  | on_top_of and next_to implemented and tested; facing and inside NOT implemented  |
| GRAPH-04    | 03-01           | Edge confidence scores based on clarity of spatial relationship                                    | SATISFIED | on_top_of: 1.0 - bbox_y_overlap; next_to: 1.0 - dist/threshold; tested         |

**REQUIREMENTS.md Overclaim:** Lines 30 and 118 mark GRAPH-03 as `[x]` complete. This is incorrect — `facing` (covariance orientation) and `inside` (containment) are specified in GRAPH-03 but not implemented.

**No orphaned requirements:** All GRAPH-01 through GRAPH-04 are claimed by 03-01-PLAN.md and 03-02-PLAN.md. No Phase-3-assigned requirements appear in REQUIREMENTS.md that were not picked up by a plan.

### Anti-Patterns Found

| File                                     | Line | Pattern                       | Severity    | Impact                                                                       |
|------------------------------------------|------|-------------------------------|-------------|------------------------------------------------------------------------------|
| `server/routers/scene.py`                | 75   | `label: str` placeholder      | Info        | Node label is intentionally `object_N` by design; Phase 4 adds LLM names. Not a stub — documented in SUMMARY. |
| `.planning/REQUIREMENTS.md`              | 30   | `[x] GRAPH-03` overclaim      | Warning     | GRAPH-03 marked complete but facing/inside not implemented. Misleading for Phase 4 planning. |

No blockers found in the implemented code. The `object_N` label placeholder is documented and intentional. All functions return real computed data, not empty arrays or hardcoded values.

### Human Verification Required

None — all automated checks are sufficient for this phase's artifacts. JAC integration gap is objectively verifiable (no files exist).

### Gaps Summary

**Gap 1 — JAC graph not implemented (blockers phase goal statement):**

The ROADMAP phase goal and Success Criterion #2 both require JAC graph integration — `SpatialObject` nodes, `SpatialRelation` edges, and a `POST /walker/build_scene` endpoint callable from JAC. The 03-02 plan objective explicitly deferred this: "For now, the Python server owns the graph data structure; JAC integration comes in Phase 4." The plan deviated from the ROADMAP goal to de-risk complexity. The Python scene graph payload (`nodes`, `edges`, `hierarchy`) is fully built and ready to be ingested by a JAC walker — the data contract is there — but the JAC layer itself does not exist.

This is not an implementation bug. It is a scope mismatch between what ROADMAP says Phase 3 delivers and what the Phase 3 plans actually scoped. The correct resolution is either: (a) implement JAC graph files in Phase 3 to close this gap, or (b) explicitly update ROADMAP.md to move JAC integration to Phase 4 and adjust Phase 3's goal statement.

**Gap 2 — GRAPH-03 partial: facing and inside edge types missing:**

The `facing` edge type (derived from Gaussian covariance orientation) and `inside` edge type (derived from 3D bounding box containment) are specified in GRAPH-03 and were not scoped into either plan. The plan tasks only specified `on_top_of` and `next_to`. REQUIREMENTS.md marks GRAPH-03 as complete, which is an overclaim. These two edge types are either needed for Phase 3 correctness or should be explicitly deferred and REQUIREMENTS.md updated.

**What works correctly (6 of 9 truths verified):**

- DBSCAN with dynamic epsilon from k-distance elbow: fully implemented, tested, never hardcodes epsilon
- Cluster metadata (centroid, bbox, gaussian_indices, confidence, mean_embedding): complete
- Scene-relative spatial thresholds (on_top_of, next_to): implemented, tested, scene-scale adaptive
- Edge confidence scores: mathematically derived, not arbitrary
- CLIP hierarchy (similar_to edges): implemented and tested
- POST /scene/build endpoint: fully wired, cached, Pydantic-validated, 503-guarded
- Full pipeline test coverage: 60/60 tests pass including 36 new Phase 3 tests

---

_Verified: 2026-04-04T22:00:00Z_
_Verifier: Claude (gsd-verifier)_
