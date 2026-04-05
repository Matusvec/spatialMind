---
phase: 05-frontend-insforge
verified: 2026-04-04T00:00:00Z
status: gaps_found
score: 7/9 must-haves verified
gaps:
  - truth: "Typing a query in chat causes matching Gaussians to visibly change color/opacity in the 3D scene"
    status: failed
    reason: "SparkRenderer fetches from /api/clip/highlight (under the /api prefix), but the actual backend endpoint is /clip/highlight with no /api prefix. The adapter.py does not proxy /api/clip/* routes. The fetch will 404 at runtime."
    artifacts:
      - path: "room-cognito/src/components/SparkRenderer.tsx"
        issue: "Fetches `${API_URL}/api/clip/highlight` but backend route is /clip/highlight (no /api prefix)"
      - path: "server/routers/adapter.py"
        issue: "No /api/clip/highlight or /api/clip/probability proxy route — CLIP endpoints are unreachable via /api prefix"
    missing:
      - "Add @router.post('/clip/highlight') and @router.post('/clip/probability') proxy routes to adapter.py, forwarding to the internal /clip/* endpoints"
      - "OR change SparkRenderer.tsx to call `${API_URL}/clip/highlight` and `${API_URL}/clip/probability` directly"
  - truth: "Heatmap toggle renders probability cloud as continuous opacity gradient across all Gaussians"
    status: failed
    reason: "Same root cause as highlight: SparkRenderer fetches /api/clip/probability which does not exist. The backend serves this at /clip/probability."
    artifacts:
      - path: "room-cognito/src/components/SparkRenderer.tsx"
        issue: "Fetches `${API_URL}/api/clip/probability` but backend route is /clip/probability (no /api prefix)"
      - path: "server/routers/adapter.py"
        issue: "No /api/clip/probability proxy — returns 404 at runtime"
    missing:
      - "Same fix as above — proxy /api/clip/* in adapter.py or align frontend URL to /clip/*"
  - truth: "Scene selector dropdown fetches scene list from /api/scenes and allows switching"
    status: partial
    reason: "/api/scenes IS fetched on mount (Index.tsx line 94) and stored in availableScenes state, but availableScenes is never passed to Viewport3D or rendered. The scene selector cards in Viewport3D.tsx (lines 178-216) still use a hardcoded static array, not the fetched data."
    artifacts:
      - path: "room-cognito/src/pages/Index.tsx"
        issue: "availableScenes state is populated but never passed to Viewport3D or any child component — it is dead state"
      - path: "room-cognito/src/components/Viewport3D.tsx"
        issue: "Scene selector cards use a hardcoded array instead of the availableScenes data from the backend"
    missing:
      - "Pass availableScenes as a prop to Viewport3D and render it in the scene selector instead of the hardcoded array"
human_verification:
  - test: "Load app, click Explore, type a query like 'chairs' in chat"
    expected: "Chair Gaussians in the 3D scene should turn golden yellow after the query is answered"
    why_human: "Requires running backend + frontend together, and a .splat file to verify visual highlight effect"
  - test: "Toggle heatmap button after making a query"
    expected: "Blue-to-red gradient appears across all Gaussians based on CLIP probability scores"
    why_human: "Same — requires live backend with loaded scene"
  - test: "Verify OrbitControls allow pan/zoom/rotate in 3D scene"
    expected: "Mouse drag rotates scene, scroll zooms, right-click pans"
    why_human: "Interactive browser behavior"
---

# Phase 05: Frontend InsForge Verification Report

**Phase Goal:** Browser renders the photorealistic splat scene at interactive framerates, semantic highlighting responds to query results in real-time, graph overlay floats in 3D space, and chat interface is wired to QueryWalker
**Verified:** 2026-04-04T00:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Frontend health check at /api/health returns {status: 'ok'} and sets backendConnected=true | VERIFIED | adapter.py GET /api/health returns status:ok; Index.tsx line 88-92 fetches and sets backendConnected=true |
| 2 | Frontend explore button triggers /api/explore/start and polls /api/explore/status, receiving objects incrementally | VERIFIED | adapter.py has POST /api/explore/start (asyncio.create_task) and GET /api/explore/status; Index.tsx line 198 fetches /api/explore/start with setInterval polling |
| 3 | Frontend chat query sends to /api/query and receives a natural language answer with matched object labels | VERIFIED | adapter.py POST /api/query calls QueryWalker, shapes response with description/nodes/reasoning; Index.tsx line 359 fetches /api/query POST |
| 4 | Spark.js loads .splat file served from /api/scene/splat endpoint at interactive framerates | VERIFIED | adapter.py GET /api/scene/splat uses FileResponse; SparkRenderer.tsx SPLAT_URL = `${API_URL}/api/scene/splat`; OrbitControls and animation loop present; build succeeds (1679 modules) |
| 5 | Typing a query in chat causes matching Gaussians to visibly change color/opacity in the 3D scene | FAILED | SparkRenderer fetches `${API_URL}/api/clip/highlight` (line 196) but backend endpoint is /clip/highlight — no /api prefix. Adapter never proxies /api/clip/*. Runtime 404. |
| 6 | Heatmap toggle renders probability cloud as continuous opacity gradient across all Gaussians | FAILED | Same root cause: SparkRenderer fetches `/api/clip/probability` (line 260) but backend exposes /clip/probability only. |
| 7 | Object nodes render as labeled spheres at real 3D centroid positions | VERIFIED | GraphOverlay3D.tsx uses SphereGeometry(0.05,8,8) + CanvasTexture labels at obj.position; wired into Viewport3D with threeScene prop |
| 8 | Spatial edges render as lines connecting nodes in 3D space | VERIFIED | GraphOverlay3D.tsx uses THREE.Line + LineBasicMaterial(0x6366f1, opacity:0.4) per graphEdges |
| 9 | Scene selector dropdown fetches scene list from /api/scenes and allows switching | PARTIAL | /api/scenes fetched in Index.tsx line 94, stored in availableScenes. But availableScenes is never passed to Viewport3D — scene selector cards (Viewport3D.tsx lines 178-216) still use hardcoded static array. |

**Score:** 7/9 truths verified (5 full, 2 failed, 1 partial)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `server/routers/adapter.py` | Thin /api/* adapter routes | VERIFIED | 7 endpoints present: /api/health, /api/explore/start, /api/explore/status, /api/query, /api/scenes, /api/scene/splat, /api/memory/{session_id}. Substantive (330 lines). Mounted in main.py line 178. |
| `server/main.py` | Mounts adapter router | VERIFIED | Line 18 imports adapter; line 178 `app.include_router(adapter.router)` |
| `room-cognito/src/config.ts` | API URL pointing to backend port 8001 | VERIFIED | `export const API_URL = "http://localhost:8001"` |
| `room-cognito/src/components/SparkRenderer.tsx` | Spark.js renderer with highlight and heatmap | PARTIAL | Contains highlight + heatmap + dyno + setColor + OrbitControls + resetAllColors + /api/scene/splat. BUT the CLIP endpoint URLs are wrong (/api/clip/* vs /clip/*) — highlight and heatmap will 404 at runtime. |
| `room-cognito/src/components/Viewport3D.tsx` | Updated viewport with lastQuery, highlight, heatmap, graph overlay | VERIFIED | Contains lastQuery, highlightQuery, heatmapQuery, heatmapEnabled, onSceneReady, GraphOverlay3D integration, graph overlay toggle |
| `room-cognito/src/components/GraphOverlay3D.tsx` | Three.js 3D graph overlay | VERIFIED | Exports GraphOverlay3D; SphereGeometry, LineBasicMaterial, CanvasTexture, label sprites, edge lines, highlight support. Returns null from render (Three.js injection pattern). |
| `room-cognito/src/pages/Index.tsx` | lastQuery state, /api/scenes fetch, INS-03 stub | PARTIAL | lastQuery state exists (line 68); setLastQuery on chat send (line 321); /api/scenes fetched (line 94); INS-03 comment stub (line 14). But availableScenes never reaches the scene selector UI. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| room-cognito/src/pages/Index.tsx | /api/health | fetch in useEffect | WIRED | Line 88: `fetch(\`${API_URL}/api/health\`)`, response sets backendConnected=true |
| room-cognito/src/pages/Index.tsx | /api/explore/start, /api/explore/status | fetch + setInterval polling | WIRED | Lines 198, 204: start POST + status GET with setInterval |
| room-cognito/src/pages/Index.tsx | /api/query | fetch POST on chat send | WIRED | Lines 359-363: POST with query + session_id |
| room-cognito/src/components/SparkRenderer.tsx | /api/clip/highlight | fetch on highlightQuery prop change | NOT_WIRED | Frontend calls `/api/clip/highlight` (line 196) but backend only exposes `/clip/highlight` (no /api prefix). No adapter proxy exists. |
| room-cognito/src/components/SparkRenderer.tsx | /api/clip/probability | fetch on probabilityQuery prop change | NOT_WIRED | Frontend calls `/api/clip/probability` (line 260) but backend only exposes `/clip/probability`. |
| room-cognito/src/components/Viewport3D.tsx | SparkRenderer | highlightQuery and heatmapQuery props | WIRED | Lines 134-136: highlightQuery, heatmapQuery, heatmapEnabled all passed |
| room-cognito/src/components/Viewport3D.tsx | GraphOverlay3D | scene, objects, graphNodes, graphEdges props | WIRED | Lines 141-148: all props passed including threeScene from onSceneReady |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| SparkRenderer.tsx | mask (highlight) | POST /api/clip/highlight | NO — endpoint does not exist at /api prefix | DISCONNECTED |
| SparkRenderer.tsx | probabilities (heatmap) | POST /api/clip/probability | NO — endpoint does not exist at /api prefix | DISCONNECTED |
| SparkRenderer.tsx | splat (3D scene) | GET /api/scene/splat | YES — adapter proxies FileResponse from artifacts dir | FLOWING |
| GraphOverlay3D.tsx | objects + graphEdges | Passed from Viewport3D -> Index.tsx state | YES — populated from /api/explore/status response (real walker data when backend connected) | FLOWING |
| Index.tsx | availableScenes | GET /api/scenes (line 94) | YES — fetched and populated | HOLLOW_PROP — fetched but never rendered |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Frontend build succeeds without errors | `cd room-cognito && pnpm build` | `✓ 1679 modules transformed, built in 2.31s` | PASS |
| Adapter router has all 7 /api/* routes | grep pattern in adapter.py | /api/health, /api/explore/start, /api/explore/status, /api/query, /api/scenes, /api/scene/splat, /api/memory/{session_id} all present | PASS |
| Adapter mounted in main.py | grep adapter in main.py | Line 18 import, line 178 include_router | PASS |
| CLIP highlight reachable via /api prefix | grep api/clip in adapter.py | No results — /api/clip/* routes absent from adapter | FAIL |
| availableScenes wired to scene selector | grep availableScenes in Index.tsx | Only on line 70 (state declaration) — never passed as prop or rendered | FAIL |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| UI-01 | 05-01, 05-02 | Spark.js renders photorealistic Gaussian splat at interactive framerates | SATISFIED | SparkRenderer.tsx loads from /api/scene/splat with animation loop; build succeeds |
| UI-02 | 05-01 | Chat interface accepts natural language queries and displays agent responses | SATISFIED | handleChatSend in Index.tsx POSTs to /api/query, populates chatMessages |
| UI-03 | 05-02 | Semantic highlighting — matching Gaussians change color/opacity based on query | BLOCKED | SparkRenderer fetches /api/clip/highlight but that route returns 404 — highlights will never apply |
| UI-04 | 05-03 | 3D graph overlay — nodes and edges rendered at 3D positions | SATISFIED | GraphOverlay3D.tsx renders SphereGeometry+Lines, integrated in Viewport3D |
| UI-05 | 05-02 | Probability cloud heatmap | BLOCKED | SparkRenderer fetches /api/clip/probability which returns 404 — heatmap will never render |
| INS-01 | 05-01 | S3-compatible storage hosts PLY/npy/weights; splat served from backend | SATISFIED | /api/scene/splat FileResponse from artifacts dir; INS-01 is a hackathon stub per plan |
| INS-02 | 05-01 | PostgreSQL stores scene metadata; /api/scenes returns scene list | SATISFIED | /api/scenes returns hardcoded list — intentional hackathon stub per plan |
| INS-03 | 05-03 | User auth via InsForge; bypassed for hackathon | SATISFIED | Comment stub at Index.tsx line 14; no login wall; documented upgrade path |
| INS-04 | 05-03 | Frontend fetches scene list from backend | PARTIAL | /api/scenes is fetched (Index.tsx line 94) and stored in availableScenes, but that state is never passed to the scene selector UI — the visual selector still shows hardcoded cards |

**Orphaned requirements check:** All 9 IDs (UI-01 through UI-05, INS-01 through INS-04) are accounted for across the three plans. No orphaned requirements found.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| room-cognito/src/components/SparkRenderer.tsx | 196 | `fetch(\`${API_URL}/api/clip/highlight\`)` — /api prefix mismatch | Blocker | Semantic highlighting (UI-03) never fires; runtime 404 |
| room-cognito/src/components/SparkRenderer.tsx | 260 | `fetch(\`${API_URL}/api/clip/probability\`)` — /api prefix mismatch | Blocker | Heatmap visualization (UI-05) never fires; runtime 404 |
| room-cognito/src/pages/Index.tsx | 70 | `availableScenes` state never passed to any child — dead state | Warning | INS-04 partially unmet; scene selector stays hardcoded |

---

### Human Verification Required

#### 1. Semantic Highlight Visual Effect (blocked pending gap fix)

**Test:** With backend running and a .splat file loaded, type "chairs" in the chat and wait for response
**Expected:** Chair Gaussians in the 3D view turn golden yellow within ~1 second of the answer appearing
**Why human:** Requires live backend + real scene file + visual inspection

#### 2. Probability Heatmap Visual Effect (blocked pending gap fix)

**Test:** After a query is answered, toggle the Heatmap button
**Expected:** All Gaussians shift to a blue-to-red color gradient reflecting CLIP probability scores
**Why human:** Same as above — visual, requires live backend

#### 3. OrbitControls Interactive Navigation

**Test:** With scene loaded, drag mouse to rotate, scroll to zoom, right-click drag to pan
**Expected:** Smooth camera movement with damping, target at (0, 0.5, 0)
**Why human:** Browser interaction, can't verify programmatically

#### 4. 3D Graph Overlay at Real Positions

**Test:** After exploration completes with a real backend, toggle the Graph button on/off
**Expected:** Labeled indigo spheres appear at spatial positions matching scanned objects; lines connect related nodes
**Why human:** Requires real backend with scene graph data to populate positions

---

### Gaps Summary

Two of the three plans executed cleanly and all their artifacts are wired. The primary gap is a URL prefix mismatch introduced in plan 05-02: SparkRenderer was written to call `/api/clip/highlight` and `/api/clip/probability`, but the clip router mounts at `/clip/*` with no `/api` prefix and the adapter was never extended to proxy those routes. This blocks both UI-03 (semantic highlighting) and UI-05 (heatmap), which are the visually distinctive features of the demo.

A secondary gap is that `availableScenes` state in Index.tsx is populated from `/api/scenes` but is never connected to the visual scene selector cards in Viewport3D.tsx, leaving INS-04 only partially satisfied.

**Fix is simple:** Either add two proxy routes to adapter.py (`/api/clip/highlight` and `/api/clip/probability`) or update SparkRenderer.tsx to drop the `/api` prefix for those two calls. Either approach is ~10 lines.

---

_Verified: 2026-04-04T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
