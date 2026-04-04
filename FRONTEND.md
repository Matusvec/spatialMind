# SpatialMind — Frontend Build Guide

**For:** Frontend partner
**Stack:** Vite + React + TypeScript + Three.js + Spark.js
**Your job:** Make this look like it came from a funded startup, not a hackathon.

---

## What You're Building

A browser app that renders a photorealistic 3D room (Gaussian splat), lets users ask natural language questions about the space, and visually shows the AI "understanding" the room — objects light up, a knowledge graph floats in 3D space, and a continuous probability heatmap pulses across the scene.

The backend handles all the AI. You handle what judges see.

---

## App Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  ┌─ HEADER BAR ───────────────────────────────────────────────┐  │
│  │  SpatialMind logo    scene name    connection status        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─ 3D VIEWPORT (80% width) ──────┐  ┌─ SIDEBAR (20%) ────────┐ │
│  │                                 │  │                         │ │
│  │   Spark.js Gaussian Splat       │  │  KNOWLEDGE GRAPH        │ │
│  │   Scene Rendering               │  │  (force-directed mini)  │ │
│  │                                 │  │                         │ │
│  │   [Graph overlay nodes          │  │  ┌─────────────────┐   │ │
│  │    float in 3D space]           │  │  │  ◉ chair         │   │ │
│  │                                 │  │  │  ├── next_to ──▶ │   │ │
│  │   [Probability cloud heatmap    │  │  │  ◉ table         │   │ │
│  │    glows across Gaussians]      │  │  │  ├── on_top_of ▶ │   │ │
│  │                                 │  │  │  ◉ lamp          │   │ │
│  │                                 │  │  └─────────────────┘   │ │
│  │                                 │  │                         │ │
│  │                                 │  │  OBJECT DETAILS         │ │
│  │                                 │  │  (click a node)         │ │
│  │                                 │  │  Label: desk chair      │ │
│  │                                 │  │  Hierarchy:             │ │
│  │                                 │  │   furniture > seating   │ │
│  │                                 │  │   > chair > desk chair  │ │
│  │                                 │  │  Confidence: 0.94       │ │
│  │                                 │  │  Relations: 3           │ │
│  └─────────────────────────────────┘  └─────────────────────────┘ │
│                                                                    │
│  ┌─ CHAT BAR (bottom, full width) ────────────────────────────┐  │
│  │  💬 Ask about this space...                        [Send]  │  │
│  │                                                             │  │
│  │  ┌─ RESPONSE AREA (expandable) ─────────────────────────┐  │  │
│  │  │ "The lamp is on the side table, next to the armchair  │  │  │
│  │  │  against the north wall. There are 3 books on the     │  │  │
│  │  │  table underneath it."                                 │  │  │
│  │  └───────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Design System

### Color Palette

```
Background:        #0a0a0f (near-black, lets the splat scene pop)
Surface:           #14141f (panels, sidebar, chat)
Surface Elevated:  #1e1e2e (hover states, active panels)
Border:            #2a2a3a (subtle dividers)

Primary Accent:    #6366f1 (indigo — used for active states, buttons)
Highlight Glow:    #fbbf24 (amber — semantic highlight on Gaussians)
Probability Low:   #3b82f6 (blue — low confidence in heatmap)
Probability High:  #ef4444 (red — high confidence in heatmap)

Text Primary:      #f1f5f9 (off-white)
Text Secondary:    #94a3b8 (muted slate)
Text Muted:        #64748b (very muted)

Success:           #22c55e (connections, verified)
Warning:           #f59e0b (loading, processing)
Error:             #ef4444

Graph Node:        #8b5cf6 (purple spheres)
Graph Edge:        #6366f1 at 40% opacity (indigo lines)
Graph Node Active: #fbbf24 (amber — when matched by query)
```

### Typography

```
Font:              "Inter", system-ui, sans-serif
Mono (code/data):  "JetBrains Mono", monospace

Header Bar:        14px semibold, tracking wide
Sidebar Headers:   12px bold uppercase, text-secondary
Node Labels:       11px medium, text-primary
Chat Input:        15px regular
Chat Response:     14px regular, line-height 1.6
Object Details:    13px regular
Confidence Score:  13px mono, tabular-nums
```

### Motion

```
Highlight fade-in:     300ms ease-out (Gaussians glow on)
Highlight fade-out:    500ms ease-in (slower fade — feels organic)
Graph node appear:     400ms spring (slight overshoot)
Graph edge draw:       600ms linear (line draws from source to target)
Camera smooth:         2000ms ease-in-out (orbit/fly transitions)
Chat response:         Token-by-token streaming, 30ms per token
Panel slide:           200ms ease-out
Probability pulse:     2000ms ease-in-out infinite (subtle breathing)
```

---

## Component Breakdown

### 1. Spark.js 3D Viewport

The hero component. This is what judges see first.

**Setup:**

```typescript
import * as THREE from "three";
import { SplatMesh } from "@sparkjsdev/spark";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);

const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });

// Load Gaussian splat
const splatMesh = new SplatMesh({ url: SPLAT_URL });
scene.add(splatMesh);

// Orbit controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.rotateSpeed = 0.5;
```

**File format:** The backend will serve either `.ply` or `.spz` files. Spark.js handles both. The URL comes from InsForge storage or a direct backend endpoint.

**Camera defaults:** Start pulled back enough to see the whole room. Smooth auto-rotate on idle (stop on interaction).

---

### 2. Semantic Highlighting (Dyno Shader)

When a query matches Gaussians, the backend sends a highlight mask — an array of Gaussian indices + confidence scores. You apply this as a color/opacity tint on those specific Gaussians.

**Spark.js Dyno system** allows per-Gaussian GPU manipulation. Research says to check `sparkjs.dev/docs/dyno-overview/` for the exact API. The pattern:

```typescript
// Conceptual — verify exact Dyno API from Spark.js docs
// The goal: given a list of Gaussian indices and scores,
// tint those Gaussians with the highlight color (amber #fbbf24)
// and scale opacity by the score.

function applyHighlightMask(
  splatMesh: SplatMesh,
  mask: { indices: number[]; scores: number[] }
) {
  // Option A: Dyno shader graph (preferred — GPU-side)
  // Build a Dyno that reads a uniform buffer of highlighted indices
  // and outputs modified RGBA for matching Gaussians.

  // Option B: splatRgba modifier (confirmed in Spark.js docs)
  // SplatMesh has a per-splat RGBA buffer you can write to.
  // Less performant but guaranteed to work.

  // The highlight color
  const highlightR = 0xfb / 255; // amber
  const highlightG = 0xbf / 255;
  const highlightB = 0x24 / 255;

  // Reset all Gaussians to default first
  // Then for each index in mask, blend toward highlight color
  // with alpha = score
}
```

**Important:** The backend returns `gaussian_indices` (integers) and `scores` (0-1 floats). You need to map these to the loaded SplatMesh's internal Gaussian ordering. The indices are 0-based into the PLY file's point array.

**Visual effect:** Matched Gaussians should glow amber. Fade in over 300ms. Unmatched Gaussians stay normal (don't dim them — the photorealistic scene should always look good).

---

### 3. Probability Cloud Heatmap

This is the wow-factor visual. Instead of binary "match/no match," every Gaussian gets a soft score. Render this as a continuous color gradient across the entire scene.

**When active:** The scene goes from photorealistic to having a thermal-vision-like overlay where semantically relevant areas glow warm (red/amber) and irrelevant areas stay cool (blue/transparent).

```
Score 0.0 → fully transparent (scene looks normal)
Score 0.1 → very faint blue tint
Score 0.3 → blue
Score 0.5 → purple (midpoint)
Score 0.7 → amber
Score 0.9 → bright red/amber
Score 1.0 → intense amber glow
```

**Implementation:** Same Dyno/splatRgba mechanism as highlighting, but applied to ALL Gaussians with their soft scores. The backend endpoint for this is different from the top-K highlight — it returns scores for every Gaussian.

**Toggle:** User can switch between "highlight mode" (top-K only, amber glow) and "heatmap mode" (all Gaussians, color gradient). Default to highlight mode. Heatmap is the demo wow-moment.

**Subtle animation:** The heatmap should have a very slow breathing pulse — scores oscillate ±5% over 2 seconds. Makes it feel alive, not static.

---

### 4. 3D Graph Overlay

Object nodes and spatial edges rendered as 3D geometry floating in the scene, overlaid on the Gaussian splat.

**Nodes:** Small spheres (Three.js `SphereGeometry`) at each object's centroid position. Purple (`#8b5cf6`). Semi-transparent. Label text floating above each sphere using CSS2DRenderer or Three.js sprite text.

```typescript
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

function createGraphNode(
  label: string,
  position: [number, number, number],
  isActive: boolean
): THREE.Group {
  const group = new THREE.Group();

  // Sphere
  const geo = new THREE.SphereGeometry(0.05, 16, 16);
  const mat = new THREE.MeshBasicMaterial({
    color: isActive ? 0xfbbf24 : 0x8b5cf6,
    transparent: true,
    opacity: isActive ? 0.9 : 0.6,
  });
  const sphere = new THREE.Mesh(geo, mat);
  group.add(sphere);

  // Label
  const div = document.createElement("div");
  div.className = "graph-node-label";
  div.textContent = label;
  const cssObj = new CSS2DObject(div);
  cssObj.position.set(0, 0.08, 0);
  group.add(cssObj);

  group.position.set(...position);
  return group;
}
```

**Edges:** Lines between connected nodes. Use `THREE.Line` with `LineDashedMaterial` or `LineBasicMaterial`. Color: indigo at 40% opacity. Draw animation on first render (600ms, line extends from source to target).

**Edge labels:** Show relation type ("on_top_of", "next_to") as small CSS text at edge midpoint. Only show on hover or when the connected nodes are active.

**Interaction:** Click a graph node → highlight its Gaussians, show details in sidebar, zoom camera toward it.

**Toggle:** Checkbox to show/hide the graph overlay. Default: visible. When hidden, the scene is pure photorealistic splat.

---

### 5. Sidebar — Knowledge Graph Mini View

A 2D force-directed graph showing the same nodes and edges as the 3D overlay, but in a classic network visualization. This gives judges a second way to understand the structure.

**Library:** Use `d3-force` or just CSS/SVG manual layout. Keep it simple — this is supplementary.

```
Nodes: circles, colored by type (purple default, amber when active)
Edges: lines with labels
Layout: force-directed, auto-stabilizing
Interaction: click node → highlights in 3D viewport too
```

**Object Details Panel (below graph):**

When a node is selected (clicked in either 3D or sidebar), show:

```
┌─────────────────────────┐
│ desk chair               │
│ ─────────────────────── │
│ Hierarchy                │
│  furniture > seating     │
│  > chair > desk chair    │
│                          │
│ Confidence  0.94         │
│ Gaussians   1,247        │
│ Position    (2.1, 0.5,   │
│              -0.8)       │
│                          │
│ Relationships            │
│  next_to → desk (0.6m)  │
│  on_top_of → rug         │
│  facing → monitor        │
└─────────────────────────┘
```

---

### 6. Chat Interface

Bottom bar. Full width. Clean and minimal.

**Input:** Single-line text input with send button. Placeholder: "Ask about this space..."

**Response area:** Expandable panel above the input. Shows the AI's response with streaming text (token-by-token, 30ms delay for dramatic effect). Supports markdown-light formatting (bold for object names).

**Response structure from backend:**

```json
{
  "description": "The lamp is on the side table, next to the armchair.",
  "nodes": [
    { "id": "obj_004", "label": "lamp", "centroid": [2.1, 1.2, 0.3] },
    { "id": "obj_007", "label": "side table", "centroid": [1.9, 0.6, 0.3] }
  ],
  "highlight_mask": {
    "indices": [4096, 4097, 4098, ...],
    "scores": [0.92, 0.88, 0.85, ...]
  }
}
```

On response:
1. Stream the `description` text into the response area
2. Simultaneously apply `highlight_mask` to the 3D viewport (amber glow fades in)
3. Activate matched nodes in the graph overlay (turn amber)
4. If sidebar node is clicked, smooth-fly camera to that object

**Query history:** Keep last 5 queries visible as collapsed chips above the input. Clicking replays the query.

**Loading state:** While waiting for response, show a subtle pulsing ring around the chat input. Don't block the 3D viewport — user can still orbit.

---

### 7. Header Bar

Minimal. Left-aligned logo, center scene name, right status indicators.

```
┌──────────────────────────────────────────────────────────────┐
│  ◆ SpatialMind          "JacHacks Venue"         ● Connected │
└──────────────────────────────────────────────────────────────┘
```

- **Logo:** "SpatialMind" in semibold Inter, with a small purple `◆` diamond icon
- **Scene name:** Whatever the current scene is called
- **Status:** Green dot = backend connected. Yellow = loading. Red = disconnected.

---

## API Contract

All calls go to the backend server. The JAC server and Python server are behind one API surface (or two ports — see architecture notes).

### Scene Load

```
GET /api/scenes
→ [{ id: "venue", name: "JacHacks Venue", ply_url: "https://...", status: "ready" }]

GET /api/scenes/:id/graph
→ {
    nodes: [
      { id: "obj_001", label: "chair", centroid: [1.2, 0.5, -0.8], bbox: [...],
        gaussian_count: 1247, confidence: 0.94, description: "..." }
    ],
    edges: [
      { from: "obj_001", to: "obj_002", relation: "next_to", confidence: 0.87, distance: 0.6 }
    ]
  }
```

### Spatial Query

```
POST /api/query
Body: { "query": "where is the lamp?", "session_id": "abc123" }
→ {
    description: "The lamp is on the side table next to the armchair.",
    nodes: [{ id, label, centroid }],
    highlight_mask: { indices: number[], scores: number[] }
  }
```

### Probability Cloud

```
POST /api/query/heatmap
Body: { "query": "furniture" }
→ {
    scores: Float32Array  // one score per Gaussian, length = total Gaussians
  }
```

### Exploration Status

```
GET /api/explore/status
→ { status: "exploring" | "complete", objects_found: 23, progress: 0.85 }
```

### Health Check

```
GET /api/health
→ { status: "ok", gaussians: 48231, objects: 23, graph_ready: true }
```

---

## InsForge Integration

### Auth

```typescript
import { createClient } from "@insforge/sdk";

const insforge = createClient({
  url: import.meta.env.VITE_INSFORGE_URL,
  key: import.meta.env.VITE_INSFORGE_ANON_KEY,
});

// Sign in
const { data: { user } } = await insforge.auth.signInWithPassword({
  email,
  password,
});

// Get session token for API calls
const { data: { session } } = await insforge.auth.getSession();
```

### Scene File Loading

```typescript
// Get PLY URL from InsForge storage
const { data } = insforge.storage
  .from("scenes")
  .getPublicUrl("venue/scene.ply");

// Pass to Spark.js
const splatMesh = new SplatMesh({ url: data.publicUrl });
```

### Scene Metadata

```typescript
// Fetch scenes from InsForge Postgres
const { data: scenes } = await insforge
  .from("scenes")
  .select("*")
  .eq("status", "ready");
```

---

## The Three-Beat Demo Sequence

This is what wins. Practice this flow.

### Beat 1: "Show me the chairs" (Object Detection)

User types "chairs" → chair Gaussians glow amber → graph nodes for chairs activate → response: "I found 4 chairs in the space..."

**What judges see:** Photorealistic room + specific objects light up instantly. Gasps.

### Beat 2: "What's on the tables?" (Spatial Reasoning)

User types "what's on the tables?" → table Gaussians glow + objects on top glow differently → graph shows on_top_of edges → response: "The main table has a laptop, two notebooks, and a water bottle. The side table has a lamp and a phone charger."

**What judges see:** The AI doesn't just find objects — it understands spatial relationships. "On top of" is a graph relationship, not a language hack.

### Beat 3: Unrehearsed Query (Proof of Understanding)

Let a judge type something. "Is there anything behind the couch?" or "What's the biggest object in the room?" or "Describe the area near the door."

**What judges see:** This isn't scripted. The system actually understands the space.

### Bonus: Probability Cloud Toggle

After Beat 2, toggle heatmap mode. Type "furniture" → the entire scene shifts to a thermal heatmap where all furniture glows warm and walls/floor stay cool. This is the "holy shit" moment.

---

## Exploration Discovery Animation

On scene load, trigger the ExplorationWalker. While it runs:

1. Show a subtle "Discovering space..." indicator in the header
2. As each object is discovered, its graph node appears with a spring animation
3. Edges draw themselves between nodes as relationships are computed
4. Counter in sidebar: "Objects found: 12... 15... 23 ✓"
5. When complete: brief flash of the full graph overlay, then settle

This makes the first 10 seconds of the demo feel alive — the AI is actively thinking, not waiting for input.

---

## Tech Stack

```
Framework:     Vite + React 19 + TypeScript
3D Engine:     Three.js 0.178+ (required by Spark.js)
Splat Render:  @sparkjsdev/spark (SplatMesh, Dyno)
State:         Zustand (lightweight, no boilerplate)
Styling:       Tailwind CSS
Graph (2D):    d3-force (sidebar mini graph)
Graph (3D):    Three.js geometry + CSS2DRenderer
HTTP:          Native fetch (no axios needed)
Auth:          @insforge/sdk
Build:         Vite, deploy to Vercel or serve static
```

### Install

```bash
pnpm create vite spatial-mind-ui --template react-ts
cd spatial-mind-ui
pnpm add three @sparkjsdev/spark zustand d3-force @insforge/sdk
pnpm add -D tailwindcss @tailwindcss/vite @types/three @types/d3-force
```

---

## Environment Variables

```env
VITE_API_URL=http://localhost:8001
VITE_JAC_URL=http://localhost:8000
VITE_INSFORGE_URL=https://your-project.insforge.dev
VITE_INSFORGE_ANON_KEY=your-anon-key
VITE_DEFAULT_SCENE=venue
```

---

## File Structure

```
src/
├── main.tsx
├── App.tsx                    # Layout shell
├── stores/
│   ├── sceneStore.ts          # Zustand: scene state, loaded Gaussians
│   ├── queryStore.ts          # Zustand: query state, highlight mask
│   └── graphStore.ts          # Zustand: graph nodes/edges
├── components/
│   ├── Viewport.tsx           # Spark.js 3D scene + orbit controls
│   ├── HighlightController.tsx # Applies highlight masks to Gaussians
│   ├── GraphOverlay3D.tsx     # Three.js nodes + edges in 3D
│   ├── ProbabilityCloud.tsx   # Heatmap mode rendering
│   ├── Sidebar.tsx            # Knowledge graph panel
│   │   ├── GraphMini.tsx      # d3-force 2D graph
│   │   └── ObjectDetails.tsx  # Selected node info
│   ├── ChatBar.tsx            # Query input + response display
│   ├── HeaderBar.tsx          # Logo, scene name, status
│   └── ExploreIndicator.tsx   # "Discovering space..." animation
├── hooks/
│   ├── useSplatScene.ts       # Load + manage SplatMesh
│   ├── useQuery.ts            # POST /api/query + handle response
│   ├── useGraph.ts            # Fetch + manage graph data
│   └── useExplore.ts          # Poll exploration status
├── api/
│   ├── client.ts              # Base fetch wrapper
│   ├── scenes.ts              # Scene CRUD
│   ├── queries.ts             # Spatial queries
│   └── insforge.ts            # InsForge client init
├── utils/
│   ├── colorMap.ts            # Score → heatmap color conversion
│   └── projection.ts          # 3D → 2D projection for labels
└── styles/
    └── globals.css            # Tailwind + custom properties
```

---

## Performance Targets

| Metric | Target | Why |
|--------|--------|-----|
| Scene load | < 5s | Judges won't wait |
| Query response | < 3s (including highlight) | Must feel interactive |
| Highlight render | < 300ms after data arrives | Visual feedback must be instant |
| Frame rate | 30fps minimum during orbit | Stuttery 3D = broken demo |
| Memory | < 1.5GB browser tab | Chrome will kill tabs above 2GB |

### If PLY is too large for the browser:

Convert to `.spz` format (Spark.js native, much smaller). Use the SuperSplat editor at `superspl.at/editor` to crop and optimize. Target 1-3M Gaussians for smooth browser rendering.

---

## What NOT to Build

- **Auth UI / signup flow** — hardcode a demo user, skip login screens entirely
- **Scene upload interface** — scenes are pre-loaded, no upload needed
- **Settings/preferences** — zero config, everything works out of the box
- **Loading screens with spinners** — use skeleton states and progressive loading instead
- **Responsive mobile layout** — desktop Chrome only, full screen for demo
- **Error pages** — if something breaks, show a toast, don't navigate away
- **Animations for animation's sake** — every motion should communicate state change

---

## Demo Day Checklist

- [ ] Scene renders in Chrome full-screen without lag
- [ ] Chat input is focused and ready to type on load
- [ ] "chairs" query highlights correct Gaussians in < 5s
- [ ] "what's on the tables?" returns correct spatial answer
- [ ] Graph overlay visible and correct
- [ ] Probability heatmap toggle works and looks stunning
- [ ] Exploration animation plays on scene load
- [ ] Fallback scene loads if primary fails
- [ ] No console errors visible (open devtools before demo, close them)
- [ ] Browser zoom at 100% (not accidentally zoomed in/out)
