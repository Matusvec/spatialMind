# SpatialMind

## An Autonomous Agent That Sees, Navigates, and Reasons Inside Photorealistic 3D Worlds

---

## The Vision

Imagine pointing your phone at a room, capturing a 30-second video, and within minutes having an AI agent that can navigate that room in photorealistic 3D, identify every object, understand spatial relationships, answer questions like "what's behind the couch?", plan navigation paths, and remember the entire layout across sessions. That's SpatialMind.

SpatialMind is a multi-agent system built on JAC that combines browser-based Gaussian splatting rendering (via Spark.js) with autonomous walker agents that build, traverse, and reason over a persistent spatial knowledge graph. The agent doesn't just see a 3D scene — it *understands* it as a structured graph of objects, rooms, and relationships, and can autonomously explore, answer questions, and navigate through the space.

This is the kind of project that makes judges say "there's no way a freshman built this in 24 hours."

---

## Why This Will Drop Jaws

### The Visual Impact

The moment you load a Gaussian splat scene in the browser, you've already won attention. Gaussian splatting renders photorealistic 3D environments — not game-engine-looking meshes, but actual photographs you can fly through. Most hackathon attendees have never seen this technology running live in a browser tab. Spark.js makes this trivially easy — about 10 lines of code to load and render a splat scene with full camera controls.

Now layer autonomous agent behavior on top of that. The camera smoothly flies to objects the agent identifies. Bounding boxes and labels appear overlaid on the 3D scene. A spatial knowledge graph builds itself in a sidebar panel as walkers traverse and discover the environment. The agent speaks natural language: "I found a bookshelf against the north wall, next to a floor lamp. There are approximately 15 books on the three shelves."

No other project at the hackathon will look like this. Period.

### The Technical Depth

Spatial reasoning is one of the hardest unsolved problems in AI. The best multimodal LLMs achieve roughly 48% accuracy on spatial reasoning benchmarks (SpatialMQA) — compared to 98% for humans. The reason is architectural: LLMs process spatial information as flat token sequences, losing the structural relationships that make spatial reasoning possible.

SpatialMind's core innovation is **externalizing spatial knowledge into a graph**. Instead of stuffing "the lamp is left of the couch which is 3 meters from the wall" into a context window, we model it as JAC nodes and edges — exactly the kind of structured spatial representation that the 3DGraphLLM paper (ICCV 2025) showed dramatically improves spatial VQA performance. JAC's walker paradigm then lets agents traverse this graph to perform structured spatial reasoning — finding paths, computing distances, resolving ambiguous references.

This is publishable-quality research applied at hackathon speed.

### The Product-Market Fit

Y Combinator's Spring 2026 RFS explicitly calls out "Large Spatial Models" as a category they believe could produce "the next AI foundation model on the scale of OpenAI." World Labs raised $1B for spatial intelligence. SpAItial raised $13M for spatial AI agents. Niantic Spatial is building the infrastructure layer. The spatial AI market is on fire, and SpatialMind sits right at the intersection of three hot categories: spatial computing, agentic AI, and 3D reconstruction.

Real applications include: real estate virtual tours with intelligent agents, warehouse inventory management, facilities inspection, accessibility navigation for visually impaired users, AR/VR scene understanding, and robotic pre-planning (simulate navigation before deploying a robot).

---

## Technical Architecture

### The Rendering Layer: Spark.js + Gaussian Splatting

Spark.js is an open-source 3D Gaussian Splatting renderer for Three.js, built by the team at World Labs. It's the state-of-the-art for browser-based splat rendering.

Key capabilities for SpatialMind:
- Loads `.spz`, `.ply`, `.splat`, `.ksplat`, and `.sog` files
- Renders photorealistic scenes at 60fps on standard hardware (WebGL2, no WebGPU required)
- `SplatMesh` extends `THREE.Object3D` — full scene hierarchy integration with standard Three.js
- Supports procedural splats for overlaying annotations (bounding boxes, labels, path markers)
- Works on desktop, mobile, and WebXR (Quest 3, Vision Pro)

Loading a scene is literally this:
```javascript
import * as THREE from "three";
import { SplatMesh } from "@sparkjsdev/spark";

const scene = new THREE.Scene();
const splatURL = "https://example.com/room.spz";
const room = new SplatMesh({ url: splatURL });
scene.add(room);
```

For the hackathon demo, pre-capture 2-3 Gaussian splat scenes before the event. Options:
- **Polycam** (iOS/Android app): Walk around a room, export as `.ply`. Free account for viewing, Pro for export (free trial available).
- **Scaniverse** (iOS): Free, exports `.spz` natively. Best option for quick captures.
- **vid2scene.com**: Free web tool — upload a smartphone video, get a splat back in ~90 minutes.
- **Pre-made scenes**: AR Code offers free `.spz` downloads. The INRIA dataset has classic scenes (bicycle, garden, etc.).

**Important**: Capture your scenes the day before the hackathon. Use Scaniverse on an iPhone to scan your dorm room, a common area, or a building lobby. This isn't pre-building code — it's capturing data, which is allowed. Aim for 2-3 scenes of different complexity.

### The Perception Layer: Gemini Vision + Viewpoint Sampling

The agent needs to "see" what's in the 3D scene. Since we can't run object detection directly on Gaussian splats (they're not images), we use a clever approach: **render viewpoints as 2D frames, then analyze them with vision AI**.

The perception pipeline:
1. Define a set of viewpoints (camera positions) distributed through the scene
2. Render each viewpoint to a 2D image using Three.js's `renderer.render(scene, camera)` + `canvas.toDataURL()`
3. Send each rendered frame to Gemini Flash for object detection with bounding boxes
4. Gemini returns structured JSON: `[{"label": "couch", "box_2d": [y0, x0, y1, x1]}, ...]`
5. Back-project 2D bounding boxes to approximate 3D positions using camera parameters
6. Merge detections across viewpoints (same object seen from different angles)
7. Create JAC nodes and edges from the merged detections

Gemini's bounding box detection is remarkably capable. It returns coordinates normalized to 0-1000 with descriptive labels. The prompt is:

```
"Detect all objects in this image. Return a JSON array where each entry 
has 'label' (descriptive name), 'box_2d' ([ymin, xmin, ymax, xmax] 
normalized 0-1000), and 'spatial_description' (where the object is 
relative to other objects)."
```

Gemini Flash is free-tier with generous rate limits — more than enough for a hackathon demo. Using it via `by llm()` in JAC with LiteLLM's Gemini support means zero API key complexity.

### The JAC Graph Architecture: Where the Magic Happens

This is where SpatialMind transcends "cool demo" and becomes "deeply technical." The spatial knowledge graph is modeled entirely in JAC using Object-Spatial Programming.

#### Node Types

```jac
node Room {
    has name: str;
    has description: str = "";
    has bounds: dict = {};  # {min_x, min_y, min_z, max_x, max_y, max_z}
    has area_sqm: float = 0.0;
    has scanned: bool = False;
}

node SpatialObject {
    has label: str;
    has object_class: str;     # "furniture", "appliance", "decoration", etc.
    has position: dict = {};   # {x, y, z} in scene coordinates
    has dimensions: dict = {}; # {width, height, depth} estimated
    has confidence: float = 0.0;
    has description: str = "";
    has first_seen: str = "";
    has view_count: int = 0;   # how many viewpoints confirmed this object
}

node Viewpoint {
    has camera_position: dict = {};  # {x, y, z}
    has camera_rotation: dict = {};  # {pitch, yaw, roll}
    has frame_ref: str = "";         # base64 or URL of rendered frame
    has analyzed: bool = False;
    has objects_detected: int = 0;
}

node NavigationWaypoint {
    has position: dict = {};
    has is_accessible: bool = True;
    has clearance: float = 0.0;  # meters of free space around this point
}

node SpatialQuery {
    has question: str;
    has answer: str = "";
    has relevant_objects: list = [];
    has reasoning_trace: str = "";
}
```

#### Edge Types

```jac
edge Contains {
    has spatial_region: str = "";  # "center", "north_wall", "corner", etc.
}

edge NearTo {
    has distance_m: float = 0.0;
    has direction: str = "";  # "left_of", "right_of", "above", "below", "behind", "in_front_of"
}

edge VisibleFrom {
    has distance: float = 0.0;
    has occlusion: float = 0.0;  # 0.0 = fully visible, 1.0 = fully occluded
    has view_angle: str = "";    # "front", "side", "top", etc.
}

edge PathTo {
    has distance_m: float = 0.0;
    has passable: bool = True;
    has waypoints: list = [];
}

edge OnTopOf {}
edge InsideOf {}
edge NextTo {}
edge Supports {}  # table Supports lamp
```

#### Walker Agents

**PerceptionWalker** — The eyes of the system. Spawns at Viewpoint nodes, renders the frame, sends it to Gemini for object detection, creates SpatialObject nodes, and wires up spatial relationship edges.

```jac
walker PerceptionWalker {
    has detected_objects: list = [];
    has current_viewpoint: dict = {};

    can perceive_frame(frame_image: str, camera_params: dict) -> list 
        by llm(temperature=0.3);

    can estimate_3d_position(bbox_2d: dict, camera_params: dict) -> dict {
        # Back-project 2D bbox center to approximate 3D position
        # Uses camera intrinsics + depth heuristics
    }

    can merge_with_existing(new_obj: dict, existing_objects: list) -> str
        by llm(temperature=0.1);
        # Returns "new" or the ID of the existing object to merge with

    can start with `root entry {
        visit [-->(`?Viewpoint) where !analyzed];
    }

    can process with Viewpoint entry {
        # 1. Render frame from this viewpoint
        # 2. Send to Gemini for detection
        # 3. For each detection, check if it matches an existing SpatialObject
        # 4. Create new nodes or update existing ones
        # 5. Create spatial relationship edges
        here.analyzed = True;
        here.objects_detected = len(self.detected_objects);
        disengage;
    }
}
```

**NavigationWalker** — The pathfinder. Takes a natural-language destination ("the red armchair"), resolves it to a SpatialObject node via `by llm()`, then traverses PathTo/NearTo edges to compute a waypoint sequence from the current camera position.

```jac
walker NavigationWalker {
    has destination: str;
    has current_position: dict = {};
    has path: list = [];
    has narration: str = "";

    can resolve_destination(query: str, available_objects: list[str]) -> str
        by llm(temperature=0.1);
        # Maps "the red armchair" -> "armchair_003"

    can plan_path(start: dict, end: dict, obstacles: list[dict]) -> list[dict]
        by llm(temperature=0.2);
        # Returns waypoints as [{x, y, z}, ...]

    can narrate_journey(path: list[dict], objects_along_path: list[str]) -> str
        by llm(temperature=0.7);
        # "Heading past the coffee table, turning left at the bookshelf..."
}
```

**QueryWalker** — The brain. Answers natural-language spatial questions by traversing the knowledge graph to gather relevant context, then synthesizing an answer via `by llm()`. This is where structured graph traversal dramatically outperforms flat context window reasoning.

```jac
walker QueryWalker {
    has question: str;
    has gathered_context: list = [];
    has answer: str = "";

    can identify_relevant_objects(question: str, all_objects: list[str]) -> list[str]
        by llm(temperature=0.1);
        # "what's near the window?" -> ["window_001"]

    can gather_spatial_context with SpatialObject entry {
        # Traverse NearTo, Contains, OnTopOf edges from this object
        # Collect all neighboring objects and their relationships
        # Build structured context: "couch_001 is NearTo lamp_003 (0.5m, left_of)"
    }

    can synthesize_answer(question: str, spatial_context: list[str]) -> str
        by llm(temperature=0.5);
        # Uses structured graph context for accurate spatial reasoning
}
```

**ExplorationWalker** — The curiosity engine. Autonomously identifies under-explored regions of the scene (viewpoints not yet analyzed, areas with low object density), generates new viewpoints to investigate, and dispatches PerceptionWalkers. This creates the "autonomous exploration" behavior that makes the demo feel alive.

```jac
walker ExplorationWalker {
    has coverage_map: dict = {};
    has exploration_targets: list = [];

    can identify_gaps(existing_viewpoints: list[dict], 
                      scene_bounds: dict) -> list[dict]
        by llm(temperature=0.3);
        # Returns unexplored camera positions

    can prioritize_targets(gaps: list[dict], 
                          current_objects: list[str]) -> list[dict]
        by llm(temperature=0.2);
        # Ranks exploration targets by expected information gain
}
```

**MemoryWalker** — The long-term memory system. Periodically traverses the full spatial graph, generates high-level summaries ("this is a living room with a couch, two armchairs, a coffee table, and a TV"), detects changes between sessions, and stores persistent memory via Backboard.

```jac
walker MemoryWalker {
    has room_summaries: dict = {};
    has change_log: list = [];

    can summarize_room(room_name: str, objects: list[dict], 
                       relationships: list[dict]) -> str
        by llm(temperature=0.5);

    can detect_changes(previous_summary: str, 
                       current_summary: str) -> list[str]
        by llm(temperature=0.2);
}
```

### Why JAC Is Perfect for This

The spatial environment IS a graph. Rooms are nodes. Objects are nodes. "The lamp is on the table" is an edge. "The couch is near the window" is an edge. This isn't a metaphor — it's literally how spatial scene graphs are defined in the research literature (3DSSG, ScanNet, Replica benchmarks).

JAC's walker paradigm maps perfectly onto spatial agents: an agent exploring a room IS a walker traversing nodes. Pathfinding IS graph traversal. Spatial queries ARE subgraph retrieval. No other programming language makes this correspondence so direct.

The `by llm()` construct eliminates all prompt engineering boilerplate. Each function's name, docstring, and type signature become the prompt automatically. This means you can write 5 agent functions in the time it takes to write 1 with manual prompt construction.

---

## The Demo Script (3 Minutes)

### 0:00–0:15 — The Hook

"What if an AI could walk through a photograph? Not a 3D model — an actual photograph of a real room, rendered in real-time in your browser."

Load the Gaussian splat scene. The audience sees a photorealistic room they can orbit around. Gasps.

### 0:15–0:45 — Autonomous Exploration

"Watch what happens when I tell the agent to explore."

Trigger the ExplorationWalker. The camera automatically moves through the scene, pausing at different viewpoints. As it pauses, object labels and bounding boxes appear overlaid on the 3D scene. The sidebar shows the spatial knowledge graph building in real-time — nodes appearing, edges connecting. A counter ticks up: "12 objects detected... 15... 23..."

### 0:45–1:15 — Spatial Queries

"Now I can ask it anything about the space."

Type: "What's on the coffee table?"
Agent responds: "The coffee table has a stack of books, a remote control, and a small plant. The remote is closest to the edge near the couch."

Type: "What's behind the TV?"
Agent responds: "Behind the TV there's a power strip with 4 cables and a small router. The router's indicator light appears to be on."

Show the reasoning trace in the sidebar — the QueryWalker traversing from the TV node through NearTo and Behind edges, gathering context from connected nodes.

### 1:15–1:45 — Intelligent Navigation

"Now watch it navigate."

Type: "Take me to the bookshelf."

The camera smoothly animates through the scene along a planned path — around the coffee table, past the couch, arriving at the bookshelf. The agent narrates: "Heading past the coffee table on the left, skirting the couch, and arriving at the bookshelf on the north wall. I can see approximately 20 books across three shelves."

Show the planned path as a glowing line through the 3D scene.

### 1:45–2:20 — The Knowledge Graph

Switch to the graph visualization tab. Show the full spatial knowledge graph — rooms as large nodes, objects as smaller nodes, edges labeled with spatial relationships. Point out the density and structure.

"This entire graph was built autonomously by JAC walkers traversing a photorealistic Gaussian splat scene. No manual annotation. No training data. No pre-built models. Just autonomous agents that see, reason, and remember."

### 2:20–2:45 — Persistent Memory

"And here's the part that changes everything: the agent remembers."

Show a "returning visit" — the agent loads the scene and immediately recalls the layout from Backboard memory. "Welcome back. The room looks the same as last time — 23 objects detected, matching my previous scan."

Move an object (if doing a live capture demo) or show a mock: "I notice the plant has been moved from the coffee table to the windowsill."

### 2:45–3:00 — The Close

"45 billion dollars has been invested in spatial AI in the last 18 months. Y Combinator calls Large Spatial Models 'the next foundation model.' SpatialMind is our answer: autonomous agents that understand 3D space as a graph, built in 24 hours on JAC."

---

## Sponsor Integration

### Backboard — Persistent Spatial Memory (Critical Integration)

This is the deepest and most natural sponsor integration. Backboard's persistent memory API stores the entire spatial knowledge graph across sessions. When the agent revisits a scene, it doesn't re-scan from scratch — it loads the previous spatial graph from Backboard and validates/updates it.

Specific integration points:
- Store room summaries, object inventories, and spatial relationships as Backboard memory entries
- Use Backboard's multi-model API to run perception through different vision models and compare results
- Enable cross-session spatial reasoning: "The last time I scanned this room, the couch was against the east wall. It's now against the north wall."

### InsForge — Infrastructure Layer

InsForge provides:
- PostgreSQL storage for scene metadata, captured frames, and detection results
- User authentication for multi-user spatial maps
- S3 storage for Gaussian splat files and rendered viewpoint frames
- Edge functions for background perception processing

### Lovable — Frontend Generation

Use Lovable to scaffold the React frontend:
- Three.js / Spark.js splat viewer with orbit controls
- Chat interface for spatial queries and navigation commands
- Sidebar with real-time knowledge graph visualization (D3.js force graph or vis.js)
- Object detection overlay panel showing bounding boxes and labels
- Navigation path renderer

### Base44 — Onboarding & Sharing

Base44 generates:
- Upload interface for new Gaussian splat scenes
- Shareable links for explored environments
- Demo landing page with pre-loaded example scenes

---

## 24-Hour Build Plan

### Pre-Hackathon (Before April 4)

**Critical**: These are preparation steps, not pre-built code.

- Install Scaniverse on your iPhone. Capture 2-3 rooms: your dorm room, a study lounge, a building lobby. Export as `.spz` files. Transfer to your laptop.
- Create a free Google Cloud API key for Gemini. Confirm Gemini Flash free-tier access.
- Bookmark: Spark.js docs (sparkjs.dev), JAC docs (docs.jaseci.org), Backboard API docs.
- Have a boilerplate HTML file ready that loads Spark.js from CDN (this is like having jQuery bookmarked — it's a library, not project code).
- Bring: laptop, phone (for live capture demo if ambitious), charger, headphones.

### Hour-by-Hour

| Hours | Task | Milestone |
|-------|------|-----------|
| 0–1 | Opening ceremony + JAC workshop | JAC environment set up, `jac` CLI working |
| 1–2 | JAC Intro + Agentic AI workshops | Understanding of `by llm()`, walkers, nodes |
| 2–3 | Attend YC talk (3–3:30 PM) + sketch architecture | Architecture diagram on paper, node/edge types defined |
| 3–5 | **Core JAC graph schema** — define all node types, edge types, walker skeletons in `.jac` files | `spatialmind.jac` compiles and runs with mock data |
| 5–7 | **Spark.js viewer** — load .spz scene, add orbit controls, implement viewpoint sampling (render frames from different camera angles) | Photorealistic scene rendering + frame export working |
| 7–9 | **PerceptionWalker** — send rendered frames to Gemini, parse bounding box JSON, create SpatialObject nodes and spatial edges | Object detection from viewpoints → JAC graph pipeline working |
| 9–11 | **QueryWalker** — implement spatial query answering via graph traversal + `by llm()` synthesis | "What's on the table?" returns accurate answers from graph |
| 11–12 | **Midnight refuel + NavigationWalker** — path planning via graph traversal, camera animation along waypoints | "Take me to the bookshelf" → smooth camera flight |
| 12–14 | **ExplorationWalker** — autonomous viewpoint generation and iterative scene discovery | Agent autonomously explores and builds the graph |
| 14–16 | **Frontend polish** — Lovable-generated React shell, D3.js knowledge graph sidebar, chat interface, bounding box overlays on 3D scene | Polished UI with all panels working |
| 16–18 | **Backboard integration** — persistent memory for spatial graph, cross-session recall | Returning visit demo working |
| 18–20 | **InsForge + Base44 integration** — auth, storage, shareable links | Full sponsor integration complete |
| 20–22 | **End-to-end testing** — run full pipeline on all captured scenes, fix edge cases, handle API failures gracefully | Reliable demo flow from start to finish |
| 22–23 | **Demo prep** — record backup video, write Devpost submission, prepare pitch slides (1 architecture slide) | Submission materials ready |
| 23–24 | **Rehearse pitch** — practice 3-minute demo 3x. Submit to Devpost before 11 AM. | Submitted and rehearsed |

### Risk Mitigation

**Gemini API rate limits**: Cache all detection results locally. If the API goes down during demo, use cached results and show the graph from a previous run.

**Spark.js rendering issues**: Have a pre-recorded screen capture of the scene rendering as backup. The JAC agent logic works independently of the renderer.

**Time overrun on perception pipeline**: The QueryWalker and NavigationWalker can work with manually-created graph data. If perception takes too long to build, manually seed the graph with realistic data and focus on the query/navigation demo.

**Scene file too large for browser**: Use SuperSplat Editor (superspl.at/editor) to crop and optimize splat files before the hackathon. Target 1-3M Gaussians for smooth browser rendering.

---

## Prize Targets

| Prize | How SpatialMind Qualifies | Probability |
|-------|---------------------------|-------------|
| **Agentic AI Track 1st ($750)** | Multi-agent system with reasoning, tool use, multi-step workflows, real-world impact. Four specialized walkers with autonomous exploration, spatial reasoning, and persistent memory. | HIGH |
| **Best JAC Builder ($400)** | The most natural possible use of JAC's paradigm — spatial environments ARE graphs, navigation IS walker traversal, `by llm()` powers every reasoning step. | HIGH |
| **Best Startup Idea / SH1P Interview** | $45B+ invested in spatial AI, YC RFS explicitly calls for Large Spatial Models, World Labs at $1B valuation validates the category. | MEDIUM |
| **Best Use of Backboard (1-Year Sub)** | Persistent spatial memory is the core differentiator — the agent remembers rooms across sessions. This is Backboard's exact use case. | HIGH |
| **Best Use of InsForge (Credits + Merch)** | Auth, storage, and backend via MCP endpoints. | MEDIUM |
| **Best Use of Lovable (3 Months Free)** | Frontend dashboard generated via Lovable. | MEDIUM |
| **Best Use of Base44 (Credits)** | Onboarding flow and demo landing page. | LOW-MEDIUM |
| **Content Prizes ($100 each)** | Gaussian splatting + AI agents = viral content. The visual hook is extraordinary. | HIGH |

**Realistic prize haul: $1,150+ cash + SH1P interview + sponsor prizes + content prize potential.**

---

## Why This Is Legendary

This project operates at the intersection of three of the hottest areas in tech right now: Gaussian splatting (the future of 3D), agentic AI (the future of software), and spatial intelligence (the future of how machines understand the physical world). Each of these individually would make a strong hackathon project. Combining all three — with photorealistic rendering, autonomous exploration, structured spatial reasoning, and persistent memory — creates something that feels like it came from a research lab, not a 24-hour hackathon.

The technical narrative is airtight: "LLMs are bad at spatial reasoning because they process space as flat text. We externalize spatial knowledge into a JAC graph and let walker agents traverse it — turning a 48% accuracy problem into structured graph reasoning. The result is an autonomous agent that navigates photorealistic 3D worlds and answers spatial questions no other system can."

And the demo is literally jaw-dropping. Photorealistic rooms. Autonomous camera flight. Real-time object detection overlays on 3D splats. A growing knowledge graph. Natural-language spatial queries. Persistent memory. No other project at JacHacks will look, feel, or perform like this.

Build this. Win this. Ship this.
