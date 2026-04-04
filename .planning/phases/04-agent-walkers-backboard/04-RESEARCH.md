# Phase 4: Agent Walkers + Backboard - Research

**Researched:** 2026-04-04
**Domain:** JAC walker agents, Backboard persistent memory, LLM integration via `by llm()`
**Confidence:** MEDIUM (JAC walker patterns HIGH via official docs; `by llm()` configuration MEDIUM via jac-byllm README; Backboard Python SDK MEDIUM via cookbook + quickstart; hybrid JAC/Python architecture MEDIUM -- documented but novel for this project)

## Summary

Phase 4 bridges the Phase 3 Python scene graph (a cached dict of nodes/edges in `app_state`) with JAC walker agents that traverse it and Backboard for persistent conversational memory. The critical architectural decision is **hybrid mode**: define the spatial graph schema and walker logic in `.jac` files (satisfying the "Best JAC Builder" prize criteria of "deep leverage of `by llm()`, walkers, graph-native data modeling"), while delegating heavy CLIP compute to the existing Python FastAPI server via HTTP calls from within walkers.

JAC is a native Python superset -- it compiles to Python bytecode and can import any Python library directly. Walkers defined in `.jac` files are auto-exposed as `POST /walker/{name}` endpoints via `jac start`. The `by llm()` mechanism uses LiteLLM under the hood, supporting 100+ LLM providers via environment variables (e.g., `OPENAI_API_KEY`). Backboard's Python SDK (`backboard-sdk` v1.5.11) provides async `BackboardClient` with `create_assistant()`, `create_thread()`, and `add_message()` for persistent memory via `memory="Auto"`.

**Primary recommendation:** Use JAC `.jac` files for graph schema (nodes/edges) and walker definitions (QueryWalker, ExplorationWalker) with `by llm()` for NL generation. Use `jac start` to serve walkers as REST endpoints on port 8000. The existing FastAPI server on port 8001 remains the CLIP/similarity compute engine, called from JAC walkers via `httpx`. Backboard integration lives in the JAC walkers (not Python server) since it's the walker that needs conversational context.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GRAPH-05 | QueryWalker traverses spatial graph and synthesizes NL answers via Backboard LLM routing + `by llm()` | JAC walker traversal + `by llm()` LiteLLM integration + Backboard thread context |
| GRAPH-06 | ExplorationWalker autonomously catalogs all objects and writes catalog to Backboard memory | JAC walker with visited-set pattern + Backboard `add_message(memory="Auto")` |
| BB-01 | Backboard assistant created with spatial reasoning system prompt | `BackboardClient.create_assistant(name, system_prompt)` |
| BB-02 | Each scanned scene gets its own Backboard thread for persistent spatial memory | `BackboardClient.create_thread(assistant_id)` -- one thread per scene |
| BB-03 | ExplorationWalker writes full object catalog to Backboard thread via `add_message()` with `memory="Auto"` | Backboard `add_message(thread_id, content, memory="Auto")` -- confirmed in SDK |
| BB-04 | QueryWalker routes spatial questions through Backboard for LLM reasoning enriched with spatial graph context | Backboard `add_message()` with spatial context injected into content string |
| BB-05 | Multi-turn conversational context -- follow-up queries retain context | Backboard thread maintains context across messages automatically |
| BB-06 | Cross-session recall -- revisiting a scene reads spatial memory from Backboard | Backboard memory persists at assistant level across threads; scene thread_id stored in app_state |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- Package manager: pnpm (frontend), pip in venv (Python)
- Language: TypeScript for frontend, Python for backend
- Backend: Node.js default overridden -- this project uses Python FastAPI + JAC
- Always handle errors with try/catch
- Never hardcode API keys -- use environment variables
- Keep files under 300 lines
- Git: conventional commits (feat:, fix:, chore:, docs:)

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| jaseci | 2.3.12 | JAC runtime + byllm + jac-client + jac-scale | Meta-package installs full JAC ecosystem; released April 3 2026; Python >=3.11 |
| jaclang | 0.13.5 | Jac language compiler/interpreter | Core walker/node/edge engine; bundled with jaseci |
| byllm | 0.6.3 | `by llm()` LLM integration via LiteLLM | Turns function signatures into auto-generated LLM prompts; bundled with jaseci |
| backboard-sdk | 1.5.11 | Backboard Python SDK for persistent memory | Async client for assistant/thread/message management; lightweight (httpx + pydantic) |
| litellm | 1.82.6 | LLM provider routing (bundled via byllm) | Supports 100+ providers via env vars; no direct dependency needed |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| httpx | 0.28.x | HTTP calls from JAC walkers to FastAPI CLIP server | Already in project; used for JAC -> Python server calls |
| python-dotenv | 1.x | Load OPENAI_API_KEY, BACKBOARD_API_KEY from .env | Already in project requirements |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JAC walkers | Pure Python FastAPI endpoints implementing walker pattern | Loses "Best JAC Builder" prize eligibility; no `by llm()`; must hand-roll graph traversal |
| Backboard for memory | In-memory dict / Redis | Loses "Best Use of Backboard" prize; no cross-session persistence |
| `by llm()` via byllm | Direct OpenAI API calls | More control but no auto-prompt from type signatures; more boilerplate |

**Installation:**
```bash
# In project venv
pip install jaseci backboard-sdk
```

**Version verification:** jaseci 2.3.12 (PyPI April 3 2026), backboard-sdk 1.5.11 (PyPI, dry-run confirmed installable on Python 3.14).

## Architecture Patterns

### Recommended Project Structure
```
server/
  jac/
    main.jac              # Entry point: graph schema + walker definitions
    walkers/
      query_walker.jac     # QueryWalker: NL query -> graph traversal -> answer
      explore_walker.jac   # ExplorationWalker: catalog all nodes
    nodes.jac              # SpatialObject node, SpatialRelation edge schema
  services/
    backboard_client.py   # BackboardClient wrapper (create assistant, thread, memory ops)
    graph_builder.py      # (existing) Scene graph builder
  routers/
    walker_proxy.py       # FastAPI proxy endpoints that forward to JAC server
    scene.py              # (existing) Scene build endpoint
```

### Pattern 1: Hybrid JAC/Python -- JAC for Graph Logic, Python for Compute

**What:** JAC `.jac` files define the spatial graph schema and walker traversal logic. Walkers call the Python FastAPI server over HTTP for CLIP similarity computation. The JAC server (`jac start`) runs on port 8000; the Python FastAPI server runs on port 8001.

**When to use:** When the project requires both JAC's graph-native walker paradigm (for hackathon prizes) AND heavy Python compute (CLIP, numpy, DBSCAN).

**Example -- JAC walker calling Python server:**
```jac
import:py from httpx { Client }

glob http_client = Client(base_url="http://localhost:8001");

walker:pub QueryWalker {
    has query: str;
    has session_id: str;
    has results: list = [];
    has visited: set = set();

    can generate_answer(context: str, query: str) -> str
        by llm(temperature=0.7);

    with `root entry {
        visit [-->](`?SpatialObject);
    }

    with SpatialObject entry {
        if here.id in self.visited {
            disengage;
        }
        self.visited.add(here.id);

        # Check if this node is relevant to the query
        if here.label in self.query or self.is_relevant(here) {
            self.results.append({
                "id": here.id,
                "label": here.label,
                "centroid": here.centroid,
                "description": here.description
            });
        }

        visit [-->](`?SpatialObject);
    }
}
```
Source: JAC walker syntax from [docs.jaseci.org](https://docs.jaseci.org/tutorials/production/local/), [jac-byllm](https://github.com/jaseci-labs/jaseci/tree/main/jac-byllm)

### Pattern 2: Backboard Memory Integration from Python Services

**What:** A Python wrapper around `BackboardClient` manages assistant creation, thread lifecycle, and memory operations. Walkers (or their Python proxy layer) call this service to store spatial facts and retrieve conversational context.

**When to use:** For BB-01 through BB-06. Backboard SDK is async Python; easier to manage from the Python layer than directly from JAC.

**Example -- Backboard service:**
```python
from backboard import BackboardClient

class SpatialMemoryService:
    """Manages Backboard assistant and threads for spatial memory."""

    def __init__(self, api_key: str):
        self.client = BackboardClient(api_key=api_key)
        self.assistant_id = None
        self.scene_threads: dict[str, str] = {}  # scene_id -> thread_id

    async def initialize(self, system_prompt: str):
        assistant = await self.client.create_assistant(
            name="SpatialMind",
            system_prompt=system_prompt,
        )
        self.assistant_id = assistant.assistant_id

    async def get_or_create_thread(self, scene_id: str) -> str:
        if scene_id not in self.scene_threads:
            thread = await self.client.create_thread(self.assistant_id)
            self.scene_threads[scene_id] = thread.thread_id
        return self.scene_threads[scene_id]

    async def store_catalog(self, scene_id: str, catalog: str):
        thread_id = await self.get_or_create_thread(scene_id)
        await self.client.add_message(
            thread_id=thread_id,
            content=catalog,
            stream=False,
        )

    async def query_with_context(self, scene_id: str, query: str, spatial_context: str) -> str:
        thread_id = await self.get_or_create_thread(scene_id)
        enriched = f"Spatial context: {spatial_context}\n\nUser question: {query}"
        response = await self.client.add_message(
            thread_id=thread_id,
            content=enriched,
            stream=False,
        )
        return response.content
```
Source: [Backboard quickstart](https://docs.backboard.io/quickstart), [Backboard cookbook](https://github.com/Backboard-io/backboard_io_cookbook)

### Pattern 3: JAC Library Mode as Fallback

**What:** If `jac start` proves too complex or unstable during the hackathon, use JAC's Library Mode instead. Define nodes/walkers as Python classes inheriting from `jaclang.lib` (Node, Edge, Walker), spawn walkers from FastAPI endpoints.

**When to use:** If the JAC server is unreliable or adds too much complexity. This still counts as "using JAC" for the prize.

**Example -- Library mode walker in Python:**
```python
from jaclang.lib import Node, Edge, Walker, on_entry, spawn, root, connect

class SpatialObject(Node):
    id: str
    label: str
    centroid: list[float]
    bbox: list[float]
    gaussian_indices: list[int]
    confidence: float
    description: str = ""
    visited: bool = False

class SpatialRelation(Edge):
    relation: str
    confidence: float
    distance: float

class ExplorationWalker(Walker):
    catalog: list = []

    @on_entry
    def explore_node(self, here: SpatialObject) -> None:
        if not here.visited:
            here.visited = True
            self.catalog.append({
                "id": here.id,
                "label": here.label,
                "centroid": here.centroid,
            })

# Spawn from FastAPI endpoint
result = spawn(ExplorationWalker(), root())
```
Source: [JAC Library Mode docs](https://docs.jaseci.org/reference/language/library-mode/)

### Pattern 4: Scene Graph Ingestion -- Python Dict to JAC Graph

**What:** Phase 3 produces a scene graph as a Python dict (cached in `app_state`). Phase 4 must ingest this into JAC's graph store. Two approaches: (a) a SceneBuilder walker that accepts JSON via HTTP and creates JAC nodes/edges, or (b) Library Mode to build the graph directly in Python.

**When to use:** At scene load time, after `POST /scene/build` populates the Python-side graph.

### Anti-Patterns to Avoid

- **Running JAC and Python as tightly coupled**: Keep them as separate services with REST boundary (port 8000 JAC, port 8001 Python). No subprocess calls, no shared memory.
- **Synchronous Backboard calls in the query hot path**: Fire Backboard memory writes as fire-and-forget. Pre-fetch memory context at session start, not per-query. Backboard adds ~200-500ms per call.
- **Skipping the visited set in walkers**: Spatial graphs have cycles (object -> room -> object). Without `visited` tracking, walkers loop infinitely. Always include `has visited: set = set()` and check at entry.
- **Using Backboard as the LLM for `by llm()`**: Backboard's LLM routing is for conversational memory. `by llm()` uses LiteLLM directly. These are separate LLM pathways with different purposes.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LLM prompt engineering for spatial descriptions | Custom prompt templates | `by llm()` with typed function signatures | JAC extracts function name + types + docstring as the prompt automatically; no prompt engineering |
| Conversational memory across sessions | Custom Redis/SQLite memory store | Backboard `add_message(memory="Auto")` | Backboard automatically extracts, stores, and retrieves relevant memories across threads |
| Walker-to-REST-endpoint mapping | Custom Flask/FastAPI wrappers around walker logic | `jac start` auto-endpoint generation | Every `:pub` walker becomes `POST /walker/{name}` automatically |
| LLM provider switching | Custom provider abstraction layer | LiteLLM (via byllm) | Set OPENAI_API_KEY or ANTHROPIC_API_KEY env var; byllm handles routing |
| Graph traversal with cycle detection | Custom BFS/DFS with visited set | JAC walker `visit` + `disengage` + `has visited: set` | Walker paradigm is JAC's core abstraction; use it natively |

## Common Pitfalls

### Pitfall 1: JAC Walker Infinite Traversal on Cyclic Spatial Graph
**What goes wrong:** Spatial graphs have cycles (Room -> Table -> Chair -> Room). A walker traversing without loop prevention revisits nodes indefinitely.
**Why it happens:** JAC's `visit` continues to adjacent nodes. Without `disengage` or visited-set, walkers loop forever. Beginners confuse `return` (exits function, walker continues) with `disengage` (terminates walker).
**How to avoid:** Always include `has visited: set = set()` in walker state. At top of every `with entry` block, check `if here.id in self.visited { disengage; }` then add to visited set. Add max_depth counter as safety net.
**Warning signs:** Walker endpoint doesn't return within 5 seconds; JAC server CPU at 100%.

### Pitfall 2: Backboard Latency in Query Hot Path
**What goes wrong:** Awaiting Backboard memory retrieval before returning query results adds 200-500ms per query.
**Why it happens:** Backboard is an external API. Network latency is unavoidable.
**How to avoid:** Return query results immediately. Fire Backboard memory write as background task. Pre-fetch memory context at session start (when thread is created), cache it locally.
**Warning signs:** Query response time > 1 second when it should be < 500ms.

### Pitfall 3: `by llm()` Requires LLM API Key Configuration
**What goes wrong:** `by llm()` calls fail silently or throw cryptic errors when no LLM provider is configured.
**Why it happens:** byllm uses LiteLLM under the hood. It needs an API key via environment variable.
**How to avoid:** Set `OPENAI_API_KEY` in `.env` before starting JAC server. Configure default model in `jac.toml`:
```toml
[plugins.byllm]
system_prompt = "You are a spatial reasoning AI that describes objects and their relationships in 3D scenes."
[plugins.byllm.model]
default_model = "gpt-4o-mini"
```
**Warning signs:** Walker endpoints return empty descriptions or 500 errors.

### Pitfall 4: Python 3.14 Compatibility Risk
**What goes wrong:** The project venv runs Python 3.14.3. Jaseci 2.3.12 specifies Python >=3.11 but may not be tested against 3.14.
**Why it happens:** Arch Linux ships Python 3.14 as system default. Jaseci wheels are `py3-none-any` (pure Python) so they install, but runtime behavior on 3.14 is untested.
**How to avoid:** Dry-run install succeeded. Test `jac start` with a minimal `.jac` file immediately after install. If it fails, create a separate Python 3.12 venv.
**Warning signs:** Import errors, `ast` module changes in 3.14 breaking JAC's compiler.

### Pitfall 5: Two Servers Need Coordinated Startup
**What goes wrong:** JAC server (port 8000) and Python FastAPI server (port 8001) must both be running. If only one starts, walkers fail to reach CLIP endpoints.
**Why it happens:** Two independent processes with a REST dependency.
**How to avoid:** Create a startup script that launches both. Health check from JAC -> Python before accepting walker requests. Add a startup probe in the JAC main.jac that pings `http://localhost:8001/health`.
**Warning signs:** Walker returns "connection refused" errors when calling Python server.

## Code Examples

### JAC Node/Edge Schema for Spatial Graph
```jac
# nodes.jac
node SpatialObject {
    has id: str;
    has label: str;
    has centroid: list[float];
    has bbox: list[float];
    has gaussian_indices: list[int];
    has confidence: float;
    has description: str = "";
    has visited: bool = False;
}

edge SpatialRelation {
    has relation: str;     # "on_top_of" | "next_to" | "inside" | "facing"
    has confidence: float;
    has distance: float;
}

edge HierarchyRelation {
    has relation: str;     # "similar_to" | "part_of"
    has similarity: float;
}
```
Source: Architecture from [ARCHITECTURE.md research](../../research/ARCHITECTURE.md)

### Backboard Assistant Initialization
```python
SPATIAL_SYSTEM_PROMPT = """You are SpatialMind, an AI that understands 3D spaces.
You have access to a spatial knowledge graph of objects in a scanned room.
When answering questions:
- Reference specific objects by name and their spatial relationships
- Use directional language (left of, above, near, facing)
- If you remember objects from previous conversations, reference that memory
- Be concise but spatially precise"""

async def init_backboard():
    client = BackboardClient(api_key=os.getenv("BACKBOARD_API_KEY"))
    assistant = await client.create_assistant(
        name="SpatialMind",
        system_prompt=SPATIAL_SYSTEM_PROMPT,
    )
    return client, assistant.assistant_id
```
Source: [Backboard quickstart](https://docs.backboard.io/quickstart)

### ExplorationWalker Catalog Generation
```python
def format_catalog(nodes: list[dict]) -> str:
    """Convert scene graph nodes into a natural language catalog for Backboard."""
    lines = ["Scene Object Catalog:"]
    for node in nodes:
        cx, cy, cz = node["centroid"]
        lines.append(
            f"- {node['label']} (id: {node['id']}): "
            f"located at ({cx:.1f}, {cy:.1f}, {cz:.1f}), "
            f"confidence {node['confidence']:.2f}"
        )
    return "\n".join(lines)
```

### JAC `by llm()` for Spatial Descriptions
```jac
can describe_spatial_context(
    object_name: str,
    neighbors: list[dict],
    relationships: list[dict]
) -> str
    """Given an object and its spatial neighbors with relationships,
    generate a natural language description of where the object is
    and what surrounds it."""
    by llm(temperature=0.7);
```
Source: [jac-byllm README](https://github.com/jaseci-labs/jaseci/tree/main/jac-byllm)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual prompt engineering for LLM calls | `by llm()` auto-prompt from function signatures | Jaseci 2.3.x (2026) | No prompt strings needed; type safety on outputs |
| Separate walker server via jac-cloud | `jac start` local dev server with auto-endpoints | jaclang 0.13.x (2026) | Simpler deployment; walkers = REST endpoints automatically |
| Custom memory stores (Redis, SQLite) | Backboard managed memory with `memory="Auto"` | Backboard 2025-2026 | Automatic extraction + retrieval of relevant facts |

## Open Questions

1. **Python 3.14 compatibility with jaseci**
   - What we know: Dry-run install succeeds; wheels are pure Python
   - What's unclear: Runtime behavior of jaclang compiler on Python 3.14 (ast module changes)
   - Recommendation: Test immediately after install; fall back to Library Mode if broken

2. **Backboard `add_message` memory parameter in Python SDK**
   - What we know: REST API supports `"memory": "Auto"`; SDK shows `add_message(thread_id, content, stream)`
   - What's unclear: Whether Python SDK `add_message()` accepts a `memory` parameter directly
   - Recommendation: Check SDK source after install; if not supported, use REST API directly via httpx

3. **JAC jac-scale MongoDB persistence**
   - What we know: jac-scale uses MongoDB for node/edge persistence
   - What's unclear: Whether we need MongoDB running for `jac start` or if it falls back to in-memory
   - Recommendation: Start without MongoDB; if persistence is needed, use SQLite or skip jac-scale

4. **Minimum JAC for "Best JAC Builder" prize**
   - What we know: Prize values "deep leverage of by llm(), walkers, graph-native data modeling"
   - What's unclear: Whether Library Mode (Python classes) counts equally vs. `.jac` files
   - Recommendation: Use `.jac` files for graph schema and walkers; this is more visually "JAC" to judges

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.14 | All backend | Yes | 3.14.3 | -- |
| jaseci (pip) | JAC walkers | No (not installed) | 2.3.12 (dry-run OK) | Library Mode via jaclang.lib |
| backboard-sdk (pip) | Backboard memory | No (not installed) | 1.5.11 (dry-run OK) | Direct REST API via httpx |
| MongoDB | jac-scale persistence | Not checked | -- | In-memory graph (no persistence across restarts) |
| OPENAI_API_KEY | `by llm()` LLM calls | Not checked | -- | ANTHROPIC_API_KEY as alternative |
| BACKBOARD_API_KEY | Backboard SDK | Not checked | -- | None -- blocking if missing |

**Missing dependencies with no fallback:**
- BACKBOARD_API_KEY env var -- must be provided for BB-01 through BB-06
- At least one LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) -- required for `by llm()`

**Missing dependencies with fallback:**
- jaseci package -- install via pip; fallback to Library Mode if runtime fails on Python 3.14
- backboard-sdk -- install via pip; fallback to direct REST API via httpx
- MongoDB -- not required for hackathon; in-memory graph sufficient for single-scene demo

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ with pytest-asyncio |
| Config file | None -- uses default pytest discovery in server/tests/ |
| Quick run command | `python -m pytest server/tests/ -x -q` |
| Full suite command | `python -m pytest server/tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GRAPH-05 | QueryWalker traverses graph and returns NL answer | integration | `python -m pytest server/tests/test_query_walker.py -x` | No -- Wave 0 |
| GRAPH-06 | ExplorationWalker catalogs all nodes and writes to Backboard | integration | `python -m pytest server/tests/test_exploration_walker.py -x` | No -- Wave 0 |
| BB-01 | Backboard assistant created with system prompt | unit | `python -m pytest server/tests/test_backboard_service.py::test_create_assistant -x` | No -- Wave 0 |
| BB-02 | Scene gets its own Backboard thread | unit | `python -m pytest server/tests/test_backboard_service.py::test_create_thread -x` | No -- Wave 0 |
| BB-03 | ExplorationWalker writes catalog to Backboard with memory=Auto | integration | `python -m pytest server/tests/test_backboard_service.py::test_store_catalog -x` | No -- Wave 0 |
| BB-04 | QueryWalker routes questions through Backboard with spatial context | integration | `python -m pytest server/tests/test_query_walker.py::test_backboard_enriched_query -x` | No -- Wave 0 |
| BB-05 | Multi-turn context persists in thread | integration | `python -m pytest server/tests/test_backboard_service.py::test_multi_turn -x` | No -- Wave 0 |
| BB-06 | Cross-session recall reads from Backboard | integration | `python -m pytest server/tests/test_backboard_service.py::test_cross_session -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest server/tests/ -x -q`
- **Per wave merge:** `python -m pytest server/tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `server/tests/test_query_walker.py` -- covers GRAPH-05, BB-04
- [ ] `server/tests/test_exploration_walker.py` -- covers GRAPH-06
- [ ] `server/tests/test_backboard_service.py` -- covers BB-01 through BB-06
- [ ] `server/tests/conftest.py` -- add fixtures for mock Backboard client and mock JAC graph

## Sources

### Primary (HIGH confidence)
- [jaseci PyPI](https://pypi.org/project/jaseci/) -- v2.3.12, April 3 2026, confirmed installable
- [jac-byllm GitHub](https://github.com/jaseci-labs/jaseci/tree/main/jac-byllm) -- by llm() mechanism, LiteLLM integration, model config
- [JAC local server docs](https://docs.jaseci.org/tutorials/production/local/) -- `jac start`, walker auto-endpoints, port config
- [JAC Library Mode](https://docs.jaseci.org/reference/language/library-mode/) -- Node/Edge/Walker Python classes, spawn(), connect()
- [Backboard quickstart](https://docs.backboard.io/quickstart) -- BackboardClient, create_assistant, create_thread, add_message
- [Backboard cookbook](https://github.com/Backboard-io/backboard_io_cookbook) -- 8 Python recipes, memory patterns

### Secondary (MEDIUM confidence)
- [JacHacks 2026 Devpost](https://jachacks-2026.devpost.com/) -- Prize criteria: "Best JAC Builder" $400, deep use of by llm(), walkers, graph-native modeling
- [JAC Python interop blog](https://blogs.jaseci.org/blog/2025/09/17/python-getting-jacd/) -- Python/JAC bidirectional imports
- [Backboard-sdk PyPI](https://pypi.org/project/backboard-sdk/) -- v1.5.11, dry-run installable

### Tertiary (LOW confidence)
- Backboard `memory` parameter on Python SDK `add_message()` -- REST API confirms `memory="Auto"` but SDK code examples don't explicitly show this parameter. Needs validation after install.
- JAC `jac start` stability on Python 3.14 -- untested; pure-Python wheels install but runtime may have ast compatibility issues.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- jaseci 2.3.12 and backboard-sdk 1.5.11 both confirmed on PyPI, installable
- Architecture: MEDIUM -- hybrid JAC/Python pattern is documented but novel for this project; Library Mode fallback adds safety
- Pitfalls: HIGH -- walker infinite loops, Backboard latency, API key config all well-documented

**Research date:** 2026-04-04
**Valid until:** 2026-04-11 (hackathon-specific; jaseci and backboard are fast-moving)
