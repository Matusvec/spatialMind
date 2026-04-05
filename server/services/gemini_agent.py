"""Gemini-backed tool-calling agent for SpatialMind chat."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from server.services.dense_query import ground_query

logger = logging.getLogger(__name__)

AGENT_STEP_ORDER = [
    ("agent_plan", "Plan with Gemini"),
    ("agent_tools", "Execute scene tools"),
    ("agent_respond", "Compose answer"),
]

SYSTEM_PROMPT = """
You are SpatialMind, an agent for grounding spatial language in a 3D Gaussian scene.

Rules:
- For scene queries and edit requests, prefer calling tools instead of guessing.
- Use `find_relevant_splats` to inspect a query in the scene.
- Use `highlight_query` when the user wants to find, show, or point out something.
- Use `recolor_query` when the user wants to change an object's color.
- Use `move_query_region` only if the user clearly requests moving something. If the tool says movement is unsupported, explain that plainly.
- Keep answers concise and concrete.
- If a tool reports no match, say so directly.
- Each tool accepts a `level` parameter to control semantic granularity:
  0 = whole objects (chair, table, person)
  1 = parts (armrest, table leg, shoe)
  2 = subparts (button, hair, zipper)
  Default is level 2 for most queries. Use level 0 for broad objects, level 1 for object parts, level 2 for fine details.
""".strip()

_LEVEL_PARAM = {
    "type": "integer",
    "description": "Semantic level: 0=whole objects (chair, table), 1=parts (armrest, leg), 2=subparts (hair, button). Default 2.",
    "enum": [0, 1, 2],
}

TOOL_DECLARATIONS = [
    {
        "name": "find_relevant_splats",
        "description": "Ground a natural-language scene query to relevant Gaussian splats and summarize the matched region.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The object, part, or region to ground in the 3D scene.",
                },
                "level": _LEVEL_PARAM,
            },
            "required": ["query"],
        },
    },
    {
        "name": "highlight_query",
        "description": "Highlight the exact Gaussian splats that match a query. Use this for show/find/highlight requests.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The object, part, or region to highlight.",
                },
                "color": {
                    "type": "string",
                    "description": "Hex color to use for highlighting, like #7c3aed.",
                },
                "level": _LEVEL_PARAM,
            },
            "required": ["query"],
        },
    },
    {
        "name": "recolor_query",
        "description": "Permanently recolor the Gaussian splats that match a query. Use this for edit commands.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The object, part, or region to recolor.",
                },
                "color": {
                    "type": "string",
                    "description": "Hex color to apply, like #0000ff.",
                },
                "level": _LEVEL_PARAM,
            },
            "required": ["query", "color"],
        },
    },
    {
        "name": "move_query_region",
        "description": "Attempt to move a grounded region in 3D. This is a forward-looking tool and may be unavailable.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The object or region to move.",
                },
                "dx": {
                    "type": "number",
                    "description": "X-axis translation in scene units.",
                },
                "dy": {
                    "type": "number",
                    "description": "Y-axis translation in scene units.",
                },
                "dz": {
                    "type": "number",
                    "description": "Z-axis translation in scene units.",
                },
            },
            "required": ["query", "dx", "dy", "dz"],
        },
    },
]


def gemini_enabled(state: dict) -> bool:
    """Return True when Gemini agent config is present."""
    config = state.get("config")
    return bool(config and getattr(config, "gemini_api_key", ""))


def init_agent_steps() -> list[dict]:
    """Return the default Gemini agent progress steps."""
    return [
        {"key": key, "label": label, "status": "pending", "detail": None}
        for key, label in AGENT_STEP_ORDER
    ]


class GeminiSceneAgent:
    """Small tool-calling agent that grounds chat requests in the scene."""

    def __init__(self, *, state: dict, progress_cb=None) -> None:
        self.state = state
        self.progress_cb = progress_cb
        self.config = state.get("config")
        if self.config is None or not getattr(self.config, "gemini_api_key", ""):
            raise RuntimeError("Gemini agent requested but GEMINI_API_KEY is not configured.")

        self.base_url = getattr(self.config, "gemini_api_base", "https://generativelanguage.googleapis.com").rstrip("/")
        self.model = getattr(self.config, "gemini_model", "gemini-2.5-flash")
        self.api_key = self.config.gemini_api_key
        self.pending_payload: dict[str, Any] | None = None

    def run(self, *, query: str, mode: str = "query") -> dict[str, Any]:
        """Run a short Gemini tool loop for one user query."""
        prompt = f"{SYSTEM_PROMPT}\n\nUser request: {query}"
        contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": prompt}]}]

        for turn in range(4):
            if turn == 0:
                self._progress("agent_plan", "active")
            response = self._generate(contents)
            if turn == 0:
                self._progress("agent_plan", "completed", detail=self.model)

            candidate = (response.get("candidates") or [{}])[0]
            model_content = candidate.get("content") or {"role": "model", "parts": []}
            function_calls = self._extract_function_calls(model_content)

            if function_calls:
                contents.append(model_content)
                self._progress("agent_tools", "active", detail=f"{len(function_calls)} tool call(s)")
                tool_parts = []
                tool_names = []
                for call in function_calls:
                    name = call.get("name", "")
                    args = call.get("args") or {}
                    tool_names.append(name)
                    tool_result = self._execute_tool(name, args, mode=mode)
                    tool_parts.append(
                        {
                            "functionResponse": {
                                "name": name,
                                "response": {"result": tool_result},
                            }
                        }
                    )
                contents.append({"role": "tool", "parts": tool_parts})
                self._progress("agent_tools", "completed", detail=", ".join(tool_names))
                continue

            text = self._extract_text(model_content)
            self._progress("agent_respond", "active")
            result = self._compose_result(query=query, text=text, mode=mode)
            self._progress("agent_respond", "completed", detail="response ready")
            return result

        raise RuntimeError("Gemini agent exceeded tool loop limit.")

    def _generate(self, contents: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": contents,
            "tools": [{"functionDeclarations": TOOL_DECLARATIONS}],
            "generationConfig": {
                "temperature": 0.2,
            },
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, params={"key": self.api_key}, json=payload)
            response.raise_for_status()
            return response.json()

    def _progress(self, step_key: str, status: str, detail: str | None = None) -> None:
        if self.progress_cb is not None:
            self.progress_cb(step_key, status, detail)

    @staticmethod
    def _extract_function_calls(model_content: dict[str, Any]) -> list[dict[str, Any]]:
        calls = []
        for part in model_content.get("parts", []):
            if "functionCall" in part:
                calls.append(part["functionCall"])
        return calls

    @staticmethod
    def _extract_text(model_content: dict[str, Any]) -> str:
        texts = [part.get("text", "") for part in model_content.get("parts", []) if part.get("text")]
        return "\n".join(texts).strip()

    def _execute_tool(self, name: str, args: dict[str, Any], *, mode: str) -> dict[str, Any]:
        if name == "find_relevant_splats":
            return self._tool_find_relevant_splats(args)
        if name == "highlight_query":
            return self._tool_highlight_query(args)
        if name == "recolor_query":
            return self._tool_recolor_query(args)
        if name == "move_query_region":
            return self._tool_move_query_region(args)
        return {"status": "error", "message": f"Unknown tool '{name}'."}

    def _run_dense(self, query: str, level: int | None = None) -> dict[str, Any]:
        preferred_levels = (level,) if level is not None else (0, 1, 2)
        return ground_query(
            text=query,
            state=self.state,
            persist_instances=True,
            preferred_levels=preferred_levels,
        )

    @staticmethod
    def _summarize_dense(result: dict[str, Any], *, query: str) -> dict[str, Any]:
        nodes = result.get("nodes", [])
        labels = [node.get("label", "region") for node in nodes[:6]]
        return {
            "status": "ok" if nodes else "no_match",
            "query": query,
            "label_count": len(nodes),
            "labels": labels,
            "matched_gaussian_count": len(result.get("highlight_indices", [])),
            "semantic_levels": result.get("semantic_levels", []),
            "reasoning": result.get("reasoning", ""),
        }

    def _tool_find_relevant_splats(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        level = args.get("level")
        if isinstance(level, (int, float)):
            level = int(level)
        else:
            level = None
        result = self._run_dense(query, level=level)
        self.pending_payload = None
        return self._summarize_dense(result, query=query)

    def _tool_highlight_query(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        color = str(args.get("color") or "#7c3aed").strip() or "#7c3aed"
        level = args.get("level")
        if isinstance(level, (int, float)):
            level = int(level)
        else:
            level = None
        result = self._run_dense(query, level=level)
        self.pending_payload = {
            "action": "highlight",
            "color": color,
            "result": result,
        }
        summary = self._summarize_dense(result, query=query)
        summary["color"] = color
        return summary

    def _tool_recolor_query(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        color = str(args.get("color") or "#7c3aed").strip() or "#7c3aed"
        level = args.get("level")
        if isinstance(level, (int, float)):
            level = int(level)
        else:
            level = None
        result = self._run_dense(query, level=level)
        self.pending_payload = {
            "action": "recolor",
            "color": color,
            "result": result,
        }
        summary = self._summarize_dense(result, query=query)
        summary["color"] = color
        return summary

    def _tool_move_query_region(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        return {
            "status": "unsupported",
            "query": query,
            "message": "Region movement is not implemented yet. Grounding and recoloring are supported now.",
        }

    def _compose_result(self, *, query: str, text: str, mode: str) -> dict[str, Any]:
        # Always ground the ORIGINAL user query for highlighting —
        # Gemini may have rewritten it to something narrower for tool calls.
        # Use the level from the pending payload if Gemini specified one.
        fresh_result = self._run_dense(query)

        if self.pending_payload is None:
            return format_dense_payload(fresh_result, mode=mode, query=query, answer=text or fresh_result.get("answer", ""))

        action = self.pending_payload["action"]
        color = self.pending_payload.get("color")
        # Use fresh grounding for highlight data, Gemini text for the answer
        payload = format_dense_payload(fresh_result, mode=mode, query=query, answer=text or fresh_result.get("answer", ""))
        if action == "highlight":
            payload["highlight_color"] = color
            payload["description"] = text or payload.get("description", "")
            payload["answer"] = text or payload.get("answer", "")
            return payload
        if action == "recolor":
            matched = result.get("nodes", [])
            target_label = matched[0].get("label", "region") if matched else "region"
            payload.update(
                {
                    "action": "recolor",
                    "target": target_label,
                    "color": color or "#7c3aed",
                    "matched_gaussians": result.get("highlight_indices", []),
                    "matched_nodes": [
                        {"label": n.get("label", ""), "confidence": 0.9}
                        for n in matched
                    ],
                }
            )
            if not text:
                payload["description"] = f"Changed the {target_label} to {color or '#7c3aed'}."
                payload["answer"] = payload["description"]
            return payload
        return payload


def format_dense_payload(result: dict[str, Any], *, mode: str, query: str, answer: str) -> dict[str, Any]:
    """Normalize dense-query output into the frontend payload shape."""
    matched_nodes = result.get("nodes", [])
    payload = {
        "description": answer or result.get("answer", ""),
        "answer": answer or result.get("answer", ""),
        "nodes": [
            {"label": n.get("label", ""), "confidence": 0.9}
            for n in matched_nodes
        ],
        "reasoning": result.get("reasoning", ""),
        "highlight_regions": result.get("highlight_regions", []),
        "highlight_match": result.get("highlight_match", {"indices": [], "scores": [], "level": 2}),
        "semantic_fallback": result.get("semantic_fallback", False),
        "agent_provider": "gemini",
        "query": query,
    }
    if mode == "edit":
        payload["action"] = "recolor"
        payload["matched_gaussians"] = result.get("highlight_indices", [])
    return payload
