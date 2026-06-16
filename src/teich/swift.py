"""Convert normalized Teich training rows into ms-swift dataset rows.

Teich's ``convert``/``convert_traces_to_training_data`` step produces OpenAI-style
rows (``prompt``/``messages``/``tools``/``metadata``). ms-swift (used to fine-tune
Qwen3-style reasoning models) expects a different shape:

- reasoning is inlined into assistant ``content`` as ``<think>...</think>`` rather
  than a separate ``reasoning_content`` field;
- tool calls become standalone ``tool_call`` role messages whose content is a JSON
  string ``{"name": ..., "arguments": {...}}``;
- tool outputs use the ``tool_response`` role;
- ``tools`` is serialized to a JSON string instead of a list.

This module performs that ``message -> message2`` conversion so the pipeline is
``trace -> message (converter) -> message2 (swift)``.
"""

from __future__ import annotations

import json
from typing import Any

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _normalize_swift_role(role: str) -> str:
    if role == "developer":
        return "system"
    if role == "model":
        return "assistant"
    return role


def _coerce_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _wrap_thinking(content: str, reasoning: str) -> str:
    block = f"{_THINK_OPEN}\n{reasoning.strip()}\n{_THINK_CLOSE}"
    if content:
        return f"{block}\n\n{content}"
    return block


def _parse_tool_arguments(arguments: Any) -> Any:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _tool_call_content(tool_call: Any) -> str | None:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = _parse_tool_arguments(function.get("arguments"))
    return json.dumps({"name": name.strip(), "arguments": arguments}, ensure_ascii=False)


def to_ms_swift_messages(
    messages: list[dict[str, Any]],
    *,
    keep_intermediate_thinking: bool = False,
) -> list[dict[str, str]]:
    """Convert Teich messages to ms-swift native (agent) messages.

    By default ``<think>`` is kept only on the final assistant turn, matching the
    Qwen3 convention where historical reasoning is not present at inference time.
    Set ``keep_intermediate_thinking=True`` to retain reasoning on every assistant
    turn (often desirable for agent traces).
    """
    last_assistant_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, dict) and _normalize_swift_role(str(message.get("role") or "")) == "assistant":
            last_assistant_index = index

    swift_messages: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = _normalize_swift_role(str(message.get("role") or "").strip())
        if not role:
            continue
        if role == "assistant":
            content = _coerce_content_text(message.get("content"))
            reasoning = message.get("reasoning_content")
            include_thinking = keep_intermediate_thinking or index == last_assistant_index
            if include_thinking and isinstance(reasoning, str) and reasoning.strip():
                content = _wrap_thinking(content, reasoning)
            if content:
                swift_messages.append({"role": "assistant", "content": content})
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    call_content = _tool_call_content(tool_call)
                    if call_content is not None:
                        swift_messages.append({"role": "tool_call", "content": call_content})
            continue
        if role == "tool":
            swift_messages.append({"role": "tool_response", "content": _coerce_content_text(message.get("content"))})
            continue
        swift_messages.append({"role": role, "content": _coerce_content_text(message.get("content"))})
    return swift_messages


def to_ms_swift_row(
    row: dict[str, Any],
    *,
    keep_intermediate_thinking: bool = False,
    include_tools: bool = True,
    tools_as_json_string: bool = True,
) -> dict[str, Any]:
    """Convert one normalized Teich row into an ms-swift dataset row."""
    messages = row.get("messages") if isinstance(row, dict) else None
    if not isinstance(messages, list):
        messages = []
    swift_row: dict[str, Any] = {
        "messages": to_ms_swift_messages(messages, keep_intermediate_thinking=keep_intermediate_thinking),
    }
    tools = row.get("tools") if isinstance(row, dict) else None
    if include_tools and isinstance(tools, list) and tools:
        swift_row["tools"] = json.dumps(tools, ensure_ascii=False) if tools_as_json_string else tools
    return swift_row


def convert_to_ms_swift(
    rows: list[dict[str, Any]],
    *,
    keep_intermediate_thinking: bool = False,
    include_tools: bool = True,
    tools_as_json_string: bool = True,
) -> list[dict[str, Any]]:
    """Convert a list of normalized Teich rows into ms-swift dataset rows."""
    return [
        to_ms_swift_row(
            row,
            keep_intermediate_thinking=keep_intermediate_thinking,
            include_tools=include_tools,
            tools_as_json_string=tools_as_json_string,
        )
        for row in rows
    ]
