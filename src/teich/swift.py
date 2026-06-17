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


# Roles that ms-swift treats as "assistant-side" (model output) vs "user-side" (model input).
_OUTPUT_ROLES = ("assistant", "tool_call")
_INPUT_TOOL_PREDECESSORS = ("tool_call", "tool_response")


def _clean_ms_swift_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize converted ms-swift messages into a sequence ms-swift can train on.

    ms-swift requires user-side (``user``/``tool_response``) and assistant-side
    (``assistant``/``tool_call``) turns to alternate, with at most one leading
    ``system`` message, ending on an ``assistant`` turn. This collapses the messy
    structure real agent traces carry while preserving valid patterns:

    - all ``system`` messages are merged into one leading system message;
    - consecutive ``user`` (and consecutive ``assistant``) messages are merged;
    - an unanswered tool round before a new ``user`` turn is dropped, so a
      ``tool_response`` is never directly followed by a ``user``;
    - orphan ``tool_response`` messages (no preceding ``tool_call``) are dropped;
    - trailing messages after the last ``assistant`` (incomplete turns) are trimmed.

    Valid parallel (``tool_call`` x N then ``tool_response`` x N) and sequential
    (``tool_response`` then ``tool_call``) tool patterns are preserved. Returns an
    empty list when nothing trainable (no ``assistant`` turn) remains.
    """
    system_parts = [
        message["content"]
        for message in messages
        if message.get("role") == "system" and str(message.get("content") or "").strip()
    ]

    emit: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            continue
        if role in _OUTPUT_ROLES:
            if role == "assistant":
                if not content.strip():
                    continue
                if emit and emit[-1]["role"] == "assistant":
                    emit[-1] = {"role": "assistant", "content": emit[-1]["content"] + "\n\n" + content}
                    continue
            emit.append({"role": role, "content": content})
        elif role == "user":
            if not content.strip():
                continue
            # Drop a dangling, unanswered tool round so a user never follows a tool_response.
            while emit and emit[-1]["role"] in _INPUT_TOOL_PREDECESSORS:
                emit.pop()
            if emit and emit[-1]["role"] == "user":
                emit[-1] = {"role": "user", "content": emit[-1]["content"] + "\n\n" + content}
            else:
                emit.append({"role": "user", "content": content})
        elif role == "tool_response":
            if emit and emit[-1]["role"] in _INPUT_TOOL_PREDECESSORS:
                emit.append({"role": "tool_response", "content": content})
            # else: orphan tool_response (no preceding tool_call) -> drop

    # Trim trailing incomplete turns so the conversation ends on the assistant target.
    while emit and emit[-1]["role"] != "assistant":
        emit.pop()
    if not any(message["role"] == "assistant" for message in emit):
        return []
    if system_parts:
        emit.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return emit


def validate_ms_swift_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Return a list of ms-swift trainability problems in a message list (empty = OK)."""
    issues: list[str] = []
    roles = [message.get("role") for message in messages]
    for index, role in enumerate(roles):
        if role == "system" and index != 0:
            issues.append(f"system at position {index} (must be leading)")
    if roles.count("system") > 1:
        issues.append("multiple system messages")
    prev = None
    for index, role in enumerate(roles):
        if role == "user" and prev == "user":
            issues.append(f"consecutive user at position {index}")
        if role == "user" and prev == "tool_response":
            issues.append(f"user directly after tool_response at position {index}")
        if role == "tool_response" and prev not in _INPUT_TOOL_PREDECESSORS:
            issues.append(f"tool_response without preceding tool_call at position {index}")
        if role != "system":
            prev = role
    answerable = [role for role in roles if role != "system"]
    if not answerable:
        issues.append("no conversation turns")
    elif answerable[-1] not in _OUTPUT_ROLES:
        issues.append(f"conversation ends on '{answerable[-1]}' (must end on an assistant-side turn)")
    if not any(role in _OUTPUT_ROLES for role in roles):
        issues.append("no assistant-side turn to train on")
    return issues


def to_ms_swift_messages(
    messages: list[dict[str, Any]],
    *,
    keep_intermediate_thinking: bool = False,
    clean: bool = True,
) -> list[dict[str, str]]:
    """Convert Teich messages to ms-swift native (agent) messages.

    By default ``<think>`` is kept only on the final assistant turn, matching the
    Qwen3 convention where historical reasoning is not present at inference time.
    Set ``keep_intermediate_thinking=True`` to retain reasoning on every assistant
    turn (often desirable for agent traces).

    With ``clean=True`` (default) the result is normalized into a sequence
    ms-swift can train on (see ``_clean_ms_swift_messages``); this can return an
    empty list when no trainable assistant turn remains.
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
    if clean:
        swift_messages = _clean_ms_swift_messages(swift_messages)
    return swift_messages


def to_ms_swift_row(
    row: dict[str, Any],
    *,
    keep_intermediate_thinking: bool = False,
    clean: bool = True,
    include_tools: bool = True,
    tools_as_json_string: bool = True,
) -> dict[str, Any]:
    """Convert one normalized Teich row into an ms-swift dataset row.

    With ``clean=True`` the messages are normalized for ms-swift trainability and
    may be empty when nothing trainable remains (such rows are dropped by
    ``convert_to_ms_swift``).
    """
    messages = row.get("messages") if isinstance(row, dict) else None
    if not isinstance(messages, list):
        messages = []
    swift_row: dict[str, Any] = {
        "messages": to_ms_swift_messages(
            messages,
            keep_intermediate_thinking=keep_intermediate_thinking,
            clean=clean,
        ),
    }
    tools = row.get("tools") if isinstance(row, dict) else None
    if include_tools and isinstance(tools, list) and tools:
        swift_row["tools"] = json.dumps(tools, ensure_ascii=False) if tools_as_json_string else tools
    return swift_row


def ms_swift_content_length(messages: list[dict[str, Any]]) -> int:
    """Total character length of all message ``content`` strings in a row."""
    return sum(len(message.get("content") or "") for message in messages)


def _round_cut_indices(messages: list[dict[str, Any]]) -> list[int]:
    # End of each user round: an assistant turn whose next turn is a user turn or end.
    cuts: list[int] = []
    total = len(messages)
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        next_role = messages[index + 1].get("role") if index + 1 < total else None
        if next_role is None or next_role == "user":
            cuts.append(index)
    return cuts


def _step_cut_indices(messages: list[dict[str, Any]]) -> list[int]:
    # End of each assistant-side run = each model action (a tool_call step or a final answer).
    cuts: list[int] = []
    total = len(messages)
    for index, message in enumerate(messages):
        if message.get("role") not in _OUTPUT_ROLES:
            continue
        next_role = messages[index + 1].get("role") if index + 1 < total else None
        if next_role not in _OUTPUT_ROLES:
            cuts.append(index)
    return cuts


_GRANULARITY_CUTS = {"round": _round_cut_indices, "step": _step_cut_indices}


def progressive_prefixes(
    messages: list[dict[str, Any]],
    granularity: str = "round",
) -> list[list[dict[str, Any]]]:
    """Split a (cleaned) ms-swift message list into accumulating prefixes.

    The prefixes accumulate and include the leading ``system`` message, so the
    last one is always the full conversation. This is for on-policy distillation
    where each example trains only its final turn. The cut points depend on
    ``granularity``:

    - ``"round"`` (default): one prefix per user round, each ending on that
      round's final-answer ``assistant`` turn. ``tool_call``/``tool_response``
      cycles stay inside their round::

          user -> output -> user -> output -> user -> output
          => [u->o,  u->o->u->o,  u->o->u->o->u->o]

    - ``"step"``: one prefix per model action -- each ending on an assistant-side
      run (a ``tool_call`` step or a final ``assistant`` answer), so intermediate
      tool calls become last-turn targets too. Parallel tool calls in one step
      stay together::

          user, tool_call, tool_response, tool_call, tool_response, assistant
          => [user..tool_call,
              user..tool_call->tool_response->tool_call,
              full]
    """
    try:
        cut_fn = _GRANULARITY_CUTS[granularity]
    except KeyError:
        raise ValueError(f"granularity must be one of {sorted(_GRANULARITY_CUTS)}; got {granularity!r}")
    return [messages[: cut + 1] for cut in cut_fn(messages)]


def convert_to_ms_swift(
    rows: list[dict[str, Any]],
    *,
    keep_intermediate_thinking: bool = False,
    clean: bool = True,
    drop_untrainable: bool = True,
    progressive: bool = False,
    granularity: str = "round",
    max_content_length: int | None = None,
    include_tools: bool = True,
    tools_as_json_string: bool = True,
) -> list[dict[str, Any]]:
    """Convert a list of normalized Teich rows into ms-swift dataset rows.

    With ``clean=True`` (default) each row is normalized for ms-swift trainability
    and, when ``drop_untrainable=True``, rows left without any trainable turn are
    dropped from the output.

    With ``progressive=True`` each conversation is expanded into accumulating
    prefixes (see ``progressive_prefixes``) for on-policy distillation that trains
    only the final turn of each example. ``granularity`` controls the cut points:
    ``"round"`` (one example per user round, ending on the final answer) or
    ``"step"`` (one example per model action, so intermediate ``tool_call`` steps
    become last-turn targets too).

    With ``max_content_length`` set, any output row whose total message-content
    character length exceeds it is dropped (applied after progressive expansion).
    """
    if progressive and granularity not in _GRANULARITY_CUTS:
        raise ValueError(f"granularity must be one of {sorted(_GRANULARITY_CUTS)}; got {granularity!r}")
    converted: list[dict[str, Any]] = []
    for row in rows:
        swift_row = to_ms_swift_row(
            row,
            keep_intermediate_thinking=keep_intermediate_thinking,
            clean=clean,
            include_tools=include_tools,
            tools_as_json_string=tools_as_json_string,
        )
        messages = swift_row["messages"]
        if clean and drop_untrainable and not messages:
            continue
        tools = swift_row.get("tools")
        variants = progressive_prefixes(messages, granularity=granularity) if progressive else [messages]
        for variant in variants:
            if not variant:
                continue
            if max_content_length is not None and ms_swift_content_length(variant) > max_content_length:
                continue
            new_row: dict[str, Any] = {"messages": variant}
            if tools is not None:
                new_row["tools"] = tools
            converted.append(new_row)
    return converted
