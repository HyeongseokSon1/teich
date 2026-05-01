from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingExample:
    source_file: Path
    prompt: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "messages": self.messages,
            "tools": self.tools,
            "metadata": self.metadata,
        }


def _first_text_block(content_blocks: Any) -> str:
    if isinstance(content_blocks, str):
        return content_blocks.strip()
    if not isinstance(content_blocks, list):
        return ""
    parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"input_text", "output_text", "text"}:
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts).strip()


def _has_same_system_message(messages: list[dict[str, Any]], content: str) -> bool:
    return any(
        message.get("role") == "system" and message.get("content") == content
        for message in messages
    )


def _pi_reasoning_content(content_blocks: Any) -> str | None:
    if not isinstance(content_blocks, list):
        return None
    parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "thinking":
            continue
        thinking = block.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            parts.append(thinking.strip())
    result = "\n\n".join(parts).strip()
    return result or None


def _tool_result_content_text(payload: dict[str, Any]) -> str:
    return _first_text_block(payload.get("content"))


def _is_tool_not_found_result(tool_name: str | None, payload: dict[str, Any]) -> bool:
    content = _tool_result_content_text(payload).strip()
    if tool_name:
        return content == f"Tool {tool_name} not found"
    return content == "Tool  not found"


def _reasoning_summary(payload: dict[str, Any]) -> str | None:
    summary = payload.get("summary")
    parts: list[str] = []
    if isinstance(summary, list):
        for item in summary:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    result = "\n\n".join(parts).strip()
    if result:
        return result

    content = payload.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "reasoning_text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    result = "\n\n".join(parts).strip()
    return result or None


def _parse_function_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments if arguments is not None else {}
    stripped = arguments.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return arguments


def _schema_identity(schema: dict[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, ensure_ascii=False)


def _infer_schema_from_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        item_schemas = [_infer_schema_from_value(item) for item in value]
        schema: dict[str, Any] = {"type": "array"}
        if item_schemas:
            schema["items"] = _merge_schemas(item_schemas)
        return schema
    if isinstance(value, dict):
        return _infer_tool_parameters_schema([value])
    return {}


def _merge_object_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    properties_by_name: dict[str, list[dict[str, Any]]] = {}
    required_sets: list[set[str]] = []
    additional_properties = False
    for schema in schemas:
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name, value in properties.items():
                if isinstance(value, dict):
                    properties_by_name.setdefault(name, []).append(value)
        required = schema.get("required")
        if isinstance(required, list):
            required_sets.append({item for item in required if isinstance(item, str)})
        else:
            required_sets.append(set())
        if schema.get("additionalProperties", True) is not False:
            additional_properties = True
    merged: dict[str, Any] = {
        "type": "object",
        "properties": {
            name: _merge_schemas(property_schemas)
            for name, property_schemas in sorted(properties_by_name.items())
        },
        "additionalProperties": additional_properties,
    }
    if required_sets:
        required = sorted(set.intersection(*required_sets))
        if required:
            merged["required"] = required
    return merged


def _merge_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for schema in schemas:
        if not schema:
            continue
        identity = _schema_identity(schema)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(schema)
    if not unique:
        return {}
    if len(unique) == 1:
        return unique[0]
    schema_types = {schema.get("type") for schema in unique if isinstance(schema.get("type"), str)}
    if schema_types == {"object"}:
        return _merge_object_schemas(unique)
    if schema_types == {"array"}:
        item_schemas = [schema.get("items") for schema in unique if isinstance(schema.get("items"), dict)]
        merged: dict[str, Any] = {"type": "array"}
        if item_schemas:
            merged["items"] = _merge_schemas(item_schemas)
        return merged
    return {"anyOf": unique}


def _infer_tool_parameters_schema(argument_samples: list[Any]) -> dict[str, Any]:
    dict_samples = [sample for sample in argument_samples if isinstance(sample, dict)]
    if not dict_samples:
        return {"type": "object", "properties": {}, "additionalProperties": True}
    properties: dict[str, dict[str, Any]] = {}
    all_keys = sorted({key for sample in dict_samples for key in sample})
    for key in all_keys:
        observed = [_infer_schema_from_value(sample[key]) for sample in dict_samples if key in sample]
        properties[key] = _merge_schemas(observed)
    required = sorted(set.intersection(*(set(sample.keys()) for sample in dict_samples))) if dict_samples else []
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }
    if required:
        schema["required"] = required
    return schema


def _parse_tool_descriptions_from_text(text: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    in_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not in_section:
            if line == "Available tools:":
                in_section = True
            continue
        if not line:
            if descriptions:
                break
            continue
        if not line.startswith("- "):
            if descriptions:
                break
            continue
        name, separator, description = line[2:].partition(":")
        tool_name = name.strip()
        tool_description = description.strip()
        if separator and tool_name and tool_description:
            descriptions[tool_name] = tool_description
    return descriptions


def _normalize_role(role: str) -> str:
    if role == "developer":
        return "system"
    return role


def _build_tool_entry(name: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    function: dict[str, Any] = {"name": name}
    if isinstance(schema, dict) and schema:
        function.update(schema)
    return {"type": "function", "function": function}


def _detect_trace_type(events: list[dict[str, Any]]) -> str:
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type in {"session_meta", "turn_context", "response_item", "event_msg"}:
            return "codex"
        if event_type in {
            "session",
            "message",
            "session_info",
            "model_change",
            "thinking_level_change",
            "compaction",
            "branch_summary",
            "custom",
            "custom_message",
            "label",
        }:
            return "pi"
    return "codex"


def load_trace_file(trace_file: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with trace_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _convert_codex_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    pending_reasoning: str | None = None
    tool_names: set[str] = set()
    tool_schemas: dict[str, dict[str, Any]] = {}
    tool_argument_samples: dict[str, list[Any]] = {}
    tool_descriptions: dict[str, str] = {}
    tool_call_names: dict[str, str] = {}
    session_meta: dict[str, Any] = {}
    turn_contexts: list[dict[str, Any]] = []
    prompt = ""

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        payload = event.get("payload")
        if event_type == "session_meta" and isinstance(payload, dict):
            session_meta = payload
            base_instructions = payload.get("base_instructions")
            if isinstance(base_instructions, dict):
                text = base_instructions.get("text")
                if isinstance(text, str) and text.strip() and not _has_same_system_message(messages, text):
                    messages.append({"role": "system", "content": text})
                    tool_descriptions.update(_parse_tool_descriptions_from_text(text))
            continue
        if event_type == "turn_context" and isinstance(payload, dict):
            turn_contexts.append(payload)
            continue
        if event_type != "response_item" or not isinstance(payload, dict):
            continue

        payload_type = payload.get("type")
        if payload_type == "reasoning":
            pending_reasoning = _reasoning_summary(payload)
            continue

        if payload_type == "message":
            role = payload.get("role")
            if not isinstance(role, str):
                continue
            normalized_role = _normalize_role(role)
            content = _first_text_block(payload.get("content"))
            if normalized_role == "user" and content and not prompt:
                prompt = content
            message: dict[str, Any] = {
                "role": normalized_role,
                "content": content,
            }
            if normalized_role == "assistant" and pending_reasoning:
                message["reasoning_content"] = pending_reasoning
                pending_reasoning = None
            messages.append(message)
            continue

        if payload_type == "function_call":
            name = payload.get("name")
            call_id = payload.get("call_id")
            if not isinstance(name, str) or not isinstance(call_id, str):
                continue
            tool_names.add(name)
            tool_call_names[call_id] = name
            arguments = _parse_function_arguments(payload.get("arguments"))
            tool_argument_samples.setdefault(name, []).append(arguments)
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
            if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                messages[-1]["tool_calls"].append(tool_call)
            else:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call],
                }
                if pending_reasoning:
                    assistant_message["reasoning_content"] = pending_reasoning
                    pending_reasoning = None
                messages.append(assistant_message)
            continue

        if payload_type == "function_call_output":
            call_id = payload.get("call_id")
            if not isinstance(call_id, str):
                continue
            tool_name = tool_call_names.get(call_id)
            if tool_name:
                tool_names.add(tool_name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name or "unknown_tool",
                    "content": str(payload.get("output") or ""),
                }
            )
            continue

        if payload_type == "tool_schema":
            name = payload.get("name")
            schema = payload.get("schema")
            if isinstance(name, str) and isinstance(schema, dict):
                tool_names.add(name)
                tool_schemas[name] = schema

    tools = []
    for name in sorted(tool_names):
        schema = dict(tool_schemas.get(name) or {})
        if name in tool_descriptions and "description" not in schema:
            schema["description"] = tool_descriptions[name]
        if "parameters" not in schema:
            schema["parameters"] = _infer_tool_parameters_schema(tool_argument_samples.get(name, []))
        tools.append(_build_tool_entry(name, schema))
    if not prompt:
        prompt = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )

    metadata = {
        "source_file": trace_file.name,
        "session_id": session_meta.get("id") or trace_file.stem,
        "trace_type": session_meta.get("source") or "codex",
        "model_provider": session_meta.get("model_provider"),
        "cwd": session_meta.get("cwd"),
        "cli_version": session_meta.get("cli_version"),
        "turn_count": len(turn_contexts),
    }
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def _convert_pi_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    tool_argument_samples: dict[str, list[Any]] = {}
    tool_descriptions: dict[str, str] = {}
    session_header: dict[str, Any] = {}
    model_change: dict[str, Any] = {}
    session_names: list[str] = []
    thinking_level: str | None = None
    prompt = ""
    invalid_tool_call_ids: set[str] = set()

    for event in events:
        if not isinstance(event, dict) or event.get("type") != "message":
            continue
        payload = event.get("message")
        if not isinstance(payload, dict) or payload.get("role") != "toolResult":
            continue
        tool_call_id = payload.get("toolCallId")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        tool_name = payload.get("toolName") if isinstance(payload.get("toolName"), str) else None
        if _is_tool_not_found_result(tool_name, payload):
            invalid_tool_call_ids.add(tool_call_id)

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "session":
            session_header = event
            continue
        if event_type == "model_change":
            model_change = event
            continue
        if event_type == "thinking_level_change":
            level = event.get("thinkingLevel")
            if isinstance(level, str) and level.strip():
                thinking_level = level.strip()
            continue
        if event_type == "session_info":
            name = event.get("name")
            if isinstance(name, str) and name.strip():
                session_names.append(name.strip())
            continue
        if event_type != "message":
            continue

        payload = event.get("message")
        if not isinstance(payload, dict):
            continue
        role = payload.get("role")
        if not isinstance(role, str):
            continue

        if role == "toolResult":
            tool_call_id = payload.get("toolCallId")
            if not isinstance(tool_call_id, str):
                continue
            if tool_call_id in invalid_tool_call_ids:
                continue
            tool_name = payload.get("toolName")
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name or "unknown_tool",
                "content": _first_text_block(payload.get("content")),
            }
            if payload.get("isError") is True:
                tool_message["is_error"] = True
            messages.append(tool_message)
            continue

        normalized_role = _normalize_role(role)
        content_blocks = payload.get("content")
        content = _first_text_block(content_blocks)

        if role == "developer" and content:
            tool_descriptions.update(_parse_tool_descriptions_from_text(content))

        if normalized_role == "user":
            if content and not prompt:
                prompt = content
            messages.append({"role": normalized_role, "content": content})
            continue

        message: dict[str, Any] = {
            "role": normalized_role,
            "content": content,
        }
        if normalized_role == "assistant":
            reasoning_content = _pi_reasoning_content(content_blocks)
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "toolCall":
                        continue
                    tool_call_id = block.get("id")
                    tool_name = block.get("name")
                    if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                        continue
                    if not tool_call_id or not tool_name or tool_call_id in invalid_tool_call_ids:
                        continue
                    tool_names.add(tool_name)
                    arguments = _parse_function_arguments(block.get("arguments"))
                    tool_argument_samples.setdefault(tool_name, []).append(arguments)
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        }
                    )
            if tool_calls:
                message["tool_calls"] = tool_calls
            if not message["content"] and "reasoning_content" not in message and "tool_calls" not in message:
                continue
        elif not content:
            continue
        messages.append(message)

    tools = [
        _build_tool_entry(
            name,
            {
                **({"description": tool_descriptions[name]} if name in tool_descriptions else {}),
                "parameters": _infer_tool_parameters_schema(tool_argument_samples.get(name, [])),
            },
        )
        for name in sorted(tool_names)
    ]
    if not prompt:
        prompt = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )

    metadata: dict[str, Any] = {
        "source_file": trace_file.name,
        "session_id": session_header.get("id") or trace_file.stem,
        "trace_type": "pi",
        "model_provider": model_change.get("provider"),
        "model": model_change.get("modelId"),
        "cwd": session_header.get("cwd"),
        "cli_version": None,
        "turn_count": sum(1 for message in messages if message.get("role") == "user"),
    }
    if thinking_level:
        metadata["thinking_level"] = thinking_level
    if session_names:
        metadata["session_names"] = session_names
        metadata["session_name"] = session_names[-1]
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def convert_trace_to_training_example(trace_file: Path) -> TrainingExample:
    events = load_trace_file(trace_file)
    trace_type = _detect_trace_type(events)
    if trace_type == "pi":
        return _convert_pi_trace_to_training_example(trace_file, events)
    return _convert_codex_trace_to_training_example(trace_file, events)


def convert_traces_to_training_data(traces_dir: Path | str) -> list[dict[str, Any]]:
    directory = Path(traces_dir)
    trace_files = sorted(path for path in directory.glob("*.jsonl") if path.is_file())
    return [convert_trace_to_training_example(path).to_dict() for path in trace_files]
