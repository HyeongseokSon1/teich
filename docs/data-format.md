# Data Format

Teich normalizes supported sources into structured training examples.

To write this normalized format as standalone JSONL from raw or extracted traces, run:

```bash
teich convert data --out teich-training.jsonl
```

Core fields:

- `prompt`: initial task description
- `follow_up_prompts`: optional additional user turns after the initial prompt
- `messages`: chat history
- `tools`: tool schemas available to the session, including tools that were not called
- `metadata`: session info, model, timestamps, usage, and provenance when available

## Messages

`messages` follows an OpenAI-style chat shape:

```json
[
  {"role": "user", "content": "Build a todo app"},
  {
    "role": "assistant",
    "content": "I will inspect the project.",
    "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "Read",
          "arguments": {"file_path": "README.md"}
        }
      }
    ]
  },
  {"role": "tool", "tool_call_id": "call_1", "name": "Read", "content": "..."}
]
```

Supported roles include:

- `system`
- `developer`
- `user`
- `assistant`
- `tool`

Assistant messages can include:

- `content`: text response
- `reasoning_content`: reasoning traces when the source provides them
- `tool_calls`: function calls with arguments

Some providers split a single model turn across multiple native events. Teich normalizes those fragments so semantic order is:

1. `reasoning_content`
2. optional assistant `content`
3. `tool_calls`

Reasoning that arrives after assistant text or a tool-call fragment is moved back in front of the output it explains.

## Tools

`tools` contains function schemas available to the session:

```json
[
  {
    "type": "function",
    "function": {
      "name": "Read",
      "description": "Read a file.",
      "parameters": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string"}
        },
        "required": ["file_path"],
        "additionalProperties": true
      }
    }
  }
]
```

Teich preserves configured tool snapshots where the source provides them. This matters because a training row can need the tool schema even when the model did not call that tool.

Tool schema sources include:

- configured tools embedded in generated traces
- generated dataset `README.md` snapshots
- `tools.json` snapshots
- native provider tool declarations
- fallback inference from observed tool calls when no explicit schema exists

## Metadata

Common metadata keys:

- `source_file`
- `source_line`
- `session_id`
- `trace_type`
- `model_provider`
- `model`
- `cwd`
- `cli_version`
- `turn_count`
- `usage`
- `total_cost_usd`
- `first_message_timestamp`

When the source format exposes per-message timestamps, converted rows include `metadata.first_message_timestamp` from the first timestamp-bearing source event that becomes a user message. It is not synthesized from session-start metadata.

## Native Claude Code Context

Native Claude Code and Claude Desktop traces can contain runtime context that the model saw but that is not ordinary user text.

Teich preserves this as masked `system` messages and mirrors it into `metadata.system_prompt`.

Examples include:

- Claude Desktop skill listings
- MCP instruction deltas
- deferred tool declarations
- command permission context
- date changes
- hook context
- away summaries
- session recaps

Local slash-command artifacts such as `/model` are filtered. `/goal` contributes the actual user goal text. Queued prompts become real user turns.

Advertised native Claude Code / Claude Desktop tools receive schemas even when a tool is only declared through deferred-tool context.

## Structured Chat Rows

The `chat` provider writes structured training rows directly instead of raw traces.

Single-turn rows can include:

- `messages`
- `prompt`
- `thinking`
- `response`
- `model`

Multi-turn follow-up rows can also include:

- `follow_up_prompts`
- `responses`
- final `response`

`system` is prompt-specific when provided. If a prompt row does not include `system`, Teich does not inject a default system prompt.

## ms-swift Output Format

To train with the [ms-swift](https://github.com/modelscope/ms-swift) framework (e.g. Qwen3-style
reasoning models), convert the normalized Teich rows into ms-swift's native dataset shape:

```bash
teich convert data --format ms-swift --out qwen3-swift.jsonl
```

This runs the same `trace -> message` conversion and then applies a `message -> message2` step
(`teich.swift.convert_to_ms_swift`). The differences from the Teich format are:

- `reasoning_content` is inlined into assistant `content` as `<think>\n...\n</think>\n\n<answer>`,
  matching the Qwen3 best-practice format. Pair this with `--loss_scale ignore_empty_think` in
  ms-swift so assistant turns without reasoning are not penalized.
- assistant `tool_calls` become standalone `tool_call` role messages whose `content` is a JSON
  string `{"name": ..., "arguments": {...}}` (`arguments` is a JSON object).
- `tool` role messages become `tool_response` role messages.
- the `developer` role is mapped to `system`.
- `tools` is serialized to a JSON string (OpenAI-style function schemas, unchanged).
- only `messages` and `tools` are emitted; `prompt`, `follow_up_prompts`, and `metadata` are dropped.

```json
{"messages": [{"role": "user", "content": "Build a todo app"}, {"role": "assistant", "content": "<think>\nCheck the README first.\n</think>\n\nI will inspect the project."}, {"role": "tool_call", "content": "{\"name\": \"Read\", \"arguments\": {\"file_path\": \"README.md\"}}"}, {"role": "tool_response", "content": "..."}, {"role": "assistant", "content": "Done."}], "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"Read\", \"parameters\": {\"type\": \"object\"}}}]"}
```

By default `<think>` is kept only on the final assistant turn, mirroring how Qwen3 drops historical
reasoning at inference. Agent traces often carry their reasoning on the intermediate (tool-deciding)
turns rather than the final answer, so pass `--keep-intermediate-thinking` to retain `<think>` on
every assistant turn when that reasoning is the signal you want to train on.

### Trainability cleanup

ms-swift requires user-side (`user`/`tool_response`) and assistant-side (`assistant`/`tool_call`)
turns to alternate, with at most one leading `system` message, ending on an `assistant` turn. Real
agent traces violate this (multiple runtime `system` messages, back-to-back user turns, tool results
with no answer before the user speaks again, conversations ending mid tool call). Cleanup is on by
default (`--no-clean` to disable) and normalizes each row:

- all `system` messages are merged into one leading `system`;
- consecutive `user` (and consecutive `assistant`) messages are merged;
- orphan `tool_response` messages and unanswered tool rounds before a new `user` turn are dropped;
- trailing incomplete turns are trimmed so the row ends on the assistant target;
- rows left with no trainable assistant turn are dropped from the output.

Valid parallel (`tool_call` x N then `tool_response` x N) and sequential (`tool_response` then
`tool_call`) tool patterns are preserved. `teich.swift.validate_ms_swift_messages(messages)` returns
the list of remaining trainability problems for a row (empty means OK).

## Incomplete Traces

Rows ending on a tool result are incomplete without a follow-up assistant turn.

`load_traces()` drops those rows by default. Pass `drop_incomplete_traces=False` only when you intentionally want to inspect or repair them.

## Generated Dataset Cards

Generated datasets include a `README.md` with summary metadata and, when available, embedded tool-schema snapshots.

Generated dataset guidance is produced by `src/teich/trace_readme.py`, so behavior changes should update both top-level docs and that template.
