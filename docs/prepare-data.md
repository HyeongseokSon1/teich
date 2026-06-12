# Preparing Data

Use `prepare_data()` when you already have data and want it rendered for a target tokenizer.

Supported sources:

- local JSONL files
- folders of JSONL traces
- Hugging Face dataset repos
- already-loaded `datasets.Dataset` objects
- source mixes with explicit ratios

## Basic Usage

```python
from teich import prepare_data

train_dataset = prepare_data(
    "TeichAI/Claude-Opus-4.6-Reasoning-887x",
    tokenizer,
    max_length=32768,
    oversized_policy="trim_followups",
    tokenize=True,
    chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
)
```

`prepare_data()` returns rendered `text`, Teich span metadata, and optionally `input_ids` / `attention_mask`.

Call [Training](training.md)'s `mask_data()` step after constructing your trainer to convert those spans into token-level labels.

## Reports and Provenance

For audit-friendly preparation, request a report and keep provenance columns:

```python
train_dataset, prep_report = prepare_data(
    "TeichAI/Claude-Opus-4.6-Reasoning-887x",
    tokenizer,
    max_length=32768,
    oversized_policy="drop",
    preserve_columns=True,
    return_report=True,
    tokenize=True,
)

print(prep_report.max_token_length)
print(prep_report.oversized_rows[:3])
```

`PrepareReport` includes dropped rows, oversized rows, trimmed rows, token lengths, max token length, kept-row ids, and returned row count.

Original columns are removed after formatting unless `preserve_columns=True` or an explicit list is passed. The default provenance set is `source`, `metadata`, `raw_index`, and `source_key`.

## Oversized Rows

When `max_length` is set, use one of:

- `oversized_policy="drop"`: drop oversized rows
- `oversized_policy="trim_followups"`: for multi-turn rows, remove the final user follow-up and everything after it before dropping the whole row
- `oversized_policy="error"`: raise instead of filtering

The older `drop_oversized_examples` and `trim_oversized_followups` flags still work as compatibility aliases, but `oversized_policy` is the preferred API.

## Mixed Sources

Mix datasets with true ratios:

```python
train_dataset = prepare_data(
    {
        "max_examples": 1000,
        "reasoning-agent": {
            "source": "badlogicgames/pi-mono",
            "percentage": 80,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        },
        "instruct-chat": {
            "source": "TeichAI/polaris-alpha-1000x",
            "percentage": 20,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
        },
    },
    tokenizer,
    max_length=32768,
    oversized_policy="trim_followups",
    tokenize=True,
    chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
)
```

`percentage`, `proportion`, and `weight` are treated as true ratios.

If one source cannot fill its share after filtering or context-window drops, Teich scales the total row count down instead of silently changing the realized mix.

Global `chat_template_kwargs` are the default for every source. A source-level `chat_template_kwargs` mapping overrides those keys for that dataset only.

You can also pass a simple list of sources:

```python
train_dataset = prepare_data(
    ["username/chat-traces", "username/tool-traces"],
    tokenizer,
    max_length=32768,
    oversized_policy="trim_followups",
    tokenize=True,
    chat_template_kwargs={"enable_thinking": True},
)
```

## Tool Validation

Teich can fail early on undeclared or malformed tool calls:

```python
train_dataset = prepare_data(
    "./output",
    tokenizer,
    validate_tools=True,
    strict=True,
)
```

`validate_tools=True` checks tool-call names and required arguments against each row's declared `tools`.

## Plain Next-Token Training

If you do not want Teich response-only labels, turn masking metadata off:

```python
train_dataset = prepare_data(
    "./data.jsonl",
    tokenizer,
    teich_masking=False,
)
```

Rows contain rendered `text` only, plus tokens if `tokenize=True`.

## Manual Flow with `load_traces`

Use `load_traces()` directly when you want to own chat-template rendering, filtering, tokenization, label masking, and packing policy.

```python
from teich import load_traces, row_fits_context, validate_tool_calls

dataset = load_traces("./output")
example = dataset[0]

validate_tool_calls(example).raise_for_errors()
if not row_fits_context(example, tokenizer, 32768, {"enable_thinking": True}):
    raise ValueError("example does not fit the target context window")

rendered = tokenizer.apply_chat_template(
    example["messages"],
    tools=example.get("tools") or [],
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=True,
)
tokenized = tokenizer(rendered, truncation=True, max_length=32768)
```

`load_traces()` drops rows that end on a tool result by default, because those traces are incomplete without a follow-up assistant turn. Pass `drop_incomplete_traces=False` only when you intentionally want to inspect or repair those rows.

## Preflight Helpers

- `row_fits_context(row, tokenizer, max_length, chat_template_kwargs)`: render and measure one row
- `validate_tool_calls(row)`: check declared tool names and required arguments
- `trace_is_complete(row)`: flag rows that end on a tool result
- `detect_trace_type(events)`: identify supported raw trace events, or return `None`
