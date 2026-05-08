# Teich

Turn coding agent sessions into auditable supervised fine-tuning data.

---

Run `codex` or `pi` to capture raw coding-agent traces, or use `chat` mode to generate text-only training rows directly.

Load local folders, local files, or Hugging Face dataset repos; normalize them into `messages`/`tools`; and prepare pre-tokenized, audited SFT datasets with a Teich-owned data collator.

## ⚡ Quick Start

```bash
pip install teich
```

```bash
teich init my-project && cd my-project
teich generate -c config.yaml
```

Or use [astral-uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uvx teich init my-project && cd my-project
uvx teich generate -c config.yaml
```

> Be sure to edit your config.yaml and prompts.csv file as needed

## ⭐ What Teich Does

- **Trace-first data collection**: Run real coding agents and keep raw session traces as the source of truth
- **Multi-agent support**: Works with Codex, Pi, and a text-only `chat` mode
- **Structured conversion**: Converts traces into chat messages with tool calls, reasoning, tool results, metadata, and configured tool snapshots
- **SFT-ready preparation**: Applies tokenizer chat templates, masks labels, builds a Teich collator, and audits the dataset before training
- **Hugging Face integration**: Publishes dataset cards plus `tools.json`, and loads local or Hub datasets through one API

## 📥 Prerequisites

Requirements for agent trace generation:

- Docker
- OpenAI/OpenRouter API key (or local OpenAI-compatible endpoint)

`agent.provider: chat` does not require Docker. The Python utilities also work without Docker if you already have traces or structured JSONL datasets.

Training examples use your existing finetuning stack. For the TRL example below, install compatible versions of `transformers`, `trl`, and your model-loading stack separately.

## 🚀 Usage

### Generate traces from prompts

```bash
# Initialize project
teich init my-project
cd my-project

# Add prompts to prompts.csv, then:
export OPENAI_API_KEY=sk-...
teich generate -c config.yaml
```

Outputs:

- `codex` / `pi`: raw traces in `output/`, sandboxes in `sandbox/`, and a `README.md`
- `chat`: text-only JSONL training rows in `output/` and a dataset `README.md`

If `publish.repo_id` is configured, Teich also creates or updates the matching Hugging Face **dataset** repo and uploads the generated JSONL, README, and `tools.json` automatically.

If a long run is interrupted, use:

```bash
teich generate -c config.yaml --resume
```

Teich will scan existing outputs and skip prompts that already converted into completed training examples.

Prompt files can be CSV, text, JSONL/NDJSON, or JSON. JSONL is recommended for very long or multiline prompts.

### Generate a text-only chat dataset

```yaml
agent:
  provider: chat

model:
  model: gpt-4.1-mini

api:
  provider: openai
  wire_api: responses
```

Each generated JSONL line will look like:

```json
{"messages":[{"role":"system","content":"You are a helpful assistant","thinking":null},{"role":"user","content":"Hello","thinking":null},{"role":"assistant","content":"Hi!","thinking":"I should greet the user."}],"system":"You are a helpful assistant","prompt":"Hello","thinking":"I should greet the user.","response":"Hi!","model":"gpt-4.1-mini"}
```

### Prepare for training

```python
from teich import prepare_sft_dataset

prepared = prepare_sft_dataset(
    "badlogicgames/pi-mono",
    tokenizer,
    max_length=32768,
    chat_template_kwargs={"enable_thinking": True},
)

training_data = prepared.dataset
data_collator = prepared.collator
print(prepared.preview())
```

`prepare_sft_dataset` loads local folders, local files, or Hugging Face datasets; applies the tokenizer chat template; creates masked SFT labels; builds a Teich data collator; and runs dataset/collator audits by default.

### Train with TRL `SFTTrainer`

Teich prepares pre-tokenized `input_ids` / `attention_mask` / `labels` rows and returns the collator/config knobs needed to keep those labels intact inside TRL:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from teich import prepare_sft_dataset

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prepared = prepare_sft_dataset(
    "badlogicgames/pi-mono",
    tokenizer,
    max_length=32768,
    chat_template_kwargs={"enable_thinking": True},
)

trainer = SFTTrainer(
    model=model,
    train_dataset=prepared.dataset,
    data_collator=prepared.collator,
    args=SFTConfig(
        **prepared.sft_config_kwargs,
        output_dir="outputs",
        per_device_train_batch_size=1,
    ),
)
trainer.train()
```

`prepared.sft_config_kwargs` includes `dataset_kwargs={"skip_prepare_dataset": True}` so TRL does not re-template or overwrite Teich's labels.

### Advanced load and format flow

```python
from teich import format_and_mask, load_traces

tool_dataset = load_traces("badlogicgames/pi-mono", split="train")
chat_dataset = load_traces("./chat-output/chat.jsonl")

training_data = format_and_mask(
    [tool_dataset, chat_dataset],
    tokenizer,
    max_length=32768,
    chat_template_kwargs={"enable_thinking": True},
    strict=True,
)
```

### Manual tokenizer flow with `load_traces`

```python
from teich import load_traces

dataset = load_traces("./output")
example = dataset[0]

rendered = tokenizer.apply_chat_template(
    example["messages"],
    tools=example.get("tools") or [],
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=True,
)
tokenized = tokenizer(rendered, truncation=True, max_length=32768)
```

## 📋 Configuration

`config.yaml`:

```yaml
agent:
  provider: codex  # or pi or chat

model:
  model: codex-mini-latest
  approval_policy: never
  sandbox: danger-full-access

prompts_file: prompts.csv

output:
  traces_dir: ./output
  sandbox_dir: ./sandbox
  pretty_name: "My Agent Traces"

publish:
  repo_id: armand0e/my-dataset
  hf_token: hf_xxx
  private: false
```

Dataset tags are auto-generated from the provider and model:

- `codex` / `pi`: `agent-traces`, `<provider>`, `distillation`, `<model>`, `teich`
- `chat`: `conversational`, `distillation`, `teich`, `<model>`

If `publish.hf_token` is omitted, Teich also accepts `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `TEICH_HF_TOKEN` from the environment.

### Local providers (LM Studio, Ollama)

```bash
export TEICH_PROVIDER=LMstudio
export TEICH_MODEL=gemma-4
export TEICH_BASE_URL=http://localhost:1234/v1
export TEICH_API_KEY=llm

teich generate -c config.yaml
```

## 🏗️ Data Structure

Training examples include:

- `prompt`: initial task description
- `messages`: chat history (system, user, assistant, tool)
- `tools`: tool schemas used in the session
- `metadata`: session info, model, timestamps, and usage when available

Structured chat datasets can also include convenience top-level fields like:

- `system`
- `thinking`
- `response`
- `model`

Assistant messages capture:

- `content`: text response
- `reasoning_content`: chain-of-thought traces
- `tool_calls`: function calls with arguments

## 🔧 Python API

```python
from teich import (
    prepare_sft_dataset, # Load, format, mask, collate, and audit for SFT
    TeichDataCollator,   # Collator for pre-tokenized Teich SFT data
    load_traces,         # Load from folder, file, or HF dataset
    format_and_mask,     # Apply chat template + assistant masks
    preview_sft_example, # Preview supervised vs masked tokens
    Config,              # Load config.yaml
    TrainingExample,     # Typed training example
)
```

`README.md` is the package readme used for PyPI, so these examples are the canonical public package docs.

## 📦 Trace-First Workflow

Teich preserves the **raw agent session** as the source of truth:

1. **Collect**: Run agents on real tasks → raw `.jsonl` traces
2. **Inspect/Share**: Traces are human-readable and uploadable
3. **Convert**: Transform to structured examples when ready
4. **Prepare**: Apply model-specific chat templates, mask labels, collate, and audit for training

If you choose `agent.provider: chat`, Teich skips the trace-preservation step and writes structured text-only JSONL rows directly.

This means you can:

- Re-convert with different logic later
- Share raw traces before releasing training data
- Train on the same sessions with different model templates

## 🛠️ Development

```bash
uv pip install -e ".[dev]"
uv run pytest --ignore=tests/test_integration.py -q
```

## 📌 Status

Teich is **alpha**. The core workflow is stable and usable. APIs may evolve as more agent types and training workflows are added.

## 📄 License

Apache-2.0
