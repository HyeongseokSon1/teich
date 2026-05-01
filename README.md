# Agentic Datagen v2

`v2/` is the experimental trace-first package for collecting raw agent sessions and converting them into training-ready data.

## What it does today

- Runs Codex and Pi in a shared Docker runtime with `uv`, `npm`, `@openai/codex`, and `@mariozechner/pi-coding-agent`
- Configures Codex through a mounted `CODEX_HOME/config.toml`
- Configures Pi through an isolated mounted `~/.pi/agent/settings.json`
- Exports raw session traces from mounted Codex and Pi session directories
- Writes a trace-folder `README.md` for upload
- Exposes Python conversion helpers for training data preparation

## Usage

```bash
# Initialize a project
uvx agentic-datagen init my-project
cd my-project

# Run with the configured agent provider and model settings
uvx agentic-datagen generate -c config.yaml
```

## Local OSS providers

If you want Codex to talk to a local provider like LM Studio or Ollama, set the provider in config or env:

```powershell
$env:AGENTIC_DATAGEN_PROVIDER='LMstudio'
$env:AGENTIC_DATAGEN_MODEL='gemma-4'
$env:AGENTIC_DATAGEN_API_KEY='llm'
$env:AGENTIC_DATAGEN_BASE_URL='http://localhost:1234/v1'
python -m agentic_datagen generate -c test_run/config.yaml
```

`v2` maps `LMstudio` and `ollama` onto Codex's native `--oss --local-provider ...` flow.

## Configuration model

Important fields in `config.yaml`:

```yaml
agent:
  provider: codex  # or pi

model:
  model: codex-mini-latest
  approval_policy: never
  sandbox: danger-full-access
  reasoning_effort: null

api:
  provider: openai
  base_url: null
  api_key: null
```

Legacy `model.approval_mode` is still accepted and normalized internally.

## Python conversion API

```python
from pathlib import Path
from agentic_datagen import convert_traces_to_training_data

examples = convert_traces_to_training_data(Path("./output"))
```

The converter currently maps example-style raw traces into message/tool records with:

- system/developer instructions
- user messages
- assistant messages
- `reasoning_content`
- tool calls
- tool results

## Development

```bash
uv pip install -e ".[dev]"
pytest tests/test_config.py tests/test_cli.py tests/test_runner.py -q
```

## Architecture

- **Shared Docker runtime**: container image includes Node.js, `uv`, `uvx`, `@openai/codex`, and `@mariozechner/pi-coding-agent`
- **Isolated Pi config**: Pi runs with a mounted per-run `~/.pi/agent` directory inside the container
- **Codex config**: generated `config.toml` under a mounted `CODEX_HOME`
- **Session export**: raw JSONL sessions are copied from mounted Codex or Pi session storage into the user output directory
- **Upload-first output**: traces are preserved in raw form before later conversion
- **Provider-aware boundary**: `agent.provider` selects either the Codex or Pi raw-trace path

## Project Structure

```text
v2/
├── docker/
│   └── codex-runtime.Dockerfile
├── src/agentic_datagen/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── converter.py
│   ├── runner.py
│   └── trace_readme.py
└── tests/
    ├── test_cli.py
    ├── test_config.py
    └── test_runner.py
