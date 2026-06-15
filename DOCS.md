# Documentation

Teich documentation now lives in focused pages under [`docs/`](docs/index.md).

- [Generation](docs/generation.md): create new Codex, Pi, Claude Code, Hermes, or chat datasets, or extract existing local sessions with `teich extract`.
- [CLI Reference](docs/cli.md): command usage for `init`, `generate`, `extract`, `convert`, `anonymize`, `studio`, and `pool`.
- [Teich Studio](docs/studio.md): configure projects, manage prompts, run batches, and save interactive sessions from a browser.
- [Preparing Data](docs/prepare-data.md): load local files, folders, Hugging Face datasets, `datasets.Dataset` objects, and source mixes.
- [Training](docs/training.md): use Teich with TRL / Unsloth and apply response-only labels with `mask_data()`.
- [Data Format](docs/data-format.md): understand `messages`, `tools`, metadata, native Claude context, and structured chat rows.
- [Python API](docs/python-api.md): public functions and validation helpers.
- [Pipeline Flow](docs/pipeline.md): generation, preparation, and masking diagrams.

The top-level [`README.md`](README.md) is the package/PyPI front page and intentionally stays short.
