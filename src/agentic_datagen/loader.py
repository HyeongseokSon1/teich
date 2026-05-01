from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from huggingface_hub import snapshot_download

from .converter import convert_traces_to_training_data


def _trace_directory(root: Path, split: str | None) -> Path:
    if split:
        candidate = root / split
        if candidate.is_dir():
            return candidate
    return root


def load_traces(
    source: str | Path,
    split: str | None = "train",
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> Dataset:
    source_path = Path(source)
    if source_path.exists():
        root = source_path
    else:
        root = Path(
            snapshot_download(
                repo_id=str(source),
                repo_type="dataset",
                revision=revision,
                token=token,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_dir=str(local_dir) if local_dir is not None else None,
                allow_patterns=["*.jsonl", "**/*.jsonl"],
            )
        )
    traces_dir = _trace_directory(root, split)
    rows = convert_traces_to_training_data(traces_dir)
    if not rows:
        location = traces_dir if traces_dir != root else root
        if split and traces_dir == root:
            raise ValueError(f"No trace files found in {location} for split '{split}'.")
        raise ValueError(f"No trace files found in {location}.")
    return Dataset.from_list(rows)
