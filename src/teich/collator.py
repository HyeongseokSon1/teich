from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TeichDataCollator:
    tokenizer: Any | None = None
    pad_token_id: int | None = None
    label_pad_token_id: int = -100
    return_tensors: str | None = "pt"
    padding_side: str | None = None

    def __post_init__(self) -> None:
        if self.pad_token_id is None and self.tokenizer is not None:
            self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if self.pad_token_id is None:
                self.pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if self.pad_token_id is None:
            raise ValueError("TeichDataCollator requires pad_token_id or tokenizer.pad_token_id/tokenizer.eos_token_id")
        if self.padding_side is None and self.tokenizer is not None:
            self.padding_side = getattr(self.tokenizer, "padding_side", None)
        if self.padding_side is None:
            self.padding_side = "right"
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            return self._tensorize({"input_ids": [], "attention_mask": [], "labels": []})
        for index, example in enumerate(examples):
            missing = {"input_ids", "attention_mask", "labels"}.difference(example)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"example {index} is missing required columns: {names}")
        max_length = max(len(_as_list(example["input_ids"])) for example in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for example in examples:
            input_ids = _as_list(example["input_ids"])
            attention_mask = _as_list(example["attention_mask"])
            labels = _as_list(example["labels"])
            if not (len(input_ids) == len(attention_mask) == len(labels)):
                raise ValueError("input_ids, attention_mask, and labels must have equal lengths before collation")
            pad_length = max_length - len(input_ids)
            batch["input_ids"].append(_pad(input_ids, pad_length, self.pad_token_id, self.padding_side))
            batch["attention_mask"].append(_pad(attention_mask, pad_length, 0, self.padding_side))
            batch["labels"].append(_pad(labels, pad_length, self.label_pad_token_id, self.padding_side))
        return self._tensorize(batch)

    def _tensorize(self, batch: dict[str, list[list[int]]]) -> dict[str, Any]:
        if self.return_tensors is None:
            return batch
        if self.return_tensors != "pt":
            raise ValueError("TeichDataCollator currently supports return_tensors='pt' or None")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("return_tensors='pt' requires torch to be installed") from exc
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _as_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _pad(values: list[int], pad_length: int, pad_value: int, padding_side: str | None) -> list[int]:
    padding = [pad_value] * pad_length
    if padding_side == "left":
        return padding + values
    return values + padding
