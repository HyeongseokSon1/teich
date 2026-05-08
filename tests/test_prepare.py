from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from teich import TeichDataCollator, prepare_sft_dataset


class TinyChatTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._reverse_vocab: dict[int, str] = {}

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
        rendered = "".join(f"<{message['role']}>{message.get('content', '')}</{message['role']}>" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self(rendered)
        return rendered

    def __call__(self, text, add_special_tokens=False, return_attention_mask=True):
        input_ids: list[int] = []
        for character in text:
            token_id = self._vocab.setdefault(character, len(self._vocab) + 1)
            self._reverse_vocab[token_id] = character
            input_ids.append(token_id)
        output = {"input_ids": input_ids}
        if return_attention_mask:
            output["attention_mask"] = [1] * len(input_ids)
        return output

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._reverse_vocab[token_id] for token_id in token_ids)


def _dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "tools": [],
            }
        ]
    )


def _write_structured_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "prompt": "hello",
                "response": "world",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_prepare_sft_dataset_accepts_dataset_and_returns_training_artifacts():
    tokenizer = TinyChatTokenizer()
    prepared = prepare_sft_dataset(
        _dataset(),
        tokenizer,
        audit=True,
        verbose=False,
        collator=TeichDataCollator(tokenizer=tokenizer, return_tensors=None),
    )

    assert prepared.dataset.num_rows == 1
    assert prepared.dataset_report is not None
    assert prepared.dataset_report.ok
    assert prepared.batch_report is not None
    assert prepared.batch_report.ok
    assert prepared.sft_config_kwargs == {"dataset_kwargs": {"skip_prepare_dataset": True}, "dataset_num_proc": 1}
    assert "world</assistant>" in prepared.preview()


def test_prepare_sft_dataset_loads_local_source(tmp_path: Path):
    dataset_file = tmp_path / "chat.jsonl"
    _write_structured_dataset(dataset_file)
    tokenizer = TinyChatTokenizer()

    prepared = prepare_sft_dataset(
        dataset_file,
        tokenizer,
        split=None,
        audit=True,
        verbose=False,
        collator=TeichDataCollator(tokenizer=tokenizer, return_tensors=None),
    )

    assert prepared.dataset.num_rows == 1
    assert prepared.batch_report is not None
    assert prepared.batch_report.ok


def test_prepare_sft_dataset_raises_when_audit_fails():
    tokenizer = TinyChatTokenizer()
    empty_target_dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                ],
                "tools": [],
            }
        ]
    )

    with pytest.raises(ValueError, match="fully masked"):
        prepare_sft_dataset(
            empty_target_dataset,
            tokenizer,
            audit=True,
            verbose=False,
            collator=TeichDataCollator(tokenizer=tokenizer, return_tensors=None),
        )
