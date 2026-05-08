from __future__ import annotations

import pytest

from teich import TeichDataCollator


class PadTokenizer:
    pad_token_id = 0
    eos_token_id = 2


class EosOnlyTokenizer:
    pad_token_id = None
    eos_token_id = 2


class LeftPadTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"


def test_teich_data_collator_pads_labels_with_ignore_index():
    collator = TeichDataCollator(tokenizer=PadTokenizer(), return_tensors=None)

    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4], "attention_mask": [1], "labels": [4]},
        ]
    )

    assert batch["input_ids"] == [[1, 2, 3], [4, 0, 0]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"] == [[-100, 2, 3], [4, -100, -100]]


def test_teich_data_collator_falls_back_to_eos_pad_token_id():
    collator = TeichDataCollator(tokenizer=EosOnlyTokenizer(), return_tensors=None)

    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )

    assert collator.pad_token_id == 2
    assert batch["input_ids"] == [[1, 2], [3, 2]]
    assert batch["labels"] == [[-100, 2], [3, -100]]


def test_teich_data_collator_respects_left_padding_side():
    collator = TeichDataCollator(tokenizer=LeftPadTokenizer(), return_tensors=None)

    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )

    assert batch["input_ids"] == [[1, 2], [0, 3]]
    assert batch["attention_mask"] == [[1, 1], [0, 1]]
    assert batch["labels"] == [[-100, 2], [-100, 3]]


def test_teich_data_collator_rejects_missing_pad_token_id():
    with pytest.raises(ValueError, match="pad_token_id"):
        TeichDataCollator(return_tensors=None)
