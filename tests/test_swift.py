import json

import pytest

from teich import (
    convert_to_ms_swift,
    ms_swift_content_length,
    progressive_prefixes,
    to_ms_swift_messages,
    to_ms_swift_row,
    validate_ms_swift_messages,
)


def test_basic_roles_passthrough_and_developer_maps_to_system():
    messages = [
        {"role": "developer", "content": "dev context"},
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = to_ms_swift_messages(messages, clean=False)
    assert result == [
        {"role": "system", "content": "dev context"},
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_reasoning_is_inlined_as_think_on_final_assistant_turn():
    messages = [
        {"role": "user", "content": "where is the capital of Zhejiang?"},
        {"role": "assistant", "content": "Hangzhou.", "reasoning_content": "Zhejiang's capital is Hangzhou."},
    ]
    result = to_ms_swift_messages(messages)
    assert result == [
        {"role": "user", "content": "where is the capital of Zhejiang?"},
        {"role": "assistant", "content": "<think>\nZhejiang's capital is Hangzhou.\n</think>\n\nHangzhou."},
    ]


def test_intermediate_thinking_dropped_by_default_but_kept_with_flag():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
    ]

    default = to_ms_swift_messages(messages)
    assert default[1] == {"role": "assistant", "content": "a1"}
    assert default[3] == {"role": "assistant", "content": "<think>\nr2\n</think>\n\na2"}

    kept = to_ms_swift_messages(messages, keep_intermediate_thinking=True)
    assert kept[1] == {"role": "assistant", "content": "<think>\nr1\n</think>\n\na1"}
    assert kept[3] == {"role": "assistant", "content": "<think>\nr2\n</think>\n\na2"}


def test_tool_calls_become_tool_call_role_messages_with_json_string_content():
    messages = [
        {"role": "user", "content": "read the file"},
        {
            "role": "assistant",
            "content": "I'll read it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": {"file_path": "README.md"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "Read", "content": "# Title"},
        {"role": "assistant", "content": "Done."},
    ]
    result = to_ms_swift_messages(messages)
    assert result[1] == {"role": "assistant", "content": "I'll read it."}
    assert result[2]["role"] == "tool_call"
    assert json.loads(result[2]["content"]) == {"name": "Read", "arguments": {"file_path": "README.md"}}
    assert result[3] == {"role": "tool_response", "content": "# Title"}
    assert result[4] == {"role": "assistant", "content": "Done."}


def test_string_tool_arguments_are_parsed_into_objects():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}
            ],
        }
    ]
    result = to_ms_swift_messages(messages, clean=False)
    assert len(result) == 1
    assert result[0]["role"] == "tool_call"
    assert json.loads(result[0]["content"]) == {"name": "Bash", "arguments": {"command": "ls"}}


def test_assistant_with_only_tool_calls_emits_no_empty_assistant_message():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "LS", "arguments": {}}}],
        }
    ]
    result = to_ms_swift_messages(messages, clean=False)
    assert [message["role"] for message in result] == ["tool_call"]


def test_to_ms_swift_row_serializes_tools_to_json_string_and_drops_extra_fields():
    row = {
        "prompt": "hi",
        "follow_up_prompts": ["more"],
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "tools": [
            {"type": "function", "function": {"name": "Read", "description": "Read a file.", "parameters": {"type": "object"}}}
        ],
        "metadata": {"model": "qwen3"},
    }
    swift_row = to_ms_swift_row(row)
    assert set(swift_row) == {"messages", "tools"}
    assert isinstance(swift_row["tools"], str)
    assert json.loads(swift_row["tools"]) == row["tools"]
    assert swift_row["messages"][-1] == {"role": "assistant", "content": "hello"}


def test_to_ms_swift_row_omits_tools_when_absent_or_empty():
    assert "tools" not in to_ms_swift_row({"messages": [{"role": "user", "content": "hi"}], "tools": []})
    assert "tools" not in to_ms_swift_row({"messages": [{"role": "user", "content": "hi"}]})


def test_convert_to_ms_swift_maps_each_row():
    rows = [
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {"messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]},
    ]
    converted = convert_to_ms_swift(rows)
    assert len(converted) == 2
    assert converted[0]["messages"][0] == {"role": "user", "content": "a"}
    assert converted[1]["messages"][-1] == {"role": "assistant", "content": "d"}


def test_clean_merges_system_and_consecutive_user():
    messages = [
        {"role": "system", "content": "s1"},
        {"role": "user", "content": "u1"},
        {"role": "developer", "content": "s2"},  # mid-conversation system after developer mapping
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a"},
    ]
    result = to_ms_swift_messages(messages, clean=True)
    assert result == [
        {"role": "system", "content": "s1\n\ns2"},
        {"role": "user", "content": "u1\n\nu2"},
        {"role": "assistant", "content": "a"},
    ]


def test_clean_drops_orphan_tool_response():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "tool", "content": "orphan result"},
        {"role": "assistant", "content": "a"},
    ]
    result = to_ms_swift_messages(messages, clean=True)
    assert [m["role"] for m in result] == ["user", "assistant"]


def test_clean_drops_unanswered_tool_round_before_user():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "T", "arguments": {}}}]},
        {"role": "tool", "content": "res"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "answer"},
    ]
    result = to_ms_swift_messages(messages, clean=True)
    assert result == [
        {"role": "user", "content": "q1\n\nq2"},
        {"role": "assistant", "content": "answer"},
    ]


def test_clean_trims_trailing_incomplete_to_end_on_assistant():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "tool_calls": [{"type": "function", "function": {"name": "T", "arguments": {}}}]},
        {"role": "tool", "content": "res"},
    ]
    result = to_ms_swift_messages(messages, clean=True)
    assert [m["role"] for m in result] == ["user", "assistant"]
    assert result[-1] == {"role": "assistant", "content": "a"}


def test_clean_preserves_parallel_tool_calls():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "A", "arguments": {}}},
                {"type": "function", "function": {"name": "B", "arguments": {}}},
            ],
        },
        {"role": "tool", "content": "ra"},
        {"role": "tool", "content": "rb"},
        {"role": "assistant", "content": "done"},
    ]
    result = to_ms_swift_messages(messages, clean=True)
    assert [m["role"] for m in result] == ["user", "tool_call", "tool_call", "tool_response", "tool_response", "assistant"]
    assert validate_ms_swift_messages(result) == []


def test_convert_drops_untrainable_rows_after_cleaning():
    rows = [
        {"messages": [{"role": "user", "content": "only a user, nothing to train"}]},
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
    ]
    converted = convert_to_ms_swift(rows, clean=True)
    assert len(converted) == 1
    assert converted[0]["messages"][-1] == {"role": "assistant", "content": "a"}


def test_validate_ms_swift_messages_flags_problems_and_passes_clean():
    assert validate_ms_swift_messages(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    ) == []
    consecutive_user = validate_ms_swift_messages(
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}]
    )
    assert any("consecutive user" in issue for issue in consecutive_user)
    ends_on_user = validate_ms_swift_messages([{"role": "user", "content": "q"}])
    assert any("end" in issue or "assistant" in issue for issue in ends_on_user)


def test_progressive_prefixes_accumulate_each_round():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    prefixes = progressive_prefixes(messages)
    assert [[m["content"] for m in p] for p in prefixes] == [
        ["u1", "a1"],
        ["u1", "a1", "u2", "a2"],
        ["u1", "a1", "u2", "a2", "u3", "a3"],
    ]
    assert prefixes[-1] == messages  # last prefix is the full conversation


def test_progressive_prefixes_cut_only_at_round_boundaries_and_keep_system():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "thinking"},
        {"role": "tool_call", "content": '{"name": "T", "arguments": {}}'},
        {"role": "tool_response", "content": "res"},
        {"role": "assistant", "content": "answer1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "answer2"},
    ]
    prefixes = progressive_prefixes(messages)
    assert len(prefixes) == 2  # two user rounds; mid-round tool_call is not a cut
    assert [m["role"] for m in prefixes[0]] == ["system", "user", "assistant", "tool_call", "tool_response", "assistant"]
    assert prefixes[0][-1]["content"] == "answer1"
    for prefix in prefixes:
        assert prefix[0]["role"] == "system"  # leading system carried into every prefix
        assert validate_ms_swift_messages(prefix) == []  # each prefix is a valid ms-swift conversation


def test_ms_swift_content_length_sums_content_chars():
    assert ms_swift_content_length([{"role": "user", "content": "aa"}, {"role": "assistant", "content": "bbb"}]) == 5


def test_convert_to_ms_swift_progressive_expands_rows():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ],
            "tools": [{"type": "function", "function": {"name": "T", "parameters": {"type": "object"}}}],
        }
    ]
    converted = convert_to_ms_swift(rows, progressive=True)
    assert len(converted) == 2
    assert [m["content"] for m in converted[0]["messages"]] == ["u1", "a1"]
    assert [m["content"] for m in converted[1]["messages"]] == ["u1", "a1", "u2", "a2"]
    assert all("tools" in row for row in converted)  # tools copied into each prefix row


def test_progressive_step_granularity_targets_each_tool_call():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "let me check"},
        {"role": "tool_call", "content": '{"name": "A", "arguments": {}}'},
        {"role": "tool_response", "content": "r1"},
        {"role": "tool_call", "content": '{"name": "B", "arguments": {}}'},
        {"role": "tool_response", "content": "r2"},
        {"role": "assistant", "content": "answer"},
    ]
    step = progressive_prefixes(messages, "step")
    assert len(step) == 3
    # second prefix ends on a tool_call: previous - user - tool_call - tool_result - tool_call
    assert [m["role"] for m in step[1]] == ["user", "assistant", "tool_call", "tool_response", "tool_call"]
    assert step[1][-1]["role"] == "tool_call"
    assert validate_ms_swift_messages(step[1]) == []  # ending on a tool_call is a valid last-turn target
    assert step[-1] == messages

    # round granularity keeps the whole tool sequence as one example (single user round)
    rounds = progressive_prefixes(messages, "round")
    assert len(rounds) == 1
    assert rounds[0] == messages


def test_progressive_step_keeps_parallel_calls_in_one_step():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "tool_call", "content": '{"name": "A", "arguments": {}}'},
        {"role": "tool_call", "content": '{"name": "B", "arguments": {}}'},
        {"role": "tool_response", "content": "ra"},
        {"role": "tool_response", "content": "rb"},
        {"role": "assistant", "content": "done"},
    ]
    step = progressive_prefixes(messages, "step")
    assert [m["role"] for m in step[0]] == ["user", "tool_call", "tool_call"]  # parallel calls not split
    assert len(step) == 2


def test_convert_to_ms_swift_step_granularity_and_invalid_value():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "tool_call", "content": '{"name": "A", "arguments": {}}'},
                {"role": "tool_response", "content": "r"},
                {"role": "assistant", "content": "done"},
            ]
        }
    ]
    converted = convert_to_ms_swift(rows, progressive=True, granularity="step")
    assert len(converted) == 2
    assert converted[0]["messages"][-1]["role"] == "tool_call"
    assert converted[1]["messages"][-1]["role"] == "assistant"

    with pytest.raises(ValueError):
        convert_to_ms_swift(rows, progressive=True, granularity="bogus")


def test_convert_to_ms_swift_max_content_length_drops_long_rows():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "x" * 10},
                {"role": "assistant", "content": "y" * 10},
                {"role": "user", "content": "z" * 100},
                {"role": "assistant", "content": "w" * 100},
            ]
        }
    ]
    # progressive yields prefix1 (20 chars) and prefix2 (220 chars); cap keeps only prefix1.
    converted = convert_to_ms_swift(rows, progressive=True, max_content_length=50)
    assert len(converted) == 1
    assert [m["content"] for m in converted[0]["messages"]] == ["x" * 10, "y" * 10]
