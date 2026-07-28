"""CPU tests for the ``areno summarize`` CLI command.

Covers: success path with multi-tool trajectories, JSON/table output fields,
tool and failure-reason filters, invalid inputs, boundary values, malformed
JSON, inference helpers, privacy-safe output, and integration via CliRunner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click import UsageError
from click.testing import CliRunner

from areno.cli.summarize import (
    _infer_finish_reason,
    _infer_tool_failure,
    load_trajectory_records,
    summarize_trajectories,
    summarize_command,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _agentic_record(
    *,
    tool_calls=None,
    tool_results=None,
    final_answer="42",
    loss_mask_true=5,
    loss_mask_total=10,
    prompt="test prompt",
    prompt_idx=0,
    sample_idx=0,
) -> dict:
    return {
        "kind": "agentic",
        "epoch": 0,
        "step": 0,
        "prompt_idx": prompt_idx,
        "sample_idx": sample_idx,
        "prompt": prompt,
        "messages": [],
        "final_answer": final_answer,
        "tool_calls": tool_calls or [],
        "tool_results": tool_results or [],
        "loss_mask_true": loss_mask_true,
        "loss_mask_total": loss_mask_total,
        "first_loss_idx": 0,
        "loss_mask": [],
        "tokens": [],
    }


def _rollout_record() -> dict:
    return {
        "kind": "rollout",
        "epoch": 0,
        "step": 0,
        "completion": "test",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


SEARCH_CALL = {"name": "search", "arguments": '{"query": "x"}'}
SUBMIT_CALL = {"name": "submit", "arguments": "{}"}
MOVE_CALL = {"name": "choose_square", "arguments": '{"square": 5}'}

SEARCH_RESULT_OK = {"name": "search", "tool_call_id": "c1", "content": "Found 3 results"}
SEARCH_RESULT_ERR = {"name": "search", "tool_call_id": "c2", "content": "Error: connection failed"}
MOVE_RESULT_OK = {"name": "choose_square", "tool_call_id": "c3", "content": "OK"}


MULTI_TOOL_RECORDS = [
    _agentic_record(
        tool_calls=[SEARCH_CALL, MOVE_CALL],
        tool_results=[SEARCH_RESULT_OK, MOVE_RESULT_OK],
        final_answer="done",
        loss_mask_true=8,
        loss_mask_total=12,
        prompt_idx=0,
        sample_idx=0,
    ),
    _agentic_record(
        tool_calls=[SEARCH_CALL],
        tool_results=[SEARCH_RESULT_ERR],
        final_answer="",
        loss_mask_true=0,
        loss_mask_total=10,
        prompt_idx=0,
        sample_idx=1,
    ),
    _agentic_record(
        tool_calls=[],
        tool_results=[],
        final_answer="42",
        loss_mask_true=5,
        loss_mask_total=8,
        prompt_idx=1,
        sample_idx=0,
    ),
    _agentic_record(
        tool_calls=[SUBMIT_CALL],
        tool_results=[{"name": "submit", "content": "accepted"}],
        final_answer="",
        loss_mask_true=0,
        loss_mask_total=6,
        prompt_idx=1,
        sample_idx=1,
    ),
]


# --------------------------------------------------------------------------- #
# Success path
# --------------------------------------------------------------------------- #


def test_summarize_multi_tool_trajectories():
    summary = summarize_trajectories(MULTI_TOOL_RECORDS)

    assert summary["overview"]["total_trajectories"] == 4
    # Records: 2+1 tools, 0 tools, 1 tool => 2+1+0+1 = 4 tool_calls entries
    assert summary["tool_calls"]["total"]["calls"] == 4
    # search called 2x, choose_square 1x, submit 1x
    assert summary["tool_calls"]["by_tool"]["search"]["calls"] == 2
    assert summary["tool_calls"]["by_tool"]["choose_square"]["calls"] == 1
    assert summary["tool_calls"]["by_tool"]["submit"]["calls"] == 1
    # search has 1 failure (SEARCH_RESULT_ERR)
    assert summary["tool_calls"]["by_tool"]["search"]["failures"] == 1


def test_summarize_completion_rate():
    summary = summarize_trajectories(MULTI_TOOL_RECORDS)
    # Record 0: tool_calls non-empty -> "tool_calls"
    # Record 1: tool_calls non-empty -> "tool_calls" (takes priority over length)
    # Record 2: no tools + final_answer="42" -> "stop" (completed)
    # Record 3: tool_calls non-empty -> "tool_calls"
    assert summary["overview"]["completed"] == 1
    assert summary["overview"]["completion_rate"] == 0.25
    assert summary["endings"]["stop"] == 1
    assert summary["endings"]["tool_calls"] == 3
    assert summary["endings"]["length"] == 0


def test_summarize_endings_reconcile_with_total():
    """Endings counts must sum to total_trajectories."""
    summary = summarize_trajectories(MULTI_TOOL_RECORDS)
    total = summary["overview"]["total_trajectories"]
    endings_sum = sum(summary["endings"].values())
    assert endings_sum == total


def test_summarize_json_output_fields():
    """JSON output must contain all top-level sections."""
    summary = summarize_trajectories(MULTI_TOOL_RECORDS)
    for key in ("overview", "tool_calls", "endings", "loss_mask"):
        assert key in summary
    assert "by_tool" in summary["tool_calls"]
    assert "total" in summary["tool_calls"]


def test_summarize_loss_mask_stats():
    summary = summarize_trajectories(MULTI_TOOL_RECORDS)
    # loss_mask_true: 8+0+5+0 = 13, avg = 13/4 = 3.25
    assert summary["loss_mask"]["avg_loss_tokens"] == pytest.approx(3.25, abs=0.01)
    # loss_mask_total: 12+10+8+6 = 36, avg = 36/4 = 9.0
    assert summary["loss_mask"]["avg_total_tokens"] == pytest.approx(9.0, abs=0.01)


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def test_summarize_tool_filter():
    summary = summarize_trajectories(MULTI_TOOL_RECORDS, tool_filter="search")
    assert summary["tool_calls"]["by_tool"]["search"]["calls"] == 2
    # Other tools should not appear
    assert "choose_square" not in summary["tool_calls"]["by_tool"]
    assert "submit" not in summary["tool_calls"]["by_tool"]


def test_summarize_failure_reason_filter_length():
    # Add a record with no tool_calls, empty answer, loss_mask_true=0 -> "length"
    records = MULTI_TOOL_RECORDS + [
        _agentic_record(
            tool_calls=[],
            tool_results=[],
            final_answer="",
            loss_mask_true=0,
            loss_mask_total=10,
            prompt_idx=2,
            sample_idx=0,
        )
    ]
    summary = summarize_trajectories(records, failure_reason_filter="length")
    assert summary["overview"]["total_trajectories"] == 1
    assert summary["endings"]["length"] == 1


def test_summarize_failure_reason_filter_timeout_empty():
    # Add a record with loss_mask_true=0 but non-empty final_answer
    records = MULTI_TOOL_RECORDS + [
        _agentic_record(
            tool_calls=[],
            tool_results=[],
            final_answer="partial",
            loss_mask_true=0,
            loss_mask_total=10,
        )
    ]
    summary = summarize_trajectories(records, failure_reason_filter="timeout_empty")
    assert summary["overview"]["total_trajectories"] == 1
    assert summary["endings"]["timeout_empty"] == 1


# --------------------------------------------------------------------------- #
# Invalid inputs
# --------------------------------------------------------------------------- #


def test_metrics_dir_not_found(tmp_path):
    nonexistent = tmp_path / "does-not-exist"
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(nonexistent)])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_metrics_dir_not_a_directory(tmp_path):
    file_path = tmp_path / "a-file"
    file_path.write_text("hello")
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(file_path)])
    assert result.exit_code != 0
    assert "not a directory" in result.output


def test_negative_limit(tmp_path):
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(tmp_path), "--limit", "-1"])
    assert result.exit_code != 0
    assert "non-negative" in result.output


def test_invalid_failure_reason(tmp_path):
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(tmp_path), "--failure-reason", "bogus"])
    assert result.exit_code != 0
    assert "must be one of" in result.output


# --------------------------------------------------------------------------- #
# Boundary values
# --------------------------------------------------------------------------- #


def test_empty_directory(tmp_path):
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No agentic trajectory records found" in result.output


def test_no_agentic_records(tmp_path):
    jsonl = tmp_path / "rollout_samples.123.jsonl"
    _write_jsonl(jsonl, [_rollout_record(), _rollout_record()])
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No agentic trajectory records found" in result.output


def test_malformed_json_lines(tmp_path):
    jsonl = tmp_path / "rollout_samples.456.jsonl"
    jsonl.write_text(
        json.dumps(_agentic_record()) + "\n"
        "this is not json\n"
        json.dumps(_agentic_record(final_answer="second")) + "\n"
        "\n"  # blank line
    )
    records, files_scanned, loaded = load_trajectory_records(tmp_path)
    assert files_scanned == 1
    assert loaded == 2
    assert len(records) == 2
    assert records[0]["final_answer"] == "42"
    assert records[1]["final_answer"] == "second"


def test_limit_truncates_records(tmp_path):
    jsonl = tmp_path / "rollout_samples.789.jsonl"
    _write_jsonl(jsonl, [_agentic_record(sample_idx=i) for i in range(20)])
    records, _, loaded = load_trajectory_records(tmp_path, limit=5)
    assert loaded == 5
    assert len(records) == 5


def test_limit_zero_means_unlimited(tmp_path):
    jsonl = tmp_path / "rollout_samples.000.jsonl"
    _write_jsonl(jsonl, [_agentic_record(sample_idx=i) for i in range(10)])
    records, _, loaded = load_trajectory_records(tmp_path, limit=0)
    assert loaded == 10
    assert len(records) == 10


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #


def test_infer_finish_reason_stop():
    assert _infer_finish_reason([], "42", 5) == "stop"


def test_infer_finish_reason_tool_calls():
    assert _infer_finish_reason([SEARCH_CALL], "anything", 5) == "tool_calls"


def test_infer_finish_reason_length():
    assert _infer_finish_reason([], "", 0) == "length"


def test_infer_finish_reason_timeout_empty():
    assert _infer_finish_reason([], "partial", 0) == "timeout_empty"


def test_infer_tool_failure_by_content():
    assert _infer_tool_failure({"content": "Error: something went wrong"}) is True
    assert _infer_tool_failure({"content": "Traceback (most recent call last)..."}) is True
    assert _infer_tool_failure({"content": "failed to connect"}) is True


def test_infer_tool_failure_normal_content():
    assert _infer_tool_failure({"content": "Found 3 results"}) is False
    assert _infer_tool_failure({"content": "OK"}) is False


def test_infer_tool_failure_empty_content():
    assert _infer_tool_failure({"content": ""}) is False
    assert _infer_tool_failure({"content": None}) is False
    assert _infer_tool_failure({}) is False


# --------------------------------------------------------------------------- #
# Privacy
# --------------------------------------------------------------------------- #


def test_output_no_raw_prompt_or_completion():
    """Output must not contain any raw prompt or completion text."""
    records = [
        _agentic_record(
            prompt="SECRET PROMPT TEXT",
            final_answer="SECRET ANSWER TEXT",
        )
    ]
    summary = summarize_trajectories(records)
    summary_str = json.dumps(summary)
    assert "SECRET PROMPT TEXT" not in summary_str
    assert "SECRET ANSWER TEXT" not in summary_str


def test_table_output_no_raw_prompt(tmp_path):
    jsonl = tmp_path / "rollout_samples.999.jsonl"
    _write_jsonl(jsonl, [_agentic_record(prompt="CLASSIFIED", final_answer="CLASSIFIED_ANSWER")])
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "CLASSIFIED" not in result.output


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #


def test_integration_cli_command_table(tmp_path):
    """Full CLI invocation with table output on a tiny fixture."""
    jsonl = tmp_path / "rollout_samples.111.jsonl"
    _write_jsonl(jsonl, MULTI_TOOL_RECORDS)
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Agentic Trajectory Summary" in result.output
    assert "Overview" in result.output
    assert "Tool Calls" in result.output
    assert "Endings" in result.output
    assert "Loss Mask" in result.output
    assert "search" in result.output


def test_integration_cli_command_json(tmp_path):
    """Full CLI invocation with JSON output."""
    jsonl = tmp_path / "rollout_samples.222.jsonl"
    _write_jsonl(jsonl, MULTI_TOOL_RECORDS)
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["overview"]["total_trajectories"] == 4
    assert "by_tool" in parsed["tool_calls"]


def test_integration_reconciliation(tmp_path):
    """Sum of endings must equal total_trajectories in CLI output."""
    jsonl = tmp_path / "rollout_samples.333.jsonl"
    _write_jsonl(jsonl, MULTI_TOOL_RECORDS)
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    total = parsed["overview"]["total_trajectories"]
    endings_sum = sum(parsed["endings"].values())
    assert endings_sum == total


def test_integration_tool_filter_cli(tmp_path):
    jsonl = tmp_path / "rollout_samples.444.jsonl"
    _write_jsonl(jsonl, MULTI_TOOL_RECORDS)
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(tmp_path), "--json", "--tool", "search"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "search" in parsed["tool_calls"]["by_tool"]
    assert "choose_square" not in parsed["tool_calls"]["by_tool"]


def test_integration_mixed_rollout_and_agentic(tmp_path):
    """Non-agentic records are skipped, agentic records are counted."""
    jsonl = tmp_path / "rollout_samples.555.jsonl"
    _write_jsonl(jsonl, [_rollout_record(), _agentic_record(), _rollout_record(), _agentic_record()])
    runner = CliRunner()
    result = runner.invoke(summarize_command, ["--metrics-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["overview"]["total_trajectories"] == 2