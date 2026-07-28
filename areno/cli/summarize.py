"""Summarize agentic trajectory outcomes from saved run artifacts.

Reads ``rollout_samples.*.jsonl`` files produced by the agentic RL training
loop (``_log_agentic_sample_completions`` in ``policy_only.py``), aggregates
turns, tool calls, tool failures, endings, and loss-mask statistics, and
prints a human-readable table or JSON summary.

The command operates entirely on local JSONL artifacts -- no model, worker, or
GPU is involved -- so it runs on any machine without CUDA.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import click

# Mirrors ``areno/api/defaults.py`` without triggering the heavy
# ``areno.api.__init__`` import chain (which pulls in torch).
DEFAULT_METRICS_LOG_DIR = "/tmp/areno/tfevent"

# Keywords that indicate a tool result is an error (case-insensitive).
_TOOL_FAILURE_KEYWORDS = ("error", "exception", "traceback", "failed", "failure")

# Valid values for --failure-reason filter.
_VALID_FAILURE_REASONS = {"length", "timeout_empty", "tool_error"}


@click.command(name="summarize", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--metrics-dir",
    "metrics_dir",
    type=click.Path(),
    default=DEFAULT_METRICS_LOG_DIR,
    show_default=True,
    help="Directory containing rollout_samples.*.jsonl files.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of a table.")
@click.option("--tool", "tool_filter", default=None, help="Only count tool calls matching this tool name.")
@click.option(
    "--failure-reason",
    "failure_reason_filter",
    default=None,
    help="Only count trajectories with this inferred failure reason (length, timeout_empty, tool_error).",
)
@click.option("--limit", "limit", default=0, show_default=True, help="Maximum JSONL lines to scan (0 = unlimited).")
def summarize_command(metrics_dir: str, as_json: bool, tool_filter: str | None, failure_reason_filter: str | None, limit: int):
    """Summarize agentic trajectory outcomes from saved run artifacts."""

    metrics_path = Path(metrics_dir)
    _validate_inputs(metrics_path, limit, failure_reason_filter)

    records, files_scanned, records_loaded = load_trajectory_records(metrics_path, limit=limit)

    if not records:
        click.echo(f"No agentic trajectory records found in {metrics_path}")
        return

    summary = summarize_trajectories(records, tool_filter=tool_filter, failure_reason_filter=failure_reason_filter)

    if as_json:
        click.echo(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        _print_table(summary)


def _validate_inputs(metrics_path: Path, limit: int, failure_reason_filter: str | None) -> None:
    """Validate CLI inputs before scanning files."""

    if not metrics_path.exists():
        raise click.UsageError(f"--metrics-dir does not exist: {metrics_path}")
    if not metrics_path.is_dir():
        raise click.UsageError(f"--metrics-dir is not a directory: {metrics_path}")
    if limit < 0:
        raise click.UsageError("--limit must be non-negative")
    if failure_reason_filter is not None and failure_reason_filter not in _VALID_FAILURE_REASONS:
        raise click.UsageError(
            f"--failure-reason must be one of: {', '.join(sorted(_VALID_FAILURE_REASONS))}"
        )


def load_trajectory_records(
    metrics_dir: Path,
    *,
    limit: int = 0,
) -> tuple[list[dict[str, Any]], int, int]:
    """Scan ``rollout_samples.*.jsonl`` and return agentic records.

    Reads files line by line so the full file set is never loaded into memory
    at once. Lines that are not valid JSON or lack ``kind == "agentic"`` are
    silently skipped.

    Returns:
        ``(records, files_scanned, records_loaded)``
    """

    records: list[dict[str, Any]] = []
    files_scanned = 0
    for jsonl_path in sorted(metrics_dir.glob("rollout_samples.*.jsonl")):
        files_scanned += 1
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("kind") != "agentic":
                    continue
                records.append(record)
                if limit > 0 and len(records) >= limit:
                    return records, files_scanned, len(records)
    return records, files_scanned, len(records)


def summarize_trajectories(
    records: list[dict[str, Any]],
    *,
    tool_filter: str | None = None,
    failure_reason_filter: str | None = None,
) -> dict[str, Any]:
    """Aggregate trajectory statistics from agentic JSONL records.

    All finish_reason and tool_failure values are inferred from indirect
    signals in the record (see ``_infer_finish_reason`` and
    ``_infer_tool_failure``). They are heuristic, not authoritative.
    """

    total_trajectories = len(records)
    total_turns = 0
    endings: Counter[str] = Counter()
    tool_calls_by_name: Counter[str] = Counter()
    tool_failures_by_name: Counter[str] = Counter()
    loss_tokens_sum = 0
    total_tokens_sum = 0
    completed = 0

    for record in records:
        tool_calls = record.get("tool_calls") or []
        tool_results = record.get("tool_results") or []
        final_answer = record.get("final_answer") or ""
        loss_mask_true = int(record.get("loss_mask_true", 0))
        loss_mask_total = int(record.get("loss_mask_total", 0))

        finish_reason = _infer_finish_reason(tool_calls, final_answer, loss_mask_true)
        endings[finish_reason] += 1
        if finish_reason == "stop":
            completed += 1

        # Turns = 1 (base turn) + number of tool calls that received results.
        turns = 1 + min(len(tool_calls), len(tool_results))
        total_turns += turns

        for tc in tool_calls:
            name = _tool_call_name(tc)
            if name is None:
                continue
            if tool_filter is not None and name != tool_filter:
                continue
            tool_calls_by_name[name] += 1

        for tr in tool_results:
            name = tr.get("name")
            if name is None:
                continue
            if tool_filter is not None and name != tool_filter:
                continue
            if _infer_tool_failure(tr):
                tool_failures_by_name[name] += 1

        loss_tokens_sum += loss_mask_true
        total_tokens_sum += loss_mask_total

    # Apply failure-reason filter to the trajectory-level stats.
    if failure_reason_filter is not None:
        filtered_records = [
            r
            for r in records
            if _infer_finish_reason(
                r.get("tool_calls") or [],
                r.get("final_answer") or "",
                int(r.get("loss_mask_true", 0)),
            )
            == _failure_reason_to_ending(failure_reason_filter)
        ]
        filtered_summary = summarize_trajectories(filtered_records, tool_filter=tool_filter)
        filtered_summary["source_dir"] = None
        filtered_summary["files_scanned"] = 0
        filtered_summary["records_loaded"] = len(records)
        filtered_summary["records_filtered"] = len(filtered_records)
        filtered_summary["filtered_by_failure_reason"] = failure_reason_filter
        return filtered_summary

    completion_rate = completed / total_trajectories if total_trajectories else 0.0
    avg_turns = total_turns / total_trajectories if total_trajectories else 0.0
    avg_loss_tokens = loss_tokens_sum / total_trajectories if total_trajectories else 0.0
    avg_total_tokens = total_tokens_sum / total_trajectories if total_trajectories else 0.0
    avg_loss_coverage = (loss_tokens_sum / total_tokens_sum) if total_tokens_sum else 0.0

    by_tool: dict[str, dict[str, Any]] = {}
    all_names = set(tool_calls_by_name) | set(tool_failures_by_name)
    for name in sorted(all_names):
        calls = tool_calls_by_name.get(name, 0)
        failures = tool_failures_by_name.get(name, 0)
        by_tool[name] = {
            "calls": calls,
            "failures": failures,
            "failure_rate": failures / calls if calls else 0.0,
        }
    total_calls = sum(tc["calls"] for tc in by_tool.values())
    total_failures = sum(tc["failures"] for tc in by_tool.values())

    return {
        "files_scanned": 0,
        "records_loaded": total_trajectories,
        "records_filtered": total_trajectories,
        "overview": {
            "total_trajectories": total_trajectories,
            "total_turns": total_turns,
            "avg_turns_per_trajectory": round(avg_turns, 4),
            "completion_rate": round(completion_rate, 4),
            "completed": completed,
        },
        "tool_calls": {
            "by_tool": by_tool,
            "total": {
                "calls": total_calls,
                "failures": total_failures,
                "failure_rate": total_failures / total_calls if total_calls else 0.0,
            },
        },
        "endings": {
            "stop": endings.get("stop", 0),
            "tool_calls": endings.get("tool_calls", 0),
            "length": endings.get("length", 0),
            "timeout_empty": endings.get("timeout_empty", 0),
        },
        "loss_mask": {
            "avg_loss_tokens": round(avg_loss_tokens, 4),
            "avg_total_tokens": round(avg_total_tokens, 4),
            "avg_loss_coverage": round(avg_loss_coverage, 4),
        },
    }


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #


def _infer_finish_reason(tool_calls: list, final_answer: str, loss_mask_true: int) -> str:
    """Infer finish_reason from indirect signals in a JSONL record.

    Rules (checked in order):
    1. ``tool_calls`` non-empty -> ``"tool_calls"`` (agent is still calling tools).
    2. ``final_answer`` empty and ``loss_mask_true == 0`` -> ``"length"`` (budget cut).
    3. ``loss_mask_true == 0`` but ``final_answer`` non-empty -> ``"timeout_empty"``.
    4. Otherwise -> ``"stop"`` (normal completion).
    """

    if tool_calls:
        return "tool_calls"
    if not final_answer and loss_mask_true == 0:
        return "length"
    if loss_mask_true == 0:
        return "timeout_empty"
    return "stop"


def _infer_tool_failure(tool_result: dict[str, Any]) -> bool:
    """Infer whether a tool result represents a failure.

    Checks ``content`` for error-indicating keywords (case-insensitive).
    """

    content = tool_result.get("content")
    if not isinstance(content, str) or not content:
        return False
    lowered = content.lower()
    return any(keyword in lowered for keyword in _TOOL_FAILURE_KEYWORDS)


def _tool_call_name(tool_call: dict[str, Any]) -> str | None:
    """Extract the tool name from a tool_call dict (OpenAI format)."""

    # Direct {"name": ...} format (used by reward_record).
    name = tool_call.get("name")
    if name:
        return str(name)
    # OpenAI {"function": {"name": ...}} format.
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if name:
            return str(name)
    return None


def _failure_reason_to_ending(failure_reason: str) -> str:
    """Map a --failure-reason value to an inferred ending label."""

    if failure_reason == "tool_error":
        return "stop"
    return failure_reason


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _print_table(summary: dict[str, Any]) -> None:
    """Print a human-readable summary table."""

    overview = summary["overview"]
    endings = summary["endings"]
    tool_data = summary["tool_calls"]
    loss = summary["loss_mask"]
    total_trajectories = overview["total_trajectories"]

    click.echo("Agentic Trajectory Summary")
    click.echo("=" * 41)
    click.echo()
    click.echo(f"Source: {summary.get('source_dir', '(unknown)')} ({summary.get('files_scanned', 0)} files, {total_trajectories} records)")
    click.echo()

    # Overview
    click.echo("Overview")
    click.echo(f"  Total trajectories:     {total_trajectories}")
    click.echo(f"  Total turns:            {overview['total_turns']}")
    click.echo(f"  Avg turns/trajectory:   {overview['avg_turns_per_trajectory']:.2f}")
    completed = overview["completed"]
    rate = overview["completion_rate"]
    click.echo(f"  Completion rate:        {rate:.1%}  ({completed}/{total_trajectories})")
    click.echo()

    # Tool Calls
    click.echo("Tool Calls")
    click.echo(f"  {'Tool':<20s} {'Calls':>6s}   {'Failures':>8s}   {'Failure%':>8s}")
    by_tool = tool_data["by_tool"]
    for name in sorted(by_tool):
        td = by_tool[name]
        click.echo(f"  {name:<20s} {td['calls']:>6d}   {td['failures']:>8d}   {td['failure_rate']:>7.1%}")
    total = tool_data["total"]
    click.echo(f"  {'Total':<20s} {total['calls']:>6d}   {total['failures']:>8d}   {total['failure_rate']:>7.1%}")
    click.echo()

    # Endings
    click.echo("Endings")
    click.echo(f"  {'Reason':<25s} {'Count':>5s}   {'Percentage':>10s}")
    ending_labels = [
        ("stop (completed)", endings.get("stop", 0)),
        ("tool_calls (ongoing)", endings.get("tool_calls", 0)),
        ("length (budget)", endings.get("length", 0)),
        ("timeout/empty", endings.get("timeout_empty", 0)),
    ]
    for label, count in ending_labels:
        pct = count / total_trajectories if total_trajectories else 0.0
        click.echo(f"  {label:<25s} {count:>5d}   {pct:>9.1%}")
    click.echo()

    # Loss Mask
    click.echo("Loss Mask")
    click.echo(f"  Avg loss tokens:       {loss['avg_loss_tokens']:.1f}")
    click.echo(f"  Avg total tokens:      {loss['avg_total_tokens']:.1f}")
    click.echo(f"  Avg loss coverage:     {loss['avg_loss_coverage']:.1%}")