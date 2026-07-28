:orphan:

Summarize CLI reference
=======================

``areno summarize`` aggregates agentic trajectory outcomes from saved run
artifacts and prints a summary in the terminal. It reads the
``rollout_samples.*.jsonl`` files that the agentic RL training loop writes
during ``_log_agentic_sample_completions`` -- no model, worker, or GPU is
involved.

Quick start
-----------

After an agentic training run (e.g. ``--algo gspo --agent-fn ...``), summary
data is already on disk in the metrics directory:

.. code-block:: bash

   areno summarize

This scans ``/tmp/areno/tfevent`` (the default ``--metrics-dir``) and prints a
table.

Options
-------

``--metrics-dir PATH``
    Directory containing ``rollout_samples.*.jsonl`` files.
    Default: ``/tmp/areno/tfevent``.

``--json``
    Emit machine-readable JSON instead of a human-readable table.

``--tool NAME``
    Only count tool calls matching this tool name. Other tools are excluded
    from the ``Tool Calls`` section; overview and endings still cover all
    trajectories.

``--failure-reason REASON``
    Only count trajectories with this inferred failure reason. Valid values:
    ``length``, ``timeout_empty``, ``tool_error``.

``--limit N``
    Maximum JSONL lines to scan across all files (0 = unlimited). Use this to
    bound memory on large run directories.

Input validation
-----------------

All validation happens before file scanning:

* ``--metrics-dir`` must exist and be a directory.
* ``--limit`` must be non-negative.
* ``--failure-reason`` must be one of the valid values.

If the directory contains no ``rollout_samples.*.jsonl`` files or no
``kind: "agentic"`` records, the command prints a message and exits 0.

Output fields
-------------

Table format (default)
^^^^^^^^^^^^^^^^^^^^^^

::

   Agentic Trajectory Summary
   ==========================

   Source: /tmp/areno/tfevent (2 files, 80 records)

   Overview
     Total trajectories:      80
     Total turns:            160
     Avg turns/trajectory:  2.00
     Completion rate:       82.5%  (66/80)

   Tool Calls
     Tool              Calls   Failures   Failure%
     choose_square       100         5       5.0%
     search               40         3       7.5%
     Total               140         8       5.7%

   Endings
     Reason                Count   Percentage
     stop (completed)        66       82.5%
     tool_calls (ongoing)    10       12.5%
     length (budget)          3        3.8%
     timeout/empty            1        1.3%

   Loss Mask
     Avg loss tokens:       7.2
     Avg total tokens:     10.5
     Avg loss coverage:    68.6%

JSON format (``--json``)
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "files_scanned": 2,
     "records_loaded": 80,
     "records_filtered": 80,
     "overview": {
       "total_trajectories": 80,
       "total_turns": 160,
       "avg_turns_per_trajectory": 2.0,
       "completion_rate": 0.825,
       "completed": 66
     },
     "tool_calls": {
       "by_tool": {
         "choose_square": {"calls": 100, "failures": 5, "failure_rate": 0.05},
         "search": {"calls": 40, "failures": 3, "failure_rate": 0.075}
       },
       "total": {"calls": 140, "failures": 8, "failure_rate": 0.057}
     },
     "endings": {
       "stop": 66,
       "tool_calls": 10,
       "length": 3,
       "timeout_empty": 1
     },
     "loss_mask": {
       "avg_loss_tokens": 7.2,
       "avg_total_tokens": 10.5,
       "avg_loss_coverage": 0.686
     }
   }

Privacy
-------

Output contains only aggregate statistics. No raw ``prompt``, ``messages``,
``final_answer``, ``tokens``, or ``loss_mask`` content from the original
records appears in either output format.

Inferred fields
---------------

The JSONL records written by the training loop do not contain explicit
``finish_reason`` or ``tool_failure`` fields. The summarize command infers
them from indirect signals:

**finish_reason** (checked in priority order):

1. ``tool_calls`` non-empty -> ``tool_calls`` (agent is still calling tools)
2. ``tool_calls`` empty + ``final_answer`` empty + ``loss_mask_true == 0`` -> ``length`` (budget cut)
3. ``tool_calls`` empty + ``loss_mask_true == 0`` + ``final_answer`` non-empty -> ``timeout_empty``
4. Otherwise -> ``stop`` (normal completion)

**tool_failure**: ``tool_results[i].content`` contains any of
``error``, ``exception``, ``traceback``, ``failed``, ``failure``
(case-insensitive).

These are heuristic inferences, not authoritative values.

Limitations
-----------

* ``finish_reason`` and ``tool_failure`` are inferred, not read from an
  explicit field. Future versions may add explicit fields to the JSONL.
* If ``ARENO_LOG_COMPLETIONS=0`` was set during training, no samples were
  written and there is nothing to summarize.
* The command does not deduplicate records across files. If a JSONL file was
  copied into the same directory, its records will be counted twice.

Examples
--------

Summarize the default metrics directory:

.. code-block:: bash

   areno summarize

Specify a custom metrics directory:

.. code-block:: bash

   areno summarize --metrics-dir /path/to/run/output

JSON output for scripting:

.. code-block:: bash

   areno summarize --json

Filter to a single tool:

.. code-block:: bash

   areno summarize --tool search

Filter to budget-cut trajectories:

.. code-block:: bash

   areno summarize --failure-reason length

Limit memory on large directories:

.. code-block:: bash

   areno summarize --limit 1000