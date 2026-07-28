# 需求文档：Summarize Agentic Trajectory Outcomes in the Terminal (#259)

## 1. 问题背景

### 1.1 现状

AReno 的 agentic RL 训练流程中，`PolicyOnlyTrainer._log_agentic_sample_completions`
（`areno/api/trainers/policy_only.py:475`）会将 agent 轨迹样本以 JSONL 格式写入本地
metrics 目录（默认 `/tmp/areno/tfevent`），文件名为 `rollout_samples.{pid}.jsonl`。

每条 `kind: "agentic"` 的记录包含 `tool_calls`、`tool_results`、`messages`、
`loss_mask`、`final_answer` 等字段，但**目前没有任何内置工具来汇总和展示这些轨迹
结果的统计概览**。用户要了解训练中 agent 的行为表现（工具调用频率、失败率、超时
情况、完成率等），只能手动读 JSONL 或写一次性脚本。

### 1.2 目标

提供一个 CLI 命令 `areno summarize`，从已保存的轨迹元数据 JSONL 文件中聚合统计
指标，在终端以表格或 JSON 格式输出，支持按工具名和失败原因过滤。

## 2. 数据来源分析

### 2.1 JSONL 文件位置

| 属性 | 值 |
|------|-----|
| 目录 | `--metrics-dir` 参数指定，默认 `/tmp/areno/tfevent`（`areno/api/defaults.py:3`） |
| 文件名模式 | `rollout_samples.{pid}.jsonl`（`areno/api/metrics.py:31`） |
| 一次训练可能产生多个文件 | 每个 worker 进程写自己的文件 |

### 2.2 现有 JSONL 记录结构（`kind: "agentic"`）

以下字段由 `_log_agentic_sample_completions`（`policy_only.py:475-507`）写入：

```json
{
  "kind": "agentic",
  "epoch": 0,
  "step": 0,
  "prompt_idx": 0,
  "sample_idx": 0,
  "prompt": "Solve the problem...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "name": "search", "content": "..."}
  ],
  "final_answer": "The answer is 42",
  "tool_calls": [
    {"name": "search", "arguments": "{\"query\": \"...\"}"}
  ],
  "tool_results": [
    {"name": "search", "tool_call_id": "call_abc", "content": "Result..."}
  ],
  "loss_mask_true": 5,
  "loss_mask_total": 10,
  "first_loss_idx": 3,
  "loss_mask": [false, false, false, true, true, true, true, true, ...],
  "tokens": [1, 2, 3, ...],
  "pid": 12345
}
```

### 2.3 缺失字段的推断策略

现有 JSONL 中没有显式的 `finish_reason`、`tool_failure`、`timeout` 字段。
以下推断规则基于对 `areno/api/agentic.py` 源码的阅读：

| 目标字段 | 推断规则 | 源码依据 |
|----------|----------|----------|
| `finish_reason: "tool_calls"` | `tool_calls` 非空（**最高优先级**） | `agentic.py:581`：`response_kind = "assistant_tool_call" if tool_calls` |
| `finish_reason: "length"` | `tool_calls` 为空 + `final_answer` 为空 + `loss_mask_true == 0` | `agentic.py:829`：`_filtered_chat_response` 返回空 content + `finish_reason: "length"` |
| `finish_reason: "timeout_empty"` | `tool_calls` 为空 + `loss_mask_true == 0` + `final_answer` 非空 | `agentic.py:484-486`：agent 超时后仍有部分 messages 但 loss_mask 可能全 False |
| `finish_reason: "stop"` | `tool_calls` 为空 + `final_answer` 非空 + `loss_mask_true > 0` | `agentic.py:595`：无 tool_calls 时 trace 写入 `finish_reason: "stop"` |
| `tool_failure` | `tool_results[i].content` 包含 error/exception/traceback/failed/failure 关键字（不区分大小写） | 无显式 failure 字段；tool 的 content 由 agent 代码写入，error 内容是唯一可获取的信号 |
| `completion` | `finish_reason == "stop"` | 正常完成 = 无 tool_calls 且有 final_answer |

**推断优先级**：`tool_calls` 非空 → `"tool_calls"` > 空 answer + `loss_mask_true=0` → `"length"` > `loss_mask_true=0` + 非空 answer → `"timeout_empty"` > 否则 → `"stop"`

**推断的局限性**：这些推断是基于间接信号的启发式规则，不保证 100% 准确。
文档和输出中应标注为 `inferred`。

## 3. 功能需求

### 3.1 CLI 命令

```
areno summarize [OPTIONS]
```

**描述**：从已保存的轨迹元数据中聚合统计指标，在终端展示。

### 3.2 CLI 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--metrics-dir` | PATH | `/tmp/areno/tfevent` | 扫描 `rollout_samples.*.jsonl` 的目录 |
| `--json` | flag | `False` | 输出 JSON 格式而非人类可读表格 |
| `--tool` | TEXT | 无（不过滤） | 只统计指定工具名的调用 |
| `--failure-reason` | TEXT | 无（不过滤） | 只统计指定失败原因的轨迹（可选值：`length`、`timeout_empty`、`tool_error`） |
| `--limit` | INT | `0`（无限制） | 限制扫描的 JSONL 行数，控制内存 |
| `-h, --help` | flag | | 显示帮助 |

### 3.3 输入校验

在模型/worker 初始化**之前**执行（本命令不涉及模型初始化，但校验仍需在文件
扫描前完成）：

| 校验项 | 失败行为 |
|--------|----------|
| `--metrics-dir` 目录不存在 | `UsageError: --metrics-dir does not exist: {path}` |
| `--metrics-dir` 不是目录 | `UsageError: --metrics-dir is not a directory: {path}` |
| `--limit` < 0 | `UsageError: --limit must be non-negative` |
| 目录下无 `rollout_samples.*.jsonl` 文件或无 `kind: "agentic"` 记录 | 输出提示信息 `"No agentic trajectory records found in {dir}"`，正常退出（exit code 0） |

### 3.4 输出字段

#### 3.4.1 表格格式（默认）

```
Agentic Trajectory Summary
==========================

Source: /tmp/areno/tfevent (3 files, 150 records)

Overview
  Total trajectories:    150
  Total turns:           320
  Avg turns/trajectory:  2.13
  Completion rate:       78.0%  (117/150)

Tool Calls
  Tool              Calls   Failures   Failure%
  choose_square        180        12       6.7%
  search                95         8       8.4%
  submit                45         0       0.0%
  Total                 320        20       6.3%

Endings
  Reason                Count   Percentage
  stop (completed)        117       78.0%
  tool_calls (ongoing)     25       16.7%
  length (budget)           6        4.0%
  timeout/empty             2        1.3%

Loss Mask
  Avg loss tokens:       8.5
  Avg total tokens:     12.3
  Avg loss coverage:    69.1%
```

#### 3.4.2 JSON 格式（`--json`）

```json
{
  "source_dir": "/tmp/areno/tfevent",
  "files_scanned": 3,
  "records_loaded": 150,
  "records_filtered": 150,
  "overview": {
    "total_trajectories": 150,
    "total_turns": 320,
    "avg_turns_per_trajectory": 2.13,
    "completion_rate": 0.78,
    "completed": 117
  },
  "tool_calls": {
    "by_tool": {
      "choose_square": {"calls": 180, "failures": 12, "failure_rate": 0.067},
      "search": {"calls": 95, "failures": 8, "failure_rate": 0.084},
      "submit": {"calls": 45, "failures": 0, "failure_rate": 0.0}
    },
    "total": {"calls": 320, "failures": 20, "failure_rate": 0.063}
  },
  "endings": {
    "stop": 117,
    "tool_calls": 25,
    "length": 6,
    "timeout_empty": 2
  },
  "loss_mask": {
    "avg_loss_tokens": 8.5,
    "avg_total_tokens": 12.3,
    "avg_loss_coverage": 0.691
  }
}
```

#### 3.4.3 隐私安全

输出**只包含聚合统计**，不输出任何原始 `prompt`、`messages`、`final_answer`、
`tokens`、`loss_mask` 内容。即使用户指定 `--tool` 或 `--failure-reason` 过滤，
输出也只是缩小了统计范围，不会暴露被过滤掉的原始数据。

### 3.5 过滤逻辑

| 过滤器 | 作用域 | 行为 |
|--------|--------|------|
| `--tool <name>` | tool_calls 维度 | 只统计 `tool_calls` 中 `name == <name>` 的调用；endings/overview 仍基于全部轨迹 |
| `--failure-reason <reason>` | 轨迹维度 | 只统计推断为该失败原因的轨迹；可选值：`length`、`timeout_empty`、`tool_error` |

### 3.6 内存控制

- 使用 `--limit N` 限制扫描的 JSONL 总行数（跨所有文件），超出后停止读取。
- 文件逐行读取（`json.loads` per line），不一次性加载全部文件到内存。
- 默认 `--limit 0` 表示无限制。

## 4. 技术设计

### 4.1 文件结构

```
areno/cli/summarize.py                         # 新建：CLI 命令 + 聚合逻辑
tests/test_summarize_cpu.py                    # 新建：CPU 测试
```

修改：
```
areno/cli/main.py                              # 注册 "summarize" 命令
```

### 4.2 模块设计（`areno/cli/summarize.py`）

```
┌──────────────────────────────────────────────────────┐
│                   summarize_command                   │  Click 入口
│                   (CLI options)                       │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│               load_trajectory_records                 │  纯函数
│   扫描目录 → 逐行读 JSONL → 过滤 kind=="agentic"      │
│   → 返回 list[dict]                                   │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│              summarize_trajectories                   │  纯函数
│   推断 finish_reason → 聚合统计 → 返回 SummaryStats   │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│          _print_table / _print_json                   │  输出
└──────────────────────────────────────────────────────┘
```

### 4.3 核心函数签名

```python
def load_trajectory_records(
    metrics_dir: Path,
    *,
    limit: int = 0,
) -> tuple[list[dict], int, int]:
    """扫描 metrics_dir 下的 rollout_samples.*.jsonl，返回 agentic 记录。

    Returns:
        (records, files_scanned, records_loaded)
    """
```

```python
def summarize_trajectories(
    records: list[dict],
    *,
    tool_filter: str | None = None,
    failure_reason_filter: str | None = None,
) -> dict:
    """从 agentic 轨迹记录中聚合统计指标。

    Returns:
        包含 overview / tool_calls / endings / loss_mask 的统计字典。
    """
```

```python
def _infer_finish_reason(record: dict) -> str:
    """从 JSONL 记录推断 finish_reason。

    推断规则：
    - tool_calls 非空 → "tool_calls"
    - final_answer 为空/"" 且 loss_mask_true == 0 → "length"
    - loss_mask_true == 0 且 final_answer 非空 → "timeout_empty"
    - 否则 → "stop"
    """
```

```python
def _infer_tool_failure(tool_result: dict) -> bool:
    """从 tool_result 推断是否失败。

    推断规则：content 包含 error/exception/traceback/failed 关键字（不区分大小写）。
    """
```

### 4.4 命令注册

在 `areno/cli/main.py` 的 `_COMMANDS` 字典中新增：

```python
"summarize": ("areno.cli.summarize", "summarize_command",
              "Summarize agentic trajectory outcomes from saved run artifacts."),
```

### 4.5 依赖约束

- **不引入新依赖**。表格输出使用 `click.echo` + 简单文本格式化（对齐用
  `str.ljust/rjust`），不依赖 `rich.table`（虽然 `rich>=13` 是现有依赖，但
  保持与 `diagnostics.py` 的 `click.echo` 模式一致更简单）。
- 仅使用标准库（`json`、`pathlib`、`collections`）+ `click`。

### 4.6 向后兼容

- 新增命令，不修改任何现有命令、配置、训练流程或 SDK API。
- 不改变 `_log_agentic_sample_completions` 的写入行为。
- 不改变 `rollout_samples.*.jsonl` 的格式。
- 默认行为完全不受影响。

## 5. 测试需求

### 5.1 测试文件

`tests/test_summarize_cpu.py`（CPU 安全，无需 GPU）

### 5.2 测试矩阵

| 测试类别 | 测试名 | 验证内容 |
|----------|--------|----------|
| **成功路径** | `test_summarize_multi_tool_trajectories` | 多工具轨迹的 tool_calls 聚合、endings 计数、completion_rate 正确 |
| **成功路径** | `test_summarize_json_output_fields` | JSON 输出包含所有必需字段（overview/tool_calls/endings/loss_mask） |
| **成功路径** | `test_summarize_table_output_contains_key_sections` | 表格输出包含 Overview/Tool Calls/Endings/Loss Mask 段落 |
| **过滤** | `test_summarize_tool_filter` | `--tool search` 只统计 search 工具的调用 |
| **过滤** | `test_summarize_failure_reason_filter` | `--failure-reason length` 只统计 length 结尾的轨迹 |
| **非法输入** | `test_metrics_dir_not_found` | 不存在的目录 → `click.UsageError` |
| **非法输入** | `test_metrics_dir_not_a_directory` | 路径是文件不是目录 → `click.UsageError` |
| **非法输入** | `test_negative_limit` | `--limit -1` → `click.UsageError` |
| **边界值** | `test_empty_directory` | 空目录（无 JSONL）→ 正常退出，输出提示信息 |
| **边界值** | `test_no_agentic_records` | 只有 `kind: "rollout"` 的记录 → 0 条 agentic 轨迹 |
| **边界值** | `test_malformed_json_lines` | 部分行 JSON 格式错误 → 跳过坏行，继续处理 |
| **边界值** | `test_limit_truncates_records` | `--limit 5` 只读 5 行 |
| **推断** | `test_infer_finish_reason_stop` | 无 tool_calls + 有 final_answer → "stop" |
| **推断** | `test_infer_finish_reason_tool_calls` | 有 tool_calls → "tool_calls" |
| **推断** | `test_infer_finish_reason_length` | 空 final_answer + loss_mask_true=0 → "length" |
| **推断** | `test_infer_tool_failure_by_content` | tool_result.content 含 "error" → 标记为 failure |
| **推断** | `test_infer_tool_failure_normal_content` | tool_result.content 正常 → 不标记为 failure |
| **隐私** | `test_output_no_raw_prompt_or_completion` | 输出中不含任何 record 的 prompt/messages/final_answer 原文 |
| **集成** | `test_integration_cli_command` | 用 `click.testing.CliRunner` 调用完整 CLI 命令，验证 exit code 0 |
| **集成** | `test_integration_reconciliation` | 各分类计数之和 == total_trajectories |

### 5.3 测试 Fixture

测试使用 `tmp_path` 创建临时 JSONL 文件，不依赖外部数据库或网络。示例 fixture：

```python
AGENTIC_RECORDS = [
    {
        "kind": "agentic", "epoch": 0, "step": 0,
        "prompt_idx": 0, "sample_idx": 0,
        "prompt": "test prompt", "final_answer": "42",
        "tool_calls": [{"name": "search", "arguments": "{}"}],
        "tool_results": [{"name": "search", "content": "result"}],
        "loss_mask_true": 5, "loss_mask_total": 10,
        "messages": [], "loss_mask": [], "tokens": [],
    },
    {
        "kind": "agentic", "epoch": 0, "step": 0,
        "prompt_idx": 0, "sample_idx": 1,
        "prompt": "test prompt", "final_answer": "",
        "tool_calls": [],
        "tool_results": [],
        "loss_mask_true": 0, "loss_mask_total": 10,
        "messages": [], "loss_mask": [], "tokens": [],
    },
]
```

## 6. 验收标准

- [ ] CLI 命令 `areno summarize` 可用，`areno --help` 中可见
- [ ] 支持 `--metrics-dir`、`--json`、`--tool`、`--failure-reason`、`--limit` 选项
- [ ] 输入校验在文件扫描前完成，错误信息明确指向受影响的选项
- [ ] 表格输出包含 Overview、Tool Calls、Endings、Loss Mask 四个段落
- [ ] JSON 输出包含 overview/tool_calls/endings/loss_mask 结构
- [ ] 输出不含原始 prompt/messages/final_answer/tokens 内容
- [ ] 多工具轨迹正确聚合
- [ ] 畸形 JSON 行被跳过不崩溃
- [ ] endings 各分类计数之和 == total_trajectories
- [ ] CPU 测试覆盖成功、非法输入、边界值、推断、隐私、集成
- [ ] 默认行为（不运行 `areno summarize`）完全向后兼容
- [ ] 不引入新依赖
- [ ] 文档包含可运行示例和输出字段说明

## 7. 非目标

- 不修改训练侧代码（`_log_agentic_sample_completions` 不变）
- 不修改 JSONL 写入格式
- 不引入外部数据库或托管服务
- 不自动修改用户配置、删除 artifact 或终止进程
- 不替换 trainer、rollout engine、dashboard 或 SDK 架构

## 8. 示例用法

### 8.1 基本用法

```bash
# 汇总默认 metrics 目录下的 agentic 轨迹
areno summarize

# 指定 metrics 目录
areno summarize --metrics-dir /path/to/run/output

# JSON 格式输出（用于脚本处理）
areno summarize --json

# 只统计 search 工具的调用
areno summarize --tool search

# 只看因 budget/length 截断的轨迹
areno summarize --failure-reason length

# 限制扫描行数（内存控制）
areno summarize --limit 1000
```

### 8.2 输出示例（表格）

```
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
  submit               20         0       0.0%
  Total               160         8       5.0%

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
```

### 8.3 输出示例（JSON）

```bash
$ areno summarize --json --metrics-dir /tmp/test-run
```

```json
{
  "source_dir": "/tmp/test-run",
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
      "search": {"calls": 40, "failures": 3, "failure_rate": 0.075},
      "submit": {"calls": 20, "failures": 0, "failure_rate": 0.0}
    },
    "total": {"calls": 160, "failures": 8, "failure_rate": 0.05}
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
```

## 9. 局限性

1. **finish_reason 和 tool_failure 是推断值**，基于启发式规则从间接信号推导，
   不保证 100% 准确。未来可在 `_log_agentic_sample_completions` 中补充显式字段。
2. **turns 计数依赖 `tool_calls` 数量**，而非实际的 agent 往返次数。一条记录
   可能包含多次 agent-model 交互（通过 `_append_sample_response` 合并），但
   JSONL 中只记录最终合并后的结果。
3. **只读已保存的 JSONL**，如果 `ARENO_LOG_COMPLETIONS=0`（环境变量禁用了
   日志记录），则无数据可汇总。
4. **跨文件去重**不做：如果同一 PID 的 JSONL 被复制到同一目录，会被重复统计。