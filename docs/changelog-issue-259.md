# 变更文档：Summarize Agentic Trajectory Outcomes in the Terminal (#259)

## 概述

新增 `areno summarize` CLI 命令，从 agentic RL 训练产生的本地 JSONL 制品中
聚合轨迹统计指标（turns、tool calls、tool failures、endings、completion rate、
loss mask 覆盖率），在终端以表格或 JSON 格式输出。支持按工具名和失败原因过滤。

## 动机

AReno 的 agentic RL 训练循环通过 `_log_agentic_sample_completions`
（`areno/api/trainers/policy_only.py:475`）将 agent 轨迹样本以 `kind: "agentic"`
的 JSONL 行写入 `rollout_samples.{pid}.jsonl`。但训练完成后，用户缺少内置工具来
汇总这些轨迹的行为表现。本变更为此提供一个标准化的 CLI 命令。

## 变更范围

### 新建文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `areno/cli/summarize.py` | ~373 | CLI 命令 + 聚合逻辑（纯函数，无 torch 依赖） |
| `tests/test_summarize_cpu.py` | ~446 | CPU 测试套件，31 个测试用例 |
| `docs/cli/summarize.rst` | ~180 | 面向用户的 RST 文档 |
| `docs/requirements-issue-259.md` | ~350 | 需求设计文档（Markdown） |
| `docs/changelog-issue-259.md` | — | 本变更文档 |

### 修改文件

| 文件 | 变更行数 | 说明 |
|------|----------|------|
| `areno/cli/main.py` | +1 | `_COMMANDS` 字典新增 `"summarize"` 条目 |
| `docs/reference/cli.rst` | +1 | CLI 参考索引新增 `:doc:/cli/summarize` |

### 未修改文件

以下文件**未做任何改动**，确保向后兼容：

- `areno/api/` 下所有文件（训练侧、配置、SDK 接口）
- `areno/engine/` 下所有文件（引擎、worker、推理/训练运行时）
- `pyproject.toml`（无新依赖）
- `areno/cli/train.py` / `serve.py`（不改变现有 CLI 选项）

## 功能说明

### CLI 用法

```bash
# 汇总默认 metrics 目录下的 agentic 轨迹
areno summarize

# 指定 metrics 目录
areno summarize --metrics-dir /path/to/run/output

# JSON 格式输出
areno summarize --json

# 按工具名过滤
areno summarize --tool search

# 按失败原因过滤（length / timeout_empty / tool_error）
areno summarize --failure-reason length

# 限制扫描行数（内存控制）
areno summarize --limit 1000
```

### CLI 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--metrics-dir` | PATH | `/tmp/areno/tfevent` | 扫描 `rollout_samples.*.jsonl` 的目录 |
| `--json` | flag | `False` | 输出 JSON 格式而非人类可读表格 |
| `--tool` | TEXT | 无 | 只统计指定工具名的调用 |
| `--failure-reason` | TEXT | 无 | 只统计指定失败原因的轨迹 |
| `--limit` | INT | `0` | 限制扫描的 JSONL 行数（0 = 无限制） |

### 输出字段

表格和 JSON 输出均包含四个段落：

1. **Overview** — total_trajectories, total_turns, avg_turns, completion_rate, completed
2. **Tool Calls** — per-tool calls/failures/failure_rate, 以及 total 汇总
3. **Endings** — stop / tool_calls / length / timeout_empty 四种结尾的计数和百分比
4. **Loss Mask** — avg_loss_tokens, avg_total_tokens, avg_loss_coverage

### 隐私安全

输出**只包含聚合统计**，不输出任何原始 `prompt`、`messages`、`final_answer`、
`tokens`、`loss_mask` 内容。

### 推断规则

JSONL 制品中没有显式 `finish_reason` 或 `tool_failure` 字段，以下推断规则基于对
`areno/api/agentic.py` 源码的阅读：

**finish_reason**（按优先级检查）：

1. `tool_calls` 非空 -> `"tool_calls"`（agent 仍在调用工具）
2. `tool_calls` 空 + `final_answer` 空 + `loss_mask_true == 0` -> `"length"`（budget 截断）
3. `tool_calls` 空 + `loss_mask_true == 0` + `final_answer` 非空 -> `"timeout_empty"`
4. 否则 -> `"stop"`（正常完成）

**tool_failure**：`tool_results[i].content` 包含 `error`、`exception`、`traceback`、
`failed`、`failure` 关键字之一（不区分大小写）。

## 架构设计

```
summarize_command (Click 入口)
    |
    v
_validate_inputs  --- 输入校验（目录存在性、limit 非负、failure-reason 合法值）
    |
    v
load_trajectory_records  --- 逐行扫描 JSONL，过滤 kind=="agentic"，跳过坏行
    |
    v
summarize_trajectories  --- 推断 + 聚合统计，返回结构化 dict
    |
    v
_print_table / json.dumps  --- 输出
```

核心函数均为纯函数，无副作用，可独立测试。

## 依赖约束

- **不引入新依赖**。仅使用 `click`（现有依赖）+ 标准库（`json`、`pathlib`、
  `collections`、`typing`）。
- `summarize.py` 内联了 `DEFAULT_METRICS_LOG_DIR = "/tmp/areno/tfevent"` 常量，
  而非从 `areno.api.defaults` 导入，以避免触发 `areno.api.__init__` -> torch
  的重依赖导入链。该常量与 `areno/api/defaults.py:3` 保持一致。

## 向后兼容

- 纯新增 CLI 命令，不修改任何现有命令、配置、训练流程或 SDK API。
- 不改变 `_log_agentic_sample_completions` 的写入行为。
- 不改变 `rollout_samples.*.jsonl` 的格式。
- 默认行为完全不受影响——不运行 `areno summarize` 时无任何副作用。

## 测试

### 测试文件

`tests/test_summarize_cpu.py`（CPU 安全，无需 GPU）

### 测试覆盖（31 个用例）

| 类别 | 用例数 | 覆盖内容 |
|------|--------|----------|
| 成功路径 | 5 | 多工具聚合、completion rate、endings 对账、JSON 字段、loss mask 统计 |
| 过滤器 | 3 | tool filter、failure-reason length、failure-reason timeout_empty |
| 非法输入 | 4 | 目录不存在、不是目录、负数 limit、无效 failure-reason |
| 边界值 | 5 | 空目录、无 agentic 记录、畸形 JSON、limit 截断、limit=0 无限制 |
| 推断 | 7 | finish_reason 4 种场景 + tool_failure 3 种场景 |
| 隐私 | 2 | JSON 输出不泄露原始数据 + 表格输出不泄露原始数据 |
| 集成 | 5 | CLI 表格输出、JSON 输出、reconciliation、tool filter CLI、mixed records |

### 测试运行结果

在本 session 中通过手动测试运行器（pytest stub + `CliRunner`）实际执行，
**31/31 通过**。

**注意**：本机环境为 Python 3.9（不满足项目要求的 3.10+），pytest 因网络问题
未能安装。`pytest tests/test_summarize_cpu.py` 本身未在标准 pytest 框架下运行。
在满足 Python 3.10+ 和 pytest 的 Linux 环境中应可直接运行。

## 验收标准对照

| 验收标准 | 状态 | 说明 |
|----------|------|------|
| multi-tool + malformed + bounded memory + privacy + table/JSON + reconciliation | 达标 | 31 个测试覆盖全部子项 |
| 使用现有 contracts，无外部数据库 | 达标 | 仅用 click + 标准库 |
| 默认行为向后兼容 | 达标 | 纯新增命令，不改动现有代码 |
| 自动化测试覆盖 success/invalid/boundary | 达标 | 31/31 通过（pytest 框架本身未运行） |
| 用户文档含可运行示例和输出说明 | 达标 | `docs/cli/summarize.rst` 已创建并注册到索引 |

## 文件清单

### 新建

```
areno/cli/summarize.py
tests/test_summarize_cpu.py
docs/cli/summarize.rst
docs/requirements-issue-259.md
docs/changelog-issue-259.md
```

### 修改

```
areno/cli/main.py          (+1 行: _COMMANDS 新增 summarize 条目)
docs/reference/cli.rst     (+1 行: 索引新增 :doc:/cli/summarize)
```

## 已知局限

1. **finish_reason 和 tool_failure 是推断值**，基于启发式规则从间接信号推导，
   不保证 100% 准确。
2. **turns 计数依赖 `tool_calls` 和 `tool_results` 数量**，而非实际的 agent
   往返次数。`_append_sample_response` 合并的多轮交互在 JSONL 中只记录最终结果。
3. **只读已保存的 JSONL**，如果 `ARENO_LOG_COMPLETIONS=0` 禁用了日志记录，则无
   数据可汇总。
4. **跨文件去重不做**：如果 JSONL 文件被复制到同一目录，会被重复统计。
5. **pytest 未在标准环境下运行**：测试逻辑已验证正确，但 `pytest
   tests/test_summarize_cpu.py` 命令本身未在 pytest 框架下执行。