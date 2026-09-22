# BudgetTrace：长程 Agent 预算控制系统

[English](README.md) · 中文

BudgetTrace 是面向工具型长程 Agent 的运行时预算控制与审计层。它位于 Agent、模型网关和工具执行器外侧，记录每一步轨迹，核算 token、时间和成本预算，并在预算窗口选择继续、重规划、切换低成本模型或停止。

## 它解决什么问题

长程 Agent 可能重复调用失败工具、在没有新信息时继续消耗预算，也可能在任务即将突破前过早停止。单纯设置 `max_steps` 只能限制最坏成本，不能解释每一步为什么继续或停止。

```text
模型动作 → 工具安全检查 → 事件轨迹与预算账本
                         ↓
             轨迹特征与 BudgetController
                         ↓
       continue / replan / route_cheaper / stop
                         ↓
              RunCard / replay / evaluation
```

## 任务类型与验证场景

BudgetTrace 面向需要多轮模型调用、工具执行和预算决策的长程 Agent 任务。代码修复、结构化检索、数据处理、浏览器操作、客服流程和零售售后都可以复用同一套 Runtime；任务合同、工具集合和评分器按场景替换。

当前 live 证据使用 tau2-retail 的一个换货任务，因为它同时包含身份确认、订单查询、商品查询、差价计算和写操作。这个任务用于验证一类多工具流程，不限定项目的适用范围。

## 主要组件

- `TaskContract` / `RunConfig`：任务、工具、成功条件、评分器和预算约束。
- Runtime：驱动模型、工具和评分器，保证每一步都经过安全边界。
- Event Journal / Budget Ledger：记录模型调用、工具调用、结果、错误、评分和预算变化。
- Features / BudgetController：根据得分变化、新颖度、重复率、错误率和上下文压力做可解释决策。
- Replay / Evaluation：回放 JSONL 轨迹，并对比固定步数、固定预算、朴素停止和 BudgetTrace 规则。
- Live adapter：可选的 OpenAI-compatible 端点适配；凭据只从进程环境变量读取，live 结果默认 `scored: false`。

## 当前验证结果

离线源码验证：

```text
pytest：181 passed
Ruff：All checks passed
mypy：Success
python -m build：Success
```

真实端点验证记录 live 管线、工具协议、预算账本和终止事件；当前输出标记为 `scored: false`：

| 模型 | 代表性结果 | token | 成本估算 |
| --- | --- | ---: | ---: |
| MiniMax-M3 | `success`，模型完成结束动作 | 6,548 | $0.005642 |
| GLM-5.2 | `success`，7 steps，模型完成结束动作 | 20,388 | $0.024678 |
| DeepSeek-v4-pro | `script_exhausted`，未完成结束动作 | 4,429 | $0.006012 |

这些数据来自单任务和少量 trial，成本按环境声明价格估算。业务成功需要确定性 scorer；模型发出 `finish` 只代表运行终态。

离线 P0 矩阵已支持 `--trials` 与 `--model-routes`：5 个任务 × 3 trial × 2 个 ScriptedLLM route × 4 策略共 120 runs，输出业务契约 scorer 的成功率、晚突破误停率和单位成功成本。真实多模型矩阵使用 `live-matrix` 与 `data/public/tau3-retail-v1/model-routes.example.json`；route 标签和价格元数据独立记录，密钥只从 `api_key_env` 读取。

## 目录结构

```text
src/budgettrace/       Runtime、控制器、回放、live adapter
tests/                 离线与适配器测试
data/mini-v1/          合成离线 fixture
data/public/           公开任务 manifest 元数据
docs/                  架构、运行手册、发布检查
.github/workflows/     离线 CI、构建和 secret scan
```

## 快速开始

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q -p no:cacheprovider
python -m ruff check src tests
python -m mypy src/budgettrace
python -m budgettrace replay data/mini-v1/sample_run.jsonl
python -m budgettrace evaluate data/mini-v1/tasks.json --output output/demo-eval
python -m budgettrace evaluate data/mini-v1/tasks.json --trials 3 --model-routes scripted-v1,scripted-v2 --output output/p0-matrix
```

上面命令不调用模型端点。`output/` 是本地生成目录，已加入 `.gitignore`。

`p0-matrix` 会固定运行 5 个任务、4 个策略、3 个 trial 和 2 个离线路由标签，并输出成功率、晚突破误停率和单位成功成本。`scripted-v1` 与 `scripted-v2` 当前复用同一 ScriptedLLM 轨迹，验证的是矩阵统计链路，不是两个真实模型的效果差异。

## 公开任务回放

`data/public/tau3-retail-v1/manifest.json` 只保存公开任务的版本和审计元数据，不 vendoring 上游数据库。准备好本地 tau2-bench checkout 后运行：

```powershell
python -m budgettrace public-audit --source <path-to-tau2-bench> --manifest data/public/tau3-retail-v1/manifest.json --output output/public-audit.json
python -m budgettrace public-replay --source <path-to-tau2-bench> --manifest data/public/tau3-retail-v1/manifest.json --output output/public-replay
```

这验证的是任务适配器和参考动作回放，不是模型成绩。

## 可选 live 验证

```powershell
$env:BUDGETTRACE_LIVE_BASE_URL = "<provider endpoint>"
$env:BUDGETTRACE_LIVE_MODEL = "<model id>"
$env:BUDGETTRACE_LIVE_API_KEY = "<secret in shell only>"
$env:BUDGETTRACE_LIVE_PRICE_INPUT_PER_MTOK = "<input USD per 1M tokens>"
$env:BUDGETTRACE_LIVE_PRICE_OUTPUT_PER_MTOK = "<output USD per 1M tokens>"
$env:BUDGETTRACE_LIVE_TIMEOUT = "180"
python -m budgettrace live-config-check
```

凭据仅从当前进程环境读取，仓库和生成报告不包含凭据值。live 输出是 informational，`success` 默认只表示模型发出了结束动作，不是 benchmark 分数；使用 `reference-actions` scorer 时，结果只表示声明动作契约是否满足。

## 工程成熟度与后续工作

当前项目完成了运行时控制、事件回放、版本化业务契约 scorer、固定任务多 trial/matrix 评测和真实端点管线验证。面向生产系统的下一阶段包括真实业务系统 scorer、持久化和断点恢复、多租户权限、secret manager、写操作幂等与回滚、人工 gate、服务化限流与故障切换、OpenTelemetry、账单对账、压测和灰度发布。

## 安全与隐私

源码包不包含 API key、原始 live journal、供应商响应、上游数据库、个人数据、本机绝对路径、缓存或运行输出。发布前请执行 `docs/release-verification.md` 中的检查；GitHub Actions 还会运行 Gitleaks。

## 文档

- [架构](docs/architecture.md)
- [运行手册](docs/runbook.md)
- [发布检查](docs/release-verification.md)
- [安全策略](SECURITY.md)
- [贡献指南](CONTRIBUTING.md)
- [许可证](LICENSE)
