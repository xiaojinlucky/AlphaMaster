# AlphaMaster 历史实盘方向交接包

更新日期：2026-07-19。本文只保留当时的实盘方向背景，已被 2026-07-23 确认的“大 A 优先、服务器 Slurm 训练 → 回测 → 虚拟信号”主线取代。现役入口是根目录 `CONTEXT.md`、`docs/REQUIREMENTS_CHANGELOG.md` 和 `docs/VALIDATION_EVIDENCE.md` 顶部；下文的市场顺序不再是当前执行计划。

## 1. 用户已确认的目标

AlphaMaster 的长期目标是打通：

```text
训练 → 回测 → 受控全自动实盘
```

市场推进顺序固定为：

```text
加密货币 → 港股/美股 → A 股
```

每个市场的工作只覆盖四个环节：

1. 历史数据与实时行情；
2. 下单、订单状态与持仓管理；
3. 运行状态、异常告警与人工可见的审计记录；
4. 仓位、回撤、止损等风险控制。

成熟、维护良好的社区工具优先于自研。但任何候选工具都必须在实际实施时重新核验官方 API、许可证、维护状态、账户与市场覆盖、模拟盘/沙盒、失败语义和安全边界。

本文件不授权真实交易：没有市场专属工单、模拟盘证据、风险门槛、账户授权和用户明确确认，任何代码都不得连接真实账户、读取真实交易凭据或发单。

## 2. 上游与 fork 的明确边界

用户选择“选择性移植”，而不是整体 merge 上游。

| 区域 | 当前决定 |
|---|---|
| 量化、金融、模型训练、通用数据处理 | 优先与上游逻辑保持一致；需要改动时以小工单移植。 |
| AI 大模型接入 | 保留 fork 定制，不因上游同步覆盖。 |
| Slurm 训练调度 | 保留 fork 定制，不因上游同步覆盖。 |
| 前端页面 | 保留 fork 定制，不因上游同步覆盖。 |
| 实盘接入 | 是后续主线；不能因上游同步被遗忘或删除。 |

本轮明确不移植：数据驱动年化估算、秒/毫秒静默修复、含凭据的 tqsdk/期货实现、前端重试、AI provider 微调和调试脚本。它们不是“永久禁止”，而是需要单独的需求、证据和工单。

## 3. 本地已实施并验证的三张上游同步工单

这些改动已随主交接提交 `4a897de` 发布到远程。目标 GitHub 仓库当前公开，但用户已于 2026-07-20 明确授权本次 36 路精确快照临时公开发布。

| 工单 | 变更 | 已有证据 |
|---|---|---|
| 标准化上游对齐 | 删除 `model_core/formula_contract.py`；`StackVM._normalize_output()` 与 `JUMP` 使用用户选择的上游 expanding 统计；词表版本只由有序 token 名称派生。 | 范围单测、回归、独立六维复验均已完成。 |
| 通达信指数路由 | 指数使用 `get_index_bars`，股票使用 `get_security_bars`；增加 CST 时间解析与异常日期拦截。 | 真实请求验证 `sh000001`、`600519`；模拟未来 K 线验证只删除未形成 bar。 |
| Parquet 文件名 | 在 `parse_parquet_filename()` 入口接受 `60min`、`60m`、`1h`、`5m` 等别名，并归一到标准 token。 | 新增专门单测；下游周期合同保持严格。 |

验证汇总：当前受影响公式、数据与 Web 契约测试 151 通过、0 失败；AI provider 定向测试 28 通过、0 失败。完整 unit 为 513 通过、9 失败；排除本轮相关测试后为 402 通过、相同 9 个失败，未发现本轮新增失败。详细证据见 `docs/VALIDATION_EVIDENCE.md`。

独立审查首轮发现“无条件删除尾部 K 线”这一阻塞问题：它会在收盘后错误丢弃最新已收盘 bar。修复为只删除时间晚于当前时刻的未来 bar 后，同一审查者按需求完整性、逻辑正确性、边界情况、代码质量、测试覆盖和实际运行结果复验为 `PASS`，阻塞项为 0。精确上游比对随后发现 `1m`/月线别名歧义，已按上游语义修复；最终独立复验确认本地 36 路候选为 `PASS`，阻塞项为 0。

上游 expanding 实现的全序列门控（`global_std`、截面/时序标准差阈值）会用整段数据决定走哪条回退分支，因此后续数据可能改变历史前缀的分支选择。用户已经明确优先“按上游一致”；这项上游行为只记录为已知例外，不得仅为消除它重新引入 200-bar 合同。若未来要改变，必须先提出可复现证据和用户决策。

## 4. 当前事实、历史事实与待核验线索

### 当前已核验事实

- 本地提交基线为 `e747eadf78f95777315ea961671c23491d116a7e`。
- 本地查询 `upstream` 的远程 `main` 时得到 `9c42ca0b394f2b42a0fda3b6141137521b38a55b`；本地 tracking ref 可能滞后，任何后续上游决策必须先刷新远程。
- `xiaojinlucky/AlphaMaster` 当前由 GitHub CLI 显示为公开仓库。用户已于 2026-07-20 明确授权本次 36 路精确快照临时公开发布，主交接提交为 `4a897de`；该授权不覆盖新增文件或任何密钥、真实配置、原始数据、模型、数据库和日志。
- 现有发布基线没有真实订单执行能力；Slurm、AI 接入和前端是 fork 定制能力。
- 用户已授权把当前 AlphaMaster 的 AI provider/前端源码与对应测试纳入本次精确快照；真实 `web_settings.json`、环境变量、登录态和 token 继续在仓库外。
- AI provider 定向测试已通过，但本轮未做官方 Codex CLI 的受控真实调用或真实浏览器页面验证；它仍需在不泄露登录态、不触发未授权外部行为的条件下完成端到端验收。

### 仅作待复核线索，不是当前结论

- 外部模型提供的上游提交数量、PR 状态、星标/许可证数字、社区稳定性评价、数据源可用性结论、DSR 数字、五层过拟合防护是否充分，以及“没有需要保护的旧 200-bar 产物”。
- 候选名称如 CCXT、IBKR、lite-qmt-executor、RiskGuard、mx-risk-guard、Prometheus/Grafana/Alertmanager。它们只能触发后续资料核验，不能直接被称为“最优”或写入生产依赖。
- `/mnt/results/...` 外部产物路径不属于本项目工作区，不能作为本仓库可验证证据。

## 5. 尚未开始的工作与授权门槛

| 事项 | 当前状态 | 开始前必须具备 |
|---|---|---|
| 加密货币实盘 | 未开始 | 交易所选择、官方 API 核验、模拟盘、风控规则、告警、资金与账户授权。 |
| 港股/美股实盘 | 未开始 | 券商选择、市场/账户覆盖、模拟盘、时区与交易时段、风控与人工恢复方案。 |
| A 股实盘 | 未开始 | 券商/miniQMT 路线核验、T+1/涨跌停/黑名单规则、模拟盘或受控验证、风控与账户授权。 |
| 过拟合与时间语义问题 | 已登记，未由本文件裁决 | 需从当前代码、合成序列和独立测试提出最小修复工单；不得凭旧结论直接改核心算法。 |

## 6. 给外部总控与本机 Codex 的协作协议

1. 网页版 GPT 只读取本仓库已提交的 `main`，不读取本机、其他项目、凭据、原始数据、`.git` 或未提交改动。默认应为私有仓库；当前公开仅适用于用户 2026-07-20 授权的这份精确快照。
2. 它负责复原事实链、排优先级、写一张最小工单和二元硬验收；不直接修改代码、创建分支、运行训练或授权真实交易。
3. 每个阻塞建议必须写明：违反的原始需求、代码/证据、实际风险、最小修复、基线失败测试、修复后硬验收与禁止扩大范围。无法给出这三类依据的意见只能列为可选优化或放弃。
4. 本机 Codex 必须重新核验 Git 状态、当前代码、skills、Windows/Slurm/数据/凭据和官方 API；外部模型的工单不是直接执行授权。
5. 本机实现完成后，独立审查线程只依据原始目标和硬验收进行六维复验；只修复阻塞项，循环到阻塞项为 0。

网页版 GPT 的可复制启动指令在 `docs/WEB_GPT_CONTROLLER_PROMPT.md`。

## 7. 本次发布快照的精确范围

用户授权将当前 AlphaMaster 的源代码、测试、规则与交接材料一起打包；本次范围固定为以下 36 个文本路径。用户已于 2026-07-20 额外授权临时公开发布这份精确清单；清单不得扩大。

### 选择性上游同步（12 个）

```text
data_pipeline/parquet_manager.py
model_core/formula_contract.py                         [删除]
model_core/ops.py
model_core/vm.py
model_core/vocab.py
tests/unit/test_checkpoint_identity.py
tests/unit/test_formula_execution_contract.py          [删除]
tests/unit/test_jump_causality.py
tests/unit/test_parquet_filename.py                    [新增]
tests/unit/test_training_package_security.py
tests/unit/test_vm_causal_normalization.py
web/data_sources/tongdaxin_source.py
```

### AI / 前端 fork 定制（14 个）

```text
README.md
tests/unit/test_ai_provider_frontend_contract.py       [新增]
tests/unit/test_ai_provider_registry.py                [新增]
tests/unit/test_ai_provider_route.py                   [新增]
tests/unit/test_ai_providers_aliases.py
tests/unit/test_codex_subscription.py                  [新增]
web/ai_analyze.py
web/ai_providers.py
web/app.py
web/codex_subscription.py                              [新增]
web/settings.py
web/static/app.js
web/static/index.html
web_settings.example.json
```

### 上下文、规则与交接包（10 个）

```text
CONTEXT.md
docs/CODEX_PROJECT_RULES.md
docs/AlphaMaster新手使用指南.md
docs/GPT5_6SOL_HANDOFF.md
docs/GPT_WEB_PRO_EXTENDED_TASK.md
docs/LOCAL_EXECUTION_CONTEXT.md
docs/LIVE_TRADING_DIRECTION_HANDOFF.md                 [新增]
docs/REQUIREMENTS_CHANGELOG.md
docs/VALIDATION_EVIDENCE.md
docs/WEB_GPT_CONTROLLER_PROMPT.md                      [新增]
```

明确排除：`.env`、真实 `web_settings.json`、`ai_analysis_history.json`、原始行情/交易/训练数据、数据库、checkpoint、模型权重、日志、临时测试环境和已跟踪但未修改的 `training_time_XAUUSD.json`。正式发布时必须再次检查 staged 清单、内容类型、文件大小、密钥扫描、用户对公开远程的本次授权和实时 SHA；严禁使用 `git add .` 或 `git add -A`。
