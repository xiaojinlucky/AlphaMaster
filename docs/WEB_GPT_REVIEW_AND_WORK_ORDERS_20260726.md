# 网页版 GPT 审核与本机本地化工单（2026-07-26）

## 1. 本文边界

审查对象：

- 当前规划快照：`e0fb283afeeb783d53371bc52071aaccad1d5091`
- 目标实现提交：`b3bbefef6aa14ca314a3f3db6ad8babb9d595ea2`
- 目标实现范围：FreeStockDB D1 数据合同与 Slurm 来源 sidecar 身份链

网页版 GPT 对目标实现提交给出 `PASS`，对本阶段系统给出 `RETURN`。它的
首轮输出没有完成强制文件补读；唯一一次纠偏也明确承认无法再次调用 GitHub
只读接口。因此，本文只吸收其方向和反方意见，不把它描述为完整独立代码审计。

本机 Codex 已按当前仓库、真实 Skills 和三路只读对抗审查完成本地化。本轮
只形成工单，没有执行任何工单，没有连接 SSH/Slurm、写数据库、恢复队列、
运行训练/回测、下载数据或揭示 sealed OOS。BioMNI 不参与。

## 2. 证据等级

| 对象 | 当前可确认到哪里 | 仍需现场核验 |
|---|---|---|
| FreeStockDB / Slurm sidecar | 固定提交中的来源合同、上传/finalize、Worker 物化、严格 loader 等代码机制属于 `PUBLIC_COMMIT_VERIFIED` | 真实 run 的成功、失败、恢复、重试和 published 路径属于 `LOCAL_CODEX_VERIFY` |
| Slurm `570548` | 完成状态、退出码和身份哈希来自提交文档，只是 `HANDOFF_CLAIM_ONLY` | 真实结果包、下载、后处理和本地队列状态属于 `LOCAL_CODEX_VERIFY` |
| `255 / 949 / 273 / 27 / 649` | 这些数字来自提交文档，只是 `HANDOFF_CLAIM_ONLY` | G 盘清单、权重文件和覆盖报告复算属于 `LOCAL_CODEX_VERIFY`；`649` 表示尚未导出到 v2，不等于源库缺失 |
| A 股虚拟执行器 | T+1、整手、零股卖出、先卖后买、显式停牌/涨跌停输入、费用、滑点和现金约束的代码机制属于 `PUBLIC_COMMIT_VERIFIED` | 历史时点成交状态源、生产 replay 和真实测试结果属于 `LOCAL_CODEX_VERIFY` |
| 账本 | `PortfolioDecisionLedger` 与 `SignalLedger` 的代码、静态校验和幂等机制属于 `PUBLIC_COMMIT_VERIFIED` | 生产订单、成交和账户持久化尚未闭环；真实 SQLite 内容属于 `LOCAL_CODEX_VERIFY` |
| sealed OOS v2 | campaign v2、统一成本、策略/数据哈希、一次性揭示锁和禁止覆盖的代码机制属于 `PUBLIC_COMMIT_VERIFIED` | 真实策略是否冻结、封存数据是否未揭示、锁注册表和最终评估结果属于 `LOCAL_CODEX_VERIFY` |

测试源码只证明测试已经编写，不证明测试实际运行通过。README、CONTEXT、
CHANGELOG 和验证说明中的运行数字仍是交接说法，不能升级为公开代码验证。

## 3. 最短依赖

```text
OPS-01（独立，只读现场审计）

RND-01 PIT 合同与覆盖矩阵
  → RND-02 历史代码小样本真实性检查
  → RND-03 649 个未导出代码的覆盖补证/扩展

并行：
  RND-04A 历史成交状态源与 PIT 语义审计
  RND-04B 现有账本的执行持久化缺口设计
  RND-05A 现有 sealed OOS v2 静态贯通审计

RND-03 + RND-04A + RND-04B
  → RND-04C 动态 PIT 组合 replay
  → RND-05B sealed OOS 端到端贯通复验
```

RND-04 必须按 A/B/C 拆开。原网页版工单把历史成交状态和生产账本列为
replay 前置条件，却没有任何工单产出这两项；这是本机对抗审查发现并修正的
唯一依赖 blocker。

## 4. 本地化工单

以下 6 张工单在本轮都只有设计和审查授权，没有实现、数据写入、远程操作、
训练、回测、下载、队列恢复或 sealed OOS 揭示授权。各单中的“目标、复用
模块、硬验收”描述未来获批后的执行合同，不表示已经开始执行。

### OPS-01：Slurm `570548` 结果身份链审计

- 裁决：`接受`
- 能力映射：`monitor-strategy-run`；仅在基础设施故障时使用
  `codex-remote-ops`
- 复用模块：`web/slurm_training_manager.py`、
  `web/slurm_training_client.py`、`web/training_batch.py`、
  `scripts/slurm_control.py`、`scripts/train_slurm_worker.py`
- 目标：只读核验 job、run、源码、数据、result manifest、产物哈希、
  下载可见性和后处理证据，产出唯一可解释的审计结论
- 非目标：不恢复队列、不写 SQLite、不重提作业、不修改模型或冻结远端运行时
- 硬验收：结果身份、源码身份、数据身份和运行身份能够唯一关联；缺失或冲突
  必须明确标为未知/失败，不能自动写成 `READY`
- 停止条件：结果包缺失、哈希不一致、manifest 无法关联或现场状态不唯一
- 授权：本轮没有执行授权；任何状态写入、Web 恢复或剩余队列续派都另行决定

### RND-01：历史沪深300 PIT 合同与覆盖矩阵

- 裁决：`接受`
- 能力映射：`build-universe` + `explore-data`
- 复用模块：`portfolio_manager/universe.py::UniverseContract`、
  `data_pipeline/free_stockdb_data.py`、`data_pipeline/dataset_contracts.py`、
  `data_pipeline/parquet_manager.py`
- 目标：复算已有 255 个权重时点，提出可知时间、生效交易日、退出语义和
  代码规范化合同，生成 949 代码覆盖/隔离矩阵及 2009-12-31 的 298 只异常报告
- 非目标：不新建第二套股票池系统，不破坏 A50 静态可信哈希，不批量训练
- 硬验收：每个时点、代码、权重、来源、可知/生效区间和覆盖状态都有可复算
  输入哈希；追加未来月份不能改变历史查询
- 停止条件：权重来源或可知/生效时点无法由权威证据唯一确定
- 授权：本轮只批准工单设计，不批准实现或数据写入

### RND-02：历史代码覆盖真实性小样本

- 裁决：`接受并改写`
- 能力映射：`explore-data`，辅以 `build-universe`
- 目标：从历史退出现任池中选择可解释小样本，验证代码映射、复权、来源
  manifest、时间范围和 loader；作为 RND-03 的可逆门
- 非目标：不跑完整沪深300组合，不新增生产模块，不把隔离样本冒充全量失败
- 硬验收：每只样本给出可用/隔离/未导出原因与输入输出哈希
- 依赖：RND-01

### RND-03：649 个未导出代码的覆盖补证与扩展

- 裁决：`接受并改写`
- 能力映射：`explore-data` + `build-universe`
- 目标：依据 RND-02 的真实抽取结果，逐步补齐或隔离 v2 尚未导出的历史代码，
  生成 949 代码的最终覆盖状态与原因矩阵
- 非目标：不把“未导出”写成“源库缺失”，不替换 FreeStockDB，不另建
  provider、sidecar、上传或 Worker 身份链
- 硬验收：所有代码都有可复算来源、范围、质量、导出/隔离状态和哈希证据
- 停止条件：v2 导出范围与真实源覆盖不一致且原因无法唯一解释
- 依赖：RND-01 → RND-02

### RND-04：历史成交状态、账本适配与动态组合 replay

- 裁决：`改写并拆分`
- 能力映射：`build-universe` + `configure-trade-execution` + `build-rule`；
  产物完成后用 `monitor-strategy-run` 审计
- 复用模块：`portfolio_manager/universe.py`、`controller.py`、
  `calibration.py`、`pipeline_adapter.py`、`execution.py`、`ledger.py` 与
  `web/signal_ledger.py`
- RND-04A：核验历史停牌、ST、涨跌停、交易日历、费用与公司行动等时点状态
  的来源和可知语义
- RND-04B：设计现有 `PortfolioDecisionLedger` / `SignalLedger` 与
  `PortfolioExecutionResult` 的订单、成交、账户持久化适配；不得新建第三套账本
- RND-04C：在 RND-03、04A、04B 都通过后，将动态 PIT 股票池接入现有
  controller/execution/ledger 做确定性组合 replay
- 非目标：不新增撮合器、不建立第二个 portfolio manager、不把 replay
  描述成真实交易或 sealed OOS
- 硬验收：信号日、可用信息日、执行日、行情状态、账户前后状态和订单身份
  可复算；崩溃恢复幂等
- 停止条件：历史成交状态不能按时点重建，或账本不能幂等绑定决策、账户、
  订单和成交身份

### RND-05：现有 sealed OOS v2 贯通审计

- 裁决：`接受并改写`
- 能力映射：`audit-strategy-spec` + `audit-runtime-semantics` +
  `monitor-strategy-run`
- 复用模块：`evaluation/sealed_oos_campaign.py`、`run_backtest.py` 及对应测试
- RND-05A：现在即可只读审计现有 v2 合同、50 只统一门槛、成本合同、
  数据/策略身份和全局一次性揭示锁是否有静态断点
- RND-05B：只有 RND-04C 通过后，才复验训练、选择、replay 与最终 OOS
  身份是否端到端贯通
- 非目标：不新建 sealed OOS 子系统，不修改成绩，不揭示封存数据
- 硬验收：训练流程无法读取封存数据；同一密封字节换标签、窗口、合同或策略
  仍只能揭示一次；结果绑定冻结策略、数据、成本和运行身份
- 停止条件：sealed 数据已经揭示，或 reveal lock、策略、数据与运行身份不一致
- 授权：真正揭示会永久消耗该密封数据身份，必须另行明确授权

## 5. 禁止重复建设

不得重建 DatasetManifest、FreeStockDB sidecar、上传/finalize、Worker 物化、
严格 loader、Slurm 身份链、撮合器、第三套 SQLite、平行 Universe 合同或
新的 sealed OOS 合同。现有模块只补真实断点。

## 6. 当前状态

- 目标实现提交：`PASS`
- 系统阶段：`RETURN`
- 本地化工单：已形成，未执行
- 外部审核：仅网页版 GPT；BioMNI 已退役
