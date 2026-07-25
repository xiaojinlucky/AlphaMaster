> **历史审查快照（已被取代，不得执行）**：本文件描述的是 `d4dcb75`
> 之前的未提交候选与当时阻断项。下文角色、任务和修复要求只作历史证据。
> 现行入口是根目录 `CONTEXT.md`、`docs/WEB_GPT_CONTROLLER_PROMPT.md`
> 和 `docs/VALIDATION_EVIDENCE.md`。

# AlphaMaster — GPT-5.6 Pro Extended Thinking 深度交接

## 1. 你的任务

你是 AlphaMaster 候选实现的独立架构与代码审查者。不要默认当前方案正确，也不要直接给泛化建议。请从原始目标出发，对需求完整性、逻辑正确性、边界情况、代码质量、测试覆盖和实际运行结果六个方面做严格审查，并给出能够交回开发线程执行的最小修复方案。

当前交接的核心不是训练出策略，而是判断下面这组数据身份与产物同源设计是否正确、是否能安全进入稳定主线。

## 2. 精确快照

- 当前目标仓库：`xiaojinlucky/AlphaMaster`（当前为公开仓库）
- 稳定分支：`main`
- 代码审查基线：`093e3172a24f1e4f7621e590b2641c4c5c61e83d`
- 前端完成提交：`186217ecb7c9658187b8a335c1445af576263f02`
- `PACKAGE_METADATA.json` 中的 `workspace_head_commit` 是打包时实际 HEAD；`TRACKED_DIFF.patch` 始终相对上述代码审查基线生成，并验证该基线是实际 HEAD 的祖先
- 当前候选：实际 HEAD 及其本地工作区相对代码审查基线的差异；不得把它当作已发布实现
- 交接入口：先读根目录 `CONTEXT.md`、`lessons.md`、`README.md`，再读本文件和候选差异

源码包不含 `.git`。原因不是节省空间，而是 Git 历史包含一个已经从当前 HEAD 删除、但尚未净化的旧 Tushare token，以及历史 checkpoint、训练 ZIP 和日志。旧 token 必须在服务端撤销/轮换；未经用户单独授权，不得改写历史。

## 3. 稳定完成的内容

### 前端

01 模型训练、02 策略回测、03 实时分析已经按同一视觉基线统一：大字体、较高行高、舒展间距、统一侧边栏、卡片、标题栏、控件和按钮。完成提交为 `186217e`，视觉合同测试 21 项通过。

### 当前版本凭证处理

`times.py` 不再保存 Tushare 明文 token，只从 `TUSHARE_TOKEN` 环境变量读取；`.env.example` 只保留占位符。该修复已在 `093e317` 推送，但历史泄露仍需轮换和单独的历史治理决策。

## 4. 待审查候选实现

### A 股旧数据转换

候选新增 `data_pipeline/a_share_data.py` 和 `scripts/convert_a_share_parquet.py`：

- 只接受 `stocks` 目录下 `6位代码_(5min|15min|60min|daily).parquet`
- 映射为 `M5/M15/H1/D1`
- 严格要求列顺序为 `time/open/high/low/close/tick_volume`
- 不静默排序、去重、补行或转换有损数值
- 从旧 1000 秒桶反演上海交易时段 bar-close UTC 秒
- 原始文件只读，输出规范副本和 sidecar manifest
- manifest 记录来源、行数、时区、年化周期、最低 bars、源/目标 SHA-256 和 `dataset_id`

请重点质疑：旧时间编码是否真的能唯一、可靠地重建所有历史交易日；交易所节假日、半日市、停牌、早期夏令时和不同周期的完整性合同是否合理。

### 数据身份与 checkpoint

候选把身份合同贯穿到 Parquet 加载、训练、checkpoint、策略、Slurm manifest、发布指针和训练包：

- `dataset_id = sha256:<data_sha256>`
- checkpoint 路径为 `checkpoints/<timeframe>/<data_sha256>/<run_id>/ckpt_<symbol>_step_<N>.pt`
- `--from-scratch` 新建隔离 run，旧检查点保留
- 旧版扁平 checkpoint 不允许被静默续训
- Slurm 产物按同一 run 和同一数据身份发布

### 回测合同

候选 `run_backtest.py` 要求策略记录的数据身份与回测 Parquet 完全一致。这个规则能阻止误用数据，但也会禁止正常样本外、向前和新时间窗口回测。请不要把“严格”直接等同于“正确”，需要重新定义：

- 训练数据身份
- 评估数据身份
- 二者允许不同的前提
- 必须一致的字段（品种、周期、字段合同、特征版本、时间不重叠规则等）
- 报告中如何同时记录两份数据证据

### 训练包 v2

候选训练包实现了成员 allowlist、路径校验、数量/大小上限、CRC、逐文件 SHA-256、checkpoint/策略/历史身份一致性、暂存验证和失败回滚。导出候选可用；`web/app.py` 的导入 API 仍固定返回 403，因此不能把内部导入函数描述成用户已可用功能。

## 5. 已知阻断项

### 5.1 旧训练入口与新 checkpoint 合同冲突

`main.py` 和 `train_single.py` 创建 `AlphaEngine` 后没有设置完整数据身份，仍搜索旧版扁平 checkpoint。候选 `model_core/engine.py` 在保存和加载时强制完整身份，因此旧入口会在首次保存时失败，且无法续训新版嵌套 checkpoint。

请判断最简正确路径：迁移这两个入口、显式弃用并早失败，还是收敛为唯一 `train_file.py` 入口。不能继续保留“看似可用、训练到中途才失败”的状态。

### 5.2 回测身份合同可能破坏量化研究目标

当前完全相同哈希的要求只支持训练集重放，不支持样本外验证。这是产品与研究语义问题，不是单纯补一个测试即可解决。

### 5.3 Windows 长路径原子发布未闭环

训练包读取 checkpoint 已使用 Windows 扩展路径，但最终原子发布仍有普通路径操作。在较长项目根目录下，合法包可能在发布阶段失败。现有长路径测试只覆盖读取，没有覆盖最终发布和回滚。

### 5.4 多数据身份下 UI/导出可能选错 checkpoint

本地 `web/progress.py` 会汇总同一 symbol 在所有数据哈希目录下的 checkpoint 并按 mtime 取最新；训练包导出复用这个结果。同一品种存在多份数据时，UI 可能显示错误身份的进度，导出也可能与策略身份冲突。

## 6. 当前验证证据

- 候选数据身份/Slurm/训练包测试 139 项通过；前端视觉合同 9 项通过；合计 148 项通过。精确文件为 `test_a_share_data.py`、`test_checkpoint_identity.py`、`test_run_backtest_contract.py`、`test_slurm_training_manager.py`、`test_train_file_cli.py`、`test_train_slurm_worker.py`、`test_training_package_security.py` 和 `test_frontend_visual_contract.py`。命令为 `python -m pytest tests/unit/<上述8个文件> -q --basetemp=<项目 scratch 下唯一目录>`，运行环境为项目 `.venv` 的 Windows Python 3.11。
- 整个 `tests/unit` 在同一环境最终复跑：385 项通过、8 项失败。命令为：`python -m pytest tests/unit -q --tb=short -ra --basetemp=<项目 scratch 下唯一目录>`。
- 8 项失败集中在算子/特征数量断言、Runner 本机策略状态和固定旧时间的训练计时测试；从文件差异看它们似乎与当前候选无关，但尚未在干净 HEAD 副本中复核，不能称为已确认的历史失败。它们没有被候选修复，也不能计入通过。
- 没有启动训练，没有执行 TradingView 配置，没有用 Mock 结果冒充真实运行。

## 7. 安全边界

不得要求或假定源码包中存在以下内容：

- `.env`、token、密码、私钥、证书、券商账户信息
- 原始行情、交易、训练数据或数据库
- checkpoint、模型权重、训练 ZIP、训练历史、策略运行态
- 本机日志、缓存、虚拟环境、`scratch`、`local_runs`、`published_training`
- `.git` 历史

项目级 `.codex/config.toml` 可以存在，因为它只引用环境变量 `ALPHAMASTER_STITCH_API_KEY`，不保存明文值。

## 8. 希望你输出的结果

请按以下顺序给出：

1. 一句话总判断：当前候选是否可以进入稳定主线。
2. 会改变结论或实现的缺陷清单：每项含影响、证据、根因、最小修法和剩余风险。
3. 对“训练身份 vs 评估身份”的正式数据模型与允许关系。
4. 对旧训练入口的唯一推荐处理方式及迁移策略。
5. Windows 长路径、安全导入和多身份 checkpoint 选择的最小实现方案。
6. 应新增或修改的测试清单，覆盖成功、失败、并发、长路径、跨身份和回滚。
7. 分阶段开发顺序；每阶段都应可独立验证，不要大爆炸重构。
8. 明确哪些结论仍需要真实样本、真实 Windows 运行或真实 Slurm 运行才能验证。

不要修改业务内容、前端视觉、TradingView 或训练参数。不要启动正式训练。若你认为当前方案大方向错误，请直接给出更简单、可证明的替代方案。
