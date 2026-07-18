# CONTEXT

## 当前目标

在原始 AlphaMaster fork 上做轻量二次开发，重点打通 Windows 本机控制端与 Linux Slurm 服务器之间稳定、可审计、可恢复的联通和交互：本机选择与校验数据，自动上传并提交任务，持续查看状态和日志，支持取消与重启恢复，训练完成后自动下载、校验并发布产物。网页版 GPT 只读取 AlphaMaster 私有仓库，为这一目标编写小工单和硬验收。

## 稳定基线

- 私有工作仓库：`https://github.com/Jinqingchang/AlphaMaster`，分支 `main`，上游为原作者仓库。
- 本轮交接前的本地与远程功能基线：`775dad54acd0bd75e5fc75d58c424536af117e38`；最终交接 SHA 由 GitHub `main` 动态读取，不在提交内自我硬编码。
- 前端三页统一完成于 `186217ecb7c9658187b8a335c1445af576263f02`；相关视觉合同测试 21 项通过，01/02/03 已统一大字体、舒展排版、侧边栏、卡片和控件基线。
- `093e317` 已将 `times.py` 的 Tushare 明文 token 改为只读取本机 `TUSHARE_TOKEN` 环境变量；但旧 Git 历史仍含旧值和历史训练产物，必须先轮换 token，且不得把 `.git` 交给外部模型。

## 已完成功能

- A 股旧数据严格转换与统一数据来源合同。
- 旧 MT5 两阶段可审计登记；新 MT5/OKX 自动 sidecar。
- `timeframe + data_sha256 + run` checkpoint 隔离。
- 策略、checkpoint、Slurm run/result、发布指针和训练包完整身份贯穿。
- 回测独立测试数据、replay / out-of-sample / diagnostic-overlap 和评估数据年化。
- 训练包 v2 路径、成员、大小、CRC、哈希、身份与失败回滚校验。
- StackVM 公式输出已改为固定 200 根、严格因果的滚动 z-score；任意时点只使用当前及历史数据，多品种同一时点有截面分散时保留截面标准化，否则回退到单品种因果时序标准化。
- 公式执行合同已纳入 `VOCAB_VERSION`：旧 normalization 语义的策略、checkpoint、训练包、回测、实时监控和 `--from-scratch` 分数基线全部失败关闭，不能被重新标记为当前版本。
- 主线程相关回归 140 通过、0 失败；独立审查扩大回归 229 通过、0 失败；完整 unit 444 通过、8 个既有失败。
- 因果 normalization 工单受影响回归 155 通过、0 失败；完整 unit 459 通过、7 个既有失败；独立审查经两轮阻塞修复后六维 PASS。
- JUMP 因果工单已将算子改为 200 根当前 bar 包含的稳定滚动样本标准化，并补齐 Web 旧策略和 Slurm 混合 checkpoint 的失败关闭；相关回归 137 通过、0 失败、1 项因本机无 CUDA 跳过，完整 unit 501 通过、8 个既有失败、1 项跳过，独立审查经两轮复验六维 PASS。
- 01/02/03 真实浏览器验证中控制台、页面和 HTTP 错误均为 0。
- 大模型供应商能力正在独立项目中先行修复，本轮 AlphaMaster 不提交供应商候选实现；后续只借鉴经过验证且符合本项目边界的结果。
- 当前优先级重新冻结为 AM 自身的历史成绩真实性：继续审计数据对齐、特征/算子因果性、信号—目标收益—PnL 时间对齐，以及反复试验造成的回测过拟合。
- 已确认但尚未修复：多品种时间交集不足时，`data_pipeline/data_manager.py` 会降级为并集并执行 `ffill().bfill()`；其中 `bfill()` 会把未来首次报价填入更早时点。
- 已确认但尚未修复：`model_core/backtest.py::_turnover_quality()` 对 `tanh` 连续仓位执行 `int(p)`，绝大多数 `(-1, 1)` 仓位被截成 0，交易频率奖励会失真。
- 高优先待审查：训练循环反复使用 walk-forward validation 分数更新冠军、精英池和搜索方向，因此这些 validation 折更接近 selection 数据，不应在没有独立密封测试证据时直接称为最终样本外成绩。
- 待合成序列确认：PnL 使用 `position[t] * target_ret[t]`，但 IC 使用 `factor[t]` 对齐 `target_ret[t+1]`；必须证明两者预测窗口是否一致。

## 本机数据状态

- 2026-07-16 已从当前登录的 MT5 精确品种 `NVDA` 导出 50,000 根已收盘 M5 K 线到 `local_data/MT5_K线数据/AlphaMaster_verified_20260716_142959/`；Parquet 与 manifest 配套，SHA-256 为 `3e2f634780ba3f18505d9c3df6654b78451e8b34d403f80d6f8e24594ffb67b6`，远程训练数据合同校验通过。该目录属于本机原始行情数据，不进入 Git；本次未启动训练。
- `local_data/MT5_K线数据` 指向 `E:\QQ文件\自动挖掘因子项目\MT5_K线数据`。只读批量注册计划扫描 668 个旧文件：537 个可注册、125 个不足 3000 bars、6 个含非法 OHLC / volume；计划保存在 `scratch/legacy_mt5_bulk_plan_20260716.json`，尚未 apply，原目录未新增 sidecar。
- 现有 `local_data/BTCUSDT_H1.parquet` 属于旧 OKX 归档数据，身份为 `okx_legacy_attested`；新版下载器仅保留 OKX `confirm=1` 的已完成 K 线，并生成 `okx` sidecar。

## 已知关键决策与约束

- 登录节点只作 SSH 跳板；远端计算入口在指定计算节点中选择，训练节点由 Slurm 调度。
- 12 CPU 是当前用户配置，尚无同口径性能基准；BTC 两步演示耗时约 17 分 50 秒，因此 9000 步与 30 分钟上限明显不相容，正式预算未决前不得启动。
- 项目级 Stitch 配置只引用 `ALPHAMASTER_STITCH_API_KEY` 环境变量，不保存明文密钥。
- 训练包导出 v2 已实现并审查；Web/API 导入入口仍固定关闭，不得宣称用户可导入。
- manifest 不是 AlphaMaster 算法的固有要求：本地读取 / 本地训练不强制；Slurm 远程训练强制。旧 MT5 / OKX 分别登记为 `mt5_legacy_attested` / `okx_legacy_attested`，不得伪装成新版导出器验证数据。
- 回测允许训练数据与测试数据哈希不同，但必须保持品种、周期、来源族和列合同一致；默认从训练结束后的第一根可用 K 线开始样本外评分，评估数据负责回测年化。
- Slurm run、Worker 结果、策略 JSON 和发布指针必须同时匹配 `data_rows`、`data_start`、`data_end`、`columns`；任何结果或策略范围篡改都不得进入 `READY`。
- AlphaMaster 是完全独立的项目。不得把其他仓库的目标、上下文、源码、任务、提交、文件包或执行流程混入本项目。
- 当前不建设券商订单执行通道；实时分析只负责已收盘 K 线信号展示与可选提醒。
- 当前公式兼容版本为 `v75353a8fc5bc`；仓库内更早版本的历史策略仅保留作旧证据，不能续训、回测或实时使用，后续必须基于新语义重新训练，不做静默迁移。

## Canonical 移交入口

- 项目规则：`docs/CODEX_PROJECT_RULES.md`
- 本机环境与 skills/memory 边界：`docs/LOCAL_EXECUTION_CONTEXT.md`
- 用户问题、需求、修改意见和完成记录：`docs/REQUIREMENTS_CHANGELOG.md`
- 当前验证证据：`docs/VALIDATION_EVIDENCE.md`
- GPT-5.6 移交说明：`docs/GPT5_6SOL_HANDOFF.md`
- 网页版 GPT 总控指令：`docs/GPT_WEB_PRO_EXTENDED_TASK.md`

## 下一步

本轮只提交项目上下文、规则、已确认问题、既有因果修复证据和网页版 GPT 总控指令，不提交大模型供应商候选代码。推送后，网页版 GPT 应从 `docs/GPT_WEB_PRO_EXTENDED_TASK.md` 开始，通过 GitHub MCP 审查未来函数、时间对齐和回测过拟合，并只输出一张最小工单和二元硬验收；本地 Codex 再按真实 skills、memory、进程和环境修订。旧策略已经失效，正式训练预算未确定，本轮不得启动正式训练或小资金实盘。
