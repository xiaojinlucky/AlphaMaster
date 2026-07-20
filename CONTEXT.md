# CONTEXT

## 当前目标

在原始 AlphaMaster fork 上做轻量二次开发，长期目标是打通“训练 → 回测 → 受控全自动实盘”的完整链路，市场顺序为加密货币 → 港股/美股 → A 股。当前基础设施仍是 Windows 本机控制端与 Linux Slurm 服务器之间稳定、可审计、可恢复的联通和交互：本机选择与校验数据，自动上传并提交任务，持续查看状态和日志，支持取消与重启恢复，训练完成后自动下载、校验并发布产物。

当前提交不接入真实账户、不发单、不保存交易凭据。网页版 GPT 只读取 AlphaMaster 的已提交 GitHub 快照，充当总控和工单设计者；本机 Codex 才能在真实环境中复核并执行。

## 2026-07-19 当前方向与上游同步边界

- 用户确认采用“选择性移植”，永不整体 merge 上游：量化、金融、模型训练与通用数据处理优先保持上游逻辑；AI 大模型接入、Slurm 训练调度、前端页面和实盘接入方向必须保留。
- 本轮三张上游同步工单已在本地工作树完成：StackVM/JUMP 改为上游 expanding 标准化并删除 formula contract；通达信指数改走指数接口；Parquet 文件名入口支持常见周期别名。
- 本轮已完成本地范围测试、真实通达信请求和独立六维复验。2026-07-20 用户明确授权将当前公开仓库作为临时发布目标；36 路精确快照已随 `4a897de` 推送到 `main`，`.env`、真实配置、密钥、数据、模型和日志未上传。
- 不移植上游的数据驱动年化估算、秒/毫秒静默修复、tqsdk 凭据/期货实现、前端重试和 AI provider 微调，除非用户为相应区域另开工单。

## 稳定基线

- 目标发布仓库：`https://github.com/xiaojinlucky/AlphaMaster`，分支 `main`，上游为原作者仓库。默认要求私有发布；该仓库当前公开，但用户已于 2026-07-20 仅对本次 36 路精确快照明确授权临时公开发布，主交接提交为 `4a897de`。后续发布仍须重新取得用户确认。
- 本轮选择性上游同步开始前的本地与远程功能基线：`e747eadf78f95777315ea961671c23491d116a7e`；最终交接 SHA 由 GitHub `main` 动态读取，不在提交内自我硬编码。
- 前端三页统一完成于 `186217ecb7c9658187b8a335c1445af576263f02`；相关视觉合同测试 21 项通过，01/02/03 已统一大字体、舒展排版、侧边栏、卡片和控件基线。
- `093e317` 已将 `times.py` 的 Tushare 明文 token 改为只读取本机 `TUSHARE_TOKEN` 环境变量；但旧 Git 历史仍含旧值和历史训练产物，必须先轮换 token，且不得把 `.git` 交给外部模型。

## 已完成功能

- A 股旧数据严格转换与统一数据来源合同。
- 旧 MT5 两阶段可审计登记；新 MT5/OKX 自动 sidecar。
- `timeframe + data_sha256 + run` checkpoint 隔离。
- 策略、checkpoint、Slurm run/result、发布指针和训练包完整身份贯穿。
- 回测独立测试数据、replay / out-of-sample / diagnostic-overlap 和评估数据年化。
- 训练包 v2 路径、成员、大小、CRC、哈希、身份与失败回滚校验。
- 历史 200 根滚动标准化与公式执行合同的修复证据仍保留在 Git 历史和验证文档中；用户随后决定以原作者上游的 expanding 标准化为当前语义，因此运行时代码已删除 `formula_contract.py`。
- 当前 `StackVM._normalize_output()` 与 `JUMP` 使用 expanding 统计，`VOCAB_VERSION` 只由有序 token 名称派生；旧版带执行合同的产物版本不再匹配当前词表版本。
- 主线程相关回归 140 通过、0 失败；独立审查扩大回归 229 通过、0 失败；完整 unit 444 通过、8 个既有失败。
- 当前受影响公式、数据与 Web 契约测试 151 通过、0 失败；AI provider 定向测试 28 通过、0 失败。完整 unit 为 513 通过、9 个既有失败；排除本轮相关测试后为 402 通过、相同 9 个失败。
- 通达信真实请求已验证 `sh000001` 与 `600519`；`drop_forming=True` 只会移除未来未形成 K 线，不会丢掉最新已收盘 K 线。
- 01/02/03 真实浏览器验证中控制台、页面和 HTTP 错误均为 0。
- AI 大模型接入是 AlphaMaster 的 fork 定制边界。用户已明确授权把当前 AI provider/前端源码和对应测试一起纳入本次精确快照；`.env`、真实 `web_settings.json`、浏览器/客户端会话、token 和其他运行态继续排除。
- 当前优先级重新冻结为 AM 自身的历史成绩真实性：继续审计数据对齐、特征/算子因果性、信号—目标收益—PnL 时间对齐，以及反复试验造成的回测过拟合。
- 2026-07-20 已修复并随 `67e81f3` 推送：多品种时间交集不足时仅做 `ffill()`，统一裁掉没有历史报价的前段；裁剪后不足 `MIN_BARS` 会失败关闭。定向单元 11 通过、时间轴属性 1 通过，独立六维审查 `PASS`。
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
- 当前发布基线没有券商订单执行通道；实盘是后续主线，但必须逐市场、逐工单完成模拟盘/风控/监控与用户授权后才能接真实账户或发单。
- 当前公式兼容版本为 `v9217a2c0d91a`；仓库内更早版本的历史策略仅保留作旧证据，不能续训、回测或实时使用，后续必须基于当前语义重新训练，不做静默迁移。

## Canonical 移交入口

- 项目规则：`docs/CODEX_PROJECT_RULES.md`
- 本机环境与 skills/memory 边界：`docs/LOCAL_EXECUTION_CONTEXT.md`
- 用户问题、需求、修改意见和完成记录：`docs/REQUIREMENTS_CHANGELOG.md`
- 当前验证证据：`docs/VALIDATION_EVIDENCE.md`
- GPT-5.6 移交说明：`docs/GPT5_6SOL_HANDOFF.md`
- 当前方向交接：`docs/LIVE_TRADING_DIRECTION_HANDOFF.md`
- 网页版 GPT 总控指令：`docs/WEB_GPT_CONTROLLER_PROMPT.md`
- 历史未来函数专项总控指令：`docs/GPT_WEB_PRO_EXTENDED_TASK.md`

## 下一步

主交接提交 `4a897de` 已完成 staged 密钥、类型、大小和远程 SHA 复核并推送。网页版 GPT 现在可从 `docs/WEB_GPT_CONTROLLER_PROMPT.md` 启动，只输出最小工单和二元硬验收。本地 Codex 必须结合真实 skills、memory、进程、数据和环境复核后才执行。正式训练预算、模拟盘门槛和真实账户授权仍未确定。
