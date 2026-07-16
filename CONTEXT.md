# CONTEXT

## 当前目标

把 AlphaMaster 的真实代码、上下文、项目规则、用户问题/需求/修改意见、已完成内容和验证证据收敛为 GitHub 可直接读取的 canonical 文档链，供网页版 GPT 与 PA_Agent 私有仓库一起做只读总控规划。用户的高层核心目标是 PA_Agent 打通到实盘；AlphaMaster 继续作为独立研究/训练项目，不自动成为 PA 的执行层。

## 稳定基线

- 私有工作仓库：`https://github.com/Jinqingchang/AlphaMaster`，分支 `main`，上游为原作者仓库。
- 当前已推送功能基线：`d4dcb75279387c282e13b56e20e843d6801d4065`。
- 前端三页统一完成于 `186217ecb7c9658187b8a335c1445af576263f02`；相关视觉合同测试 21 项通过，01/02/03 已统一大字体、舒展排版、侧边栏、卡片和控件基线。
- `093e317` 已将 `times.py` 的 Tushare 明文 token 改为只读取本机 `TUSHARE_TOKEN` 环境变量；但旧 Git 历史仍含旧值和历史训练产物，必须先轮换 token，且不得把 `.git` 交给外部模型。

## 已完成功能

- A 股旧数据严格转换与统一数据来源合同。
- 旧 MT5 两阶段可审计登记；新 MT5/OKX 自动 sidecar。
- `timeframe + data_sha256 + run` checkpoint 隔离。
- 策略、checkpoint、Slurm run/result、发布指针和训练包完整身份贯穿。
- 回测独立测试数据、replay / out-of-sample / diagnostic-overlap 和评估数据年化。
- 训练包 v2 路径、成员、大小、CRC、哈希、身份与失败回滚校验。
- 主线程相关回归 140 通过、0 失败；独立审查扩大回归 229 通过、0 失败；完整 unit 444 通过、8 个既有失败。
- 01/02/03 真实浏览器验证中控制台、页面和 HTTP 错误均为 0。

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
- PA_Agent 是实盘主项目；AlphaMaster 只作为独立研究项目和审计模式参考。未经另行授权不接入 PA 信号或订单生命周期。

## Canonical 移交入口

- 项目规则：`docs/CODEX_PROJECT_RULES.md`
- 本机环境与 skills/memory 边界：`docs/LOCAL_EXECUTION_CONTEXT.md`
- 用户问题、需求、修改意见和完成记录：`docs/REQUIREMENTS_CHANGELOG.md`
- 当前验证证据：`docs/VALIDATION_EVIDENCE.md`
- GPT-5.6 移交说明：`docs/GPT5_6SOL_HANDOFF.md`
- 网页版 GPT 总控指令：`docs/GPT_WEB_PRO_EXTENDED_TASK.md`

## 下一步

验证文档事实与链接后，按精确文件范围提交私有 GitHub。用户随后在网页版 GPT Pro Extended Thinking 中粘贴总控指令，并把工单+硬验收通过 `@` 当前对话交回；本地 Codex 再按真实 skills、memory、代码、环境和测试进行修订。旧 MT5 批量计划未获 apply 授权，正式训练预算未决，均不得执行。
