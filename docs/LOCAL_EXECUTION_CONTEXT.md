# AlphaMaster — 本地执行环境、Skills 与 Memory 摘要

> **历史环境快照（截至 2026-07-25），不是当前运行态。**
> 当前状态只看根目录 `CONTEXT.md` 与
> `docs/WORK_ORDERS_CLAUDE_20260726.md`；本文件仅帮助外部审查者理解
> 本机能力和早期环境背景。

网页版 GPT 无法看到本机真实环境。本文件提供与 AlphaMaster 直接相关的脱敏快照；任何实际执行前，本地 Codex 仍须实时复核。

## 1. 本机与仓库

- 操作系统：Windows。
- 项目路径：`D:\Desktop\Quant\AlphaMaster`。
- Python：项目 `.venv`，当前主验证环境为 Python 3.11。
- Web：本机回环地址 `127.0.0.1:8765`；桌面快捷方式只启动控制台，不会自动提交训练。
- 目标发布仓库：`xiaojinlucky/AlphaMaster`，`main` 跟踪 `origin/main`。用户已明确要求仓库始终保持公开，以便网页版 GPT 直接读取；不得再把改为私有作为发布前提。密钥、真实配置、原始数据、数据库、模型、checkpoint、训练历史、原始日志和机器可消费的运行态文件仍是硬阻断项；只允许脱敏、带时间戳且不能作为控制输入的摘要证据。当前 goal 内提交与推送已有持续授权，但仍逐次精确暂存并检查 staged 快照。
- 原作者仓库：`rosemarycox5334-debug/AlphaMaster`，本机只读 remote 为 `upstream`。

## 2. 远程训练环境

- Windows 负责选择数据、提交、查看日志和下载结果。
- 登录节点只作 SSH `ProxyCommand` 跳板。
- 健康计算入口在固定别名中动态选择；实际训练节点由 Slurm 调度。
- 2026-07-25 23:24 +08:00 的状态查询是：本机 Web 未运行，
  `squeue -u jinqc` 无该用户的活动作业；
  SQLite 仍显示 22 `READY`、1 个过期 `TRAINING`、22 `QUEUED`。过期项
  `600276` / Slurm `570548` 已由 `sacct -X` 确认完成且退出码为 0，
  但结果下载和后处理尚未核验。恢复前必须先对账，不能重复提交或直接把它
  写成 `READY`。
- 共享远端 Git HEAD 为
  `1d44177496a87a8d881cdc87e26af5835cdaab18`，当前训练源码 SHA-256 为
  `02350ba4e3c696461f2be9f042cb00912568442b3f9986151a79621f571eeb8d`；
  剩余 22 项是否续派尚未决定，不能先覆盖该运行时。
- 首次尝试 Windows SQLite URI 只读参数时，参数被错误解析并在仓库根生成
  0 字节空文件 `=ro`；该文件随后移入
  `scratch/runtime_audit_sqlite_uri_artifact_20260725.empty`，没有改写
  SQLite、行情或训练文件。后续查询改用 `sqlite3 -readonly`。
- 当前默认申请 12 CPU，但没有与 8 CPU 同口径的稳定性能结论。
- 当前正式默认 `9000` 步与 `00:30:00` 时限明显不相容；预算未确认前不得启动正式训练。
- Linux Worker 使用独立 Python 3.11 和依赖锁，不安装 MetaTrader5 或本机凭据。

## 3. 本机数据状态

- `local_data/MT5_K线数据` 指向仓库外的旧 MT5 数据目录。
- 旧 MT5 批量注册计划共扫描 668 个文件：537 个可登记，125 个不足最低 bars，6 个违反 OHLC/volume 合同。
- 批量计划尚未 apply；旧目录没有因此新增 sidecar。
- 已从当前登录 MT5 导出一份 50,000 根、已收盘的 `NVDA_M5` 验证数据及 sidecar；原始行情不进入 Git。
- `local_data/BTCUSDT_H1.parquet` 是旧 OKX 归档身份 `okx_legacy_attested`；新版下载器只保留 `confirm=1` 的已完成 K 线。
- Git 当前仍跟踪历史运行态文件 `training_time_XAUUSD.json`。新移交包必须显式排除；本轮不擅自删除或改写历史。

## 4. 与本项目相关的本机 Skills

截至 2026-07-25，`.agents/skills` 与 `.claude/skills` 各有 29 个同名
Junction，全部可解析到本机共享 Skill；两边清单一致。这些入口被
`.git/info/exclude` 排除，不在公开 Git 快照中。网页版 GPT 看不到，也不能
直接调用。

AlphaMaster 当前也不是 OpenXQuant workspace：仓库没有
`.open-xquant/workspace.yaml`。下列 OpenXQuant Skills 用来提供量化语义、
审计门和任务路由，本机 Codex 必须把它们映射到 AlphaMaster 的现有模块、
数据和测试；不得照抄其中的 `uv run oxq ...` 命令。

| 本机能力 | 当前用途 | 外部计划交回后的本地映射 |
|---|---|---|
| `skill-scope-router` | 从任务文字定位真实项目和最小 Skill 集 | 必须先路由；返回 `no-match` 时不自动创建项目配置或反复改写任务重试 |
| `open-xquant` | 项目定位后的量化任务总路由 | 识别真实缺口，再加载对应叶子 Skill |
| `build-universe` + `explore-data` | 历史成分/权重、时点股票池、数据覆盖与质量 | 映射到沪深300已有 255 个权重时点、949 个历史代码和 FreeStockDB 数据资产 |
| `configure-trade-execution` + `build-rule` | 成交时钟、费用、滑点、整手、T+1、停牌和涨跌停语义 | 先审计并复用现有 A 股虚拟执行内核，只补时点状态数据、生产账本和未闭环路径 |
| `audit-strategy-spec` + `audit-runtime-semantics` | 用户决定、训练/验证/OOS 语义和运行时一致性 | 借用审计门，映射到 AlphaMaster 的冻结合同、manifest、Worker 和 sealed OOS |
| `monitor-strategy-run` | 运行身份、产物哈希、可复现性和研究偏差证据 | 优先复用已完成真实 run，验证 `result → checkpoint → run → dataset → source` |
| `browser:control-in-app-browser` | 本机 Web 真值、控制台和 replay 标签验收 | 静态源码或公开页面不能代替本机真实运行验收 |
| `codex-remote-ops` | Windows 到远程节点链路排障 | 只处理基础设施故障，不替代 Slurm 训练合同 |
| `neat-freak` | 阶段收口、文档对齐和临时产物清理 | 只在真实阶段完成后使用 |
| `meta-ship` | 高风险量化改动的 Git 审查 | 仍服从本项目的精确文件确认、公开仓库和禁止擅自开分支规则 |

旧版文档列出的 `github:github`、`github:yeet` 当前不可调用，不能成为工单
前置条件。GitHub 事实、密钥扫描和发布由本机 Git、实时远程检查和
`meta-ship` 适配完成。

缺少 `.open-xquant/workspace.yaml` 只说明 AlphaMaster 不是 OpenXQuant
workspace；不得因此自动执行 init、migrate、生成 spec 或改变仓库结构。
只有用户另行明确要求把项目迁移为 OpenXQuant workspace 时才讨论该动作。

## 5. 与本项目有关的既定偏好

- 用户要求真实文件、真实测试、真实渲染和真实运行证据，不接受只给抽象建议。
- 前端最重要的视觉方向是大字体、较高行高、舒展留白和三页一致。
- 用户要求明确区分“原作者既有机制”和 fork 后新增机制。
- manifest 是 fork 后为 Slurm 增加的远程身份合同，不得说成原仓库算法要求。
- 不启动未经确认的正式训练，不配置本轮未授权的 TradingView。
- 完成重要开发后，必须循环独立严格审查直到阻塞项清零。
- AlphaMaster 与任何其他仓库完全隔离，不做跨项目分析、提交、打包或执行。
- AI 大模型接入属于 AlphaMaster 的 fork 定制边界；用户已授权把当前 AlphaMaster 的 provider/前端源码和测试纳入本次精确快照。不得把其他项目的源码、运行态或提交直接混入本仓库，真实配置与凭据仍不提交。
- 网页版 GPT 当前优先审查两条独立主线：A50 恢复对账，以及
  沪深300 PIT → 历史覆盖 → 动态组合 replay → 成交状态 → sealed OOS。
  它看不到本机未提交工作区、真实训练数据、进程和远程状态，所有执行
  命令与环境假设必须由本地 Codex 重新核验。
- 外部规划者负责方向、架构、依赖和反方意见；本机 Codex 负责现场核验、Skill 路由、现有实现去重、代码、测试、真实数据、Slurm、Browser 和 Git。外部工单不是执行授权。
- 用户要求进攻式推进：关键路径一旦被真实证据确定，主线立即开工，独立证据线并行；不再用泛化风险清单、重复调研或“一次只做一张最小工单”拖慢整体完工。

## 6. 外部工单的本地进攻式落地协议

1. 一次刷新 Git、进程、Web、Slurm、真实数据和测试基线；将外部结论标为
   `接受`、`改写`、`合并` 或 `删除`。
2. 对照现有模块和产物去重。已经存在的身份链、虚拟执行内核、sealed OOS
   合同或数据冻结机制只做缺口审计，不从零重建。
3. 先经 `skill-scope-router` 定位项目和最小 Skill 集，再进入
   `open-xquant` 与叶子 Skill。将接受的方向映射到真实模块、测试和启动
   方式；不存在的 Skill、命令、路径或 OpenXQuant workspace 命令直接删掉。
4. 运营恢复和研究研发两条主线立即并行；身份链证据、测试复现、页面真值等
   独立工作也不等待另一条主线结束。
5. 只有不可逆操作、会改变结果语义的业务分叉、或真实证据缺失足以改变
   结论时才停下询问。已经授权范围内的可逆本地实现和只读验证不重复请示。
6. 禁止 Mock、静默换源、降低测试门槛、把 replay 冒充样本外或真实交易、
   中断现役训练、跨项目混入，以及无关重构。
7. 每个阶段用真实文件、真实测试、真实数据或真实运行产物验收；计划和代码
   完成不等于系统完成。

当前两条独立交付主线如下：

```text
运营恢复线
  570548 结果包/下载/后处理对账
    → 本地队列状态收敛
    → 复核冻结运行时后决定 Web 恢复与剩余 22 项续派

研究研发线
  现有 255 时点 PIT 合同 + 949 代码覆盖矩阵
    → 历史退出现任池小样本抽取验证
    → 649 未导出代码的覆盖扩展与隔离报告
    → 扩展 portfolio_manager/universe.py::UniverseContract
    → 动态历史股票池接入现有组合控制/执行器 replay
    → 时点成交状态与账本生产化
       （复用 portfolio_manager/ledger.py、web/signal_ledger.py）
    → sealed OOS 与端到端可复现验收

并行证据线
  现有身份链真实 run 审计 → monitor-strategy-run
  测试与 Web 真值固化 → pytest + Browser

路由
  PIT/覆盖 → build-universe + explore-data
  成交/账本 → configure-trade-execution + build-rule
  OOS/运行 → audit-strategy-spec + audit-runtime-semantics
             + monitor-strategy-run
```
