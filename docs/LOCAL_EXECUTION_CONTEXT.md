# AlphaMaster — 本地执行环境、Skills 与 Memory 摘要

网页版 GPT 无法看到本机真实环境。本文件提供与 AlphaMaster 直接相关的脱敏快照；任何实际执行前，本地 Codex 仍须实时复核。

## 1. 本机与仓库

- 操作系统：Windows。
- 项目路径：`D:\Desktop\Quant\AlphaMaster`。
- Python：项目 `.venv`，当前主验证环境为 Python 3.11。
- Web：本机回环地址 `127.0.0.1:8765`；桌面快捷方式只启动控制台，不会自动提交训练。
- 私有仓库：`Jinqingchang/AlphaMaster`，`main` 跟踪 `origin/main`。
- 原作者仓库：`rosemarycox5334-debug/AlphaMaster`，本机只读 remote 为 `upstream`。

## 2. 远程训练环境

- Windows 负责选择数据、提交、查看日志和下载结果。
- 登录节点只作 SSH `ProxyCommand` 跳板。
- 健康计算入口在固定别名中动态选择；实际训练节点由 Slurm 调度。
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

| Skill | 实际用途 | 外部计划的适配要求 |
|---|---|---|
| `browser:control-in-app-browser` | 本机 Web 页面、控制台错误、布局和运行状态验证 | 外部模型不能把静态代码检查当成真实页面验收 |
| `neat-freak` | 对照代码收敛 CONTEXT、README、docs 和规则 | 外部计划应引用 canonical 文档，不继续追加冲突快照 |
| `github:yeet` | 精确 Git 范围、密钥扫描、私有远程校验和发布 | 本项目禁止未经确认开分支/PR，实际发布按项目 Git 规则适配 |
| `codex-remote-ops` | Windows 到远程 Codex/节点链路排障 | 仅在远程基础设施故障时使用，不替代 Slurm 训练合同 |

## 5. 与本项目有关的既定偏好

- 用户要求真实文件、真实测试、真实渲染和真实运行证据，不接受只给抽象建议。
- 前端最重要的视觉方向是大字体、较高行高、舒展留白和三页一致。
- 用户要求明确区分“原作者既有机制”和 fork 后新增机制。
- manifest 是 fork 后为 Slurm 增加的远程身份合同，不得说成原仓库算法要求。
- 不启动未经确认的正式训练，不配置本轮未授权的 TradingView。
- 完成重要开发后，必须循环独立严格审查直到阻塞项清零。
- AlphaMaster 与任何其他仓库完全隔离，不做跨项目分析、提交、打包或执行。

## 6. 外部工单转为本地任务的检查表

1. 刷新当前 Git HEAD、工作区、暂存区、进程和本机 Web 状态。
2. 将每一步映射到真实模块、类、配置字段、测试文件和启动方式。
3. 核对外部模型引用的 skill、命令和路径在本机是否存在。
4. 涉及 MT5、OKX、Slurm、SSH、PyTorch 或第三方 API 时重新查官方资料/实际接口。
5. 补回外部计划容易遗漏的凭据隔离、数据身份、幂等、未知状态、重启恢复和失败关闭。
6. 删除无关重构、TradingView 配置、正式训练、券商接入和任何跨项目内容。
7. 每张工单写清入口、禁止行为、测试证据、实际运行证据、用户确认点和回退。
8. 本地 Codex 修订后的工单再由用户确认执行。
