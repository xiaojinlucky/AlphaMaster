# AlphaMaster

基于深度神经网络强化学习的量化因子挖掘中心：从 Parquet / MT5 / OKX K 线自动搜索可解释因子公式，支持 Web 端训练、回测与实时信号分析。

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

![Web 控制台总览](docs/images/00_hero.png)

目标发布仓库：[github.com/Jinqingchang/AlphaMaster](https://github.com/Jinqingchang/AlphaMaster)；原作者仓库保留为只读 `upstream`：[rosemarycox5334-debug/AlphaMaster](https://github.com/rosemarycox5334-debug/AlphaMaster)。默认只向私有仓库发布；当前仓库公开，但用户已于 2026-07-20 明确授权临时发布本次已审查的 36 路快照，主交接提交为 `4a897de`。`.env`、真实配置、密钥、数据、模型和日志始终不上传，后续发布仍须重新确认。

当前数据身份、checkpoint、旧 MT5 注册、样本外回测和训练包 v2 已随 `d4dcb75` 发布并通过独立严格审查。接手或让外部模型规划前，先读 [`CONTEXT.md`](CONTEXT.md) 和 [`docs/GPT5_6SOL_HANDOFF.md`](docs/GPT5_6SOL_HANDOFF.md)。

网页版 GPT Pro Extended Thinking 的 AlphaMaster 专属总控入口为 [`docs/GPT_WEB_PRO_EXTENDED_TASK.md`](docs/GPT_WEB_PRO_EXTENDED_TASK.md)。当前轻量二次开发重点是打通 Windows 本机控制端与 Linux Slurm 服务器之间的训练提交、状态/日志交互、恢复和产物回传。

---

## 它做什么

AlphaMaster 把「挖因子」做成一条可操作的流水线：

1. **训练**：用强化学习在特征 + 算子空间里搜索公式，按验证集表现选优  
2. **回测**：用 `tanh(因子)` 连续仓位在历史行情上模拟交易，看资金曲线与绩效  
3. **实时分析**：按周期收盘后重算信号，展示方向与把握；方向转折可推飞书提醒  

公式以 token 序列保存（如 `strategies/best_BTCUSDT.json`），可用 StackVM 解释执行，训练 / 回测 / 实时共用同一套信号逻辑。

---

## Web 控制台（推荐入口）

```bash
pip install -r requirements.txt
python run_web.py --port 8765
```

浏览器打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。界面分三步：

第一次使用建议先读 [`docs/AlphaMaster新手使用指南.md`](docs/AlphaMaster新手使用指南.md)。

Windows 已可使用桌面 `AlphaMaster` 快捷方式：双击后自动启动本机 Web 并打开浏览器；若服务已经运行，只会再次打开页面，不会重复启动。快捷方式对应的命令为：

```powershell
.venv\Scripts\python.exe run_web.py --host 127.0.0.1 --port 8765 --open-browser
```

服务窗口默认最小化在任务栏；关闭该窗口即可停止本机 Web，不会取消已经提交到 Slurm 的训练任务。

| 步骤 | 作用 |
|------|------|
| **01 模型训练** | 选 Parquet、开始 / 重新训练、看曲线与日志、导出策略与检查点 |
| **02 策略回测** | 选策略 JSON，设手续费 / 滑点，看绩效与资金曲线 |
| **03 实时分析** | 多数据源监控，收盘后更新信号；可选飞书转折提醒 |

### 模型训练

![训练页](docs/images/01_train.png)

- Parquet 命名：`{品种}_{周期}.parquet`，例如 `BTCUSDT_H1.parquet`、`XAUUSD_H1.parquet`  
- **本地后端**：普通续训只在当前数据身份的最新 run 中寻找最高 step；重新训练会新建隔离 run 并从头训练，旧 run 保留
- **manifest 边界**：AlphaMaster 核心读取本地 Parquet 不依赖 manifest；Slurm 远程训练必须用 sidecar 绑定文件哈希、来源和数据范围。旧 MT5 文件可在页面一次确认后自动注册，新版 MT5 / OKX 导出器会自动生成
- **Slurm 第一阶段**：每个 run 独立训练；网络中断会重连同一 run/job，跨 run checkpoint 续训尚未开放；每次 `READY` 会以单一原子指针发布该 run 的 checkpoint、训练历史和策略整套产物，不跨 run 混用
- 展示最优分数、验证分数、训练曲线与最优公式；可选 AI 分析当前训练情况  

### AI 模型供应商

训练页可直接切换 Codex 订阅、DeepSeek、Kimi 和小米 MiMo，并按模型控制 Thinking 与推理强度。DeepSeek、Kimi、MiMo 默认从仓库外的 `D:\Desktop\Quant\env` 读取 `DEEPSEEK_*`、`MOONSHOT_*`、`MIMO_*`；也可在界面填写并保存在本机 `web_settings.json`（该文件不得提交）。Codex 不需要 API Key，使用官方 Codex CLI 当前的 `codex login` 登录。

Codex 的订阅可用性、额度、限流和计费取决于当时的 ChatGPT 计划与官方 rate card；不得假定“永远不按 token/额度计费”。部署或使用前请核对 [OpenAI 的 Codex 计划说明](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan) 与 [当前 rate card](https://help.openai.com/en/articles/20001106-codex-rate-card)。AlphaMaster 会在空临时目录运行它，禁用 Shell、网页、应用、MCP 与子 Agent，只接收最终文本。

AI 通道采用固定白名单，不读取浏览器、IDE、Electron、WorkBuddy、QClaw 或其他客户端的登录会话和 token。旧的 OpenClaw/WorkBuddy 通道与 Key 别名已停用。

### 策略回测

![回测页](docs/images/02_backtest.png)

- 仓位：`position = tanh(factor)`，信号越强仓位越大  
- 成本：手续费 + 滑点（默认约 0.02% / 0.01%）  
- 数据：策略 JSON 与测试 Parquet 分开选择；同一数据执行训练集重放，不同哈希但同品种 / 周期 / 来源族的数据自动按训练结束时间切出样本外评分区间
- 年化：回测指标使用测试数据实际覆盖范围推导的年化周期，同时在报告中保留训练与测试两套数据身份
- 输出：总收益、夏普、索提诺、盈亏比、滚动夏普与资金曲线  

![资金曲线示例](docs/images/04_equity.png)

### 实时分析

![实时分析页](docs/images/03_realtime.png)

- 数据源：MT5 / OKX 等（以界面可用源为准）  
- **只在当前周期 K 线收盘后**重新判断；未收盘 bar 不参与信号  
- 卡片展示方向（看涨 / 看跌 / 不确定）与把握程度  
- 可选飞书 Webhook：仅在方向转折时推送文字提醒  

---

## 项目结构

```
AlphaMaster/
├── web/                 # FastAPI Web UI（训练 / 回测 / 实时）
├── model_core/          # 特征、算子、StackVM、训练引擎、回测评分
├── data_pipeline/       # Parquet / MT5 K 线加载与对齐
├── strategy_manager/    # 信号与仓位状态逻辑；真实 trader 已移除
├── execution/           # MT5 实时报价兼容层，不含下单适配器
├── backtest_viz/        # 回测引擎与图表
├── strategies/          # best_{symbol}.json 策略文件
├── checkpoints/         # 训练检查点
├── run_web.py           # 启动 Web 控制台
├── train_file.py        # CLI：从单个 Parquet 训练
└── requirements.txt
```

---

## 环境要求

- Python **3.10+**（建议 3.11）  
- PyTorch、pandas、FastAPI、uvicorn 等（见 `requirements.txt`）  
- 可选：MetaTrader 5 终端（实时 MT5 行情；当前仓库不提供可用下单适配器）
- 复制 `.env.example` 为 `.env` 填写 MT5 等凭证（`.env` 已 gitignore）  

```bash
pip install -r requirements.txt
# 可选可视化等：pip install -r requirements-optional.txt
```

本仓库同时提供可复现锁文件：

```powershell
# Windows：Web、MT5 导出与本机控制端
uv venv --python 3.11
uv pip sync --python .venv\Scripts\python.exe requirements-windows.lock

# 开发与测试环境
uv pip sync --python .venv\Scripts\python.exe requirements-dev.lock
```

Linux Slurm Worker 使用独立的 `requirements-linux-worker.lock`，不安装 `MetaTrader5`、Web 服务或本机凭据。

---

## Windows Parquet（MT5 / OKX）→ Slurm CPU 训练

第一阶段支持本机 MT5 已收盘 K 线导出或经过来源清单校验的 OKX Parquet，通过本机 Web 控制台提交到 Slurm。`login-node` 仅作为 SSH 配置中的跳板，不运行命令；每次远端操作先调用节点选择器确定健康的交互入口，真正的训练节点始终由 Slurm 调度。

```powershell
# 1. 从当前已登录的 MT5 导出准确品种/周期；默认排除未收盘 bar
.venv\Scripts\python.exe scripts\export_mt5_parquet.py --symbol XAUUSD --timeframe H1 --bars 50000

# 也可下载 OKX 主流永续合约；仅保存 confirm=1 的已完成 K 线
.venv\Scripts\python.exe download_okx_klines.py --out "D:\OKX_K线数据"

# 中断后续跑：只跳过已有且 Parquet + sidecar 哈希与已完成 K 线合同均有效的文件
.venv\Scripts\python.exe download_okx_klines.py --out "D:\OKX_K线数据" --resume

# 2. 复制并检查配置；不要把真实凭据提交到 Git
Copy-Item .env.example .env

# 3. 仅在回环地址启动本机 Web
.venv\Scripts\python.exe run_web.py --host 127.0.0.1 --port 8765
```

Slurm 模式固定使用 `cpu` 分区、`normal` QOS 和服务器项目 Python 3.11 环境；失败会原样暴露，不会退回本机训练。上传数据、源码身份和下载产物均通过 manifest 与 SHA-256 校验。详细设计见 [`docs/slurm-deployment-design.md`](docs/slurm-deployment-design.md)。

当前正式任务默认申请 12 CPU；实际训练节点仍由 Slurm 根据集群可用性分配，不固定节点。

`TRAINING_BACKEND` 必须显式配置；缺失或未知值会拒绝启动。只有明确设置 `TRAINING_BACKEND=local` 才允许本机训练。

### 旧 MT5 数据如何接入

旧 Parquet 没有 sidecar 不代表数据格式错误。远程训练前可用两阶段注册把“用户确认来源的精确文件字节”记录为旧数据身份；注册不会修改 Parquet，也不会冒充新版导出器验证的数据。

```powershell
# 第一步只读扫描；--source-report 可重复传入
.venv\Scripts\python.exe scripts\register_legacy_mt5_parquet.py plan `
  --input-dir "D:\K线数据" `
  --source-report "D:\K线数据\_bulk_sync_report.json" `
  --source-report "D:\K线数据\_bulk_sync_retry_report.json" `
  --feed-id "legacy_mt5_bulk_20260709" `
  --output-plan "scratch\legacy_mt5_plan.json"

# 第二步必须使用计划输出的精确 plan_sha256，并明确确认来源
.venv\Scripts\python.exe scripts\register_legacy_mt5_parquet.py apply `
  --plan "scratch\legacy_mt5_plan.json" `
  --plan-sha256 "<plan_sha256>" `
  --acknowledge-source MetaTrader5 `
  --output-report "scratch\legacy_mt5_report.json"
```

批量计划允许部分文件被拒绝：不足 `3000` bars、字段 / 时间 / 数值合同不合格的文件不会生成 manifest。Web 页面对当前选中的单个旧 MT5 文件提供同一套显式确认入口。

---

## 常用命令

```powershell
# Web 控制台
python run_web.py --port 8765

# CLI 训练（自动续训；加 --from-scratch 则重新训练）
python train_file.py --data-file D:\K线数据\BTCUSDT_H1.parquet
python train_file.py --data-file D:\K线数据\BTCUSDT_H1.parquet --from-scratch
python train_file.py --data-file D:\K线数据\XAUUSD_H1.parquet --from-scratch --train-steps 20

# 回测可显式选择独立测试数据；auto 会自动判断重放或样本外
python run_backtest.py --single --strategy-file strategies\best_BTCUSDT.json `
  --data-file D:\K线数据\BTCUSDT_H1_future.parquet --evaluation-mode auto
```

策略输出默认在 `strategies/best_{symbol}.json`。

---

## 信号口径（训练 / 回测 / 实时一致）

- 因子经 StackVM 算出标量序列  
- `position = tanh(factor)` ∈ (-1, 1)  
- `|position|` 小于阈值时视为无信号（观望）  
- 实时侧只用**已收盘** K 线，避免盘中抖动与回测不一致  

---

## 截图更新

仓库内展示图是 2026-07-12 的历史参考；当前三页统一版以提交 `186217e` 的代码和视觉合同测试为准。需要刷新展示图时可用：

```bash
python scripts/capture_readme_shots.py
```

（需本机已启动 `python run_web.py --port 8765`，并已安装 Playwright + Chromium。）

---

## License

本项目采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)。  
修改、分发或通过网络提供服务时，须按相同协议公开对应源代码。

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=rosemarycox5334-debug/AlphaMaster&type=date&legend=top-left)](https://www.star-history.com/#rosemarycox5334-debug/AlphaMaster&type=date&legend=top-left)
