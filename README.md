# AlphaMaster

基于深度神经网络强化学习的量化因子挖掘中心：从 A 股 / MT5 / OKX Parquet 自动搜索可解释因子公式，模型训练固定走服务器 Slurm，并在 Web 端贯通回测与实时信号分析。

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

![Web 控制台总览](docs/images/00_hero.png)

目标发布仓库：[github.com/xiaojinlucky/AlphaMaster](https://github.com/xiaojinlucky/AlphaMaster)；原作者仓库保留为只读 `upstream`：[rosemarycox5334-debug/AlphaMaster](https://github.com/rosemarycox5334-debug/AlphaMaster)。用户已明确要求目标仓库始终保持公开，以便网页版 GPT 直接读取已提交源码和证据；不得再把改为私有作为发布前提。公开不放宽安全边界：`.env`、真实配置、密钥、原始数据、数据库、模型、checkpoint、训练历史、原始日志和机器可消费的运行态文件始终不上传；只允许提交脱敏、带时间戳且不能作为控制输入的摘要证据。当前 goal 内的提交与推送已有持续授权，但每次仍须精确暂存并通过 staged 安全检查。

现役数据身份、Slurm 批次、checkpoint、回测、虚拟信号和封存评估边界以 [`CONTEXT.md`](CONTEXT.md) 与 [`docs/VALIDATION_EVIDENCE.md`](docs/VALIDATION_EVIDENCE.md) 为准；历史提交不能替代当前运行证据。外部模型还必须读取 [`docs/LOCAL_EXECUTION_CONTEXT.md`](docs/LOCAL_EXECUTION_CONTEXT.md)：它只能提供方向和反方意见，本机 Codex 才能按真实 Skills、数据、代码和运行态重写并执行工单。

当前唯一外部审核入口是 [`docs/WEB_GPT_CONTROLLER_PROMPT.md`](docs/WEB_GPT_CONTROLLER_PROMPT.md)：网页版 GPT 审核公开提交并给出方向和工单，本机 Codex 再按真实 Skills、memory、代码、数据和运行态本地化。BioMNI 不再参与当前 goal；其旧提示词只保留为历史证据。当前主线是普通大 A 账户的只做多信号：Windows 本机负责数据、控制、回测和信号展示；所有模型训练固定提交到 Linux Slurm 服务器。

当前 50 只中证 A50 仅作为龙头基线池，用于先跑通批量训练、回测和虚拟信号；它不等于“近一年轮动表现最强 50 只”。官方依据、成熟轮动框架候选、夏普门槛和未完成调研见 [`docs/A50_ROTATION_RESEARCH_EVIDENCE.md`](docs/A50_ROTATION_RESEARCH_EVIDENCE.md)。

---

## 它做什么

AlphaMaster 把「挖因子」做成一条可操作的流水线：

1. **服务器训练**：本机上传经过身份校验的 Parquet，Slurm 单节点用强化学习搜索公式，本机不训练
2. **成本回测**：训练产物回传后，用 `tanh(因子)` 连续仓位和手续费 / 滑点完成工程重放或独立样本外评分
3. **虚拟与实时信号**：读取通达信已收盘 K 线，以普通 A 股账户只做多语义生成买入、加仓、减仓、离场、止盈和止损事件，并可推送飞书

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
| **01 模型训练** | 选 Parquet、提交服务器 Slurm、看资源与曲线，并自动继续回测和虚拟信号 |
| **02 策略回测** | 选策略 JSON，设手续费 / 滑点，看绩效与资金曲线 |
| **03 实时分析** | 多数据源监控，收盘后更新只做多交易动作；可选飞书信号推送 |

### 模型训练

![训练页](docs/images/01_train.png)

- Parquet 命名：`{品种}_{周期}.parquet`，例如 `BTCUSDT_H1.parquet`、`XAUUSD_H1.parquet`  
- **manifest 边界**：AlphaMaster 核心读取本地 Parquet 不依赖 manifest；Slurm 远程训练必须用 sidecar 绑定文件哈希、来源和数据范围。旧 MT5 文件可在页面一次确认后自动注册，新版 MT5 / OKX 导出器会自动生成
- **Slurm 第一阶段**：每个 run 独立训练；网络中断会重连同一 run/job，跨 run checkpoint 续训尚未开放；每次 `READY` 会以单一原子指针发布该 run 的 checkpoint、训练历史和策略整套产物，不跨 run 混用
- **大 A 自动后处理**：大 A run 达到 `READY` 后，以同一 `run_id` 自动执行含手续费 / 滑点的 replay 回测，再用通达信最新已收盘 K 线生成虚拟动作；页面明确提示 replay 不是样本外收益
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

- 数据源：普通 A 股优先使用通达信；其他来源以界面可用源为准
- **只在当前周期 K 线收盘后**重新判断；未收盘 bar 不参与信号  
- 卡片展示方向（看涨 / 看跌 / 不确定）与把握程度  
- 普通 A 股账户只做多：负因子不会生成做空建议；空仓时观望，持仓时减仓或离场
- SQLite 虚拟仓位与事件账本：同一策略、品种、周期、K 线只生成一个决策，重启后不重复
- 动作包括买入、加仓、减仓、离场、止盈和止损；系统只发信号，不连接券商、不自动下单
- 可选飞书 Webhook：只推送新的可执行交易动作，继续持有和观望不打扰
- 飞书网络失败、限流或服务端错误自动重试，最终投递状态保留在信号历史中
- 已有确定性虚拟执行内核处理显式停牌/涨跌停状态、T+1、买入整手、合法零股卖出和费用；它尚未接生产执行账本或券商，不能解释为真实订单校验

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
├── experiments/         # 隔离差分实验；不进入生产运行时
├── strategies/          # best_{symbol}.json 策略文件
├── checkpoints/         # 训练检查点
├── run_web.py           # 启动 Web 控制台
├── train_file.py        # CLI：从单个 Parquet 训练
└── requirements.txt
```

### 隔离研究实验

大 A 候选框架的固定样本差分位于
[`experiments/a_share_execution_diff`](experiments/a_share_execution_diff/README.md)
和
[`experiments/a_share_research_layer_diff`](experiments/a_share_research_layer_diff/README.md)。
统一的全新目录运行入口是
[`experiments/run_a_share_diff_suite.py`](experiments/run_a_share_diff_suite.py)。
这些实验只验证明确合同，不替换生产回测；结果、边界和已发现的评分时钟/
尾部裁剪 P0 见
[`docs/A_SHARE_QUANT_EXPERIMENTS_20260724.md`](docs/A_SHARE_QUANT_EXPERIMENTS_20260724.md)。

---

## 环境要求

- Python **3.10+**（建议 3.11）  
- PyTorch、pandas、FastAPI、uvicorn 等（见 `requirements.txt`）  
- Windows 使用已验证的 PyTorch 2.8 CPU；不要自动升级到本机已复现 `c10.dll` 启动失败的 2.12/2.13
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

## Windows Parquet（A 股 / MT5 / OKX）→ Slurm CPU 训练

支持严格转换后的本机 A 股 Parquet、MT5 已收盘 K 线或经过来源清单校验的 OKX Parquet，通过本机 Web 控制台提交到 Slurm。free-stockdb 前复权日线使用独立来源 `ashare_free_stockdb_qfq`，当前只接受 D1，并且必须携带绑定文件哈希、源快照、抽取脚本、复权方式、15:00 收盘语义和成交量单位的 sidecar；成交量按原始股数保存，不随价格前复权调整。该来源不能冒充旧 `ashare_local` 或 AKShare 数据。`login-node` 仅作为 SSH 配置中的跳板，不运行命令；每次远端操作先调用节点选择器确定健康的交互入口，真正的训练节点始终由 Slurm 调度。

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

`TRAINING_BACKEND` 必须为 `slurm`；缺失、`local` 或其他值都会拒绝启动，远端失败也绝不退回 Windows 本机训练。

训练门同时位于模型引擎和单文件训练入口：只有 Linux 计算节点上的受控 Slurm Worker 能真正进入训练循环。Web 后台会持续观察当前 job；浏览器关闭不影响训练完成后的回测与虚拟信号。后处理只接受哈希验证后的 Slurm 发布包，临时行情/回测超时最多自动重试 3 次。

### 2026-07-23 大 A 真实闭环证据

- 输入：`600519_D1.parquet`，5,955 根日线，数据 SHA-256 为 `4cb8912466781de76735f3bcd7873dac204098bc5a5e8647e3b8a009fd4627b8`
- Slurm：run `run_20260723T151419Z_bdc5e5a0`、job `568306`，单节点 `cu16`、12 CPU、32 GB 申请；20 步完成，墙钟 `00:01:25`、累计 CPU `10:09.454`、峰值内存 `1254524K`
- 最优公式分数：`0.8649603724`；策略、checkpoint、训练历史三类产物均完成哈希校验并回传
- replay 回测：累计对数收益 `2.912246`、夏普 `0.5306`、索提诺 `0.6649`、盈亏比 `2.3136`、588 笔；这是同训练数据的工程闭环验证，不是独立样本外成绩
- 通达信最新 500 根已收盘日线生成 `BUY`，目标虚拟仓位 `64.63%`，参考价 `1292.01`，止损 `1266.1698`，止盈 `1343.6904`；本次流水线验证未发送飞书

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
# 本机只启动控制台；训练按钮会创建并提交服务器 Slurm run
.venv\Scripts\python.exe run_web.py --host 127.0.0.1 --port 8765

# 回测可显式选择独立测试数据；auto 会自动判断重放或样本外
.venv\Scripts\python.exe run_backtest.py --single `
  --strategy-file strategies\best_600519.json `
  --data-file D:\大A数据\600519_D1_future.parquet `
  --evaluation-mode auto
```

`train_file.py` 是 Slurm Worker 内部入口，不是 Windows 本机训练命令。不要在本机直接运行它；正式模型训练统一从 Web 提交服务器 Slurm。

策略输出默认在 `strategies/best_{symbol}.json`。

---

## 信号口径（训练 / 回测 / 实时一致）

- 因子经 StackVM 算出标量序列  
- 目标收益为 `log(open[t+2] / open[t+1])`；`factor[t]` 与
  `target_ret[t]` 同索引配对，最后两根没有未来收益，不进入评分、成本或
  回测曲线
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
