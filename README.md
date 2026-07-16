# AlphaMaster

基于深度神经网络强化学习的量化因子挖掘中心：从 Parquet / MT5 / OKX K 线自动搜索可解释因子公式，支持 Web 端训练、回测与实时信号分析。

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

![Web 控制台总览](docs/images/00_hero.png)

当前私有工作仓库：[github.com/Jinqingchang/AlphaMaster](https://github.com/Jinqingchang/AlphaMaster)；原作者仓库保留为只读 `upstream`：[rosemarycox5334-debug/AlphaMaster](https://github.com/rosemarycox5334-debug/AlphaMaster)。

当前数据身份、checkpoint、A 股转换和训练包候选实现仍在深度审查，接手前先读 [`CONTEXT.md`](CONTEXT.md) 与 [`docs/GPT56_PRO_EXTENDED_HANDOFF.md`](docs/GPT56_PRO_EXTENDED_HANDOFF.md)，不要把未提交候选当成稳定功能。

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
- **本地后端**：开始训练会从项目检查点续训，重新训练会清除该品种检查点后从头搜索
- **Slurm 第一阶段**：每个 run 独立训练；网络中断会重连同一 run/job，跨 run checkpoint 续训尚未开放；每次 `READY` 会以单一原子指针发布该 run 的 checkpoint、训练历史和策略整套产物，不跨 run 混用
- 展示最优分数、验证分数、训练曲线与最优公式；可选 AI 分析当前训练情况  

### 策略回测

![回测页](docs/images/02_backtest.png)

- 仓位：`position = tanh(factor)`，信号越强仓位越大  
- 成本：手续费 + 滑点（默认约 0.02% / 0.01%）  
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
├── strategy_manager/    # 实盘信号与仓位逻辑（与回测口径一致）
├── execution/           # MT5 下单接口
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
- 可选：MetaTrader 5 终端（实时 MT5 源 / 实盘相关脚本）  
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

# 2. 复制并检查配置；不要把真实凭据提交到 Git
Copy-Item .env.example .env

# 3. 仅在回环地址启动本机 Web
.venv\Scripts\python.exe run_web.py --host 127.0.0.1 --port 8765
```

Slurm 模式固定使用 `cpu` 分区、`normal` QOS 和服务器项目 Python 3.11 环境；失败会原样暴露，不会退回本机训练。上传数据、源码身份和下载产物均通过 manifest 与 SHA-256 校验。详细设计见 [`docs/slurm-deployment-design.md`](docs/slurm-deployment-design.md)。

当前正式任务默认申请 12 CPU；实际训练节点仍由 Slurm 根据集群可用性分配，不固定节点。

`TRAINING_BACKEND` 必须显式配置；缺失或未知值会拒绝启动。只有明确设置 `TRAINING_BACKEND=local` 才允许本机训练。

---

## 常用命令

```bash
# Web 控制台
python run_web.py --port 8765

# CLI 训练（自动续训；加 --from-scratch 则重新训练）
python train_file.py --data-file D:\K线数据\BTCUSDT_H1.parquet
python train_file.py --data-file D:\K线数据\BTCUSDT_H1.parquet --from-scratch
python train_file.py --data-file D:\K线数据\XAUUSD_H1.parquet --from-scratch --train-steps 20
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
