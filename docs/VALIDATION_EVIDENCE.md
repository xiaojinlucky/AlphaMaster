# AlphaMaster — 当前验证证据

本文件只记录已经得到最终汇总的验证。没有最终汇总、仅有 Mock、仅到达外部端点或仍待提交的事项都单独标明。

## 1. Git 基线与范围

- 私有仓库：`Jinqingchang/AlphaMaster`
- 分支：`main`
- 本轮交接前基线：`775dad54acd0bd75e5fc75d58c424536af117e38`
- 该基线已与本地 `HEAD`、`origin/main` 一致。
- 本轮候选只包括 canonical 上下文、项目规则、问题记录、既有因果修复证据和网页版 GPT 总控指令；大模型供应商候选代码与测试不进入提交。
- 最终提交 SHA 必须从 GitHub `main` 动态读取；本文件不在同一提交中自我硬编码。

## 2. StackVM normalization 因果修复

- 修复提交：`fd3e1302a77dce0900fb54513d51c3551319ef92`
- `StackVM._normalize_output()` 已从全样本时间统计改为固定 200 根、只使用当前及历史数据的滚动标准化。
- 多品种同一时点有截面分散时保留截面标准化；单品种或无截面分散时使用因果时序标准化。
- 公式执行合同已进入 `VOCAB_VERSION`；旧 normalization 语义的策略、checkpoint、训练包、回测和实时监控均失败关闭。
- 受影响回归：155 通过、0 失败。
- 当时完整 unit：459 通过、7 个既有失败。
- 独立审查经两轮阻塞修复后六维 `PASS`。

核心验收不是“使用 rolling z-score”这一实现名字，而是：

```text
任意前缀长度 L，1 <= L <= T：
full_series_output[..., :L]
=
execute(input[..., :L])[..., :L]
```

同时必须覆盖未来追加、未来扰动、不同 batch 组成、常数序列、短序列、NaN/Inf 和旧产物拒绝；并用 `output[i] = input[i+1]` 这类故意前视的反例证明前缀最后一个索引不会被漏检。

## 3. JUMP 算子因果修复

- 修复提交：`775dad54acd0bd75e5fc75d58c424536af117e38`
- JUMP 类算子已改为 200 根、包含当前 bar 的稳定滚动样本标准化，不再使用整段序列的全局均值和标准差。
- Web 旧策略与 Slurm 混合 checkpoint 已补齐公式执行合同校验；旧语义产物不能续训、回测或实时使用。
- 受影响回归：137 通过、0 失败、1 项因本机无 CUDA 跳过。
- 当时完整 unit：501 通过、8 个既有失败、1 项跳过。
- 独立审查经两轮复验后六维 `PASS`。

## 4. 已确认但尚未修复的问题

### 4.1 多品种对齐的 `bfill()` 前视偏差

`data_pipeline/data_manager.py` 在严格时间交集少于 `MIN_BARS` 时，会降级为并集并执行：

```python
reindexed = reindexed.ffill().bfill()
```

`bfill()` 会把某品种未来出现的第一条报价填入更早的时间点。该问题已由当前代码直接确认，尚无修复提交和自动化验收，因此受该路径影响的历史成绩不能作为可信证据。

本机只读最小复现使用两个完全错开的两行品种，交集为 0；品种 B 在时间 1 的填充值等于它到时间 3 才出现的首个真实 open，证明未来报价被写入历史时点。

### 4.2 `turnover_quality` 连续仓位统计失真

`model_core/backtest.py::_turnover_quality()` 对 `tanh` 产生的 `(-1, 1)` 连续仓位执行 `int(p)`。例如 `int(0.8) == 0`、`int(-0.8) == 0`，会把绝大多数真实暴露误判为空仓。该组件进入 reward，可能改变公式搜索方向。

本机只读最小复现输入 `[0.8, 0.8, -0.8, -0.8]`：真实仓位变化计数为 1，但 `_turnover_quality()` 返回无交易路径的 `-2.0`。

### 4.3 validation 被反复用于选择

`model_core/engine.py` 在每个训练 step 读取 walk-forward validation 分数，并用它更新：

- `best_score` / `best_formula`
- `elite_pool`
- `factor_pool`
- 后续搜索与重启状态

因此这些 validation 折属于训练选择反馈，不是策略冻结后才访问的密封最终测试。当前代码有时间切分和 gap，但这两者不能自动消除“反复试验后对 validation 过拟合”的风险。

### 4.4 IC 与 PnL horizon 待合成序列确认

- `target_ret[t] = log(open[t+2] / open[t+1])`
- PnL 使用 `position[t] * target_ret[t]`
- IC 使用 `factor[t]` 与 `target_ret[t+1]`

二者看起来相差一期。当前只登记为待验证问题；必须用合成价格、固定 oracle 和唯一执行时点证明后，才能决定是否修改。

## 5. 当前测试基线

提交 `775dad54acd0bd75e5fc75d58c424536af117e38` 的最终记录：

- 受 JUMP 影响的回归：137 通过、0 失败、1 项因本机无 CUDA 跳过。
- 完整 unit：501 通过、8 个既有失败、1 项跳过。
- 退出码：1。

现有失败未被静默忽略，因此当前仓库不能描述为“全仓全绿”。本轮只提交文档交接包，不把本机未提交的大模型供应商候选代码及其测试数字混入 GitHub 基线。

## 6. 数据、远程与训练边界

- 新 MT5 `NVDA_M5`：50,000 根已收盘 K 线，sidecar 与 Parquet 哈希合同通过；未启动训练。
- 旧 OKX `BTCUSDT_H1`：身份为 `okx_legacy_attested`。
- 旧 MT5 批量计划：668 个文件；537 可登记、125 不足 bars、6 数据合同失败；尚未 apply。
- 已有 Slurm 冒烟曾证明上传、提交、训练、下载和产物校验链路可完成。
- 本轮没有连接服务器、提交/取消 Slurm 任务、启动正式训练、配置 TradingView 或接入订单执行。

## 7. 安全与 Git 最终门禁

- `.env`、真实 `web_settings.json`、token、密码、私钥、账户信息、原始行情、数据库、checkpoint、训练历史、日志和运行态不得进入提交。
- 历史 Git 对象曾含旧 Tushare token；当前代码已改为环境变量读取，但旧值仍应视为已暴露并轮换。
- 当前 Git 树仍跟踪历史运行态文件 `training_time_XAUUSD.json`；本轮不得把它作为新改动重新暂存。
- 本轮只能暂存 canonical 文档和项目规则；大模型供应商生产代码、测试、前端改动和本机配置必须保持未提交。
- 最终提交前必须对精确 staged snapshot 复核文件列表、文件类型/大小并运行 gitleaks；最终结果由本地提交线程报告。

## 8. 独立审查

- StackVM normalization 和 JUMP 因果修复已有各自独立六维 `PASS`。
- 本轮 8 份文档交接包已经由独立严格审查线程复核事实准确性、范围隔离、问题分类、硬验收和网页版 GPT 指令。
- 首轮审查发现 1 个阻塞项：因果验收用 `:t` 表示“截止到索引 t”，实际会漏掉当前索引。总控指令和本文件随后统一改用前缀长度 `L` 与切片 `:L`，并要求用 `output[i] = input[i+1]` 的故意前视反例验证最后一个前缀索引不会漏检。
- 同一独立审查者复验六个维度均为 `PASS`，阻塞落地问题为 0；没有要求修复可选优化，也没有修改或审查大模型供应商候选代码。
