# GPT-5.6-sol 移交说明

## 1. 项目定位

AlphaMaster 是基于原始公开仓库的私有轻量二次开发项目。现有核心路线保持为：

```text
AlphaGPT 生成公式
→ StackVM 执行公式
→ 因子映射为连续仓位
→ 回测和奖励评价
```

本轮不扩张到 Strategy Genome、Alpha Pool 平台、多品种通用 Alpha 或券商订单执行。当前重点是：

1. 保证公式和算子严格因果，历史回测与实时计算语义一致。
2. 保证旧语义策略/checkpoint 失败关闭。
3. 保持 Windows 本机与 Linux Slurm 的可审计、可恢复联通。
4. 识别并关闭会让历史成绩失真或无法复现实盘的其他未来函数、时间错位和回测过拟合。

本移交只覆盖 `Jinqingchang/AlphaMaster`。不得读取、比较、规划、修改或执行任何其他仓库。

## 2. Git 事实源

- 私有仓库：`Jinqingchang/AlphaMaster`
- 分支：`main`
- 本轮交接前基线：`775dad54acd0bd75e5fc75d58c424536af117e38`
- 原作者仓库：`rosemarycox5334-debug/AlphaMaster`，只用于区分原始机制与 fork 新增机制。
- 最终交接提交是包含本文件的 GitHub `main` 最新 SHA；请通过 GitHub MCP 动态读取，不要使用本文件猜测。

## 3. 已确认的因果修复

### StackVM 最终输出

提交 `fd3e1302a77dce0900fb54513d51c3551319ef92` 已修复 `_normalize_output()` 的全样本时间统计前视偏差：

- 固定 200 根滚动窗口。
- 任意时点只使用当前及历史数据。
- 多品种有截面分散时保留同一时点截面标准化。
- 单品种或无截面分散时使用因果时序标准化。
- 旧公式执行合同产物失败关闭。

### JUMP 算子

提交 `775dad54acd0bd75e5fc75d58c424536af117e38` 已修复 JUMP 类算子的全局标准化前视偏差：

- 使用 200 根、包含当前 bar 的稳定滚动样本统计。
- 补齐 Web 旧策略和 Slurm 混合 checkpoint 的合同拒绝。
- 旧策略不能被静默重新标记为当前版本。

两类修复的核心验收都是前缀不变性和未来扰动不变性，而不是简单检查函数名或存在 rolling 代码。

## 4. 当前端到端远程链路

```text
Windows 选择 Parquet
→ 本机校验文件名、K 线合同、来源 sidecar 与 SHA-256
→ 动态选择健康计算入口
→ OpenSSH/SCP 准备并上传独立 run
→ 服务器固定控制器幂等提交 Slurm
→ Web 查询 PENDING/RUNNING/终态并读取日志
→ 支持取消、网络中断恢复和 Web 重启接管同一 run/job
→ Worker 复核输入与源码身份并执行训练
→ 生成结果 manifest
→ 本机下载策略、checkpoint 和训练历史
→ 复核路径、大小、哈希与数据身份
→ 原子发布同一 READY run 的完整产物
```

登录节点只作 SSH 跳板；实际训练节点由 Slurm 调度。远程失败不得静默退回本机训练。

## 5. 当前验证状态

- StackVM normalization：受影响回归 155 通过、0 失败；独立六维 `PASS`。
- JUMP：受影响回归 137 通过、0 失败、1 跳过；独立六维 `PASS`。
- JUMP 修复提交时完整 unit：501 通过、8 个既有失败、1 跳过，不能宣称全仓全绿。
- 大模型供应商候选代码和测试保留在本机未提交工作区，不属于本次 GitHub 交接基线。

详细证据及失败清单见 `docs/VALIDATION_EVIDENCE.md`。

## 6. 已确认和待验证的问题

- `R-01`：reward 是否过度依赖交易结果、掩盖预测质量。当前只登记，不修改；须在因果性、回测/实时一致性和模拟盘证据之后再评估。
- `P0-CANDIDATE-DATA-BFILL`：多品种时间交集不足时执行 `ffill().bfill()`；其中 `bfill()` 已确认会把未来首次报价写入历史时点，尚未修复。
- `P1-CANDIDATE-TURNOVER`：`_turnover_quality()` 对 `tanh` 连续仓位执行 `int(p)`，已确认会把绝大多数仓位误判为 0。
- `P0-CANDIDATE-SELECTION`：walk-forward validation 被每个训练 step 反复用于冠军、精英池和搜索方向，属于 selection 反馈；尚未证明存在策略冻结后才访问的密封最终测试。
- `P1-CANDIDATE-HORIZON`：PnL 使用 `position[t] * target_ret[t]`，IC 使用 `factor[t]` 对齐 `target_ret[t+1]`；必须用合成序列证明唯一 horizon 后再决定是否修改。
- 当前 `9000` 步与 `00:30:00` Slurm 时限明显不相容；正式训练预算未冻结。
- 12 CPU 没有与 8 CPU 同口径的稳定性能结论。
- 旧 MT5 668 文件批量计划尚未 apply。
- Web/API 训练包导入继续固定关闭。
- TradingView 继续暂缓。
- 8 个完整 unit 既有失败尚未单独治理。
- 工程链路通过不代表策略盈利；仍需严格样本外、成本、稳健性、模拟盘和受控小资金验证。

## 7. 外部模型必须遵守的边界

- 从第一性原理和 GitHub 当前代码证据出发，不照搬旧对话结论。
- 开发建议遵守剃刀定律，不扩张系统边界，不主动重写已满足要求的部分。
- 验收遵守墨菲定律，覆盖未来扰动、前缀一致、旧产物、损坏状态、权限、超时、并发和重启。
- 问题分“阻塞落地的问题”和“可选优化”，只要求修复前者。
- 每个阻塞建议必须写清违反的需求、实际风险和验证修复的方法；无明确依据时保留现状。
- 网页版 GPT 看不到本机 skills、memory、进程、凭据、数据、SSH/Slurm 状态和未提交工作区，所有工单必须由本地 Codex 校正。
- 本轮只读规划，不连接服务器、不启动训练、不修改仓库、不创建 PR。
- 大模型供应商能力正在独立项目中先行修复；本轮不得为 AlphaMaster 重复规划或提交供应商代码。

## 8. 网页版 GPT 的任务

1. 使用 GitHub MCP 完整读取 `docs/GPT_WEB_PRO_EXTENDED_TASK.md` 规定的文件。
2. 先核对 StackVM/JUMP 因果修复和旧产物失败关闭是否仍有可复现缺口。
3. 系统审计数据对齐、特征/算子、信号、目标收益、PnL、walk-forward 和候选选择是否存在其他未来函数或过拟合。
4. 对已确认问题逐项核实代码证据，并按影响历史成绩真实性和实盘一致性的优先级排序。
5. 只输出一张最小工单、禁止项和二元硬验收，不直接编码或运行。
6. 首轮只问一个真正改变实现的业务问题；已由代码和文档确认的事项不要重复询问。

启动提示词见 `docs/GPT_WEB_PRO_EXTENDED_TASK.md`。
