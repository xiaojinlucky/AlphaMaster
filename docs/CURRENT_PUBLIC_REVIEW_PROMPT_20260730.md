# AlphaMaster 当前公开快照：全量审查与推进方向指令

把下面整段内容交给审查者。审查对象是公开仓库当前 `main`，不是本机运行态，
也不是某个历史专项提交。

```text
你是 AlphaMaster 的独立架构、量化风险、数据完整性和工程验收审查者。
请只读审查公开仓库：

https://github.com/xiaojinlucky/AlphaMaster

你的目标不是重复项目文档，而是：

1. 判断项目真实推进到哪个阶段；
2. 找出距离“可持续训练、可信回放、可审计虚拟信号、受控 sealed OOS”
   完整闭环还差什么；
3. 对当前实现给出 P0/P1/P2 问题；
4. 给出最短关键路径、可并行工作和不超过 6 张的下一阶段工单。

## 一、冻结审查对象

先读取远端 `main` 的实际完整 SHA，并在输出第一行记录：

`REVIEWED_MAIN_SHA=<40位SHA>`

后续所有源码、测试、文档和结论都绑定这个 SHA。不要审查移动中的分支，
不要用历史提交替换当前 `main`，不要创建 Issue、PR、分支或任何写操作。

## 二、必读顺序

1. `README.md`
2. `CONTEXT.md`
3. `docs/WORK_ORDERS_CLAUDE_20260726.md`
4. `docs/CODEX_PROJECT_RULES.md`
5. `lessons.md`
6. 以下主线证据：
   - `docs/evidence/wo_am04_training_manifest_contract_recovery_20260729.md`
   - `docs/evidence/wo_am04_000617_node_fail_checkpoint_recovery_20260729.md`
   - `docs/evidence/wo_am04_000725_ready_backup_20260730.md`
   - `docs/evidence/wo_am04_000792_ready_backup_20260730.md`
   - `docs/evidence/wo_am08_live_sqlite_backup_contract_20260729.md`
   - `docs/evidence/wo_am07a_v4_mainline_accept_20260729.md`
   - `docs/evidence/wo_am07b_replay_eligibility_20260727.md`
   - `docs/evidence/wo_am07f_daily_gap_closeout_20260729.md`
   - `docs/evidence/rnd04a_execution_overlay_adjudication_20260726.md`

`docs/LOCAL_EXECUTION_CONTEXT.md` 与 `docs/VALIDATION_EVIDENCE.md` 只作为
已明确标注日期的历史背景读取，不得用其中的旧状态覆盖 `CONTEXT.md`、
总工单或上述正式证据。

然后按调用链读取对应源码和测试，至少覆盖：

- `model_core/`：目标收益、特征、StackVM、训练/验证语义；
- `data_pipeline/`：数据身份、用途门禁、执行状态 overlay；
- `web/slurm_training_client.py`、`web/slurm_training_manager.py`、
  `web/training_batch.py`、`web/training_queue.py`；
- `scripts/slurm_result_adapter.py`、`scripts/slurm_checkpoint_import.py`、
  `scripts/backup_am_assets.py`；
- `portfolio_manager/`：universe、controller、execution、ledger、replay；
- `evaluation/`：sealed OOS 授权与审计边界；
- 与上述模块对应的 `tests/unit/`、`tests/property/`、`tests/smoke/`。

## 三、事实和证据纪律

每个关键结论标记以下一种：

- `CODE_VERIFIED`：当前 SHA 的源码和测试逻辑可直接证明；
- `PUBLISHED_EVIDENCE`：仓库中的脱敏证据记录了固定哈希、命令或验收结果，
  但你没有接触本机/HPC，因此只能证明公开记录存在；
- `LOCAL_VERIFY_REQUIRED`：需要本机数据库、数据、Web、进程、Slurm、备份端
  或未提交工作区才能证明；
- `USER_DECISION_REQUIRED`：涉及揭示 sealed OOS、真实账户、数据采购、
  删除、迁移或其他会改变业务语义的决定。

不得把训练分数、validation、同训练数据 replay 写成 sealed OOS。
不得把文档中的队列、Slurm、SQLite 或备份状态冒充你亲自实时核验的事实。
仓库不包含凭据、原始行情、数据库、checkpoint、训练历史或原始日志，
不要要求把这些敏感/运行态资产上传 GitHub。
已知历史例外：Git 树仍跟踪旧文件 `training_time_XAUUSD.json`；它不属于
当前审查证据或现役控制面，不得据此推断仓库完全没有历史运行态痕迹。

## 四、当前主线边界

- 主线是普通大 A 只做多研究闭环：
  Slurm 训练 → 含成本 replay → 已收盘 K 线虚拟信号。
- Windows 本机不训练；现役批次不能因审查被停止、重提或换源码。
- 当前不接券商、不发单；没有真实 QMT/PTrade 渠道。
- 34 个训练文件在批次结束前冻结；sealed OOS 未经授权不得揭示。
- WO-AM-07A/B/F 已完成。不要回到 99 点公司公告、StockDB local 包等
  已退出主线的增强支线。
- 审查应进攻式推进：只把会真正阻断结论的缺口列为阻塞，不用泛化风险
  清单、前端美化或重复造轮子占据关键路径。

## 五、必须完成的审查

### 1. 系统闭环

重建并审查以下链路：

数据身份与用途 → PIT universe → 特征/StackVM → factor/仓位 →
训练/validation → 训练产物合同 → 成本 replay → 虚拟信号 →
队列状态 → 增量备份 → sealed OOS 授权门。

检查成功、失败、超时、重试、节点故障恢复、进程重启和并发状态是否保持
同一 run/job/source/scoring 身份，是否存在跨 run 混产物或静默降级。

### 2. 数据和量化语义

重点检查：

- `open[t+1] → open[t+2]` 目标收益及尾部裁剪是否全链一致；
- train / validation / sealed test 是否物理和语义隔离；
- qfq/hfq、成交量单位、停牌/ST/涨跌停、整手和 T+1 是否跨层混用；
- PIT 股票池的 known_at/effective_at 两时钟是否在 replay 中贯彻；
- 同训练数据 replay 的标签是否诚实；
- sealed OOS 是否存在绕过、重复揭示或把结果反馈训练的通路。

### 3. 产物、恢复和备份合同

重点检查：

- 远端原始多 checkpoint 结果是否必须先逐件哈希验证，再由固定适配器
  归一化为 checkpoint、策略、历史 3 件；
- checkpoint 恢复是否保持逻辑 run、物理 execution run、父 job、恢复 step
  和历史前缀身份；
- SQLite 在线快照、WAL/SHM 中和、generation 构建和 `CURRENT` 原子切换
  是否在中途失败、掉电和并发下保持可恢复；
- 备份是否只新增、不删除远端历史版本。

### 4. 测试可信度

不要只数测试文件。检查关键断言是否真的约束生产语义、是否误用 mock 代替
真实边界、是否遗漏 property/smoke、是否有陈旧测试基线掩盖生产失败。
指出必须补的最小测试和真实验收，不要提出泛化“增加覆盖率”。

## 六、输出格式

### A. 审查对象与读取证明

- `REVIEWED_MAIN_SHA`
- 仓库可见性
- 已读文件清单
- 未能读取的文件及原因

### B. 当前阶段裁决

给出：

- 系统阶段：一句话；
- 完成度：0–100%，列出计算口径；
- 已完成；
- 未完成；
- 当前真正阻塞项；
- 结论：`PASS` / `RETURN` / `BLOCKED`。

### C. P0/P1/P2 发现

每项必须包含：

- 级别；
- 文件、类/函数和尽可能精确的行号；
- 可复现触发条件；
- 会造成的真实后果；
- 当前证据等级；
- 最小修复；
- 验收办法。

没有源码位置、触发条件和后果的意见不得列为问题。

### D. 最短推进路线

画出依赖顺序，并分成：

- 训练运营线；
- 数据/replay 研发线；
- sealed OOS 前置线；
- 2–3 条可同时推进的独立工作。

明确哪些现有模块必须复用，哪些旧支线不得重开。

### E. 最终工单

最多 6 张，每张包含：

工单 ID:
目标:
非目标:
依赖:
可并行性:
允许修改范围:
必须复用:
硬验收:
真实证据:
停止条件:
是否需要用户授权:

最后给出：

`ALPHAMASTER_CURRENT_MAIN_REVIEW_COMPLETE`
```
