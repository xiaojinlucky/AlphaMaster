# WO-AM-04：000988 READY 验收与增量备份失败关闭（2026-07-30）

## 结论

`000988` 已完成 9000 步训练，并通过训练产物合同、回测、虚拟信号和队列四层验收，正式进入 `READY`。随后的一次有效增量备份尝试因下一项 `002027` 正在改写 `tail.log`，在远端 staging 闭合校验阶段失败关闭；新 generation 未发布，旧 `CURRENT` 保持不变，临时目录已清空。

因此本轮结论分成两部分：

- `000988` 训练与本机持久产物：**验收通过**；
- `000988` 最新完整产物的异地备份：**尚未完成**，不得写成已备份。

本次未停止或重提作业，未修改 34 个冻结训练文件，未触碰 sealed OOS 或旧数据。

## 训练与身份

- run：`run_20260723T235959Z_26c8877c`
- Slurm：`583711`，cu05，`COMPLETED/0:0`，墙钟 `08:14:45`
- runtime commit：`bb280ca2dada4b087ce1bf5d86f1292b73c576b9`
- 34 个训练源文件聚合 SHA-256：`0e5ca8c3ee924b94757e453e01637d4e1a7f85c7ae9a50ad9baacebdf0564ab0`
- 数据 SHA-256：`9e6a40d22de42d2c0765ac68437ad907f8c832c5277a0dd4ea3df144659202f3`
- run manifest SHA-256：`2f42376f4bfd5f8de4a94544c7c101d7ac0698070a7918e8f4b56c236db8ab11`
- 结果 manifest SHA-256：`790f92284ce6fd4b7f7b43bf7f7548c53312738c5ca665072f077e22137466ec`
- 评分合同：`open_t1_t2_same_index_tail2_v1`
- 固定适配器 SHA-256：`fdddd12bf0ac1c1fa471807790533b386fb3a02216c9ab3b5c233eabc69373ce`

## 训练产物合同

25 项只读合同审计结果为 **25/0**，包括：

- run、symbol、job、runtime commit、训练数据和 34 个源文件身份闭合；
- Slurm 终态为 `COMPLETED/0:0`；
- 以本机固定适配器字节流远端重验原始 452 件产物；
- 原始结果 manifest SHA-256 为 `59ec0a9bbd4024973e2c8acd4d2a0394e7aeccc066e2c2000931e30435d03417`；
- 按 `highest_step_single_run_v1` 省略 449 个旧 checkpoint，只发布 step 9000 checkpoint、策略和训练历史 3 件；
- checkpoint 的模型、优化器、best snapshot、9000 步历史及策略语义均通过。

3 件归一化产物：

| 产物 | 字节 | SHA-256 |
|---|---:|---|
| `ckpt_000988_step_9000.pt` | 6,812,796 | `d574cefccafe38e895a74b6e10afab21c0f2d72baa33452997347c3162ed1093` |
| `strategies/best_000988.json` | 1,012 | `54641831292ad787516fc22aedb8e88d9a30d1bbbeef8dca419d9351b85e5ab1` |
| `training_history_000988.json` | 2,240,817 | `d8db8864abbb34fd4da3835f4a6f18128e6737c691fe3d5a5f961dda5083f075` |

训练 best score 为 `1.5741169452667236`。

## 四层 READY

- 训练层：READY。
- 回测层：READY；报告 SHA-256 `5e80ccb1f160e1da37fad5ac5d9099ebf92a9183e689578c2a585868dbfea257`。
- 虚拟信号层：READY；输出 SHA-256 `a0033113d5a90d75bad8cfdb20f46519ace4d17e095dc57bb6830cb5e7dc8728`。原始方向为 SHORT；普通 A 股账户只做多且当前空仓，因此生命周期动作是 HOLD、目标仓位 0、交付状态 `NOT_REQUIRED`，未发送飞书。
- 队列层：ordinal 4 / `000988` / `583711` / execution run 绑定正确，状态 READY；队列 SQLite `integrity_check=ok`。

回测使用训练数据上的含成本 replay，`total_return=3.197581`、`Sharpe=1.1902`、`Sortino=1.6918`、`profit_loss_ratio=1.4606`、324 笔交易。这些数字不是 sealed OOS 成绩。

## 增量备份失败关闭

本轮一次有效备份尝试的输入规模为 1,308 个文件、3,234.8 MB，其中 SQLite 在线一致性快照 38 个；远端上一完整 generation 有 1,290 个文件。

失败原因：

```text
远端 staging 校验失败：
缺失=[]
多余=[]
大小差异=['local_runs/<active-run>/logs/tail.log']
```

公开证据对仍在运行的 run ID 做了脱敏；原始错误保留在本机日志。
`002027` 已被队列自动领取并持续写入该日志。备份脚本正确拒绝把传输中发生变化的文件发布为完整版本。失败后只读核验：

- `CURRENT` 仍为 `gen-1785374003929226888-1112810-9d40a766076a531c5931cd45`；
- 远端仍保留两个完整 generation；
- `.incoming=[]`、`.building=[]`；
- 当前 generation 仍为 1,290 个普通文件、3,218,579,641 字节；
- 37/37 SQLite 均 `integrity_check=ok`、journal mode 为 DELETE、WAL/SHM 为 0；
- 旧 generation 只含 `000988` 训练中期的 5 个普通文件；与最新本机 17 个普通持久文件相比，缺少 12 个、另有 4 个哈希已变化，因此不能冒充 `000988` 最新完整备份。

没有自动重试，也没有修改备份合同或训练运行态。

## 对抗复核

独立只读对抗复核结论为 **ACCEPT，P0/P1/P2=`0/0/0`**。审查者独立复跑并确认：

- 训练产物合同审计 25/0，34 个冻结源文件、runtime commit、评分合同和 452→3 件产物语义一致；
- 000988 的 state、pipeline、队列、manifest、checkpoint 9000 步及三件产物 SHA 一致；
- 文档没有把备份失败写成完成；
- 远端 `CURRENT` 未变，两个完整 generation 保留，临时目录为空；
- 当前 generation 的 1,290 个文件、3,218,579,641 字节和 37/37 SQLite 健康状态成立；
- 旧 generation 对 000988 确为本机 17 件、远端 5 件：缺 12 件、4 件哈希变化、1 件一致；
- CONTEXT、总工单、正式证据和 PROGRESS 口径一致。

## 自动推进

队列已自动推进到 ordinal 5 / `002027`；活动 run/job 身份只保留在本机巡检证据。故障后只读核验时：

- 队列为 `READY=5 / TRAINING=1 / QUEUED=44`，SQLite `integrity_check=ok`；
- 活动项在 cu05 RUNNING，12 CPU；
- Web 8765 仍在回环监听；
- 冻结源码 SHA、runtime commit 和评分合同未漂移。

后续必须先明确解决“活动普通文件在备份窗口内变化”的备份合同问题，再对 `000988` 执行一次新的增量备份与独立核验；在此之前，`000988` 只能标记为本机四层 READY、异地备份未闭环。
