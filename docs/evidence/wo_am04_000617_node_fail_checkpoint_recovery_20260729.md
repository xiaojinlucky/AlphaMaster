# WO-AM-04：000617 节点故障与 checkpoint 续训门

时间：2026-07-29 12:50–18:27（Asia/Shanghai）

## 结论

- 000617 / Slurm 581389 于 12:50:09 进入 `NODE_FAIL`。主作业运行
  01:57:19，`ExitCode=0:0`；batch/extern 因节点消失被取消。
- cu19 随后完成重启。12:59 直接查询 cu19 时 uptime 约 14 分钟；
  Slurm 仍把节点标记为 `DOWN* / Not responding`。
- 这是计算节点失联，不是训练代码、数据、评分合同或模型报错。
- 队列失败关闭正确：`READY=1 / NEEDS_ATTENTION=1 / QUEUED=48`，
  批次状态 `NEEDS_ATTENTION`；没有自动派发下一件。
- Web 监护父子进程仍存活，PID 68008 在 `127.0.0.1:8765` 正常监听。
- 旧 run、旧 job、原始 checkpoint 和训练历史均保持不变；未重提、
  未停止、未写队列、未触碰 sealed OOS 或旧数据。

## 可恢复资产

- 失败运行：`run_20260723T235959Z_867bfc69`
- 数据：000617 / D1 /
  `66a9bb626ed65fc1b0e5f0eaf168f69344e2bd3789e848dd47c2ac5e38e802af`
- runtime commit：
  `bb280ca2dada4b087ce1bf5d86f1292b73c576b9`
- 冻结训练源码集合：
  `0e5ca8c3ee924b94757e453e01637d4e1a7f85c7ae9a50ad9baacebdf0564ab0`
- 评分合同：`open_t1_t2_same_index_tail2_v1`
- 已生成 138 个完整 checkpoint；最新完整 checkpoint 为 step 2760：
  `checkpoints/D1/66a9bb626ed65fc1b0e5f0eaf168f69344e2bd3789e848dd47c2ac5e38e802af/run_01785293593994423452/ckpt_000617_step_2760.pt`
- checkpoint 大小 5,965,894 字节，SHA-256：
  `5da8a7baf29010329a34ede6af70e5547c064701bfd4a8ad0e257edf661259ff`
- 训练历史大小 723,267 字节，SHA-256：
  `b369c42a63c5d9337defb8e4e043d41021448e976d484f85803b5e7a6e378f9c`
- checkpoint 已用 runtime `.venv` 的 `torch.load(weights_only=True)` 独立读取；
  step、数据身份、源码来源、周期参数、评分合同、词表版本、模型、优化器、
  最优快照、重启计数和训练历史均完整。目标总步数仍为 9000，因此还需执行
  2760 至 8999，共 6240 步。

## 现有控制面为何不能直接恢复

- 批处理固定以 `from_scratch=True` 发射。
- 同一 planned run 已绑定 581389；提交器会把再次提交视为同一旧作业，
  不会生成新 job。
- 队列禁止改写已经冻结的 job ID；`recover_submitted_item` 只允许恢复
  “同一仍存在的作业”，不允许把终态 `NODE_FAIL` 换成新作业。
- 新 run 虽可使用 `from_scratch=False`，但每个 run 的 checkpoint 根目录
  独立；现有准备/上传/提交链中没有“提交前导入已验证父 checkpoint”的步骤。

因此，直接重试会失败；绕过队列改写 job ID、删除旧服务器 receipt 或直接
从头训练都会破坏现有身份合同。

## 一次性恢复合同

恢复必须同时满足：

1. 旧 run 及 581389 的终态、receipt、日志和 checkpoint 全部只读保留。
2. 为恢复尝试创建新的物理 run ID 和新的 Slurm job ID；原 planned run ID
   继续作为队列逻辑项目身份，显式记录父 run、父 job、父 checkpoint 路径、
   step、大小和 SHA-256。
3. 新 run 沿用同一训练 Parquet、34 个冻结源码字节、runtime commit、
   评分合同和 9000 总步数；唯一训练参数变化是
   `from_scratch: true -> false`。
4. checkpoint 只能在新 run 已完成输入冻结、但尚未 `sbatch` 之前导入；
   导入前后逐字节 SHA-256 必须等于上述固定值，目标目录必须是新 run 内
   与 D1、data SHA 和 symbol 一致的身份路径。
5. Slurm Worker 仍由原冻结脚本启动；`train_file.py` 必须真实打印从
   step 2760 恢复，首个新 checkpoint 必须大于 2760。
6. 最终结果仍走已经验收的 452→3 清单适配器、回测、虚拟信号、队列 READY
   和增量备份全链验收。

该合同需要新增“逻辑队列项目—物理恢复尝试”的显式映射和提交前 checkpoint
导入动作。它属于新的恢复授权，不属于已完成的训练产物清单合同修复授权。

## 控制面实现与验收（2026-07-29 13:27）

- 实现只修改冻结训练集合之外的
  `web/training_queue.py`、`web/training_batch.py`、
  `web/slurm_training_manager.py`、`web/slurm_training_client.py`，并新增
  `scripts/slurm_checkpoint_import.py`；34 个冻结训练文件零改动，集合
  SHA-256 仍为
  `0e5ca8c3ee924b94757e453e01637d4e1a7f85c7ae9a50ad9baacebdf0564ab0`。
- 队列继续以原 `planned_run_id` 表示逻辑项目，只为一次 NODE_FAIL 恢复
  写入新的 `execution_run_id` 和 attempt=1，并保留父 run、父 job、
  checkpoint 路径、step、大小与 SHA-256。旧 run/job/checkpoint 不改写。
- 恢复门同时由队列锚定错误合同和远端 Slurm 真实状态复核；仅
  `Slurm NODE_FAIL[:...]` 且父作业真实 `state=NODE_FAIL` 可以进入。
- 新 run 完成数据与 manifest 上传/冻结后，每次进入 `SUBMITTING` 都先执行
  固定字节导入器，再调用 `sbatch`。提交响应中断时会重新做幂等导入核验，
  不因本机旧回执跳过。
- 导入器以 `openat + O_NOFOLLOW` 固定祖先目录和叶文件；复制、大小和
  SHA-256 校验绑定同一源文件句柄；语义验收在同一目标句柄上再次完成
  大小、SHA-256、`torch.load(weights_only=True)` 与前后文件身份核对；
  回执落盘后再做最终路径字节核对。
- 导入器实际 SHA-256 与客户端固定值一致：
  `de851b775be22b0f8da648bb496bb62ad0fc3d485770db9fd1b1b4124fd90e06`。
- 固定测试 152 通过、0 失败；静态编译通过。最终独立对抗审查结论
  `ACCEPT`，P0/P1/P2=`0/0/0`。
- 同一 Unix 账户主动在导入返回后替换目标 checkpoint 仍属于现有单用户可信
  HPC 威胁模型之外的权限边界；不修改冻结 worker/control 或目录所有权模型
  无法消除该越权能力。正常恢复链没有该写入者。

控制面代码已经就绪，但生产实施仍保持冻结：生产队列尚未迁移，Web 未重启，
checkpoint 未导入，新 Slurm 作业未提交。等待用户明确授权后再执行。

## 生产同构副本演练（2026-07-29 13:32）

- 使用 SQLite 在线备份把运行中的生产队列复制到
  `scratch/goal_am_custody/node_fail_recovery_rehearsal_20260729T0530Z/`；
  全部演练只作用于该副本。
- 迁移前副本与演练后反查的生产库均为 50 行，生产完整行摘要为
  `c4f63cab622be5961e55ec8b890af4c55123636e2e1616a8b65eefac31e92cd8`。
- 副本 schema 迁移后，旧 50 行的全部旧列逐列不变，batch 行逐列不变；
  `execution_run_id=planned_run_id` 为 50/50，`attempt_number=0` 为 50/50，
  `idx_batch_items_execution_run_id` 唯一索引存在，`integrity_check=ok`。
- 在副本上以真实 000617 身份执行一次性恢复事务：
  `planned_run_id` 仍为 `run_20260723T235959Z_867bfc69`，演练物理 run 为
  `run_20260729T053000Z_c0ffee07`；parent job=581389，checkpoint 路径、
  SHA-256、大小 5,965,894 和 step 2760 均与正式合同一致。
- 事务后冻结业务字段摘要保持
  `4bc08daf0bcc0d01f2440d06bb03bc7c477fec9cec0d72191024e0ecc6b62735`；
  其余 49 项全部旧列精确不变。副本状态为
  `READY=1/DISPATCHING=1/QUEUED=48`、batch=`ACTIVE`，
  `integrity_check=ok`。
- 演练后以 SQLite `mode=ro` 反查生产库：仍为旧 16 列 schema，
  `READY=1/NEEDS_ATTENTION=1/QUEUED=48`；000617 仍绑定旧 job 581389
  和 `Slurm NODE_FAIL: NODE_FAIL`。生产本机不存在演练 run，Web 仍由
  PID 68008 监听 127.0.0.1:8765。

结论：生产数据库迁移和一次性恢复事务已在生产同构副本上验证；生产状态
没有被演练改变。

## 终态 Slurm 状态门实探与修复（2026-07-29 13:42）

- 使用最终客户端从 `compute-node-12` 调用冻结 `status_run` 时，活动队列
  `squeue -j 581389` 因终态作业已退出而返回
  `slurm_load_jobs error: Invalid job id specified`；冻结控制器因此尚未进入
  后续 `sacct` 分支。这会真实阻断 checkpoint 导入。
- 分别在 `compute-node-11`、`compute-node-12`、`compute-node-13` 只读
  执行固定 `sacct` 查询，三节点均唯一返回：
  581389、用户 jinqc、job name
  `alphamaster_run_20260723T235959Z_867bfc69`、state=`NODE_FAIL`、
  ExitCode=`0:0`、10:52:50–12:50:09、elapsed=`01:57:19`、cu19。
- 导入器不再调用会先经过 `squeue` 的 `status_run`；它通过冻结控制器的
  `_run_slurm` 以固定参数列表执行 `sacct -n -P -X -j <纯数字 job>`，
  只接受一条、恰好 11 字段且 JobIDRaw 精确等于父 job 的主作业行，再核对
  当前 Unix 用户、固定 job name 和规范化后的 `NODE_FAIL`。`.batch`、
  `.extern`、缺失、重复、错误 owner/name/state 和错误字段数全部拒绝。
- 新增正例包含 `.batch` 噪声行并精确断言完整 argv；参数化反例覆盖重复主行、
  仅 step 行、wrong owner、wrong job name 和 12 字段。最终指定测试
  152/0；独立增量复审 `ACCEPT`，P0/P1/P2=`0/0/0`。
- 最终 importer actual/pin：
  `de851b775be22b0f8da648bb496bb62ad0fc3d485770db9fd1b1b4124fd90e06`；
  34 个冻结训练源码 SHA-256 仍为
  `0e5ca8c3ee924b94757e453e01637d4e1a7f85c7ae9a50ad9baacebdf0564ab0`。

该实探和修复均未创建远端 run、未导入 checkpoint、未提交或停止作业。

## 生产恢复实施与续训验收（2026-07-29 13:45–13:57）

- 用户明确授权实施 000617 一次性 checkpoint 恢复后，先以 SQLite 在线
  一致性备份冻结生产队列：
  `scratch/goal_am_custody/000617_production_recovery_20260729T054526Z/training_queue_v2.before.sqlite3`，
  SHA-256 为
  `4ad78a6bdf373aaa81f28109e48e56120da63f0c142b14affdbc20d7abb2bc5b`，
  `integrity_check=ok`。迁移后、绑定恢复事务前的第二份快照 SHA-256 为
  `7fbcf905a35f24d1f48b9d09c9127b5a3466250da57896be43d2c24fd2197e6a`，
  `integrity_check=ok`。
- Web 只使用显式生产队列路径
  `local_runs/training_queue_v2.sqlite3` 和远端根
  `/hwdata/home/jinqc/Quant/AlphaMaster-runtime-v2` 重启；未停止或修改
  任何 Slurm 作业。schema 迁移后旧 50 项和 batch 原字段保持不变，
  execution identity 回填完成，唯一索引存在。
- 一次性恢复事务保持逻辑 planned run
  `run_20260723T235959Z_867bfc69` 不变，新建物理 execution run
  `run_20260729T054526Z_1141da4d`，attempt=1；父 run、父 job 581389、
  checkpoint 路径、step 2760、大小和 SHA-256 全部按恢复合同写入。
- 首次提交前导入正确失败关闭，错误为“导入 checkpoint 无法安全读取”；
  失败发生在 `sbatch` 之前，因此没有产生 Slurm job，也没有派发下一项。
  根因是导入器误用不含 torch 的通用 Python
  `/hwdata/home/jinqc/.local/bin/python3.11`。客户端只把
  `checkpoint-import` 改为使用目标 runtime 自带
  `.venv/bin/python`；固定测试 152 通过、0 失败，importer actual/pin
  仍为 `de851b77…0e06`，34 个冻结训练源码集合仍为
  `0e5ca8c3…64ab0`。
- 对同一 execution run 执行一次提交前重试；checkpoint 导入回执通过，
  target manifest SHA-256 为
  `5eb41169a77dd8577b5584731897051ab67b9159e1658897dac4768ef5655ea0`，
  parent manifest SHA-256 为
  `aff4bc2e667a7ee9b93dfaade531fde5479374817128d0b490d2a97446910bf7`，
  checkpoint 身份与原 step 2760 完全一致。随后唯一新作业 **581519**
  于 13:52:56 在 cu01 开始运行。
- 13:57 生产队列为
  `READY=1 / TRAINING=1 / QUEUED=48`，000617 行保持原 planned run，
  execution run 为新 run，job=581519，parent job=581389，
  `integrity_check=ok`。
- 实际续训已通过 checkpoint 字节和语义验收。首个新 checkpoint 为
  `ckpt_000617_step_2780.pt`，大小 5,968,646 字节，SHA-256
  `a99fb51fda17fe1cbe1635e86d178f3befe82bf27ee6848c04f00fd74ebb451a`；
  它的 symbol、timeframe、data SHA、dataset ID、数据源、年化周期、
  最小样本、评分合同和词表身份与 step 2760 全部相同，18 个训练历史字段
  的既有前缀逐项相等，模型、优化器、最优快照和训练历史均可加载。
  随后训练历史和新 checkpoint 已到 step 2860，证明没有从 step 0 重跑。

结论：000617 一次性 checkpoint 恢复已正式实施并通过“新 job 已运行 +
首个新 checkpoint 大于 2760 + 历史前缀不变”的生产验收。581519 继续运行，
尚未到 9000 步，不能提前记为 READY；后续继续按原训练验收、结果适配和
增量备份链监护。

## 9000 步终态、四层 READY 与备份验收（2026-07-29 18:15–18:27）

- Slurm 581519 于 18:15:12 在 cu01 正常完成：
  `COMPLETED / ExitCode=0:0 / elapsed=04:22:16 / 12 CPU /
  MaxRSS=1,499,968K`。训练历史完整覆盖 step 0–8999，共 9000 条；
  最终 best score 为 1.5200775862。
- 远端原始结果清单 SHA-256 为
  `edf8fa1ab71c65a3407471d3f9382c0435f3e6ec8bbc256fa74e1bea37e6b2f4`，
  完整列出 315 件、2,001,758,285 字节：313 个 checkpoint、1 个策略、
  1 个训练历史。run/job、runtime commit
  `bb280ca2dada4b087ce1bf5d86f1292b73c576b9`、34 个 source files、
  数据 SHA 和评分合同
  `open_t1_t2_same_index_tail2_v1` 全部与冻结输入一致。
- 固定适配器 actual/pin 均为
  `fdddd12bf0ac1c1fa471807790533b386fb3a02216c9ab3b5c233eabc69373ce`，
  在完整核对原 315 件后发布 3 件。归一化结果清单 SHA-256 为
  `c28bbbe25d187540a75ac5bc8d621cefaf7e5c05a0efd901cf0aeab2a54c72d7`：
  - step 9000 checkpoint：6,785,344 字节，SHA-256
    `8dd58ebe7c753e5d23e4b4e1898d58cee548d381cbcb973f84489ce85f33ef07`；
  - 策略：1,020 字节，SHA-256
    `d69c362115a4c71a916a04a53d25b9fa743ffee0d0bd63de1b0b6ca9f9845463`；
  - 训练历史：2,338,161 字节，SHA-256
    `07c16da7ff10308989cd6b8a7688d2696612b6ed5dcd24d3c86c879e87181e34`。
- step 9000 checkpoint 已独立加载：step、symbol、timeframe、数据 SHA、
  评分合同均正确；模型、优化器、最优快照和 9000 条训练历史完整。
- 后处理四层全部 READY：训练 best score 1.5200775862；同训练数据、含成本
  replay 的 Sharpe=0.9352、Sortino=1.3349、总收益=2.27173、162 笔；
  该数值不是 sealed OOS 成绩。虚拟信号完成，当前事件为 HOLD。生产队列
  000617 正式进入 READY。
- 队列随即按既有串行合同派发 000725 / run
  `run_20260723T235959Z_15080906` / Slurm 581904；18:27 在 cu01
  RUNNING。批次为 `READY=2 / TRAINING=1 / QUEUED=47`，
  SQLite `integrity_check=ok`。
- READY 后增量备份完成，远端固定根现有 1,250 文件、
  3,186,377,616 字节，`.incoming` 为空。000617 本轮 17 个普通持久文件、
  9,462,913 字节在本机与远端逐文件 SHA-256 完全一致，差异 0。
  生产队列在线快照 SHA-256 为
  `c67c0c8359fe42e3d175a135966a9c113b861e9323d576983579a84a41b1a18a`，
  000617 READY 身份与恢复谱系正确；信号账本快照 SHA-256 为
  `a2e851532fe83f5a48cc850d35c46b10d9bf40b9af4264349ceb7231f7c4cb7d`。
  两库均 `integrity_check=ok`、journal mode=DELETE，远端 WAL/SHM 均为
  0 字节。

最终结论：000617 的 NODE_FAIL 恢复已从授权、谱系绑定、checkpoint 导入、
真实续训、9000 步完成、315→3 结果适配、回测、信号、队列 READY 到异地
备份全部闭环。旧 run 581389 及其证据保持不变，sealed OOS 未触碰。
