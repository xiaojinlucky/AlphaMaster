# WO-AM-04 训练产物清单合同修复与首件恢复验收

时间：2026-07-29 10:33–11:00（Asia/Shanghai）

## 结论

- 用户授权单独解冻训练产物清单合同后，修复只落在训练源码冻结集合之外：
  `scripts/slurm_result_adapter.py` 与 `web/slurm_training_client.py`。
- 34 个冻结训练文件未改；当前源码合同 SHA-256 仍为
  `0e5ca8c3ee924b94757e453e01637d4e1a7f85c7ae9a50ad9baacebdf0564ab0`。
- 适配器完整校验原始 452 件产物的路径、大小和 SHA-256，再只发布同一
  checkpoint run 中唯一最高步数 checkpoint、一个策略和一个训练历史；
  不修改或删除远端原始清单及 450 个 checkpoint。
- 真实首件 000333 / 577313 校验结果：
  原始清单 SHA-256
  `4f80b73ec01eaf2030705415149b9baaa7710efa65f668244204c5c3f7618807`，
  原始 452 件，发布 3 件，最高步数 9000，省略 449 个中间 checkpoint。
- 首轮独立对抗审查拒绝放行并发现 3 个 P1：祖先目录符号链接、适配器
  身份未锁定、清单解析和记账哈希不属于同一字节快照。修复后目录链和叶文件
  使用 `openat + O_NOFOLLOW` 固定句柄读取；清单从同一句柄字节解析和哈希；
  客户端强制核对适配器 SHA-256。
- 第二轮复审进一步指出“适配器自报 SHA”不构成独立认证。最终实现由客户端
  先校验本机适配器字节 SHA-256，再把同一串已校验字节经 SSH 标准输入交给
  远端 Python 执行；远端磁盘副本不再参与执行或身份判断。当前受信字节
  SHA-256 为
  `fdddd12bf0ac1c1fa471807790533b386fb3a02216c9ab3b5c233eabc69373ce`。
- 最终独立复审：`ACCEPT`，P0/P1/P2=`0/0/1`；唯一 P2 是固定测试尚未
  直接覆盖符号链接、读取期变化、本地哈希不匹配和 SSH stdin 同字节正反例。
  真实 452→3 全链已通过，因此不阻断。
- 固定测试：134 通过、0 失败。

## 首件恢复

- 恢复前精确核对 run、job、旧错误、源码 SHA、runtime commit、
  Slurm `COMPLETED/0:0`；恢复前 state 与队列库快照位于
  `scratch/goal_am_custody/manifest_contract_recovery_20260729T024825Z/`。
- 只恢复同一 run/job，不重提训练：000333 最终训练状态、回测、虚拟信号和
  队列项目均为 `READY`，队列错误清空。
- 本机归一化结果清单 SHA-256：
  `bab628d6b4aca07b4db9c3b8c718fef231ccd59710e0ef7ce01f12065ea9daa1`。
- 结果仍绑定 runtime commit
  `bb280ca2dada4b087ce1bf5d86f1292b73c576b9`、评分合同
  `open_t1_t2_same_index_tail2_v1` 和 34 个源文件。

## 队列续跑与备份

- 000333 进入 `READY` 后，000617 按原冻结源码和 runtime commit 提交为
  Slurm 581389；11:00 时为 `RUNNING`。
- 第 29 次增量备份传输后，独立核对首件 19 个持久资产、
  16,326,593 字节，本机与
  `compute-node-11:/hwdata/home/jinqc/AlphaMaster-backup/`
  逐文件 SHA-256 完全一致。
- 在线队列数据库另用 SQLite 一致性快照备份；本地/远端 SHA-256 均为
  `64721b27cd0e9674c70f4d3d2c01b8e2f9865df596a0ab8dac51ab7455972ba8`，
  远端 `PRAGMA integrity_check` 返回 `ok`。
- 备份脚本唯一报错项是 Web 运行中持续变化的
  `training_queue_v2.sqlite3-wal`。WAL/SHM 是运行态旁路文件，不能脱离主库
  独立恢复；本次以在线一致性快照替代，未停止 Web、未删除任何备份文件。
