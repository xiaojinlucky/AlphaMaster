# Windows MT5 与 Slurm Worker 对接设计

## 目标与边界

第一阶段只实现：

```text
Windows MT5 已收盘 K 线
→ 本机 Parquet 与数据 manifest
→ 本机 AlphaMaster Web 控制台
→ 系统 OpenSSH / SCP
→ 计算节点上的固定 Slurm 控制器
→ Slurm CPU Worker
→ 哈希校验后的策略、checkpoint 与结果 manifest 回传
```

不实现自动交易、GPU 重构、公网 Web、服务器常驻 API、自建队列或策略盈利性结论。

## 硬约束

- `login-node` 只作为 SSH `ProxyCommand` 跳板，不执行安装、调试、测试、提交或训练命令。
- 每批远端操作前调用 `D:\Desktop\codex-remote-tools\check-best-node.ps1`，只允许返回 `compute-node-11/12/13`。
- 推荐计算节点只承担交互与 `sbatch` 提交；训练落点由 Slurm 调度器决定，禁止 `--nodelist`。
- Slurm 显式使用 `cpu` 分区和 `normal` QOS，不依赖默认 `fat`，不使用账号无权访问的 `tmp` 分区。
- 远端显式使用 `/hwdata/home/jinqc/.local/bin/python3.11` 创建项目 `.venv`；禁止使用默认 Python 3.6。
- Worker 不安装 `MetaTrader5`，不接收或继承 MT5、AI、飞书等凭据。
- 远程训练失败时不得退回本机训练。

## 最小架构

### 本机

- `scripts/export_mt5_parquet.py`：从正在运行且已登录的 MT5 导出准确品种名的已收盘 K 线；按时间升序、去重、原子写 Parquet，并生成包含 UTC 时间范围、行数和 SHA-256 的数据 manifest。
- `web/slurm_training_client.py`：仅用系统 `ssh` / `scp` 参数数组，负责节点选择、上传、提交、查询、取消、日志与结果下载。
- `web/slurm_training_manager.py`：持久化 run 状态，向现有 Web API 提供与本地训练管理器兼容的 `start/status/stop/tail_log` 接口。Web 重启后从本地状态文件恢复未完成任务。
- `local_runs/<run_id>/`：保存本机状态、上传 manifest 和下载 staging；完成哈希校验后才发布正式结果。
- `published_training/current_<symbol>.json`：单一原子提交点，绑定同一 READY run 的 checkpoint、训练历史、策略、数据身份与产物哈希。Web 只从该指针读取 Slurm 产物，根目录兼容副本不会参与训练包组装。

### 服务器共享目录

```text
/hwdata/home/jinqc/Quant/AlphaMaster/runs/<run_id>/
├── input/
│   ├── XAUUSD_H1.parquet
│   └── run_manifest.json
├── logs/
│   ├── slurm.out
│   └── slurm.err
├── checkpoints/
├── strategies/
└── output/
    └── result_manifest.json
```

- `scripts/slurm_control.py`：纯标准库固定控制器。严格校验命令、run ID、job ID、数据文件、分区与资源范围；使用 `subprocess.run([...], shell=False)` 调用 `sbatch/squeue/sacct/scancel`；取消前核对当前用户、job name 与 run ID。
- `scripts/train_alphamaster.sbatch`：固定脚本，只接收合法 run ID；`umask 077`，使用干净环境，显式设置线程变量和 `.venv/bin/python`，在 run 目录运行 Worker。
- `scripts/train_slurm_worker.py`：复核输入哈希、源码提交和文件指纹；记录实际计算节点与环境版本；运行 `train_file.py`；无论成功失败都尽力生成真实 result manifest，失败不伪装完成。

训练进程以 run 目录为工作目录。现有相对路径 `checkpoints/`、`strategies/` 和 `training_history_*.json` 因而天然按 run 隔离，不改训练引擎的路径体系。

## 状态机与恢复

本机状态只能按以下方向前进：

```text
PREPARING → UPLOADING → SUBMITTING → SUBMITTED → PENDING → RUNNING
→ COMPLETED → DOWNLOADING → READY
                                  ↘ CANCELLING → CANCELLED
```

失败终态为 `FAILED` 或 `CANCELLED`。上传完成前不得提交；`sbatch --parsable` 未返回纯数字 job ID 不得进入 `PENDING`；Slurm 终态必须由 `squeue/sacct` 判断；取消请求只进入 `CANCELLING`，不能把 `scancel` 返回成功当作已取消。

节点选择、SSH/SCP 和超时属于可重试传输错误：保持同一 run 状态并在 Web 重启或下次查询时幂等重放。取消状态若仍看到远端作业处于排队或运行，会幂等重发 `scancel`。确定性身份、路径、资源、manifest 或哈希错误才进入 `FAILED`。进入 `READY` 前必须逐项匹配本机 run manifest 的数据、源码、训练参数和资源，并同时存在可解析的目标策略、checkpoint 和训练历史；三类产物通过原子指针作为同一 run 的不可拆分集合发布，Slurm 模式的每个通用读取入口都会复算全部必需产物哈希，任一不符即整套拒绝。

本地状态采用临时文件加原子替换。每次查询可重新选择健康计算节点，因为 `/hwdata` 为共享存储；计算节点切换不改变 run ID 或 job ID。

Slurm 第一阶段的恢复含义是“网络或 Web 中断后接管同一 run/job”，不包含把上一个 READY run 的 checkpoint 上传到新 run 继续训练。验证后的 checkpoint 与 history 会发布到本机项目目录供 Web 查看和导出；跨 run 续训需另行增加 checkpoint 来源身份合同。

## 安全设计

- Web 强制绑定回环地址；删除任意来源 CORS。
- Host/Origin 只允许同端口的 `127.0.0.1` 或 `localhost`。
- 所有写请求要求启动时随机生成、由同源 session 接口读取的控制 token；跨站表单和无 token 请求失败。
- API 不返回 AI key、飞书 webhook/secret、SSH 路径或 traceback；调试日志接口仅在明确 debug 模式开放。
- 第一阶段禁用外部 ZIP/PT 训练包导入和飞书测试接口。
- 生产 checkpoint 加载显式使用 `weights_only=True`；SHA-256 只用于完整性，不把不可信 pickle 变成可信。
- 远端 host、用户、根目录、控制器路径不接受 Web 请求覆盖；节点选择结果必须映射到固定计算节点别名。
- SSH 使用 `BatchMode=yes`、`StrictHostKeyChecking=yes`、超时和现有 SSH config；不使用密码，不读取私钥内容。
- 上传先写 `.partial`，服务器复核大小与哈希后原子发布；下载先进入 staging，并按 manifest 白名单、大小和哈希验证；发布后的通用读取与训练包导出都会再次复算 SHA-256，防止本机落盘后的替换或损坏被继续读取。

## 训练参数与资源

- `train_file.py` 增加严格正整数 `--train-steps`，默认仍为项目现有值；不修改正式默认训练预算。
- 冒烟数据使用 `XAUUSD / H1`，至少满足 `MIN_BARS=3000`，只含已收盘 K 线。
- 先运行 1、4、8 CPU 的短基准并通过 `sacct` 记录 `Elapsed/TotalCPU/MaxRSS`，再选择正式冒烟资源；基准前不固化 CPU、内存和时限。
- 为得到 checkpoint，端到端冒烟至少运行到现有 checkpoint 周期；该结果只证明系统链路，不证明策略有效。

## 放弃的复杂方案

- 不增加 FastAPI/消息队列常驻在服务器，减少暴露面和运维状态。
- 不引入 Paramiko；复用已验证的系统 OpenSSH、host key 与 ProxyCommand。
- 不手工固定 Slurm 节点；动态提交入口不能替代调度器。
- 不重构 GPU 或公式批处理；当前模型明确为 CPU 路径。
- 不修复并重新开放不可信 ZIP/PT 导入；第一阶段直接关闭。

## 前端决定

本轮不新增页面、路由、导航、布局、组件层级、角色可见性或视觉风格；现有“开始/停止/日志/进度”交互保持不变，后端切换为 Slurm 并返回兼容状态。因此没有新的前端设计任务，不进入 Stitch；若实现中出现上述任一变化，必须停止并先走 Google Stitch。

## 验证

- 单元：输入校验、状态跃迁、原子持久化、节点白名单、命令参数、Slurm 状态映射、manifest 与哈希、路径包含、秘密脱敏、CSRF/Host/Origin。
- 本机：Python 3.11 隔离环境、MT5 import/initialize、准确品种、已收盘导出、Parquet schema/排序/去重/UTC/哈希。
- SSH：host key、只读命令、文件往返、网络失败、禁止 login-node。
- Slurm：固定脚本最小任务、合法 job ID、PENDING/RUNNING/终态、日志、取消、失败状态、实际计算节点共享路径。
- AlphaMaster：缩短训练、结果下载、checkpoint/策略/result manifest 哈希、Web 重启恢复、无本机回退。
- 交付：主 Agent 自审和独立多 Agent 否定性/追问性审查；清理 `scratch/` 与可再生临时文件；只报告通过、失败、未执行数量。

## 2026-07-14 CPU 短基准

同一份 `XAUUSD / H1`、50000 根已收盘 K 线、10 个训练步、从头训练；实际节点由 Slurm 分配。

| CPU | Slurm job | 实际节点 | Elapsed | TotalCPU | MaxRSS（Slurm `.batch`） |
|---:|---:|---|---:|---:|---:|
| 1 | 542366 | cu01 | 00:02:05 | 01:35.812 | 1376K |
| 4 | 542370 | cu03 | 00:02:35 | 04:32.944 | 3204K |
| 8 | 542374 | cu01 | 00:01:20 | 08:49.811 | 3204K |

短基准中 8 CPU 的单次墙钟最短。2026-07-15 按用户明确要求将正式任务默认资源调整为 12 CPU；12 CPU 尚未做同口径短基准，因此这项调整是资源配置决定，不声称已经证明比 8 CPU 更快。三次历史运行跨节点且仅各测一次，不外推为稳定性能规律。该集群 `.batch` 的 `MaxRSS` 数值异常偏低，仅按 Slurm 原始记账字段留存，不据此声称模型真实内存占用。

## 最终 Web 冒烟

- 本机 Web：`127.0.0.1:8765`；无控制令牌访问受保护 API 返回 403，带启动期随机令牌可提交。
- run/job：`run_20260714T145726Z_5f5c3b75` / `542383`。
- Slurm：8 CPU、20 步、实际节点 `cu01`、`Elapsed=00:02:27`、`TotalCPU=17:32.661`、退出码 `0:0`。
- 数据：`XAUUSD_H1.parquet`，50000 根已收盘 K 线，SHA-256 `aa6df4466a0b16efa96d3a0ba643380dc1b6c2dc8d7aad1aefb5b1c23df8217d`。
- 回传：checkpoint、strategy、training history 共 3 个产物；大小和 SHA-256 全部通过，checkpoint 使用 `torch.load(..., weights_only=True)` 加载通过。
- 这只验证工程链路和产物完整性，不验证策略盈利性；正式默认训练步数仍为项目既有 9000，冒烟的 20 步不写入持久 `.env`。

## 对抗审查修复后的回归

- 成功路径：最终回归 run `run_20260714T155605Z_1e87413c` / job `542541`，8 CPU，实际节点 `cu01`，2 步，`Elapsed=00:00:43`，严格身份校验后 3 个产物进入 `READY`；原子发布指针、根目录兼容策略与导出训练包均绑定该 run，产物哈希无不匹配。
- 取消路径：run `run_20260714T153405Z_0db8307c` / job `542524`，1 CPU，实际节点 `cu01`；Web 在 RUNNING 后请求取消，最终由 `sacct` 确认 `CANCELLED`，未下载或发布不完整结果。
- 失败路径单元测试覆盖：准备/上传/提交响应丢失、Web 重启恢复、下载瞬断重试、完整性错误终止、取消与自然完成竞态、取消请求丢失后重发、低分新 run 不得形成跨 run 混合训练包、缺失后端配置拒绝启动。
