# GPT-5.6-sol 移交说明

## 1. 项目定位

AlphaMaster 是基于原始 fork 的轻量二次开发项目，保留因子训练、策略回测和实时信号能力。当前核心工程目标是打通 Windows 本机控制端与 Linux Slurm 服务器之间稳定、可审计、可恢复的联通和交互。

本移交只覆盖 `Jinqingchang/AlphaMaster`。不得读取、比较、规划、修改或执行任何其他仓库，也不得把其他项目的业务目标写入 AlphaMaster 工单。

## 2. 当前端到端链路

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

这条链路的工程完整性是当前开发重点；训练步数或策略收益不属于联通验收的替代指标。

## 3. 当前已发布状态

- 私有仓库：`Jinqingchang/AlphaMaster`
- 分支：`main`
- 当前功能提交：`d4dcb75279387c282e13b56e20e843d6801d4065`
- 三页前端已统一大字体、舒展排版、侧边栏、卡片、表单、按钮和 Tab。
- 新 MT5/OKX 数据可自动生成 sidecar；旧 MT5 使用显式两阶段登记。
- A 股旧数据可通过严格转换生成规范 Parquet 与 manifest。
- checkpoint 按 `timeframe + data_sha256 + run` 隔离。
- 策略、checkpoint、Slurm run/result、发布指针和训练包贯穿完整数据身份。
- 回测可独立选择测试数据，区分 replay、out-of-sample 和 diagnostic-overlap。
- 训练包 v2 已实现路径、成员、大小、CRC、哈希、身份和失败回滚检查；Web 导入入口仍固定关闭。

## 4. manifest 的真实边界

- 原作者仓库的 AlphaMaster 核心算法不要求 sidecar。
- fork 提交 `c3717be` 为 Slurm 远程训练新增 manifest 身份合同。
- 本地 Parquet 读取/训练不强制 manifest。
- Slurm 远程训练强制 sidecar，以绑定来源、哈希、数据范围和列合同。
- 旧 MT5/OKX 使用 `mt5_legacy_attested` / `okx_legacy_attested`；不得冒充新版导出器验证身份。

## 5. 当前验证

- 主开发线程直接相关回归：140 通过、0 失败。
- 独立严格审查扩大相关回归：229 通过、0 失败。
- 完整 unit：444 通过、8 个既有失败，不能宣称全仓全绿。
- 三页真实浏览器验证：控制台错误 0、页面错误 0、HTTP 错误 0。
- NVDA 裸旧数据在 Slurm 模式正确显示“本地未登记 / 需要先注册”，训练按钮禁用。
- 旧 MT5 668 文件批量计划只读生成，未 apply。
- 已有真实 Slurm 冒烟证明上传、提交、训练、下载和产物校验链路可完成。
- 本轮不启动正式训练，不配置 TradingView，不接入任何订单执行系统。

完整证据见 `docs/VALIDATION_EVIDENCE.md`。

## 6. 已知开放项

- 当前 `9000` 步与 `00:30:00` Slurm 时限明显不相容；正式训练预算未冻结。
- 12 CPU 没有与 8 CPU 同口径的稳定性能结论。
- 网络中断、状态未知、重复提交、取消竞态、下载中断和本机重启恢复仍需持续用硬验收防回归。
- 历史策略缺少完整训练数据身份时，不能直接宣称可做严格样本外。
- 旧 MT5 批量计划尚未获得 apply 授权。
- Web/API 训练包导入仍固定关闭。
- TradingView 继续暂缓。
- 工程链路通过不代表策略盈利；仍需成本、样本外、稳健性和过拟合研究。

## 7. 安全与数据边界

- `.env`、真实设置、token、密码、私钥和账户信息不得进入 Git 或移交包。
- 原始行情、训练数据、数据库、checkpoint、模型、训练历史、日志和运行态不得提交。
- 当前 Git 树仍有历史跟踪文件 `training_time_XAUUSD.json`；新安全包必须排除。是否删除当前文件或净化历史需用户单独授权。
- 历史 Git 对象曾含旧 Tushare token；当前代码已改为环境变量读取，但旧值仍应视为已暴露并轮换。
- 外部模型不得连接本机、远程节点、MT5 或 Slurm，不得启动训练、取消任务或修改服务器状态。
- 登录节点只作 SSH 跳板；交互控制只能在允许的计算节点入口执行，实际训练节点由 Slurm 调度。

## 8. 对网页版 GPT 的任务

1. 只按 AlphaMaster 私有仓库的 GitHub 代码证据重建系统事实。
2. 以“轻量二次开发并打通本机—服务器联通和交互”为唯一总目标。
3. 重点审查上传、提交、状态、日志、取消、恢复、下载、身份校验和 Web 交互的缺口。
4. 输出依赖有序的小工单和二元硬验收，不直接编码或运行。
5. 对无法看到的本机 skills、memory、进程、凭据、数据和未提交状态列为本地适配项。
6. 不把单元测试、Mock 或一次成功冒烟自动等同于长期联通可靠。
7. 不引入券商、订单、跨项目集成或与当前目标无关的大规模重构。

启动提示词见 `docs/GPT_WEB_PRO_EXTENDED_TASK.md`。
