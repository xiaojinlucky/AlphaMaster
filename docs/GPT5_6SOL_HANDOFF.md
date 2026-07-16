# GPT-5.6-sol 移交说明

## 1. 项目定位

AlphaMaster 是独立的因子训练、策略回测、实时信号与 Slurm 训练管理项目。它不是当前“PA_Agent 打通到实盘”目标的订单执行层，也没有与 PA_Agent 共享运行态、数据库、凭据或执行代码。

用户允许网页版 GPT 同时只读两个私有仓库，以便获得更完整的工作背景；这不等于授权跨项目接入。PA_Agent 仍是实盘主项目，AlphaMaster 只可作为数据身份、训练、回测、状态机、哈希和审计模式的参考。

## 2. 当前已发布状态

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

## 3. manifest 的真实边界

- 原作者仓库的 AlphaMaster 核心算法不要求 sidecar。
- fork 提交 `c3717be` 为 Slurm 远程训练新增 manifest 身份合同。
- 本地 Parquet 读取/训练不强制 manifest。
- Slurm 远程训练强制 sidecar，以绑定来源、哈希、数据范围和列合同。
- 旧 MT5/OKX 使用 `mt5_legacy_attested` / `okx_legacy_attested`；不得冒充新版导出器验证身份。

## 4. 当前验证

- 主开发线程直接相关回归：140 通过、0 失败。
- 独立严格审查扩大相关回归：229 通过、0 失败。
- 完整 unit：444 通过、8 个既有失败，不能宣称全仓全绿。
- 三页真实浏览器验证：控制台错误 0、页面错误 0、HTTP 错误 0。
- NVDA 裸旧数据在 Slurm 模式正确显示“本地未登记 / 需要先注册”，训练按钮禁用。
- 旧 MT5 668 文件批量计划只读生成，未 apply。
- 本轮未启动正式训练、真实交易或 TradingView 配置。

完整证据见 `docs/VALIDATION_EVIDENCE.md`。

## 5. 已知开放项

- 当前 `9000` 步与 `00:30:00` Slurm 时限明显不相容；正式训练预算未冻结。
- 12 CPU 没有与 8 CPU 同口径的稳定性能结论。
- 历史策略缺少完整训练数据身份时，不能直接宣称可做严格样本外。
- 旧 MT5 批量计划尚未获得 apply 授权。
- Web/API 训练包导入仍固定关闭。
- TradingView 继续暂缓。
- 工程链路通过不代表策略盈利；仍需成本、样本外、稳健性和过拟合研究。

## 6. 安全与数据边界

- `.env`、真实设置、token、密码、私钥和账户信息不得进入 Git 或移交包。
- 原始行情/交易/训练数据、数据库、checkpoint、模型、训练历史、日志和运行态不得提交。
- 当前 Git 树仍有历史跟踪文件 `training_time_XAUUSD.json`；新安全包必须排除。是否删除当前文件或净化历史需用户单独授权。
- 历史 Git 对象曾含旧 Tushare token；当前代码已改为环境变量读取，但旧值仍应视为已暴露并轮换。
- 外部模型不得连接本机、远程节点、MT5、Slurm 或券商私有接口，不得启动训练或发送订单。

## 7. 与 PA_Agent 实盘目标的关系

PA_Agent 的完整实盘闭环至少需要 TradeIntent、确定性风险闸门、人工确认/临时武装、券商路由、强订单账本、幂等、未知状态对账、部分成交、保护单、持仓/资金/PnL 监控、离场和状态回写。

AlphaMaster 当前没有经过验证的生产订单生命周期。外部模型可以比较以下模式是否值得在 PA 中重新实现：

- 数据/产物身份和哈希贯穿
- 失败关闭
- 原子状态发布
- 恢复同一 run/job
- 独立严格审查

不得直接复制 AlphaMaster 的策略、checkpoint、运行态或未验证执行代码到 PA。

## 8. 对网页版 GPT 的任务

1. 先按 GitHub 代码证据重建 PA_Agent 和 AlphaMaster 的真实边界。
2. 以 PA_Agent 打通实盘为唯一总目标，AlphaMaster 只作只读参考。
3. 输出依赖有序的小工单和二元硬验收，不直接编码。
4. 对无法看到的本机 skills、memory、进程、凭据和未提交状态明确列为本地适配项。
5. 不把 Mock、demo、只读预检或代码完成冒充实盘打通。
6. 最终真实 canary 必须作为独立、再次授权的工单。

启动提示词见 `docs/GPT_WEB_PRO_EXTENDED_TASK.md`。
