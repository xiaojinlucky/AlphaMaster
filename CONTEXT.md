# CONTEXT

正在做：`BTCUSDT_H1` 两步真实 Slurm 新手教学训练、使用文档、界面收口和私有 GitHub 备份均已完成；等待用户决定正式训练的步数与时限。

上次停在：用户授权后已取消误启动的 XAUUSD job `542862`；BTC run `run_20260715T031117Z_2ad7721d` / job `542890` 在 Slurm 分配的 `cu09` 上以 12 CPU 完成，2 步耗时 `00:17:50`，状态 `READY`。57,553 根闭合 OKX H1 K 线的数据身份贯穿 run/result/published strategy/export package；3 个产物哈希全部通过。演示进程已结束，桌面快捷方式已按日常配置重启，当前 9000 步、调试关闭；界面按最近已发布 run 正确显示 `2/2、100%、completed`，没有活动 Slurm 作业。

近期决定：2026-07-15 按用户要求将正式 Slurm 任务默认 CPU 从 8 调整为 12；12 CPU 尚未做同口径基准，不作性能更优声明。当前 `9000` 步与 `00:30:00` 时限对 BTC 明显不相容，在用户确认正式预算前不得直接提交。快捷方式保持“项目虚拟环境可执行文件 + 固定工作目录”结构；关闭 Web 服务窗口不取消 Slurm 作业。登录节点只作 SSH 跳板，训练节点由 Slurm 调度。

Git 管理：2026-07-15 已建立私有仓库 `https://github.com/Jinqingchang/AlphaMaster`，本机 `main` 默认推送到该仓库；原作者仓库保留为 `upstream`。`.env`、Parquet、检查点、训练产物和本机运行状态继续由 `.gitignore` 排除。

AI 配置：2026-07-15 已在本机忽略文件 `web_settings.json` 中配置 DeepSeek 用户密钥，通道固定为官方 `deepseek-v4-flash` / `https://api.deepseek.com`；已通过 AlphaMaster 自身流式调用完成真实响应验证。密钥不写入项目文档、Git 索引或远端仓库。
