# AlphaMaster — 当前验证证据

## 1. 已发布基线

- 私有仓库：`Jinqingchang/AlphaMaster`
- 分支：`main`
- 当前功能提交：`d4dcb75279387c282e13b56e20e843d6801d4065`
- 提交范围：43 个文件，7,430 行新增，373 行删除
- 本地 HEAD、`origin/main` 与 GitHub `main` 在推送后完全一致

## 2. 自动化测试

- 主开发线程的直接相关回归：140 通过、0 失败。
- 独立严格审查扩大相关回归：229 通过、0 失败。
- 完整 `tests/unit`：444 通过、8 失败。
- 2026-07-17 仅修改文档后尝试复跑相关回归，两次 watchdog 分别在约 180 秒和 120 秒到期，均未返回最终 pytest 汇总；进程随后退出且临时目录已清理。因此这两次不计入通过或失败，也不改写上述已有有效计数。
- 8 个失败集中在互相矛盾的算子/词表数量旧断言、旧特征维度断言、依赖本机策略文件的 Runner 测试和依赖固定历史时间的训练计时测试。
- 这 8 项没有计入通过，也没有被描述为全仓全绿。

## 3. 语法与静态检查

- `web/static/app.js` 通过 `node --check`。
- 变更涉及的 Python 模块通过 `py_compile` / `compileall`。
- `git diff --check` 通过。
- 提交前 staged snapshot：43 个文件、14 新增、29 修改、0 删除、0 重命名。
- staged 文件中没有二进制、敏感扩展名或超过 50 MiB 的文件。

## 4. 安全扫描

- 使用 gitleaks v8.30.1 对 staged snapshot 扫描：0 泄漏。
- 旧源码 ZIP 解包后 gitleaks 为 0，但规范复核发现 Git 树中的历史文件 `training_time_XAUUSD.json` 被一并归档。
- 该旧 ZIP 没有上传 GitHub，现标记为 superseded；新移交包必须显式排除 `.git`、`.env*`、原始数据、日志、checkpoint、数据库、`training_time_XAUUSD.json` 和其他运行态，再重新扫描。
- 历史 Git 对象曾包含旧 Tushare token；当前代码已改为环境变量读取，但历史凭据仍应视为已暴露并轮换。

## 5. 真实页面验证

- 使用隔离的本机 Edge/CDP 加载 `127.0.0.1:8765`，禁用缓存后检查三页。
- 01/02/03 的侧边栏激活项宽度均为 255 px，高度均为 52 px。
- 菜单文字 17 px / 26 px 行高；页面标题 26 px / 36 px；卡片标题 22 px / 30 px。
- 02 的四个 Tab 自然宽度排列，右侧保留空白，无异常拉伸。
- NVDA 裸旧文件显示“本地未登记 / 需要先注册”，远程训练按钮禁用，注册入口可见。
- replay、out-of-sample、diagnostic-overlap 三种评估标签均由真实浏览器上下文执行 helper 验证。
- 最终报告：控制台错误 0、页面错误 0、HTTP 错误 0。

## 6. 真实数据与只读计划

- 新 MT5 `NVDA_M5`：50,000 根已收盘 K 线，sidecar 与 Parquet 哈希合同通过；未启动训练。
- 旧 OKX `BTCUSDT_H1`：身份为 `okx_legacy_attested`；本地 loader 与 Slurm run 身份一致。
- 旧 MT5 批量计划：668 个文件；537 可登记、125 不足 bars、6 数据合同失败。
- 计划 SHA 和来源报告 SHA 已复核；sidecar、partial、lock 数量均为 0，证明未执行 apply。

## 7. 独立审查结论

独立审查者从需求完整性、逻辑正确性、边界情况、代码质量、测试覆盖和实际运行结果六方面确认本轮数据通道目标通过。审查未授权或执行正式训练、真实交易、TradingView 配置、旧 MT5 批量 apply。
