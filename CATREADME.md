# AlphaMaster 仓库速读

**开源协议**：GNU Affero General Public License v3.0 (AGPL-3.0)。修改、分发或通过网络提供服务时，须按相同协议公开源代码。详见根目录 `LICENSE`。

AlphaMaster 是一个因子训练、策略回测、实时信号和 Slurm 远程训练控制项目。当前二次开发保持轻量，核心工程目标是打通 Windows 本机控制端与 Linux 服务器的联通和交互。

## 当前主流程

1. 本机选择并校验 Parquet 数据及来源 sidecar。
2. Web 控制端通过固定 OpenSSH/SCP 通道准备并上传独立 run。
3. 服务器固定控制器向 Slurm 提交训练任务。
4. 本机持续查询状态和日志，并支持取消、网络恢复和 Web 重启接管。
5. 训练完成后下载策略、checkpoint、训练历史和结果 manifest。
6. 校验 run、数据身份、文件大小和 SHA-256 后原子发布产物。
7. 训练、回测和实时分析共用同一套公式解释与信号口径。

## 代码组织

- `web/`：本机 FastAPI 控制台、训练状态与 Slurm 客户端/管理器。
- `scripts/`：MT5 数据导出、Slurm 固定控制器、Worker 和数据登记工具。
- `data_pipeline/`：Parquet、MT5、数据来源和身份合同。
- `model_core/`：特征、算子 DSL、StackVM、训练和评分。
- `strategy_manager/`：信号与仓位状态逻辑，不含可用的真实下单器。
- `execution/`：MT5 实时报价兼容层，不含生产订单适配器。
- `backtest_viz/`：回测与图表。

本项目当前不建设券商订单执行通道，也不把实时信号展示描述为自动交易。完整现行说明以根目录 `README.md`、`CONTEXT.md` 和 `docs/CODEX_PROJECT_RULES.md` 为准。
