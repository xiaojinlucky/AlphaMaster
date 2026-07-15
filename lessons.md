# Lessons

- 2026-07-14：AlphaMaster 的远端安装、调试、测试和 `sbatch` 提交不得在 `login-node` 执行。每批远端操作前使用 `D:\Desktop\codex-remote-tools\check-best-node.ps1` 在 `compute-node-11/12/13` 中选择实时最优节点；`login-node` 仅作为 SSH `ProxyCommand` 跳板。提交所在节点不等于训练节点，训练资源与落点由 Slurm 调度器决定，不使用 `--nodelist` 固定节点。
