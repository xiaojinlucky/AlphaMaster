> **历史提示词（已被取代，不得执行）**：现行网页版总控入口为
> `docs/WEB_GPT_CONTROLLER_PROMPT.md`；`docs/GPT_WEB_PRO_EXTENDED_TASK.md`
> 也只是历史未来函数专项。

# 给 GPT-5.6 Pro Extended Thinking 的提示词

你收到的是 AlphaMaster 的“稳定基线 + 未提交候选实现”安全审查包，不是一个已经完成的发布版本。

请先完整阅读：

1. `CONTEXT_GPT56.md`
2. `source/CONTEXT.md`
3. `source/lessons.md`
4. `source/README.md`
5. `SELECTED_WORKTREE_STATUS.txt`
6. `EXCLUDED_WORKTREE_FILES.txt`
7. `TRACKED_DIFF.patch`
8. `SELECTED_UNTRACKED_FILES.tsv`
9. `SOURCE_MANIFEST.tsv`

然后重点检查 `source/docs/GPT56_PRO_EXTENDED_HANDOFF.md` 列出的四个阻断项，并从以下六方面独立验证：需求完整性、逻辑正确性、边界情况、代码质量、测试覆盖、实际运行结果。

要求：

- 不要假设现有数据身份设计正确。
- 不要把“严格校验”自动等同于“研究语义正确”。
- 特别分析训练数据身份与样本外评估数据身份应该如何分离。
- 对旧训练入口、Windows 长路径原子发布、多数据身份 checkpoint 选择给出唯一推荐方案。
- 只保留会改变结论、实现或验证方式的问题。
- 每个问题给出影响、证据、根因、最小修法、测试和剩余风险。
- 区分代码可静态证明的结论与必须靠真实 Windows、真实 A 股样本或真实 Slurm 运行验证的结论。
- 不启动训练，不配置 TradingView，不要求任何密钥、原始数据或 checkpoint。
- 使用中文回答。

最终输出应能直接作为主开发线程的修复清单，并明确说明当前候选是否可以进入稳定主线。
