# A 股量化候选框架隔离实验报告（2026-07-24）

> 执行状态说明（2026-07-25）：本文的隔离实验结果和证据边界仍有效，但第八节
> 的执行顺序已按最新运行态更新。本文不是启动训练、改写队列或写入数据库的
> 授权；当前唯一外部总控入口是 `docs/WEB_GPT_CONTROLLER_PROMPT.md`，
> 本机执行边界以 `CONTEXT.md` 为准。

## 一、结论

两组固定样本已在本机隔离环境中用全新输出目录完整复跑，且没有接入
AlphaMaster 生产链：

| 实验 | 结果 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| AlphaMaster / AKQuant / RQAlpha 成交费用差分 | `PASS_WITH_DOCUMENTED_DIFFERENCE` | AlphaMaster 与手算、AKQuant 在固定一手买卖上完全一致；RQAlpha 默认股票费用实现对齐佣金倍率后仍少 0.02 元过户费 | 三套框架完整订单生命周期、公司行动和恢复语义等价 |
| AlphaMaster / Qlib / vn.py 研究层差分 | `PASS_WITH_RUNTIME_BOUNDARY_AND_ADOPTION_FINDINGS` | AlphaMaster 真实目标收益构造函数、Qlib 上一时点信号源码合同、vn.py 正常区间及三个反例均被固定证据复现 | 尚未证明完整 PnL 端到端时钟；Qlib 未完整运行；生产历史股票池尚未落地 |

本轮最重要的新发现不是第三方框架，而是 AlphaMaster 自身的四处 P0
评分口径问题：

1. `AlphaEngine._compute_ic` 额外右移一根。
2. `MT5Backtest._ts_ic_stability` 有相同额外右移。
3. PnL/换手成本路径没有裁掉目标收益末尾两个补零，最后两根可能只有
   换手成本、没有可实现的未来收益。
4. 模块级 `model_core.evaluator.score_all(..., horizon=1)` 和
   `EffectivenessEvaluator(target_horizon=1)` 都默认只裁 1 根；
   `prune_features.py` 调用模块级 `score_all` 时没有显式传入 2，而当前
   目标收益需要裁 2 根。

这些问题会影响训练评分、冠军选择和特征评估。隔离实验本身只复现并冻结
证据；随后已在新的本地源码身份中修复生产评分代码。正在运行的旧源码
Slurm 批次没有被取消、重提或覆盖，其结果仍只能作为旧评分语义诊断基线，
不能与新合同结果混用。

生产修复后的唯一合同是：

```text
factor[t] ↔ target_ret[t]
有效评分窗口 = [0, T-2)
```

训练 IC、walk-forward、PnL、手续费、换手惩罚、暴露度、因子相关性池、
特征评估、Web 回测和仓库内独立审计脚本均使用该窗口。新 checkpoint 和
策略写入
`scoring_contract_version=open_t1_t2_same_index_tail2_v1`；旧合同
checkpoint 拒绝续训，旧策略分数不能作为新训练下限或参与跨合同最高分
选择。

## 二、固定版本与输入身份

| 项目 | 固定版本 |
|---|---|
| AlphaMaster | 实验运行时 `41ae1c66fb7f041fc939b9f451a0d92d03fe5559` |
| AKQuant | PyPI `0.3.20`；源码测试快照 `30054523fb905adb1c3f250749e1b5ff61cf8452` |
| RQAlpha | `3503ab57932540cd36bf8375134e52c6923bf0d2`；运行版本 `6.3.0` |
| Qlib | Windows 发布版 `0.9.7`；最新源码快照 `79633dd9506ea689e5400dea0197717b5b3d74b7` |
| vn.py | `1b78494979deb4c4996f6b864f234d9839f2f239`；源码版本 `4.4.0` |

成交费用样本 SHA-256：
`2c85be84c86a4dbe82bd15843748d9a7669b9811dc3b148f876f0fff6918d90d`。

研究层样本 SHA-256：
`ad68d63868786d1973de9d50e72842552a8f7640aac93643de602cc0825db61f`。

第三方源码、虚拟环境和运行输出都位于 `scratch/`，不进入生产依赖，也
不提交第三方源码。

正式通过轮次为 `20260724T151436Z`，总入口合同为
`a-share-diff-suite-v3`。总入口要求结果目录原先不存在，
顺序执行 6 个样本生成器/源码检查器和 2 个比较器，并记录每条命令、输出
文件、标准输出/错误日志、Python 版本、完整已安装包清单原文及其
SHA-256。13 个实际运行脚本/fixture 也逐文件绑定，合并源码合同
SHA-256 为
`1ef52ba8a207d56a97272fc8acaf3f8748c768ec1a443ef67c1a8021dec59201`。
当前没有保存 Python 安装包本身的哈希，因此只能称“本机环境身份可
核对”，不能称跨机器字节级供应链复现。

## 三、成交与费用差分

固定样本是一笔 100 股、10 元买入并以同价卖出：

- 初始现金：100,000 元
- 佣金：0.03%，最低 5 元
- 卖出印花税：0.05%
- 过户费：0.001%
- 滑点：0

结果：

| 引擎 | 买入费用 | 卖出费用 | 期末现金 | 解释 |
|---|---:|---:|---:|---|
| 手算 | 5.01 | 5.51 | 99,989.48 | 权威固定口径 |
| AlphaMaster | 5.01 | 5.51 | 99,989.48 | 与手算完全一致 |
| AKQuant | 5.01 | 5.51 | 99,989.48 | 与 AlphaMaster 完全一致 |
| RQAlpha 默认费用器 | 5.00 | 5.50 | 99,989.50 | 默认股票费用实现对齐佣金倍率后仍没有计算过户费，现金差 0.02 元 |

这说明 RQAlpha 默认费用器不能直接作为 AlphaMaster 的 A 股费用真值。
如果未来复用 RQAlpha，必须把过户费和按日期生效的费用政策显式放在
AlphaMaster 侧，不能靠调整佣金倍率把总结果“调成一样”。

AKQuant 的正式样本使用同周期收盘成交，只为对齐两次终态成交并比较
费用。它不证明 AlphaMaster 的下一开盘时钟，也不能替代 T+1 和跨日订单
生命周期回放。

这个样本只有 100 股、10 元、最低佣金、零滑点且费用落在整分，不能外推
到大额佣金、非整分舍入、滑点或按日期变化的费用政策。

## 四、信号、标签、成交时钟

AlphaMaster 真实目标收益构造方法在 8 个不等距开盘价上得到 6 个真实
非零/非等值收益，并在末尾补两个零：

```text
target_ret =
[0.1670540, -0.0800427, 0.2231435, -0.0689929,
 0.2513144, -0.0571584, 0, 0]
```

这直接验证的只是目标收益构造合同：

```text
target_ret[t] = log(open[t+2] / open[t+1])
```

`position[t] × target_ret[t]` 来自生产源码检查，`t+1` 开盘成交来自另一
组独立执行样本；本轮没有运行一条把信号、成交、持仓和 PnL 串起来的完整
生产回测，因此不把三段证据合并冒充端到端验证。

Qlib `TopkDropoutStrategy.generate_trade_decision` 的发布版和最新源码快照
都使用 `get_step_time(trade_step, shift=1)` 读取预测；交易日历的真实实现
使用 `当前索引 - shift`，因此正数确实表示更早一根。两个版本相关方法的
语法树哈希一致。该结论只覆盖这两个方法，不覆盖热身边界、多频率、时区
和预测实际可用时间，也不是完整回测运行证据。

AlphaMaster 的 IC 反例结果：

- 同钟诊断序列的正确 IC：`1.0`
- 只额外偏移、但排除补零后的 IC：约 `-0.968580`
- 额外偏移并纳入补零后的 IC：约 `-0.775391`
- 当前 `AlphaEngine._compute_ic` 输出：约 `-0.775391`

这同时证明额外右移和补零污染。后续生产修复回归又覆盖了 PnL/成本尾部
不变性、walk-forward 不能触达补零区、模块级评估器默认裁两根，以及 Web
回测只返回真实可实现收益窗口。原实验数字仍是旧语义反例，不改写为新语义
结果。

## 五、历史时点股票池

vn.py 固定提交中的原始 `AlphaDataset.prepare_data` 已用真实 Polars
数据执行。两只股票分别设置生效区间：

```text
000001.SSE: 2024-01-02 至 2024-01-03
000002.SSE: 2024-01-03 至 2024-01-04
```

正常样本输出恰好 4 行，两个区间的起止日期都被包含，区间外数据全部
排除。但三个对抗样本同时复现：

- 同一股票的区间重叠时，重叠行出现两次。
- `filters={}` 时完全绕过筛选，所有行都会泄漏进结果。
- 所有区间都为空时，抛出
  `ValueError("cannot concat empty list")`。

因此只能借鉴“生效区间”数据结构，不能直接复制方法。生产实现必须先合并
区间、强制 `(交易日, 股票)` 唯一、空输入失败关闭，并把带时分秒的行情
规范化成交易日后再匹配。

当前 AlphaMaster 仍只有 2026 年冻结 A50 成分；本实验没有把 vn.py 代码
接进生产，也没有取得真实历史 A50 成分数据，因此幸存者偏差问题尚未
解决。

## 六、实际踩坑

1. Qlib 最新源码在本机需要 Microsoft C++ 14 以上编译工具；缺失时无法
   构建 Cython 扩展。
2. Qlib 0.9.7 有 Windows 预编译包，但直接配最新 NumPy 2.x 和 CVXPY
   依赖时出现原生访问冲突。降级 NumPy 后又与最新版 CVXPY 的依赖要求
   冲突。
3. 因此 Qlib 本轮只能给出“发布包源码 + 最新快照源码合同”证据，不能写成
   运行时 E3。
4. vn.py alpha 的数据模块在包导入时会连带 Alphalens、模型、策略和实验室
   模块。正式实验只为未调用的 Alphalens 报告函数提供导入占位，历史成分
   筛选方法保持原样执行。
5. RQAlpha 默认股票费用口径和 AlphaMaster 不同；框架“支持 A 股”不等于
   默认费用合同就是本项目所需合同。
6. 第一次正式总入口运行在环境清单阶段失败，因为项目 `.venv` 没有
   `pip`；现已改用 Python 标准库读取安装包清单。第二次运行因误用了
   AKQuant 新版示例中的 `result.positions` 失败；固定版 0.3.20 的真实
   接口是 `get_positions_dict()`。两次失败均保留失败清单，没有改写成
   通过。
7. `a-share-diff-suite-v2` 首次运行的外层等待在 5 秒时超时；当时读取的
   快照是 `RUNNING` 和 4/8 步，但原进程随后继续并自动写成 `PASSED`
   8/8，没有启动第二个进程复用该目录。后续 v2 在另一个全新目录完成
   8/8，再由 v3 增加脚本/fixture 字节绑定并在新目录完成最终 8/8。

## 七、验证

- 正式总入口：8 个步骤通过、0 个失败，其中包含 2 个比较合同。
- 正式执行比较 SHA-256：
  `a47b4782c4b14539663bd33dc4cba7a28cde2ce231d3c7664cfb582f1e196d96`。
- 正式研究比较 SHA-256：
  `ecfeb4adfebb483215c8e603a6ebed396812b316ecdb826e1c96f48ad24ffb0b`。
- 正式运行清单 SHA-256：
  `d63503770b4e0ea52de1140c2b7615f1eba8677ef6c7fff815593dbe4437cf3a`。
- 本轮直接相关回归：54 个通过，0 个失败，其中 AlphaMaster 27 个、
  AKQuant 10 个、RQAlpha 17 个。标准 JUnit 结果保存在
  `scratch/third_party_diff_20260724/validation/20260724T152500Z/`：
  - AlphaMaster：
    `39868886185005d6cf95a3ffc24b12960f36c6828ba3a91488530fdb1e932ce3`
  - AKQuant：
    `d6386fd27aaa4340778266f96c76593f089c0a22c7da1cdd49515810d19a67b8`
  - RQAlpha：
    `44bdd0c16b9b2c08c99e09d60a04ccaf553cd004c9bf6608da1cc546677394fd`
- 扩大到一个既有 MT5 fetcher 属性测试时：26 个通过，1 个失败。失败路径
  是模拟 MT5 已返回数据但另一层仍按“MT5 不可用”清空结果，与本轮新增
  实验代码无关；问题没有被隐藏或改成通过。
- `experiments/` 通过 Ruff、Python 编译和 `git diff --check`；Ruff JSON
  SHA-256 为
  `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`。

## 八、放行与当前续接顺序

本轮只放行以下“思想和合同”，不放行整套框架替换：

1. 借鉴 Qlib/vn.py 的生效区间数据结构，但自行实现合并、唯一性、空输入
   失败关闭和交易日规范化，不能直接复制当前 vn.py 方法。
2. 保留 Qlib 的“上一时点预测、当前时点交易”显式时钟写法。
3. 用 AKQuant/RQAlpha 继续做逐订单黄金差分，权威费用仍由手算与
   AlphaMaster 合同决定。

当前优先级：

1. `OPS-01` 先对账已完成的 Slurm `570548`，核验结果包、下载、后处理和
   本地队列状态；在对账完成和用户批准前，不默认启动新训练或覆盖冻结运行时。
2. `RND-01` 与运营恢复线并行：先冻结已有 255 个沪深300权重时点的
   PIT 可知/生效合同、949 代码覆盖/隔离矩阵和 2009-12-31 的 298 只异常，
   再用历史退出现任池的小样本验证抽取器并扩展历史覆盖。
3. 身份链真实 run 审计、测试证据复现，以及历史时点 ST、板块、涨跌停、
   停牌和成交量状态调查同时推进，不等待 50/50 个单股信号。
4. 接通动态股票池组合 replay 后，复用现有执行器和账本补齐生产化，再做
   跨日挂单、公司行动、按日期生效的费用政策回放和 sealed OOS。

正式实验入口：

- [`experiments/a_share_execution_diff`](../experiments/a_share_execution_diff/README.md)
- [`experiments/a_share_research_layer_diff`](../experiments/a_share_research_layer_diff/README.md)
- [`experiments/run_a_share_diff_suite.py`](../experiments/run_a_share_diff_suite.py)
