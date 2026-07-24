# 大 A 量化模型训练与回测生态调研

> 日期：2026-07-24
> 状态：广搜、深挖与第一轮隔离实验完成；小红书、抖音完整正文与评论区仍待登录补证，完整框架 E3/E4 仍未执行
> 服务对象：AlphaMaster 个人研究系统
> 原则：先广泛搜集，再按源码、测试、真实故障和 AlphaMaster 适配性深入筛选

## 一、结论

这轮没有找到一个适合整体替换 AlphaMaster 的框架。真正值得吸收的是分层能力：

1. **保留 AlphaMaster 的核心执行与审计链。**
   当前 `portfolio_manager/execution.py` 的源码和单元测试已有 T+1、显式停牌/涨跌停状态、整手、合法碎股清仓、最低佣金、印花税、过户费、滑点和确定性执行身份，但尚未接入现役 Web 长驻链路。第一轮固定样本只把 AKQuant/RQAlpha 的窄范围费用差分和 vn.py 历史成分方法推进到 E3；尚未证明任何第三方框架完整覆盖这组能力，已确认的模块级缺口见第三节。
2. **优先复用成熟的非核心轮子。**
   - QuantStats：只做收益序列报告和图表，不能覆盖 AlphaMaster 的权威指标。
   - Alphalens 活跃分支：只做因子 IC、分层收益、换手和衰减诊断。
   - skfolio：未来用于组合优化的隔离基准，输出权重后仍交给 AlphaMaster 执行。
3. **把 AKQuant 和 RQAlpha 当作独立对照器。**
   它们最适合贡献 A 股订单、账户、公司行动和费用的黄金测试，不适合成为第二套生产执行内核。
4. **Qlib 和 vn.py alpha 只吸收研究层。**
   重点是历史成分区间、模型统一接口、信号—成交时钟、实验记录和因子处理器；不复用其 A 股撮合器。
5. **TradingAgents、RD-Agent、AlphaPilot、wq-alpha-research 只进入研究工作流。**
   它们可以提出假设、生成候选、记录失败和组织评审，但不得读取最终封存集，也不得直接生成目标仓位或订单。
6. **QMT、聚宽、BigQuant、SuperMind 只提供工程参考或隔离执行通道。**
   QMT 有可追溯的社区故障案例；PTrade 本轮证据不足，暂不作同等判断。商业平台的回测结果不能成为 AlphaMaster 的真值；版本升级前必须跑固定订单回归集。

最优先的问题不是再增加模型，而是：

```text
历史时点股票池
→ 时点有效的 ST / 板块 / 涨跌停 / 停牌状态
→ t 日信号、t+1 成交的统一时钟
→ 选择集与最终封存集彻底分离
→ 训练代理回测与 A 股真实执行回放分开命名
```

## 二、调研方法与覆盖

### 2.1 研究地图

从三种角色检查同一问题：

- **开发者**：代码质量、接口边界、测试、依赖、恢复、运行身份和维护状态。
- **使用者**：安装、数据稳定性、回测与模拟/实盘差异、版本升级、错误可见性和支持质量。
- **评估者**：未来泄漏、幸存者偏差、反复调参、费用、可交易性、统计口径和可复现性。

候选只有在以下证据逐步升级后才进入复用清单：

- E0：README、宣传或单个截图。
- E1：多个独立用户给出具体版本、环境或失败场景。
- E2：源码、测试、维护者确认或修复提交。
- E3：固定版本、固定数据的独立复现。
- E4：在 AlphaMaster 隔离环境中通过对抗性差分测试。

广搜阶段把核心候选推进到 E2；后续第一轮隔离实验只在固定样本的窄范围内把 AKQuant/RQAlpha/vn.py 推进到 E3。Qlib 仍停在源码合同 E2，完整框架差分仍是下一阶段；不把源码阅读或模块实验说成已经集成验证。

### 2.2 来源覆盖

| 来源 | 覆盖 | 证据边界 |
|---|---|---|
| GitHub 仓库、源码、测试、Issue、Discussion | 深入 | 本轮最强证据 |
| QMT 社区、VeighNa 社区 | 深入到具体用户帖子 | QMT 有可追溯案例；VeighNa 用于开发者/使用者交叉检查 |
| 聚宽社区 | 读取少量平台社区案例 | 不是独立于平台的外部评估 |
| BigQuant | 以官方文档和平台托管 Wiki 为主 | 属 E0/E2 的产品/接口证据，不是独立用户共识 |
| Reddit | 搜索通用回测与 vectorbt/Qlib | 未形成高于 E0/E1 的大 A 专用保留证据 |
| B站 | 读取公开视频页面和作者信息 | 未登录时完整评论线程不稳定；未形成高于 E0 的保留证据 |
| 微信公开文章搜索 | 广搜并筛选技术文章 | 独立、可复现的用户证据很少；未把营销文章作为结论依据 |
| 小红书 | 已到搜索页 | 登录弹窗阻断，未读取正文和评论 |
| 抖音 | 搜索到公开视频线索 | 无合法登录态，未读取完整评论 |

本轮的工作判断是：后续普通泛搜的新增结果主要重复四类问题——数据接口变化、版本升级破坏、研究/模拟/实盘语义分叉、执行近似失真。这个判断没有用查询数量和重复率做严格统计，不作为“已经搜尽”的证明。剩余不确定性优先靠登录后的评论取证和 AlphaMaster 本地 E3/E4 实验解决。

## 三、优先候选

### 3.1 A 股执行与回测

| 项目 | 证据 | 当前决策 | 值得吸收 | 关键反证 |
|---|---|---|---|---|
| [AKQuant](https://github.com/akfamily/akquant) | E3（仅固定一手费用终态） | 高优先级隔离复验 | T+1、订单状态、黄金基线和恢复测试 | 项目很新；[#329](https://github.com/akfamily/akquant/issues/329)、[#334](https://github.com/akfamily/akquant/issues/334) 是同一报告者发现、现已关闭并有回归测试的历史缺陷；同周期终态样本不证明完整跨日生命周期 |
| [RQAlpha](https://github.com/ricequant/rqalpha) | E3（仅默认费用模块） | A 股语义参考与对照器 | 可卖持仓、碎股清仓、公司行动、退市、时点印花税、分笔最低佣金 | 高质量历史数据依赖 RQData；默认股票费用决策器在佣金倍率对齐后仍未计 AlphaMaster 已有的过户费；未运行完整回测；自定义非商业许可 |
| [Qlib](https://github.com/microsoft/qlib) | E2 | 只吸收研究层 | 历史成分区间、数据表达式、模型、Recorder、Top-K Dropout、前一周期信号 | 本轮固定快照的账户模型未找到普通 A 股 T+1；费用和涨跌停是通用近似；指定价格缺失存在回退路径 |
| [vn.py alpha](https://github.com/vnpy/vnpy/tree/master/vnpy/alpha) | E3（仅历史成分方法） | 只借数据结构，不复制当前筛选方法 | Polars 因子处理器、Alpha101/158、统一模型接口、按日成分列表 | 正常区间通过，但重叠区间会重复行、空字典会绕过过滤、全空区间会报错；alpha 回测器仍存在用合成旧价 bar 撮合的路径，且没有独立现金校验；不能泛化为整个 vn.py |
| [vectorbt](https://github.com/polakowo/vectorbt) | E1/E2 | 只做快速筛选 | 大规模参数与信号矩阵筛选；其 Portfolio 使用 Numba 顺序处理，并非完全不能表达路径依赖 | 缺 A 股订单、账户和市场规则真值；完整订单管理和大矩阵内存仍是边界 |
| [LEAN](https://github.com/QuantConnect/Lean) | E2 | 架构参考 | 完整事件、订单、经纪商和回归体系 | C#/Python 大型运行时，A 股适配成本远大于收益 |
| [Backtrader](https://github.com/mementum/backtrader) | E1/E2 | 低优先参考 | 通用事件式策略接口 | GPL、维护节奏较慢、A 股语义需重做 |
| [Zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) | E1/E2 | 低优先参考 | Pipeline/事件式研究 | 主要面向美股日历和资产模型 |
| [PyBroker](https://github.com/edtechre/pybroker) | E2 | 延后 | walk-forward 与缓存思想 | Apache 2.0 + Commons Clause；仍不是 A 股执行真值 |
| [backtesting.py](https://github.com/kernc/backtesting.py) | E2 | 不进入主线 | 简单策略原型 | AGPL；组合和 A 股执行能力不足 |
| [bt](https://github.com/pmorissette/bt) | E1/E2 | 不进入执行层 | 组合权重实验 | 不提供真实订单、现金和 A 股规则 |

核心源码快照（读取于 2026-07-24）：

- AKQuant [`30054523fb905adb1c3f250749e1b5ff61cf8452`](https://github.com/akfamily/akquant/tree/30054523fb905adb1c3f250749e1b5ff61cf8452)
- RQAlpha [`3503ab57932540cd36bf8375134e52c6923bf0d2`](https://github.com/ricequant/rqalpha/tree/3503ab57932540cd36bf8375134e52c6923bf0d2)
- Qlib [`79633dd9506ea689e5400dea0197717b5b3d74b7`](https://github.com/microsoft/qlib/tree/79633dd9506ea689e5400dea0197717b5b3d74b7)
- vn.py [`1b78494979deb4c4996f6b864f234d9839f2f239`](https://github.com/vnpy/vnpy/tree/1b78494979deb4c4996f6b864f234d9839f2f239)

关键源码复核入口：

- AKQuant：[股票市场规则与状态](https://github.com/akfamily/akquant/blob/30054523fb905adb1c3f250749e1b5ff61cf8452/src/market/stock.rs)、[A 股教材与涨跌停边界](https://github.com/akfamily/akquant/blob/30054523fb905adb1c3f250749e1b5ff61cf8452/docs/zh/textbook/06_stock_a.md)。
- RQAlpha：[持仓与可卖数量](https://github.com/ricequant/rqalpha/blob/3503ab57932540cd36bf8375134e52c6923bf0d2/rqalpha/mod/rqalpha_mod_sys_accounts/position_model.py)、[股票下单数量](https://github.com/ricequant/rqalpha/blob/3503ab57932540cd36bf8375134e52c6923bf0d2/rqalpha/mod/rqalpha_mod_sys_accounts/api/api_stock.py)、[默认费用决策器](https://github.com/ricequant/rqalpha/blob/3503ab57932540cd36bf8375134e52c6923bf0d2/rqalpha/mod/rqalpha_mod_sys_transaction_cost/deciders.py)。
- Qlib：[交易所价格与限制路径](https://github.com/microsoft/qlib/blob/79633dd9506ea689e5400dea0197717b5b3d74b7/qlib/backtest/exchange.py)、[持仓账户](https://github.com/microsoft/qlib/blob/79633dd9506ea689e5400dea0197717b5b3d74b7/qlib/backtest/position.py)、[信号时钟](https://github.com/microsoft/qlib/blob/79633dd9506ea689e5400dea0197717b5b3d74b7/qlib/contrib/strategy/signal_strategy.py)、[历史指数成分](https://github.com/microsoft/qlib/blob/79633dd9506ea689e5400dea0197717b5b3d74b7/scripts/data_collector/index.py)。
- vn.py alpha：[历史成分数据集](https://github.com/vnpy/vnpy/blob/1b78494979deb4c4996f6b864f234d9839f2f239/vnpy/alpha/dataset/template.py)、[模型接口](https://github.com/vnpy/vnpy/blob/1b78494979deb4c4996f6b864f234d9839f2f239/vnpy/alpha/model/models/lgb_model.py)、[alpha 回测器](https://github.com/vnpy/vnpy/blob/1b78494979deb4c4996f6b864f234d9839f2f239/vnpy/alpha/strategy/backtesting.py)。

### 3.2 报告、因子诊断与组合

| 项目 | 当前决策 | 使用边界 |
|---|---|---|
| [QuantStats](https://github.com/ranaroussi/quantstats) | E2 候选，可先做小型报告适配 | 只消费已验证收益序列；只展示 CAGR、波动率、最大回撤、Sharpe 等已通过手算差分的字段；任何字段不一致就隐藏该字段；“胜率”必须标为收益周期胜率 |
| [Empyrical-reloaded](https://github.com/stefan-jansen/empyrical-reloaded) | 借公式和测试样例 | 不必增加第二个权威指标运行时 |
| [pyfolio-reloaded](https://github.com/stefan-jansen/pyfolio-reloaded) | 延后 | 可借 tear sheet 结构，不替代订单审计 |
| [Alphalens-reloaded](https://github.com/stefan-jansen/alphalens-reloaded) | E2 因子诊断候选 | AlphaMaster 必须生成并审计价格矩阵与时间对齐，或提供预计算收益适配器；上线前注入错一日、缺失交易日、停牌和未来价格测试 |
| [skfolio](https://github.com/skfolio/skfolio) | E2 组合优化首选候选 | 只在选择集做 walk-forward，与等权/逆波动基线比较并记录全部调参次数；禁止随机 KFold；连续权重必须交给 AlphaMaster 执行 |
| [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) | 延后 | 风险度量丰富，但求解器和配置复杂度高 |
| [PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt) | 借收缩协方差和 HRP | `DiscreteAllocation` 不是 A 股执行器 |
| [cvxportfolio](https://github.com/cvxgrp/cvxportfolio) | 低优先 | GPL，且集成复杂度较高 |

近期候选读取快照（2026-07-24）：QuantStats
[`fbd10daed0227aa0d10da6513f1b15e7e98d7fae`](https://github.com/ranaroussi/quantstats/tree/fbd10daed0227aa0d10da6513f1b15e7e98d7fae)、
Alphalens-reloaded
[`f0a07c22d554e4b4036983cc80320b432714fe7e`](https://github.com/stefan-jansen/alphalens-reloaded/tree/f0a07c22d554e4b4036983cc80320b432714fe7e)、
skfolio
[`109ed13fee0125ff9b001b8be33643b17e791578`](https://github.com/skfolio/skfolio/tree/109ed13fee0125ff9b001b8be33643b17e791578)。

### 3.3 训练、实验和 Slurm

| 项目 | 当前决策 | 理由 |
|---|---|---|
| [MLflow](https://github.com/mlflow/mlflow) | 借接口，是否引依赖后定 | AlphaMaster 已有更强的数据/源码/产物哈希；先补统一查询面，不建立第二套运行身份 |
| [Optuna](https://github.com/optuna/optuna) | 延后到评估合同稳定后 | 自动搜索会放大当前验证集污染和多重检验 |
| [DVC](https://github.com/iterative/dvc) | 当前不引入 | 现有数据清单和哈希已工作；没有证明需要第二套数据版本控制 |
| [Submitit](https://github.com/facebookincubator/submitit) + [Hydra](https://github.com/facebookresearch/hydra) | 不改现役 Slurm 主链 | 不能替代 run/job/源码/数据/产物身份，会增加 checkpoint 和日志语义 |
| [Parsl](https://github.com/Parsl/parsl) | 不进入当前主线 | 当前没有需要第二套工作流运行时的证据 |

### 3.4 Agent、AI 与自动因子研究

| 项目 | 当前决策 | 可吸收内容 | 禁止用途 |
|---|---|---|---|
| [RD-Agent](https://github.com/microsoft/RD-Agent) | E2，高研究价值，延后集成 | 提案—实现—评估—记忆循环 | 访问最终封存集、无限搜索、直接发布策略；准入前必须通过随机标签、封存集访问拦截、缓存污染和同名实验覆盖测试 |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | 只借工作流 | 多角色反方、checkpoint、决策日志 | 当作历史回测器或收益证据 |
| [TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN) | 只借非 UI 核心思想 | 中文数据接口和多角色工程 | `app/`、`frontend/` 有独立专有许可 |
| [AlphaPilot](https://github.com/ai-yang/AlphaPilot) | 按组件审查 | 因子资产、任务门户、实盘模式隔离、审计账本 | 其核心回测仍依赖 Qlib；大量功能不等于 A 股真实性 |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | 研究基准 | 强化学习训练流程 | 直接把奖励回测当样本外收益 |
| [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster) | 研究基准 | 多任务 RL 实验结构 | 生产 A 股执行 |
| [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) | 文本研究参考 | 金融 Agent 组织 | 目标仓位或订单 |
| [AlphaGen](https://github.com/ICT-FinD-Lab/alphagen) | 论文/方法参考 | 符号因子生成 | 无许可证代码并入公开仓库 |
| [AlphaForge](https://github.com/DulyHao/AlphaForge) | 论文/方法参考 | 公式生成与评估 | 无许可证代码并入公开仓库；固定年份切分不能直接复用 |

RD-Agent 固定于
[`4f9ecb005881cddc08df0124a2e894c018007679`](https://github.com/microsoft/RD-Agent/tree/4f9ecb005881cddc08df0124a2e894c018007679)，读取日期 2026-07-24。

TradingAgents 的真实反证：

- [#1007](https://github.com/TauricResearch/TradingAgents/issues/1007)：历史回测曾混入 2025—2026 新闻，后来增加日期过滤和回归测试。
- [#805](https://github.com/TauricResearch/TradingAgents/issues/805)：即使数据过滤正确，2026 年训练的语言模型仍可能知道 2023 年之后发生的事实。
- [#969](https://github.com/TauricResearch/TradingAgents/issues/969)：社区仍在追问完整历史回测能力；当前架构不能当周期组合回测器。

## 四、用户点名项目

| 项目 | 结论 |
|---|---|
| [wq-alpha-research](https://github.com/QuantML-Research/wq-alpha-research) | E2；借“失败经验账本、相关性去重、换手约束、人工审核后应用”；它面向 WorldQuant BRAIN 的美国市场，不是 A 股代码；README 声明 CC BY-NC 4.0，复制前仍核对完整条款 |
| [StockTradebyZ](https://github.com/SebastienZh/StockTradebyZ) | E2；借“Tushare 初筛 → 图像复评 → 结构化建议”作为人机研究界面；没有可信回测主链；README 声明 CC BY-NC 4.0，复制前仍核对完整条款 |
| [quantitative_analysis](https://github.com/henrylin99/quantitative_analysis) | E2；可借 Parquet+SQLite、本地 Web、因子表达式白名单和验收文档；[核心回测](https://github.com/henrylin99/quantitative_analysis/blob/1ce6fa128d52a5d0127b2d3e2b233736ee8edf9c/app/services/backtest_engine.py)会逐日捕获异常后继续，且未找到 T+1/涨跌停/历史股票池的完整覆盖，不复用主引擎；README 声明 MIT |
| [tdx_quant](https://github.com/henrylin99/tdx_quant) | E2；用户点名项目中最值得深挖。可借 [pytdx 多服务器与稀疏分页](https://github.com/henrylin99/tdx_quant/blob/b95d8e915aa2fa4b703e64c38ca48eb51a6fa96e/scripts/data_pipeline/connectors/pytdx_client.py)、[证券列表](https://github.com/henrylin99/tdx_quant/blob/b95d8e915aa2fa4b703e64c38ca48eb51a6fa96e/scripts/data_pipeline/jobs/security_list_job.py)、除权除息、真实网络集成测试和空结果失败关闭；准入前必须验证分页完整性、除权、时区、股/手单位、停牌与财务公告时点；未检测到明确许可证 |
| [cn_stocks](https://github.com/henrylin99/cn_stocks) | E2；当前排除。技术指标投票、MySQL 和 BaoStock 与现有架构重复；模型、回测和风险管理仍在路线图；README 声明 MIT |

上述项目的读取快照依次为：`86d7531fcd6ddb50a87765ed75981f88c1d5aa29`、`df3c70199828a12389ee4759d0c498f7df81de30`、`1ce6fa128d52a5d0127b2d3e2b233736ee8edf9c`、`b95d8e915aa2fa4b703e64c38ca48eb51a6fa96e`、`9210b55752e2d5c98a93eeaa30ef0ab7f62debb9`，读取日期均为 2026-07-24。

## 五、这轮挖出的个人项目

| 项目 | 结论 |
|---|---|
| [tickflow-stock-panel](https://github.com/shy3130/tickflow-stock-panel) | E2；值得借动态 ST/板块涨跌停、历史日期校验、回测矩阵和 walk-forward 测试；但截至 2026-07-24 项目才建立约一月、变化极快、数据强依赖 TickFlow，默认仍允许 `close_t` 成交，不能直接用作真值 |
| [free-stockdb](https://github.com/hello245m/free-stockdb) | E0；截至 2026-07-24 未观察到仓库自动测试；核心 C++/数据源与宣传性能缺乏可审计证据，先作为线索 |
| [OSkhQuant](https://github.com/khscience/OSkhQuant) | E0/E1；截至 2026-07-24 未观察到自动测试；可参考 MiniQMT 本地化产品形态，但回测与数据真实性证据不足 |
| [a-share-quant-sim](https://github.com/fkchaos/a-share-quant-sim) | E2；“回测和模拟共用逻辑”方向正确；但大量测试只验证合成 DataFrame 或手写 JSON 结构，仓库混入大量历史结果，不能把测试数量当可信度 |
| [BreadFree-Simu](https://github.com/FeiCoder/BreadFree-Simu) | E0/E1；事件循环和策略/执行分离可参考；提交少、无自动测试，Agent 策略和回测能力主要停留在 README |

其他广搜项目包括 `efinance`、`mootdx`、已归档 `pytdx`、`KHunter`、`AlphaGPT_Tushare`、`backquant`、`EasyQuant`、`Smart-Stock`、`VCStockRank`、`quant-buddy-skills`、`a-share-mcp`、`ashare-ai`、`dualalpha-lite` 和 `Quantia`。它们没有出现足以超过上述候选的源码、测试或独立用户证据，保留为线索，不进入近期集成队列。

## 六、商业平台可借鉴的工程框架

### 6.1 BigQuant

[BigTrader 官方说明](https://bigquant.com/wiki/doc/I2Wvg4wJWd)公开了几项值得借鉴的设计：

- 标准事件回测和快速向量化回测明确分开，快速模式承认准确性较低。
- 当前 K 线产生的订单在下一根 K 线撮合。
- 快照行情下记录排队量、阶段成交量和部分成交。
- 回测、模拟和实盘尽量共用代码，但持久状态需单独处理。

社区和官方文档同时表明“同代码”并不等于“同状态”：

- 模拟交易每天重新触发初始化，状态必须写入 `user_store`。
- [实盘数据同步说明](https://bigquant.com/wiki/doc/fIXuWtlfE0)承认真实账户因断网或失败订单会与模拟账户逐日漂移。
- [回测与模拟差异案例](https://bigquant.com/wiki/doc/4zfEvJRjzY)显示日期绑定和运行环境会改变数据区间。

对 AlphaMaster 的启发：保留“同一订单意图、不同执行适配器”，但账户、成交和持仓只能以账本与真实回报为准。

### 6.2 聚宽

[聚宽 API 文档](https://cdn.joinquant.com/help/img/JoinQuantAPI.pdf)明确区分研究、回测和模拟，并提示模拟重启、真实价格选项会导致同日期程序结果不同。社区帖子还暴露：

- [常见策略骗人把戏](https://www.joinquant.com/community/post/detailMobile?postId=56381)：错误 ST 过滤、虚假撮合、不可成交套利和费用口径。
- [回测正常、模拟重启后状态为空](https://www.joinquant.com/community/post/detailMobile?postId=61684)：依赖历史状态的策略在不同起点或重启后结果变化。

可借的是环境分层、社区策略可复现入口和模拟前向观察；不能复制的是把平台回测曲线当真实可成交证据。

### 6.3 SuperMind

[官方常见问题](https://quant.10jqka.com.cn/view/help/16)和[平台首页](https://quant.10jqka.com.cn/view/)强调数据、回测、仿真和本地化实盘的一体化。公开社区可复现的独立负面证据较少，搜索结果中教程、培训和开户链接较多。

可借的是“研究 IDE + 仿真 + 风控 + 券商本地部署”的产品分层；技术正确性仍需真实订单和状态回放，不能从宣传反推。

### 6.4 QMT

QMT 社区的具体故障比宣传更有价值：

- [同代码同数据，隔数日结果变化](https://www.xuntou.net/forum.php?mobile=no&mod=viewthread&tid=6408)：该帖用户报告同一条件下结果发生变化，不能外推为所有 QMT 版本。
- [客户端升级后下单失效](https://www.xuntou.net/forum.php?mod=viewthread&tid=1022)：该帖用户报告升级后的下单问题。
- [实盘接口不能直接在回测使用](https://www.xuntou.net/forum.php?mobile=no&mod=viewthread&tid=5914)：该帖讨论回测与实盘接口边界。
- [缺行情被默认解释为停牌](https://www.xuntou.net/forum.php?mobile=2&mod=viewthread&tid=1931)：该帖用户把本地行情缺失观察为停牌语义。

因此，QMT 只能作为 Windows 隔离执行端：

```text
AlphaMaster 冻结决策
→ 薄适配器翻译为券商订单
→ 订单/成交/资金/持仓回报
→ 持久账本与启动对账
```

训练、特征、主回测和权威指标不能依赖券商客户端。

### 6.5 PTrade

本轮没有获得与 QMT 同等强度、可复现的 PTrade 用户故障证据。由于它同样是券商定制、版本分散的闭源环境，暂时把“薄适配器 + 启动对账”作为保守工程假设，而不是已经证明的产品缺陷。只有取得目标券商当前版本的官方接口材料和真实社区复现后，才决定是否接入。

## 七、真实用户踩坑形成的硬规则

### 7.1 数据

- [Qlib #1875](https://github.com/microsoft/qlib/issues/1875)：历史上网页采集接口 404 可让数据脚本长期卡住，维护者说明 [#1758](https://github.com/microsoft/qlib/pull/1758) 已修复相关 Phoenix 列表问题。当前价值是保留超时、失败关闭和下载/训练解耦测试，不把它说成现版本仍故障。
- [Qlib #1268](https://github.com/microsoft/qlib/issues/1268)：复权因子与行情软件口径不同。价格、复权因子和原始价必须共同冻结。
- [AKShare #6986](https://github.com/akfamily/akshare/issues/6986)、[#6100](https://github.com/akfamily/akshare/issues/6100)、[#5857](https://github.com/akfamily/akshare/issues/5857)：这些是特定时间的上游断连、返回异常或部分股票缺失案例，部分已关闭；“IP 被限流”多为用户推断，不能当作已确认原因。当前规则仍是训练期间禁止动态抓取。
- “缺数据”和“真实停牌”必须是两个不同状态，不能自动互相转换。

### 7.2 实验

- [Qlib #1930](https://github.com/microsoft/qlib/issues/1930)：复用实验名可能加载同一模型，导致不同模型出现相同回测。
- 每次实验必须绑定代码提交、数据哈希、股票池版本、特征/标签合同、全部随机种子、模型哈希、预测哈希和回测输入。
- 必须记录所有尝试，包括失败和被拒绝的公式；只保存冠军会隐藏真实搜索预算。

### 7.3 报告

- [QuantStats #334](https://github.com/ranaroussi/quantstats/issues/334)：短序列 CAGR 历史上出现除零，现已关闭。
- [QuantStats #463](https://github.com/ranaroussi/quantstats/issues/463)：0.0.71—0.0.73 曾出现 Series 布尔判断回归，0.0.74 已修复。
- 报告库必须锁版本、保存输入收益序列，并通过空序列、单点、重复日期、缺失日、跨年、时区、对数/算术收益测试。

### 7.4 向量化回测

- [vectorbt Discussion #185](https://github.com/polakowo/vectorbt/discussions/185)：没有完整订单管理，订单立即成交或拒绝。
- [Discussion #209](https://github.com/polakowo/vectorbt/discussions/209)：复杂路径依赖和大矩阵会提高建模与内存成本。vectorbt 的 Portfolio 本身使用 Numba 顺序处理，不能把它简化成“纯向量化完全无法处理路径依赖”。
- 最安全的流程是“向量化粗筛 → AlphaMaster 事件式 A 股执行回放”。

## 八、对 AlphaMaster 的对抗性审查

### 8.1 源码和测试已具备、但尚未全部部署的强项

- 训练只在 Slurm Worker 运行，源码、运行提交、数据、结果和发布包有身份链。
- 最终封存评估已有全局揭盲锁，成本必须大于 0。
- 组合决策源码和测试已绑定现金、持仓和冻结行情，篡改会在下单前失败；现役生产代码尚未实际调用 `execute_portfolio_decision()`。
- 虚拟执行源码和测试已有 T+1、整手、碎股清仓、停牌/锁板、先卖后买、最低佣金、印花税、过户费和滑点；尚未接入现役 Web 长驻链路。
- 现有执行层不猜涨跌停和停牌，要求调用方显式传入行情状态，这个边界应保留。

### 8.2 P0 漏洞

| 漏洞 | 当前影响 | 应吸收的轮子 |
|---|---|---|
| 当前 A50 是 2026 冻结成分，没有历史生效区间 | 隔离实验只证明 vn.py 正常单区间的双端包含行为，同时复现了重叠、空字典和全空区间三类接入阻断点；生产仍未实现，组合级历史回测仍有幸存者偏差 | 借 Qlib/vn.py 的生效区间数据结构，自行实现合并、唯一性、空输入失败关闭和交易日规范化 |
| 缺历史时点 ST、板块、涨跌停、停牌和成交量状态 | 执行器虽然正确，但上游无法给出完整历史 `ExecutionQuote` | AKQuant 的显式规则字段、tickflow 的日期化涨跌停测试 |
| 信号、IC、标签和 PnL 时钟不一致 | 运行探针确认真实目标函数构造 `open[t+1]→open[t+2]` 收益；源码审查发现两个 IC 实现额外右移一根，IC/PnL 又会纳入末尾两个补零；模块级 `score_all` 和 `EffectivenessEvaluator` 的 horizon 默认值也少裁一根。成交与 PnL 是分段证据，尚无完整端到端运行证明 | Qlib 的前一周期信号与当前交易步结构；正式反例见隔离实验报告 |
| 训练循环反复使用 validation 指导冠军、精英池和搜索方向 | validation 已是选择集，不能称最终样本外 | skfolio 的时间序列评估思想；最终仍由 AlphaMaster 密封合同控制 |
| 连续仓位单费率回测与 A 股虚拟执行是两套语义 | 固定一手买卖已与 AKQuant/手算一致；RQAlpha 默认费用器少过户费；完整路径仍未对齐 | AKQuant/RQAlpha 只作为逐订单对照 |

### 8.3 P1 漏洞

| 漏洞 | 建议 |
|---|---|
| 缺分红、送转、拆股、退市和现金返还账本 | 参考 RQAlpha position state，新增公司行动事件 |
| 执行器要求调用方显式传入 `AShareFeeSchedule`，但缺少按交易日期自动解析、可审计的费用政策时间表 | 至少覆盖 2023-08-28 印花税变化，保留过户费，并把解析结果继续写入执行身份 |
| 缺 bar、零成交量和多交易日停牌订单生命周期尚未形成完整回放 | 当前执行层先明确区分“缺 bar”与“零成交量停牌”；未来持久回放层再借鉴 AKQuant #329/#334 已修复后的回归测试，覆盖跨日挂单、恢复和终态 |
| 保存/恢复等价性分属两层 | 现有训练 checkpoint 补随机状态和历史缓冲等价；未来持久执行回放另测订单、现金、持仓和权益恢复 |
| 只保存冠军会低估多重检验 | 建立全局试验账本，记录失败、拒绝、重复和搜索预算；随机标签下必须暴露虚假发现 |
| 报告口径分散 | 权威指标保留在 AlphaMaster，第三方只负责展示并做差分 |

## 九、建议的落地顺序

### 阶段 1：先补合同和黄金测试，不加运行时依赖

建议新增：

- `tests/unit/test_point_in_time_universe.py`
- `tests/unit/test_a_share_market_rules.py`
- `tests/unit/test_signal_execution_clock.py`
- `tests/unit/test_portfolio_replay.py`
- `tests/unit/test_corporate_actions.py`
- `tests/unit/test_resume_equivalence.py`
- `tests/unit/test_validation_selection_contract.py`
- `tests/unit/test_authoritative_backtest_contract.py`

现有 `tests/unit/test_portfolio_execution.py` 已覆盖 T+1、停牌/锁板、最低佣金、整手和碎股清仓，不重复写同类测试。

### 阶段 2：隔离差分，不替换生产

1. 固定 AKQuant、RQAlpha、Qlib、vn.py 的 commit。
2. 构造同一组小型 A 股行情、组合目标和费用政策。
3. 比较订单、成交、现金、可卖持仓、费用、权益和失败原因，不只比较夏普。
4. 第三方与 AlphaMaster 不一致时，先判断交易语义，不做“调到结果一样”的后处理。

2026-07-24 已完成第一轮固定样本，详见
[`A_SHARE_QUANT_EXPERIMENTS_20260724.md`](A_SHARE_QUANT_EXPERIMENTS_20260724.md)。
成交费用、下一开盘目标收益构造、历史成分正常区间与三个反例已有本机
固定环境证据；Qlib 仍只有源码合同，完整端到端时钟、跨日订单、公司行动
和生产历史股票池仍未完成。

逐项放行条件：

- QuantStats：仅放行与 AlphaMaster/手算一致的展示字段；不一致字段隐藏并保留差分证据。
- Alphalens：错一日、缺失交易日、停牌和未来价格注入必须全部被测试捕获。
- skfolio：只读选择集；与等权、逆波动比较；记录全部调参；连续权重交给 AlphaMaster 执行。
- RD-Agent：随机标签不得稳定产出优势；封存集访问、缓存污染和同名实验覆盖必须失败关闭。
- `tdx_quant`：分页完整性、除权、时区、股/手单位、停牌和公告时点全部通过后，才允许成为数据候选。

### 阶段 3：选择性吸收

优先顺序：

```text
历史成分区间
→ 行情规则适配层
→ 多交易日执行回放
→ 公司行动与时点费用
→ 全局试验账本
→ QuantStats/Alphalens 展示适配
→ skfolio 组合基准
→ 受限 RD-Agent 研究循环
```

## 十、复制代码的边界

用户已明确：系统仅供个人研究，因此本轮不因“仅限非商业使用”自动降级候选。但用户授权不能替代仓库许可证和平台条款；代码只在各自许可允许的范围内进入本地隔离试验。

但 AlphaMaster 当前是公开 AGPL 仓库，仍需区分：

- MIT / Apache / BSD：可复制、修改和公开，但保留版权与许可证通知。
- GPL / AGPL：本地研究可用；公开组合或修改版需满足对应许可证。
- Commons Clause / RQAlpha 自定义非商业许可：个人研究可用；公开再分发和未来用途按原许可单独审查。
- **没有 LICENSE 的仓库**：公开可见不等于授权复制。只阅读公开源码、观察公开行为和按公开思想独立实现；运行、修改或复制原代码需作者授权或单独许可审查。
- 闭源商业工具：只学习公开文档、用户反馈和工程框架，不复制未公开实现。

## 十一、仍待补证

1. 用户完成小红书登录后，定向读取：
   - A 股量化回测踩坑
   - QMT/PTrade 回测实盘差异
   - 因子挖掘未来函数
   - A 股数据源避雷
2. 抖音登录后只抽取带版本、日志、复现或明确迁移原因的正文/评论，过滤培训和开户链接。
3. AKQuant/RQAlpha/Qlib/vn.py 第一轮窄范围隔离差分已完成；仍需补完整端到端时钟、Qlib 完整运行、跨日订单、公司行动和真实历史股票池。
4. 取得目标券商当前版本的官方接口材料后，再决定 QMT 或 PTrade 适配器。

## 十二、核心证据台账

| 来源 | 访问日 | 类型/等级 | 状态 | 支持的结论 | 限制 |
|---|---|---|---|---|---|
| AKQuant 固定快照、#329、#334 | 2026-07-24 | 源码/测试/固定样本，E3 | 一手费用差分通过；固定快照中 10 项相关 Python 测试在隔离环境通过 | A 股费用与状态回归测试值得复用 | 没有运行完整测试套件或 Rust `cargo test`；同周期终态成交不证明完整下一开盘和跨日生命周期 |
| RQAlpha 固定快照 | 2026-07-24 | 源码/费用模块固定样本，E3 | 默认费用差异已复现 | 可卖持仓、公司行动结构可借；默认费用缺过户费 | 只运行费用模块；RQData 和自定义许可限制 |
| Qlib 固定快照、#1875、#1930 | 2026-07-24 | 发布包与快照源码合同，E2 | 上一时点信号合同通过；完整导入被 Windows 原生依赖阻断 | 研究层强，时钟写法和实验身份值得借 | 不能声称完整运行 E3 |
| vn.py alpha 固定快照 | 2026-07-24 | 原始方法固定样本，E3 | 正常成分区间通过，三个边界反例复现 | 只借生效区间数据结构 | 当前方法有重叠重复、空字典绕过和全空报错，不能直接复制；不验证 alpha 撮合 |
| QuantStats 固定快照、#334、#463 | 2026-07-24 | 源码/Issue，E2 | 历史问题均已处理/关闭 | 适合作展示，但需字段差分 | 旧故障不是排除当前版本的理由 |
| vectorbt #185、#209 | 2026-07-24 | 维护者 Discussion，E1/E2 | 历史讨论 | 适合粗筛，不提供 A 股执行真值 | Numba 顺序模拟可处理部分路径依赖 |
| 迅投 QMT 社区 6408/1022/5914/1931 | 2026-07-24 | 四个帖子合并后包含多个用户，E1 | 版本/券商环境各异 | 需要隔离适配、版本回归和缺行情显式状态 | 单个帖子本身不单独达到 E1；只能说“该帖用户报告”，不能外推产品整体 |
| BigQuant、聚宽、SuperMind | 2026-07-24 | 官方/平台托管文档，E0/E2 | 在线 | 提供环境分层、状态持久化和适配器思路 | 不是独立社区共识 |
| Reddit、B站、微信公开搜索 | 2026-07-24 | 搜索线索，E0/E1 | 覆盖受限 | 用于扩展候选和反证关键词 | 本轮未保留足以支撑核心判断的独立大 A 证据 |
| 小红书、抖音 | 2026-07-24 | 未取得正文 | 登录阻断 | 无 | 不得声称读过正文或评论 |
