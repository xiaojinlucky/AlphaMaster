# A50 龙头轮动：给网页版 GPT 与 BioMe 的证据包

更新时间：2026-07-24

## 1. 本文边界

本文只整理已核实的一手资料、当前仓库的真实状态和仍未完成的证据缺口，不替网页版 GPT 或 BioMe 决定最终架构。

当前服务器正在跑的 50 只基线队列不能被描述为“近一年轮动龙头最终名单”。它使用的是 2026-07-23 冻结的中证 A50 成分，目的是先打通 50 只串行训练、回测和虚拟信号工程链。

## 2. 已确认事实

### 2.1 中证 A50 是龙头候选池，不是轮动成绩榜

中证指数官方编制方案说明：

- 样本来自各行业龙头上市公司；
- 过去一年日均成交金额需位于样本空间前 90%；
- 候选证券需在所属中证三级行业中过去一年日均自由流通市值排名第一、总市值位于全市场前 300，并属于沪股通或深股通范围；
- 最终按过去一年日均自由流通市值选 50 只，并保证各中证二级行业至少一只；
- 成分每半年调整一次，通常在 6 月和 12 月实施。

因此，中证 A50 能作为“高流动性、跨行业、机构可交易龙头”的母池，但官方规则没有要求成分股在过去一年出现过可获利的轮动信号，也没有按动量、相对强弱或轮动收益排序。

一手来源：

- [中证 A50 指数编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/930050_Index_Methodology_cn.pdf)
- [中证 A50 指数事实表](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/930050factsheet.pdf)
- [中证 A50 官方成分 XLS](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/930050cons.xls)

### 2.2 当前仓库已经冻结可审计的 50 只基线

- 成分合同：`universes/csi_a50_20260723.json`
- 成分数：50
- 官方 XLS SHA-256：`172c1ebfa12f2916a3606ee492f1d0e74d940b572bc49d27c4b41f41268694be`
- 成分合同 SHA-256：`987387945fba0cb778b648860bc7579a3cf49e9c3b788596464f714e968bb896`
- 训练/封存切分合同：`universes/csi_a50_20260723_sealed_20250724.json`
- 切分合同 SHA-256：`ec5a9549587802b3f0f82fd566d953315b3ecbf0a42d24f21795a8978a60cf73`
- 50 只训练数据均截止 2025-07-23。
- 统一封存评分窗口为 2025-07-24 至 2026-07-23；封存数据物理隔离，不能进入训练。

当前批次 `batch_a50_26e3472c68ca5c04` 共 50 项。第一项 `000333` / Slurm `568571` 的训练分数为 `1.2064923048`、同数据含成本 replay 夏普为 `0.7105`；第二项 `000617` / Slurm `568740` 的训练分数为 `1.2361105680`、同数据含成本 replay 夏普为 `0.6114`；第三项 `000725` / Slurm `568756` 的训练分数为 `1.4638690948`、同数据含成本 replay 夏普为 `1.0163`；第四项 `000792` / Slurm `568779` 的训练分数为 `1.9909664392`、同数据含成本 replay 夏普为 `0.9750`。四者都不是最终样本外通过证据；第五项 `000988` / Slurm `568800` 已提交并等待调度优先级。在最终一次性封存评估完成前，不得声称任何一只股票已经达到样本外夏普大于 1。

## 3. 可直接借鉴的成熟工程框架

### 3.1 Qlib：优先借鉴实验记录和横截面轮动骨架

微软 Qlib 已提供：

- 对每只股票每天输出预测分数，再按分数构造组合的横截面工作流；
- 多模型基准，结果按不同随机种子重复运行并汇总均值和波动；
- Recorder 实验记录器，可记录参数、指标、模型和产物；
- `TopkDropoutStrategy`，按预测排名卖出持仓中排名靠后的股票、买入未持仓中排名靠前的股票，形成可控换手的 Top-K 轮动。

这些能力与后续“50 只龙头之间智能调控”的工程问题高度重合。它们适合作为复用候选和对照基线，最终是否接入、接入哪一层，由后续架构工单决定。

一手来源：

- [Qlib 官方 benchmark](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)
- [Qlib Recorder 官方文档](https://qlib.readthedocs.io/en/stable/component/recorder.html)
- [Qlib TopkDropoutStrategy 官方文档](https://qlib.readthedocs.io/en/latest/component/strategy.html)

### 3.2 AKShare：接口可用不等于数据源稳定

AKShare 社区已有 `stock_zh_a_hist` 和个股信息接口被远端断连的真实问题记录。当前 AlphaMaster 使用新浪后复权日线、下载后冻结 Parquet 与 manifest 哈希，避免训练期间因上游响应变化而悄悄换数据；但这只能保证本批输入不漂移，不能替代复权正确性、停牌、涨跌停和退市处理的独立核验。

社区证据：

- [AKShare Issue #7051：A 股历史行情接口断连](https://github.com/akfamily/akshare/issues/7051)

## 4. 夏普大于 1 的硬门槛怎样才算真的通过

“不断训练直到 50 只每一只夏普都大于 1”只能用于训练集和选择集上的搜索目标。最终验收必须同时满足：

1. 每只股票的策略在评分前已经冻结；
2. 50 只使用同一个未被训练访问的最终时间窗；
3. 一次性整批揭示，不能看完某只结果后再训练再揭示；
4. 使用扣除交易成本后的结果；
5. 记录总试验次数，并用 Deflated Sharpe Ratio 或 Probability of Backtest Overfitting 检查“试得太多后偶然挑中高夏普”的问题；
6. 只有 50 只全部通过，才允许宣称整批通过。

当前仓库的物理封存、一次性揭示锁和逐只报告合同正是为这个硬门槛服务。第一轮 200 步队列的同数据重放只证明工程链可运行，不计入最终样本外夏普门槛。

当前封存合同为 v2：合同与逐只报告会冻结手续费、滑点和实际 `cost_rate`，零总成本或“报告写了成本但回测实际没用该成本”都会失败。揭盲锁只由 50 份密封文件 SHA-256 的排序多重集合生成；股票代码标签、评分窗口、合同路径、结果路径和策略 SHA 都不参与锁身份。因此看完结果后改标签、改窗口、换策略或换合同文件，也不能再次读取同一批物理字节。该门禁只能防止同一 AlphaMaster 运行身份下的工程性误用，不能替代外部审计或阻止有人蓄意删除锁文件、改代码后作弊。

一手来源：

- [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

## 5. 与 A 股轮动直接相关的研究提醒

中国 A 股不能直接照搬美国市场的普通价格动量结论。已有 A 股研究报告称，传统动量未产生显著利润，而残差动量在其样本中有预测力；其他研究也显示动量与反转效应会随市场状态变化。

这意味着后续的“近一年轮动表现”筛选至少要把原始相对强弱、残差动量、波动和市场状态分开比较，不能先验认定单一动量排名就是最终答案。

一手研究：

- [Residual momentum and the cross-section of stock returns: Chinese evidence](https://www.sciencedirect.com/science/article/pii/S1544612318303325)
- [Wax and wane of the cross-sectional momentum and contrarian effects: Evidence from the Chinese stock markets](https://arxiv.org/abs/1707.05552)

## 6. 尚未完成，不能冒充结论

- 尚未用严格的“近一年轮动表现”定义重新筛选最终 50 只；当前中证 A50 只是官方龙头基线。
- 尚未确定智能调控最终采用相对强弱、Top-K Drop、状态切换、组合优化还是其他机制。
- 小红书、抖音、微信公众号、雪球、知乎等社区的开发者、使用者和评估者证据仍不完整，不能写成已完成社区调研。
- 当前没有真实 QMT、PTrade 或券商实盘渠道；只交付服务器训练、成本回测、虚拟信号和后续飞书推送。
