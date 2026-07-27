# RND-04A 准入边界独立裁决（2026-07-26）

> 台账合法表述（唯一口径）：**完整 04A = BLOCKED 不变；仅放行严格限域 execution-state overlay 子集（PARTIAL_READY）**。
> 本文是 `scratch/CLAUDE_HANDOFF_20260726.md` 第 9 节第一步要求的独立裁决产物，裁决依据为第 7 节审计 JSON（SHA-256 `157dbb37ed1eee8ae8947d6113ddd591abf1667eba277eb6397bc9bcf222f387`，已复核匹配、内部算术自洽）与第 8 节两时钟语义。
> 关键代码锚点已人工复核：`portfolio_manager/execution.py` 的 `ExecutionQuote`（symbol/session_date/price/status/lot_size=100，status 四值枚举）、`universe.py:55/266` 的受信 `combined_sha256`。

## 一、04C 日线 replay 的真实必要字段（全部只进执行时钟）

| # | 字段 | 来源 | 用途与关键约束 |
|---|---|---|---|
| 1 | 逐日停牌日集合 | `suspended_days.h5` | 派生 `status=SUSPENDED`。停牌唯一来源；禁止由 bar 存在/价格不变/volume==0 推断（反例样本 000002/2006-05-30：停牌日 bar 是填充值） |
| 2 | `limit_up`/`limit_down` 显式非零价格边界 | `stocks.h5` | 派生 `LIMIT_UP_LOCKED`/`LIMIT_DOWN_LOCKED`。执行器无比例猜测逻辑，显式价格是唯一合法输入；0 值＝"该日无普通日涨跌停边界"（科创板上市初期，样本 688981），禁止判 locked |
| 3 | `open/high/low/close`（未复权） | `stocks.h5` | **仅作** execution 层"触及显式限价"判断的同空间比较基准。v3 价格是前复权（qfq），禁止拿 qfq 价与未复权限价跨口径比较；RQAlpha OHLC 永不充当估值/成交/信号价格源 |
| 4 | `prev_close`（排除 990018.XSHG） | `stocks.h5` | 限价制度交叉核验分母（10%/20% 切换、ST 5%）。990018.XSHG 的 432 行空值失败关闭；该代码**在 949 覆盖矩阵内且为唯一 source_missing**（v3 未导出，成分期 2005-04-29～2006-09-29），本就被硬条件 A 与禁止声明第 7 条失败关闭；741 available 交集不含它，排除其 prev_close 语义不损失可 replay 覆盖（2026-07-26 对抗审查更正原"不在 949 映射集内"误述） |
| 5 | 逐日 ST 状态 | `st_stock_days.h5` | ST 进入/退出验证与制度交叉核验。存储为**严格降序**，读取方必须显式处理，不得假定升序 |
| 6 | 交易日历 | 冻结 FreeStockDB `trade_calendar.parquet`（主）；RQAlpha `trading_dates.npy`（一致性哨兵） | 共同区间 5,235 个交易日两源集合完全相等（审计已证）；RQAlpha 日历 2026-06-30 后部分禁用 |
| 7 | `trading_code→order_book_id` 映射 + `listed_date`/`de_listed_date` | `instruments.pk`（静态安全解码，拒绝 find_class/persistent_load） | 949/949 唯一主表匹配、上市有效期内 949/949 全覆盖；不得退回文件名/后缀猜测 |

## 二、扩展需求（不阻塞 04C）

公司行动 known-at/修订链（dividends/split_factor/ex_cum_factor，审计 BLOCKED 且存在同日冲突，qfq 冻结价格架构下本就不消费）；历史简称事件链（仅展示需求）；真实封板/盘口队列/可成交量（日线 replay 用"保守日线触及规则"命名即可诚实运行）；2026-07-01 后执行状态（换新 bundle 重审）；历史费率精确化（`AShareFeeSchedule` 显式参数已满足）；分钟线（域外）。

## 三、裁决：ACCEPT（有条件、严格限域）

理由：① 需求—供给精确匹配——04C 执行层全部状态需求就是 `ExecutionQuote.status` 的派生，overlay 的 4 项 permitted_semantics 恰好覆盖；执行时钟准绳是 `effective_at <= fill_ts`，不需要 known-at 证明，审计中全部 BLOCKED 项都不在必要条件集内。② 质量有独立实证——22,690 缺口 100% 归因（22,511 停牌 + 169 FreeStockDB 漏数 + 10 截止日后未决）、7/8 边界样本组通过（唯一失败在截止日后、域外）、日历双源精确相等。③ 无更优替代，等待完整 04A＝无限期搁置。④ 残余风险（月度事后打包、供应商事后修订不可检测）已声明不隐藏。

**三项硬条件（缺一退回 REJECT）：**
- **A 覆盖缺口失败关闭**：历史时点成分含 quarantine 股或撞上 169 个漏数日时，该时点 replay 失败关闭；不得静默缩池、不得用 RQAlpha 价格填补。接受 overlay ≠ 承诺每个历史时点都能 replay。
- **B 来源身份上移绑定**：`ExecutionQuote` 不携带来源字段，RQAlpha source identity（bundle SHA、成员 SHA、允许字段、日期/代码域、派生规则版本）必须绑定在 replay 运行层身份中。
- **C 状态派生空间隔离**：LOCKED 判断只在 RQAlpha 未复权空间内完成；RQAlpha 价格永不进入估值/成交/信号。

## 四、禁止声明清单（防洗白，13 条）

1. 禁止表述为"RND-04A 通过/完成"；只允许"完整 04A=BLOCKED，仅放行限域 overlay 子集"。
2. 禁止"strict PIT 执行状态""archival PIT"等措辞；禁止把 `effective_at` 写成 `known_at`。
3. 禁止 RQAlpha 任何字段进入决策时钟（选股/信号/权重/模型输入/校准历史）。
4. 禁止把 LOCKED 状态解释为真实封单/排队/不可成交；派生规则必须命名为"保守日线触及规则"，source identity 中显式否认盘口语义（现有枚举名 `*_LOCKED` 是洗白高危点）。
5. 禁止 `limit_up==0 && limit_down==0` → locked 或任何状态推断。
6. 禁止由 bar/价格/成交量推断停牌；唯一来源 `suspended_days.h5`。
7. 禁止 207 quarantine / 1 source_missing 因"RQAlpha 有数据"升级；禁止用 overlay 掩盖决策 universe 覆盖缺口。
8. 禁止使用 2026-07-01 及之后执行状态；10 个未决缺口（600000×2、688072×8）永久失败关闭，直至冻结覆盖 ≥2026-07-13 的同源新 bundle 并重复同一只读审计。
9. 禁止 990018.XSHG 的 `prev_close` 语义。
10. 禁止把 RQAlpha 状态写回 FreeStockDB Parquet 或混同 provenance；禁止 RQAlpha OHLC 充当估值/成交价格源。
11. 禁止声称历史简称、公司行动 known-at/修订链已具备。
12. 禁止把 04C replay 写成真实交易或 sealed OOS；本裁决只放行 overlay 作为 04C 输入，04C 仍需实现、定向测试、独立对抗审查三道门。
13. 禁止把月度事后打包 bundle 描述为"逐日快照留档"；供应商事后修订风险作为已声明残余风险保留。

## 五、正式适配器失败关闭测试清单（19 组）

交接第 9 节第二步已列（1–12，全部必要）：① 供应链身份（bundle SHA `f2d8a0…`、8 成员 SHA、inventory SHA 任一不符拒载）；② 日期上限 >2026-06-30 拒绝（含 10 个未决缺口日期用例）；③ 代码域＝741 available 交集，quarantine/source_missing 注入拒绝；④ 停牌只读 suspended_days（000002/20060530 填充 bar 反例、20170717→0718 复牌边界）；⑤ st_stock_days 降序显式处理（升序假设注入检出；000518/600228/002602 进入退出四点边界）；⑥ 10%→20% 切换（300750 20200821/24 prev_close 复算）；⑦ 科创板上市初期 0 限价（688981 前 5 日不判 locked）；⑧ 未来状态污染（追加合成未来状态后历史输出字节不变或拒绝）；⑨ 990018.XSHG prev_close 失败关闭；⑩ FreeStockDB Parquet 只读约束；⑪ pickle 安全解码策略复用；⑫ h5py 按项目 requirements/锁文件体系进入，不本机临时装。

选样勘误（2026-07-26 对抗审查确认）：第⑤⑦组点名的 688981 / 000518 / 600228 在 v3 中均为 quarantine，位于 741 代码域外、适配器必拒。数值断言由同语义的 available 样本承载（ST 四点边界改用 002602，科创板上市初期 0 限价改用 688072），quarantine 三股保留为"防升级拒绝"断言；该替代不削弱任何禁止声明。

04C 侧义务（对抗审查 P2 建议，采纳）：04C 必须把生产 bundle 的 overlay `identity_sha256` 固化为常量并在运行前比对，闭合 trusted_anchor 注入口的身份链（配合硬条件 B）。

本裁决补充（13–19）：⑬ dataset 缺失语义显式固定（suspended 3,440 vs st 5,553 个 dataset；"无 dataset＝无停牌记录"必须显式声明并测试，741 交集逐一确认两文件存在性映射）；⑭ 派生空间一致性（构造有除权史股票证明 qfq×未复权跨口径比较会误判，该路径必须被禁止）；⑮ lot_size 来源（688xxx 整手 200 股，默认 100 是错的；必须来自经验证显式来源——接入前先核实 instruments 的 `round_lot` 字段——并有 688 标的失败关闭测试）；⑯ 日历一致性运行时哨兵（加载时重验共同区间集合相等）；⑰ replay 层来源身份绑定（篡改 overlay 状态或来源声明 → 身份链检出，对齐 RND-04B 12 入口风格）；⑱ 169 个 FreeStockDB 漏数日（含 000338×127）失败关闭或显式拒绝，禁止填补、禁止归类为停牌；⑲ 决策 universe 覆盖缺口（含 quarantine 成分的历史时点 → controller signals 完整性门必须失败，属 04C 集成测试，与 2009-12-31 空窗同组）。
