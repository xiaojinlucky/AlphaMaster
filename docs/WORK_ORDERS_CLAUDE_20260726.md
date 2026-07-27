# AlphaMaster 总控工单板（2026-07-26 起）

> 本板是 AlphaMaster 主线的**唯一现行工单真值**。总控/大脑/审查者 = Claude（Fable 5）；执行者 = 任何接手模型（GPT/Claude/其他）。
> 执行者纪律：只做工单列明的事；每完成一个重要步骤，**更新本板对应工单的状态栏和证据栏**，并同步 `CONTEXT.md` 顶部；测试结果只报通过/失败数量；不确定就写"不确定"，禁止把探索说成验证。
> 姊妹项目 PA_Agent 由独立会话负责（入口工单在其启动提示词与 `D:\Desktop\Quant\MULTI_MARKET_ROADMAP.md` P1–P3），本板不覆盖。

## 1. 系统架构一页（接手必读）

- **AlphaMaster 是什么**：用强化学习搜索因子公式的符号回归系统。Transformer 策略网络 + REINFORCE 在 token 词表上采样公式，StackVM 解释执行（`model_core/engine.py`、`vm.py`、`vocab.py`）；目标收益 `target_ret[t]=log(open[t+2]/open[t+1])`，评分合同 `open_t1_t2_same_index_tail2_v1`（`model_core/target_contract.py:16`）；仓位 `tanh(factor)`。主线=大 A；训练强制服务器 Slurm（本机训练被硬门禁拒绝）；当前不接券商、不发单，落点是"训练→回测→虚拟信号→飞书推送"。
- **数据三层口径（A1 已定，不可回退）**：训练段 train（梯度更新）/ 验证段 validation（fitness 门控与选优；**模型选择口径，绝不是样本外证据**）/ 封存段 sealed test（唯一合法样本外证据，只能经 `evaluation/sealed_oos_campaign.py` 受控揭示）。守卫测试 `tests/unit/test_backtest.py::test_fitness_source_never_labels_validation_as_oos` 正则扫全 model_core。
- **两时钟语义（RND-04A 裁决核心）**：决策时钟 `known_at<=decision_ts`（选股/信号/权重/模型输入，由受信历史 universe 合同管）；执行时钟 `effective_at<=fill_ts`（停牌/ST/涨跌停只进执行层）。RQAlpha 数据只允许进执行时钟。
- **关键模块地图**：`portfolio_manager/universe.py`（沪深300 历史成分受信合同，RND-01/02）→ `controller.py`（signals 完整覆盖门）→ `execution.py`（ExecutionQuote 四值状态/费用/T+1）→ `ledger.py`（不可篡改账本，RND-04B + replay 绑定扩展）→ `replay.py`（RND-04C 动态组合日线回放）。执行状态来源：`data_pipeline/rqalpha_execution_overlay.py`（只读适配器，"保守日线触及规则"，生产身份固化常量比对）。数据用途门禁：`data_pipeline/dataset_purpose.py`（RND-05A）。训练链：`scripts/enqueue_a50_training_batch.py` → `web/training_queue.py`/`training_batch.py`（Web 进程驱动，单活动项串行）→ `web/slurm_training_manager.py`+`slurm_training_client.py`（SSH/SCP）→ 服务器 `scripts/slurm_control.py`+`train_slurm_worker.py`（逐文件哈希+合同+用途三重失败关闭）。
- **必读文档**：`scratch/CLAUDE_HANDOFF_20260726.md`（现场交接）；`docs/evidence/rnd04a_execution_overlay_adjudication_20260726.md`（裁决：七必要字段/三硬条件/13 禁止声明/19 组测试）；`CONTEXT.md`（滚动状态）；`lessons.md`。

## 2. 不可违反边界（红线汇总）

1. **sealed OOS 揭盲需用户届时单独明确授权**；持续 Git 授权≠揭盲授权；绝不在冒烟训练产物上揭盲（一次性不可逆消耗）。
2. 不本机训练；不 reset/checkout/revert 工作区；不创建分支/worktree；不删数据/事故源/G 盘/node13 临时目录（删除前单独确认）；G 盘只读。
3. Git：禁止 `git add .`/`-A`；只按精确路径分批暂存；staged 快照过 gitleaks（`C:\Users\Administrator\.codex-shared\tools\gitleaks\v8.30.1\gitleaks.exe git --staged --no-banner --redact`）+ 文件类型/体积检查；**提交与推送前须向用户一次性确认清单**；只推 `origin`（xiaojinlucky/AlphaMaster，用户明令保持公开），严禁推 `upstream`。数据/日志/checkpoint/凭据/运行态硬阻断。
4. RND-04A 唯一合法表述："完整 04A=BLOCKED 不变；仅放行严格限域 execution-state overlay 子集"。禁止 strict PIT 措辞洗白；RQAlpha 状态不进信号层；LOCKED≠真实封单；`limit=0` 不判 locked；停牌唯一来源 `suspended_days.h5`；qfq 价格与未复权限价禁止跨口径比较；990018.XSHG prev_close 禁用；207 quarantine/1 source_missing 不升级。
5. replay 只是工程证据，禁止宣传为策略绩效或 sealed 成绩；validation 成绩禁止宣传为样本外。
6. 每次收口提醒用户**密钥轮换**（Codex 日志明文泄漏 OKX/长桥/模型密钥与交易密码，仅用户本人能在后台轮换；完成前每轮必提）。

## 3. 已完成并验证（证据指针）

| 项 | 状态 | 证据 |
|---|---|---|
| A1 适应度诚实化（验证段正名+三层口径+守卫测试） | ✅ 含对抗审查修复 | `model_core/backtest.py` 头部声明；32/32 相关测试；全量 unit 与基线一致（8 历史失败零新增） |
| RND-04A 独立裁决 | ✅ 已按审查更正 990018 误述+选样勘误 | 裁决文档 `docs/evidence/rnd04a_execution_overlay_adjudication_20260726.md` |
| RQAlpha overlay 适配器 | ✅ 实现+对抗审查通过（P0=0，P1×2/P2×3 已修复） | `data_pipeline/rqalpha_execution_overlay.py`（1068 行）+ 63 用例（18 真实/45 合成）；h5py==3.16.0 走锁文件流程入 requirements-windows.in + 两 Windows 锁；round_lot 实测 688xxx 全 200 |
| RND-04C 动态组合回放 | ✅ 实现+定向测试；**独立对抗审查未完成**（见 WO-AM-01） | `portfolio_manager/replay.py`(1244)+`scripts/run_dynamic_replay.py`(144)+19/19 测试（13 合成/6 真实）；ledger 最小扩展 +219 行（新表 `portfolio_replay_bindings`）；相关回归 176/0（总控亲测 53/53）；e2e 产物 `local_runs/replay_runs/20260726_rnd04c_*`；真实发现（2026-07-27 审查修正数字）：255/255 个 CSI300 时点均含非 available 成分，每时点 **23–111 只**（2005 年 108–111、近年 23–28）→ 全量真实 replay 必然失败关闭（正确行为） |
| 干净训练发射侦察 | ✅ 只读完成 | 结论全文并入 WO-AM-04；服务器 squeue 空、磁盘充足、SSH 通 |
| 验证盲区定位 | ✅ | tests/property+smoke 有 9 个 HEAD 即坏历史失败（vocab/features 上游漂移陈旧断言簇）→ WO-AM-06 |

## 4. 工单队列（按序执行；每张工单完成后更新本表）

### WO-AM-01 复跑 RND-04C 独立对抗审查 【状态：✅ 完成——**第三门 PASS，0 blocker**（2026-07-27 凌晨复跑，11 向量全结论，四测试文件实测 116/0）。P1×3 处置：P1-1 文档数字失实已更正（quarantine 每时点实为 23–111 只）；P1-2 verify() 绑定副本字段重算比对已落地并加 8 组自洽伪造反例（21/21 过）；P1-3 worktree+分支残留待用户确认删除（并入 WO-AM-02 询问）。P2 处置：②负样本测试、⑤日历防御检查已落地；①绑定行 created_at 不入身份——**书面豁免**：绑定表是审计副本，核心身份链已在决策/执行行由 04B 机制锁死，若未来绑定表升级为权威源则必须纳入行身份；③run 头记录、④摘要加长——可选延后。已知无害残留：非过渡记录 transition_mode 翻 true 且 forced 为空属无效谎言，不在检出范围（已注记在测试 docstring）】

- **目标**：04C 三道门（实现/定向测试/独立审查）的第三道；0 blocker 才可进入 WO-AM-05 的授权询问。
- **审查对象**：`portfolio_manager/replay.py`、`scripts/run_dynamic_replay.py`、`tests/unit/test_dynamic_replay.py`、`ledger.py` 扩展段、两个 `local_runs/replay_runs/20260726_*` 产物。
- **攻击向量（逐条给结论）**：a) RQAlpha 状态注入信号层（含间接：执行状态过滤 universe、lot_size/限价影响信号排序）；b) 越过 2026-06-30（区间/执行日/日历/绑定/verify 各路径）；c) quarantine 升级或静默缩池（含 990018；抽验 FAIL_CLOSED 产物点名的 25 只与覆盖矩阵一致性）；d) bar/volume 推断停牌、limit=0 判 locked；e) 协调篡改五类身份（构造"改绑定 payload+重算内嵌哈希"的自洽篡改，verify() 与账本链须检出；`PRODUCTION_OVERLAY_IDENTITY_SHA256` 可从锚重算且 CLI 强制比对、无绕过参数）；f) ledger 扩展是否稀释 RND-04B（既有路径零改动；只写 binding 不写 execution 不得通过；重跑 3 个 04B 反例）；g) 幂等与确定性（崩溃注入点、run id 确定性、无时间随机源入身份）；h) 测试名实相符抽查 6 个（至少 3 个真实 G 盘用例：002049 停牌 10 拒→复牌触涨停→次日成交、002050 触涨停、000800/002384 真实调样）；i) 产物诚实性（工程证据声明、无绩效宣传）；j) 越界检查（触碰集=声明 5 文件+CONTEXT+scratch 保留脚本）；k) 实测复跑四个测试文件报数量。
- **纪律**：只读；pytest basetemp 放 `scratch/pytest-04creview-*` 完工删除；P0/P1/P2 定级+文件:行号证据+最小修复建议。
- **验收（二元）**：11 个向量逐条有结论；P0=0；P1 全部有修复方案；总判决明确"第三门 PASS/FAIL"。
- **完成后**：修复项落地→复验→更新本板与 CONTEXT。

### WO-AM-02 Git 分批提交与推送 【状态：待用户确认（先完成 WO-AM-01 修复再冻结清单）】

- **目标**：把工作区全部已验证改动按归属分批提交推送 `origin/main`，产出干净提交 C\*（训练发射的源码身份基准）。
- **前置**：WO-AM-01 完成（避免提交后又改）；用户对本工单清单的一次性确认（**必须**，规则见边界 3；确认时同时呈报：仓库=公开是用户明令、暂存区已有 23 个删除、fetch/push 均指向 xiaojinlucky/AlphaMaster）。
- **远程核验（2026-07-27 凌晨已完成）**：`origin` fetch/push 均为 `https://github.com/xiaojinlucky/AlphaMaster.git`；GitHub API 实查 `visibility=PUBLIC`、`viewerPermission=ADMIN`；`upstream`=rosemarycox5334-debug（严禁推送）。本地 HEAD `13393b7` 与 `origin/main` 同步（ahead/behind=0，侦察时核实）。
- **分批草案**（每批一个提交，中文模块化信息；staged 快照逐批过 gitleaks+类型/体积检查）：
  1. 大扫除：23 个已暂存删除（`_launch_*.bat`、`check_ckpt*.py` 等死脚本）+ `config.py`（作废 Sharpe 表清空）+ `tests/unit/test_research_script_scoring_contract.py`（同批合同测试更新）。
  2. RND-01/02/03 历史股票池与 v3 导出：`portfolio_manager/universe.py`、`data_pipeline/parquet_manager.py`（其 RND 部分）、`scripts/build_csi300_historical_am_inputs_v3*.py`（2 个，LF 字节敏感）、`docs/evidence/csi300_historical_am_inputs_v3_release_binding_20260726.json`、`.gitattributes`、`tests/unit/test_csi300_pit_universe.py`、`tests/unit/test_csi300_historical_export.py`。
  3. RND-04B+05A 账本与封存门禁：`portfolio_manager/controller.py`、`__init__.py`、`evaluation/sealed_oos_campaign.py`、`data_pipeline/a_share_akshare.py`、`data_pipeline/dataset_purpose.py`、`run_backtest.py`、`scripts/build_a50_sealed_split.py`、`scripts/train_slurm_worker.py`、`train_file.py`、对应 6+ 组测试（`test_portfolio_ledger/test_portfolio_execution_ledger/test_sealed_oos_campaign/test_a50_sealed_split/test_a_share_sealed_slices/test_run_backtest_contract/test_train_slurm_worker`）。
  4. A1 适应度诚实化：`model_core/backtest.py`、`model_core/engine.py`、`tests/unit/test_backtest.py`、`tests/property/test_backtest_props.py`、`tests/unit/test_target_return_contract.py`。
  5. RND-04A 裁决+overlay：`docs/evidence/rnd04a_execution_overlay_adjudication_20260726.md`、`data_pipeline/rqalpha_execution_overlay.py`、`tests/unit/test_rqalpha_execution_overlay.py`、`requirements-windows.in`、`requirements-windows.lock`、`requirements-dev.lock`。
  6. RND-04C 回放：`portfolio_manager/replay.py`、`portfolio_manager/ledger.py`（含 04B 层+04C 扩展，提交信息注明双层）、`scripts/run_dynamic_replay.py`、`tests/unit/test_dynamic_replay.py`。
  7. 文档收尾：`CONTEXT.md`、`docs/VALIDATION_EVIDENCE.md`、本板、（如审查产生的其他 docs）。
- **排除（永不提交）**：`.env`、`local_runs/**`、`local_data/**`、checkpoints、logs、scratch/**、任何数据文件。`scratch/replay_dev/run_engineering_e2e.py` 默认不提交（工程证据脚本，留 scratch）。`.claude/`（含另一会话的 worktree 与项目 skills）默认不提交，确认时向用户单独提一句。
- **全库对账（2026-07-27 凌晨完成）**：`git status --porcelain` 共 67 条=30 M + 14 ??（含本板与裁决文档→第 7 批）+ 23 staged D（→第 1 批），逐条归入七批草案，**零未归属文件**；`.claude/` 为唯一排除目录。清单在 WO-AM-01 修复落地后冻结为最终版再向用户呈报。
- **执行细则**：逐批 `git add <精确路径>` → `gitleaks git --staged` → `git diff --cached --stat` 核对 → commit；全部批次完成后一次 `git push origin main`（先核 `HEAD==origin/main` 实时 SHA、ahead/behind=0）；推送后验证三方 SHA 一致。钩子失败禁止 `--no-verify`。
- **验收（二元）**：每批 gitleaks 0 泄漏；push 后本地 HEAD==远端==GitHub API 返回 SHA；`git status` 只剩预期未提交项（scratch 等）。

### WO-AM-03 服务器 runtime-v2 重部署到 C\* 【状态：待 WO-AM-02；含一次性授权项】

- **目标**：`/hwdata/home/jinqc/Quant/AlphaMaster-runtime-v2` 从 detached `1d441774`+28 补丁 → 干净 checkout C\*。
- **须用户知情/授权**：checkout 会覆盖 28 个补丁文件=**正式放弃旧批次（batch_8e173c5a…等）续跑能力**（与"干净训练必须新批次"一致，但不可逆）。与 WO-AM-02 的确认合并为一次询问。
- **步骤**：把 C\* 同步进服务器本地镜像 `…/Quant/AlphaMaster`（runtime-v2 的 origin）或改从 GitHub 拉取（二选一，倾向镜像路径少动配置）→ runtime-v2 `fetch + checkout C*` → 工作树干净核验 → 三件到位抽查：`model_core/target_contract.py`（含 open_t1_t2_same_index_tail2_v1）、`data_pipeline/dataset_purpose.py`、`universes/csi_a50_20260723_sealed_20250724.json`。
- **验收（二元）**：远端 `git rev-parse HEAD`==C\*；`git status` 干净；三件抽查存在且哈希与本地一致。

### WO-AM-04 干净训练发射（50 只 A50 新批次） 【状态：待 WO-AM-02/03】

- **数据就绪（侦察已证）**：切分合同 `universes/csi_a50_20260723_sealed_20250724.json`（`ec5a9549…`）；训练切片 50 parquet+manifest 在 `local_data/a_share_akshare_sina_hfq/20260723_train_pre_20250724/`；sealed 切片物理隔离；数据无需重切。A50 批与 v3/741 无依赖（不同数据源族）。
- **发射参数**：`--train-steps 9000 --time-limit 10:00:00`（默认 200 步/30 分钟是跑通档，禁止用于正式）；12 CPU/32G 沿用。
- **步骤**：①`git status` 干净=源码身份冻结；②新队列 DB：enqueue 加 `--queue-db local_runs/training_queue_v2.sqlite3`，Web 进程带 `ALPHAMASTER_TRAINING_QUEUE_DB=同路径`（旧 DB 原样保留，规避被过期 TRAINING 行锁死的全局单活动项索引）；③`scripts/enqueue_a50_training_batch.py --split-contract universes/csi_a50_20260723_sealed_20250724.json --train-steps 9000 --time-limit 10:00:00 --check-only` 过目输出（注意：合同复核会读 sealed parquet 字节做哈希核对，属既有完整性设计、无评分揭示）；④去掉 `--check-only` 正式入队（新 batch_id 自动生成，与旧批次幂等隔离）；⑤`python run_web.py` 点火——**第一观察点**：监控线程会先收敛 `local_runs/current.json` 指向的过期 run（600276/570548 旧合同产物下载→新代码按合同失败关闭→终态），确认其未被提升 READY/未污染发布指针后，队列才领取新批次第 1 项；⑥首件 READY 后人工验收 `result_manifest.json` 的 `scoring_contract_version/git_commit/source_files(34 项)`，确认新合同贯穿再放手串行。
- **时长预期**：单只 5–8h，50 只串行 ≈11–15 天；逐项产出 READY，无需整批等待。失败关闭语义：任何漂移/FAILED/TIMEOUT→该项 NEEDS_ATTENTION、批次挂起、绝不自动跳过。
- **验收（二元）**：首件 manifest 三字段全对；批次推进中无被静默跳过的项；旧批次/旧 DB 零改动。

### WO-AM-05 sealed OOS 揭盲授权门 【状态：远期；三前置缺一不可】

- 前置：①WO-AM-01 第三门 PASS 0 blocker；②WO-AM-04 产出真实训练候选（严禁用 train_steps=2/20 的冒烟产物）；③**用户单独明确授权"同意揭示"**。
- 询问时呈报：候选清单（标的/run/validation 成绩注明"模型选择口径"）、揭示消耗的一次性语义、reveal 锁机制、预期产物（sealed report+receipt）。未获授权：不读 sealed parquet、不建 reveal claim、不生成报告。
- 揭盲后：报告分层呈现 train/validation/sealed-test，sealed 成绩才是样本外结论；对照路线图 A1 的组合层目标语义（组合扣成本 Sharpe>1 + 分位数要求，替代"50 只逐只>1"）。

### WO-AM-06 property/smoke 盲区清理 【状态：排队】

- 对象：9 个 HEAD 即坏失败（`test_feature_props`×2、`test_prop_features`×2、`test_prop_ops`×4、`test_config_fields`×1）+ unit 的 8 个同簇历史失败（e2e vocab/op 计数等）。
- 方向：逐个判定"断言过期（fork 演进后 vocab=131/特征维数变化）"还是"真实缺陷"；过期者按当前语义更新断言并写明依据，真实缺陷立修。禁止 skip 掩盖。
- 验收：`pytest tests/` 全量失败数归零（或每个残留失败有书面豁免理由）；CI 范围扩到 property/smoke。

### WO-AM-07 quarantine 207 源修复（真实全量 CSI300 replay 前置） 【状态：排队；独立数据工单】

- 事实（2026-07-27 审查修正）：255/255 个历史时点均含非 available 成分，每时点 **23–111 只**（2005 年最重 108–111，2021–22 年最低 23，2024 年以来 24–28）→ 全量真实 replay 每时点必然失败关闭（硬条件 A 的正确行为）。补源范围评估必须按 23–111 的全域分布做，不能按近年 24–28 低估。
- 方向：逐只归因 quarantine 原因（FreeStockDB 事故源隔离），评估补源路径（RQAlpha 仅执行层不可作价格源；候选=FreeStockDB 修复版/其他 qfq 一致来源），任何补源必须走独立 source_id+manifest，禁止混 provenance、禁止静默升级。
- 验收：给出"可完整 replay 的最早连续区间"清单与证据；升级任何标的必须有独立审计记录。

### WO-AM-08 质量配套（路线图 A1/A4 持续项） 【状态：排队】

BRAIN 五项检验闸门进流水线（换手率区间/单票权重截断/覆盖率门槛/sub-universe 复测/入库因子相关性<0.7 去重）；AlphaEval 扰动保真检验；skfolio CombinatorialPurgedCV/WalkForward 替换手写分割；MLflow 本地模式（file/SQLite）管实验与 checkpoint；**checkpoint 异地备份到 /hwdata**（当前模型资产 100% 仅存本机，磁盘故障即归零）。每项独立小工单，动手前先读官方文档确认函数与参数。

## 5. 对抗审查方案模板（所有复杂工单完成后适用）

只读审查者，立场=推翻；按攻击向量逐条给 OK/DEFECT(P0 阻断/P1 必修/P2 建议)+文件:行号证据+最小修复；必须实测复跑相关测试并报数量；越界检查（触碰集 vs 声明）；总判决二元。审查者与实现者必须是不同上下文。

## 6. 验收标准总表（速查）

| 工单 | 二元验收（全部满足才算过） |
|---|---|
| WO-AM-01 | 11 攻击向量逐条有结论；P0=0；P1 均有修复方案并落地复验；总判决明确第三门 PASS |
| WO-AM-02 | 每批 gitleaks 0 泄漏；逐批 `--cached` diff 与清单一致；push 后本地/跟踪分支/GitHub 三方 SHA 相等；无 `git add .` |
| WO-AM-03 | 远端 HEAD==C\*；工作树干净；`target_contract.py`/`dataset_purpose.py`/`universes/*.json` 三件在位且哈希与本地一致 |
| WO-AM-04 | `--check-only` ok:true/50 项；首件 `result_manifest.json` 的合同版本/git_commit/34 源文件哈希全对；无被跳过项；旧批次与旧 DB 零改动 |
| WO-AM-05 | 三前置齐备（审查 PASS + 真实训练候选 + 用户书面"同意揭示"）；揭示走唯一受控 runner；报告三层分列 |
| WO-AM-06 | `pytest tests/` 全量失败归零或逐条书面豁免；CI 覆盖 property/smoke |
| WO-AM-07 | 产出"可完整 replay 的最早连续区间"清单；任何升级有独立审计记录 |
| WO-AM-08 | 每子项：官方文档确认 API 后实现 + 测试 + 一次审查；checkpoint 异地备份可恢复演练通过 |

## 7. 进度账本（每个重要步骤后追加一行；接手模型必须维护）

| 时间 | 执行者 | 事件 | 证据 |
|---|---|---|---|
| 07-26 晚 | Fable5 总控 | A1 适应度诚实化落地+验证 | 32/32；unit 基线零新增 |
| 07-26 晚 | 子代理+总控采信 | RND-04A 裁决落盘 | 裁决文档 v1 |
| 07-26 晚 | 子代理 | overlay 适配器实现 | 63/63（18 真实+45 合成） |
| 07-26 晚 | 审查代理 | 阶段审查 P0=0/P1=2/P2=3 → 总控全部修复 | 修复后 32/32 |
| 07-26 深夜 | 子代理 | RND-04C 实现+定向测试 | 19/19+回归 176/0；总控亲测 53/53 |
| 07-26 深夜 | 侦察代理 | 训练发射侦察（B1–B7 阻塞清单） | 并入 WO-AM-04 |
| 07-27 凌晨 | 审查代理 | 04C 审查首跑因额度中断（部分：gitignore 覆盖 OK） | 待 WO-AM-01 复跑 |
| 07-27 凌晨 | Fable5 总控 | 本板落盘；远程核验（PUBLIC/ADMIN/同步）；02:47 自动续跑已排 | 本板 |
| 07-27 早 | 审查代理 | WO-AM-01 复跑：04C 第三门 PASS 0 blocker（P0=0/P1=3/P2=5）；四文件实测 116/0 | 审查报告全文在会话任务输出 |
| 07-27 早 | Fable5 总控 | P1-1 三处文档更正；P1-2 verify() 字段级比对+8 反例；P2-②⑤落地；21/21 复验 | replay.py / test_dynamic_replay.py |
| 07-27 早 | Fable5 总控 | PA_Agent CI 失败（时区依赖测试）只读定位并转交 PA 会话修复 | run 30225188125 |

## 8. 用户侧待办（只有用户本人能做）

1. **密钥轮换**（最高优先）：OKX 与长桥后台轮换全部 API 凭据+交易密码（Codex 会话日志明文泄漏）；完成前每轮收口必提醒。
2. **WO-AM-02+03 合并确认**（WO-AM-01 完成后我会来问）：分批提交清单 + 推送公开仓 + 服务器重部署（覆盖 28 补丁=放弃旧批次续跑）。
3. **WO-AM-04 知情项**：50 只串行 ≈11–15 天墙钟；接受即发射（无需额外操作）。
4. **WO-AM-05 揭盲授权**（远期，届时单独问）。
5. **Norgate/Sharadar 美股数据采购拍板**（管线验证后才问，见 MULTI_MARKET_ROADMAP D5）。
6. PA_Agent 专属会话启动卡片（如尚未点击）。

## 9. 状态更新纪律

每个工单状态变化→更新对应"状态"栏+第 3 节证据表+第 7 节进度账本+`CONTEXT.md` 顶部段；重大新事实（数据/服务器/额度）记入对应工单"状态"栏。**额度紧张时**：优先主循环顺序执行，把并行对抗审查降级为按第 5 节模板的书面审查（逐向量自查+留痕），并在账本注明"降级审查，待额度恢复补独立审查"。用户可见收口三段式：已完成/未完成/有风险待确认 + 下一步 + 密钥轮换提醒。

---
更新日志：
- 2026-07-26 深夜 初版（Claude Fable 5 总控会话）。
- 2026-07-27 凌晨 增补 6–9 节（与 PA_Agent 的 `docs/WORKORDER_MASTER_20260727.md` 结构对齐）；远程核验证据入 WO-AM-02。当前推进位置：WO-AM-01 待复跑（02:47 自动续跑已排）；WO-AM-02 清单草案待冻结。
