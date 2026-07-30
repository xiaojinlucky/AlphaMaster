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

0. **训练批次期间源码冻结（2026-07-27 起，批次 batch_a50_d93b6b4bd5fc7704 完成前有效）**：队列每领取新一项都重校验"本地工作区训练源码哈希==批次冻结值 0e5ca8c3…"。因此以下 34 文件集在批次期间本地禁改：`config.py`、`train_file.py`、`data_pipeline/*.py`（含 overlay/dataset_purpose）、`model_core/*.py`、`strategy_manager/__init__.py`、`strategy_manager/signal.py`、`utils/train_logging.py`、`utils/training_runtime.py`、`scripts/slurm_control.py`、`scripts/train_slurm_worker.py`、`scripts/train_alphamaster.sbatch`。改任何一个=下一项领取 SourceHashDriftError 挂起。本地后处理链（run_backtest/回放/信号）虽不入哈希，也应冻结以保证 50 件结果可比。测试文件、docs、evaluation/ 新增模块、新脚本文件不受限。Web 控制台进程与本机不关机同为批次生命线。

1. **sealed OOS 揭盲需用户届时单独明确授权**；持续 Git 授权≠揭盲授权；绝不在冒烟训练产物上揭盲（一次性不可逆消耗）。
2. 不本机训练；不 reset/checkout/revert 工作区；不创建分支/worktree；不删数据/事故源/G 盘/node13 临时目录（删除前单独确认）；G 盘只读。
3. Git：禁止 `git add .`/`-A`；只按精确路径分批暂存；staged 快照过 gitleaks（`C:\Users\Administrator\.codex-shared\tools\gitleaks\v8.30.1\gitleaks.exe git --staged --no-banner --redact`）+ 文件类型/体积检查；**提交与推送前须向用户一次性确认清单**；只推 `origin`（xiaojinlucky/AlphaMaster，用户明令保持公开），严禁推 `upstream`。数据/日志/checkpoint/凭据/运行态硬阻断。
4. RND-04A 唯一合法表述："完整 04A=BLOCKED 不变；仅放行严格限域 execution-state overlay 子集"。禁止 strict PIT 措辞洗白；RQAlpha 状态不进信号层；LOCKED≠真实封单；`limit=0` 不判 locked；停牌唯一来源 `suspended_days.h5`；qfq 价格与未复权限价禁止跨口径比较；990018.XSHG prev_close 禁用；207 quarantine/1 source_missing 不升级。
5. replay 只是工程证据，禁止宣传为策略绩效或 sealed 成绩；validation 成绩禁止宣传为样本外。
6. 凭据与账户安全事项只保留在非 Git 本地安全记录；公开工单不得记录具体服务、事件媒介或账户细节。

## 3. 已完成并验证（证据指针）

| 项 | 状态 | 证据 |
|---|---|---|
| A1 适应度诚实化（验证段正名+三层口径+守卫测试） | ✅ 含对抗审查修复 | `model_core/backtest.py` 头部声明；32/32 相关测试；全量 unit 与基线一致（8 历史失败零新增） |
| RND-04A 独立裁决 | ✅ 已按审查更正 990018 误述+选样勘误 | 裁决文档 `docs/evidence/rnd04a_execution_overlay_adjudication_20260726.md` |
| RQAlpha overlay 适配器 | ✅ 实现+对抗审查通过（P0=0，P1×2/P2×3 已修复） | `data_pipeline/rqalpha_execution_overlay.py`（1068 行）+ 63 用例（18 真实/45 合成）；h5py==3.16.0 走锁文件流程入 requirements-windows.in + 两 Windows 锁；round_lot 实测 688xxx 全 200 |
| RND-04C 动态组合回放 | ✅ 实现+定向测试；**独立对抗审查未完成**（见 WO-AM-01） | `portfolio_manager/replay.py`(1244)+`scripts/run_dynamic_replay.py`(144)+19/19 测试（13 合成/6 真实）；ledger 最小扩展 +219 行（新表 `portfolio_replay_bindings`）；相关回归 176/0（总控亲测 53/53）；e2e 产物 `local_runs/replay_runs/20260726_rnd04c_*`；真实发现（2026-07-27 审查修正数字）：255/255 个 CSI300 时点均含非 available 成分，每时点 **23–111 只**（2005 年 108–111、近年 23–28）→ 全量真实 replay 必然失败关闭（正确行为） |
| 干净训练发射侦察 | ✅ 只读完成 | 结论全文并入 WO-AM-04；服务器 squeue 空、磁盘充足、SSH 通 |
| 验证盲区定位 | ✅ | tests/property+smoke 有 9 个 HEAD 即坏历史失败（vocab/features 上游漂移陈旧断言簇）→ WO-AM-06 |
| WO-AM-07F 日级漏数精确化 | ✅ 数据产物 PASS；叙述勘误已正式收口 | 公开审查摘要 `docs/evidence/wo_am07f_daily_gap_closeout_20260729.md`；完整逐日表 `c11ce7a1…e811a` 与逐股汇总 `a99345b0…499c` 保留在 scratch，不上传运行数据 |
| WO-AM-07A 首批 26 只 qfq 修复 | ✅ 主线 v4 构建与独立审计 PASS | `G:\QuantData\free-stockdb\am_exports\20260727_csi300_qfq_repair_v4`：105 文件/5,448,565 字节，manifest `b4a58231…f0628`；26 只 `continuity_violations=0`、`ohlc_violations=0`；父 v3 manifest 仍为 `e07fffd0…404c`。证据 `docs/evidence/wo_am07a_v4_mainline_accept_20260729.md`。99 点裁决、local 8 点与逐公司公告搜证降为非阻塞增强研究支线 |

## 4. 工单队列（按序执行；每张工单完成后更新本表）

### WO-AM-01 复跑 RND-04C 独立对抗审查 【状态：✅ 完成——**第三门 PASS，0 blocker**（2026-07-27 凌晨复跑，11 向量全结论，四测试文件实测 116/0）。P1×3 处置：P1-1 文档数字失实已更正（quarantine 每时点实为 23–111 只）；P1-2 verify() 绑定副本字段重算比对已落地并加 8 组自洽伪造反例（21/21 过）；P1-3 worktree+分支残留待用户确认删除（并入 WO-AM-02 询问）。P2 处置：②负样本测试、⑤日历防御检查已落地；①绑定行 created_at 不入身份——**书面豁免**：绑定表是审计副本，核心身份链已在决策/执行行由 04B 机制锁死，若未来绑定表升级为权威源则必须纳入行身份；③run 头记录、④摘要加长——可选延后。已知无害残留：非过渡记录 transition_mode 翻 true 且 forced 为空属无效谎言，不在检出范围（已注记在测试 docstring）】

- **目标**：04C 三道门（实现/定向测试/独立审查）的第三道；0 blocker 才可进入 WO-AM-05 的授权询问。
- **审查对象**：`portfolio_manager/replay.py`、`scripts/run_dynamic_replay.py`、`tests/unit/test_dynamic_replay.py`、`ledger.py` 扩展段、两个 `local_runs/replay_runs/20260726_*` 产物。
- **攻击向量（逐条给结论）**：a) RQAlpha 状态注入信号层（含间接：执行状态过滤 universe、lot_size/限价影响信号排序）；b) 越过 2026-06-30（区间/执行日/日历/绑定/verify 各路径）；c) quarantine 升级或静默缩池（含 990018；抽验 FAIL_CLOSED 产物点名的 25 只与覆盖矩阵一致性）；d) bar/volume 推断停牌、limit=0 判 locked；e) 协调篡改五类身份（构造"改绑定 payload+重算内嵌哈希"的自洽篡改，verify() 与账本链须检出；`PRODUCTION_OVERLAY_IDENTITY_SHA256` 可从锚重算且 CLI 强制比对、无绕过参数）；f) ledger 扩展是否稀释 RND-04B（既有路径零改动；只写 binding 不写 execution 不得通过；重跑 3 个 04B 反例）；g) 幂等与确定性（崩溃注入点、run id 确定性、无时间随机源入身份）；h) 测试名实相符抽查 6 个（至少 3 个真实 G 盘用例：002049 停牌 10 拒→复牌触涨停→次日成交、002050 触涨停、000800/002384 真实调样）；i) 产物诚实性（工程证据声明、无绩效宣传）；j) 越界检查（触碰集=声明 5 文件+CONTEXT+scratch 保留脚本）；k) 实测复跑四个测试文件报数量。
- **纪律**：只读；pytest basetemp 放 `scratch/pytest-04creview-*` 完工删除；P0/P1/P2 定级+文件:行号证据+最小修复建议。
- **验收（二元）**：11 个向量逐条有结论；P0=0；P1 全部有修复方案；总判决明确"第三门 PASS/FAIL"。
- **完成后**：修复项落地→复验→更新本板与 CONTEXT。

### WO-AM-02 Git 分批提交与推送 【状态：✅ 完成（2026-07-27 08:10，七批提交并推送，三方 SHA 一致）】

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

### WO-AM-03 服务器 runtime-v2 重部署到 C\* 【状态：✅ 完成（2026-07-27 08:13，远端固定为 `bb280ca`）】

- **目标**：`/hwdata/home/jinqc/Quant/AlphaMaster-runtime-v2` 从 detached `1d441774`+28 补丁 → 干净 checkout C\*。
- **须用户知情/授权**：checkout 会覆盖 28 个补丁文件=**正式放弃旧批次（batch_8e173c5a…等）续跑能力**（与"干净训练必须新批次"一致，但不可逆）。与 WO-AM-02 的确认合并为一次询问。
- **步骤**：把 C\* 同步进服务器本地镜像 `…/Quant/AlphaMaster`（runtime-v2 的 origin）或改从 GitHub 拉取（二选一，倾向镜像路径少动配置）→ runtime-v2 `fetch + checkout C*` → 工作树干净核验 → 三件到位抽查：`model_core/target_contract.py`（含 open_t1_t2_same_index_tail2_v1）、`data_pipeline/dataset_purpose.py`、`universes/csi_a50_20260723_sealed_20250724.json`。
- **验收（二元）**：远端 `git rev-parse HEAD`==C\*；`git status` 干净；三件抽查存在且哈希与本地一致。

### WO-AM-04 干净训练发射（50 只 A50 新批次） 【状态：ACTIVE——000333/000617/000725/000792 READY；000988 训练中】

- **数据就绪（侦察已证）**：切分合同 `universes/csi_a50_20260723_sealed_20250724.json`（`ec5a9549…`）；训练切片 50 parquet+manifest 在 `local_data/a_share_akshare_sina_hfq/20260723_train_pre_20250724/`；sealed 切片物理隔离；数据无需重切。A50 批与 v3/741 无依赖（不同数据源族）。
- **发射参数**：`--train-steps 9000 --time-limit 10:00:00`（默认 200 步/30 分钟是跑通档，禁止用于正式）；12 CPU/32G 沿用。
- **步骤**：①`git status` 干净=源码身份冻结；②新队列 DB：enqueue 加 `--queue-db local_runs/training_queue_v2.sqlite3`，Web 进程带 `ALPHAMASTER_TRAINING_QUEUE_DB=同路径`（旧 DB 原样保留，规避被过期 TRAINING 行锁死的全局单活动项索引）；③`scripts/enqueue_a50_training_batch.py --split-contract universes/csi_a50_20260723_sealed_20250724.json --train-steps 9000 --time-limit 10:00:00 --check-only` 过目输出（注意：合同复核会读 sealed parquet 字节做哈希核对，属既有完整性设计、无评分揭示）；④去掉 `--check-only` 正式入队（新 batch_id 自动生成，与旧批次幂等隔离）；⑤`python run_web.py` 点火——**第一观察点**：监控线程会先收敛 `local_runs/current.json` 指向的过期 run（600276/570548 旧合同产物下载→新代码按合同失败关闭→终态），确认其未被提升 READY/未污染发布指针后，队列才领取新批次第 1 项；⑥首件 READY 后人工验收 `result_manifest.json` 的 `scoring_contract_version/git_commit/source_files(34 项)`，确认新合同贯穿再放手串行。
- **时长预期**：单只 5–8h，50 只串行 ≈11–15 天；逐项产出 READY，无需整批等待。失败关闭语义：任何漂移/FAILED/TIMEOUT→该项 NEEDS_ATTENTION、批次挂起、绝不自动跳过。
- **验收（二元）**：首件 manifest 三字段全对；批次推进中无被静默跳过的项；旧批次/旧 DB 零改动。
- **2026-07-27 15:19 首件真实结果**：000333 / 作业 577313 在 cu18 完成
  9000/9000 步，Slurm=COMPLETED、elapsed=05:52:48、exit=0:0；远端
  `result_manifest.json` 自报 COMPLETED，`scoring_contract_version`、
  `git_commit=bb280ca2dada4b087ce1bf5d86f1292b73c576b9`、`source_files=34`
  三字段均正确。但训练引擎每 20 步保存一次 checkpoint，9000 步形成 450 个；
  worker 又把 450 checkpoint+策略+训练历史全部收入 `artifacts`（452 项，
  2,799,966,432 bytes），控制端 `result_run()` 对产物列表设置 `len<=64`
  的失败关闭门，因此报“result manifest 产物列表非法”。队列现为
  NEEDS_ATTENTION=1/QUEUED=49/READY=0，批次状态 NEEDS_ATTENTION；
  Web 正常监听。未改队列、未重试、未停止/重启任何进程或作业。
- **根因边界**：这是冻结提交内的规模合同不一致，不是训练失败、下载丢失或
  三字段漂移：`model_core/engine.py:986-987` 固定每 20 步保存，
  `scripts/train_slurm_worker.py:581-610` 收集全部匹配文件，
  `scripts/slurm_control.py:904-911` 拒绝超过 64 项；既有单测只构造单个
  checkpoint，未覆盖 9000 步会产生 450 项的跨模块规模组合。按本批冻结红线，
  未修改这些文件；恢复训练需要用户另行授权源码修复与批次恢复方式。
- **2026-07-29 11:00 产物合同收口**：用户授权单独解冻训练产物清单合同后，
  在 34 个冻结训练文件之外新增固定 SHA 的结果适配器。它先完整核对远端
  452 项原始产物，再只交付最高 step checkpoint、策略和训练历史 3 项。
  000333/577313 已通过训练、回测、虚拟信号和队列四层验收进入 READY；
  冻结训练源码 SHA-256 仍为 `0e5ca8c3…64ab0`。
- **2026-07-29 13:27 NODE_FAIL 恢复门**：000617/581389 因 cu19 重启进入
  `NODE_FAIL`，step 2760 checkpoint 完整可恢复。冻结集合外已实现逻辑
  `planned_run_id` 与物理 `execution_run_id` 分离、一次性 NODE_FAIL 事务、
  父 run/job/checkpoint 谱系、提交前固定字节 checkpoint 导入和断线幂等
  重验。13:42 最终固定测试 152/0，独立审查 `ACCEPT`，
  P0/P1/P2=`0/0/0`；导入器 actual/pin 均为 `de851b77…0e06`。
  只读实探发现冻结控制器对终态 job 的 `squeue` 查询会先报 Invalid job id；
  最终导入器改用固定 argv 的 `sacct` 唯一主行验签，并核对 owner、job name
  和 NODE_FAIL；三个计算节点均返回同一 581389 NODE_FAIL 事实。生产
  队列/Web/Slurm 均未实施新恢复，等待用户授权。
- **2026-07-29 13:57 生产恢复已实施**：用户授权后完成生产 SQLite 在线快照、
  schema 迁移和一次性恢复事务。逻辑 planned run 保持
  `run_20260723T235959Z_867bfc69`，物理 execution run 为
  `run_20260729T054526Z_1141da4d`，新 Slurm job **581519** 在 cu01
  RUNNING。首次 checkpoint 导入因通用 Python 不含 torch 在 `sbatch` 前
  失败关闭，未生成 job；只把导入动作切到 runtime `.venv/bin/python` 后，
  同一 execution run 重试成功。首个新 checkpoint step 2780 的旧历史前缀
  与 step 2760 在 18 个字段逐项相等，训练随后到 step 2860，确认不是从 0
  重跑。队列为 `READY=1/TRAINING=1/QUEUED=48`，数据库完整性通过；
  581519 尚未完成，继续监护到 9000 步、结果适配、READY 和增量备份。
- **2026-07-29 18:27 000617 全链闭环**：581519 在 cu01
  `COMPLETED/0:0`，墙钟 04:22:16，完整训练到 9000 步。远端原始结果
  315 件/2,001,758,285 字节经固定适配器完整核验后归一化为 step 9000
  checkpoint、策略、训练历史 3 件；runtime commit、34 源文件、数据 SHA、
  scoring contract 和恢复谱系全对。训练、含成本 replay、虚拟信号和队列
  四层均 READY；replay Sharpe=0.9352 仍是同训练数据成绩，不是 sealed
  OOS。READY 后增量备份完成，17 个普通持久文件逐文件 SHA 差异 0，
  队列与信号 SQLite 在线快照均 integrity=ok、DELETE 模式、WAL/SHM=0。
  队列已自动推进 000725 / 581904，当前
  `READY=2/TRAINING=1/QUEUED=47`。
- **2026-07-30 01:44 000725 全链闭环**：581904 在 cu01
  `COMPLETED/0:0`，墙钟 06:15:01，完整训练到 9000 步。远端原始结果
  452 件经固定适配器逐件核验后归一化为 step 9000 checkpoint、策略、
  训练历史 3 件；run/job/runtime commit/34 源文件/数据/scoring contract
  全对，本机合同验收 25/0（含远端原始 452 件逐件复验）。训练、含成本
  replay、虚拟信号和队列四层
  READY；replay Sharpe=1.2393 仍是同训练数据成绩，不是 sealed OOS。
  READY 后首次完整版本生产迁移完成：1,268 文件/3,202.5 MB/201 秒，
  `CURRENT` 原子指向首个 generation。36/36 SQLite 快照
  integrity=ok、DELETE、WAL/SHM=0；000725 的 17 个非 SQLite 持久文件
  逐 SHA 一致，signal SQLite 业务行一致；独立验收 8/0、对抗复审
  ACCEPT，P0/P1/P2=`0/0/0`。队列已自动推进 000792 / 582419，当前
  `READY=3/TRAINING=1/QUEUED=46`。
- **2026-07-30 09:15 000792 全链闭环**：582419 在 cu05
  `COMPLETED/0:0`，墙钟 08:24:18，完整训练到 9000 步。远端原始结果
  452 件经本机固定适配器逐件核验后归一化为 step 9000 checkpoint、
  策略、训练历史 3 件；run/job/runtime commit/34 源文件/数据/scoring
  contract 全对，合同验收 25/0。训练、含成本 replay、虚拟信号和队列
  四层 READY；replay Sharpe=0.9940 是同训练数据成绩，不是 sealed OOS。
  READY 后只执行一次增量备份：1,288 文件/3,218.5 MB/221 秒；
  `CURRENT` 原子切换到第二个 generation。37/37 SQLite 快照
  integrity=ok、DELETE、WAL/SHM=0；000792 的 17 个非 SQLite 持久文件
  逐 SHA 一致，signal SQLite 业务行一致；独立验收 8/0、对抗复审
  ACCEPT，P0/P1/P2=`0/0/0`。队列已自动推进 000988；活动 job 身份不进入
  公开快照，当前
  `READY=4/TRAINING=1/QUEUED=45`。

### WO-AM-05 sealed OOS 揭盲授权门 【状态：远期；三前置缺一不可】

- 前置：①WO-AM-01 第三门 PASS 0 blocker；②WO-AM-04 产出真实训练候选（严禁用 train_steps=2/20 的冒烟产物）；③**用户单独明确授权"同意揭示"**。
- 询问时呈报：候选清单（标的/run/validation 成绩注明"模型选择口径"）、揭示消耗的一次性语义、reveal 锁机制、预期产物（sealed report+receipt）。未获授权：不读 sealed parquet、不建 reveal claim、不生成报告。
- 揭盲后：报告分层呈现 train/validation/sealed-test，sealed 成绩才是样本外结论；对照路线图 A1 的组合层目标语义（组合扣成本 Sharpe>1 + 分位数要求，替代"50 只逐只>1"）。

### WO-AM-06 property/smoke 盲区清理 【状态：✅ 完成（2026-07-27，总控复核 48/48 + 冻结集零改动）】

**结果**：17 个历史失败全数清零=12 过期断言 + 5 环境/flaky，**真实源码缺陷 0、skip 0**；tests/unit 1090/0 + property/smoke 48/0，**测试全树首次全绿**。9 个测试文件 +223/−84，3 个嵌错误数字的测试更名。修法原则=数字断言改为从被测模块常量推导（防再漂移）。
**关键勘误（更正本板旧口径）**：真实词表=**65 特征 + 62 算子 = 127 词元**（07-03 扩到 66/131 后，07-04 `23a5b1e`"移除二值算子"删 LT/GT 等 4 个，IF_GT 因输出连续保留）。旧"65+66=131"出自 7 月中的 ChatGPT 审核对话，已过时。
**批次后工单（源码冻结解除后执行）**：① 根 `config.py:126` 遗留 `INPUT_DIM=20` 与 `data_pipeline/data_manager.py:162-176` 不可达 ImportError 兜底（违反 let-it-crash）清理 + 两个冻结值断言同步；② `model_core/config.py:32,44` 过时注释勘误；③ 新建 GitHub Actions workflow（仓库现无 `.github/workflows`）——总控决定：做，但须先解决"真实 G 盘用例在 runner 上的显式环境跳过语义"（skip 必须带 reason 且计数可见，不得静默），与①②同批。

- 对象：9 个 HEAD 即坏失败（`test_feature_props`×2、`test_prop_features`×2、`test_prop_ops`×4、`test_config_fields`×1）+ unit 的 8 个同簇历史失败（e2e vocab/op 计数等）。
- 方向：逐个判定"断言过期（fork 演进后 vocab=131/特征维数变化）"还是"真实缺陷"；过期者按当前语义更新断言并写明依据，真实缺陷立修。禁止 skip 掩盖。
- 验收：`pytest tests/` 全量失败数归零（或每个残留失败有书面豁免理由）；CI 范围扩到 property/smoke。

### WO-AM-07 quarantine 207 源修复（真实全量 CSI300 replay 前置） 【状态：主线 07A/07B/07F ✅；07C/07D 排队】

**归因结论（真实查询，零估算；逐只清单在会话 scratchpad `wo_am07_result.json`）**：六桶=A 仅复权断裂 133 只 / B 无因子+断裂 54 / C 仅无因子 16 / D 次新不足484根 3 / E 单行脏记录 1（600228）/ F 上游无此券 1（990018）。A∪B∪C=203 只（98.1%）同根因=**FreeStockDB 发布者复权因子表覆盖不完整**（工作副本与发布者 manifest 296/299 SHA 一致→非本地损耗，重同步零增益；事故源与此无关）。**单修任何一桶解锁 0 时点**，必须 A+B+C 一揽子。敏感度：修 203 只→224 时点（2006-10-31~2025-05-30 连续）；+D→237 时点至 2026-06-30；**最小集 29 只→解锁 2026-01-23~2026-06-30（6 时点，首个真实全量回放区间）**。三道墙分层口径：状态层 237；退市残留 11 只/12 时点需 07C 语义裁决（否则最长连续段 96 时点 2017-02~2025-01）；日级另有 12 只 available 股窗口内缺口待 07F 拆分（000338×185 天最重）。**正式建议放弃 990018 所在的 2005-04~2006-09 段（18 时点），全量回放史实起点=2006-10-31。**

子工单（验收标准见归因报告，要点）：
- **07A 复权因子重建**（203 只，主工单；分期：首批 29 只）：先探发布者 manifest 新版/ProApi 额度恢复走同源；否则外源（AKShare 等）——qfq 核对必须比逐日收益而非绝对价；独立 source_id+v4 manifest+逐只审计；v3/事故源零字节改动；203 只断裂清零；过独立对抗审查。
- **07B replay 资格分级** ✅（2026-07-27 总控裁决落盘 `docs/evidence/wo_am07b_replay_eligibility_20260727.md`：484 根=训练合同参数不适用 replay 价格层；三只次新股 source_covers_membership 全 true；`replay_price_eligible` 四条取值规则已定义，物理列随 v4 落地）。
- **07A 首批 29 只中的 26 只复权修复【✅ PASS，2026-07-29 10:26】**：按原始 `/goal` 的六个成分时点重推，非 available 并集精确为 29 只；001280、001391、600930 归 07B，其余 26 只从冻结 capture `wo07a-20260729T005459992563Z-10ff978b` 的新浪原始行情与 qfq factor 重建。按已拍板的回放起点 2006-10-31 和父 v3 截止日 2026-07-24 生成独立输出 `G:\QuantData\free-stockdb\am_exports\20260727_csi300_qfq_repair_v4`，共 105 文件、5,448,565 字节，manifest SHA-256 `b4a582318e4c2f94bbd10feec99894cb82b4077ab0f691ce83450f3eeebf0628`。独立输出重算审计：26/26、`continuity_violations=0`、`ohlc_violations=0`；父 v3 manifest SHA-256 仍精确为 `e07fffd04c9d53a897ae688ad05897a03273acf14010f799e1aca85579a8404c`。正式证据 `docs/evidence/wo_am07a_v4_mainline_accept_20260729.md`。此前扩展的 99 点裁决体系、StockDB local 8 点和逐公司公告搜证超出原任务完成条件，已有产物保留为增强型诊断资料，但不再作为 v4 阻塞门，也不得宣传为 PIT 或 sealed 证据。
- **07C 退市残留语义裁决**（11 只/12 时点，设计工单）：有交易所退市证据时显式剔除+逐只审计，非静默缩池；裁决前维持失败关闭。
- **07D 600228 单行修复**（搭车，优先级最低，时点增益 0）。
- **07F 日级漏数精确化 ✅ 数据产物 PASS；叙述勘误已正式收口（2026-07-27 复核）**：窗口缺口 29,616=显式停牌 29,479+真漏数 137；真漏数涉及 8 只，目标区间 2006-10-31～2026-06-30 内有 136 个威胁交易日，余 4,642 个交易日组成 7 段连续安全区间：2006-10-31～2015-08-04（2,132 日）、2015-08-06～2017-07-13（471 日）、2017-07-17～2018-06-29（234 日）、2019-01-07～2020-04-10（306 日）、2020-04-14～2023-06-14（770 日）、2023-06-16～2026-05-22（708 日）、2026-05-26～2026-06-24（21 日）。旧口径 `169 点/12 只` 来自 v2 的 273 个 ready 文件与“RQAlpha 正常 bar”条件；新口径改为 v3 的 741 个 available 文件+成员窗口+非停牌条件，精确桥接为减 35 点/7 只（601238×27、600372×3、000596/002028/302132/600415/600845 各×1），加 3 点/3 只（002607/600256/688065 各×1），即 `169-35+3=137`、`12-7+3=8`。原始 `REPORT.md` 的两处叙述须以本板为准：① `S0_available_members_data_hole_points=0` 仅证明日期落在文件首末范围，不能证明快照日实际有 bar；② 000338 成员窗口 185=停牌 58+真漏 127，全 span 189=停牌 62+真漏 127。公开审查摘要为 `docs/evidence/wo_am07f_daily_gap_closeout_20260729.md`；完整逐日表 `c11ce7a1…e811a`、逐股汇总 `a99345b0…499c` 和原始报告 `60825487…0c7` 只保留在 scratch，不上传运行数据。
- 执行通道治理（2026-07-27 用户授权）：本机 Codex CLI（gpt-5.6-sol @ high）可承接执行层工单（`codex exec --sandbox workspace-write`，headless 不开 GUI）；Claude 总控出工单+验收，Codex 产物一律经总控复核才落账。Codex 会话日志仍明文落 `~/.codex/sessions`——工单提示词里禁止出现任何密钥。
- **不修清单**：990018（当前不可修，唯一他源被红线禁止）；26 只 2006-10 前退出者（放弃段内，修了不增加时点）。

（原第一阶段描述保留存档）

- 事实（2026-07-27 审查修正）：255/255 个历史时点均含非 available 成分，每时点 **23–111 只**（2005 年最重 108–111，2021–22 年最低 23，2024 年以来 24–28）→ 全量真实 replay 每时点必然失败关闭（硬条件 A 的正确行为）。补源范围评估必须按 23–111 的全域分布做，不能按近年 24–28 低估。
- 方向：逐只归因 quarantine 原因（FreeStockDB 事故源隔离），评估补源路径（RQAlpha 仅执行层不可作价格源；候选=FreeStockDB 修复版/其他 qfq 一致来源），任何补源必须走独立 source_id+manifest，禁止混 provenance、禁止静默升级。
- 验收：给出"可完整 replay 的最早连续区间"清单与证据；升级任何标的必须有独立审计记录。

### WO-AM-08 质量配套（路线图 A1/A4 持续项） 【状态：备份子项已上线，其余排队】

- **checkpoint 异地备份 ✅ 已上线（2026-07-27；完整版本原子切换合同于 2026-07-29 修复；首次生产迁移于 2026-07-30 通过）**：`scripts/backup_am_assets.py`。大文件按当前版本的同路径同字节数增量跳过，小文件/账本每次重传；运行中的 SQLite 使用在线一致性快照并生成 0 字节 WAL/SHM。全部文件先进入唯一 staging，随后以上一完整版本的硬链接副本为基底构造新版本；新版本闭合后才原子替换 `CURRENT` 指针，上一完整版本始终保留，避免主库与 sidecar 跨文件崩溃窗口。符号链接删除边界、掉电持久性、中段失败残留和 `CURRENT` 提交边界均已通过故障注入与独立复审。000725 READY 后的首次生产迁移为 1,268 文件/3,202.5 MB、generation 1,270 文件、36 个 SQLite；000792 READY 后已推进到第二代 generation 1,290 文件、37 个 SQLite。两次均 `.incoming/.building` 为空，独立验收与对抗复审通过，P0/P1/P2=`0/0/0`。目标 `compute-node-11:/hwdata/home/jinqc/AlphaMaster-backup/`；继续由本任务 30 分钟心跳逐件监护。
- 其余排队（BRAIN 五闸门/AlphaEval/skfolio/MLflow）：**注意批次源码冻结（第 2 节第 0 条）**——skfolio 换 CV 属 model_core 改动、BRAIN/AlphaEval 若接入现有 evaluation 流水线也受"后处理冻结"约束；批次期间只允许做"新增独立模块+独立测试"的准备工作，接线一律等批次完成或明确接受批次挂起再做。动手前先读官方文档确认函数与参数。

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
| 07-27 08:10 | Fable5 总控（用户全案批准） | WO-AM-02 ✅：七批提交（9ba2f5b→bb280ca）逐批 gitleaks 零泄漏；推送 origin/main 三方 SHA 一致 bb280ca | git log；GitHub API |
| 07-27 08:11 | Fable5 总控 | P1-3 ✅：违规 worktree+分支已删（tip 13393b7 无损失） | git worktree list / branch |
| 07-27 08:13 | Fable5 总控 | WO-AM-03 ✅：runtime-v2 经 bundle 离线部署 checkout bb280ca（28 补丁按授权覆盖），跟踪树零脏，三件抽查+合同串确认；部署包两端已清理 | ssh 输出 |
| 07-27 08:15 | Fable5 总控 | WO-AM-04 点火：check-only VALIDATED → 新批次 `batch_a50_d93b6b4bd5fc7704`（50 项/9000 步/10h/源码 0e5ca8c3…）入 `training_queue_v2.sqlite3`；Web 控制台分离进程启动、8765 监听；旧 run 收敛与首项领取观察中 | enqueue JSON；端口探测 |
| 07-27 08:20 | Fable5 总控 | 旧 run 被新代码按评分合同正确失败关闭（未污染 READY）；首项 000333 派发成功，**Slurm 作业 577313 在 cu18 RUNNING**（排队≈0） | state.json；squeue |
| 07-27 08:5x | Fable5 总控 | WO-AM-08 备份子项上线：基线 1148 文件/3.15GB 同步 HPC 完成、逐一核对大小；首跑踩 WSL-bash/GBK 解码坑已修（纯 Python 双进程管道）；6h 增量+巡检 cron 挂起 | backup_am_assets.py 输出 |
| 07-27 08:5x | Fable5 总控 | 源码冻结红线（批次期间 34 文件禁改）写入本板第 2 节第 0 条；WO-AM-06/07 两代理并行开工 | 本板 |
| 07-27 09:0x | 分析代理 | WO-AM-07 第一阶段归因 ✅：六桶 207+1、98.1% 同根因（上游复权因子表缺失）、单桶解锁 0 时点、最小集 29 只解锁 2026-01-23 起 6 时点、990018 段建议放弃；子工单 07A–07F 草案 | 本板 WO-AM-07；scratch/wo_am07_attribution/ |
| 07-27 09:1x | Fable5 总控 | leader 任务书交付（批次监护+07B+07A 首批 29 只，/goal 可粘贴）；归因清单落 scratch/wo_am07_attribution/ | 会话交付 |
| 07-27 09:2x | 实现代理 | WO-AM-06 ✅：17 失败清零（12 过期断言+5 环境类、源码缺陷 0、skip 0），全树首绿 1090/0+48/0；词表勘误 65+62=127；总控复核 48/48+冻结集零改动 | 本板 WO-AM-06 |
| 07-27 09:01 | Fable5 总控（用户确认） | 第二轮提交推送 ✅：测试清零/备份脚本/文档三批，逐批 gitleaks 零泄漏，bb280ca→76112be 三方一致；34 文件源码集哈希未变，批次领取不受影响 | git log；GitHub API |
| 07-27 11:19 | /goal 接手总控 | 只读接管核验 ✅：批次 `batch_a50_d93b6b4bd5fc7704` 为 TRAINING=1/QUEUED=49/READY=0；000333、Slurm 577313 在 cu18 RUNNING（12 CPU，03:02:05/10:00:00）；Web `127.0.0.1:8765` 正常监听，未停止、重启或写队列 | SQLite `mode=ro`；squeue/sacct；本机监听端口 |
| 07-27 11:31 | /goal 接手总控+独立审查 | WO-AM-07F ✅：两份 CSV 独立重算通过，137/8、区间内 136 威胁日及 7 段安全区间确认；原始报告叙述勘误由本板覆盖。WO-AM-08 增量备份 ✅：1148 文件/3154.3 MB 同步并逐一核对大小，耗时 443 秒 | 本板 WO-AM-07F；`scratch/wo07f/` 哈希；`scripts/backup_am_assets.py` 输出 |
| 07-27 15:19 | /goal 接手总控 | WO-AM-04 首件训练成功但回传门禁失败关闭：000333/577313 完成 9000 步，05:52:48，合同/git/34 源文件正确；450 checkpoint+策略+历史=452 项，超过控制端 64 项硬上限。队列 NEEDS_ATTENTION=1/QUEUED=49/READY=0；未改状态、未重试 | 远端 result_manifest；`engine.py:986-987`、`train_slurm_worker.py:581-610`、`slurm_control.py:904-911`；SQLite mode=ro |
| 07-27 15:26 | /goal 接手总控 | WO-AM-08 增量备份 ✅：1,148 文件/3,154.3 MB 同步 HPC，远端逐一核对大小通过，耗时 281 秒；未把未过结果门禁的首件远端产物计入本机已验收资产 | `backup_patrol_20260727_1519.stdout.log`；巡检证据 |
| 07-27 16:03 | /goal 接手总控+三组证据核验 | 当时按错误日期裁切得到 22/30 与 8 点未决结论；**该点级结论已被 16:27 原生周期重算撤销**。75 个未解释因子切换与 v4 失败关闭状态仍有效 | 旧桥接仅保留追溯，不再作为裁决证据 |
| 07-27 16:27 | /goal 接手总控+原生周期复核 | 撤销 07A 的 `22/8` 与“000876 供应商冲突”结论：旧桥接先裁父日期再算收益。30/30 正确重算后为 19 个 qfq close 修复候选、6 个旧审计假阳性候选、3 个三方冲突、2 个父序列缺行；000876 实缺 24 个成交日，600803 缺 1 日。该增强路线后来被主线替代 | 本地增强支线现场，未进入公开主线提交 |
| 07-27 15:49 | /goal 接手总控 | 训练第 3 轮只读巡检无变化：批次/首件仍 NEEDS_ATTENTION、其余 49 件 QUEUED、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变，Web 仍监听；零运行态写入 | SQLite `mode=ro`；sacct；sha256sum；本机监听端口 |
| 07-27 17:18 | /goal 接手总控 | 训练第 4 轮巡检无状态变化：首件 NEEDS_ATTENTION、其余 49 件 QUEUED、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变，Web 仍监听。本机与备份端均为 1,148 项，无 READY 增量，距上轮不足 6 小时，未重复传输。18:10 追认：本轮 `HEAD /` 曾让 Web 追加 48 字节 405 日志；不影响业务状态，后续只查端口 | `scratch/goal_am_custody/training_patrol_20260727_1718.md` SHA-256 `d1d4c3fc…bab2`；SQLite `mode=ro`；sacct；远端 sha256sum |
| 07-27 17:44 | /goal 接手总控+独立审查 | 07A 增强型三道完成门曾实现收口，但正式批准账仍为空、v4 不存在；该路线后来被 07-29 主线替代，源码与证据不进入公开主线提交 | 本地增强支线现场，未发布 |
| 07-27 18:10 | /goal 接手总控 | 训练第 5 轮巡检无状态变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变，Web PID 45836 仍监听；两端均 1,148 项。巡检 `HEAD /` 会把 405 追加到 Web 错误日志，本轮本机日志较备份端多 96 字节（其中 48 字节由本轮触发）；无业务状态变化，后续改为只查端口 | `scratch/goal_am_custody/training_patrol_20260727_1810.md` SHA-256 `0eef27fe…e8d6`；SQLite `mode=ro`；sacct；远端 sha256sum |
| 07-27 18:44 | /goal 接手总控 | 训练第 6 轮巡检无状态变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变，Web PID 45836 仍监听；两端均 1,148 项。本轮只查 TCP 监听，未发送 HTTP、未写队列或传输备份；远端真实工作目录由 `sacct WorkDir` 精确绑定为 `AlphaMaster-runtime-v2` | `scratch/goal_am_custody/training_patrol_20260727_1844.md` SHA-256 `5a83f720…b810a`；SQLite `mode=ro`；sacct WorkDir；远端 sha256sum |
| 07-27 19:32 | /goal 接手总控 | 训练第 7 轮巡检仍无状态变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变，Web PID 45836 仍监听；本机/备份端均为 1,148 项，唯一差异仍为既有 96 字节 405 日志。距 15:26 不足 6 小时，未传输；全程只读、无 HTTP | `scratch/goal_am_custody/training_patrol_20260727_1932.md` SHA-256 `fde89f41…8b66`；SQLite `-readonly`+`query_only`；sacct WorkDir；远端 sha256sum |
| 07-28 16:51 | /goal 接手总控+独立审查 | 07A adjudication draft/finalize 固定树迁移通过：生产/测试全交叉隔离、四类输入租约、并发赢家保持及正式 sources 自包含闭合；P0/P1/P2=0/0/0，测试 34/0。共享 packages 仍拒绝，未运行任何生产构建 | `scratch/goal_am_custody/wo07a_adjudication_prepare_final_accept_20260728_1651.md` SHA-256 `d1f96833…589b2` |
| 07-28 17:42 | /goal 接手总控+独立审查 | 07A 共享 package 现有整树固定快照守卫通过：ASCII/casefold、固定哈希向量、全父链与整树 File ID、文件单次固定句柄读取、读后/终验及异常清理闭合；P0/P1/P2=0/0/0，测试 28/0。调用方 schema 与 builder/auditor 仍在门内 | `scratch/goal_am_custody/wo07a_package_snapshot_guard_final_accept_20260728_1742.md` SHA-256 `3e9fa046…fa911` |
| 07-28 18:14 | /goal 接手总控 | 训练第 22 轮只读巡检无变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变；8765/17911 均无监听，系统未重启，冻结路径无改动。距 15:04 完整备份约 3 小时 10 分钟，无 READY 增量，未重复传输 | `scratch/goal_am_custody/training_patrol_20260728_1814.md` SHA-256 `286f3902…f7faa`；SQLite `-readonly`+`query_only`；sacct WorkDir；远端 sha256sum |
| 07-28 18:23 | /goal 接手总控+独立审查 | 07A 共享包 auditor 复审 REJECT（P0/P1/P2=0/1/2）：本地 StockDB 的配置/进程/命令行/cwd/数据根/LevelDB manifest 路径链未闭合；跨 producer 重用 package ID 未拒绝；Tencent/local 完整 validator 攻击测试不足。既有测试 92/0，但新增对抗探针 0/2；已退回 auditor，并同步要求 builder 补同类门，仍禁止生产构建 | `scratch/goal_am_custody/wo07a_auditor_shared_packages_review_20260728_1823.md` SHA-256 `8dac441e…54018` |
| 07-28 18:26 | /goal 接手总控+独立审查 | 07A 共享包 contract、capture producer 与 RQ/Tencent producer 最终 ACCEPT：P0/P1/P2=0/0/0，四套离线测试 142/0、补充只读检查 5/0、编译 8/0。h5py 当前版本与精确 H5 dtype 在 production/TestEnvironment 双向失败关闭；99 点、六点和 600887 负例未漂移。只读收尾确认暂存区空、冻结路径干净、formal/v4 不存在，现场全保留 | `scratch/goal_am_custody/wo07a_package_contract_toolchain_final_accept_20260728_1826.md` SHA-256 `94806d12…7b13` |
| 07-28 18:58 | /goal 接手总控 | 训练第 23 轮只读巡检无变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变；8765/17911 均无监听，系统未重启，冻结路径无改动。距 15:04 完整备份约 3 小时 54 分钟，无 READY 增量，未重复传输 | `scratch/goal_am_custody/training_patrol_20260728_1858.md` SHA-256 `a018eeec…c680a`；SQLite `-readonly`+`query_only`；sacct WorkDir；远端 sha256sum |
| 07-28 19:00 | /goal 接手总控+独立审查 | 07A 共享包 builder 最终 ACCEPT：P0/P1/P2=0/0/0，目标测试 34/0、编译 3/0、AST 1/0、独立攻击探针 15/0。正式根整树租约、三包独立重算、全局 package ID/usage closure、H5 link/dtype、本地 StockDB 全身份链、网络守卫和输出根晚创建闭合；只读收尾确认 formal/v4 不存在，未运行生产构建 | `scratch/goal_am_custody/wo07a_builder_shared_packages_final_accept_20260728_1900.md` SHA-256 `e46b7647…b37cf` |
| 07-28 19:04 | /goal 接手总控+独立审查 | 07A 共享包 auditor 最终 ACCEPT，替代 18:23 REJECT：P0/P1/P2=0/0/0，shared 48/0、capture 59/0、编译 3/0，主控复跑一致。本地 StockDB 配置到 LevelDB 的完整身份链、跨 producer package ID 三层唯一性、usage 后租约终验、Tencent/local 完整攻击夹具、H5 offset/字段顺序和 AST 门均闭合；未运行生产 | `scratch/goal_am_custody/wo07a_auditor_shared_packages_final_accept_20260728_1904.md` SHA-256 `9f685810…abf8cb` |
| 07-28 19:42 | /goal 接手总控 | 训练第 24 轮只读巡检无变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变；8765/17911 均无监听，系统未重启，冻结路径无改动。距 15:04 完整备份约 4 小时 38 分钟，无 READY 增量，未重复传输 | `scratch/goal_am_custody/training_patrol_20260728_1942.md` SHA-256 `381cb168…719c0`；SQLite `-readonly`+`query_only`；sacct WorkDir；远端 sha256sum |
| 07-28 20:25 | /goal 接手总控 | 训练第 25 轮只读巡检无变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变；8765/17911 均无监听，系统未重启，冻结路径无改动。距 15:04 完整备份约 5 小时 21 分钟，无 READY 增量，未重复传输 | `scratch/goal_am_custody/training_patrol_20260728_2025.md` SHA-256 `0b423675…712b1`；SQLite `-readonly`+`query_only`；sacct WorkDir；远端 sha256sum |
| 07-28 20:54 | /goal 接手总控+独立审查 | 07A publication guard Windows 长路径修复最终 ACCEPT：生产等价 270 字符叶路径不再触发 WinError 206；内部扩展路径与公开逻辑路径隔离，外部 `\\?\`/`\\.\` 输入失败关闭。P0/P1/P2=0/0/0，直接测试 33/0、其它消费者 262/0、编译 2/0；正式/v4 仍不存在，未运行生产 | `scratch/goal_am_custody/wo07a_publish_guard_long_path_final_accept_20260728_2054.md` SHA-256 `bd2a88ed…0d4be` |
| 07-28 21:05 | /goal 接手总控+独立审查 | 07A `prepare_adjudications` 三类共享包接入最终 ACCEPT：P0/P1/P2=0/0/0，scoped 51/0、编译 2/0，主控同哈希复跑一致。Tencent 30 点/26 请求/19-6-3-2、RQ 独立 69 点 oracle、local 8 点、三层 TOCTOU、H5 dtype、三生产者 whole-package 发布及输出租约终验均闭合；未运行生产，formal/v4 不存在 | `scratch/goal_am_custody/wo07a_prepare_adjudications_shared_packages_final_accept_20260728_2105.md` SHA-256 `07cb40f9…f0d56` |
| 07-28 21:04 | /goal 接手总控 | 训练第 26 轮巡检及完整增量备份完成：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 与远端 manifest 均未变。1,148 文件/3,154,302,705 字节同步到固定 HPC 备份根，用时 294 秒；21:09 独立重查两端路径、总字节及逐文件大小完全一致，缺失/多余/差异均为 0 | `scratch/goal_am_custody/training_backup_20260728_2104.md` SHA-256 `76c66e0c…9884d` |
| 07-28 21:42 | /goal 接手总控 | 增强支线的生产抓取曾暴露 RQAlpha H5 日期语义缺口并完成消费者修复；该取证链后来不再作为主线 v4 前置门 | 本地增强支线现场，未发布 |
| 07-28 23:28 | /goal 接手总控 | 训练第 27 轮只读巡检无变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变；8765/17911 均无监听，冻结路径无改动。距 21:04 完整备份不足 2.5 小时，无 READY 增量，未重复传输 | `scratch/goal_am_custody/training_patrol_20260728_2328.md` SHA-256 `757c835a…f9b` |
| 07-29 00:10 | /goal 接手总控+独立审查 | 增强支线的父 v3 receipt 与生产证据包曾通过验收，但 99 点 review 仍为空、v4 不存在；该链后来不再作为主线前置门 | 本地增强支线现场，未发布 |
| 07-29 00:24 | /goal 接手总控 | 训练第 28 轮只读巡检仍无变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 COMPLETED/0:0，远端 manifest SHA 未变；8765/17911 均无监听，冻结路径无改动。距 21:04 完整备份约 3 小时 20 分钟，无 READY 增量，未重复传输 | `scratch/goal_am_custody/training_patrol_20260729_0024.md` SHA-256 `d08fe9f4…7861` |
| 07-29 00:37 | /goal 接手总控+独立审查 | 增强支线的 local 8 点静态预检发现 LevelDB workcopy 会恢复写入，随后拆分不可变源与一次性副本；既有服务未触碰。该链后来不再作为主线前置门 | 本地增强支线现场，未发布 |
| 07-29 00:50 | /goal 接手总控 | 训练第 29 轮只读巡检仍无变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 主作业/batch/extern 均 COMPLETED/0:0，远端 manifest SHA 未变；17911 无监听，冻结路径无改动。距 21:04 完整备份不足 4 小时，无 READY 增量，未重复传输 | `scratch/goal_am_custody/training_patrol_20260729_0050.md` SHA-256 `f37b3f57…9b24` |
| 07-29 01:25 | /goal 接手总控 | 训练第 30 轮只读巡检仍无变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 主作业/batch/extern 均 COMPLETED/0:0，远端 manifest SHA 未变；8765、17911 均无监听，冻结路径无改动。距 21:04 完整备份不足 6 小时，无 READY 增量，未重复传输 | `scratch/goal_am_custody/training_patrol_20260729_0125.md` SHA-256 `7986c315…defea` |
| 07-29 03:04 | /goal 接手总控 | 训练第 31 轮只读巡检及第 27 次完整增量备份完成：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313、远端 manifest、8765/17911 与冻结路径均未变化。同步 1,148 文件；独立重查两端均为 3,154,302,705 字节，缺失/多余/大小差异均为 0 | `scratch/goal_am_custody/training_patrol_20260729_0304.md` SHA-256 `0b4a60b2…0bc1e`；`training_backup_20260729_0304.md` SHA-256 `19952154…baf0` |
| 07-29 03:12 | /goal 接手总控+独立审查 | 增强支线的一次性副本工具链通过终审，但未运行正式 capture/finalize/builder/auditor/v4；该链后来不再作为主线前置门 | 本地增强支线现场，未发布 |
| 07-29 08:54 | /goal 接手总控 | 用户授权受控启动一次性 StockDB 副本；新 capture/RQ/Tencent/review 完成。新 capture manifest `9b1dbebb…dfa8`；RQ 69 点（31 内/38 外）；Tencent 30 点/26 请求、`19/6/3/2`；99 点裁定草案已生成。一次性副本 296 项、10,181,877,487 字节校验通过，但官方二进制全局单实例锁被既有 PID 24840 持有，17911 未启动；既有服务未触碰 | capture `wo07a-20260729T005459992563Z-10ff978b`；review packet SHA-256 `7f09040f…35ce`；manual template SHA-256 `ba3b9aa3…7c98e` |
| 07-29 09:12 | /goal 接手总控 | 训练第 32 轮只读巡检及第 28 次完整增量备份完成：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 与冻结路径未变化。同步 1,148 文件；独立重查两端均为 3,154,302,705 字节，缺失/多余/大小差异均为 0 | `scratch/goal_am_custody/training_patrol_20260729_0912.md` SHA-256 `0f1a9918…32f5`；`training_backup_20260729_0913.md` SHA-256 `4f1b5b89…206a` |
| 07-29 09:35 | /goal 接手总控 | 07A 99 点证据覆盖审计落盘：旧断点 30 点均只有非合格 Tencent+RQ 候选；因子 69 点中 RQ 阈值内 31、阈值外 38。阈值外包含 10 个仅按股票绑定官方文档的待核点和 28 个无合格 required role 候选的硬缺口。正式 decision 仍为 0/99，未运行正式链路 | `scratch/goal_am_custody/wo07a_evidence_coverage_audit/coverage_audit.json` SHA-256 `221037a9…7fb7`；CSV `f191af69…5e8d`；报告 `6e021911…be71` |
| 07-29 10:06 | /goal 接手总控 | 训练第 33 轮只读巡检无变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313 COMPLETED/0:0，远端 manifest 与冻结路径均未变化。距 09:13 完整备份不足 1 小时，无 READY 增量，未重复传输 | `scratch/goal_am_custody/training_patrol_20260729_1006.md` SHA-256 `15103405…c9f2` |
| 07-29 10:18 | /goal 接手总控 | 07A 第一方原件继续强攻：天齐锂业第一方 2018/2020 年报与港交所招股章程完整闭合 2017、2019 两次配股异常点；使用名义配股权而非事后认购率，倍率误差分别为 `3.93E-13`、`2.04E-13`。累计完整闭合 4 点、剩 23 个硬来源点，正式 decision 仍为 0/99 | `scratch/goal_am_custody/wo07a_official_docs_primary_batch3_tianqi/`；overlay SHA-256 `67c1a2b7…902d`；报告 `057437ce…f492` |
| 07-29 10:26 | /goal 接手总控 | 用户指出路线偏航后回到原始任务树干：停止扩展 99 点/逐公司公告支线，直接用冻结 capture 重建 26 只 qfq v4。正式输出 105 文件/5,448,565 字节；独立审计 26/26、continuity=0、OHLC=0；父 v3 哈希未变，WO-AM-07A 主线 PASS | `docs/evidence/wo_am07a_v4_mainline_accept_20260729.md`；v4 manifest `b4a58231…f0628` |
| 07-29 10:29 | /goal 接手总控 | 原任务书指定的正式入口 `scripts/wo07a_audit.py` 已对齐主线并实跑 PASS；随后在 GitHub 发布审查中彻底解耦增强支线，正式入口只加载独立主线审计器 | `AUDIT PASS repair_codes=26 continuity_violations=0 ohlc_violations=0`；v3 `e07fffd0…404c`；v4 `b4a58231…f0628` |
| 07-29 10:30 | /goal 接手总控 | 训练第 34 轮只读巡检无变化：NEEDS_ATTENTION=1、QUEUED=49、READY=0；577313、远端 manifest、runtime commit、冻结路径与端口均未变化；无 READY 增量，未重复备份 | `scratch/goal_am_custody/training_patrol_20260729_1030.md` |
| 07-29 11:00 | /goal 接手总控 | 用户授权的训练产物清单合同修复完成：34 个冻结源文件零改动；冻结集合外适配器真实校验原始 452 件后只发布 step 9000 checkpoint+策略+历史 3 件。000333/577313 训练、回测、信号、队列四层 READY；000617 已提交为 581389 RUNNING，队列 READY=1/TRAINING=1/QUEUED=48。首件 19 个持久资产逐文件异地哈希一致，队列 SQLite 在线快照远端 integrity=ok | `docs/evidence/wo_am04_training_manifest_contract_recovery_20260729.md`；固定测试 134/0 |
| 07-29 11:18 | /goal 接手总控+独立审查 | 训练产物合同两轮对抗修复后最终 ACCEPT：P0/P1/P2=0/0/1。客户端独立校验本地 adapter 固定 SHA 后把同一字节流交给远端 Python；远端副本不参与信任。openat/O_NOFOLLOW 全祖先固定、同句柄产物哈希和 manifest 同字节解析/哈希闭合。真实 452→3、固定测试 134/0；Web 热重启加载新客户端，581389 持续 RUNNING | `docs/evidence/wo_am04_training_manifest_contract_recovery_20260729.md`；adapter SHA `fdddd12b…73ce` |
| 07-29 11:40 | /goal 接手总控+独立审查 | WO-AM-08 在线 SQLite 备份合同修复完成：34 库在线快照纳入 WAL 已提交事务，主库转 DELETE 模式；每库以 0 字节 WAL/SHM 先行原子替换远端旧日志，再发布主库。唯一 staging、精确集合校验、管道失败关闭和目标祖先符号链接门闭合。最终 ACCEPT，P0/P1/P2=0/0/0；真实只读演练 1224 文件/3170.0 MB，未提前执行大传输 | `docs/evidence/wo_am08_live_sqlite_backup_contract_20260729.md`；脚本 SHA `607f2b5e…dae343` |
| 07-29 11:49 | /goal 接手总控 | 第 35 轮训练巡检：队列仍 READY=1/TRAINING=1/QUEUED=48，Web 正常。Slurm 控制器 `squeue` 查询超时，但 cu19 上 000617 的 Worker/训练进程均存活，约 11 核满载；训练历史推进到 step 1401/9000，best score 1.443386，slurm.err 为空。确认是控制器查询通道异常，不是训练停止；未干预作业 | `scratch/goal_am_custody/training_patrol_20260729_1149.md` SHA-256 `32d02531…91ff` |
| 07-29 11:51 | /goal 接手总控 | Slurm 查询通道自行恢复：`squeue` 再次返回 581389 RUNNING/cu19/57:50，Web `last_poll_error` 已清空；训练历史继续推进到 step 1452/9000。未做恢复操作 | squeue；本机 state；远端 training history |
| 07-29 11:52 | /goal 接手总控 | 剩余队列只读派发预检通过：50/50 训练文件存在且逐文件 SHA 与冻结值一致，总计 3,651,128 字节；50 个 ordinal、symbol、planned run ID 全部唯一；50/50 资源合同统一为 9000 步/12 CPU/32G/10h，问题 0 | `scratch/goal_am_custody/remaining_queue_preflight_20260729_1152.md` SHA-256 `11e66a55…8a18` |
| 07-29 12:00 | /goal 接手总控 | 000617 身份预验收通过：本机/远端 run manifest SHA `aff4bc2e…bf7` 一致，`.slurm_job.json` 绑定 581389；34/34 源文件与集合 SHA、训练 Parquet 本机/远端 SHA、runtime commit、资源及评分合同全部一致，问题 0。训练仍在运行，未提前验结果 | `scratch/goal_am_custody/000617_identity_preaccept_20260729_1200.md` SHA-256 `f40ff77f…c387` |
| 07-29 12:12 | /goal 接手总控 | 第 36 轮训练巡检：10 分钟内 20/20 次状态采样均 RUNNING，轮询错误和 run error 均为 0；581389 在 cu19 运行 01:19:49，000617 推进到 step 1992/9000，best score 1.443386，slurm.err 为空 | `scratch/goal_am_custody/training_patrol_20260729_1212.md` SHA-256 `ab9fdce5…0f63` |
| 07-29 12:13 | /goal 接手总控 | 剩余 48 个 planned run ID 目录冲突预检通过：本机同名 run 目录 0；服务器现有 37 个 run 目录中同名 0。后续自动领取不存在既有目录碰撞 | `scratch/goal_am_custody/queued_run_collision_preflight_20260729_1213.md` SHA-256 `30e23408…e767` |
| 07-29 12:14 | /goal 接手总控 | 50 份训练 Parquet 语义预检通过：列结构 50/50 一致，行数 859–2321，时间严格递增；50/50 统一截止 2025-07-23 07:00 UTC，未混入 2025-07-24 起 sealed OOS，问题 0 | `scratch/goal_am_custody/training_parquet_semantics_preflight_20260729_1214.md` SHA-256 `aeb7f100…e181` |
| 07-29 12:15 | /goal 接手总控 | 50 份训练 Parquet 数值预检通过：107,665 行，有限值 250/250、OHLC 正价格 200/200、成交量非负 50/50、OHLC 高低关系 100/100，问题 0 | `scratch/goal_am_custody/training_parquet_numeric_preflight_20260729_1215.md` SHA-256 `74b55816…c5bf` |
| 07-29 12:32 | /goal 接手总控 | 第 37 轮训练巡检：15 分钟 30/30 次采样均 RUNNING，轮询错误/run error 0；581389 在 cu19 运行 01:39:45，000617 到 step 2481/9000，best score 1.443386，slurm.err 为空；队列 READY=1/TRAINING=1/QUEUED=48 | `scratch/goal_am_custody/training_patrol_20260729_1232.md` SHA-256 `295301f3…4b44` |
| 07-29 13:00 | /goal 接手总控 | 000617/581389 于 12:50:09 因 cu19 失联进入 NODE_FAIL；01:57:19、ExitCode=0:0，节点随后重启。队列正确失败关闭为 READY=1/NEEDS_ATTENTION=1/QUEUED=48，未派发下一件。138 个 checkpoint 完整保留；step 2760 checkpoint（5,965,894 字节，SHA `5da8a7ba…59ff`）已独立安全加载，训练身份和优化器状态完整。现有控制面没有终态 job 的安全续训路径；已冻结“新物理 run/job + 父 checkpoint 显式谱系 + from_scratch=false + 9000 总步数”的一次性恢复合同，用户授权前不重提、不写队列 | `docs/evidence/wo_am04_000617_node_fail_checkpoint_recovery_20260729.md` |
| 07-29 13:27 | /goal 接手总控+独立审查 | 冻结集合外的一次性 NODE_FAIL 恢复控制面完成：逻辑/物理 run 分离、一次性恢复事务、父谱系、远端真实 NODE_FAIL 复核、同句柄 checkpoint 哈希+语义验收、每次 SUBMITTING 幂等重验后才 sbatch。固定测试 146/0；独立审查 ACCEPT，P0/P1/P2=0/0/1；importer actual/pin=`c69f00e9…d3224`，34 冻结源码仍 `0e5ca8c3…64ab0`。未迁移生产 DB、未重启 Web、未导入/提交恢复作业 | `docs/evidence/wo_am04_000617_node_fail_checkpoint_recovery_20260729.md` |
| 07-29 13:32 | /goal 接手总控 | 对生产 SQLite 在线一致性副本完成恢复同构演练：迁移后旧 50 行与 batch 逐列不变，execution backfill=50/50、attempt=0=50/50、唯一索引存在；恢复事务后 immutable digest 仍 `4bc08daf…2735`，其余 49 项旧列精确不变，状态成为 READY=1/DISPATCHING=1/QUEUED=48、batch=ACTIVE，000617 父谱系及 step 2760 字节身份全对，integrity=ok。随后反查生产库仍是旧 schema、READY=1/NEEDS_ATTENTION=1/QUEUED=48、完整行摘要 `c4f63cab…2cd8`；演练 run 不在生产，Web PID 68008 未变 | `scratch/goal_am_custody/node_fail_recovery_rehearsal_20260729T0530Z/`；同一正式证据 |
| 07-29 13:42 | /goal 接手总控+独立审查 | 生产只读实探发现 frozen status 路径被终态 job 的 `squeue Invalid job id` 阻断，但 compute-node-11/12/13 的 `sacct` 均唯一返回 581389=`NODE_FAIL/0:0`。导入器改用固定 argv `sacct` 主行验签，并与冻结 binding、owner、job name、base state 交叉核对；duplicate/仅 step/wrong owner/wrong name/12 字段全部失败关闭。最终指定测试 152/0，独立审查 ACCEPT、P0/P1/P2=0/0/0；importer actual/pin=`de851b77…0e06`，34 冻结源码不变。全程只读 Slurm，未创建 run/job | `docs/evidence/wo_am04_000617_node_fail_checkpoint_recovery_20260729.md` |
| 07-30 01:44 | /goal 接手总控+独立审查 | 000725/581904 完成 9000 步并通过训练、回测、信号、队列四层 READY；远端 452 件经固定适配器归一化为 3 件，run/job/runtime/34 源文件/scoring 全对，合同验收 25/0（含远端原始 452 件逐件复验）。首次完整版本生产迁移完成：1,268 文件/3,202.5 MB/201 秒，当前 generation 1,270 文件；36/36 SQLite integrity=ok、DELETE、WAL/SHM=0，17 个普通持久文件逐 SHA 一致，signal SQLite 逻辑行一致，备份验收 8/0、对抗复审 ACCEPT、P0/P1/P2=0/0/0。队列推进为 READY=3/TRAINING=1/QUEUED=46，000792/582419 在 cu05 RUNNING | `docs/evidence/wo_am04_000725_ready_backup_20260730.md` |
| 07-30 01:54～08:38 | /goal 接手总控 | 000792 连续 12 次只读巡检从 step 1389 推进至 8628/9000，best score 由 1.983448 提高到 2.024587；队列始终 READY=3/TRAINING=1/QUEUED=46、NEEDS_ATTENTION=0，SQLite integrity=ok，Web 回环监听、runtime commit、scoring contract 与冻结源码 SHA 均不变。无新 READY、失败或源码漂移，未重复备份、未干预作业；活动 PID 不进入公开快照，逐次证据只保留在本机 scratch | 本机 `scratch/goal_am_custody/training_patrol_20260730_*.md` |
| 07-30 09:15 | /goal 接手总控+独立审查 | 000792/582419 完成 9000 步并通过训练、回测、信号、队列四层 READY；远端 452 件经固定适配器归一化为 3 件，run/job/runtime/34 源文件/scoring 全对，合同验收 25/0。增量备份同步 1,288 文件/3,218.5 MB/221 秒，新 generation 1,290 文件；37/37 SQLite integrity=ok、DELETE、WAL/SHM=0，17 个普通持久文件逐 SHA 一致，signal SQLite 逻辑行一致，备份验收 8/0；对抗复审 ACCEPT、P0/P1/P2=0/0/0。队列推进为 READY=4/TRAINING=1/QUEUED=45，000988 在 cu05 RUNNING；活动 run/job 身份不进入公开快照 | `docs/evidence/wo_am04_000792_ready_backup_20260730.md` SHA-256 `55aeba85…628202c6` |
| 07-30 09:38～15:54 | /goal 接手总控 | 000988 连续 12 次只读巡检从 step 633 推进至 7527/9000，best score 1.574117；队列始终 READY=4/TRAINING=1/QUEUED=45、NEEDS_ATTENTION=0，SQLite integrity=ok，Web 回环监听、runtime commit、scoring contract 与冻结源码 SHA 均不变。无新 READY、失败或源码漂移，未重复备份、未干预作业；活动 run/job/PID 不进入公开快照，逐次证据只保留在本机 scratch | 本机 `scratch/goal_am_custody/training_patrol_20260730_*.md` |

## 8. 用户侧待办（只有用户本人能做）

1. **WO-AM-02+03 已完成**：七批提交、公开仓推送和服务器 runtime-v2 重部署均已于 2026-07-27 完成，无用户侧待办。
2. **WO-AM-04 checkpoint 恢复已完成**：用户已授权并完成 000617 的一次性恢复；581519 完整运行到 9000 步，训练、回测、虚拟信号和队列四层均 READY。无需再次授权或重提。
3. **WO-AM-07A 已完成**：无需停止既有 PID 24840；99 点、local 8 点和逐公司公告搜证仅在未来明确要做增强型来源研究时再单独立项。
4. **WO-AM-05 揭盲授权**（远期，届时单独问）。
5. **Norgate/Sharadar 美股数据采购拍板**（管线验证后才问，见 MULTI_MARKET_ROADMAP D5）。
6. PA_Agent 专属会话启动卡片（如尚未点击）。

## 9. 状态更新纪律

每个工单状态变化→更新对应"状态"栏+第 3 节证据表+第 7 节进度账本+`CONTEXT.md` 顶部段；重大新事实（数据/服务器/额度）记入对应工单"状态"栏。**额度紧张时**：优先主循环顺序执行，把并行对抗审查降级为按第 5 节模板的书面审查（逐向量自查+留痕），并在账本注明"降级审查，待额度恢复补独立审查"。用户可见收口三段式：已完成/未完成/有风险待确认 + 下一步。

---
更新日志：
- 2026-07-26 深夜 初版（Claude Fable 5 总控会话）。
- 2026-07-27 凌晨 增补 6–9 节（与 PA_Agent 的 `docs/WORKORDER_MASTER_20260727.md` 结构对齐）；远程核验证据入 WO-AM-02。当前推进位置：WO-AM-01 待复跑（02:47 自动续跑已排）；WO-AM-02 清单草案待冻结。
