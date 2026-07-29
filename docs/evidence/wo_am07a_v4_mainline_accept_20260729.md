# WO-AM-07A v4 主线验收

- 时间：2026-07-29 10:26（Asia/Shanghai）
- 状态：PASS
- 输出：`G:\QuantData\free-stockdb\am_exports\20260727_csi300_qfq_repair_v4`
- 输出文件：105
- 输出总字节：5,448,565
- v4 manifest SHA-256：`b4a582318e4c2f94bbd10feec99894cb82b4077ab0f691ce83450f3eeebf0628`
- v3 manifest SHA-256：`e07fffd04c9d53a897ae688ad05897a03273acf14010f799e1aca85579a8404c`

## 主线范围

从 2026-01-30 至 2026-06-30 六个沪深 300 成分时点推导出 29 只
`non-available` 股票；001280、001391、600930 按 WO-AM-07B 处理，
其余 26 只进入复权修复。回放史实起点按原任务书固定为 2006-10-31，
数据截止日与父 v3 一致为 2026-07-24。

## 构建与审计原文

```text
PRECHECK PASS repair_codes=26 continuity_violations=0 ohlc_violations=0
BUILD PASS output=G:\QuantData\free-stockdb\am_exports\20260727_csi300_qfq_repair_v4 repair_codes=26 files=105 continuity_violations=0
V3_MANIFEST_SHA256=e07fffd04c9d53a897ae688ad05897a03273acf14010f799e1aca85579a8404c
AUDIT PASS repair_codes=26 continuity_violations=0 ohlc_violations=0 v3_manifest_sha256=e07fffd04c9d53a897ae688ad05897a03273acf14010f799e1aca85579a8404c v4_manifest_sha256=b4a582318e4c2f94bbd10feec99894cb82b4077ab0f691ce83450f3eeebf0628
```

构建器：`scripts/wo07a_build_v4_mainline.py`，只依赖最小主线计算核心
`scripts/wo07a_mainline_core.py`。它提供 `--v3`、`--capture`、`--output`
和 `--check-only`，不再动态加载 99 点、公司公告或 StockDB 支线。

正式验收入口：`scripts/wo07a_audit.py`，调用只读审计器
`scripts/wo07a_audit_mainline.py`。审计器不导入构建器或主线计算核心，
使用独立的 `merge_asof` 因子对齐方式重算 qfq，逐文件核对精确 105 件产物、
父 v3 与 v4 manifest 哈希、D1 列/类型/值、逐只审计记录和研究语义。
`--v3` 与 `--v4` 参数会真实传入审计器。

GitHub 发布前的解耦回归：仓内测试 4 通过、0 失败；生产 `--check-only`
与正式审计均再次通过。被替代的增强审计源码按同字节 SHA-256
`50c7ca64efa5c21d41fde7a4968b034410719fa2bf1854428c02a99ad45aff78`
保留在项目 `scratch/`，不进入主线提交。

| 正式源码 | SHA-256 |
|---|---|
| `scripts/wo07a_mainline_core.py` | `521a4c66fb3d03dcf809cebdd991b5014a28a817036bb5fd82d9b882f34e1275` |
| `scripts/wo07a_build_v4_mainline.py` | `ddff57bb8f0f4cc9a7dc5a46777257e6127dd5e7de73b1d906bc4263bfbeb0f5` |
| `scripts/wo07a_audit_mainline.py` | `70d3cb2adfe1ad9802d5a4e3f1755c1eda98831b0453ee37e28dd0a352cafcac` |
| `scripts/wo07a_audit.py` | `f3b1d26d6499b9a66ab4228e2e9651416be3970f472bee92d7939e13e8246385` |

## 路线纠偏

此前新增的 99 点裁决体系、StockDB local 8 点包和逐公司公告搜证，属于超出
原始 `/goal` 完成条件的增强型诊断支线。它们不进入当前主线源码闭包，也不再
作为 v4 的阻塞门。v4 只称回溯性前复权修复数据，不具备 PIT 或 sealed OOS
资格。
