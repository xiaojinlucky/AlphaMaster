# A 股研究层差分实验

本目录只做隔离研究，不接入 AlphaMaster 生产链，也不增加项目运行时依赖。

## 固定版本

- AlphaMaster：运行时当前 Git `HEAD`
- Qlib：
  - Windows 预编译发布版 `0.9.7`
  - 最新源码快照 `79633dd9506ea689e5400dea0197717b5b3d74b7`
- vn.py：源码快照
  `1b78494979deb4c4996f6b864f234d9839f2f239`，源码版本 `4.4.0`

第三方源码和隔离虚拟环境只能放在 `scratch/`。Qlib 和 vn.py 均为
MIT 许可证。

## 固定样本

`fixture.json` 同时固定两个合同：

1. 信号时钟：`signal[t]` 在 `open[t+1]` 执行，收益区间截至
   `open[t+2]`；同时用反例检查 IC 是否与这段收益同钟。
2. 历史成分：两只股票各自只在指定的起止日期区间内有效，起止日期都
   包含在结果中。

## 环境边界

正式探针只需要下列直接依赖，且必须装在彼此隔离的 Python 3.11 虚拟
环境：

- AKQuant：`akquant==0.3.20`
- RQAlpha：固定提交
  `3503ab57932540cd36bf8375134e52c6923bf0d2`
- Qlib 源码合同：`pyqlib==0.9.7 --no-deps`；不导入 Qlib 运行时
- vn.py 历史成分探针：`polars==1.34.0`、`pandas==2.2.3`、
  `tqdm==4.69.1`、`loguru==0.7.3`

总入口通过 Python 标准库读取每个环境的完整已安装包清单，并保存清单的
SHA-256，足以判断是否复用了同一份本机环境。当前没有保存 Python 安装包
本身的哈希，因此这不是跨机器的字节级供应链复现声明。

首次搭建隔离目录可执行：

```powershell
$root = "scratch/third_party_diff_20260724"

git clone https://github.com/akfamily/akquant.git "$root/akquant"
git -C "$root/akquant" checkout --detach `
  30054523fb905adb1c3f250749e1b5ff61cf8452
git clone https://github.com/ricequant/rqalpha.git "$root/rqalpha"
git -C "$root/rqalpha" checkout --detach `
  3503ab57932540cd36bf8375134e52c6923bf0d2
git clone https://github.com/microsoft/qlib.git "$root/qlib"
git -C "$root/qlib" checkout --detach `
  79633dd9506ea689e5400dea0197717b5b3d74b7
git clone https://github.com/vnpy/vnpy.git "$root/vnpy"
git -C "$root/vnpy" checkout --detach `
  1b78494979deb4c4996f6b864f234d9839f2f239

& ".venv/Scripts/python.exe" -m venv "$root/venv_akquant"
& "$root/venv_akquant/Scripts/python.exe" -m pip install `
  akquant==0.3.20
& ".venv/Scripts/python.exe" -m venv "$root/venv_rqalpha"
& "$root/venv_rqalpha/Scripts/python.exe" -m pip install `
  "$root/rqalpha"
& ".venv/Scripts/python.exe" -m venv "$root/venv_qlib_contract"
& "$root/venv_qlib_contract/Scripts/python.exe" -m pip install `
  --no-deps pyqlib==0.9.7
& ".venv/Scripts/python.exe" -m venv "$root/venv_vnpy"
& "$root/venv_vnpy/Scripts/python.exe" -m pip install `
  polars==1.34.0 pandas==2.2.3 tqdm==4.69.1 loguru==0.7.3
```

这些命令固定直接依赖和源码提交，但传递依赖仍由当次 Python 包索引解析；
总入口记录最终环境身份，发现清单变化时必须重新审查，不能静默沿用结论。

## 推荐运行方式

从项目根目录运行总入口，输出必须写到全新 `scratch/` 目录，不能提交：

```powershell
$runId = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$out = "scratch/third_party_diff_20260724/runs/$runId"

& ".venv/Scripts/python.exe" "experiments/run_a_share_diff_suite.py" `
  --output-dir $out `
  --akquant-python "scratch/third_party_diff_20260724/venv_akquant/Scripts/python.exe" `
  --akquant-source "scratch/third_party_diff_20260724/akquant" `
  --rqalpha-python "scratch/third_party_diff_20260724/venv_rqalpha/Scripts/python.exe" `
  --rqalpha-source "scratch/third_party_diff_20260724/rqalpha" `
  --qlib-python "scratch/third_party_diff_20260724/venv_qlib_contract/Scripts/python.exe" `
  --qlib-source "scratch/third_party_diff_20260724/qlib" `
  --vnpy-python "scratch/third_party_diff_20260724/venv_vnpy/Scripts/python.exe" `
  --vnpy-source "scratch/third_party_diff_20260724/vnpy"
```

## 证据边界

- AlphaMaster 直接执行真实收益目标构造函数，证明
  `target_ret[t] = log(open[t+2] / open[t+1])`。PnL 乘法和下一开盘
  执行分别来自源码检查与另一组固定执行样本；本轮没有执行一条把三者串
  起来的端到端生产回测，不能称“完整 PnL 时钟已运行验证”。
- 当前反例稳定复现：`AlphaEngine._compute_ic` 额外右移一根，而且把收益
  构造函数末尾补的两个零纳入 IC。相同额外右移还存在于
  `MT5Backtest._ts_ic_stability`。此外，PnL 与换手成本路径也未裁掉末尾
  两个无可实现未来收益的补零；模块级
  `model_core.evaluator.score_all(..., horizon=1)` 和
  `EffectivenessEvaluator(target_horizon=1)` 默认只裁 1 根，而当前收益
  目标需要裁 2 根，`prune_features.py` 的模块级调用也没有显式传 2。
  四处都列为生产评分修复范围，本实验不改生产代码。
- Qlib 0.9.7 及最新源码快照都检查真实策略源码的语法树，同时核对交易
  日历用 `当前步 - shift` 计算索引，证明 `shift=1` 确实表示更早一根。
  这里只能证明所检查方法的源码合同，不能证明完整回测、热身边界、多
  频率/时区或预测真正可用时点。
- vn.py 直接执行固定提交中的原始 `AlphaDataset.prepare_data`。只对
  本方法不调用的 Alphalens 报告函数做导入占位，不改成分筛选代码。
- vn.py 的正常单区间样本通过，但对抗样本同时复现了三个接入阻断点：
  区间重叠会重复行、空字典会绕过过滤、所有区间为空会抛出
  `ValueError("cannot concat empty list")`。生产实现必须先合并区间、
  强制 `(交易日, 股票)` 唯一、空输入失败关闭，并把带时分秒的行情统一
  映射成交易日后再匹配，不能直接复制该方法。
- 结果不能证明 Qlib、vn.py 与 AlphaMaster 的完整训练、数据或回测语义
  等价。
