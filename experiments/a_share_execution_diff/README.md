# A 股执行差分实验

本目录只做隔离研究，不接入 AlphaMaster 生产链，也不增加项目运行时依赖。

## 固定版本

- AlphaMaster：运行时当前 Git `HEAD`
- AKQuant：PyPI `0.3.20`；对应源码测试快照
  `30054523fb905adb1c3f250749e1b5ff61cf8452`
- RQAlpha：源码快照
  `3503ab57932540cd36bf8375134e52c6923bf0d2`，安装后版本 `6.3.0`

第三方源码和虚拟环境只能放在 `scratch/`。RQAlpha 仅按其许可证用于个人
非商业研究。

## 当前样本

`fixture.json` 固定一笔 100 股、10 元买入和卖出：

- 初始现金：100,000 元
- 佣金：0.03%
- 最低佣金：5 元
- 卖出印花税：0.05%
- 过户费：0.001%
- 滑点：0

手算结果是买入费用 5.01 元、卖出费用 5.51 元、期末现金
99,989.48 元。

## 推荐运行方式

从项目根目录运行总入口。它要求输出目录不存在，顺序执行 6 个样本生成器/
源码检查器和 2 个比较器，并记录每条命令、每个输出文件和各虚拟环境
已安装包清单的 SHA-256。这样旧 JSON 不能冒充本轮结果。

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

- AlphaMaster 与 AKQuant 在这个固定样本上比较终态成交与费用数值。
- AKQuant、RQAlpha 的版本、固定 Git 提交和干净源码树都会失败关闭；
  RQAlpha 还逐字节核对实际导入的费用器源码与固定快照。
- AKQuant 使用 `same_cycle` 收盘成交，只用于对齐执行事件；这不是
  AlphaMaster 的 `t 日信号 → t+1 开盘成交` 时钟证明。
- RQAlpha 当前只隔离调用默认股票费用器，不把模块测试冒充完整回测。
- RQAlpha 的基础佣金率通过倍率对齐到本样本的 0.03%；对齐后，默认股票
  费用实现仍比带过户费的固定口径少 0.02 元。
- 当前只有“100 股、10 元、触发最低佣金、零滑点、金额落在整分”的单个
  样本，不能外推到大额佣金、非整分舍入、滑点或按日期变化的费用政策。
- 当前结果不能证明三个框架的订单生命周期、公司行动和恢复语义等价。
