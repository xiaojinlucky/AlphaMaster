"""
run_backtest.py — 多因子组合回测（含手续费/滑点、夏普、资金曲线）

训练/回测一律使用本地 Parquet，不连接 MT5 在线拉数。

用法：
    python run_backtest.py --strategy-file strategies/best_ADAUSD.json --data-file D:\\K线数据\\ADAUSD_H1.parquet
    python run_backtest.py --strategy-file path\\to\\strategy.json
        # 若策略 JSON 内含 data_file 字段，可省略 --data-file
    python run_backtest.py --commission 0.02 --slippage 0.01
        # 单边手续费/滑点（单位 %），默认 0.02 / 0.01
"""

import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.parquet_manager import ParquetDataManager
from data_pipeline.dataset_contracts import source_family
from backtest_viz import BacktestEngine
from model_core.target_contract import SCORING_CONTRACT_VERSION
from model_core.vocab import (
    FORMULA_VOCAB,
    VOCAB_VERSION,
    VocabVersionMismatchError,
)
from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless
from evaluation.sealed_oos_campaign import (
    SEALED_REPORT_FORMAT,
    normalize_cost_policy,
)

_H1_PER_YEAR = 6240
DEFAULT_COMMISSION_PCT = 0.02  # 单边手续费 %
DEFAULT_SLIPPAGE_PCT = 0.01    # 单边滑点 %
_SEALED_REPORT_FIELDS = (
    "format",
    "symbol",
    "data_sha256",
    "strategy_sha256",
    "evaluation_mode",
    "test_start",
    "test_end",
    "commission_pct",
    "slippage_pct",
    "cost_rate",
    "sharpe",
)


def decode_formula(tokens: list[int]) -> str:
    names = FORMULA_VOCAB.token_names
    return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in tokens)


def load_strategy(path: Path, *, raw_bytes: bytes | None = None) -> dict | None:
    if raw_bytes is None:
        if not path.exists():
            return None
        raw_bytes = path.read_bytes()
    data = json.loads(raw_bytes.decode("utf-8"))
    if isinstance(data, list):
        raise VocabVersionMismatchError(
            f"策略 {path} 是无兼容版本的旧格式；需重新训练/重建后回测"
        )
    if not isinstance(data, dict):
        raise ValueError(f"策略 {path} 顶层必须是 JSON 对象")
    artifact_version = data.get("vocab_version")
    if artifact_version is None:
        raise VocabVersionMismatchError(
            f"策略 {path} 缺少公式兼容版本；需重新训练/重建后回测"
        )
    FORMULA_VOCAB.verify(artifact_version)
    if data.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
        raise ValueError(f"策略 {path} 的评分合同不兼容，需重新训练")
    return data


def _utc_seconds(value: str, field: str) -> int:
    if not isinstance(value, str) or not value:
        raise ValueError(f"策略缺少训练数据范围字段: {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"策略训练数据范围字段 {field} 不是合法 ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"策略训练数据范围字段 {field} 必须包含时区")
    return int(parsed.astimezone(timezone.utc).timestamp())


def _identity_payload_from_strategy(strategy: dict) -> dict:
    return {
        "symbol": strategy["symbol"],
        "timeframe": strategy["timeframe"],
        "local_source": strategy["local_source"],
        "source_family": source_family(strategy["local_source"]),
        "periods_per_year": strategy["periods_per_year"],
        "minimum_bars": strategy["minimum_bars"],
        "dataset_id": strategy["dataset_id"],
        "data_sha256": strategy["data_sha256"],
        "data_rows": strategy.get("data_rows"),
        "data_start": strategy.get("data_start"),
        "data_end": strategy.get("data_end"),
        "columns": strategy.get("columns"),
    }


def _identity_payload_from_manager(manager: ParquetDataManager) -> dict:
    return {
        "symbol": manager.symbol,
        "timeframe": manager.timeframe,
        "local_source": manager.source,
        "source_family": source_family(manager.source),
        "periods_per_year": manager.periods_per_year,
        "minimum_bars": manager.minimum_bars,
        "dataset_id": manager.dataset_id,
        "data_sha256": manager.data_sha256,
        "data_rows": manager.data_rows,
        "data_start": manager.data_start,
        "data_end": manager.data_end,
        "columns": list(manager.columns),
    }


def _validate_strategy_data_contract(
    strategy: dict,
    manager: ParquetDataManager,
    *,
    evaluation_mode: str = "auto",
    score_start: str | None = None,
) -> dict:
    """验证训练身份与评估身份，并返回可审计评分区间。"""
    if evaluation_mode not in {
        "auto",
        "replay",
        "out_of_sample",
        "diagnostic_overlap",
        "sealed_oos",
    }:
        raise ValueError("evaluation_mode 不受支持")
    if evaluation_mode == "sealed_oos" and not score_start:
        raise ValueError("sealed_oos 必须显式提供 score_start")
    required = {
        "symbol": str,
        "timeframe": str,
        "local_source": str,
        "periods_per_year": int,
        "minimum_bars": int,
        "data_sha256": str,
        "dataset_id": str,
        "data_rows": int,
        "data_start": str,
        "data_end": str,
    }
    for field, expected_type in required.items():
        if field not in strategy:
            raise ValueError(f"策略缺少训练数据身份字段: {field}")
        actual = strategy[field]
        if expected_type is int:
            if isinstance(actual, bool) or not isinstance(actual, int):
                raise ValueError(f"策略训练数据身份字段 {field} 必须是整数")
        elif not isinstance(actual, str) or not actual:
            raise ValueError(f"策略训练数据身份字段 {field} 必须是非空字符串")
    columns = strategy.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("策略训练数据身份字段 columns 必须是无重复字符串列表")
    if strategy["data_rows"] < strategy["minimum_bars"]:
        raise ValueError("策略 data_rows 小于 minimum_bars")
    training_start = _utc_seconds(strategy["data_start"], "data_start")
    training_end = _utc_seconds(strategy["data_end"], "data_end")
    if training_start >= training_end:
        raise ValueError("策略训练数据范围必须满足 data_start < data_end")

    digest = strategy["data_sha256"]
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("策略 data_sha256 必须是 64 位小写十六进制")
    if strategy["dataset_id"] != f"sha256:{digest}":
        raise ValueError("策略 dataset_id 与 data_sha256 不一致")
    if manager.dataset_id != f"sha256:{manager.data_sha256}":
        raise ValueError("评估数据 dataset_id 与 data_sha256 不一致")

    for field, value in (
        ("symbol", manager.symbol),
        ("timeframe", manager.timeframe),
    ):
        if strategy[field] != value:
            raise ValueError(
                f"策略与评估数据的 {field} 不兼容: "
                f"{strategy[field]!r} != {value!r}"
            )
    if source_family(strategy["local_source"]) != source_family(manager.source):
        raise ValueError(
            "策略与评估数据的来源族不兼容: "
            f"{strategy['local_source']!r} != {manager.source!r}"
        )

    if columns != list(manager.columns):
        raise ValueError("策略与评估数据的 columns 不兼容")

    same_dataset = digest == manager.data_sha256
    times = manager.raw_dict["time"][0].detach().cpu().numpy()
    if same_dataset:
        for field, value in (
            ("local_source", manager.source),
            ("periods_per_year", manager.periods_per_year),
            ("minimum_bars", manager.minimum_bars),
            ("data_rows", manager.data_rows),
            ("data_start", manager.data_start),
            ("data_end", manager.data_end),
        ):
            if strategy[field] != value:
                raise ValueError(
                    f"同一数据 hash 的 {field} 与评估 loader 不一致: "
                    f"{strategy[field]!r} != {value!r}"
                )
        if evaluation_mode in {"out_of_sample", "sealed_oos"}:
            raise ValueError("样本外回测要求评估数据 hash 与训练数据不同")
        if score_start:
            raise ValueError("训练集重放不接受 score_start")
        resolved_mode = "replay"
        score_start_index = 0
    else:
        if evaluation_mode == "replay":
            raise ValueError("训练集重放要求评估数据 hash 与训练数据完全相同")
        if evaluation_mode == "diagnostic_overlap":
            resolved_mode = "diagnostic_overlap"
            if score_start:
                score_start_seconds = _utc_seconds(score_start, "score_start")
                score_start_index = int(times.searchsorted(score_start_seconds, side="left"))
            else:
                score_start_index = 0
        else:
            score_start_seconds = (
                _utc_seconds(score_start, "score_start")
                if score_start
                else training_end + 1
            )
            if score_start_seconds <= training_end:
                raise ValueError("样本外评分起点必须晚于训练数据结束时间")
            score_start_index = int(times.searchsorted(score_start_seconds, side="left"))
            resolved_mode = (
                "sealed_oos"
                if evaluation_mode == "sealed_oos"
                else "out_of_sample"
            )
        if score_start_index > len(times) - 3:
            raise ValueError("评估数据没有足够的可评分样本")

    training_identity = _identity_payload_from_strategy(strategy)
    evaluation_identity = _identity_payload_from_manager(manager)
    return {
        "evaluation_mode": resolved_mode,
        "same_dataset": same_dataset,
        "score_start_index": score_start_index,
        "score_start": datetime.fromtimestamp(
            int(times[score_start_index]), tz=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "score_end": datetime.fromtimestamp(
            int(times[-1]), tz=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "warmup_bars": score_start_index,
        "training_data": training_identity,
        "evaluation_data": evaluation_identity,
        "annualization": {
            "basis": "evaluation_data",
            "training_periods_per_year": strategy["periods_per_year"],
            "evaluation_periods_per_year": manager.periods_per_year,
            "same_periods_per_year": (
                strategy["periods_per_year"] == manager.periods_per_year
            ),
        },
    }


def _validate_sealed_report_cli(
    *,
    evaluation_mode: str,
    strategy_file: str | None,
    sealed_report: str | None,
    single_mode: bool,
) -> Path | None:
    """校验封存报告只能由显式单策略 sealed_oos 回测生成。"""
    sealed_mode = evaluation_mode == "sealed_oos"
    if sealed_mode and not sealed_report:
        raise ValueError("sealed_oos 必须显式提供 --sealed-report")
    if sealed_report and not sealed_mode:
        raise ValueError("--sealed-report 仅允许与 --evaluation-mode sealed_oos 同时使用")
    if not sealed_mode:
        return None
    if not strategy_file:
        raise ValueError("sealed_oos 必须通过 --strategy-file 指定单个策略文件")
    if single_mode:
        raise ValueError("sealed_oos 不接受 --single，只接受 --strategy-file 单策略模式")
    if not isinstance(sealed_report, str) or not sealed_report.strip():
        raise ValueError("--sealed-report 必须是非空路径")
    return Path(sealed_report)


def _build_sealed_report_payload(
    *,
    results_map: dict,
    evaluation_contract: dict,
    data_sha256: str,
    strategy_bytes: bytes,
    commission_pct: float,
    slippage_pct: float,
) -> dict:
    """生成封存评估器读取的严格带成本身份报告。"""
    if evaluation_contract.get("evaluation_mode") != "sealed_oos":
        raise ValueError("封存报告只接受 sealed_oos 评估结果")
    if len(results_map) != 1:
        raise ValueError("封存报告要求正好一个品种结果")
    if re.fullmatch(r"[0-9a-f]{64}", data_sha256) is None:
        raise ValueError("封存报告的评估数据 hash 非法")
    if not isinstance(strategy_bytes, bytes) or not strategy_bytes:
        raise ValueError("封存报告缺少策略文件原始字节")
    costs = normalize_cost_policy(
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
    )

    symbol, result = next(iter(results_map.items()))
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("封存报告的 symbol 必须是非空字符串")
    try:
        sharpe = float(result["sharpe"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("封存报告缺少合法的 Sharpe") from exc
    if not math.isfinite(sharpe):
        raise ValueError("封存报告的 Sharpe 必须是有限浮点数")
    try:
        actual_cost_rate = float(result["cost_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("封存报告缺少回测实际使用的 cost_rate") from exc
    if not math.isfinite(actual_cost_rate):
        raise ValueError("封存报告的实际 cost_rate 必须是有限浮点数")
    if actual_cost_rate != costs["cost_rate"]:
        raise ValueError(
            "封存报告声明的手续费/滑点与回测实际 cost_rate 不一致"
        )

    test_start = evaluation_contract.get("score_start")
    test_end = evaluation_contract.get("score_end")
    test_start_seconds = _utc_seconds(test_start, "test_start")
    test_end_seconds = _utc_seconds(test_end, "test_end")
    if test_start_seconds >= test_end_seconds:
        raise ValueError("封存报告必须满足 test_start < test_end")

    payload = {
        "format": SEALED_REPORT_FORMAT,
        "symbol": symbol,
        "data_sha256": data_sha256,
        "strategy_sha256": hashlib.sha256(strategy_bytes).hexdigest(),
        "evaluation_mode": "sealed_oos",
        "test_start": test_start,
        "test_end": test_end,
        **costs,
        "sharpe": sharpe,
    }
    if tuple(payload) != _SEALED_REPORT_FIELDS:
        raise RuntimeError("封存报告字段合同被意外修改")
    return payload


def _write_sealed_report_atomic(path: Path, payload: dict) -> None:
    """先完整落盘临时文件，再以不覆盖的原子硬链接发布报告。"""
    if tuple(payload) != _SEALED_REPORT_FIELDS:
        raise ValueError("封存报告字段必须严格匹配带成本身份的字段合同")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"封存报告已存在，禁止覆盖: {target}")

    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"封存报告已存在，禁止覆盖: {target}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


# ── 统计指标 ──────────────────────────────────────────────────────────────────

def calc_sharpe(pnl: np.ndarray, periods_per_year: int = _H1_PER_YEAR) -> float:
    """年化 Sharpe（无风险利率=0）。"""
    m = pnl.mean()
    s = pnl.std(ddof=0)
    if s < 1e-10:
        return 0.0
    return float(m / s * math.sqrt(periods_per_year))


def calc_sortino(pnl: np.ndarray, periods_per_year: int = _H1_PER_YEAR) -> float:
    """年化 Sortino（下行标准差）。"""
    m    = pnl.mean()
    down = pnl[pnl < 0]
    ds   = down.std(ddof=0) if len(down) > 0 else 1e-10
    ds   = max(ds, abs(m), 1e-10)
    return float(np.clip(m / ds * math.sqrt(periods_per_year), -20, 20))


def calc_rolling_sharpe(
    pnl: np.ndarray,
    window: int = 500,
    periods_per_year: int = _H1_PER_YEAR,
) -> np.ndarray:
    """滚动年化夏普；窗口不足处为 nan。"""
    T = len(pnl)
    out = np.full(T, np.nan, dtype=np.float64)
    if T == 0 or window <= 1:
        return out
    w = min(window, T)
    # 累积和 / 累积平方和 → O(T) 滑动窗口
    csum = np.concatenate([[0.0], np.cumsum(pnl, dtype=np.float64)])
    csq = np.concatenate([[0.0], np.cumsum(pnl.astype(np.float64) ** 2)])
    for i in range(w - 1, T):
        s = csum[i + 1] - csum[i + 1 - w]
        sq = csq[i + 1] - csq[i + 1 - w]
        mean = s / w
        var = sq / w - mean * mean
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-12:
            out[i] = 0.0
        else:
            out[i] = float(np.clip(mean / std * math.sqrt(periods_per_year), -20, 20))
    return out


def _fmt_pl_ratio(results_map: dict) -> str:
    vals = [
        d["profit_loss_ratio"]
        for d in results_map.values()
        if d.get("profit_loss_ratio") is not None
    ]
    if not vals:
        return "—"
    return f"{sum(vals) / len(vals):.3f}"


# ── 资金曲线图 ────────────────────────────────────────────────────────────────

def _setup_chinese_font() -> None:
    """让 matplotlib 能正确显示中文（Windows 优先微软雅黑）。"""
    from matplotlib import font_manager

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_equity_curves(
    results_map: dict,
    output_dir: str,
    times_arr: np.ndarray | None = None,
    periods_per_year: int = _H1_PER_YEAR,
):
    """绘制各品种 + 等权组合的资金曲线（中文标注）。

    Args:
        results_map: {symbol: {"pnl": np.array, "cum_pnl": np.array, ...}}
        output_dir:  输出目录
        times_arr:   时间戳数组（Unix秒），用于 X 轴刻度
    """
    _setup_chinese_font()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    syms   = list(results_map.keys())
    n_syms = len(syms)

    fig, ax_eq = plt.subplots(figsize=(18, 7), dpi=110)

    colors = ["#1565c0", "#00897b", "#e65100", "#6a1b9a", "#558b2f", "#b71c1c"]

    # 等权组合 PnL
    all_pnls = np.stack([results_map[s]["pnl"] for s in syms], axis=0)
    port_pnl = all_pnls.mean(axis=0)
    port_cum = np.cumsum(port_pnl)

    T = len(port_cum)
    x = np.arange(T)

    if n_syms == 1:
        sym = syms[0]
        cum = results_map[sym]["cum_pnl"]
        ax_eq.plot(
            x, cum, linewidth=2.0, color="#1565c0",
            label=f"{sym}（索提诺 {results_map[sym]['sortino']:+.2f}）",
        )
        ax_eq.fill_between(x, cum, 0, where=cum >= 0, alpha=0.08, color="#1565c0")
        ax_eq.fill_between(x, cum, 0, where=cum < 0,  alpha=0.08, color="#b71c1c")
        title_head = f"{sym} 资金曲线"
        show_pnl, show_cum = results_map[sym]["pnl"], cum
    else:
        for i, sym in enumerate(syms):
            cum = results_map[sym]["cum_pnl"]
            ax_eq.plot(
                x, cum, linewidth=0.8, alpha=0.65, color=colors[i % len(colors)],
                label=f"{sym}（索提诺 {results_map[sym]['sortino']:+.2f}）",
            )
        ax_eq.plot(
            x, port_cum, linewidth=2.2, color="black",
            label=f"等权组合（索提诺 {calc_sortino(port_pnl, periods_per_year):+.2f}）",
        )
        ax_eq.fill_between(x, port_cum, 0, where=port_cum >= 0, alpha=0.06, color="#1565c0")
        ax_eq.fill_between(x, port_cum, 0, where=port_cum < 0,  alpha=0.06, color="#b71c1c")
        title_head = "多因子组合资金曲线"
        show_pnl, show_cum = port_pnl, port_cum

    ax_eq.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.set_ylabel("累计对数收益", fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax_eq.grid(alpha=0.25)
    ax_eq.set_title(
        f"{title_head}  |  "
        f"总收益={show_cum[-1]:+.3f}  "
        f"夏普={calc_sharpe(show_pnl, periods_per_year):+.2f}  "
        f"索提诺={calc_sortino(show_pnl, periods_per_year):+.2f}  "
        f"盈亏比={_fmt_pl_ratio(results_map)}",
        fontsize=11, pad=8,
    )

    # X 轴时间刻度
    if times_arr is not None and len(times_arr) == T:
        from datetime import datetime, timezone
        step  = max(1, T // 10)
        ticks = x[::step]
        labels = [
            datetime.fromtimestamp(int(times_arr[i]), tz=timezone.utc).strftime("%Y-%m-%d")
            for i in range(0, T, step)
        ]
        ax_eq.set_xticks(ticks)
        ax_eq.set_xticklabels(labels[:len(ticks)], fontsize=8, rotation=20)
    ax_eq.set_xlabel("日期", fontsize=9)

    path = str(Path(output_dir) / "portfolio_equity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  资金曲线图已保存 → {path}")
    return path


def export_equity_json(
    results_map: dict,
    output_dir: str,
    times_arr: np.ndarray | None = None,
    max_points: int = 1500,
    rolling_window: int = 500,
    periods_per_year: int = _H1_PER_YEAR,
):
    """导出资金曲线原始数据为 JSON，供前端渲染交互式 HTML 图表。

    结构：
        {
          "labels": [...时间标签],
          "n_points": int, "total_bars": int,
          "rolling_window": int,
          "symbols": { sym: { equity, rolling_sharpe, sharpe, sortino,
                              total_return, profit_loss_ratio } },
          "portfolio": { ... }   # 多品种时才有
        }
    """
    syms = list(results_map.keys())
    if not syms:
        return None

    all_pnls = np.stack([results_map[s]["pnl"] for s in syms], axis=0)
    port_pnl = all_pnls.mean(axis=0)
    port_cum = np.cumsum(port_pnl)
    T = len(port_cum)

    # 均匀降采样，保证首尾点在内，避免 JSON 过大导致前端卡顿
    if T > max_points:
        idx = np.unique(np.linspace(0, T - 1, max_points).astype(int))
    else:
        idx = np.arange(T)

    def _sample(arr: np.ndarray) -> list[float | None]:
        out = []
        for i in idx:
            v = arr[i]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out.append(None)
            else:
                out.append(round(float(v), 6))
        return out

    if times_arr is not None and len(times_arr) == T:
        from datetime import datetime, timezone

        labels = [
            datetime.fromtimestamp(int(times_arr[i]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            for i in idx
        ]
    else:
        labels = [str(int(i)) for i in idx]

    out: dict = {
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "labels": labels,
        "n_points": int(len(idx)),
        "total_bars": int(T),
        "rolling_window": int(rolling_window),
        "symbols": {},
    }
    for s in syms:
        cum = results_map[s]["cum_pnl"]
        roll = calc_rolling_sharpe(
            results_map[s]["pnl"],
            window=rolling_window,
            periods_per_year=periods_per_year,
        )
        pl = results_map[s].get("profit_loss_ratio")
        out["symbols"][s] = {
            "equity": _sample(cum),
            "rolling_sharpe": _sample(roll),
            "sharpe": round(float(results_map[s]["sharpe"]), 4),
            "sortino": round(float(results_map[s]["sortino"]), 4),
            "total_return": round(float(results_map[s]["total_return"]), 6),
            "profit_loss_ratio": round(float(pl), 4) if pl is not None else None,
        }

    if len(syms) > 1:
        pl_vals = [
            results_map[s]["profit_loss_ratio"]
            for s in syms
            if results_map[s].get("profit_loss_ratio") is not None
        ]
        port_pl = float(sum(pl_vals) / len(pl_vals)) if pl_vals else None
        out["portfolio"] = {
            "equity": _sample(port_cum),
            "rolling_sharpe": _sample(
                calc_rolling_sharpe(
                    port_pnl,
                    window=rolling_window,
                    periods_per_year=periods_per_year,
                )
            ),
            "sharpe": round(float(calc_sharpe(port_pnl, periods_per_year)), 4),
            "sortino": round(float(calc_sortino(port_pnl, periods_per_year)), 4),
            "total_return": round(float(port_cum[-1]), 6),
            "profit_loss_ratio": round(port_pl, 4) if port_pl is not None else None,
        }

    path = Path(output_dir) / "equity_curve.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  资金曲线数据已保存 → {path}")
    return str(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR  = "backtest_output"
    single_mode = "--single" in sys.argv
    # 回测强制离线：只用本地 Parquet，永不连 MT5 在线
    if "--online" in sys.argv or "--mt5" in sys.argv:
        print("[ERROR] 回测已禁用在线/MT5 拉数。请使用本地 Parquet（--data-file 或策略内 data_file）。")
        sys.exit(1)

    strategy_file = None
    data_file_arg = None
    evaluation_mode = "auto"
    score_start = None
    sealed_report = None
    commission_pct = DEFAULT_COMMISSION_PCT
    slippage_pct = DEFAULT_SLIPPAGE_PCT
    for i, arg in enumerate(sys.argv):
        if arg == "--strategy-file" and i + 1 < len(sys.argv):
            strategy_file = sys.argv[i + 1]
        elif arg == "--data-file" and i + 1 < len(sys.argv):
            data_file_arg = sys.argv[i + 1]
        elif arg == "--commission" and i + 1 < len(sys.argv):
            commission_pct = float(sys.argv[i + 1])
        elif arg == "--slippage" and i + 1 < len(sys.argv):
            slippage_pct = float(sys.argv[i + 1])
        elif arg == "--evaluation-mode" and i + 1 < len(sys.argv):
            evaluation_mode = sys.argv[i + 1]
        elif arg == "--score-start" and i + 1 < len(sys.argv):
            score_start = sys.argv[i + 1]
        elif arg == "--sealed-report":
            if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
                print("[ERROR] --sealed-report 缺少输出路径")
                sys.exit(1)
            sealed_report = sys.argv[i + 1]

    try:
        sealed_report_path = _validate_sealed_report_cli(
            evaluation_mode=evaluation_mode,
            strategy_file=strategy_file,
            sealed_report=sealed_report,
            single_mode=single_mode,
        )
        if sealed_report_path is not None and sealed_report_path.exists():
            raise FileExistsError(
                f"封存报告已存在，禁止覆盖: {sealed_report_path}"
            )
    except (ValueError, FileExistsError) as exc:
        print(f"[ERROR] CLI 参数无效: {exc}")
        sys.exit(1)

    if (
        not math.isfinite(commission_pct)
        or not math.isfinite(slippage_pct)
        or commission_pct < 0
        or slippage_pct < 0
    ):
        print("[ERROR] 手续费/滑点必须是非负有限数字"); sys.exit(1)
    cost_rate_all = (commission_pct + slippage_pct) / 100.0
    print(
        f"\n交易成本（单边）: "
        f"手续费={commission_pct:g}%  滑点={slippage_pct:g}%  "
        f"→ cost_rate={cost_rate_all:.8f}"
    )
    print("数据模式: 强制离线 Parquet（不连接 MT5）")

    # ── 2. 加载策略 ─────────────────────────────────────────────────
    strategy_data_file = None
    strategy_bytes = None
    strategy_contracts: list[dict] = []
    print(f"\n{'='*62}")
    if strategy_file:
        strategy_path = Path(strategy_file)
        if not strategy_path.exists():
            print(f"[ERROR] 找不到: {strategy_file}"); sys.exit(1)
        strategy_bytes = strategy_path.read_bytes()
        data = load_strategy(strategy_path, raw_bytes=strategy_bytes)
        strategy_data_file = data.get("data_file")
        sym = data.get("symbol")
        if not sym:
            stem = Path(strategy_file).stem
            if stem.startswith("best_"):
                sym = stem.replace("best_", "", 1)
            elif stem.startswith("strategy_"):
                # strategy_ADAUSD_step0084_score2.4021 / strategy_ADAUSD (1)
                rest = stem.replace("strategy_", "", 1)
                sym = rest.split("_step")[0].split(" ")[0]
        if not sym:
            print("[ERROR] 策略文件未包含 symbol，且无法从文件名识别"); sys.exit(1)
        symbol_formulas = {sym: data["formula"]}
        strategy_contracts.append(data)
        sc = data.get("best_score", "N/A")
        score_txt = f"{sc:.3f}" if isinstance(sc, (int, float)) else str(sc)
        print(f"  模式: 单策略文件 ({Path(strategy_file).name})")
        print(f"  {sym}: score={score_txt}  {decode_formula(data['formula'])}")
        if strategy_data_file:
            print(f"  策略记录数据: {strategy_data_file}")
    elif single_mode:
        data = load_strategy(Path(Config.STRATEGY_FILE))
        if data is None:
            print(f"[ERROR] 找不到: {Config.STRATEGY_FILE}"); sys.exit(1)
        strategy_data_file = data.get("data_file")
        symbol_formulas = {sym: data["formula"] for sym in Config.SYMBOLS}
        strategy_contracts.append(data)
        print("  模式: 单公式（所有品种共用）")
    else:
        symbol_formulas = {}
        for sym in Config.SYMBOLS:
            path = Path("strategies") / f"best_{sym}.json"
            data = load_strategy(path)
            if data is None:
                print(f"  [缺失] {sym}")
                continue
            ver = data.get("vocab_version", "unknown")
            if ver != VOCAB_VERSION:
                print(f"  [跳过] {sym}: vocab_version 不符 ({ver} vs {VOCAB_VERSION})")
                continue
            symbol_formulas[sym] = data["formula"]
            strategy_contracts.append(data)
            if not strategy_data_file and data.get("data_file"):
                strategy_data_file = data.get("data_file")
            sc = data.get("best_score", "N/A")
            print(f"  {sym}: score={sc:.3f}  {decode_formula(data['formula'])}")

    if not symbol_formulas:
        print("[ERROR] 没有有效策略，请先运行训练"); sys.exit(1)

    cost_rates = {sym: cost_rate_all for sym in symbol_formulas}
    print(f"{'='*62}\n")

    # ── 3. 加载数据（仅本地 Parquet）────────────────────────────────
    if not data_file_arg and strategy_data_file:
        data_file_arg = str(strategy_data_file).strip() or None

    if not data_file_arg:
        print(
            "[ERROR] 未指定本地 Parquet。\n"
            "请传入 --data-file PATH\\TO\\SYMBOL_TF.parquet，\n"
            "或使用包含 data_file 字段的策略 JSON（本软件训练生成）。\n"
            "回测不会连接 MT5 / 不会使用在线行情。"
        )
        sys.exit(1)

    parquet_path = Path(data_file_arg)
    if not parquet_path.exists():
        print(f"[ERROR] Parquet 不存在: {parquet_path}")
        sys.exit(1)

    print(f"正在加载数据（离线 Parquet: {parquet_path}）...")
    pm = ParquetDataManager(str(parquet_path))
    pm.load()
    applicable_contracts = strategy_contracts
    if not strategy_file and not single_mode:
        applicable_contracts = [
            contract
            for contract in strategy_contracts
            if contract.get("symbol") == pm.symbol
        ]
        if not applicable_contracts:
            print(f"[ERROR] 当前数据品种 {pm.symbol} 没有对应策略")
            sys.exit(1)
    evaluation_contracts: list[dict] = []
    try:
        for contract in applicable_contracts:
            evaluation_contracts.append(
                _validate_strategy_data_contract(
                    contract,
                    pm,
                    evaluation_mode=evaluation_mode,
                    score_start=score_start,
                )
            )
    except ValueError as exc:
        print(f"[ERROR] 拒绝回测: {exc}")
        sys.exit(1)
    score_start_indices = {
        contract["score_start_index"] for contract in evaluation_contracts
    }
    resolved_modes = {
        contract["evaluation_mode"] for contract in evaluation_contracts
    }
    if len(score_start_indices) != 1 or len(resolved_modes) != 1:
        print("[ERROR] 多策略的评估区间或评估模式不一致")
        sys.exit(1)
    score_start_index = next(iter(score_start_indices))
    resolved_evaluation_mode = next(iter(resolved_modes))
    evaluation_contract = evaluation_contracts[0]
    periods_per_year = pm.periods_per_year
    raw_dict = pm.raw_dict
    syms = pm.symbols
    T = raw_dict["open"].shape[1]
    times_all = raw_dict.get("time", None)
    print(
        f"  品种: {syms}  T={T} bars  年化周期={periods_per_year}\n"
        f"  评估模式: {resolved_evaluation_mode}  "
        f"评分起点={evaluation_contract['score_start']}  "
        f"预热={score_start_index} bars\n"
    )

    # ── 4. 为每品种计算因子 + 回测 ───────────────────────────────
    vm   = StackVM()
    # 因果特征化：_robust_norm 现为滚动窗口实现，传入全量序列是安全的
    # 每个时间步 t 的归一化参数只依赖 [t-w+1..t]，无 look-ahead
    feat = MT5FeatureEngineer.compute_features(raw_dict)  # [N, F, T]，因果安全

    results_map = {}
    backtest_results = []

    for i, sym in enumerate(syms):
        if sym not in symbol_formulas:
            print(f"  [跳过] {sym}（无策略）")
            continue

        formula   = symbol_formulas[sym]
        cost_rate = cost_rates.get(sym, cost_rate_all)
        feat_i    = feat[i:i+1]
        raw_i     = {k: v[i:i+1] for k, v in raw_dict.items()}

        engine    = BacktestEngine(
            formula=formula,
            cost_rate=cost_rate,
            periods_per_year=periods_per_year,
        )
        sym_res   = engine.run(
            raw_i,
            feat_i,
            [sym],
            score_start_index=score_start_index,
        )
        backtest_results.extend(sym_res)

        r = sym_res[0]
        pnl_arr = r.pnl
        cum_arr = r.cum_pnl
        sharpe  = calc_sharpe(pnl_arr, periods_per_year)
        sortino = calc_sortino(pnl_arr, periods_per_year)
        pl_ratio = r.profit_loss_ratio

        results_map[sym] = {
            "pnl":          pnl_arr,
            "cum_pnl":      cum_arr,
            "total_return": r.total_return,
            "sharpe":       sharpe,
            "sortino":      sortino,
            "n_trades":     r.n_trades,
            "win_rate":     r.win_rate,
            "avg_hold":     r.avg_hold_bars,
            "profit_loss_ratio": pl_ratio,
            "cost_rate":    cost_rate,
        }

    if sealed_report_path is not None:
        try:
            sealed_payload = _build_sealed_report_payload(
                results_map=results_map,
                evaluation_contract=evaluation_contract,
                data_sha256=pm.data_sha256,
                strategy_bytes=strategy_bytes,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
            )
            _write_sealed_report_atomic(sealed_report_path, sealed_payload)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] 封存报告生成失败: {exc}")
            sys.exit(1)
        print(f"  封存样本外报告已生成但未展示指标 → {sealed_report_path}")
        print("完成。\n")
        return

    # ── 5. 打印各品种统计 ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  多因子回测报告")
    print(f"{'='*62}")
    header = f"{'品种':12s} {'PnL':>8} {'Sharpe':>8} {'Sortino':>8} {'盈亏比':>8} {'Trades':>7} {'WinRate':>8} {'AvgH':>6}"
    print(f"  {header}")
    print(f"  {'─'*72}")
    for sym, d in results_map.items():
        pl = d["profit_loss_ratio"]
        pl_s = f"{pl:8.3f}" if pl is not None else f"{'—':>8}"
        print(f"  {sym:12s} "
              f"{d['total_return']:+8.3f} "
              f"{d['sharpe']:+8.3f} "
              f"{d['sortino']:+8.3f} "
              f"{pl_s} "
              f"{d['n_trades']:7d} "
              f"{d['win_rate']:8.1%} "
              f"{d['avg_hold']:6.1f}h")

    # 等权组合
    p_pl_ratio = None
    if results_map:
        all_pnls = np.stack([d["pnl"] for d in results_map.values()], axis=0)
        port_pnl = all_pnls.mean(axis=0)
        port_cum = np.cumsum(port_pnl)
        p_sharpe  = calc_sharpe(port_pnl, periods_per_year)
        p_sortino = calc_sortino(port_pnl, periods_per_year)
        pl_vals = [d["profit_loss_ratio"] for d in results_map.values()
                   if d["profit_loss_ratio"] is not None]
        p_pl_ratio = float(sum(pl_vals) / len(pl_vals)) if pl_vals else None
        pl_s = f"{p_pl_ratio:8.3f}" if p_pl_ratio is not None else f"{'—':>8}"
        print(f"  {'─'*72}")
        print(f"  {'Portfolio':12s} "
              f"{port_cum[-1]:+8.3f} "
              f"{p_sharpe:+8.3f} "
              f"{p_sortino:+8.3f} "
              f"{pl_s}")
        print(f"\n  正收益品种: {sum(1 for d in results_map.values() if d['total_return']>0)}/{len(results_map)}")
        print(f"  Sharpe>1 品种: {sum(1 for d in results_map.values() if d['sharpe']>1)}/{len(results_map)}")
    print(f"{'='*62}\n")

    # ── 6. 资金曲线图 ─────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    chart_artifacts: list[dict[str, str]] = []
    equity_artifact: dict[str, str] | None = None
    if results_map:
        times_np = (
            times_all[0, score_start_index:].numpy()
            if times_all is not None
            else None
        )
        chart_path = Path(plot_equity_curves(
            results_map,
            OUTPUT_DIR,
            times_np,
            periods_per_year=periods_per_year,
        ))
        chart_artifacts.append(
            {
                "name": chart_path.name,
                "sha256": _file_sha256(chart_path),
                "label": "组合资金曲线",
                "kind": "portfolio",
            }
        )
        equity_path = Path(export_equity_json(
            results_map,
            OUTPUT_DIR,
            times_np,
            periods_per_year=periods_per_year,
        ))
        equity_artifact = {
            "name": equity_path.name,
            "sha256": _file_sha256(equity_path),
        }

    # ── 7. 资金曲线图已在步骤 6 生成；跳过 K 线/逐笔交易图以加快回测 ─────

    # ── 8. 保存 JSON 报告 ─────────────────────────────────────────────
    report = {
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "mode": "single" if single_mode else "multi_factor",
        "symbol": pm.symbol,
        "timeframe": pm.timeframe,
        "local_source": pm.source,
        "dataset_id": pm.dataset_id,
        "data_sha256": pm.data_sha256,
        "periods_per_year": periods_per_year,
        "minimum_bars": pm.minimum_bars,
        "evaluation_mode": resolved_evaluation_mode,
        "score_start": evaluation_contract["score_start"],
        "score_end": evaluation_contract["score_end"],
        "warmup_bars": evaluation_contract["warmup_bars"],
        "annualization": evaluation_contract["annualization"],
        "training_data": (
            evaluation_contracts[0]["training_data"]
            if len(evaluation_contracts) == 1
            else [row["training_data"] for row in evaluation_contracts]
        ),
        "evaluation_data": evaluation_contract["evaluation_data"],
        "cost_rates": cost_rates,
        "chart_artifacts": chart_artifacts,
        "equity_artifact": equity_artifact,
        "symbols": {},
        "portfolio": {},
    }
    for sym, d in results_map.items():
        formula = symbol_formulas.get(sym, [])
        pl = d["profit_loss_ratio"]
        report["symbols"][sym] = {
            "formula":      formula,
            "readable":     decode_formula(formula),
            "cost_rate":    d["cost_rate"],
            "total_return": round(d["total_return"], 6),
            "sharpe":       round(d["sharpe"], 4),
            "sortino":      round(d["sortino"], 4),
            "n_trades":     d["n_trades"],
            "win_rate":     round(d["win_rate"], 4),
            "avg_hold_bars":round(d["avg_hold"], 2),
            "profit_loss_ratio": round(pl, 4) if pl is not None else None,
        }
    if results_map:
        report["portfolio"] = {
            "total_return": round(float(port_cum[-1]), 6),
            "sharpe":       round(p_sharpe, 4),
            "sortino":      round(p_sortino, 4),
            "profit_loss_ratio": round(p_pl_ratio, 4) if p_pl_ratio is not None else None,
        }
    rp = f"{OUTPUT_DIR}/multi_factor_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 报告已保存 → {rp}")

    print("完成。\n")


if __name__ == "__main__":
    main()
