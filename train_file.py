"""
train_file.py — 从单个 Parquet K 线文件训练

用法:
    python train_file.py --data-file D:\\K线数据\\AAPL_H1.parquet

文件名格式: {品种}_{周期}.parquet，例如 AAPL_H1.parquet、US30.cash_H1.parquet
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.train_logging import configure_train_stdio

configure_train_stdio()

from config import Config
from data_pipeline.dataset_contracts import TRAINING_SOURCE_IDS
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.config import ModelConfig
import model_core.engine as _engine_module
from model_core.engine import AlphaEngine
from model_core.vocab import VOCAB_VERSION


DEFAULT_TRAIN_STEPS = ModelConfig.TRAIN_STEPS
CHECKPOINT_IDENTITY_FIELDS = getattr(
    _engine_module,
    "CHECKPOINT_IDENTITY_FIELDS",
    (
        "symbol",
        "timeframe",
        "dataset_id",
        "data_sha256",
        "local_source",
        "periods_per_year",
        "minimum_bars",
    ),
)


def checkpoint_identity_directory(timeframe: str, dataset_id: str) -> pathlib.Path:
    return _engine_module.checkpoint_identity_directory(timeframe, dataset_id)


def checkpoint_symbol_tag(symbol: str) -> str:
    return _engine_module.checkpoint_symbol_tag(symbol)


def _positive_int(value: str) -> int:
    """解析严格正整数，拒绝符号、小数和非 ASCII 数字。"""
    if not value or not value.isascii() or not value.isdigit():
        raise argparse.ArgumentTypeError("必须是严格正整数")
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从单个 Parquet K 线文件训练")
    parser.add_argument(
        "--data-file",
        required=True,
        help="符合 {品种}_{周期}.parquet 命名规则的输入文件",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="为当前数据身份新建隔离 run，不加载或删除旧检查点",
    )
    parser.add_argument(
        "--train-steps",
        type=_positive_int,
        default=DEFAULT_TRAIN_STEPS,
        help=f"本次训练步数，必须为严格正整数（默认: {DEFAULT_TRAIN_STEPS}）",
    )
    parser.add_argument(
        "--periods-per-year",
        type=_positive_int,
        default=None,
        help="每年 K 线数；若提供，必须与数据 manifest 的严格合同一致",
    )
    parser.add_argument(
        "--minimum-bars",
        type=_positive_int,
        default=None,
        help="来源特定最低 K 线数；仅由已验证的远程 run manifest 传入",
    )
    parser.add_argument(
        "--data-source",
        choices=tuple(sorted(TRAINING_SOURCE_IDS)),
        default=None,
        help="数据来源身份；远程 Worker 必须从 run manifest 显式传入",
    )
    return parser


def train_from_file(
    data_file: str,
    *,
    from_scratch: bool = False,
    data_source: str | None = None,
    periods_per_year: int | None = None,
    minimum_bars: int | None = None,
) -> AlphaEngine | None:
    info = inspect_parquet_file(
        data_file,
        expected_source_id=data_source,
        expected_periods_per_year=periods_per_year,
        expected_minimum_bars=minimum_bars,
    )
    symbol = info["symbol"]
    timeframe = info["timeframe"]
    data_sha256 = info["data_sha256"]
    dataset_id = info["dataset_id"]
    local_source = info["source"]

    print(f"\n{'='*60}")
    print(f"  AlphaGPT 文件训练 — {info['filename']}")
    print(f"{'='*60}")
    print(f"  品种: {symbol}")
    print(f"  周期: {timeframe}")
    print(f"  数据: 强制离线 Parquet（不连接 MT5）")
    print(f"  文件: {Path(data_file).resolve()}")
    print(f"  训练步数: {ModelConfig.TRAIN_STEPS}")
    print(f"  K线数: {info['bars']}")
    print(f"  年化周期: {info['periods_per_year']} bars/year")
    print(f"  数据来源: {info['source'] or '本地未声明来源'}")
    print(f"  模式: {'重新训练（从头）' if from_scratch else '自动续训'}")
    print(f"{'='*60}")

    try:
        mgr = ParquetDataManager(
            data_file,
            expected_source_id=data_source,
            expected_periods_per_year=periods_per_year,
            expected_minimum_bars=minimum_bars,
        )
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        print(f"  数据加载成功，共 {T} 根K线")
    except Exception as e:
        print(f"  [错误] 数据加载失败: {e}")
        return None

    loaded_identity = {
        "source": mgr.source,
        "data_sha256": mgr.data_sha256,
        "dataset_id": mgr.dataset_id,
        "periods_per_year": mgr.periods_per_year,
        "minimum_bars": mgr.minimum_bars,
    }
    for field, actual in loaded_identity.items():
        expected = info[field]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"数据在检查与加载之间发生身份变化: {field} "
                f"期望 {expected!r}，实际 {actual!r}"
            )

    engine = AlphaEngine(data_manager=mgr, target_symbol=symbol)
    engine.timeframe = timeframe
    engine.data_file = str(Path(data_file).resolve())
    engine.mode = "parquet_file"
    engine.periods_per_year = mgr.periods_per_year
    engine.minimum_bars = mgr.minimum_bars
    engine.data_sha256 = data_sha256
    engine.dataset_id = dataset_id
    engine.local_source = local_source
    engine.data_rows = mgr.data_rows
    engine.data_start = mgr.data_start
    engine.data_end = mgr.data_end
    engine.data_columns = list(mgr.columns)
    engine.train_steps = ModelConfig.TRAIN_STEPS

    identity_root = checkpoint_identity_directory(timeframe, dataset_id)
    latest_checkpoint = _latest_identity_checkpoint(identity_root, symbol)
    start_step = 0

    if from_scratch:
        engine.checkpoint_dir = _new_checkpoint_run_directory(identity_root)
        print("  [重新训练] 已创建当前数据身份的新 run；旧检查点保持不变")
        _seed_best_from_strategy(engine, symbol)
    elif latest_checkpoint is not None:
        engine.checkpoint_dir = latest_checkpoint.parent
        start_step = engine.load_checkpoint(str(latest_checkpoint))
        print(f"  [续训] 从 {latest_checkpoint} 恢复，起始步={start_step}")
    else:
        legacy = _latest_legacy_checkpoint(symbol)
        if legacy is not None:
            raise RuntimeError(
                f"检测到旧版扁平 checkpoint '{legacy}'；其路径未按 timeframe+dataset "
                "隔离，拒绝续训。请显式使用 --from-scratch 创建新身份 run"
            )
        engine.checkpoint_dir = _new_checkpoint_run_directory(identity_root)

    if start_step >= ModelConfig.TRAIN_STEPS:
        print(f"  [完成] {symbol} 已完成全部 {ModelConfig.TRAIN_STEPS} 步，跳过训练")
        _save_strategy(
            engine,
            symbol,
            timeframe,
            data_file,
            mgr.periods_per_year,
            mgr.minimum_bars,
            local_source,
            dataset_id,
            data_sha256,
        )
        return engine

    if start_step == 0 and not from_scratch:
        print("  [新训] 从第 0 步开始")

    if start_step > 0:
        engine._save_training_history_live()

    engine.train(start_step=start_step)
    _save_strategy(
        engine,
        symbol,
        timeframe,
        data_file,
        mgr.periods_per_year,
        mgr.minimum_bars,
        local_source,
        dataset_id,
        data_sha256,
    )
    return engine


def _new_checkpoint_run_directory(identity_root: pathlib.Path) -> pathlib.Path:
    """原子创建新 run 目录，确保 from_scratch 意图可被后续进程识别。"""
    run_number = time.time_ns()
    while True:
        candidate = identity_root / f"run_{run_number:020d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            run_number += 1


def _checkpoint_step(path: pathlib.Path) -> int:
    marker = "_step_"
    if marker not in path.stem:
        return -1
    raw = path.stem.rsplit(marker, 1)[1]
    return int(raw) if raw.isascii() and raw.isdigit() else -1


def _latest_identity_checkpoint(
    identity_root: pathlib.Path, symbol: str
) -> pathlib.Path | None:
    """只在最新身份 run 内选择最高 step，避免回到更老的高步数 run。"""
    symbol_tag = checkpoint_symbol_tag(symbol)
    runs = [
        path
        for path in identity_root.glob("run_*")
        if path.is_dir()
        and len(path.name) == 24
        and path.name.startswith("run_")
        and path.name[4:].isascii()
        and path.name[4:].isdigit()
    ]
    if not runs:
        return None
    latest_run = max(runs, key=lambda path: path.name)
    checkpoints = [
        path
        for path in latest_run.glob(f"ckpt_{symbol_tag}_step_*.pt")
        if path.is_file() and _checkpoint_step(path) >= 0
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: (_checkpoint_step(path), path.name))


def _latest_legacy_checkpoint(symbol: str) -> pathlib.Path | None:
    """查找旧版扁平 checkpoint，以便明确拒绝而不是静默忽略。"""
    symbol_tag = checkpoint_symbol_tag(symbol)
    candidates = [
        path
        for path in pathlib.Path("checkpoints").glob(f"ckpt_{symbol_tag}_step_*.pt")
        if path.is_file() and _checkpoint_step(path) >= 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_checkpoint_step(path), path.name))


def _strategy_identity(engine: AlphaEngine) -> dict:
    return {
        field: getattr(engine, field if field != "symbol" else "target_symbol")
        for field in CHECKPOINT_IDENTITY_FIELDS
    }


def _same_strategy_identity(data: dict, engine: AlphaEngine) -> bool:
    expected = _strategy_identity(engine)
    return all(
        field in data
        and type(data[field]) is type(expected[field])
        and data[field] == expected[field]
        for field in CHECKPOINT_IDENTITY_FIELDS
    )


def _seed_best_from_strategy(engine: AlphaEngine, symbol: str) -> None:
    """把已有 best_{symbol}.json 当作重新训练的分数下限。"""
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("策略 JSON 顶层必须是对象")
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [警告] 读取已有策略失败: {e}")
        return
    except TypeError as e:
        print(f"  [警告] 已有策略格式非法: {e}")
        return
    if data.get("vocab_version") != VOCAB_VERSION:
        print("  [重新训练] 已有最优策略的公式执行版本不同，不作为本次分数下限")
        return
    if not _same_strategy_identity(data, engine):
        print("  [重新训练] 已有最优策略的数据身份不同，不作为本次分数下限")
        return
    formula = data.get("formula")
    score = data.get("best_score")
    if not formula or score is None:
        return
    try:
        engine.best_formula = [int(t) for t in formula]
        engine.best_score = float(score)
        print(f"  [重新训练] 保留已有最优分数下限={engine.best_score:.4f}，仅更好时才会覆盖策略文件")
    except (TypeError, ValueError) as e:
        print(f"  [警告] 已有策略无法用作下限: {e}")


def _save_strategy(
    engine: AlphaEngine,
    symbol: str,
    timeframe: str,
    data_file: str,
    periods_per_year: int,
    minimum_bars: int,
    data_source: str,
    dataset_id: str,
    data_sha256: str,
) -> None:
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    path.parent.mkdir(exist_ok=True)
    # 若磁盘上已有更高分，不要用更弱结果覆盖
    if path.exists() and engine.best_formula is not None:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(old, dict):
                raise TypeError("策略 JSON 顶层必须是对象")
            old_score = old.get("best_score")
            if (
                old.get("vocab_version") == VOCAB_VERSION
                and _same_strategy_identity(old, engine)
                and old_score is not None
                and float(old_score) > float(engine.best_score)
            ):
                print(
                    f"  [策略] 保留磁盘更优结果 {float(old_score):.4f} "
                    f"> 本次 {float(engine.best_score):.4f}，未覆盖 {path}"
                )
                merged = dict(old)
                for key, val in (
                    ("timeframe", timeframe),
                    ("data_file", str(Path(data_file).resolve())),
                    ("mode", "parquet_file"),
                    ("train_steps", ModelConfig.TRAIN_STEPS),
                    ("periods_per_year", periods_per_year),
                    ("minimum_bars", minimum_bars),
                    ("local_source", data_source),
                    ("dataset_id", dataset_id),
                    ("data_sha256", data_sha256),
                    ("data_rows", getattr(engine, "data_rows", None)),
                    ("data_start", getattr(engine, "data_start", None)),
                    ("data_end", getattr(engine, "data_end", None)),
                    ("columns", getattr(engine, "data_columns", None)),
                ):
                    if val is not None and not merged.get(key):
                        merged[key] = val
                if merged != old:
                    path.write_text(
                        json.dumps(merged, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"  [策略] 已补全数据路径等元数据: {path}")
                return
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    data = {
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_file": str(Path(data_file).resolve()),
        "mode": "parquet_file",
        "periods_per_year": periods_per_year,
        "minimum_bars": minimum_bars,
        "local_source": data_source,
        "dataset_id": dataset_id,
        "data_sha256": data_sha256,
        "data_rows": getattr(engine, "data_rows", None),
        "data_start": getattr(engine, "data_start", None),
        "data_end": getattr(engine, "data_end", None),
        "columns": getattr(engine, "data_columns", None),
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula)
        if engine.best_formula
        else None,
        "best_score": engine.best_score,
        "train_steps": ModelConfig.TRAIN_STEPS,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  策略已保存: {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ModelConfig.REWARD_MODE = "ftmo"
    ModelConfig.TRAIN_STEPS = args.train_steps
    t0 = time.time()
    eng = train_from_file(
        args.data_file,
        from_scratch=args.from_scratch,
        data_source=args.data_source,
        periods_per_year=args.periods_per_year,
        minimum_bars=args.minimum_bars,
    )
    elapsed = time.time() - t0

    if eng:
        sym = eng.target_symbol or "?"
        print(f"\n<<< [{sym}] 训练完成: 最优分数={eng.best_score:.4f}，耗时 {elapsed/3600:.2f} 小时")
        if eng.best_formula:
            print(f"    {eng._decode_formula(eng.best_formula)}")
        return 0
    else:
        print("\n<<< 训练失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
