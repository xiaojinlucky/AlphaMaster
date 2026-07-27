"""
Property-based tests for model_core.ops — temporal operators.

Property 6: TS_RANK Output Range Constraint
  对任意 [N, T] 输入，TS_RANK_5/10/20 输出 ∈ [0.0, 1.0)
  **Validates: 需求 F2.4**

Property 7: TS_CORR_10 Correlation Coefficient Boundedness
  对任意 [N, T] 输入，TS_CORR_10 输出 ∈ [-1.0, 1.0]；
  当 x 或 y 为常数（窗口填满后）时，对应位置输出为 0。
  **Validates: 需求 F2.5**

Property 8: Operators Produce No NaN or Inf
  对任意 [N, T] 输入（包含零值、极大值等边界情况），OPS_CONFIG 全部注册算子
  （含时序、基础、跨品种与三元算子）输出不包含 NaN 或 Inf。
  2026-07-27 对齐 fork 当前算子语义：旧实现用开放切片 OPS_CONFIG[12:]（注释
  写"索引 12–21 共 10 个时序算子"，只在算子库=22 时成立），算子库扩到 62 后
  意外吞进 50 个算子且没有三元分派，遇到 IF_GT(arity=3) 即 TypeError 崩溃。
  现改为显式遍历全表 + 按 arity 1/2/3 分派，覆盖面 10→62 只增不减。
  **Validates: 需求 F2.6**
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model_core.ops import _ts_rank, _ts_corr_10, _ts_mean, _ts_std, OPS_CONFIG

# ── 全部注册算子（2026-07-27：替换陈旧开放切片 OPS_CONFIG[12:]，见文件头）────
_ALL_OPS = tuple(OPS_CONFIG)  # list of (name, fn, arity)，arity ∈ {1, 2, 3}


def _invoke_op(fn, arity: int, x: torch.Tensor, y: torch.Tensor,
               z: torch.Tensor) -> torch.Tensor:
    """按算子元数分派调用；未知元数直接失败（let it crash，不静默跳过）。"""
    if arity == 1:
        return fn(x)
    if arity == 2:
        return fn(x, y)
    if arity == 3:
        return fn(x, y, z)
    raise AssertionError(f"未知算子元数 arity={arity}")

# ── Strategies ────────────────────────────────────────────────────────────────

def _ohlcv_tensor(N: int, T: int) -> torch.Tensor:
    """Generate a strictly positive random [N, T] tensor (values in (1, 2])."""
    return torch.rand(N, T) + 1.0


def _randn_tensor(N: int, T: int) -> torch.Tensor:
    """Standard-normal [N, T] tensor."""
    return torch.randn(N, T)


# ─────────────────────────────────────────────────────────────────────────────
# Property 6: TS_RANK Output Range ∈ [0, 1)
# ─────────────────────────────────────────────────────────────────────────────

@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=10, max_value=100),
    d=st.sampled_from([5, 10, 20]),
)
@settings(max_examples=50)
def test_ts_rank_range(N: int, T: int, d: int) -> None:
    """
    Property 6: TS_RANK Output Range Constraint

    For any tensor x of shape [N, T] and window size d ∈ {5, 10, 20},
    _ts_rank(x, d) must satisfy: all values ∈ [0.0, 1.0).

    **Validates: 需求 F2.4**
    """
    assume(T >= d)  # window must fit in T

    torch.manual_seed(0)
    x = _randn_tensor(N, T)
    out = _ts_rank(x, d)

    assert out.shape == (N, T), (
        f"Shape mismatch: expected ({N}, {T}), got {tuple(out.shape)}"
    )
    assert (out >= 0.0).all(), (
        f"TS_RANK(d={d}) produced values < 0: min={out.min().item():.6f}"
    )
    assert (out < 1.0).all(), (
        f"TS_RANK(d={d}) produced values >= 1.0: max={out.max().item():.6f}"
    )


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=10, max_value=100),
)
@settings(max_examples=50)
def test_ts_rank_range_all_windows_via_ops_config(N: int, T: int) -> None:
    """
    Property 6 (via OPS_CONFIG): all TS_RANK_* operators satisfy [0, 1) bound.

    **Validates: 需求 F2.4**
    """
    torch.manual_seed(1)
    x = _randn_tensor(N, T)

    for name, fn, arity in _ALL_OPS:
        if not name.startswith("TS_RANK"):
            continue
        out = fn(x)
        assert (out >= 0.0).all(), (
            f"{name}: values < 0 found, min={out.min().item():.6f}"
        )
        assert (out < 1.0).all(), (
            f"{name}: values >= 1.0 found, max={out.max().item():.6f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 7: TS_CORR_10 Boundedness ∈ [-1, 1] & constant-input → 0
# ─────────────────────────────────────────────────────────────────────────────

@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=10, max_value=100),
)
@settings(max_examples=50)
def test_ts_corr_10_bounded(N: int, T: int) -> None:
    """
    Property 7a: TS_CORR_10 Correlation Coefficient Boundedness

    For any x, y of shape [N, T], _ts_corr_10(x, y) must satisfy:
    all values ∈ [-1.0 - ε, 1.0 + ε] where ε = 1e-5 (numerical tolerance).

    **Validates: 需求 F2.5**
    """
    torch.manual_seed(2)
    x = _randn_tensor(N, T)
    y = _randn_tensor(N, T)
    out = _ts_corr_10(x, y)

    assert out.shape == (N, T), (
        f"Shape mismatch: expected ({N}, {T}), got {tuple(out.shape)}"
    )
    assert (out >= -1.0 - 1e-5).all(), (
        f"TS_CORR_10 below -1: min={out.min().item():.6f}"
    )
    assert (out <= 1.0 + 1e-5).all(), (
        f"TS_CORR_10 above +1: max={out.max().item():.6f}"
    )


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=10, max_value=100),
    const_val=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_ts_corr_10_constant_x_outputs_zero_after_warmup(
    N: int, T: int, const_val: float
) -> None:
    """
    Property 7b: TS_CORR_10 outputs 0 when x is constant (after window warm-up).

    The causal window (d=10) pads d-1=9 zeros on the left, so the window is
    fully filled with the constant value starting at t=9 (0-indexed).
    From that point on std(x_window) < 1e-6 → mask → output = 0.

    **Validates: 需求 F2.5**
    """
    assume(abs(const_val) < 1e3)  # avoid degenerate floats

    torch.manual_seed(3)
    x = torch.full((N, T), const_val)
    y = _randn_tensor(N, T)
    out = _ts_corr_10(x, y)

    # Window d=10 fully fills from index 9 onward
    warm_start = 9
    if T > warm_start:
        warmed = out[:, warm_start:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"TS_CORR_10 with constant x (val={const_val:.3f}): "
            f"expected 0 from t={warm_start}, "
            f"max abs={warmed.abs().max().item():.2e}"
        )


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=10, max_value=100),
    const_val=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_ts_corr_10_constant_y_outputs_zero_after_warmup(
    N: int, T: int, const_val: float
) -> None:
    """
    Property 7c: TS_CORR_10 outputs 0 when y is constant (after window warm-up).

    **Validates: 需求 F2.5**
    """
    assume(abs(const_val) < 1e3)

    torch.manual_seed(4)
    x = _randn_tensor(N, T)
    y = torch.full((N, T), const_val)
    out = _ts_corr_10(x, y)

    warm_start = 9
    if T > warm_start:
        warmed = out[:, warm_start:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"TS_CORR_10 with constant y (val={const_val:.3f}): "
            f"expected 0 from t={warm_start}, "
            f"max abs={warmed.abs().max().item():.2e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 8: Temporal Operators Produce No NaN or Inf
# ─────────────────────────────────────────────────────────────────────────────

@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=10, max_value=100),
)
# deadline=None：每例串行执行全部 62 个算子，高负载下会超 hypothesis 默认
# 200ms/例 deadline 造成闪断；deadline 是性能护栏，放宽不降低断言强度。
@settings(max_examples=50, deadline=None)
def test_all_ts_ops_no_nan_inf_random(N: int, T: int) -> None:
    """
    Property 8a: All operators produce no NaN or Inf (random normal input).

    2026-07-27 对齐 fork 当前算子语义：由陈旧切片 OPS_CONFIG[12:]（缺三元分派，
    IF_GT 即崩）改为全表遍历 + arity 分派，覆盖 62 算子（见文件头）。

    **Validates: 需求 F2.6**
    """
    torch.manual_seed(5)
    x = _randn_tensor(N, T)
    y = _randn_tensor(N, T)
    z = _randn_tensor(N, T)

    for name, fn, arity in _ALL_OPS:
        out = _invoke_op(fn, arity, x, y, z)

        assert not torch.isnan(out).any(), (
            f"{name}: NaN found in output (random input, N={N}, T={T})"
        )
        assert not torch.isinf(out).any(), (
            f"{name}: Inf found in output (random input, N={N}, T={T})"
        )


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=10, max_value=100),
)
# deadline=None：理由同 8a（每例全表 62 算子，防高负载闪断，不降断言强度）。
@settings(max_examples=50, deadline=None)
def test_all_ts_ops_no_nan_inf_zeros(N: int, T: int) -> None:
    """
    Property 8b: All operators produce no NaN or Inf (all-zeros input).

    全零输入同时覆盖 DIV/SCALE/CS_* 等的除零防护路径。
    2026-07-27 对齐 fork 当前算子语义：全表遍历 + arity 分派（见文件头）。

    **Validates: 需求 F2.6**
    """
    x = torch.zeros(N, T)
    y = torch.zeros(N, T)
    z = torch.zeros(N, T)

    for name, fn, arity in _ALL_OPS:
        out = _invoke_op(fn, arity, x, y, z)

        assert not torch.isnan(out).any(), (
            f"{name}: NaN found with zero input (N={N}, T={T})"
        )
        assert not torch.isinf(out).any(), (
            f"{name}: Inf found with zero input (N={N}, T={T})"
        )


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=10, max_value=100),
)
# deadline=None：理由同 8a（每例全表 62 算子，防高负载闪断，不降断言强度）。
@settings(max_examples=50, deadline=None)
def test_all_ts_ops_no_nan_inf_large_values(N: int, T: int) -> None:
    """
    Property 8c: All operators produce no NaN or Inf (very large values 1e8).

    极大值输入覆盖 PRODUCT_5/POWER 等潜在 float32 溢出路径。
    2026-07-27 对齐 fork 当前算子语义：全表遍历 + arity 分派（见文件头）。

    **Validates: 需求 F2.6**
    """
    x = torch.full((N, T), 1e8)
    y = torch.full((N, T), -1e8)
    z = torch.full((N, T), 1e8)

    for name, fn, arity in _ALL_OPS:
        out = _invoke_op(fn, arity, x, y, z)

        assert not torch.isnan(out).any(), (
            f"{name}: NaN found with large-value input (N={N}, T={T})"
        )
        assert not torch.isinf(out).any(), (
            f"{name}: Inf found with large-value input (N={N}, T={T})"
        )


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=10, max_value=100),
    const_val=st.floats(
        min_value=-1e6, max_value=1e6,
        allow_nan=False, allow_infinity=False,
    ),
)
# deadline=None：理由同 8a（每例全表 62 算子，防高负载闪断，不降断言强度）。
@settings(max_examples=50, deadline=None)
def test_all_ts_ops_no_nan_inf_constant_input(N: int, T: int, const_val: float) -> None:
    """
    Property 8d: All operators produce no NaN or Inf (constant-value input).

    This tests the degenerate case where std=0 inside rolling windows, which
    is the main source of division-by-zero risk.
    2026-07-27 对齐 fork 当前算子语义：全表遍历 + arity 分派（见文件头）。

    **Validates: 需求 F2.6**
    """
    x = torch.full((N, T), const_val)
    y = torch.full((N, T), const_val + 1.0)  # distinct constant for TS_CORR_10
    z = torch.full((N, T), const_val - 1.0)  # 三元算子的第三操作数（同为退化常数）

    for name, fn, arity in _ALL_OPS:
        out = _invoke_op(fn, arity, x, y, z)

        assert not torch.isnan(out).any(), (
            f"{name}: NaN found with constant input (val={const_val:.3g}, N={N}, T={T})"
        )
        assert not torch.isinf(out).any(), (
            f"{name}: Inf found with constant input (val={const_val:.3g}, N={N}, T={T})"
        )
