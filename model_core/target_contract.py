"""目标收益的统一时间合同。

当前目标定义为::

    target_ret[t] = log(open[t+2] / open[t+1])

因此 factor[t] 必须与 target_ret[t] 同索引配对，原始序列末尾两根没有可实现
收益，任何训练评分、回测收益、成本和评估指标都必须先裁掉这两根。
"""
from __future__ import annotations

import torch


TARGET_RETURN_HORIZON = 2
SCORING_CONTRACT_VERSION = "open_t1_t2_same_index_tail2_v1"


class ScoringContractMismatchError(ValueError):
    """产物的评分合同与当前运行时不一致。"""


def validate_target_horizon(horizon: int) -> int:
    """严格校验前视跨度，禁止 bool、浮点数和字符串被静默转换。"""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("target horizon 必须是严格正整数")
    return horizon


def valid_target_length(
    total_steps: int,
    horizon: int = TARGET_RETURN_HORIZON,
) -> int:
    """返回原始目标序列中具有真实未来收益的时间步数。"""
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 0:
        raise ValueError("目标序列长度必须是非负整数")
    return max(0, total_steps - validate_target_horizon(horizon))


def align_target_return_window(
    candidate: torch.Tensor,
    target: torch.Tensor,
    horizon: int = TARGET_RETURN_HORIZON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按同一时钟配对 candidate[t] 与 target[t]，并裁掉目标尾部占位值。"""
    if candidate.ndim != 2 or target.ndim != 2:
        raise ValueError("candidate 和 target 必须都是 [N, T] 二维张量")
    if candidate.shape != target.shape:
        raise ValueError(
            "candidate 和 target 形状必须完全一致，"
            f"实际为 {tuple(candidate.shape)} 与 {tuple(target.shape)}"
        )
    length = valid_target_length(target.shape[1], horizon)
    return candidate[:, :length], target[:, :length]
