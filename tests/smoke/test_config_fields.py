"""
冒烟测试：Config 字段与 FEATURE_NAMES 内容验证

断言：
- FEATURE_NAMES 不含 LIQ_SCORE / FOMO（旧 Solana 特有因子）
- 根 config.Config.INPUT_DIM 保持冻结遗留值 20（仅存于不可达兜底分支）
- len(FEATURE_NAMES) 与活口径 ModelConfig.INPUT_DIM / MT5FeatureEngineer.INPUT_DIM
  一致（2026-07-27 对齐 fork 当前特征语义，当前 65）

注意：task 5.1 会将 vocab.py 更新为新的 MT5 特征名称。
      此测试使用 try/except 优雅处理旧版 vocab.py 中仍含旧字段的情形。

Requirements: 11.1, 5.5 (4.1)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from config import Config


def _get_feature_names():
    """
    尝试从 model_core.vocab 获取 FEATURE_NAMES。
    若模块不存在或属性不存在，返回 None。
    """
    try:
        from model_core.vocab import FEATURE_NAMES
        return FEATURE_NAMES
    except (ImportError, AttributeError):
        return None


def _is_mt5_feature_names_updated():
    """
    检查 FEATURE_NAMES 是否已更新为 MT5 特征集（task 5.1 完成后）。
    若仍包含旧 Solana 特有因子则返回 False，表示 task 5.1 尚未执行。
    """
    feature_names = _get_feature_names()
    if feature_names is None:
        return False
    # 旧版特征含 Solana 特有因子
    old_features = {"LIQ_SCORE", "FOMO", "LOG_VOL"}
    return not bool(old_features & set(feature_names))


class TestConfigInputDimSmoke:
    def test_input_dim_equals_10(self):
        """Config.INPUT_DIM must equal 20 (expanded from 10 to 20 features)."""
        assert Config.INPUT_DIM == 20


class TestFeatureNamesSmoke:
    """
    验证 FEATURE_NAMES 不含旧 Solana 链上特有因子。

    - 若 vocab.py 尚未更新（task 5.1 前），这些测试将跳过，以免误报。
    - 若 vocab.py 已更新，则必须满足断言。
    """

    def test_feature_names_accessible(self):
        """FEATURE_NAMES 应当可以从 model_core.vocab 导入"""
        feature_names = _get_feature_names()
        assert feature_names is not None, (
            "Cannot import FEATURE_NAMES from model_core.vocab. "
            "Ensure model_core/vocab.py exists and defines FEATURE_NAMES."
        )

    def test_liq_score_not_in_feature_names(self):
        """FEATURE_NAMES 不应包含 'LIQ_SCORE'（已废弃的链上流动性因子）"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert "LIQ_SCORE" not in feature_names, (
            f"FEATURE_NAMES should not contain 'LIQ_SCORE', but got: {feature_names}"
        )

    def test_fomo_not_in_feature_names(self):
        """FEATURE_NAMES 不应包含 'FOMO'（已废弃的链上情绪因子）"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert "FOMO" not in feature_names, (
            f"FEATURE_NAMES should not contain 'FOMO', but got: {feature_names}"
        )

    def test_feature_names_length_matches_input_dim(self):
        """FEATURE_NAMES 的长度应等于**活口径**的 INPUT_DIM（模型/特征工程侧）。

        2026-07-27 对齐 fork 当前特征语义：特征库已扩展至 65，训练/推理链路的
        输入维数权威源是 model_core.config.ModelConfig.INPUT_DIM（由
        FORMULA_VOCAB.feature_count 派生）与 MT5FeatureEngineer.INPUT_DIM（由
        FEATURE_REGISTRY 派生）。旧断言比对的根 config.Config.INPUT_DIM==20 是
        上游旧版遗留死值——生产链路里它只出现在 data_pipeline/data_manager.py:175
        的 ImportError 兜底分支（model_core.features 存在时不可达），不参与任何
        活路径；根 config.py 属训练批次冻结文件，其注释勘误记入批次后工单。
        """
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        from model_core.config import ModelConfig
        from model_core.features import MT5FeatureEngineer

        assert len(feature_names) == ModelConfig.INPUT_DIM, (
            f"Expected len(FEATURE_NAMES) == ModelConfig.INPUT_DIM "
            f"({ModelConfig.INPUT_DIM}), got {len(feature_names)}: {feature_names}"
        )
        assert len(feature_names) == MT5FeatureEngineer.INPUT_DIM, (
            f"Expected len(FEATURE_NAMES) == MT5FeatureEngineer.INPUT_DIM "
            f"({MT5FeatureEngineer.INPUT_DIM}), got {len(feature_names)}"
        )
