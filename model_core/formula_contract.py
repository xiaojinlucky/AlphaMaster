"""公式执行语义的固定兼容合同。"""

STACKVM_OUTPUT_NORM_WINDOW = 200
STACKVM_OUTPUT_NORM_EPS = 1e-6
STACKVM_OUTPUT_NORM_CLIP = 3.0

# 任何会改变同一公式 token 执行结果的修改都必须更新此值，使旧产物失败关闭。
FORMULA_EXECUTION_CONTRACT = (
    "stackvm-output-v1:"
    "causal-rolling-zscore-w200-population:"
    "current-cross-section-fallback-time-series:"
    "clip-3"
)
