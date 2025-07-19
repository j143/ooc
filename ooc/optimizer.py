
# The rule registry
# Pattern: (OuterOp, InnerOp), Kernel: function_to_execute
FUSION_RULES = [
    ((MultiplyScalarOp, AddOp), execute_fused_add_multiply),
    # Future rules go here, e.g., ((AddOp, MultiplyOp), fused_kernel_2)
]

