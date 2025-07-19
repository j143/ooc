# --- Purpose: To inspect a plan and choose the best execution strategy. ---

from . import backend
from .plan import Plan, EagerMatrix, AddOp, MultiplyOp, MultiplyScalarOp
# The rule registry
# Pattern: (OuterOp, InnerOp), Kernel: function_to_execute
FUSION_RULES = [
    ((MultiplyScalarOp, AddOp), execute_fused_add_multiply),
    # Future rules go here, e.g., ((AddOp, MultiplyOp), fused_kernel_2)
]

def execute(plan, output_path: str):
    """
    The main optimizer entry point. It checks for patterns and executes.
    1. Checks the plan against FUSION_RULES.
    2. If a rule matches, calls the corresponding fast kernel from lazy.py/eager.py.
    3. If no rules match, executes the plan step-by-step (default path).
    """
    # Check if the plan's op matches any fusion rule
    for pattern, kernel in FUSION_RULES:
        OuterOp, InnerOp = pattern
        if isinstance(plan.op, OuterOp) and isinstance(plan.op.left.op, InnerOp):
            print(f"✨ Optimizer: Found pattern {pattern}. Using fused kernel.")
            # Call the specialized kernel
            return kernel(...) # You'll need to pass the right components from the plan

    # If no special rule matches, execute the default, non-fused way
    print("Optimizer: No fusion pattern found. Executing default.")
    return plan.op.execute(output_path) # The original, slower path
