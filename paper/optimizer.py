# --- Purpose: To inspect a plan and choose the best execution strategy. ---

from . import backend
from .plan import Plan, EagerNode, AddNode, MultiplyNode, MultiplyScalarNode
# The rule registry
# Pattern: (OuterOp, InnerOp), Kernel: function_to_execute
FUSION_RULES = [
    ((MultiplyScalarNode, AddNode), backend.execute_fused_add_multiply),
    ((MultiplyScalarNode, MultiplyNode), backend.execute_fused_matmul_scalar),
    ((MultiplyNode, AddNode), backend.execute_fused_add_matmul),
    ((MultiplyScalarNode, MultiplyScalarNode), backend.execute_fused_double_scalar),
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
            
            # Pattern 1: (A + B) * scalar
            if OuterOp == MultiplyScalarNode and InnerOp == AddNode:
                add_node = plan.op.left.op
                matrix_A = add_node.left.op.execute(None)
                matrix_B = add_node.right.op.execute(None)
                scalar = plan.op.right
                return kernel(matrix_A, matrix_B, scalar, output_path)
            
            # Pattern 2: (A @ B) * scalar
            elif OuterOp == MultiplyScalarNode and InnerOp == MultiplyNode:
                mul_node = plan.op.left.op
                matrix_A = mul_node.left.op.execute(None)
                matrix_B = mul_node.right.op.execute(None)
                scalar = plan.op.right
                return kernel(matrix_A, matrix_B, scalar, output_path)
            
            # Pattern 3: (A + B) @ C
            elif OuterOp == MultiplyNode and InnerOp == AddNode:
                add_node = plan.op.left.op
                matrix_A = add_node.left.op.execute(None)
                matrix_B = add_node.right.op.execute(None)
                matrix_C = plan.op.right.op.execute(None)
                return kernel(matrix_A, matrix_B, matrix_C, output_path)
            
            # Pattern 4: (A * scalar1) * scalar2
            elif OuterOp == MultiplyScalarNode and InnerOp == MultiplyScalarNode:
                inner_mul_node = plan.op.left.op
                matrix_A = inner_mul_node.left.op.execute(None)
                scalar1 = inner_mul_node.right
                scalar2 = plan.op.right
                return kernel(matrix_A, scalar1, scalar2, output_path)

    # If no special rule matches, execute the default, non-fused way
    print("Optimizer: No fusion pattern found. Executing default.")
    return plan.op.execute(output_path) # The original, slower path
