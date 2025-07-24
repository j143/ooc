# --- Purpose: To inspect a plan and choose the best execution strategy. ---

from . import backend
from .plan import Plan, EagerNode, AddNode, MultiplyNode, MultiplyScalarNode

from .buffer import BufferManager, TILE_SIZE

# The rule registry
# Pattern: (OuterOp, InnerOp), Kernel: function_to_execute
FUSION_RULES = [
    ((MultiplyScalarNode, AddNode), backend.execute_fused_add_multiply),
    ((MultiplyScalarNode, MultiplyNode), backend.execute_fused_matmul_scalar),
    ((MultiplyNode, AddNode), backend.execute_fused_add_matmul),
    ((MultiplyScalarNode, MultiplyScalarNode), backend.execute_fused_double_scalar),
]

def _generate_trace_recursive(op_node):
    """
    A recursive helper to generate the I/O trace for a plan node.
    """
    if isinstance(op_node, EagerNode):
        # Eager nodes are leaves; so, doesn't have preceding operations
        return []
    
    # Recursively find the trace for the inputs
    left_trace = _generate_trace_recursive(op_node.left)
    right_trace = []
    if hasattr(op_node, 'right'):
        right_trace = _generate_trace_recursive(op_node.right)
    
    # Trace for the current kernel operation
    kernel_trace = []
    if isinstance(op_node, MultiplyNode):
        A = op_node.left.matrix
        B = op_node.right.matrix
        C_shape = (A.shape[0], B.shape[1])
        # simulate the backend.multiply kernel's access pattern
        for r_start in range(0, C_shape[0], TILE_SIZE):
            for c_start in range(0, C_shape[1], TILE_SIZE):
                for k_start in range(0, A.shape[1], TILE_SIZE):
                    kernel_trace.append((A.filepath, r_start, k_start))
                    kernel_trace.append((B.filepath, k_start, c_start))
    
    return left_trace + right_trace + kernel_trace

def generate_io_trace(plan: Plan) -> list:
    """
    Generates a complete, ordered list of all tile accesses (the I/O trace)
    given a computation plan.
    """
    print("Optimizer: Generating I/O trace for optimal caching...")

    trace = _generate_trace_recursive(plan.op_node)
    return [(os.path.basename(path), r, c) for path, r, c in trace]


def execute(plan, output_path: str, buffer_manager: BufferManager | None):
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
