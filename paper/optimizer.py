# --- Purpose: To inspect a plan and choose the best execution strategy. ---

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
from enum import Enum

from . import backend
from .plan import Plan, EagerNode, AddNode, MultiplyNode, MultiplyScalarNode

from .buffer import BufferManager
from .config import TILE_SIZE

# Configure logging
logger = logging.getLogger(__name__)


class FusionPattern(Enum):
    """Enumeration of recognized fusion patterns."""
    FUSED_ADD_MULTIPLY = "add_multiply"
    FUSED_MATMUL_SCALAR = "matmul_scalar"
    FUSED_ADD_MATMUL = "add_matmul"
    FUSED_DOUBLE_SCALAR = "double_scalar"
    NONE = "none"


@dataclass
class MatchResult:
    """
    Result of pattern matching analysis.
    Contains metadata about detected patterns without materializing data.
    """
    pattern: FusionPattern
    outer_op_type: Optional[type]
    inner_op_type: Optional[type]
    kernel_function: Optional[callable]
    # Metadata about inputs (no actual matrices)
    input_shapes: List[Tuple[int, int]]
    parameters: dict  # e.g., {'scalar': 2.5} or {'scalar1': 1.5, 'scalar2': 2.0}
    
    @property
    def is_fusable(self) -> bool:
        """Returns True if a fusion pattern was detected."""
        return self.pattern != FusionPattern.NONE
    
    def __repr__(self):
        if self.is_fusable:
            return f"MatchResult(pattern={self.pattern.value}, shapes={self.input_shapes}, params={self.parameters})"
        return "MatchResult(pattern=none)"


@dataclass
class CostEstimate:
    """
    Cost model for execution estimation.
    """
    io_operations: int  # Number of I/O operations
    compute_operations: int  # Number of compute operations
    estimated_io_bytes: int  # Estimated I/O in bytes
    cache_benefit: float  # Estimated cache hit improvement (0-1)
    
    @property
    def total_cost(self) -> float:
        """Simple cost model: weighted sum of I/O and compute."""
        # I/O is typically 100x more expensive than compute
        io_weight = 100.0
        compute_weight = 1.0
        return (self.io_operations * io_weight + 
                self.compute_operations * compute_weight)
    
    def __repr__(self):
        return (f"CostEstimate(io_ops={self.io_operations}, "
                f"compute_ops={self.compute_operations}, "
                f"io_bytes={self.estimated_io_bytes}, "
                f"total_cost={self.total_cost:.0f})")

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
    left_trace = []
    if hasattr(op_node, 'left'):
        left_trace = _generate_trace_recursive(op_node.left)
    right_trace = []
    if hasattr(op_node, 'right'):
        right_trace = _generate_trace_recursive(op_node.right)
    
    # Trace for the current kernel operation
    kernel_trace = []

    def find_leaf_matrices(node):
        if isinstance(node, EagerNode):
            return [node.matrix]
        leaves = []
        if hasattr(node, 'left'):
            leaves.extend(find_leaf_matrices(node.left))
        if hasattr(node, 'right'):
            leaves.extend(find_leaf_matrices(node.right))
        return leaves

    if isinstance(op_node, MultiplyNode):
        left_leaves = find_leaf_matrices(op_node.left)
        right_leaves = find_leaf_matrices(op_node.right)
        
        A_shape = op_node.left.shape
        B_shape = op_node.right.shape
        C_shape = (A_shape[0], B_shape[1])

        # This assumes for A@B, A comes from left leaves and B from right leaves.
        if left_leaves and right_leaves:
            A_matrix = left_leaves[0]
            B_matrix = right_leaves[0]
            for r_start in range(0, C_shape[0], TILE_SIZE):
                for c_start in range(0, C_shape[1], TILE_SIZE):
                    for k_start in range(0, A_shape[1], TILE_SIZE):
                        kernel_trace.append((os.path.basename(A_matrix.filepath), r_start, k_start))
                        kernel_trace.append((os.path.basename(B_matrix.filepath), k_start, c_start))
    
    elif isinstance(op_node, AddNode):
        left_leaves = find_leaf_matrices(op_node.left)
        right_leaves = find_leaf_matrices(op_node.right)
        if len(left_leaves) == 1 and len(right_leaves) == 1:
            A, B = left_leaves[0], right_leaves[0]
            for r_start in range(0, A.shape[0], TILE_SIZE):
                for c_start in range(0, A.shape[1], TILE_SIZE):
                    kernel_trace.append((os.path.basename(A.filepath), r_start, c_start))
                    kernel_trace.append((os.path.basename(B.filepath), r_start, c_start))
    
    elif isinstance(op_node, MultiplyScalarNode):
        left_leaves = find_leaf_matrices(op_node.left)
        if len(left_leaves) == 1:
            A = left_leaves[0]
            for r_start in range(0, A.shape[0], TILE_SIZE):
                for c_start in range(0, A.shape[1], TILE_SIZE):
                    kernel_trace.append((os.path.basename(A.filepath), r_start, c_start))

    return left_trace + right_trace + kernel_trace

def generate_io_trace(plan: Plan) -> list:
    """
    Generates a complete, ordered list of all tile accesses (the I/O trace)
    given a computation plan.
    """
    logger.info("Generating I/O trace for optimal caching...")
    logger.debug(f"Plan structure: {plan}")

    trace = _generate_trace_recursive(plan.op)
    result = [(os.path.basename(path), r, c) for path, r, c in trace]
    
    logger.info(f"Generated I/O trace with {len(result)} tile accesses")
    return result


def analyze(plan: Plan) -> Tuple[List, MatchResult]:
    """
    Stage 1: Analyze the plan without executing anything.
    
    Returns:
        - I/O trace (list of tile accesses)
        - MatchResult (detected fusion patterns with metadata only)
    """
    logger.info("=== STAGE 1: ANALYZE ===")
    
    # Generate I/O trace
    io_trace = generate_io_trace(plan)
    
    # Pattern matching (metadata only - no execute() calls)
    match_result = _detect_fusion_pattern(plan)
    
    logger.info(f"Analysis complete: {match_result}")
    return io_trace, match_result


def _detect_fusion_pattern(plan: Plan) -> MatchResult:
    """
    Detect fusion patterns using metadata only (no execute() calls).
    This is purely analytical - inspects the plan tree structure.
    """
    logger.debug("Detecting fusion patterns...")
    
    # Check against each fusion rule
    for pattern, kernel in FUSION_RULES:
        OuterOp, InnerOp = pattern
        
        # Pattern detection using isinstance checks (no execution)
        # Note: plan.op contains the operation node, not another Plan
        if isinstance(plan.op, OuterOp) and hasattr(plan.op, 'left'):
            # plan.op.left is the inner operation node (not a Plan)
            if isinstance(plan.op.left, InnerOp):
                # Pattern matched! Extract metadata
                logger.debug(f"Pattern detected: {OuterOp.__name__}({InnerOp.__name__})")
                
                # Determine pattern type and extract parameters
                if OuterOp == MultiplyScalarNode and InnerOp == AddNode:
                    add_node = plan.op.left  # This is the AddNode
                    return MatchResult(
                        pattern=FusionPattern.FUSED_ADD_MULTIPLY,
                        outer_op_type=OuterOp,
                        inner_op_type=InnerOp,
                        kernel_function=kernel,
                        input_shapes=[add_node.left.shape, add_node.right.shape],
                        parameters={'scalar': plan.op.right}
                    )
                
                elif OuterOp == MultiplyScalarNode and InnerOp == MultiplyNode:
                    mul_node = plan.op.left  # This is the MultiplyNode
                    return MatchResult(
                        pattern=FusionPattern.FUSED_MATMUL_SCALAR,
                        outer_op_type=OuterOp,
                        inner_op_type=InnerOp,
                        kernel_function=kernel,
                        input_shapes=[mul_node.left.shape, mul_node.right.shape],
                        parameters={'scalar': plan.op.right}
                    )
                
                elif OuterOp == MultiplyNode and InnerOp == AddNode:
                    add_node = plan.op.left  # This is the AddNode
                    return MatchResult(
                        pattern=FusionPattern.FUSED_ADD_MATMUL,
                        outer_op_type=OuterOp,
                        inner_op_type=InnerOp,
                        kernel_function=kernel,
                        input_shapes=[add_node.left.shape, add_node.right.shape, plan.op.right.shape],
                        parameters={}
                    )
                
                elif OuterOp == MultiplyScalarNode and InnerOp == MultiplyScalarNode:
                    inner_mul = plan.op.left  # This is the inner MultiplyScalarNode
                    return MatchResult(
                        pattern=FusionPattern.FUSED_DOUBLE_SCALAR,
                        outer_op_type=OuterOp,
                        inner_op_type=InnerOp,
                        kernel_function=kernel,
                        input_shapes=[inner_mul.left.shape],
                        parameters={'scalar1': inner_mul.right, 'scalar2': plan.op.right}
                    )
    
    # No pattern matched
    logger.debug("No fusion pattern detected")
    return MatchResult(
        pattern=FusionPattern.NONE,
        outer_op_type=None,
        inner_op_type=None,
        kernel_function=None,
        input_shapes=[],
        parameters={}
    )


def estimate_cost(plan: Plan, match_result: MatchResult) -> CostEstimate:
    """
    Estimate execution cost for a plan.
    
    Args:
        plan: The computation plan
        match_result: Result from pattern analysis
        
    Returns:
        CostEstimate with predicted I/O and compute costs
    """
    logger.debug("Estimating execution cost...")
    
    # Calculate number of tiles
    def count_tiles(shape):
        rows_tiles = (shape[0] + TILE_SIZE - 1) // TILE_SIZE
        cols_tiles = (shape[1] + TILE_SIZE - 1) // TILE_SIZE
        return rows_tiles * cols_tiles
    
    io_ops = 0
    compute_ops = 0
    io_bytes = 0
    
    if match_result.is_fusable:
        # Fused operations reduce I/O by combining operations
        for shape in match_result.input_shapes:
            tiles = count_tiles(shape)
            io_ops += tiles  # Read each input once
            io_bytes += shape[0] * shape[1] * 4  # float32
            compute_ops += tiles
        
        # Fusion benefit: ~30% fewer I/O operations
        cache_benefit = 0.3
    else:
        # Unfused: each operation reads and writes separately
        shape = plan.shape
        tiles = count_tiles(shape)
        # Estimate: 2x I/O for unfused (read + write per operation)
        io_ops = tiles * 2
        io_bytes = shape[0] * shape[1] * 4 * 2
        compute_ops = tiles
        cache_benefit = 0.0
    
    estimate = CostEstimate(
        io_operations=io_ops,
        compute_operations=compute_ops,
        estimated_io_bytes=io_bytes,
        cache_benefit=cache_benefit
    )
    
    logger.info(f"Cost estimate: {estimate}")
    return estimate


def rewrite(plan: Plan, match_result: MatchResult) -> Plan:
    """
    Stage 2: Rewrite the plan based on match results.
    
    Currently returns the original plan (future: create fused node IR).
    In a full implementation, this would create new FusedNode types.
    
    Args:
        plan: Original computation plan
        match_result: Pattern matching results from analyze()
        
    Returns:
        Rewritten plan (potentially with fused operations)
    """
    logger.info("=== STAGE 2: REWRITE ===")
    
    if match_result.is_fusable:
        logger.info(f"Plan will use fused kernel: {match_result.pattern.value}")
        # Future: Create FusedNode IR representation here
        # For now, we return the original plan and handle fusion in execute()
    else:
        logger.info("Plan will execute unfused (no optimization)")
    
    return plan


def execute_plan(plan: Plan, match_result: MatchResult, output_path: str, 
                 buffer_manager: BufferManager | None):
    """
    Stage 3: Execute the plan using the analysis results.
    
    Args:
        plan: The computation plan
        match_result: Results from analyze() stage
        output_path: Where to write the result
        buffer_manager: Optional buffer manager for caching
        
    Returns:
        Computed PaperMatrix result
    """
    logger.info("=== STAGE 3: EXECUTE ===")
    
    if match_result.is_fusable:
        logger.info(f"✨ Executing fused kernel: {match_result.pattern.value}")
        
        # Extract leaf matrices (now that we're actually executing)
        def get_leaf_matrix(node):
            """Helper to get the PaperMatrix from an EagerNode."""
            if isinstance(node, EagerNode):
                return node.matrix
            # Recursively execute to get the matrix
            return node.execute(None, buffer_manager)
        
        # Execute based on pattern
        if match_result.pattern == FusionPattern.FUSED_ADD_MULTIPLY:
            add_node = plan.op.left  # AddNode
            matrix_A = get_leaf_matrix(add_node.left)
            matrix_B = get_leaf_matrix(add_node.right)
            scalar = match_result.parameters['scalar']
            return match_result.kernel_function(matrix_A, matrix_B, scalar, output_path, buffer_manager)
        
        elif match_result.pattern == FusionPattern.FUSED_MATMUL_SCALAR:
            mul_node = plan.op.left  # MultiplyNode
            matrix_A = get_leaf_matrix(mul_node.left)
            matrix_B = get_leaf_matrix(mul_node.right)
            scalar = match_result.parameters['scalar']
            return match_result.kernel_function(matrix_A, matrix_B, scalar, output_path)
        
        elif match_result.pattern == FusionPattern.FUSED_ADD_MATMUL:
            add_node = plan.op.left  # AddNode
            matrix_A = get_leaf_matrix(add_node.left)
            matrix_B = get_leaf_matrix(add_node.right)
            matrix_C = get_leaf_matrix(plan.op.right)
            return match_result.kernel_function(matrix_A, matrix_B, matrix_C, output_path)
        
        elif match_result.pattern == FusionPattern.FUSED_DOUBLE_SCALAR:
            inner_mul = plan.op.left  # Inner MultiplyScalarNode
            matrix_A = get_leaf_matrix(inner_mul.left)
            scalar1 = match_result.parameters['scalar1']
            scalar2 = match_result.parameters['scalar2']
            return match_result.kernel_function(matrix_A, scalar1, scalar2, output_path)
    
    # Unfused execution
    logger.info("Executing unfused plan")
    return plan.op.execute(output_path, buffer_manager)


def execute(plan, output_path: str, buffer_manager: BufferManager | None):
    """
    Legacy entry point for backward compatibility.
    Now uses the three-stage optimizer pipeline.
    
    The new three-stage approach:
    1. analyze() - Generate I/O trace and detect patterns (no execution)
    2. rewrite() - Transform plan based on patterns (future: create IR)
    3. execute_plan() - Execute using analysis results
    
    DEPRECATED: Use analyze() → rewrite() → execute_plan() for new code.
    """
    logger.warning("Using legacy execute() - consider migrating to three-stage pipeline")
    
    # Three-stage pipeline
    io_trace, match_result = analyze(plan)
    cost = estimate_cost(plan, match_result)
    rewritten_plan = rewrite(plan, match_result)
    return execute_plan(rewritten_plan, match_result, output_path, buffer_manager)
