
# from . import optimizer
from .backend import add
from .backend import multiply
from .core import PaperMatrix
from .backend import execute_fused_add_multiply
from .config import TILE_SIZE, DEFAULT_CACHE_SIZE_TILES

from .buffer import BufferManager

use_buffer_manager = True  # Set to False to disable buffer manager

class Plan:
    """Represents a computation that will result in a matrix, but is not yet executed."""
    def __init__(self, op):
        # The 'op' is the plan for this matrix
        self.op = op
        self.shape = op.shape
    
    def compute(self, output_path, cache_size_tiles=None):
        """Triggers the execution of the entire computation plan."""
        
        from .optimizer import generate_io_trace

        io_trace = generate_io_trace(self)
        print(f"io_trace: {io_trace}")
        buffer_manager = None
        if use_buffer_manager:
            cache_size = cache_size_tiles if cache_size_tiles is not None else DEFAULT_CACHE_SIZE_TILES
            buffer_manager = BufferManager(max_cache_size_tiles=cache_size, io_trace=io_trace)
            print(f"Using buffer manager with cache size: {cache_size} tiles")
        else:
            print(" - No buffer manager used. Will read/write directly to disk.")
        
        result_matrix = self.op.execute(output_path, buffer_manager)
        # Return both the result matrix and the buffer manager for further use    
        return result_matrix, buffer_manager

    def __repr__(self):
        return f"Plan(plan={self.op!r})"

    def __add__(self, x: 'Plan'):
        print("Build an 'AddNode' plan...")
        # return Plan(AddNode(self, x))
        return Plan(AddNode(self.op, x.op))
    
    def __matmul__(self, x: 'Plan'):
        print("Build an 'MultiplyNode' plan")
        # return Plan(MultiplyNode(self, x))
        return Plan(MultiplyNode(self.op, x.op))

    def __mul__(self, x):
        if isinstance(x, (int, float)):
            return Plan(MultiplyScalarNode(self.op, x))
        raise NotImplementedError("Only scalar multiplication is supported.")
    
    def __rmul__(self, x):
        # Handles the case `2 * my_lazy_matrix`
        return self.__mul__(x)


class EagerNode:
    """An operation that represents an already computed matrix on disk. This is the leaf of our plan."""
    def __init__(self, matrix: PaperMatrix):
        self.matrix = matrix
        self.shape = matrix.shape

    def __repr__(self):
        return f"EagerNode(path='{self.matrix.filepath}')"
    
    def execute(self, path, buffer_manager):
        """Executing a leaf node simply means returning the handle to the existing matrix."""
        print(f" - Executing EagerNode: Providing handle to '{self.matrix.filepath}")
        return self.matrix

class AddNode:
    """An operation node representing addition in our computation plan."""
    def __init__(self, left: 'Plan', right: 'Plan'):
        if left.shape != right.shape:
            raise ValueError("Shapes must match for lazy addition.")
        self.left = left
        self.right = right
        self.shape = left.shape
    
    def __repr__(self):
        # !r calls the repr() of the inner objects, creating a nested view
        return f"AddNode(left={self.left!r}, right={self.right!r})"

    def execute(self, output_path, buffer_manager):
        """Executes the addition plan."""
        print(" - Execute AddNode: Get inputs...")
        # Create temporary paths for intermediate results if needed
        matrix_A = self.left.execute(None if output_path is None else output_path + ".left.tmp", buffer_manager)
        matrix_B = self.right.execute(None if output_path is None else output_path + ".right.tmp", buffer_manager)

        print(" - Calling 'add' to perform the computation...")
        return add(matrix_A, matrix_B, output_path, buffer_manager)

class MultiplyNode:
    """An operation node representing matrix multiplication in our computation plan."""
    def __init__(self, left: 'Plan', right: 'Plan'):
        if left.shape[1] != right.shape[0]:
            raise ValueError("Inner dimensions must match for matrix multiplication.")
        self.left = left
        self.right = right
        self.shape = (left.shape[0], right.shape[1])
    
    def __repr__(self):
        # !r calls the repr() of the inner objects, creating a nested view
        return f"MultiplyNode(left={self.left!r}, right={self.right!r})"

    def execute(self, output_path, buffer_manager):
        """Executes the multiplication plan."""
        print(" - Execute MultiplyNode: Get inputs...")
        # Create temporary paths for intermediate results if needed
        matrix_A = self.left.execute(None if output_path is None else output_path + ".left.tmp", buffer_manager)
        matrix_B = self.right.execute(None if output_path is None else output_path + ".right.tmp", buffer_manager)

        print(" - Calling 'multiply' to perform the computation...")
        return multiply(matrix_A, matrix_B, output_path, buffer_manager)

class MultiplyScalarNode:
    """An operation node representing multiplication by a scalar."""
    def __init__(self, left: 'Plan', right: float):
        self.left = left
        self.right = right
        self.shape = left.shape
    
    def __repr__(self):
        return f"MultiplyScalarNode(left={self.left!r}, scalar={self.right})"

    def execute(self, output_path, buffer_manager):
        """This is our 'mini-optimizer'. It checks if it can use a fast, fused kernel."""
        # THE OPTIMIZATION RULE:
        # If my left input is an addition operation...
        if isinstance(self.left, AddNode):
            print("Optimizer: Fused Add-Multiply pattern detected! Calling fast kernel.")
            # ...then call the special fused execution function
            add_op = self.left
            # Get the actual matrix handles from the eager nodes
            matrix_A = add_op.left.execute(None if output_path is None else output_path + ".fused.A.tmp", buffer_manager)
            matrix_B = add_op.right.execute(None if output_path is None else output_path + ".fused.B.tmp", buffer_manager)
            return execute_fused_add_multiply(
                matrix_A, # Matrix A
                matrix_B, # Matrix B
                self.right, # The scalar value
                output_path,
                buffer_manager
            )
        else:
            # Fallback to general case (non-fused)
            print("Optimizer: No fusion pattern detected. Executing step-by-step.")
            # 1. Compute the input matrix first
            TMP = self.left.execute(output_path + ".tmp", buffer_manager)
            # 2. Then, perform the scalar multiplication
            C = PaperMatrix(output_path, self.shape, mode='w+')
            for r in range(0, self.shape[0], TILE_SIZE):
                r_end = min(r + TILE_SIZE, self.shape[0])
                for c in range(0, self.shape[1], TILE_SIZE):
                    c_end = min(c + TILE_SIZE, self.shape[1])
                    C.data[r:r_end, c:c_end] = TMP.data[r:r_end, c:c_end] * self.right
            
            C.data.flush()
            # Only close TMP if it's not from an EagerNode (i.e., it was computed, not an input)
            if not isinstance(self.left, EagerNode):
                TMP.close()
            return C

