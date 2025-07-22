
# from . import optimizer
from .backend import add
from .backend import multiply
from .core import PaperMatrix
from .backend import TILE_SIZE
from .backend import execute_fused_add_multiply

from .buffer import BufferManager

use_buffer_manager = False  # Set to False to disable buffer manager

class Plan:
    """Represents a computation that will result in a matrix, but is not yet executed."""
    def __init__(self, op):
        # The 'op' is the plan for this matrix
        self.op = op
        self.shape = op.shape
    
    def compute(self, output_path):
        """Triggers the execution of the entire computation plan."""
        buffer_manager = None
        if use_buffer_manager:
            buffer_manager = BufferManager(max_cache_size_tiles=64)
        else:
            print(" - No buffer manager used. Will read/write directly to disk.")
        
        result_matrix = self.op.execute(output_path, buffer_manager)
        return result_matrix

    def __repr__(self):
        return f"Plan(plan={self.op!r})"

    def __add__(self, x: 'Plan'):
        print("Build an 'AddNode' plan...")
        return Plan(AddNode(self, x))
    
    def __matmul__(self, x: 'Plan'):
        print("Build an 'MultiplyNode' plan")
        return Plan(MultiplyNode(self, x))

    def __mul__(self, x):
        if isinstance(x, (int, float)):
            return Plan(MultiplyScalarNode(self, x))
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
        return f"EagerNode(path='{self.matrix.filepath})"
    
    def execute(self, path):
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
        matrix_A = self.left.op.execute(None)
        matrix_B = self.right.op.execute(None)

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

    def execute(self, output_path):
        """Executes the multiplication plan."""
        print(" - Execute MultiplyNode: Get inputs...")
        matrix_A = self.left.op.execute(None)
        matrix_B = self.right.op.execute(None)

        print(" - Calling 'multiply' to perform the computation...")
        return multiply(matrix_A, matrix_B, output_path)

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
        if isinstance(self.left.op, AddNode):
            print("Optimizer: Fused Add-Multiply pattern detected! Calling fast kernel.")
            # ...then call the special fused execution function
            add_op = self.left.op
            # Get the actual matrix handles from the eager nodes
            matrix_A = add_op.left.op.execute(None)
            matrix_B = add_op.right.op.execute(None)
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
            TMP = self.left.compute(output_path + ".tmp")
            # 2. Then, perform the scalar multiplication
            C = PaperMatrix(output_path, self.shape, mode='w+')
            for r in range(0, self.shape[0], TILE_SIZE):
                r_end = min(r + TILE_SIZE, self.shape[0])
                for c in range(0, self.shape[1], TILE_SIZE):
                    c_end = min(c + TILE_SIZE, self.shape[1])
                    C.data[r:r_end, c:c_end] = TMP.data[r:r_end, c:c_end] * self.right
            
            C.data.flush()
            TMP.close()
            return C

