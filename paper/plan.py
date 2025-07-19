
from . import optimizer

class EagerNodeOp:
    """An operation that represents an already computed matrix on disk. This is the leaf of our plan."""
    def __init__(self, matrix: PaperMatrix):
        self.matrix = matrix
        self.shape = matrix.shape

    def __repr__(self):
        return f"EagerNodeOp(path='{self.matrix.filepath})"
    
    def execute(self, path):
        """Executing a leaf node simply means returning the handle to the existing matrix."""
        print(f" - Executing EagerNodeOp: Providing handle to '{self.matrix.filepath}")
        return self.matrix

class AddNode:
    """An operation node representing addition in our computation plan."""
    def __init__(self, left: 'LazyMatrix', right: 'LazyMatrix'):
        if left.shape != right.shape:
            raise ValueError("Shapes must match for lazy addition.")
        self.left = left
        self.right = right
        self.shape = left.shape
    
    def __repr__(self):
        # !r calls the repr() of the inner objects, creating a nested view
        return f"AddNode(left={self.left!r}, right={self.right!r})"

    def execute(self, output_path):
        """Executes the addition plan."""
        print(" - Execute AddNode: Get inputs...")
        matrix_A = self.left.op.execute(None)
        matrix_B = self.right.op.execute(None)

        print(" - Calling 'add' to perform the computation...")
        return add(matrix_A, matrix_B, output_path)

class MultiplyNode:
    """An operation node representing addition in our computation plan."""
    def __init__(self, left: 'LazyMatrix', right: 'LazyMatrix'):
        if left.shape[1] != right.shape[0]:
            raise ValueError("Shapes must match for lazy addition.")
        self.left = left
        self.right = right
        self.shape = (left.shape[0], right.shape[1])
    
    def __repr__(self):
        # !r calls the repr() of the inner objects, creating a nested view
        return f"AddNode(left={self.left!r}, right={self.right!r})"

    def execute(self, output_path):
        """Executes the addition plan."""
        print(" - Execute AddNode: Get inputs...")
        matrix_A = self.left.op.execute(None)
        matrix_B = self.right.op.execute(None)

        print(" - Calling 'add' to perform the computation...")
        return add(matrix_A, matrix_B, output_path)

class MultiplyScalarNode:
    """An operation node representing multiplication by a scalar."""
    def __init__(self, left: 'LazyMatrix', right: float):
        self.left = left
        self.right = right
        self.shape = left.shape
    
    def __repr__(self):
        return f"MultiplyScalarNode(left={self.left!r}, scalar={self.right})"

    def execute(self, output_path):
        """This is our 'mini-optimizer'. It checks if it can use a fast, fused kernel."""
        # THE OPTIMIZATION RULE:
        # If my left input is an addition operation...
        if isinstance(self.left.op, AddNode):
            print("Optimizer: Fused Add-Multiply pattern detected! Calling fast kernel.")
            # ...then call the special fused execution function
            add_op = self.left.op
            return execute_fused_add_multiply(
                add_op.left.op.matrix, # Matrix A
                add_op.right.op.matrix, # Matrix B
                self.right, # The scalar value
                output_path
            )
        else:
            # Fallback to general case (non-fused)
            print("Optimizer: No fusion pattern detected. Executing step-by-step.")
            # 1. Compute the input matrix first
            TMP = self.left.compute(output_path + ".tmp")
            # 2. Then, peform the scalar multiplication
            C = PaperMatrix(output_path, self.shape, mode='w+')
            for r in range(0, self.shape[0], TILE_SIZE):
                for c in range(0, self.shape[1], TILE_SIZE):
                    C.data[r:r+TILE_SIZE, c:c+TILE_SIZE] = TMP.data[r:r+TILE_SIZE, c:c+TILE_SIZE] * self.right
            
            C.data.flush()
            TMP.close()
            return C

class LazyMatrix:
    """Represents a computation that will result in a matrix, but is not yet executed."""
    def __init__(self, op):
        # The 'op' is the plan for this matrix
        self.op = op
        self.shape = op.shape
    
    def __repr__(self):
        return f"LazyMatrix(plan={self.op!r})"

    def __add__(self, x: 'LazyMatrix'):
        print("Build an 'AddNode' plan...")
        return LazyMatrix(AddNode(self, x))
    
    def __matmul__(self, x: 'LazyMatrix'):
        print("Build an 'MultiplyNode' plan")
        return LazyMatrix(MultiplyNode(self, x))

    def __mul__(self, x):
        if isinstance(x, (int, float)):
            return LazyMatrix(MultiplyScalarNode(self, x))
        raise NotImplementedError("Only scalar multiplication is supported.")
    
    def __rmul__(self, x):
        # Handles the case `2 * my_lazy_matrix`
        return self.__mul__(x)

    def compute(self, output_path):
        """Triggers the execution of the entire computation plan."""
        result_matrix = self.op.execute(output_path)
        return result_matrix
