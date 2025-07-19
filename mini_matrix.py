import numpy as np
import os
import time

# --- Configuration ---
TILE_SIZE = 1000  # The dimension of the square tiles to process in memory
DATA_DIR = "data" # Directory to store large matrix files

class MiniMatrix:
    """
    Represents a matrix stored on disk using a memory-mapped file.
    It holds metadata but doesn't load the whole file into RAM.
    """
    def __init__(self, filepath, shape, dtype=np.float32, mode='r'):
        self.filepath = filepath
        self.shape = shape
        self.dtype = np.dtype(dtype)
        
        # This is the core of our out-of-core access.
        # It links a NumPy array interface to the file on disk.
        # Create a memory-map to an array stored in a binary file on disk.
        # Memory-mapped files are used for accessing small segments of large files on disk,
        # without reading the entire file into memory. NumPy’s memmap’s are array-like objects.
        self.data = np.memmap(self.filepath, dtype=self.dtype, mode=mode, shape=self.shape)
        
    def close(self):
        """Explicitly closes the underlying memory-mapped file handle."""
        
        if self.data is not None and self.data._mmap is not None:
            print("I reached mmap")
            print(self.data._mmap)
            self.data._mmap.close()
            print(self.data._mmap)
    
    def __repr__(self):
        return f"MiniMatrix(path='{self.filepath}', shape={self.shape}, dtype={self.dtype.name})"

class EagerMatrixOp:
    """An operation that represents an already computed matrix on disk. This is the leaf of our plan."""
    def __init__(self, matrix: MiniMatrix):
        self.matrix = matrix
        self.shape = matrix.shape

    def __repr__(self):
        return f"EagerMatrixOp(path='{self.matrix.filepath})"
    
    def execute(self, path):
        """Executing a leaf node simply means returning the handle to the existing matrix."""
        print(f" - Executing EagerMatrixOp: Providing handle to '{self.matrix.filepath}")
        return self.matrix

class AddOp:
    """An operation node representing addition in our computation plan."""
    def __init__(self, left: 'LazyMatrix', right: 'LazyMatrix'):
        if left.shape != right.shape:
            raise ValueError("Shapes must match for lazy addition.")
        self.left = left
        self.right = right
        self.shape = left.shape
    
    def __repr__(self):
        # !r calls the repr() of the inner objects, creating a nested view
        return f"AddOp(left={self.left!r}, right={self.right!r})"

    def execute(self, output_path):
        """Executes the addition plan."""
        print(" - Execute AddOp: Get inputs...")
        matrix_A = self.left.op.execute(None)
        matrix_B = self.right.op.execute(None)

        print(" - Calling 'add_eager' to perform the computation...")
        return add_eager(matrix_A, matrix_B, output_path)

class LazyMatrix:
    """Represents a computation that will result in a matrix, but is not yet executed."""
    def __init__(self, op):
        # The 'op' is the plan for this matrix
        self.op = op
        self.shape = op.shape
    
    def __repr__(self):
        return f"LazyMatrix(plan={self.op!r})"

    def __add__(self, x: 'LazyMatrix'):
        print("Build an 'AddOp' plan...")
        return LazyMatrix(AddOp(self, x))

    def compute(self, output_path):
        """Triggers the execution of the entire computation plan."""
        result_matrix = self.op.execute(output_path)
        return result_matrix


def create_random_matrix(filepath, shape):
    """Creates and saves a large matrix with random data, tile by tile."""
    print(f"Creating random matrix at '{filepath}' with shape {shape}...")
    
    # Create a new file for writing
    matrix = MiniMatrix(filepath, shape, mode='w+')
    
    # Iterate through the matrix in blocks and fill with random data
    for r_start in range(0, shape[0], TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, shape[0])
        for c_start in range(0, shape[1], TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, shape[1])
            tile_shape = (r_end - r_start, c_end - c_start)
            # Generate a small in-memory tile
            random_tile = np.random.rand(*tile_shape).astype(matrix.dtype)
            # Write the tile to the memory-mapped file
            matrix.data[r_start:r_end, c_start:c_end] = random_tile
            
    matrix.data.flush() # Ensure all changes are written to disk
    print("Creation complete.")
    matrix.close()
    

def add_eager(A: MiniMatrix, B: MiniMatrix, output_path: str):
    """Performs out-of-core matrix addition: C = A + B."""
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape for addition.")

    C = MiniMatrix(output_path, A.shape, mode='w+')
    
    print("Performing eager addition...")
    rows, cols = A.shape
    # Iterate through the matrices tile by tile
    for r_start in range(0, rows, TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, rows)
        for c_start in range(0, cols, TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, cols)
            
            # 1. Read tiles from A and B into memory
            tile_A = A.data[r_start:r_end, c_start:c_end]
            tile_B = B.data[r_start:r_end, c_start:c_end]
            
            # 2. Compute the result in memory
            tile_C = tile_A + tile_B
            
            # 3. Write the resulting tile to C's file
            C.data[r_start:r_end, c_start:c_end] = tile_C
    
    C.data.flush() # Ensure all data is saved to disk
    print("Addition complete.")
    return C

def multiply_eager(A: MiniMatrix, B: MiniMatrix):
    """Performs out-of-core tiled matrix multiplication: C = A @ B."""
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must match for multiplication.")
    
    C_shape = (A.shape[0], B.shape[1])
    C_path = os.path.join(DATA_DIR, "C_mul.bin")
    C = MiniMatrix(C_path, C_shape, mode='w+')

    print("Performing eager multiplication...")
    rows_A, K, cols_B = A.shape[0], A.shape[1], B.shape[1]

    # Loop over the tiles of the output matrix C
    for r_start in range(0, rows_A, TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, rows_A)
        for c_start in range(0, cols_B, TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, cols_B)

            # 1. Initialize an in-memory tile for accumulating the result
            tile_C = np.zeros((r_end - r_start, c_end - c_start), dtype=C.dtype)

            # 2. Loop through the inner dimension k
            for k_start in range(0, K, TILE_SIZE):
                k_end = min(k_start + TILE_SIZE, K)

                # load the corresponding tiles from A and B
                tile_A = A.data[r_start:r_end, k_start:k_end]
                tile_B = B.data[k_start:k_end, c_start:c_end]

                #3. Perform in-memory multiplication and accumulate
                tile_C += tile_A @ tile_B
            
            # 4. write the final computed tile to disk
            C.data[r_start:r_end, c_start:c_end] = tile_C
        
        C.data.flush()
        print("Multiplication complete.")
        return C


def main():
    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Define the shape of our first test matrix
    shape_A = (3500, 4200)
    path_A = os.path.join(DATA_DIR, "A.bin")
    path_B = os.path.join(DATA_DIR, "B_add.bin")

    # Create the matrix file if it doesn't already exist
    for path, shape in {path_A: shape_A, path_B: shape_A}.items():
        if not os.path.exists(path_A):
            writer = MiniMatrix(path_A, shape_A, mode='w+')
            writer.close()
    
    # lazy execution logic
    print("\n--- Build and execute a lazy addition plan ---")

    A_handle = MiniMatrix(path_A, shape_A, mode='r')
    B_handle = MiniMatrix(path_B, shape_A, mode='r')
    A_lazy = LazyMatrix(EagerMatrixOp(A_handle))
    B_lazy = LazyMatrix(EagerMatrixOp(B_handle))

    # 1. Build the plan using the '+' operator
    #   This calls our __add__ method.
    plan = A_lazy + B_lazy
    print(f"\n plan built: '{plan!r}'")

    # 2. Execute the plan using .compute()
    result_file_path = os.path.join(DATA_DIR, "C_lazy_add.bin")
    start_time = time.time()
    result_matrix = plan.compute(result_file_path)
    end_time = time.time()

    print(f"\nExecution finished in {end_time - start_time:.2f} seconds.")

    # 3. Clean up the file handle
    A_handle.close()
    B_handle.close()
    result_matrix.close()

    
    # # Instantiate our matrix object
    # A = MiniMatrix(path_A, shape_A)
    # print(f"Successfully loaded matrix representation: {A}")

    # # --- Setup Matrix B for Addition ---
    # # It must have the same shape as A
    # shape_B_add = (3500, 4200)
    # path_B_add = os.path.join(DATA_DIR, "B_add.bin")
    # if not os.path.exists(path_B_add):
    #     create_random_matrix(path_B_add, shape_B_add)
    # B_add = MiniMatrix(path_B_add, shape_B_add)
    # print(f"Loaded matrix representation: {B_add}")
    
    # # --- Perform Eager Addition ---
    # print("\n--- Starting Eager Addition ---")
    # start_time = time.time()
    # C_add = add_eager(A, B_add)
    # A.close()
    # B_add.close()
    # C_add.close()
    # end_time = time.time()
    # print(f"Eager Addition finished in {end_time - start_time:.2f} seconds. Result: {C_add}")

    # # --- Setup Matrix B for Multiplcation ---
    # shape_B_mul = (4200, 3800)
    # path_B_mul = os.path.join(DATA_DIR, "B_mul.bin")
    # if not os.path.exists(path_B_mul):
    #     create_random_matrix(path_B_mul, shape_B_mul)
    # B_mul = MiniMatrix(path_B_mul, shape_B_mul)
    # print(f"Loaded matrix representation: {B_mul}")

    # # --- Perform Eager Addition ---
    # print("\n--- Starting Eager Multiplication ---")
    # start_time = time.time()
    # C_mul = multiply_eager(A, B_mul)
    # A.close()
    # B_mul.close()
    # C_mul.close()
    # end_time = time.time()
    # print(f"Eager Multiplication finished in {end_time - start_time:.2f} seconds. Result: {C_mul}")


if __name__ == "__main__":
    main()