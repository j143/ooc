import numpy as np
import os
import time

# --- Configuration ---
TILE_SIZE = 1000  # The dimension of the square tiles to process in memory
DATA_DIR = "data" # Directory to store large matrix files

class MiniMatrix:
    """
    Represents a matrix stored on disk, accessed via memory-mapped files.
    This provides a way to handle the matrix as if it were an in-memory NumPy array
    without loading the entire file.
    """
    def __init__(self, filepath, shape, dtype=np.float32, mode='r'):
        self.filepath = filepath
        self.shape = shape
        self.dtype = np.dtype(dtype)
        
        # Ensure the file exists and is of the correct size if in read/read-write mode
        if mode != 'w' and os.path.exists(self.filepath):
            expected_size = self.shape[0] * self.shape[1] * self.dtype.itemsize
            if os.path.getsize(self.filepath) != expected_size:
                raise ValueError(
                    f"File size of {os.path.getsize(self.filepath)} does not match "
                    f"expected size of {expected_size} for shape {self.shape}"
                )
        
        # Use numpy's memmap for efficient, out-of-core array access
        self.data = np.memmap(self.filepath, dtype=self.dtype, mode=mode, shape=self.shape)

    def __repr__(self):
        return f"MiniMatrix(path='{self.filepath}', shape={self.shape}, dtype={self.dtype.name})"

# --- Utility Function ---

def create_random_matrix(filepath, shape):
    """Creates and saves a large matrix with random data."""
    print(f"Creating random matrix at '{filepath}' with shape {shape}...")
    # Create an empty file with the correct size
    with open(filepath, 'wb') as f:
        f.seek(shape[0] * shape[1] * np.dtype(np.float32).itemsize - 1)
        f.write(b'\0')

    # Memory-map the file and fill it with random data tile by tile
    matrix = MiniMatrix(filepath, shape, mode='r+')
    for r_start in range(0, shape[0], TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, shape[0])
        for c_start in range(0, shape[1], TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, shape[1])
            tile_shape = (r_end - r_start, c_end - c_start)
            matrix.data[r_start:r_end, c_start:c_end] = np.random.rand(*tile_shape).astype(matrix.dtype)
    matrix.data.flush() # Ensure data is written to disk
    print("Creation complete.")
    return matrix

# --- Eager Execution Engine ---

def add_eager(A: MiniMatrix, B: MiniMatrix):
    """Performs out-of-core matrix addition: C = A + B."""
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape for addition.")

    C_path = os.path.join(DATA_DIR, "C_add.bin")
    C = MiniMatrix(C_path, A.shape, mode='w+')
    
    rows, cols = A.shape
    for r_start in range(0, rows, TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, rows)
        for c_start in range(0, cols, TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, cols)
            
            # Read tiles, compute, and write result
            tile_A = A.data[r_start:r_end, c_start:c_end]
            tile_B = B.data[r_start:r_end, c_start:c_end]
            tile_C = tile_A + tile_B
            C.data[r_start:r_end, c_start:c_end] = tile_C
    
    C.data.flush()
    return C

def multiply_eager(A: MiniMatrix, B: MiniMatrix):
    """Performs out-of-core tiled matrix multiplication: C = A @ B."""
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must match for multiplication.")

    C_shape = (A.shape[0], B.shape[1])
    C_path = os.path.join(DATA_DIR, "C_mul.bin")
    C = MiniMatrix(C_path, C_shape, mode='w+')
    
    rows_A, K, cols_B = A.shape[0], A.shape[1], B.shape[1]
    
    for r_start in range(0, rows_A, TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, rows_A)
        for c_start in range(0, cols_B, TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, cols_B)
            
            # This tile will accumulate the results for C[r_start:r_end, c_start:c_end]
            tile_C = np.zeros((r_end - r_start, c_end - c_start), dtype=C.dtype)
            
            for k_start in range(0, K, TILE_SIZE):
                k_end = min(k_start + TILE_SIZE, K)
                
                tile_A = A.data[r_start:r_end, k_start:k_end]
                tile_B = B.data[k_start:k_end, c_start:c_end]
                tile_C += tile_A @ tile_B
            
            C.data[r_start:r_end, c_start:c_end] = tile_C

    C.data.flush()
    return C

# --- Lazy Execution Engine ---

class LazyMatrix:
    """Represents a computation that results in a matrix, but is not yet executed."""
    def __init__(self, op):
        self.op = op
        self.shape = op.shape

    def compute(self, output_path):
        """Triggers the computation and saves the result to a file."""
        print(f"Executing lazy plan to create '{output_path}'...")
        return self.op.execute(output_path)

    def __add__(self, other):
        return LazyMatrix(AddOp(self, other))

    def __matmul__(self, other):
        return LazyMatrix(MultiplyOp(self, other))
        
    def __repr__(self):
        return f"LazyMatrix(shape={self.shape}, op={self.op})"

class AddOp:
    def __init__(self, left, right):
        if left.shape != right.shape:
            raise ValueError("Shapes must match for lazy addition.")
        self.left = left
        self.right = right
        self.shape = left.shape

    def execute(self, output_path):
        # Resolve inputs: if an input is lazy, compute it first
        # For this simple engine, we assume inputs are already computed MiniMatrix objects
        # A more robust engine would handle recursive computation.
        A = self.left if isinstance(self.left, MiniMatrix) else self.left.compute(output_path + ".left")
        B = self.right if isinstance(self.right, MiniMatrix) else self.right.compute(output_path + ".right")
        return add_eager(A, B) # Reuse the eager implementation for execution

class MultiplyOp:
    def __init__(self, left, right):
        if left.shape[1] != right.shape[0]:
            raise ValueError("Inner dimensions must match for lazy multiplication.")
        self.left = left
        self.right = right
        self.shape = (left.shape[0], right.shape[1])

    def execute(self, output_path):
        A = self.left if isinstance(self.left, MiniMatrix) else self.left.compute(output_path + ".left")
        B = self.right if isinstance(self.right, MiniMatrix) else self.right.compute(output_path + ".right")
        return multiply_eager(A, B) # Reuse the eager implementation for execution


# --- Main Execution Block ---

def main():
    """Main function to demonstrate the out-of-core library."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Define matrix shapes
    shape_A = (3500, 4200)
    shape_B = (4200, 3800)
    
    # --- Create Test Data ---
    path_A = os.path.join(DATA_DIR, "A.bin")
    path_B = os.path.join(DATA_DIR, "B.bin")
    if not os.path.exists(path_A):
        create_random_matrix(path_A, shape_A)
    if not os.path.exists(path_B):
        create_random_matrix(path_B, shape_B)
        
    A = MiniMatrix(path_A, shape_A)
    B = MiniMatrix(path_B, shape_B)
    
    # Create a compatible matrix for addition
    shape_A_add = (3500, 4200)
    path_A_add = os.path.join(DATA_DIR, "A_add.bin")
    if not os.path.exists(path_A_add):
        create_random_matrix(path_A_add, shape_A_add)
    A_add = MiniMatrix(path_A_add, shape_A_add)

    print("\n--- Starting Eager Execution ---")
    
    # Eager Addition
    start_time = time.time()
    C_add = add_eager(A_add, A)
    end_time = time.time()
    print(f"Eager Addition finished in {end_time - start_time:.2f} seconds. Result: {C_add}")
    
    # Eager Multiplication
    start_time = time.time()
    C_mul = multiply_eager(A, B)
    end_time = time.time()
    print(f"Eager Multiplication finished in {end_time - start_time:.2f} seconds. Result: {C_mul}")

    print("\n--- Starting Lazy Execution ---")
    
    # Lazy Addition
    lazy_A_add = LazyMatrix(A_add)
    lazy_A = LazyMatrix(A)
    lazy_plan_add = lazy_A_add + lazy_A
    print(f"Lazy Addition plan created: {lazy_plan_add}")
    start_time = time.time()
    D_add = lazy_plan_add.compute(os.path.join(DATA_DIR, "D_add.bin"))
    end_time = time.time()
    print(f"Lazy Addition executed in {end_time - start_time:.2f} seconds. Result: {D_add}")
    
    # Lazy Multiplication
    lazy_A = LazyMatrix(A)
    lazy_B = LazyMatrix(B)
    lazy_plan_mul = lazy_A @ lazy_B
    print(f"Lazy Multiplication plan created: {lazy_plan_mul}")
    start_time = time.time()
    D_mul = lazy_plan_mul.compute(os.path.join(DATA_DIR, "D_mul.bin"))
    end_time = time.time()
    print(f"Lazy Multiplication executed in {end_time - start_time:.2f} seconds. Result: {D_mul}")
    
    print("\n--- Verification ---")
    
    # Verify a tile of the multiplication result
    # We load small tiles from the original files to compute an in-memory ground truth
    ground_truth_tile_A = A.data[0:TILE_SIZE, 0:TILE_SIZE]
    ground_truth_tile_B = B.data[0:TILE_SIZE, 0:TILE_SIZE]
    expected_tile_C = ground_truth_tile_A @ ground_truth_tile_B
    
    # To check this, we need to compute C_00 fully
    full_C00_tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    for k_start in range(0, A.shape[1], TILE_SIZE):
        k_end = min(k_start + TILE_SIZE, A.shape[1])
        tile_A = A.data[0:TILE_SIZE, k_start:k_end]
        tile_B = B.data[k_start:k_end, 0:TILE_SIZE]
        full_C00_tile += tile_A @ tile_B
        
    computed_tile_C = C_mul.data[0:TILE_SIZE, 0:TILE_SIZE]
    
    if np.allclose(full_C00_tile, computed_tile_C):
        print("✅ Multiplication verification successful!")
    else:
        print("❌ Multiplication verification FAILED!")
        
if __name__ == "__main__":
    main()