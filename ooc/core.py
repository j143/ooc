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
    

def main():
    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Define the shape of our first test matrix
    shape_A = (3500, 4200)
    shape_C = (4200, 4000)
    path_A = os.path.join(DATA_DIR, "A.bin")
    path_B = os.path.join(DATA_DIR, "B_add.bin")
    path_C = os.path.join(DATA_DIR, "C_mul.bin")

    # Create the matrix file if it doesn't already exist
    for path, shape in {path_A: shape_A, path_B: shape_A, path_C: shape_C}.items():
        if not os.path.exists(path):
            writer = MiniMatrix(path, shape, mode='w+')
            writer.close()
    
    # lazy execution logic
    print("\n--- Build and execute a lazy addition plan ---")

    A_handle = MiniMatrix(path_A, shape_A, mode='r')
    B_handle = MiniMatrix(path_B, shape_A, mode='r')
    C_handle = MiniMatrix(path_C, shape_C, mode='r')
    A_lazy = LazyMatrix(EagerMatrixOp(A_handle))
    B_lazy = LazyMatrix(EagerMatrixOp(B_handle))
    C_lazy = LazyMatrix(EagerMatrixOp(C_handle))

    # 1. Build the plan using the '+' operator
    #   This calls our __add__ method.
    plan = (A_lazy) * 2
    plan2 = A_lazy @ C_lazy
    print(f"\n plan built: '{plan!r}'")

    # 2. Execute the plan using .compute()
    result_file_path = os.path.join(DATA_DIR, "L_lazy_add.bin")
    start_time = time.time()
    result_matrix = plan.compute(result_file_path)
    end_time = time.time()

    print(f"\nExecution finished in {end_time - start_time:.2f} seconds.")

    A_handle.close()

    result_file_path = os.path.join(DATA_DIR, "M_lazy_add.bin")
    start_time = time.time()
    result_matrix2 = plan.compute(result_file_path)
    end_time = time.time()

    # 3. Clean up the file handle
    A_handle.close()
    B_handle.close()
    C_handle.close()
    result_matrix.close()
    result_matrix2.close()


if __name__ == "__main__":
    main()