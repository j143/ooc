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
    return matrix

# --- Main Execution Block ---
if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Define the shape of our first test matrix
    shape_A = (3500, 4200)
    path_A = os.path.join(DATA_DIR, "A.bin")

    # Create the matrix file if it doesn't already exist
    if not os.path.exists(path_A):
        create_random_matrix(path_A, shape_A)
    
    # Instantiate our matrix object
    A = MiniMatrix(path_A, shape_A)
    print(f"Successfully loaded matrix representation: {A}")