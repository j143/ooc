# benchmarks/utils.py
import numpy as np
import os
import time
import sys

# Add the project root to the Python path to allow importing 'paper'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paper.core import PaperMatrix

# --- Configuration ---
# Use a larger tile size for benchmarks if memory allows
TILE_SIZE = 1024

def create_benchmark_data(filepath, shape):
    """
    Creates a large matrix file with random data for benchmarking.
    This function is designed to be run once to set up the data.
    """
    print(f"Creating benchmark data at '{filepath}' with shape {shape}...")
    matrix = PaperMatrix(filepath, shape, mode='w+')
    
    # Fill with random data tile by tile
    for r_start in range(0, shape[0], TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, shape[0])
        for c_start in range(0, shape[1], TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, shape[1])
            tile_shape = (r_end - r_start, c_end - c_start)
            random_tile = np.random.rand(*tile_shape).astype(matrix.dtype)
            matrix.data[r_start:r_end, c_start:c_end] = random_tile
            
    matrix.data.flush()
    matrix.close()
    print(f"Creation of '{filepath}' complete.")

class Timer:
    """A simple context manager for timing code blocks."""
    def __init__(self, description):
        self.description = description
        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        print(f"\n--- Starting: {self.description} ---")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        print(f"--- Finished: {self.description} in {self.elapsed:.4f} seconds ---")

