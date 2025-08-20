"""
Tests for Paper matrix operations with out-of-core processing.
This test suite covers basic operations:
- Matrix addition (A + B)
- Fused matrix addition and scalar multiplication ((A + B) * 2)
- Matrix scalar multiplication (A * 2)
"""

import os
import time
import numpy as np
import sys
import shutil

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper import PaperMatrix, Plan, EagerNode
from paper.config import TILE_SIZE

# --- Configuration ---
TEST_DATA_DIR = "test_data"  # Directory to store test matrix files

def setup():
    """Create test data directory and test matrices."""
    # Create the test data directory if it doesn't exist, or clean it if it does
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    os.makedirs(TEST_DATA_DIR)
    
    # Define smaller matrix shapes for faster testing
    shape_A = (2000, 1500)  # Smaller than the production matrices for quicker tests
    
    # Create paths for test matrices
    path_A = os.path.join(TEST_DATA_DIR, "A.bin")
    path_B = os.path.join(TEST_DATA_DIR, "B.bin")
    
    # Create test matrices with some values
    create_test_matrix(path_A, shape_A, fill_value=1.0)  # Matrix A filled with 1.0
    create_test_matrix(path_B, shape_A, fill_value=2.0)  # Matrix B filled with 2.0
    
    return shape_A, path_A, path_B

def create_test_matrix(filepath, shape, fill_value=None):
    """Creates and saves a test matrix with specified values or random data."""
    print(f"Creating test matrix at '{filepath}' with shape {shape}...")
    
    # Create a new file for writing
    matrix = PaperMatrix(filepath, shape, mode='w+')
    
    # Iterate through the matrix in blocks and fill with data
    for r_start in range(0, shape[0], TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, shape[0])
        for c_start in range(0, shape[1], TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, shape[1])
            tile_shape = (r_end - r_start, c_end - c_start)
            
            if fill_value is not None:
                # Generate a tile filled with the specified value
                tile_data = np.full(tile_shape, fill_value, dtype=matrix.dtype)
            else:
                # Generate a random tile
                tile_data = np.random.rand(*tile_shape).astype(matrix.dtype)
                
            # Write the tile to the memory-mapped file
            matrix.data[r_start:r_end, c_start:c_end] = tile_data
    
    matrix.data.flush()  # Ensure all changes are written to disk
    print(f"Matrix creation complete: {filepath}")
    matrix.close()

def verify_result(result_path, shape, expected_value=None):
    """Verify a small sample of the result matrix to check correctness."""
    result = PaperMatrix(result_path, shape, mode='r')
    
    # Sample a small region from the center of the matrix
    r_mid = shape[0] // 2
    c_mid = shape[1] // 2
    sample = result.data[r_mid:r_mid+10, c_mid:c_mid+10]
    
    if expected_value is not None:
        # Check if the sample values are close to the expected value
        is_correct = np.allclose(sample, expected_value, rtol=1e-5)
        if is_correct:
            print(f"✓ Verification passed: Values close to {expected_value}")
        else:
            print(f"✗ Verification failed: Expected {expected_value}, got average {np.mean(sample)}")
    else:
        # Just show the sample values
        print(f"Sample values (center of matrix):\n{sample}")
    
    result.close()
    return sample

def test_matrix_addition():
    """Test A + B operation."""
    print("\n=== Testing Matrix Addition (A + B) ===")
    
    # Get matrix paths and handles
    shape_A, path_A, path_B = setup()
    A_handle = PaperMatrix(path_A, shape_A, mode='r')
    B_handle = PaperMatrix(path_B, shape_A, mode='r')
    
    # Create lazy representations
    A_lazy = Plan(EagerNode(A_handle))
    B_lazy = Plan(EagerNode(B_handle))
    
    # Build the addition plan
    addition_plan = A_lazy + B_lazy
    print(f"Plan built: {addition_plan!r}")
    
    # Execute the plan
    result_path = os.path.join(TEST_DATA_DIR, "result_addition.bin")
    start_time = time.time()
    result_matrix, _ = addition_plan.compute(result_path)
    end_time = time.time()
    
    print(f"Addition computation time: {end_time - start_time:.2f} seconds")
    
    # Verify the result (1.0 + 2.0 = 3.0)
    sample = verify_result(result_path, shape_A, expected_value=3.0)
    
    # Clean up
    A_handle.close()
    B_handle.close()
    result_matrix.close()
    
    return sample

def test_fused_add_multiply():
    """Test (A + B) * 2 operation with fusion optimization."""
    print("\n=== Testing Fused Add-Multiply ((A + B) * 2) ===")
    
    # Get matrix paths and handles
    shape_A, path_A, path_B = setup()
    A_handle = PaperMatrix(path_A, shape_A, mode='r')
    B_handle = PaperMatrix(path_B, shape_A, mode='r')
    
    # Create lazy representations
    A_lazy = Plan(EagerNode(A_handle))
    B_lazy = Plan(EagerNode(B_handle))
    
    # Build the fused operation plan
    fused_plan = (A_lazy + B_lazy) * 2
    print(f"Plan built: {fused_plan!r}")
    
    # Execute the plan
    result_path = os.path.join(TEST_DATA_DIR, "result_fused.bin")
    start_time = time.time()
    result_matrix, _ = fused_plan.compute(result_path)
    end_time = time.time()
    
    print(f"Fused operation computation time: {end_time - start_time:.2f} seconds")
    
    # Verify the result ((1.0 + 2.0) * 2 = 6.0)
    sample = verify_result(result_path, shape_A, expected_value=6.0)
    
    # Clean up
    A_handle.close()
    B_handle.close()
    result_matrix.close()
    
    return sample

def test_scalar_multiply():
    """Test A * 2 operation."""
    print("\n=== Testing Scalar Multiplication (A * 2) ===")
    
    # Get matrix paths and handles
    shape_A, path_A, _ = setup()
    A_handle = PaperMatrix(path_A, shape_A, mode='r')
    
    # Create lazy representation
    A_lazy = Plan(EagerNode(A_handle))
    
    # Build the scalar multiplication plan
    scalar_plan = A_lazy * 2
    print(f"Plan built: {scalar_plan!r}")
    
    # Execute the plan
    result_path = os.path.join(TEST_DATA_DIR, "result_scalar_mul.bin")
    start_time = time.time()
    result_matrix, _ = scalar_plan.compute(result_path)
    end_time = time.time()
    
    print(f"Scalar multiplication computation time: {end_time - start_time:.2f} seconds")
    
    # Verify the result (1.0 * 2 = 2.0)
    sample = verify_result(result_path, shape_A, expected_value=2.0)
    
    # Clean up
    A_handle.close()
    result_matrix.close()
    
    return sample

def run_all_tests():
    """Run all tests and report results."""
    print("Running all Paper matrix operation tests...")
    
    # Run the tests
    addition_sample = test_matrix_addition()
    fused_sample = test_fused_add_multiply()
    scalar_sample = test_scalar_multiply()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Matrix Addition (A + B): {'✓ Passed' if np.allclose(addition_sample, 3.0) else '✗ Failed'}")
    print(f"Fused Add-Multiply ((A + B) * 2): {'✓ Passed' if np.allclose(fused_sample, 6.0) else '✗ Failed'}")
    print(f"Scalar Multiplication (A * 2): {'✓ Passed' if np.allclose(scalar_sample, 2.0) else '✗ Failed'}")
    
    # Clean up test data directory
    print("\nTests completed. Cleaning up test data...")
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    print("Cleanup finished.")

if __name__ == "__main__":
    run_all_tests()
