"""
Tests for advanced optimization patterns in the Paper matrix library.
This test suite focuses on the optimizer's ability to detect and apply fusion patterns.
"""

import os
import time
import numpy as np
import sys
import shutil

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper import PaperMatrix, Plan, EagerNode

# --- Configuration ---
TILE_SIZE = 1000  # The dimension of the square tiles to process in memory
TEST_DATA_DIR = "test_data_optimizer"  # Directory to store test matrix files

def setup():
    """Create test data directory and test matrices."""
    # Create the test data directory if it doesn't exist, or clean it if it does
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    os.makedirs(TEST_DATA_DIR)
    
    # Define smaller matrix shapes for faster testing
    shape_A = (1500, 1000)  # A is 1500x1000
    shape_B = (1500, 1000)  # B is 1500x1000 (same as A for addition)
    shape_C = (1000, 800)   # C is 1000x800 (for matrix multiplication)
    
    # Create paths for test matrices
    path_A = os.path.join(TEST_DATA_DIR, "A.bin")
    path_B = os.path.join(TEST_DATA_DIR, "B.bin")
    path_C = os.path.join(TEST_DATA_DIR, "C.bin")
    
    # Create test matrices with specific values
    create_test_matrix(path_A, shape_A, fill_value=1.0)  # Matrix A filled with 1.0
    create_test_matrix(path_B, shape_B, fill_value=2.0)  # Matrix B filled with 2.0
    create_test_matrix(path_C, shape_C, fill_value=0.5)  # Matrix C filled with 0.5
    
    return {
        'A': {'shape': shape_A, 'path': path_A},
        'B': {'shape': shape_B, 'path': path_B},
        'C': {'shape': shape_C, 'path': path_C}
    }

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

def verify_result(result_path, shape, expected_value=None, label="Result"):
    """Verify a small sample of the result matrix to check correctness."""
    result = PaperMatrix(result_path, shape, mode='r')
    
    # Sample a small region from the center of the matrix
    r_mid = shape[0] // 2
    c_mid = shape[1] // 2
    sample = result.data[r_mid:r_mid+5, c_mid:c_mid+5]
    
    if expected_value is not None:
        # Check if the sample values are close to the expected value
        is_correct = np.allclose(sample, expected_value, rtol=1e-5, atol=1e-5)
        if is_correct:
            print(f"✓ {label} verification passed: Values close to {expected_value}")
        else:
            print(f"✗ {label} verification failed: Expected ~{expected_value}, got average {np.mean(sample)}")
            print(f"  Sample values:\n{sample}")
    else:
        # Just show the sample values
        print(f"{label} sample values (center of matrix):\n{sample}")
    
    result.close()
    return sample

def test_fused_add_multiply():
    """Test (A + B) * 2 operation with fusion optimization."""
    print("\n=== Testing Fused Add-Multiply ((A + B) * 2) ===")
    
    # Get matrix info
    matrices = setup()
    A_handle = PaperMatrix(matrices['A']['path'], matrices['A']['shape'], mode='r')
    B_handle = PaperMatrix(matrices['B']['path'], matrices['B']['shape'], mode='r')
    
    # Create lazy representations
    A_lazy = Plan(EagerNode(A_handle))
    B_lazy = Plan(EagerNode(B_handle))
    
    # Build the fused operation plan
    fused_plan = (A_lazy + B_lazy) * 2
    print(f"Plan built: {fused_plan!r}")
    
    # Execute the plan
    result_path = os.path.join(TEST_DATA_DIR, "result_fused_add_mul.bin")
    start_time = time.time()
    result_matrix = fused_plan.compute(result_path)
    end_time = time.time()
    
    print(f"Fused operation computation time: {end_time - start_time:.2f} seconds")
    
    # Verify the result ((1.0 + 2.0) * 2 = 6.0)
    sample = verify_result(result_path, matrices['A']['shape'], expected_value=6.0, 
                          label="Fused (A+B)*2")
    
    # Clean up
    A_handle.close()
    B_handle.close()
    result_matrix.close()
    
    return sample

def test_fused_matmul_scalar():
    """Test (A @ C) * 2 operation with fusion optimization."""
    print("\n=== Testing Fused Matrix-Multiply-Scalar ((A @ C) * 2) ===")
    
    # Get matrix info
    matrices = setup()
    A_handle = PaperMatrix(matrices['A']['path'], matrices['A']['shape'], mode='r')
    C_handle = PaperMatrix(matrices['C']['path'], matrices['C']['shape'], mode='r')
    
    # Create lazy representations
    A_lazy = Plan(EagerNode(A_handle))
    C_lazy = Plan(EagerNode(C_handle))
    
    # Build the fused operation plan
    fused_plan = (A_lazy @ C_lazy) * 2
    print(f"Plan built: {fused_plan!r}")
    
    # Expected output shape
    output_shape = (matrices['A']['shape'][0], matrices['C']['shape'][1])
    
    # Execute the plan
    result_path = os.path.join(TEST_DATA_DIR, "result_fused_matmul_scalar.bin")
    start_time = time.time()
    result_matrix = fused_plan.compute(result_path)
    end_time = time.time()
    
    print(f"Fused operation computation time: {end_time - start_time:.2f} seconds")
    
    # Verify the result (A[1.0] @ C[0.5] * 2 = 1.0)
    # Each row of A (with 1.0) multiplied by each column of C (with 0.5) = 0.5 * number of columns
    # For our test matrices with 1000 columns in A and all values = 1.0, and all values in C = 0.5
    # Each element in the result will be 1.0 * 0.5 * 1000 = 500, then multiplied by 2 = 1000
    expected_value = 1000.0  # (1.0 * 0.5 * 1000 * 2)
    sample = verify_result(result_path, output_shape, expected_value=expected_value, 
                          label="Fused (A@C)*2")
    
    # Clean up
    A_handle.close()
    C_handle.close()
    result_matrix.close()
    
    return sample

def test_fused_add_matmul():
    """Test (A + B) @ C operation with fusion optimization."""
    print("\n=== Testing Fused Add-Matrix-Multiply ((A + B) @ C) ===")
    
    # Get matrix info
    matrices = setup()
    A_handle = PaperMatrix(matrices['A']['path'], matrices['A']['shape'], mode='r')
    B_handle = PaperMatrix(matrices['B']['path'], matrices['B']['shape'], mode='r')
    C_handle = PaperMatrix(matrices['C']['path'], matrices['C']['shape'], mode='r')
    
    # Create lazy representations
    A_lazy = Plan(EagerNode(A_handle))
    B_lazy = Plan(EagerNode(B_handle))
    C_lazy = Plan(EagerNode(C_handle))
    
    # Build the fused operation plan
    fused_plan = (A_lazy + B_lazy) @ C_lazy
    print(f"Plan built: {fused_plan!r}")
    
    # Expected output shape
    output_shape = (matrices['A']['shape'][0], matrices['C']['shape'][1])
    
    # Execute the plan
    result_path = os.path.join(TEST_DATA_DIR, "result_fused_add_matmul.bin")
    start_time = time.time()
    result_matrix = fused_plan.compute(result_path)
    end_time = time.time()
    
    print(f"Fused operation computation time: {end_time - start_time:.2f} seconds")
    
    # Verify the result ((A[1.0] + B[2.0]) @ C[0.5])
    # (1.0 + 2.0) = 3.0, then 3.0 * 0.5 * 1000 = 1500 for each element
    expected_value = 1500.0  # (3.0 * 0.5 * 1000)
    sample = verify_result(result_path, output_shape, expected_value=expected_value, 
                          label="Fused (A+B)@C")
    
    # Clean up
    A_handle.close()
    B_handle.close()
    C_handle.close()
    result_matrix.close()
    
    return sample

def test_fused_double_scalar():
    """Test (A * 2) * 3 operation with fusion optimization."""
    print("\n=== Testing Fused Double-Scalar ((A * 2) * 3) ===")
    
    # Get matrix info
    matrices = setup()
    A_handle = PaperMatrix(matrices['A']['path'], matrices['A']['shape'], mode='r')
    
    # Create lazy representation
    A_lazy = Plan(EagerNode(A_handle))
    
    # Build the fused operation plan
    fused_plan = (A_lazy * 2) * 3
    print(f"Plan built: {fused_plan!r}")
    
    # Execute the plan
    result_path = os.path.join(TEST_DATA_DIR, "result_fused_double_scalar.bin")
    start_time = time.time()
    result_matrix = fused_plan.compute(result_path)
    end_time = time.time()
    
    print(f"Fused operation computation time: {end_time - start_time:.2f} seconds")
    
    # Verify the result (A[1.0] * 2 * 3 = 6.0)
    expected_value = 6.0
    sample = verify_result(result_path, matrices['A']['shape'], expected_value=expected_value, 
                          label="Fused (A*2)*3")
    
    # Clean up
    A_handle.close()
    result_matrix.close()
    
    return sample

def run_all_optimizer_tests():
    """Run all optimizer pattern tests and report results."""
    print("\nRunning all optimizer pattern tests...")
    
    # Run the tests
    add_mul_sample = test_fused_add_multiply()
    matmul_scalar_sample = test_fused_matmul_scalar()
    add_matmul_sample = test_fused_add_matmul()
    double_scalar_sample = test_fused_double_scalar()
    
    # Print summary
    print("\n=== Optimizer Test Summary ===")
    print(f"1. (A + B) * 2: {'✓ Passed' if np.allclose(add_mul_sample, 6.0, rtol=1e-5) else '✗ Failed'}")
    print(f"2. (A @ C) * 2: {'✓ Passed' if np.allclose(matmul_scalar_sample, 1000.0, rtol=1e-1) else '✗ Failed'}")
    print(f"3. (A + B) @ C: {'✓ Passed' if np.allclose(add_matmul_sample, 1500.0, rtol=1e-1) else '✗ Failed'}")
    print(f"4. (A * 2) * 3: {'✓ Passed' if np.allclose(double_scalar_sample, 6.0, rtol=1e-5) else '✗ Failed'}")
    
    # Clean up test data directory
    print("\nTests completed. Cleaning up test data...")
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    print("Cleanup finished.")

if __name__ == "__main__":
    run_all_optimizer_tests()
