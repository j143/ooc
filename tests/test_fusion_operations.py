"""
Unit tests for fusion operations.
Tests the correctness of fused operations vs their non-fused equivalents.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import sys

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.core import PaperMatrix
from paper import backend
from paper.buffer import BufferManager
from paper.config import TILE_SIZE


class TestFusionOperations(unittest.TestCase):
    """Test cases for fusion operation correctness."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.test_shape = (128, 128)  # Medium-sized matrices for fusion testing
        self.test_dtype = np.float32
        
        # Create test matrix files
        self.matrix_A_path = os.path.join(self.test_dir, "A.bin")
        self.matrix_B_path = os.path.join(self.test_dir, "B.bin")
        
        # Create deterministic test data for reproducible comparisons
        np.random.seed(42)
        self.data_A = np.random.rand(*self.test_shape).astype(self.test_dtype)
        self.data_B = np.random.rand(*self.test_shape).astype(self.test_dtype)
        
        # For matrix multiplication tests
        self.matmul_shape_A = (128, 64)
        self.matmul_shape_B = (64, 96)
        self.matrix_A_matmul_path = os.path.join(self.test_dir, "A_matmul.bin")
        self.matrix_B_matmul_path = os.path.join(self.test_dir, "B_matmul.bin")
        self.data_A_matmul = np.random.rand(*self.matmul_shape_A).astype(self.test_dtype)
        self.data_B_matmul = np.random.rand(*self.matmul_shape_B).astype(self.test_dtype)
        
        # Save test data to files
        self.data_A.tofile(self.matrix_A_path)
        self.data_B.tofile(self.matrix_B_path)
        self.data_A_matmul.tofile(self.matrix_A_matmul_path)
        self.data_B_matmul.tofile(self.matrix_B_matmul_path)
        
        # Create PaperMatrix objects
        self.A = PaperMatrix(self.matrix_A_path, self.test_shape, dtype=self.test_dtype, mode='r')
        self.B = PaperMatrix(self.matrix_B_path, self.test_shape, dtype=self.test_dtype, mode='r')
        self.A_matmul = PaperMatrix(self.matrix_A_matmul_path, self.matmul_shape_A, dtype=self.test_dtype, mode='r')
        self.B_matmul = PaperMatrix(self.matrix_B_matmul_path, self.matmul_shape_B, dtype=self.test_dtype, mode='r')
        
        # Create buffer manager for testing
        self.buffer_manager = BufferManager(max_cache_size_tiles=8)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Close matrix handles
        if hasattr(self, 'A'):
            self.A.close()
        if hasattr(self, 'B'):
            self.B.close()
        if hasattr(self, 'A_matmul'):
            self.A_matmul.close()
        if hasattr(self, 'B_matmul'):
            self.B_matmul.close()
            
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_fused_vs_unfused_add_multiply(self):
        """Test that fused (A + B) * scalar gives the same result as unfused."""
        scalar = 2.5
        
        # Compute fused result
        fused_output_path = os.path.join(self.test_dir, "fused_add_multiply.bin")
        fused_result = backend.execute_fused_add_multiply(self.A, self.B, scalar, fused_output_path, buffer_manager=None)
        fused_data = np.fromfile(fused_output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        # Compute unfused result: step by step
        # Step 1: A + B
        temp_add_path = os.path.join(self.test_dir, "temp_add.bin")
        temp_add_result = backend.add(self.A, self.B, temp_add_path, buffer_manager=None)
        
        # Step 2: (A + B) * scalar
        unfused_output_path = os.path.join(self.test_dir, "unfused_add_multiply.bin")
        unfused_result = backend.execute_fused_double_scalar(temp_add_result, scalar, 1.0, unfused_output_path)
        unfused_data = np.fromfile(unfused_output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        # Compare results
        np.testing.assert_allclose(fused_data, unfused_data, rtol=1e-5, atol=1e-6,
                                 err_msg="Fused and unfused (A + B) * scalar results differ")
        
        # Clean up
        fused_result.close()
        temp_add_result.close()
        unfused_result.close()
    
    def test_fused_vs_unfused_add_multiply_with_buffer_manager(self):
        """Test fused vs unfused (A + B) * scalar with buffer manager."""
        scalar = 1.8
        
        # Compute fused result with buffer manager
        fused_output_path = os.path.join(self.test_dir, "fused_add_multiply_buf.bin")
        fused_result = backend.execute_fused_add_multiply(self.A, self.B, scalar, fused_output_path, buffer_manager=self.buffer_manager)
        fused_data = np.fromfile(fused_output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        # Compute unfused result without buffer manager for comparison
        temp_add_path = os.path.join(self.test_dir, "temp_add_buf.bin")
        temp_add_result = backend.add(self.A, self.B, temp_add_path, buffer_manager=None)
        
        unfused_output_path = os.path.join(self.test_dir, "unfused_add_multiply_buf.bin")
        unfused_result = backend.execute_fused_double_scalar(temp_add_result, scalar, 1.0, unfused_output_path)
        unfused_data = np.fromfile(unfused_output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        # Compare results
        np.testing.assert_allclose(fused_data, unfused_data, rtol=1e-5, atol=1e-6,
                                 err_msg="Fused (with buffer) and unfused results differ")
        
        # Clean up
        fused_result.close()
        temp_add_result.close()
        unfused_result.close()
    
    def test_fused_vs_unfused_matmul_scalar(self):
        """Test that fused (A @ B) * scalar gives the same result as unfused."""
        scalar = 3.2
        
        # Compute fused result
        fused_output_path = os.path.join(self.test_dir, "fused_matmul_scalar.bin")
        fused_result = backend.execute_fused_matmul_scalar(self.A_matmul, self.B_matmul, scalar, fused_output_path)
        expected_shape = (self.matmul_shape_A[0], self.matmul_shape_B[1])
        fused_data = np.fromfile(fused_output_path, dtype=self.test_dtype).reshape(expected_shape)
        
        # Compute unfused result: step by step
        # Step 1: A @ B
        temp_matmul_path = os.path.join(self.test_dir, "temp_matmul.bin")
        temp_matmul_result = backend.multiply(self.A_matmul, self.B_matmul, temp_matmul_path, buffer_manager=None)
        
        # Step 2: (A @ B) * scalar
        unfused_output_path = os.path.join(self.test_dir, "unfused_matmul_scalar.bin")
        unfused_result = backend.execute_fused_double_scalar(temp_matmul_result, scalar, 1.0, unfused_output_path)
        unfused_data = np.fromfile(unfused_output_path, dtype=self.test_dtype).reshape(expected_shape)
        
        # Compare results
        np.testing.assert_allclose(fused_data, unfused_data, rtol=1e-4, atol=1e-5,
                                 err_msg="Fused and unfused (A @ B) * scalar results differ")
        
        # Clean up
        fused_result.close()
        temp_matmul_result.close()
        unfused_result.close()
    
    def test_fused_vs_unfused_add_matmul(self):
        """Test that fused (A + B) @ C gives the same result as unfused."""
        # Create a third matrix C compatible with (A + B)
        matmul_shape_C = (self.test_shape[1], 80)
        matrix_C_path = os.path.join(self.test_dir, "C_matmul.bin")
        data_C = np.random.rand(*matmul_shape_C).astype(self.test_dtype)
        data_C.tofile(matrix_C_path)
        C_matmul = PaperMatrix(matrix_C_path, matmul_shape_C, dtype=self.test_dtype, mode='r')
        
        # Compute fused result
        fused_output_path = os.path.join(self.test_dir, "fused_add_matmul.bin")
        fused_result = backend.execute_fused_add_matmul(self.A, self.B, C_matmul, fused_output_path)
        expected_shape = (self.test_shape[0], matmul_shape_C[1])
        fused_data = np.fromfile(fused_output_path, dtype=self.test_dtype).reshape(expected_shape)
        
        # Compute unfused result: step by step
        # Step 1: A + B
        temp_add_path = os.path.join(self.test_dir, "temp_add_for_matmul.bin")
        temp_add_result = backend.add(self.A, self.B, temp_add_path, buffer_manager=None)
        
        # Step 2: (A + B) @ C
        unfused_output_path = os.path.join(self.test_dir, "unfused_add_matmul.bin")
        unfused_result = backend.multiply(temp_add_result, C_matmul, unfused_output_path, buffer_manager=None)
        unfused_data = np.fromfile(unfused_output_path, dtype=self.test_dtype).reshape(expected_shape)
        
        # Compare results
        np.testing.assert_allclose(fused_data, unfused_data, rtol=1e-4, atol=1e-5,
                                 err_msg="Fused and unfused (A + B) @ C results differ")
        
        # Clean up
        fused_result.close()
        temp_add_result.close()
        unfused_result.close()
        C_matmul.close()
    
    def test_fused_double_scalar_optimization(self):
        """Test that fused double scalar is equivalent to direct computation."""
        scalar1 = 2.0
        scalar2 = 3.5
        
        # Compute fused result
        fused_output_path = os.path.join(self.test_dir, "fused_double_scalar.bin")
        fused_result = backend.execute_fused_double_scalar(self.A, scalar1, scalar2, fused_output_path)
        fused_data = np.fromfile(fused_output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        # Compute expected result directly
        expected = self.data_A * (scalar1 * scalar2)
        
        # Compare results
        np.testing.assert_allclose(fused_data, expected, rtol=1e-5, atol=1e-6,
                                 err_msg="Fused double scalar result differs from expected")
        
        # Clean up
        fused_result.close()
    
    def test_fusion_reduces_memory_footprint(self):
        """Test that fusion operations don't create intermediate files."""
        scalar = 2.0
        
        # Count files before operation
        files_before = len(os.listdir(self.test_dir))
        
        # Perform fused operation
        fused_output_path = os.path.join(self.test_dir, "fused_test.bin")
        fused_result = backend.execute_fused_add_multiply(self.A, self.B, scalar, fused_output_path, buffer_manager=None)
        
        # Count files after operation
        files_after = len(os.listdir(self.test_dir))
        
        # Only one new file should be created (the output)
        self.assertEqual(files_after - files_before, 1, 
                        "Fused operation should create only one output file")
        
        # Clean up
        fused_result.close()
    
    def test_fusion_thread_safety(self):
        """Test that fused operations work correctly with threading."""
        import threading
        import time
        
        scalar = 1.5
        results = []
        errors = []
        
        def run_fused_operation(thread_id):
            try:
                output_path = os.path.join(self.test_dir, f"fused_thread_{thread_id}.bin")
                result = backend.execute_fused_add_multiply(self.A, self.B, scalar, output_path, buffer_manager=None)
                result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(self.test_shape)
                results.append((thread_id, result_data))
                result.close()
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_fused_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that no errors occurred
        self.assertEqual(len(errors), 0, f"Thread errors occurred: {errors}")
        
        # Check that all results are identical
        self.assertEqual(len(results), 3, "All threads should complete")
        expected = (self.data_A + self.data_B) * scalar
        
        for thread_id, result_data in results:
            np.testing.assert_allclose(result_data, expected, rtol=1e-5, atol=1e-6,
                                     err_msg=f"Thread {thread_id} result differs")
    
    def test_large_tile_fusion_correctness(self):
        """Test fusion operations with matrices that require multiple tiles."""
        # Create larger matrices that will definitely require multiple tiles
        large_shape = (TILE_SIZE * 3, TILE_SIZE * 2)  # 3x2 tiles
        large_A_path = os.path.join(self.test_dir, "large_A.bin")
        large_B_path = os.path.join(self.test_dir, "large_B.bin")
        
        # Create large test data
        large_data_A = np.random.rand(*large_shape).astype(self.test_dtype)
        large_data_B = np.random.rand(*large_shape).astype(self.test_dtype)
        large_data_A.tofile(large_A_path)
        large_data_B.tofile(large_B_path)
        
        large_A = PaperMatrix(large_A_path, large_shape, dtype=self.test_dtype, mode='r')
        large_B = PaperMatrix(large_B_path, large_shape, dtype=self.test_dtype, mode='r')
        
        scalar = 2.7
        
        # Compute fused result
        fused_output_path = os.path.join(self.test_dir, "large_fused.bin")
        fused_result = backend.execute_fused_add_multiply(large_A, large_B, scalar, fused_output_path, buffer_manager=None)
        fused_data = np.fromfile(fused_output_path, dtype=self.test_dtype).reshape(large_shape)
        
        # Compute expected result
        expected = (large_data_A + large_data_B) * scalar
        
        # Compare results
        np.testing.assert_allclose(fused_data, expected, rtol=1e-5, atol=1e-6,
                                 err_msg="Large matrix fusion result differs from expected")
        
        # Clean up
        large_A.close()
        large_B.close()
        fused_result.close()


if __name__ == '__main__':
    unittest.main()