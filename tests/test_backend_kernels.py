"""
Unit tests for backend kernel operations.
Tests the correctness of out-of-core computation kernels.
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


class TestBackendKernels(unittest.TestCase):
    """Test cases for backend computation kernels."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.test_shape = (64, 64)  # Small matrices for fast testing
        self.test_dtype = np.float32
        
        # Create test matrix files
        self.matrix_A_path = os.path.join(self.test_dir, "A.bin")
        self.matrix_B_path = os.path.join(self.test_dir, "B.bin")
        self.matrix_C_path = os.path.join(self.test_dir, "C.bin")
        
        # Create test data
        self.data_A = np.random.rand(*self.test_shape).astype(self.test_dtype)
        self.data_B = np.random.rand(*self.test_shape).astype(self.test_dtype)
        
        # For matrix multiplication tests
        self.matmul_shape_A = (64, 32)
        self.matmul_shape_B = (32, 48)
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
        self.buffer_manager = BufferManager(max_cache_size_tiles=4)
    
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
    
    def test_add_kernel_correctness(self):
        """Test that the add kernel produces correct results."""
        output_path = os.path.join(self.test_dir, "result_add.bin")
        
        # Compute using backend kernel
        result = backend.add(self.A, self.B, output_path, buffer_manager=None)
        
        # Load result data
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        # Compare with expected result
        expected = self.data_A + self.data_B
        np.testing.assert_allclose(result_data, expected, rtol=1e-5, atol=1e-6)
        
        result.close()
    
    def test_add_kernel_with_buffer_manager(self):
        """Test that the add kernel works correctly with buffer manager."""
        output_path = os.path.join(self.test_dir, "result_add_buffered.bin")
        
        # Compute using backend kernel with buffer manager
        result = backend.add(self.A, self.B, output_path, buffer_manager=self.buffer_manager)
        
        # Load result data
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        # Compare with expected result
        expected = self.data_A + self.data_B
        np.testing.assert_allclose(result_data, expected, rtol=1e-5, atol=1e-6)
        
        result.close()
    
    def test_add_kernel_shape_mismatch_error(self):
        """Test that add kernel raises error for mismatched shapes."""
        # Create matrix with different shape
        wrong_shape = (32, 32)
        wrong_path = os.path.join(self.test_dir, "wrong_shape.bin")
        wrong_data = np.random.rand(*wrong_shape).astype(self.test_dtype)
        wrong_data.tofile(wrong_path)
        wrong_matrix = PaperMatrix(wrong_path, wrong_shape, dtype=self.test_dtype, mode='r')
        
        output_path = os.path.join(self.test_dir, "result_error.bin")
        
        with self.assertRaises(ValueError):
            backend.add(self.A, wrong_matrix, output_path, buffer_manager=None)
        
        wrong_matrix.close()
    
    def test_multiply_kernel_correctness(self):
        """Test that the multiply kernel produces correct results."""
        output_path = os.path.join(self.test_dir, "result_multiply.bin")
        
        # Compute using backend kernel
        result = backend.multiply(self.A_matmul, self.B_matmul, output_path, buffer_manager=None)
        
        # Load result data
        expected_shape = (self.matmul_shape_A[0], self.matmul_shape_B[1])
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(expected_shape)
        
        # Compare with expected result
        expected = self.data_A_matmul @ self.data_B_matmul
        np.testing.assert_allclose(result_data, expected, rtol=1e-4, atol=1e-5)
        
        result.close()
    
    def test_multiply_kernel_with_buffer_manager(self):
        """Test that the multiply kernel works correctly with buffer manager."""
        output_path = os.path.join(self.test_dir, "result_multiply_buffered.bin")
        
        # Compute using backend kernel with buffer manager
        result = backend.multiply(self.A_matmul, self.B_matmul, output_path, buffer_manager=self.buffer_manager)
        
        # Load result data
        expected_shape = (self.matmul_shape_A[0], self.matmul_shape_B[1])
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(expected_shape)
        
        # Compare with expected result
        expected = self.data_A_matmul @ self.data_B_matmul
        np.testing.assert_allclose(result_data, expected, rtol=1e-4, atol=1e-5)
        
        result.close()
    
    def test_multiply_kernel_dimension_mismatch_error(self):
        """Test that multiply kernel raises error for incompatible dimensions."""
        # Create matrix with incompatible dimensions for multiplication
        # A is 64x64, we need B to have incompatible first dimension
        incompatible_shape = (32, 64)  # 64x64 @ 32x64 should fail
        incompatible_path = os.path.join(self.test_dir, "incompatible.bin")
        incompatible_data = np.random.rand(*incompatible_shape).astype(self.test_dtype)
        incompatible_data.tofile(incompatible_path)
        incompatible_matrix = PaperMatrix(incompatible_path, incompatible_shape, dtype=self.test_dtype, mode='r')
        
        output_path = os.path.join(self.test_dir, "result_error.bin")
        
        with self.assertRaises(ValueError):
            backend.multiply(self.A, incompatible_matrix, output_path, buffer_manager=None)
        
        incompatible_matrix.close()
    
    def test_fused_add_multiply_kernel_correctness(self):
        """Test that the fused add-multiply kernel produces correct results."""
        output_path = os.path.join(self.test_dir, "result_fused_add_multiply.bin")
        scalar = 2.5
        
        # Compute using backend kernel
        result = backend.execute_fused_add_multiply(self.A, self.B, scalar, output_path, buffer_manager=None)
        
        # Load result data
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        # Compare with expected result
        expected = (self.data_A + self.data_B) * scalar
        np.testing.assert_allclose(result_data, expected, rtol=1e-5, atol=1e-6)
        
        result.close()
    
    def test_fused_add_multiply_with_buffer_manager(self):
        """Test that the fused add-multiply kernel works with buffer manager."""
        output_path = os.path.join(self.test_dir, "result_fused_add_multiply_buffered.bin")
        scalar = 3.0
        
        # Compute using backend kernel with buffer manager
        result = backend.execute_fused_add_multiply(self.A, self.B, scalar, output_path, buffer_manager=self.buffer_manager)
        
        # Load result data
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        # Compare with expected result
        expected = (self.data_A + self.data_B) * scalar
        np.testing.assert_allclose(result_data, expected, rtol=1e-5, atol=1e-6)
        
        result.close()
    
    def test_fused_matmul_scalar_kernel_correctness(self):
        """Test that the fused matmul-scalar kernel produces correct results."""
        output_path = os.path.join(self.test_dir, "result_fused_matmul_scalar.bin")
        scalar = 1.5
        
        # Compute using backend kernel
        result = backend.execute_fused_matmul_scalar(self.A_matmul, self.B_matmul, scalar, output_path)
        
        # Load result data
        expected_shape = (self.matmul_shape_A[0], self.matmul_shape_B[1])
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(expected_shape)
        
        # Compare with expected result
        expected = (self.data_A_matmul @ self.data_B_matmul) * scalar
        np.testing.assert_allclose(result_data, expected, rtol=1e-4, atol=1e-5)
        
        result.close()
    
    def test_fused_add_matmul_kernel_correctness(self):
        """Test that the fused add-matmul kernel produces correct results."""
        # Create a third matrix for C in (A + B) @ C
        matmul_shape_C = (self.test_shape[1], 32)  # Compatible with (A + B)
        matrix_C_path = os.path.join(self.test_dir, "C_matmul.bin")
        data_C = np.random.rand(*matmul_shape_C).astype(self.test_dtype)
        data_C.tofile(matrix_C_path)
        C_matmul = PaperMatrix(matrix_C_path, matmul_shape_C, dtype=self.test_dtype, mode='r')
        
        output_path = os.path.join(self.test_dir, "result_fused_add_matmul.bin")
        
        # Compute using backend kernel
        result = backend.execute_fused_add_matmul(self.A, self.B, C_matmul, output_path)
        
        # Load result data
        expected_shape = (self.test_shape[0], matmul_shape_C[1])
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(expected_shape)
        
        # Compare with expected result
        expected = (self.data_A + self.data_B) @ data_C
        np.testing.assert_allclose(result_data, expected, rtol=1e-4, atol=1e-5)
        
        result.close()
        C_matmul.close()
    
    def test_fused_double_scalar_kernel_correctness(self):
        """Test that the fused double scalar kernel produces correct results."""
        output_path = os.path.join(self.test_dir, "result_fused_double_scalar.bin")
        scalar1 = 2.0
        scalar2 = 3.0
        
        # Compute using backend kernel
        result = backend.execute_fused_double_scalar(self.A, scalar1, scalar2, output_path)
        
        # Load result data
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        # Compare with expected result (should be equivalent to A * (scalar1 * scalar2))
        expected = self.data_A * (scalar1 * scalar2)
        np.testing.assert_allclose(result_data, expected, rtol=1e-5, atol=1e-6)
        
        result.close()


if __name__ == '__main__':
    unittest.main()