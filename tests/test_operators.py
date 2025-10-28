"""
Unit tests for the OOCMatrix operators API.
Tests the high-level NumPy-like API for out-of-core operations.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import sys

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.operators import OOCMatrix
from paper.config import TILE_SIZE


class TestOOCMatrix(unittest.TestCase):
    """Test cases for OOCMatrix class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.test_shape = (512, 256)
        self.test_dtype = np.float32
        self.test_filepath = os.path.join(self.test_dir, "test_matrix.bin")
        
        # Create a test matrix file with known values
        self._create_test_matrix_file()
    
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_test_matrix_file(self):
        """Create a test matrix file with predictable values."""
        # Create matrix where element at (i,j) = i + j (simple for testing)
        data = np.zeros(self.test_shape, dtype=self.test_dtype)
        for i in range(self.test_shape[0]):
            for j in range(self.test_shape[1]):
                data[i, j] = i + j
        data.tofile(self.test_filepath)
    
    def test_oocmatrix_creation(self):
        """Test that OOCMatrix objects are created correctly."""
        matrix = OOCMatrix(self.test_filepath, self.test_shape, 
                          dtype=self.test_dtype, mode='r')
        
        self.assertEqual(matrix.shape, self.test_shape)
        self.assertEqual(matrix.dtype, self.test_dtype)
        self.assertTrue(os.path.exists(matrix.filepath))
        
        matrix.close()
    
    def test_oocmatrix_creation_with_create_flag(self):
        """Test creating a new OOCMatrix file."""
        new_filepath = os.path.join(self.test_dir, "new_matrix.bin")
        
        matrix = OOCMatrix(new_filepath, (100, 50), dtype=np.float32, 
                          mode='w+', create=True)
        
        self.assertTrue(os.path.exists(new_filepath))
        self.assertEqual(matrix.shape, (100, 50))
        
        matrix.close()
    
    def test_iterate_blocks(self):
        """Test iterating over matrix blocks."""
        matrix = OOCMatrix(self.test_filepath, self.test_shape, 
                          dtype=self.test_dtype, mode='r')
        
        block_count = 0
        total_elements = 0
        
        for block, (r_start, c_start) in matrix.iterate_blocks():
            block_count += 1
            total_elements += block.size
            
            # Verify block is a numpy array
            self.assertIsInstance(block, np.ndarray)
            
            # Verify block dimensions are reasonable
            self.assertLessEqual(block.shape[0], TILE_SIZE)
            self.assertLessEqual(block.shape[1], TILE_SIZE)
        
        # Verify we iterated through all elements
        expected_elements = self.test_shape[0] * self.test_shape[1]
        self.assertEqual(total_elements, expected_elements)
        self.assertGreater(block_count, 0)
        
        matrix.close()
    
    def test_blockwise_apply(self):
        """Test applying a function to each block."""
        matrix = OOCMatrix(self.test_filepath, self.test_shape, 
                          dtype=self.test_dtype, mode='r')
        
        # Apply a simple transformation: multiply by 2
        output_path = os.path.join(self.test_dir, "blockwise_result.bin")
        result = matrix.blockwise_apply(lambda x: x * 2, output_path=output_path)
        
        self.assertEqual(result.shape, matrix.shape)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify the transformation was applied
        for block, (r_start, c_start) in result.iterate_blocks():
            # Original values were i + j, multiplied by 2
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    expected = (r_start + i + c_start + j) * 2
                    actual = block[i, j]
                    self.assertAlmostEqual(actual, expected, places=5)
        
        matrix.close()
        result.close()
    
    def test_blockwise_reduce_sum(self):
        """Test blockwise reduction with sum."""
        # Create a simple matrix with all ones
        ones_path = os.path.join(self.test_dir, "ones.bin")
        ones_data = np.ones((100, 50), dtype=np.float32)
        ones_data.tofile(ones_path)
        
        matrix = OOCMatrix(ones_path, (100, 50), dtype=np.float32, mode='r')
        
        # Sum should be 100 * 50 = 5000
        total = matrix.blockwise_reduce(np.sum)
        
        expected = 100 * 50
        self.assertAlmostEqual(total, expected, places=2)
        
        matrix.close()
    
    def test_sum_method(self):
        """Test the sum() convenience method."""
        # Create a simple matrix with all ones
        ones_path = os.path.join(self.test_dir, "ones.bin")
        ones_data = np.ones((100, 50), dtype=np.float32)
        ones_data.tofile(ones_path)
        
        matrix = OOCMatrix(ones_path, (100, 50), dtype=np.float32, mode='r')
        
        total = matrix.sum()
        expected = 100 * 50
        
        self.assertAlmostEqual(total, expected, places=2)
        
        matrix.close()
    
    def test_mean_method(self):
        """Test the mean() convenience method."""
        # Create a matrix with known mean
        test_path = os.path.join(self.test_dir, "mean_test.bin")
        test_data = np.full((100, 50), 5.0, dtype=np.float32)
        test_data.tofile(test_path)
        
        matrix = OOCMatrix(test_path, (100, 50), dtype=np.float32, mode='r')
        
        mean_val = matrix.mean()
        
        self.assertAlmostEqual(mean_val, 5.0, places=5)
        
        matrix.close()
    
    def test_max_min_methods(self):
        """Test the max() and min() convenience methods."""
        # Create a matrix with known min/max
        test_path = os.path.join(self.test_dir, "minmax_test.bin")
        test_data = np.arange(100 * 50, dtype=np.float32).reshape(100, 50)
        test_data.tofile(test_path)
        
        matrix = OOCMatrix(test_path, (100, 50), dtype=np.float32, mode='r')
        
        max_val = matrix.max()
        min_val = matrix.min()
        
        self.assertAlmostEqual(max_val, 100 * 50 - 1, places=5)
        self.assertAlmostEqual(min_val, 0.0, places=5)
        
        matrix.close()
    
    def test_lazy_addition_operator(self):
        """Test lazy addition using + operator."""
        # Create two matrices
        path_a = os.path.join(self.test_dir, "A_add.bin")
        path_b = os.path.join(self.test_dir, "B_add.bin")
        
        data_a = np.ones((100, 50), dtype=np.float32) * 2
        data_b = np.ones((100, 50), dtype=np.float32) * 3
        
        data_a.tofile(path_a)
        data_b.tofile(path_b)
        
        matrix_a = OOCMatrix(path_a, (100, 50), dtype=np.float32, mode='r')
        matrix_b = OOCMatrix(path_b, (100, 50), dtype=np.float32, mode='r')
        
        # Lazy addition
        result_lazy = matrix_a + matrix_b
        
        # This should not execute yet
        self.assertIsNotNone(result_lazy._plan)
        
        # Execute the plan
        output_path = os.path.join(self.test_dir, "result_add.bin")
        result = result_lazy.compute(output_path)
        
        # Verify result
        for block, _ in result.iterate_blocks():
            # All values should be 2 + 3 = 5
            self.assertTrue(np.allclose(block, 5.0))
        
        matrix_a.close()
        matrix_b.close()
        result.close()
    
    def test_lazy_scalar_multiplication(self):
        """Test lazy scalar multiplication using * operator."""
        path = os.path.join(self.test_dir, "scalar_test.bin")
        data = np.ones((100, 50), dtype=np.float32) * 3
        data.tofile(path)
        
        matrix = OOCMatrix(path, (100, 50), dtype=np.float32, mode='r')
        
        # Lazy multiplication
        result_lazy = matrix * 2
        
        # Execute
        output_path = os.path.join(self.test_dir, "result_scalar.bin")
        result = result_lazy.compute(output_path)
        
        # Verify result
        for block, _ in result.iterate_blocks():
            # All values should be 3 * 2 = 6
            self.assertTrue(np.allclose(block, 6.0))
        
        matrix.close()
        result.close()
    
    def test_matmul_with_custom_operation(self):
        """Test matrix multiplication with custom operation."""
        # Create two small matrices
        path_a = os.path.join(self.test_dir, "A_matmul.bin")
        path_b = os.path.join(self.test_dir, "B_matmul.bin")
        
        # A is 50x30, B is 30x20
        shape_a = (50, 30)
        shape_b = (30, 20)
        
        data_a = np.random.rand(*shape_a).astype(np.float32)
        data_b = np.random.rand(*shape_b).astype(np.float32)
        
        data_a.tofile(path_a)
        data_b.tofile(path_b)
        
        matrix_a = OOCMatrix(path_a, shape_a, dtype=np.float32, mode='r')
        matrix_b = OOCMatrix(path_b, shape_b, dtype=np.float32, mode='r')
        
        # Perform matrix multiplication using numpy.dot
        output_path = os.path.join(self.test_dir, "result_matmul.bin")
        result = matrix_a.matmul(matrix_b, op=np.dot, output_path=output_path)
        
        # Verify shape
        self.assertEqual(result.shape, (50, 20))
        
        # Verify result by comparing with numpy
        expected = np.dot(data_a, data_b)
        
        # Read back the result
        result_data = np.memmap(output_path, dtype=np.float32, mode='r', shape=(50, 20))
        
        self.assertTrue(np.allclose(result_data, expected, rtol=1e-5))
        
        matrix_a.close()
        matrix_b.close()
        result.close()
    
    def test_lazy_matmul_operator(self):
        """Test lazy matrix multiplication using @ operator."""
        # Create two small matrices
        path_a = os.path.join(self.test_dir, "A_lazy_matmul.bin")
        path_b = os.path.join(self.test_dir, "B_lazy_matmul.bin")
        
        shape_a = (40, 30)
        shape_b = (30, 25)
        
        data_a = np.random.rand(*shape_a).astype(np.float32)
        data_b = np.random.rand(*shape_b).astype(np.float32)
        
        data_a.tofile(path_a)
        data_b.tofile(path_b)
        
        matrix_a = OOCMatrix(path_a, shape_a, dtype=np.float32, mode='r')
        matrix_b = OOCMatrix(path_b, shape_b, dtype=np.float32, mode='r')
        
        # Lazy matmul
        result_lazy = matrix_a @ matrix_b
        
        # Verify lazy evaluation
        self.assertIsNotNone(result_lazy._plan)
        self.assertEqual(result_lazy.shape, (40, 25))
        
        # Execute
        output_path = os.path.join(self.test_dir, "result_lazy_matmul.bin")
        result = result_lazy.compute(output_path)
        
        # Verify shape
        self.assertEqual(result.shape, (40, 25))
        
        matrix_a.close()
        matrix_b.close()
        result.close()
    
    def test_context_manager(self):
        """Test OOCMatrix as context manager."""
        with OOCMatrix(self.test_filepath, self.test_shape, 
                      dtype=self.test_dtype, mode='r') as matrix:
            # Should be able to use the matrix
            self.assertEqual(matrix.shape, self.test_shape)
            
            # Iterate over blocks
            block_count = 0
            for block, _ in matrix.iterate_blocks():
                block_count += 1
            
            self.assertGreater(block_count, 0)
        
        # Matrix should be closed after exiting context
        # (no error should occur)
    
    def test_blockwise_apply_with_numpy_operations(self):
        """Test blockwise_apply with various NumPy operations."""
        matrix = OOCMatrix(self.test_filepath, self.test_shape, 
                          dtype=self.test_dtype, mode='r')
        
        # Test normalization-like operation
        output_path = os.path.join(self.test_dir, "normalized.bin")
        result = matrix.blockwise_apply(
            lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8),
            output_path=output_path
        )
        
        self.assertEqual(result.shape, matrix.shape)
        self.assertTrue(os.path.exists(output_path))
        
        matrix.close()
        result.close()
    
    def test_std_method(self):
        """Test the std() convenience method."""
        # Create a matrix with known std
        test_path = os.path.join(self.test_dir, "std_test.bin")
        # Use values that give a predictable std
        test_data = np.array([1, 2, 3, 4, 5] * 20, dtype=np.float32).reshape(10, 10)
        test_data.tofile(test_path)
        
        matrix = OOCMatrix(test_path, (10, 10), dtype=np.float32, mode='r')
        
        std_val = matrix.std()
        expected_std = np.std(test_data)
        
        self.assertAlmostEqual(std_val, expected_std, places=4)
        
        matrix.close()


class TestOOCMatrixIntegration(unittest.TestCase):
    """Integration tests for OOCMatrix with the existing Plan infrastructure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_complex_expression_with_oocmatrix(self):
        """Test a complex expression: (A + B) * 2."""
        path_a = os.path.join(self.test_dir, "A_complex.bin")
        path_b = os.path.join(self.test_dir, "B_complex.bin")
        
        data_a = np.ones((100, 50), dtype=np.float32) * 2
        data_b = np.ones((100, 50), dtype=np.float32) * 3
        
        data_a.tofile(path_a)
        data_b.tofile(path_b)
        
        matrix_a = OOCMatrix(path_a, (100, 50), dtype=np.float32, mode='r')
        matrix_b = OOCMatrix(path_b, (100, 50), dtype=np.float32, mode='r')
        
        # Build lazy expression
        result_lazy = (matrix_a + matrix_b) * 2
        
        # Execute
        output_path = os.path.join(self.test_dir, "result_complex.bin")
        result = result_lazy.compute(output_path)
        
        # Verify result: (2 + 3) * 2 = 10
        for block, _ in result.iterate_blocks():
            self.assertTrue(np.allclose(block, 10.0))
        
        matrix_a.close()
        matrix_b.close()
        result.close()


if __name__ == '__main__':
    unittest.main()
