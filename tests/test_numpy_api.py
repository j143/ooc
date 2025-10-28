"""
Unit tests for the NumPy-compatible API layer.

Tests the numpy_api module to ensure it provides a familiar NumPy interface
while leveraging the Paper framework's out-of-core capabilities.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import sys

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper import numpy_api as pnp


class TestNumpyAPIArrayCreation(unittest.TestCase):
    """Test cases for array creation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_array_creation_from_list(self):
        """Test creating array from nested lists."""
        data = [[1, 2, 3], [4, 5, 6]]
        arr = pnp.array(data)
        
        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(arr.size, 6)
    
    def test_array_creation_from_numpy_array(self):
        """Test creating array from numpy array."""
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr = pnp.array(np_arr)
        
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype, np.float32)
        
        # Verify data correctness
        materialized = arr._materialize()
        np.testing.assert_array_equal(materialized, np_arr)
    
    def test_array_creation_with_dtype(self):
        """Test array creation with different dtypes."""
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                arr = pnp.array([[1, 2], [3, 4]], dtype=dtype)
                self.assertEqual(arr.dtype, dtype)
    
    def test_zeros_creation(self):
        """Test zeros array creation."""
        arr = pnp.zeros((3, 4))
        
        self.assertEqual(arr.shape, (3, 4))
        self.assertEqual(arr.dtype, np.float32)
        
        # Verify all zeros
        materialized = arr._materialize()
        np.testing.assert_array_equal(materialized, np.zeros((3, 4), dtype=np.float32))
    
    def test_ones_creation(self):
        """Test ones array creation."""
        arr = pnp.ones((2, 5))
        
        self.assertEqual(arr.shape, (2, 5))
        
        # Verify all ones
        materialized = arr._materialize()
        np.testing.assert_array_equal(materialized, np.ones((2, 5), dtype=np.float32))
    
    def test_random_rand_creation(self):
        """Test random array creation."""
        arr = pnp.random_rand((3, 3))
        
        self.assertEqual(arr.shape, (3, 3))
        
        # Verify values are in [0, 1)
        materialized = arr._materialize()
        self.assertTrue(np.all(materialized >= 0))
        self.assertTrue(np.all(materialized < 1))
    
    def test_eye_creation(self):
        """Test identity matrix creation."""
        arr = pnp.eye(4)
        
        self.assertEqual(arr.shape, (4, 4))
        
        # Verify identity matrix
        materialized = arr._materialize()
        np.testing.assert_array_equal(materialized, np.eye(4, dtype=np.float32))
    
    def test_array_properties(self):
        """Test array properties (shape, ndim, size, dtype)."""
        arr = pnp.array([[1, 2, 3], [4, 5, 6]])
        
        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(arr.size, 6)
        self.assertEqual(arr.dtype, np.float32)
    
    def test_to_numpy_method(self):
        """Test to_numpy() method for converting to NumPy arrays."""
        # Test with materialized array
        arr = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        numpy_arr = arr.to_numpy()
        
        self.assertIsInstance(numpy_arr, np.ndarray)
        expected = np.array([[1, 2], [3, 4]], dtype=np.float32)
        np.testing.assert_array_equal(numpy_arr, expected)
        
        # Test with lazy array
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        c = a + b
        
        self.assertTrue(c._is_lazy)
        numpy_result = c.to_numpy()
        
        self.assertIsInstance(numpy_result, np.ndarray)
        expected_lazy = np.array([[6, 8], [10, 12]], dtype=np.float32)
        np.testing.assert_array_equal(numpy_result, expected_lazy)


class TestNumpyAPIOperations(unittest.TestCase):
    """Test cases for NumPy-compatible operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_addition_operation(self):
        """Test array addition."""
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        
        c = a + b
        
        # Check that result is lazy
        self.assertTrue(c._is_lazy)
        self.assertEqual(c.shape, (2, 2))
        
        # Compute and verify
        result = c.compute()
        expected = np.array([[6, 8], [10, 12]], dtype=np.float32)
        
        materialized = result._materialize()
        np.testing.assert_array_equal(materialized, expected)
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        
        # Test both left and right multiplication
        c1 = a * 2
        c2 = 2 * a
        
        # Both should be lazy
        self.assertTrue(c1._is_lazy)
        self.assertTrue(c2._is_lazy)
        
        # Compute and verify
        result1 = c1.compute()
        result2 = c2.compute()
        expected = np.array([[2, 4], [6, 8]], dtype=np.float32)
        
        np.testing.assert_array_equal(result1._materialize(), expected)
        np.testing.assert_array_equal(result2._materialize(), expected)
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        
        c = a @ b
        
        # Check lazy and shape
        self.assertTrue(c._is_lazy)
        self.assertEqual(c.shape, (2, 2))
        
        # Compute and verify
        result = c.compute()
        expected = np.array([[1, 2], [3, 4]], dtype=np.float32) @ np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        materialized = result._materialize()
        np.testing.assert_array_almost_equal(materialized, expected, decimal=5)
    
    def test_chained_operations(self):
        """Test chaining multiple operations."""
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        
        # (a + b) * 2
        c = (a + b) * 2
        
        self.assertTrue(c._is_lazy)
        
        # Compute and verify
        result = c.compute()
        expected = (np.array([[1, 2], [3, 4]], dtype=np.float32) + 
                   np.array([[5, 6], [7, 8]], dtype=np.float32)) * 2
        
        materialized = result._materialize()
        np.testing.assert_array_equal(materialized, expected)
    
    def test_matrix_multiplication_different_sizes(self):
        """Test matrix multiplication with non-square matrices."""
        a = pnp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # 2x3
        b = pnp.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)  # 3x2
        
        c = a @ b
        
        self.assertEqual(c.shape, (2, 2))
        
        # Compute and verify
        result = c.compute()
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32) @ np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
        
        materialized = result._materialize()
        np.testing.assert_array_almost_equal(materialized, expected, decimal=5)
    
    def test_transpose_operation(self):
        """Test array transpose."""
        a = pnp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        b = a.T
        
        self.assertEqual(b.shape, (3, 2))
        
        # Verify transpose correctness
        materialized = b._materialize()
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32).T
        
        np.testing.assert_array_equal(materialized, expected)


class TestNumpyAPIErrorHandling(unittest.TestCase):
    """Test cases for error handling in NumPy API."""
    
    def test_shape_mismatch_addition(self):
        """Test that addition with mismatched shapes raises error."""
        a = pnp.array([[1, 2], [3, 4]])
        b = pnp.array([[1, 2, 3]])
        
        with self.assertRaises(ValueError):
            c = a + b
    
    def test_dimension_mismatch_matmul(self):
        """Test that matmul with incompatible dimensions raises error."""
        a = pnp.array([[1, 2], [3, 4]])  # 2x2
        b = pnp.array([[1, 2, 3]])  # 1x3
        
        with self.assertRaises(ValueError):
            c = a @ b
    
    def test_invalid_array_creation(self):
        """Test that creating array without data or shape raises error."""
        with self.assertRaises(ValueError):
            arr = pnp.ndarray()


class TestNumpyAPIFileOperations(unittest.TestCase):
    """Test cases for file I/O operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_load_array(self):
        """Test loading array from file."""
        # Create a test file
        test_path = os.path.join(self.test_dir, "test.bin")
        test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        test_data.tofile(test_path)
        
        # Load using Paper API
        arr = pnp.load(test_path, shape=(2, 2))
        
        self.assertEqual(arr.shape, (2, 2))
        
        # Verify data
        materialized = arr._materialize()
        np.testing.assert_array_equal(materialized, test_data)
    
    def test_save_array(self):
        """Test saving array to file."""
        arr = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        
        save_path = os.path.join(self.test_dir, "saved.bin")
        pnp.save(save_path, arr)
        
        self.assertTrue(os.path.exists(save_path))
        
        # Load and verify
        loaded_data = np.fromfile(save_path, dtype=np.float32).reshape((2, 2))
        expected = np.array([[1, 2], [3, 4]], dtype=np.float32)
        np.testing.assert_array_equal(loaded_data, expected)
    
    def test_save_lazy_array(self):
        """Test saving lazy (uncomputed) array."""
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        
        c = a + b  # Lazy
        
        save_path = os.path.join(self.test_dir, "lazy_saved.bin")
        pnp.save(save_path, c)
        
        self.assertTrue(os.path.exists(save_path))
        
        # Load and verify
        loaded_data = np.fromfile(save_path, dtype=np.float32).reshape((2, 2))
        expected = np.array([[6, 8], [10, 12]], dtype=np.float32)
        np.testing.assert_array_equal(loaded_data, expected)


class TestNumpyAPIHelperFunctions(unittest.TestCase):
    """Test cases for helper functions."""
    
    def test_dot_function(self):
        """Test dot product (matrix multiplication) function."""
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        
        c = pnp.dot(a, b)
        
        # Should be same as a @ b
        self.assertTrue(c._is_lazy)
        self.assertEqual(c.shape, (2, 2))
    
    def test_add_function(self):
        """Test add function."""
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        
        c = pnp.add(a, b)
        
        # Should be same as a + b
        self.assertTrue(c._is_lazy)
        self.assertEqual(c.shape, (2, 2))
    
    def test_multiply_function(self):
        """Test multiply function."""
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        
        c = pnp.multiply(a, 2)
        
        # Should be same as a * 2
        self.assertTrue(c._is_lazy)
        self.assertEqual(c.shape, (2, 2))


class TestNumpyAPILargeArrays(unittest.TestCase):
    """Test cases for larger arrays to verify out-of-core capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_large_array_creation(self):
        """Test creating and operating on larger arrays."""
        # Create moderately large arrays (1000x1000)
        a = pnp.random_rand((1000, 1000))
        b = pnp.random_rand((1000, 1000))
        
        self.assertEqual(a.shape, (1000, 1000))
        self.assertEqual(b.shape, (1000, 1000))
        
        # Test addition
        c = a + b
        self.assertTrue(c._is_lazy)
        self.assertEqual(c.shape, (1000, 1000))
    
    def test_large_matrix_multiplication(self):
        """Test matrix multiplication on larger matrices."""
        # Create moderately sized matrices
        a = pnp.random_rand((500, 600))
        b = pnp.random_rand((600, 400))
        
        c = a @ b
        
        self.assertEqual(c.shape, (500, 400))
        
        # Just verify it can be computed without errors
        # (full verification would be too slow)
        result = c.compute()
        self.assertEqual(result.shape, (500, 400))


if __name__ == '__main__':
    unittest.main()
