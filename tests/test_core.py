"""
Unit tests for the PaperMatrix class.
Tests the core functionality of out-of-core matrix operations.
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
from paper.config import TILE_SIZE


class TestPaperMatrix(unittest.TestCase):
    """Test cases for PaperMatrix class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.test_shape = (512, 256)  # Small test matrices for fast execution
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
        # Create matrix where element at (i,j) = i * shape[1] + j
        data = np.arange(self.test_shape[0] * self.test_shape[1], dtype=self.test_dtype)
        data = data.reshape(self.test_shape)
        data.tofile(self.test_filepath)
    
    def test_papermatrix_creation_with_correct_shape(self):
        """Test that PaperMatrix objects are created with the correct shape."""
        matrix = PaperMatrix(self.test_filepath, self.test_shape, dtype=self.test_dtype, mode='r')
        
        self.assertEqual(matrix.shape, self.test_shape)
        self.assertEqual(matrix.dtype, self.test_dtype)
        self.assertEqual(matrix.filepath, self.test_filepath)
        
        matrix.close()
    
    def test_papermatrix_creation_with_correct_data_type(self):
        """Test that PaperMatrix objects are created with the correct data type."""
        # Test with different data types
        for dtype in [np.float32, np.float64, np.int32]:
            with self.subTest(dtype=dtype):
                # Create test file with specific dtype
                test_path = os.path.join(self.test_dir, f"test_{dtype.__name__}.bin")
                test_data = np.ones(self.test_shape, dtype=dtype)
                test_data.tofile(test_path)
                
                matrix = PaperMatrix(test_path, self.test_shape, dtype=dtype, mode='r')
                self.assertEqual(matrix.dtype, dtype)
                matrix.close()
    
    def test_get_tile_retrieves_correct_data_slice(self):
        """Test that get_tile() method retrieves the correct data slice from a file."""
        matrix = PaperMatrix(self.test_filepath, self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Test tile retrieval from different positions
        test_cases = [
            (0, 0),      # Top-left corner
            (128, 64),   # Middle region
            (256, 128),  # Different position
        ]
        
        for r_start, c_start in test_cases:
            with self.subTest(r_start=r_start, c_start=c_start):
                tile = matrix.get_tile(r_start, c_start)
                
                # Calculate expected tile size
                r_end = min(r_start + TILE_SIZE, self.test_shape[0])
                c_end = min(c_start + TILE_SIZE, self.test_shape[1])
                expected_shape = (r_end - r_start, c_end - c_start)
                
                # Verify tile shape
                self.assertEqual(tile.shape, expected_shape)
                
                # Verify tile data correctness
                # Our test matrix has element at (i,j) = i * shape[1] + j
                for i in range(tile.shape[0]):
                    for j in range(tile.shape[1]):
                        global_i = r_start + i
                        global_j = c_start + j
                        expected_value = global_i * self.test_shape[1] + global_j
                        self.assertEqual(tile[i, j], expected_value)
        
        matrix.close()
    
    def test_get_tile_boundary_conditions(self):
        """Test get_tile() method at matrix boundaries."""
        matrix = PaperMatrix(self.test_filepath, self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Test at the edge of the matrix where tile would extend beyond bounds
        r_start = self.test_shape[0] - 64  # Near bottom edge
        c_start = self.test_shape[1] - 32  # Near right edge
        
        tile = matrix.get_tile(r_start, c_start)
        
        # Verify the tile is correctly clipped to matrix boundaries
        expected_rows = self.test_shape[0] - r_start
        expected_cols = self.test_shape[1] - c_start
        
        self.assertEqual(tile.shape, (expected_rows, expected_cols))
        
        matrix.close()
    
    def test_get_tile_returns_copy_not_view(self):
        """Test that get_tile() returns a copy, not a view of the data."""
        matrix = PaperMatrix(self.test_filepath, self.test_shape, dtype=self.test_dtype, mode='r')
        
        tile = matrix.get_tile(0, 0)
        original_value = tile[0, 0]
        
        # Modify the tile
        tile[0, 0] = original_value + 1000
        
        # Get the same tile again and verify it wasn't affected
        tile2 = matrix.get_tile(0, 0)
        self.assertEqual(tile2[0, 0], original_value, 
                        "get_tile() should return a copy, not a view")
        
        matrix.close()


if __name__ == '__main__':
    unittest.main()