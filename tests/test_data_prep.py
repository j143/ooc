"""
Tests for data preparation utilities.

This module tests the data download, conversion, and validation utilities
in the data_prep package.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_prep.download_dataset import generate_realistic_gene_expression_data, download_gene_expression_data
from data_prep.convert_to_binary import convert_to_paper_format, validate_binary_file
from paper.core import PaperMatrix


class TestDataPreparation(unittest.TestCase):
    """Test suite for data preparation utilities."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_generate_realistic_gene_expression_data(self):
        """Test generation of realistic gene expression data."""
        # Generate small dataset
        n_genes = 100
        n_samples = 50
        
        filepath, shape = generate_realistic_gene_expression_data(
            output_dir=self.test_dir,
            n_samples=n_samples,
            n_genes=n_genes,
            random_seed=42
        )
        
        # Check file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Check shape is correct
        self.assertEqual(shape, (n_genes, n_samples))
        
        # Check file size
        expected_size = n_genes * n_samples * 4  # float32 = 4 bytes
        actual_size = os.path.getsize(filepath)
        self.assertEqual(actual_size, expected_size)
        
        # Check data can be loaded
        data = np.memmap(filepath, dtype=np.float32, mode='r', shape=shape)
        
        # Check all values are non-negative (characteristic of expression data)
        self.assertTrue(np.all(data >= 0))
        
        # Check data has realistic range (not all zeros)
        self.assertGreater(np.max(data), 0)
    
    def test_download_gene_expression_data_small(self):
        """Test downloading small gene expression dataset."""
        filepath, shape = download_gene_expression_data(
            output_dir=self.test_dir,
            size="small",
            random_seed=42
        )
        
        # Check returned values
        self.assertTrue(os.path.exists(filepath))
        self.assertEqual(shape, (5000, 5000))  # small preset
    
    def test_download_gene_expression_data_reproducibility(self):
        """Test that same seed produces same data."""
        # Generate first dataset
        filepath1, shape1 = generate_realistic_gene_expression_data(
            output_dir=os.path.join(self.test_dir, "test1"),
            n_samples=100,
            n_genes=100,
            random_seed=42
        )
        
        # Generate second dataset with same seed
        filepath2, shape2 = generate_realistic_gene_expression_data(
            output_dir=os.path.join(self.test_dir, "test2"),
            n_samples=100,
            n_genes=100,
            random_seed=42
        )
        
        # Load both datasets
        data1 = np.memmap(filepath1, dtype=np.float32, mode='r', shape=shape1)
        data2 = np.memmap(filepath2, dtype=np.float32, mode='r', shape=shape2)
        
        # Check they are identical
        np.testing.assert_array_equal(data1[:], data2[:])
    
    def test_validate_binary_file(self):
        """Test binary file validation."""
        # Create a valid file
        shape = (100, 50)
        filepath = os.path.join(self.test_dir, "test.bin")
        
        data = np.random.rand(*shape).astype(np.float32)
        data.tofile(filepath)
        
        # Validate should pass
        result = validate_binary_file(filepath, shape, dtype=np.float32, n_samples=5)
        self.assertTrue(result)
    
    def test_validate_binary_file_wrong_size(self):
        """Test validation fails for wrong file size."""
        # Create a file with wrong size
        shape = (100, 50)
        wrong_shape = (100, 40)  # Wrong size
        filepath = os.path.join(self.test_dir, "test.bin")
        
        data = np.random.rand(*wrong_shape).astype(np.float32)
        data.tofile(filepath)
        
        # Validate should fail
        result = validate_binary_file(filepath, shape, dtype=np.float32)
        self.assertFalse(result)
    
    def test_validate_binary_file_missing(self):
        """Test validation fails for missing file."""
        filepath = os.path.join(self.test_dir, "nonexistent.bin")
        result = validate_binary_file(filepath, (100, 50), dtype=np.float32)
        self.assertFalse(result)
    
    def test_convert_numpy_to_paper_format(self):
        """Test conversion from NumPy to Paper format."""
        # Create a NumPy file
        shape = (100, 50)
        npy_path = os.path.join(self.test_dir, "test.npy")
        bin_path = os.path.join(self.test_dir, "test.bin")
        
        original_data = np.random.rand(*shape).astype(np.float32)
        np.save(npy_path, original_data)
        
        # Convert to Paper format
        output_path, output_shape = convert_to_paper_format(
            input_path=npy_path,
            output_path=bin_path,
            input_format="npy"
        )
        
        # Check output
        self.assertEqual(output_path, bin_path)
        self.assertEqual(output_shape, shape)
        self.assertTrue(os.path.exists(bin_path))
        
        # Verify data is identical
        converted_data = np.memmap(bin_path, dtype=np.float32, mode='r', shape=shape)
        np.testing.assert_array_almost_equal(original_data, converted_data[:])
    
    def test_download_gene_expression_data_different_seeds(self):
        """Test that different seeds produce different data."""
        # Generate first dataset
        filepath1, shape1 = generate_realistic_gene_expression_data(
            output_dir=os.path.join(self.test_dir, "test1"),
            n_samples=100,
            n_genes=100,
            random_seed=42
        )
        
        # Generate second dataset with different seed
        filepath2, shape2 = generate_realistic_gene_expression_data(
            output_dir=os.path.join(self.test_dir, "test2"),
            n_samples=100,
            n_genes=100,
            random_seed=123
        )
        
        # Load both datasets
        data1 = np.memmap(filepath1, dtype=np.float32, mode='r', shape=shape1)
        data2 = np.memmap(filepath2, dtype=np.float32, mode='r', shape=shape2)
        
        # Check they are different
        self.assertFalse(np.array_equal(data1[:], data2[:]))
    
    def test_convert_binary_to_paper_format(self):
        """Test conversion from binary to Paper format."""
        # Create a binary file
        shape = (100, 50)
        input_path = os.path.join(self.test_dir, "input.dat")
        output_path = os.path.join(self.test_dir, "output.bin")
        
        original_data = np.random.rand(*shape).astype(np.float32)
        original_data.tofile(input_path)
        
        # Convert to Paper format
        result_path, result_shape = convert_to_paper_format(
            input_path=input_path,
            output_path=output_path,
            input_format="binary",
            shape=shape
        )
        
        # Check output
        self.assertEqual(result_path, output_path)
        self.assertEqual(result_shape, shape)
        
        # Verify data
        converted_data = np.memmap(output_path, dtype=np.float32, mode='r', shape=shape)
        np.testing.assert_array_almost_equal(original_data, converted_data[:])
    
    def test_paper_matrix_can_read_generated_data(self):
        """Test that PaperMatrix can read generated data."""
        # Generate data
        filepath, shape = generate_realistic_gene_expression_data(
            output_dir=self.test_dir,
            n_samples=100,
            n_genes=100,
            random_seed=42
        )
        
        # Load with PaperMatrix
        matrix = PaperMatrix(filepath, shape, dtype=np.float32, mode='r')
        
        # Try to read a tile
        tile = matrix.get_tile(0, 0)
        
        # Check tile is valid
        self.assertIsNotNone(tile)
        self.assertEqual(tile.shape[0], min(matrix.shape[0], 512))  # TILE_SIZE
        self.assertEqual(tile.shape[1], min(matrix.shape[1], 512))
        
        # Check data is non-negative
        self.assertTrue(np.all(tile >= 0))
        
        matrix.close()


class TestDataPreparationEdgeCases(unittest.TestCase):
    """Test edge cases for data preparation utilities."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_invalid_size_preset(self):
        """Test that invalid size preset raises error."""
        with self.assertRaises(ValueError):
            download_gene_expression_data(
                output_dir=self.test_dir,
                size="invalid_size"
            )
    
    def test_auto_format_detection(self):
        """Test automatic format detection."""
        # Create NumPy file
        shape = (50, 30)
        npy_path = os.path.join(self.test_dir, "test.npy")
        bin_path = os.path.join(self.test_dir, "test.bin")
        
        data = np.random.rand(*shape).astype(np.float32)
        np.save(npy_path, data)
        
        # Convert with auto format
        output_path, output_shape = convert_to_paper_format(
            input_path=npy_path,
            output_path=bin_path,
            input_format="auto"  # Should detect .npy
        )
        
        # Verify conversion worked
        self.assertTrue(os.path.exists(bin_path))
        self.assertEqual(output_shape, shape)


if __name__ == '__main__':
    unittest.main()
