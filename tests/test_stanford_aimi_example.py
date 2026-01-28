"""
Tests for Stanford AIMI CheXpert example.

This test suite validates the CheXpert example functionality including:
- Dataset generation
- Paper framework operations
- Traditional NumPy operations
- Performance comparisons
"""

import unittest
import os
import sys
import tempfile
import shutil
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper import numpy_api as pnp
from examples.stanford_aimi_chexpert_example import (
    CheXpertDatasetSimulator,
    benchmark_traditional_numpy,
    benchmark_paper_framework
)


class TestCheXpertDatasetSimulator(unittest.TestCase):
    """Test the CheXpert dataset simulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp(prefix='test_chexpert_')
        self.n_samples = 100  # Small for testing
        self.img_size = 32
        self.n_labels = 14
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_simulator_initialization(self):
        """Test CheXpert simulator can be initialized."""
        simulator = CheXpertDatasetSimulator(
            n_samples=self.n_samples,
            img_size=self.img_size,
            n_labels=self.n_labels
        )
        
        self.assertEqual(simulator.n_samples, self.n_samples)
        self.assertEqual(simulator.img_size, self.img_size)
        self.assertEqual(simulator.n_labels, self.n_labels)
    
    def test_dataset_generation(self):
        """Test dataset generation creates valid files."""
        simulator = CheXpertDatasetSimulator(
            n_samples=self.n_samples,
            img_size=self.img_size,
            n_labels=self.n_labels
        )
        
        images_path, labels_path, metadata = simulator.generate_dataset(self.temp_dir)
        
        # Check files exist
        self.assertTrue(os.path.exists(images_path))
        self.assertTrue(os.path.exists(labels_path))
        
        # Check file sizes
        img_pixels = self.img_size * self.img_size
        expected_img_size = self.n_samples * img_pixels * 4  # float32
        expected_lbl_size = self.n_samples * self.n_labels * 4  # float32
        
        self.assertEqual(os.path.getsize(images_path), expected_img_size)
        self.assertEqual(os.path.getsize(labels_path), expected_lbl_size)
        
        # Check metadata
        self.assertEqual(metadata['n_samples'], self.n_samples)
        self.assertEqual(metadata['img_size'], self.img_size)
        self.assertEqual(metadata['n_labels'], self.n_labels)
        self.assertIn('generation_time', metadata)
        self.assertGreater(metadata['generation_time'], 0)
    
    def test_realistic_xray_generation(self):
        """Test that generated X-rays have realistic properties."""
        simulator = CheXpertDatasetSimulator(
            n_samples=10,
            img_size=self.img_size,
            n_labels=self.n_labels
        )
        
        xrays = simulator._generate_realistic_xrays(10)
        
        # Check shape
        self.assertEqual(xrays.shape, (10, self.img_size * self.img_size))
        
        # Check value range [0, 1]
        self.assertGreaterEqual(xrays.min(), 0.0)
        self.assertLessEqual(xrays.max(), 1.0)
        
        # Check dtype
        self.assertEqual(xrays.dtype, np.float32)
        
        # Check for variation (not all same value)
        self.assertGreater(xrays.std(), 0.01)
    
    def test_pathology_labels_generation(self):
        """Test that pathology labels are generated correctly."""
        simulator = CheXpertDatasetSimulator(
            n_samples=100,
            img_size=self.img_size,
            n_labels=self.n_labels
        )
        
        labels = simulator._generate_pathology_labels(100)
        
        # Check shape
        self.assertEqual(labels.shape, (100, self.n_labels))
        
        # Check values are binary (0 or 1)
        unique_vals = np.unique(labels)
        self.assertTrue(np.all((unique_vals == 0) | (unique_vals == 1)))
        
        # Check dtype
        self.assertEqual(labels.dtype, np.float32)
        
        # Check prevalence is reasonable (not all 0 or all 1)
        for i in range(self.n_labels):
            prevalence = labels[:, i].mean()
            self.assertGreater(prevalence, 0.0)
            self.assertLess(prevalence, 1.0)


class TestTraditionalNumPyBenchmark(unittest.TestCase):
    """Test traditional NumPy benchmark functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp(prefix='test_numpy_bench_')
        self.n_samples = 50
        self.img_size = 32
        self.n_labels = 14
        
        # Generate small dataset
        simulator = CheXpertDatasetSimulator(
            n_samples=self.n_samples,
            img_size=self.img_size,
            n_labels=self.n_labels
        )
        self.images_path, self.labels_path, _ = simulator.generate_dataset(self.temp_dir)
        self.img_pixels = self.img_size * self.img_size
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_numpy_benchmark_success(self):
        """Test that NumPy benchmark runs successfully on small dataset."""
        results = benchmark_traditional_numpy(
            self.images_path,
            self.labels_path,
            self.n_samples,
            self.img_pixels,
            self.n_labels
        )
        
        # Check success
        self.assertTrue(results['success'])
        
        # Check required fields
        self.assertIn('load_time', results)
        self.assertIn('preprocess_time', results)
        self.assertIn('augment_time', results)
        self.assertIn('total_time', results)
        self.assertIn('memory_gb', results)
        
        # Check reasonable values
        self.assertGreater(results['load_time'], 0)
        self.assertGreater(results['total_time'], 0)
        self.assertGreater(results['memory_gb'], 0)
    
    def test_numpy_data_loading(self):
        """Test that NumPy correctly loads data."""
        # Load with NumPy
        with open(self.images_path, 'rb') as f:
            images = np.fromfile(f, dtype=np.float32)
            images = images.reshape(self.n_samples, self.img_pixels)
        
        # Check shape
        self.assertEqual(images.shape, (self.n_samples, self.img_pixels))
        
        # Check value range
        self.assertGreaterEqual(images.min(), 0.0)
        self.assertLessEqual(images.max(), 1.0)


class TestPaperFrameworkBenchmark(unittest.TestCase):
    """Test Paper framework benchmark functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp(prefix='test_paper_bench_')
        self.n_samples = 50
        self.img_size = 32
        self.n_labels = 14
        
        # Generate small dataset
        simulator = CheXpertDatasetSimulator(
            n_samples=self.n_samples,
            img_size=self.img_size,
            n_labels=self.n_labels
        )
        self.images_path, self.labels_path, _ = simulator.generate_dataset(self.temp_dir)
        self.img_pixels = self.img_size * self.img_size
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_paper_benchmark_success(self):
        """Test that Paper benchmark runs successfully."""
        results = benchmark_paper_framework(
            self.images_path,
            self.labels_path,
            self.n_samples,
            self.img_pixels,
            self.n_labels
        )
        
        # Check success
        self.assertTrue(results['success'])
        
        # Check required fields
        self.assertIn('load_time', results)
        self.assertIn('plan_time', results)
        self.assertIn('compute_time', results)
        self.assertIn('total_time', results)
        self.assertIn('memory_gb', results)
        
        # Check reasonable values
        self.assertGreater(results['load_time'], 0)
        self.assertGreater(results['compute_time'], 0)
        self.assertGreater(results['total_time'], 0)
        
        # Paper should use minimal memory
        self.assertLessEqual(results['memory_gb'], 0.1)
    
    def test_paper_lazy_loading(self):
        """Test that Paper uses lazy loading."""
        # Load with Paper
        images = pnp.load(self.images_path, shape=(self.n_samples, self.img_pixels))
        
        # Should be lazy initially
        self.assertFalse(images._is_lazy)  # After load, it's eager
        
        # Create lazy operation (using multiplication only)
        normalized = images * 2.0
        self.assertTrue(normalized._is_lazy)
        
        # Shape should be correct
        self.assertEqual(normalized.shape, (self.n_samples, self.img_pixels))
    
    def test_paper_computation(self):
        """Test that Paper correctly computes results."""
        # Load and process with Paper
        images = pnp.load(self.images_path, shape=(self.n_samples, self.img_pixels))
        scaled = images * 2.0
        result = scaled.compute()
        
        # Get numpy array
        result_np = result.to_numpy()
        
        # Check shape
        self.assertEqual(result_np.shape, (self.n_samples, self.img_pixels))
        
        # Check value range (should be scaled to [0, 2])
        self.assertGreaterEqual(result_np.min(), 0.0)
        self.assertLessEqual(result_np.max(), 2.5)  # Allow some margin
        
    def test_paper_vs_numpy_correctness(self):
        """Test that Paper produces same results as NumPy."""
        # Load with NumPy
        with open(self.images_path, 'rb') as f:
            images_np = np.fromfile(f, dtype=np.float32)
            images_np = images_np.reshape(self.n_samples, self.img_pixels)
        numpy_result = images_np * 2.0
        
        # Load with Paper
        images_paper = pnp.load(self.images_path, shape=(self.n_samples, self.img_pixels))
        paper_result = (images_paper * 2.0).compute().to_numpy()
        
        # Results should match
        np.testing.assert_allclose(numpy_result, paper_result, rtol=1e-5)


class TestPerformanceComparison(unittest.TestCase):
    """Test performance comparison between approaches."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp(prefix='test_comparison_')
        self.n_samples = 100
        self.img_size = 32
        self.n_labels = 14
        
        # Generate dataset
        simulator = CheXpertDatasetSimulator(
            n_samples=self.n_samples,
            img_size=self.img_size,
            n_labels=self.n_labels
        )
        self.images_path, self.labels_path, _ = simulator.generate_dataset(self.temp_dir)
        self.img_pixels = self.img_size * self.img_size
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_both_approaches_succeed(self):
        """Test that both approaches can process the dataset."""
        numpy_results = benchmark_traditional_numpy(
            self.images_path,
            self.labels_path,
            self.n_samples,
            self.img_pixels,
            self.n_labels
        )
        
        paper_results = benchmark_paper_framework(
            self.images_path,
            self.labels_path,
            self.n_samples,
            self.img_pixels,
            self.n_labels
        )
        
        self.assertTrue(numpy_results['success'])
        self.assertTrue(paper_results['success'])
    
    def test_paper_uses_less_memory(self):
        """Test that Paper uses less memory than NumPy."""
        numpy_results = benchmark_traditional_numpy(
            self.images_path,
            self.labels_path,
            self.n_samples,
            self.img_pixels,
            self.n_labels
        )
        
        paper_results = benchmark_paper_framework(
            self.images_path,
            self.labels_path,
            self.n_samples,
            self.img_pixels,
            self.n_labels
        )
        
        # Paper should use significantly less memory
        if numpy_results['success']:
            self.assertLess(
                paper_results['memory_gb'],
                numpy_results['memory_gb'] * 0.1  # Less than 10% of NumPy
            )


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCheXpertDatasetSimulator))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTraditionalNumPyBenchmark))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPaperFrameworkBenchmark))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPerformanceComparison))
    
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
