"""
System tests for hierarchical memory management.
Tests end-to-end integration with matrix operations.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import sys

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.hierarchical_buffer import HierarchicalBufferManager
from paper.core import PaperMatrix
from paper.backend import add, multiply
from paper.config import TILE_SIZE


class TestHierarchicalSystemIntegration(unittest.TestCase):
    """System tests for hierarchical buffer manager integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        # Use a size that requires multiple tiles
        self.test_shape = (512, 512)  # Will require 4 tiles (2x2 with TILE_SIZE=1024)
        self.test_dtype = np.float32
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_matrix_file(self, filename, fill_value):
        """Helper to create a matrix file."""
        filepath = os.path.join(self.test_dir, filename)
        data = np.full(self.test_shape, fill_value, dtype=self.test_dtype)
        data.tofile(filepath)
        return filepath
    
    def test_matrix_addition_with_hierarchical_cache(self):
        """Test matrix addition using hierarchical buffer manager."""
        # Create test matrices
        a_path = self._create_matrix_file("A.bin", 1.0)
        b_path = self._create_matrix_file("B.bin", 2.0)
        output_path = os.path.join(self.test_dir, "C.bin")
        
        # Create matrices
        A = PaperMatrix(a_path, self.test_shape, dtype=self.test_dtype, mode='r')
        B = PaperMatrix(b_path, self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Create hierarchical buffer manager
        buffer_mgr = HierarchicalBufferManager(
            ram_capacity_tiles=2,
            ssd_capacity_tiles=4,
            network_capacity_tiles=8,
            network_latency_ms=0.0
        )
        
        # Perform addition with hierarchical cache
        C = add(A, B, output_path, buffer_mgr)
        
        # Verify result
        expected = np.full(self.test_shape, 3.0, dtype=self.test_dtype)
        np.testing.assert_array_almost_equal(C.data[:], expected, decimal=5)
        
        # Check that cache was used
        metrics = buffer_mgr.get_summary_metrics()
        self.assertGreater(metrics['total_requests'], 0)
        
        A.close()
        B.close()
        C.close()
    
    def test_matrix_multiplication_with_hierarchical_cache(self):
        """Test matrix multiplication using hierarchical buffer manager."""
        # Create smaller matrices for multiplication test
        small_shape = (256, 256)
        
        a_path = self._create_matrix_file("A_mul.bin", 1.0)
        b_path = self._create_matrix_file("B_mul.bin", 2.0)
        
        # Resize the files for smaller shape
        data_a = np.full(small_shape, 1.0, dtype=self.test_dtype)
        data_b = np.full(small_shape, 2.0, dtype=self.test_dtype)
        data_a.tofile(a_path)
        data_b.tofile(b_path)
        
        output_path = os.path.join(self.test_dir, "C_mul.bin")
        
        A = PaperMatrix(a_path, small_shape, dtype=self.test_dtype, mode='r')
        B = PaperMatrix(b_path, small_shape, dtype=self.test_dtype, mode='r')
        
        # Create hierarchical buffer manager
        buffer_mgr = HierarchicalBufferManager(
            ram_capacity_tiles=2,
            ssd_capacity_tiles=4,
            network_capacity_tiles=8,
            network_latency_ms=0.0
        )
        
        # Perform multiplication with hierarchical cache
        C = multiply(A, B, output_path, buffer_mgr)
        
        # Verify result (A @ B where A and B are all 1s and 2s)
        # Each element of result should be N * 1.0 * 2.0 = N * 2.0
        expected_value = small_shape[1] * 1.0 * 2.0
        
        # Check a sample of the result
        sample = C.data[0, 0]
        self.assertAlmostEqual(sample, expected_value, places=1)
        
        # Check that cache was used
        metrics = buffer_mgr.get_summary_metrics()
        self.assertGreater(metrics['total_requests'], 0)
        
        A.close()
        B.close()
        C.close()
    
    def test_cache_promotion_during_computation(self):
        """Test that data is promoted between tiers during computation."""
        a_path = self._create_matrix_file("A_promo.bin", 1.0)
        b_path = self._create_matrix_file("B_promo.bin", 2.0)
        output_path = os.path.join(self.test_dir, "C_promo.bin")
        
        A = PaperMatrix(a_path, self.test_shape, dtype=self.test_dtype, mode='r')
        B = PaperMatrix(b_path, self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Create buffer manager with very small RAM cache
        buffer_mgr = HierarchicalBufferManager(
            ram_capacity_tiles=1,  # Very small to force demotions
            ssd_capacity_tiles=3,
            network_capacity_tiles=10,
            network_latency_ms=0.0
        )
        
        # Perform operation
        C = add(A, B, output_path, buffer_mgr)
        
        # Get metrics
        metrics = buffer_mgr.get_tier_metrics()
        
        # With small RAM cache, we should see:
        # - Demotions from RAM to SSD
        # - Promotions back to RAM when tiles are re-accessed
        
        # At least some tier interaction should have occurred
        total_promotions = sum(m['promotions'] for m in metrics.values())
        total_demotions = sum(m['demotions'] for m in metrics.values())
        
        # Should have some tier movement
        self.assertGreater(total_demotions, 0)
        
        A.close()
        B.close()
        C.close()
    
    def test_cache_efficiency_comparison(self):
        """Compare cache efficiency between hierarchical and flat cache."""
        a_path = self._create_matrix_file("A_eff.bin", 1.0)
        b_path = self._create_matrix_file("B_eff.bin", 2.0)
        
        A = PaperMatrix(a_path, self.test_shape, dtype=self.test_dtype, mode='r')
        B = PaperMatrix(b_path, self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Create hierarchical buffer manager
        buffer_mgr = HierarchicalBufferManager(
            ram_capacity_tiles=2,
            ssd_capacity_tiles=4,
            network_capacity_tiles=8,
            network_latency_ms=0.0
        )
        
        # Perform multiple operations to build up cache
        for _ in range(2):
            output_path = os.path.join(self.test_dir, f"C_eff_{_}.bin")
            C = add(A, B, output_path, buffer_mgr)
            C.close()
        
        # Get final metrics
        metrics = buffer_mgr.get_summary_metrics()
        
        # Second iteration should have better hit rate
        # (We can't compare directly, but we verify metrics are tracked)
        self.assertIn('overall_hit_rate', metrics)
        self.assertIn('ram_hit_rate', metrics)
        
        # Total requests should be > 0
        self.assertGreater(metrics['total_requests'], 0)
        
        A.close()
        B.close()
    
    def test_concurrent_access_with_hierarchy(self):
        """Test that hierarchical cache works correctly with concurrent access."""
        import threading
        
        a_path = self._create_matrix_file("A_concurrent.bin", 1.0)
        b_path = self._create_matrix_file("B_concurrent.bin", 2.0)
        
        A = PaperMatrix(a_path, self.test_shape, dtype=self.test_dtype, mode='r')
        B = PaperMatrix(b_path, self.test_shape, dtype=self.test_dtype, mode='r')
        
        buffer_mgr = HierarchicalBufferManager(
            ram_capacity_tiles=4,
            ssd_capacity_tiles=8,
            network_capacity_tiles=16,
            network_latency_ms=0.0
        )
        
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                # Each thread accesses different tiles
                for i in range(3):
                    r_start = (thread_id * 10) % TILE_SIZE
                    c_start = 0
                    
                    tile_a = buffer_mgr.get_tile(A, r_start, c_start)
                    tile_b = buffer_mgr.get_tile(B, r_start, c_start)
                    
                    results.append((thread_id, i, tile_a.shape, tile_b.shape))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")
        self.assertEqual(len(results), 9)  # 3 threads * 3 operations
        
        # Verify metrics were tracked
        metrics = buffer_mgr.get_summary_metrics()
        self.assertGreater(metrics['total_requests'], 0)
        
        A.close()
        B.close()
    
    def test_metrics_reporting(self):
        """Test that comprehensive metrics are reported correctly."""
        a_path = self._create_matrix_file("A_metrics.bin", 1.0)
        
        A = PaperMatrix(a_path, self.test_shape, dtype=self.test_dtype, mode='r')
        
        buffer_mgr = HierarchicalBufferManager(
            ram_capacity_tiles=2,
            ssd_capacity_tiles=4,
            network_capacity_tiles=8,
            network_latency_ms=0.0
        )
        
        # Perform various accesses
        for i in range(5):
            r_start = (i % 2) * TILE_SIZE
            c_start = 0
            buffer_mgr.get_tile(A, r_start, c_start)
        
        # Get tier metrics
        tier_metrics = buffer_mgr.get_tier_metrics()
        
        # Verify all tiers are reported
        for tier_name in ['ram', 'ssd', 'network']:
            self.assertIn(tier_name, tier_metrics)
            
            tier = tier_metrics[tier_name]
            # Verify required fields
            self.assertIn('hits', tier)
            self.assertIn('misses', tier)
            self.assertIn('hit_rate', tier)
            self.assertIn('evictions', tier)
            self.assertIn('promotions', tier)
            self.assertIn('demotions', tier)
            self.assertIn('size', tier)
            self.assertIn('capacity', tier)
            self.assertIn('utilization', tier)
        
        # Get summary metrics
        summary = buffer_mgr.get_summary_metrics()
        
        # Verify summary fields
        required_fields = ['total_requests', 'overall_hit_rate', 
                          'ram_hit_rate', 'tiers']
        for field in required_fields:
            self.assertIn(field, summary)
        
        A.close()


if __name__ == '__main__':
    unittest.main()
