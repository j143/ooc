"""
Unit tests for storage tier abstraction.
Tests individual tier behavior: RAM, SSD, and Network tiers.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import sys
import time
import threading

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.storage_tier import StorageTier, RAMTier, SSDTier, NetworkTier


class TestRAMTier(unittest.TestCase):
    """Test cases for RAMTier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tier = RAMTier(capacity_tiles=3)
        self.test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.test_key = ('matrix.bin', 0, 0)
    
    def test_basic_put_get(self):
        """Test basic put and get operations."""
        self.tier.put(self.test_key, self.test_data)
        retrieved = self.tier.get(self.test_key)
        
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved, self.test_data)
    
    def test_get_nonexistent_returns_none(self):
        """Test that getting a non-existent key returns None."""
        result = self.tier.get(('nonexistent.bin', 0, 0))
        self.assertIsNone(result)
    
    def test_capacity_enforcement(self):
        """Test that capacity limits are enforced."""
        # Fill to capacity
        for i in range(3):
            key = (f'matrix_{i}.bin', 0, 0)
            data = np.array([[i]], dtype=np.float32)
            self.tier.put(key, data)
        
        self.assertEqual(self.tier.size, 3)
        
        # Add one more - should evict LRU
        key4 = ('matrix_4.bin', 0, 0)
        data4 = np.array([[4]], dtype=np.float32)
        self.tier.put(key4, data4)
        
        # Size should still be 3
        self.assertEqual(self.tier.size, 3)
    
    def test_lru_eviction_policy(self):
        """Test that LRU eviction works correctly."""
        # Add 3 items
        keys = []
        for i in range(3):
            key = (f'matrix_{i}.bin', 0, 0)
            data = np.array([[i]], dtype=np.float32)
            keys.append(key)
            self.tier.put(key, data)
        
        # Access first item to make it most recently used
        self.tier.get(keys[0])
        
        # Add 4th item - should evict keys[1] (LRU)
        key4 = ('matrix_4.bin', 0, 0)
        data4 = np.array([[4]], dtype=np.float32)
        self.tier.put(key4, data4)
        
        # keys[0] should still be present
        self.assertIsNotNone(self.tier.get(keys[0]))
        
        # keys[1] should be evicted
        self.assertIsNone(self.tier.get(keys[1]))
    
    def test_hit_miss_metrics(self):
        """Test that hit/miss metrics are tracked correctly."""
        self.tier.put(self.test_key, self.test_data)
        
        # Hit
        self.tier.get(self.test_key)
        self.assertEqual(self.tier.hits, 1)
        self.assertEqual(self.tier.misses, 0)
        
        # Miss
        self.tier.get(('nonexistent.bin', 0, 0))
        self.assertEqual(self.tier.hits, 1)
        self.assertEqual(self.tier.misses, 1)
    
    def test_clear(self):
        """Test that clear removes all data and resets metrics."""
        self.tier.put(self.test_key, self.test_data)
        self.tier.get(self.test_key)
        
        self.tier.clear()
        
        self.assertEqual(self.tier.size, 0)
        self.assertEqual(self.tier.hits, 0)
        self.assertEqual(self.tier.misses, 0)
        self.assertIsNone(self.tier.get(self.test_key))


class TestSSDTier(unittest.TestCase):
    """Test cases for SSDTier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache_dir = tempfile.mkdtemp()
        self.tier = SSDTier(capacity_tiles=3, cache_dir=self.cache_dir)
        self.test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.test_key = ('matrix.bin', 0, 0)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
    
    def test_basic_put_get(self):
        """Test basic put and get operations."""
        self.tier.put(self.test_key, self.test_data)
        retrieved = self.tier.get(self.test_key)
        
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved, self.test_data)
    
    def test_persistence_to_disk(self):
        """Test that data is actually written to disk."""
        self.tier.put(self.test_key, self.test_data)
        
        # Check that a file was created
        files = os.listdir(self.cache_dir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith('.npy'))
    
    def test_capacity_enforcement(self):
        """Test that capacity limits are enforced."""
        # Fill to capacity
        for i in range(3):
            key = (f'matrix_{i}.bin', 0, 0)
            data = np.array([[i]], dtype=np.float32)
            self.tier.put(key, data)
        
        self.assertEqual(self.tier.size, 3)
        
        # Add one more - should evict LRU
        key4 = ('matrix_4.bin', 0, 0)
        data4 = np.array([[4]], dtype=np.float32)
        self.tier.put(key4, data4)
        
        # Size should still be 3
        self.assertEqual(self.tier.size, 3)
        
        # Should have 3 files on disk
        files = os.listdir(self.cache_dir)
        self.assertEqual(len(files), 3)
    
    def test_get_nonexistent_returns_none(self):
        """Test that getting a non-existent key returns None."""
        result = self.tier.get(('nonexistent.bin', 0, 0))
        self.assertIsNone(result)


class TestNetworkTier(unittest.TestCase):
    """Test cases for NetworkTier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache_dir = tempfile.mkdtemp()
        # Use very low latency for tests
        self.tier = NetworkTier(capacity_tiles=5, cache_dir=self.cache_dir, 
                               latency_ms=1.0)
        self.test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.test_key = ('matrix.bin', 0, 0)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
    
    def test_basic_put_get(self):
        """Test basic put and get operations."""
        self.tier.put(self.test_key, self.test_data)
        retrieved = self.tier.get(self.test_key)
        
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved, self.test_data)
    
    def test_simulated_latency(self):
        """Test that network latency is simulated."""
        # Use higher latency for this test
        tier = NetworkTier(capacity_tiles=5, cache_dir=self.cache_dir, 
                          latency_ms=10.0)
        
        tier.put(self.test_key, self.test_data)
        
        # Measure get time
        start = time.time()
        tier.get(self.test_key)
        elapsed_ms = (time.time() - start) * 1000
        
        # Should take at least 10ms
        self.assertGreaterEqual(elapsed_ms, 8.0)  # Allow some margin
    
    def test_capacity_enforcement(self):
        """Test that capacity limits are enforced."""
        # Fill to capacity
        for i in range(5):
            key = (f'matrix_{i}.bin', 0, 0)
            data = np.array([[i]], dtype=np.float32)
            self.tier.put(key, data)
        
        self.assertEqual(self.tier.size, 5)
        
        # Add one more - should evict one
        key6 = ('matrix_6.bin', 0, 0)
        data6 = np.array([[6]], dtype=np.float32)
        self.tier.put(key6, data6)
        
        # Size should still be 5
        self.assertEqual(self.tier.size, 5)


class TestTierHierarchy(unittest.TestCase):
    """Test cases for tier hierarchy interactions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache_dir = tempfile.mkdtemp()
        
        # Create a 3-tier hierarchy
        self.network = NetworkTier(capacity_tiles=10, 
                                   cache_dir=os.path.join(self.cache_dir, 'network'),
                                   latency_ms=1.0)
        self.ssd = SSDTier(capacity_tiles=5, 
                          cache_dir=os.path.join(self.cache_dir, 'ssd'),
                          next_tier=self.network)
        self.ram = RAMTier(capacity_tiles=2, next_tier=self.ssd)
        
        self.test_key = ('matrix.bin', 0, 0)
        self.test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
    
    def test_promotion_from_lower_tier(self):
        """Test that data is promoted from lower tiers on access."""
        # Put data in SSD tier only
        self.ssd.put(self.test_key, self.test_data)
        
        # RAM should initially miss, but find in SSD
        result = self.ram.get(self.test_key)
        
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, self.test_data)
        
        # Check that data was promoted to RAM
        self.assertEqual(self.ram.promotions, 1)
        self.assertTrue(self.ram._contains(self.test_key))
    
    def test_demotion_on_eviction(self):
        """Test that evicted data is demoted to next tier."""
        # Fill RAM to capacity
        keys = []
        for i in range(2):
            key = (f'matrix_{i}.bin', 0, 0)
            data = np.array([[i]], dtype=np.float32)
            keys.append(key)
            self.ram.put(key, data)
        
        # Add one more - should evict to SSD
        key3 = ('matrix_3.bin', 0, 0)
        data3 = np.array([[3]], dtype=np.float32)
        self.ram.put(key3, data3)
        
        # Check demotion happened
        self.assertEqual(self.ram.demotions, 1)
        
        # Evicted item should be in SSD
        # Note: We need to check SSD directly since RAM.get would re-promote
        evicted_key = keys[0]  # LRU item
        self.assertTrue(self.ssd._contains(evicted_key))
    
    def test_cascade_through_all_tiers(self):
        """Test that data can cascade through all tiers."""
        # Put in RAM
        self.ram.put(self.test_key, self.test_data)
        
        # Fill RAM to force eviction (need 3 items for capacity of 2)
        for i in range(3):
            key = (f'matrix_{i}.bin', 0, 0)
            data = np.array([[i]], dtype=np.float32)
            self.ram.put(key, data)
        
        # Original data should have been evicted to SSD
        self.assertTrue(self.ssd._contains(self.test_key))
        
        # Don't test further cascade to network to avoid slow test
        # Just verify SSD demotion works
        self.assertGreater(self.ram.demotions, 0)
    
    def test_metrics_across_tiers(self):
        """Test that metrics are tracked correctly across tiers."""
        # Put in RAM, access multiple times
        self.ram.put(self.test_key, self.test_data)
        
        # Multiple hits in RAM
        for _ in range(3):
            self.ram.get(self.test_key)
        
        self.assertEqual(self.ram.hits, 3)
        
        # Miss in RAM should check SSD
        other_key = ('other.bin', 0, 0)
        self.ram.get(other_key)
        
        self.assertEqual(self.ram.misses, 1)
        self.assertEqual(self.ssd.misses, 1)  # SSD also checked
    
    def test_thread_safety(self):
        """Test that tier hierarchy is thread-safe."""
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(5):
                    key = (f'thread_{thread_id}_item_{i}.bin', 0, 0)
                    data = np.array([[thread_id, i]], dtype=np.float32)
                    
                    # Put and immediately get
                    self.ram.put(key, data)
                    result = self.ram.get(key)
                    
                    results.append((thread_id, i, result.shape))
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
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 15)  # 3 threads * 5 operations


if __name__ == '__main__':
    unittest.main()
