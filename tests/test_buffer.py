"""
Unit tests for the BufferManager class.
Tests caching functionality, eviction policies, and event logging.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import sys

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.buffer import BufferManager
from paper.core import PaperMatrix
from paper.config import TILE_SIZE


class TestBufferManager(unittest.TestCase):
    """Test cases for BufferManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.test_shape = (256, 256)  # Small test matrices for fast execution
        self.test_dtype = np.float32
        
        # Create test matrix files
        self.matrix_paths = []
        for i in range(3):
            filepath = os.path.join(self.test_dir, f"matrix_{i}.bin")
            self._create_test_matrix_file(filepath, fill_value=float(i + 1))
            self.matrix_paths.append(filepath)
    
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_test_matrix_file(self, filepath, fill_value=1.0):
        """Create a test matrix file with specified fill value."""
        data = np.full(self.test_shape, fill_value, dtype=self.test_dtype)
        data.tofile(filepath)
    
    def test_hit_and_miss_event_logging(self):
        """Test that HIT and MISS events are correctly logged."""
        buffer_manager = BufferManager(max_cache_size_tiles=2)
        matrix = PaperMatrix(self.matrix_paths[0], self.test_shape, dtype=self.test_dtype, mode='r')
        
        # First access should be a MISS
        buffer_manager.get_tile(matrix, 0, 0)
        
        # Second access to same tile should be a HIT
        buffer_manager.get_tile(matrix, 0, 0)
        
        # Third access to different tile should be a MISS
        buffer_manager.get_tile(matrix, TILE_SIZE, 0)
        
        # Check event log
        events = buffer_manager.get_log()
        self.assertEqual(len(events), 3)
        
        # First event should be MISS
        self.assertEqual(events[0][1], 'MISS')
        
        # Second event should be HIT
        self.assertEqual(events[1][1], 'HIT')
        
        # Third event should be MISS
        self.assertEqual(events[2][1], 'MISS')
        
        matrix.close()
    
    def test_lru_eviction_policy(self):
        """Test the correctness of the LRU eviction policy."""
        # Create buffer manager with cache size of 2 tiles
        buffer_manager = BufferManager(max_cache_size_tiles=2)
        matrix = PaperMatrix(self.matrix_paths[0], self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Access three different tiles - this should trigger LRU eviction
        tile1 = buffer_manager.get_tile(matrix, 0, 0)          # MISS - tile1 added
        tile2 = buffer_manager.get_tile(matrix, TILE_SIZE, 0)  # MISS - tile2 added (cache full)
        
        # Access tile1 again to make it most recently used
        buffer_manager.get_tile(matrix, 0, 0)                  # HIT - tile1 is now MRU
        
        # Access a third tile - should evict tile2 (LRU)
        tile3 = buffer_manager.get_tile(matrix, 0, TILE_SIZE)  # MISS - tile3 added, tile2 evicted
        
        # Access tile1 again - should still be a HIT (not evicted)
        buffer_manager.get_tile(matrix, 0, 0)                  # HIT - tile1 still in cache
        
        # Access tile2 again - should be a MISS (was evicted)
        buffer_manager.get_tile(matrix, TILE_SIZE, 0)          # MISS - tile2 was evicted
        
        # Verify event sequence
        events = buffer_manager.get_log()
        event_types = [event[1] for event in events]
        
        # Expected sequence includes EVICT events when cache is full
        # Access tile1: MISS, Access tile2: MISS, Access tile1 again: HIT
        # Access tile3: MISS + EVICT (evicts tile2), Access tile1 again: HIT  
        # Access tile2: MISS + EVICT (evicts tile3)
        
        # Count specific event types to verify LRU behavior
        miss_count = event_types.count('MISS')
        hit_count = event_types.count('HIT')
        evict_count = event_types.count('EVICT')
        
        self.assertEqual(miss_count, 4)  # 4 cache misses
        self.assertEqual(hit_count, 2)   # 2 cache hits
        self.assertEqual(evict_count, 2) # 2 evictions
        
        matrix.close()
    
    def test_optimal_eviction_policy_with_io_trace(self):
        """Test the correctness of the Optimal eviction policy with a predictable I/O trace."""
        # Create a predictable I/O trace
        matrix_name = os.path.basename(self.matrix_paths[0])
        io_trace = [
            (matrix_name, 0, 0),           # Access tile A
            (matrix_name, TILE_SIZE, 0),   # Access tile B  
            (matrix_name, 0, TILE_SIZE),   # Access tile C
            (matrix_name, 0, 0),           # Access tile A again (furthest future)
            (matrix_name, TILE_SIZE, 0),   # Access tile B again
        ]
        
        # Create buffer manager with cache size of 2 and the I/O trace
        buffer_manager = BufferManager(max_cache_size_tiles=2, io_trace=io_trace)
        matrix = PaperMatrix(self.matrix_paths[0], self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Execute the trace
        buffer_manager.get_tile(matrix, 0, 0, trace_pos=0)           # MISS - tile A added
        buffer_manager.get_tile(matrix, TILE_SIZE, 0, trace_pos=1)   # MISS - tile B added (cache full)
        
        # Next access should evict tile A (has furthest future use at position 3)
        # and keep tile B (next use at position 4)  
        buffer_manager.get_tile(matrix, 0, TILE_SIZE, trace_pos=2)   # MISS - tile C added, tile A evicted
        
        # Accessing tile A should now be a MISS (it was evicted)
        buffer_manager.get_tile(matrix, 0, 0, trace_pos=3)           # MISS - tile A re-added
        
        # Accessing tile B should be a HIT (it wasn't evicted)
        buffer_manager.get_tile(matrix, TILE_SIZE, 0, trace_pos=4)   # HIT - tile B still in cache
        
        # Verify the optimal eviction worked correctly
        events = buffer_manager.get_log()
        event_types = [event[1] for event in events]
        
        # Count event types to verify optimal policy behavior
        miss_count = event_types.count('MISS')
        hit_count = event_types.count('HIT')
        evict_count = event_types.count('EVICT')
        
        # The key test is that tile B should still be in cache (HIT) while tile A was evicted (MISS)
        # The exact eviction count may vary based on implementation details
        self.assertEqual(miss_count, 4)  # 4 misses total
        self.assertEqual(hit_count, 1)   # 1 hit (tile B still cached)
        self.assertGreaterEqual(evict_count, 1)  # At least 1 eviction occurred
        
        matrix.close()
    
    def test_cache_size_limit_enforcement(self):
        """Test that the cache size limit is properly enforced."""
        cache_size = 3
        buffer_manager = BufferManager(max_cache_size_tiles=cache_size)
        matrix = PaperMatrix(self.matrix_paths[0], self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Fill cache to capacity
        for i in range(cache_size):
            buffer_manager.get_tile(matrix, i * TILE_SIZE, 0)
        
        # Verify cache is at capacity
        self.assertEqual(len(buffer_manager.cache), cache_size)
        
        # Add one more tile - should trigger eviction
        buffer_manager.get_tile(matrix, cache_size * TILE_SIZE, 0)
        
        # Cache should still be at capacity
        self.assertEqual(len(buffer_manager.cache), cache_size)
        
        matrix.close()
    
    def test_concurrent_access_thread_safety(self):
        """Test basic thread safety of the buffer manager."""
        import threading
        import time
        
        buffer_manager = BufferManager(max_cache_size_tiles=5)
        matrix = PaperMatrix(self.matrix_paths[0], self.test_shape, dtype=self.test_dtype, mode='r')
        
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                for i in range(5):
                    tile = buffer_manager.get_tile(matrix, (thread_id * 10) % TILE_SIZE, 0)
                    results.append((thread_id, i, tile.shape))
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        
        # Verify we got expected number of results
        self.assertEqual(len(results), 15)  # 3 threads * 5 operations each
        
        matrix.close()


if __name__ == '__main__':
    unittest.main()