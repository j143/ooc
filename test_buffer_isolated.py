#!/usr/bin/env python
"""
Test to isolate buffer manager and segfault issues.
"""

import sys
import os
import numpy as np
import time

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.core import PaperMatrix
from paper.config import TILE_SIZE
from paper.buffer import BufferManager

def test_buffer_manager_basic():
    """Test basic buffer manager functionality"""
    print("=== Testing Buffer Manager Basic Functionality ===")
    
    # Create a simple test matrix
    test_path = "/tmp/test_buffer.bin"
    shape = (512, 512)
    
    # Create matrix
    matrix = PaperMatrix(test_path, shape, mode='w+')
    test_data = np.ones((512, 512), dtype=np.float32)
    matrix.data[:, :] = test_data
    matrix.data.flush()
    matrix.close()
    
    # Test buffer manager
    matrix = PaperMatrix(test_path, shape, mode='r')
    buffer_mgr = BufferManager(max_cache_size_tiles=4)
    
    # Test cache hit/miss
    tile1 = buffer_mgr.get_tile(matrix, 0, 0)
    print(f"First access - tile shape: {tile1.shape}")
    
    tile2 = buffer_mgr.get_tile(matrix, 0, 0)  # Should be cache hit
    print(f"Second access - same tile: {np.array_equal(tile1, tile2)}")
    
    # Test cache eviction
    for i in range(6):  # Force evictions
        tile = buffer_mgr.get_tile(matrix, i * 100, i * 100)
        print(f"Access {i}: tile shape {tile.shape}")
    
    event_log = buffer_mgr.get_log()
    print(f"Events logged: {len(event_log)}")
    
    matrix.close()
    
    # Cleanup
    os.remove(test_path)
    print("✓ Buffer manager test completed")

def test_with_trace():
    """Test buffer manager with IO trace"""
    print("\n=== Testing Buffer Manager with IO Trace ===")
    
    test_path = "/tmp/test_trace.bin"
    shape = (512, 512)
    
    # Create matrix
    matrix = PaperMatrix(test_path, shape, mode='w+')
    test_data = np.ones((512, 512), dtype=np.float32)
    matrix.data[:, :] = test_data
    matrix.data.flush()
    matrix.close()
    
    # Create a simple trace
    trace = [
        ("test_trace.bin", 0, 0),
        ("test_trace.bin", 0, 1024),
        ("test_trace.bin", 0, 0),  # Should be cache hit
    ]
    
    matrix = PaperMatrix(test_path, shape, mode='r')
    buffer_mgr = BufferManager(max_cache_size_tiles=2, io_trace=trace)
    
    # Access according to trace
    tile1 = buffer_mgr.get_tile(matrix, 0, 0)
    tile2 = buffer_mgr.get_tile(matrix, 0, 1024)  # This should be limited by shape
    tile3 = buffer_mgr.get_tile(matrix, 0, 0)
    
    event_log = buffer_mgr.get_log()
    print(f"Events with trace: {len(event_log)}")
    for event in event_log:
        print(f"  {event}")
    
    matrix.close()
    
    # Cleanup
    os.remove(test_path)
    print("✓ Buffer manager with trace test completed")

if __name__ == "__main__":
    test_buffer_manager_basic()
    test_with_trace()
    print("\nAll buffer manager tests completed successfully!")