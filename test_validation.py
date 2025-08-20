#!/usr/bin/env python
"""
Validation tests for the consistency fixes made to the Paper matrix library.
This focuses on validating that TILE_SIZE consistency and buffer manager logic work correctly.
"""

import sys
import os
import numpy as np
import time

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.core import PaperMatrix
from paper.config import TILE_SIZE, DEFAULT_CACHE_SIZE_TILES
from paper.buffer import BufferManager
from paper import Plan, EagerNode

def test_tile_size_consistency():
    """Test that TILE_SIZE is consistent across all modules"""
    print("=== Testing TILE_SIZE Consistency ===")
    
    # Import TILE_SIZE from different modules to ensure consistency
    from paper.config import TILE_SIZE as config_tile_size
    from paper.core import TILE_SIZE as core_tile_size  
    from paper.buffer import TILE_SIZE as buffer_tile_size
    from paper.optimizer import TILE_SIZE as optimizer_tile_size
    
    sizes = [config_tile_size, core_tile_size, buffer_tile_size, optimizer_tile_size]
    
    # Check all sizes are equal
    if all(size == config_tile_size for size in sizes):
        print(f"✓ TILE_SIZE consistent across modules: {config_tile_size}")
    else:
        print(f"✗ TILE_SIZE inconsistent: {sizes}")
        return False
    
    # Check specific value
    if config_tile_size == 1024:
        print("✓ TILE_SIZE has expected value: 1024")
    else:
        print(f"✗ TILE_SIZE has unexpected value: {config_tile_size}")
        return False
        
    return True

def test_cache_size_consistency():
    """Test that cache size configuration is consistent"""
    print("\n=== Testing Cache Size Consistency ===")
    
    from paper.config import DEFAULT_CACHE_SIZE_TILES
    
    # Test buffer manager uses the default
    buffer_mgr = BufferManager()
    if buffer_mgr.max_size == DEFAULT_CACHE_SIZE_TILES:
        print(f"✓ BufferManager uses centralized default: {DEFAULT_CACHE_SIZE_TILES}")
    else:
        print(f"✗ BufferManager has different default: {buffer_mgr.max_size}")
        return False
        
    # Test custom size works
    custom_size = 32
    buffer_mgr_custom = BufferManager(max_cache_size_tiles=custom_size)
    if buffer_mgr_custom.max_size == custom_size:
        print(f"✓ BufferManager accepts custom size: {custom_size}")
    else:
        print(f"✗ BufferManager custom size failed: {buffer_mgr_custom.max_size}")
        return False
        
    return True

def test_buffer_manager_eviction_logic():
    """Test that buffer manager eviction logic works correctly"""
    print("\n=== Testing Buffer Manager Eviction Logic ===")
    
    test_path = "/tmp/eviction_test.bin"
    shape = (512, 512)
    
    # Create test matrix
    matrix = PaperMatrix(test_path, shape, mode='w+')
    test_data = np.random.rand(512, 512).astype(np.float32)
    matrix.data[:, :] = test_data
    matrix.data.flush()
    matrix.close()
    
    matrix = PaperMatrix(test_path, shape, mode='r')
    
    # Test LRU eviction
    buffer_mgr = BufferManager(max_cache_size_tiles=2)  # Small cache
    
    # Add 3 tiles to force eviction
    tile1 = buffer_mgr.get_tile(matrix, 0, 0)
    tile2 = buffer_mgr.get_tile(matrix, 100, 100)
    tile3 = buffer_mgr.get_tile(matrix, 200, 200)  # Should evict tile1
    
    # Check cache size doesn't exceed max
    if len(buffer_mgr.cache) <= buffer_mgr.max_size:
        print(f"✓ Cache size respects limit: {len(buffer_mgr.cache)}/{buffer_mgr.max_size}")
    else:
        print(f"✗ Cache size exceeded limit: {len(buffer_mgr.cache)}/{buffer_mgr.max_size}")
        matrix.close()
        os.remove(test_path)
        return False
    
    # Check event log has eviction events
    events = buffer_mgr.get_log()
    evict_events = [e for e in events if e[1] == 'EVICT']
    if len(evict_events) > 0:
        print(f"✓ Eviction events logged: {len(evict_events)}")
    else:
        print("✗ No eviction events logged")
        matrix.close()
        os.remove(test_path)
        return False
    
    matrix.close()
    os.remove(test_path)
    return True

def test_trace_bounds_checking():
    """Test that trace position bounds checking works"""
    print("\n=== Testing Trace Bounds Checking ===")
    
    test_path = "/tmp/trace_bounds_test.bin"
    shape = (256, 256)
    
    # Create test matrix
    matrix = PaperMatrix(test_path, shape, mode='w+')
    test_data = np.ones((256, 256), dtype=np.float32)
    matrix.data[:, :] = test_data
    matrix.data.flush()
    matrix.close()
    
    # Create a short trace
    trace = [
        ("trace_bounds_test.bin", 0, 0),
        ("trace_bounds_test.bin", 100, 100),
    ]
    
    matrix = PaperMatrix(test_path, shape, mode='r')
    buffer_mgr = BufferManager(max_cache_size_tiles=4, io_trace=trace)
    
    # Access more tiles than trace length
    tile1 = buffer_mgr.get_tile(matrix, 0, 0)
    tile2 = buffer_mgr.get_tile(matrix, 100, 100)
    tile3 = buffer_mgr.get_tile(matrix, 200, 200)  # Beyond trace
    
    # Should not crash and trace_pos should not exceed trace length
    if buffer_mgr.trace_pos <= len(trace):
        print(f"✓ Trace position within bounds: {buffer_mgr.trace_pos}/{len(trace)}")
    else:
        print(f"✗ Trace position out of bounds: {buffer_mgr.trace_pos}/{len(trace)}")
        matrix.close()
        os.remove(test_path)
        return False
    
    matrix.close()
    os.remove(test_path)
    return True

def test_matrix_operations_with_fixes():
    """Test that basic matrix operations still work with our fixes"""
    print("\n=== Testing Matrix Operations with Fixes ===")
    
    # Create test matrices
    test_dir = "/tmp/validation_test"
    os.makedirs(test_dir, exist_ok=True)
    
    path_A = os.path.join(test_dir, "A.bin")
    path_B = os.path.join(test_dir, "B.bin")
    result_path = os.path.join(test_dir, "result.bin")
    
    shape = (512, 512)
    
    # Create matrices
    for path, fill_value in [(path_A, 1.0), (path_B, 2.0)]:
        matrix = PaperMatrix(path, shape, mode='w+')
        for r in range(0, shape[0], TILE_SIZE):
            r_end = min(r + TILE_SIZE, shape[0])
            for c in range(0, shape[1], TILE_SIZE):
                c_end = min(c + TILE_SIZE, shape[1])
                tile_data = np.full((r_end - r, c_end - c), fill_value, dtype=np.float32)
                matrix.data[r:r_end, c:c_end] = tile_data
        matrix.data.flush()
        matrix.close()
    
    # Test addition operation
    A = PaperMatrix(path_A, shape, mode='r')
    B = PaperMatrix(path_B, shape, mode='r')
    
    A_lazy = Plan(EagerNode(A))
    B_lazy = Plan(EagerNode(B))
    
    plan = A_lazy + B_lazy
    result_matrix, buffer_mgr = plan.compute(result_path)
    
    # Verify result
    sample = result_matrix.data[0:5, 0:5]
    expected = 3.0  # 1.0 + 2.0
    
    if np.allclose(sample, expected):
        print(f"✓ Matrix addition produces correct result: {np.mean(sample)}")
        success = True
    else:
        print(f"✗ Matrix addition incorrect: expected {expected}, got {np.mean(sample)}")
        success = False
    
    # Clean up
    A.close()
    B.close()
    result_matrix.close()
    
    # Clean up files
    import shutil
    shutil.rmtree(test_dir)
    
    return success

def run_validation_tests():
    """Run all validation tests"""
    print("Running validation tests for consistency fixes...\n")
    
    tests = [
        test_tile_size_consistency,
        test_cache_size_consistency,
        test_buffer_manager_eviction_logic,
        test_trace_bounds_checking,
        test_matrix_operations_with_fixes,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All validation tests passed!")
        print("The consistency fixes are working correctly.")
    else:
        print("❌ Some validation tests failed.")
        print("Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)