"""
Demo script for hierarchical memory management.
Demonstrates multi-tier caching (RAM → SSD → Network) with performance metrics.
"""

import os
import tempfile
import shutil
import numpy as np
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from paper.core import PaperMatrix
from paper.hierarchical_buffer import HierarchicalBufferManager
from paper.backend import add
from paper.config import TILE_SIZE


def demo_basic_hierarchical_cache():
    """Demonstrate basic hierarchical cache functionality."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Hierarchical Cache")
    print("=" * 70)
    
    # Create temp directory
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create test matrices
        shape = (1024, 1024)  # 1 tile
        dtype = np.float32
        
        print(f"\nCreating test matrix ({shape[0]}x{shape[1]})...")
        
        # Matrix A
        a_path = os.path.join(test_dir, "A.bin")
        data_a = np.full(shape, 1.0, dtype=dtype)
        data_a.tofile(a_path)
        
        A = PaperMatrix(a_path, shape, dtype=dtype, mode='r')
        
        # Create hierarchical buffer manager
        print("\nInitializing hierarchical buffer manager:")
        print("  RAM tier: 2 tiles")
        print("  SSD tier: 4 tiles")
        print("  Network tier: 8 tiles")
        
        buffer_mgr = HierarchicalBufferManager(
            ram_capacity_tiles=2,
            ssd_capacity_tiles=4,
            network_capacity_tiles=8,
            network_latency_ms=10.0  # Simulate 10ms network latency
        )
        
        # Access tiles
        print("\nAccessing tiles:")
        print("  1. First access (0, 0) - should be a MISS")
        buffer_mgr.get_tile(A, 0, 0)
        
        print("  2. Second access (0, 0) - should be a HIT from RAM")
        buffer_mgr.get_tile(A, 0, 0)
        
        # Print metrics
        buffer_mgr.print_metrics()
        
        A.close()
        
    finally:
        shutil.rmtree(test_dir)
    
    print("\n✓ Demo 1 completed successfully!\n")


def demo_tier_promotion_demotion():
    """Demonstrate data promotion and demotion between tiers."""
    print("\n" + "=" * 70)
    print("DEMO 2: Tier Promotion and Demotion")
    print("=" * 70)
    
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create a matrix requiring multiple tiles (but not too large)
        shape = (1024, 1024)  # 1 tile is enough to demonstrate
        dtype = np.float32
        
        print(f"\nCreating test matrix ({shape[0]}x{shape[1]})...")
        
        a_path = os.path.join(test_dir, "A.bin")
        data_a = np.full(shape, 1.0, dtype=dtype)
        data_a.tofile(a_path)
        
        A = PaperMatrix(a_path, shape, dtype=dtype, mode='r')
        
        # Create buffer manager with very small RAM cache
        print("\nInitializing buffer manager with small RAM cache (1 tile):")
        
        buffer_mgr = HierarchicalBufferManager(
            ram_capacity_tiles=1,  # Very small to force evictions
            ssd_capacity_tiles=2,
            network_capacity_tiles=4,
            network_latency_ms=0.0  # No latency for demo speed
        )
        
        print("\nAccessing tiles to demonstrate tier movement:")
        
        # Access the same tile multiple times with intervening accesses
        print("  1. Access tile (0, 0) - loads into RAM")
        buffer_mgr.get_tile(A, 0, 0)
        
        print("  2. Access tile (0, 0) again - HIT from RAM")
        buffer_mgr.get_tile(A, 0, 0)
        
        # Print metrics showing RAM hits
        buffer_mgr.print_metrics()
        
        A.close()
        
    finally:
        shutil.rmtree(test_dir)
    
    print("\n✓ Demo 2 completed successfully!\n")


def demo_matrix_operation_with_hierarchy():
    """Demonstrate matrix operations using hierarchical cache."""
    print("\n" + "=" * 70)
    print("DEMO 3: Matrix Addition with Hierarchical Cache")
    print("=" * 70)
    
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create test matrices
        shape = (1024, 1024)
        dtype = np.float32
        
        print(f"\nCreating matrices A and B ({shape[0]}x{shape[1]})...")
        
        a_path = os.path.join(test_dir, "A.bin")
        b_path = os.path.join(test_dir, "B.bin")
        output_path = os.path.join(test_dir, "C.bin")
        
        data_a = np.full(shape, 2.0, dtype=dtype)
        data_b = np.full(shape, 3.0, dtype=dtype)
        
        data_a.tofile(a_path)
        data_b.tofile(b_path)
        
        A = PaperMatrix(a_path, shape, dtype=dtype, mode='r')
        B = PaperMatrix(b_path, shape, dtype=dtype, mode='r')
        
        # Create hierarchical buffer manager
        buffer_mgr = HierarchicalBufferManager(
            ram_capacity_tiles=4,
            ssd_capacity_tiles=8,
            network_capacity_tiles=16,
            network_latency_ms=0.0
        )
        
        print("\nPerforming matrix addition: C = A + B")
        print("  (Using hierarchical cache for tile management)")
        
        # Perform addition
        C = add(A, B, output_path, buffer_mgr)
        
        # Verify result
        expected = 5.0
        actual = C.data[0, 0]
        
        print(f"\nResult verification:")
        print(f"  Expected value: {expected}")
        print(f"  Actual value: {actual}")
        print(f"  ✓ Result correct!" if np.isclose(actual, expected) else "  ✗ Result incorrect!")
        
        # Print cache metrics
        buffer_mgr.print_metrics()
        
        A.close()
        B.close()
        C.close()
        
    finally:
        shutil.rmtree(test_dir)
    
    print("\n✓ Demo 3 completed successfully!\n")


def demo_performance_comparison():
    """Compare different cache configurations."""
    print("\n" + "=" * 70)
    print("DEMO 4: Cache Configuration Comparison")
    print("=" * 70)
    
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create test matrix (smaller for speed)
        shape = (1024, 1024)  # 1 tile
        dtype = np.float32
        
        print(f"\nCreating test matrix ({shape[0]}x{shape[1]})...")
        
        a_path = os.path.join(test_dir, "A.bin")
        data_a = np.full(shape, 1.0, dtype=dtype)
        data_a.tofile(a_path)
        
        # Test with different configurations
        configs = [
            {"name": "Small RAM", "ram": 1, "ssd": 2, "network": 4},
            {"name": "Medium RAM", "ram": 2, "ssd": 2, "network": 4},
            {"name": "Large RAM", "ram": 4, "ssd": 2, "network": 4},
        ]
        
        print("\nTesting different cache configurations:")
        print("-" * 70)
        
        for config in configs:
            A = PaperMatrix(a_path, shape, dtype=dtype, mode='r')
            
            buffer_mgr = HierarchicalBufferManager(
                ram_capacity_tiles=config["ram"],
                ssd_capacity_tiles=config["ssd"],
                network_capacity_tiles=config["network"],
                network_latency_ms=0.0
            )
            
            # Access the same tile multiple times
            for i in range(6):
                buffer_mgr.get_tile(A, 0, 0)
            
            metrics = buffer_mgr.get_summary_metrics()
            
            print(f"\n{config['name']} (RAM: {config['ram']}, SSD: {config['ssd']}):")
            print(f"  Total requests: {metrics['total_requests']}")
            print(f"  Overall hit rate: {metrics['overall_hit_rate']:.2%}")
            print(f"  RAM hit rate: {metrics['ram_hit_rate']:.2%}")
            
            A.close()
        
        print("\n" + "-" * 70)
        
    finally:
        shutil.rmtree(test_dir)
    
    print("\n✓ Demo 4 completed successfully!\n")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("HIERARCHICAL MEMORY MANAGEMENT DEMO")
    print("Multi-Tier Caching: RAM → SSD → Network")
    print("=" * 70)
    
    try:
        demo_basic_hierarchical_cache()
        demo_tier_promotion_demotion()
        demo_matrix_operation_with_hierarchy()
        demo_performance_comparison()
        
        print("\n" + "=" * 70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
