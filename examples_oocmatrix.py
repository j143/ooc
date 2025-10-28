"""
Example demonstrating the OOCMatrix API for out-of-core operations.

This example shows how to use the high-level OOCMatrix wrapper to perform
operations on large matrices that don't fit in memory, while leveraging
existing NumPy/SciPy operations for in-block computations.
"""

import numpy as np
import os
import tempfile
from paper import OOCMatrix
from paper.core import PaperMatrix
from paper.config import TILE_SIZE


def create_large_matrix(filepath, shape, value_func=None):
    """
    Create a large matrix file with specified values.
    
    Args:
        filepath: Path where to save the matrix
        shape: Shape of the matrix (rows, cols)
        value_func: Optional function(i, j) -> value for element at (i, j)
    """
    print(f"Creating matrix at '{filepath}' with shape {shape}...")
    
    # Create the matrix using PaperMatrix for initialization
    matrix = PaperMatrix(filepath, shape, mode='w+')
    
    # Fill with data tile by tile
    for r_start in range(0, shape[0], TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, shape[0])
        for c_start in range(0, shape[1], TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, shape[1])
            
            if value_func is None:
                # Random data
                tile = np.random.rand(r_end - r_start, c_end - c_start).astype(np.float32)
            else:
                # Custom values
                tile = np.zeros((r_end - r_start, c_end - c_start), dtype=np.float32)
                for i in range(tile.shape[0]):
                    for j in range(tile.shape[1]):
                        tile[i, j] = value_func(r_start + i, c_start + j)
            
            matrix.data[r_start:r_end, c_start:c_end] = tile
    
    matrix.data.flush()
    matrix.close()
    print(f"Matrix created successfully.")


def example_basic_operations():
    """Demonstrate basic operations with OOCMatrix."""
    print("\n" + "="*60)
    print("Example 1: Basic Operations")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test matrices
        path_a = os.path.join(tmpdir, "A.bin")
        path_b = os.path.join(tmpdir, "B.bin")
        
        shape = (1000, 500)
        create_large_matrix(path_a, shape)
        create_large_matrix(path_b, shape)
        
        # Load matrices using OOCMatrix
        A = OOCMatrix(path_a, shape, dtype=np.float32, mode='r')
        B = OOCMatrix(path_b, shape, dtype=np.float32, mode='r')
        
        print(f"\nMatrix A: {A}")
        print(f"Matrix B: {B}")
        
        # Compute statistics without loading entire matrix
        print(f"\nA.sum() = {A.sum():.2f}")
        print(f"A.mean() = {A.mean():.6f}")
        print(f"A.std() = {A.std():.6f}")
        print(f"A.min() = {A.min():.6f}")
        print(f"A.max() = {A.max():.6f}")
        
        A.close()
        B.close()


def example_blockwise_operations():
    """Demonstrate blockwise operations with custom functions."""
    print("\n" + "="*60)
    print("Example 2: Blockwise Operations")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a matrix
        path = os.path.join(tmpdir, "data.bin")
        shape = (800, 400)
        create_large_matrix(path, shape)
        
        # Load as OOCMatrix
        A = OOCMatrix(path, shape, dtype=np.float32, mode='r')
        
        print(f"Original matrix: {A}")
        
        # Example 1: Apply normalization using NumPy operations
        print("\n1. Normalization using blockwise_apply:")
        mean_val = A.mean()
        std_val = A.std()
        
        normalized_path = os.path.join(tmpdir, "normalized.bin")
        A_normalized = A.blockwise_apply(
            lambda x: (x - mean_val) / std_val,
            output_path=normalized_path
        )
        
        print(f"   Normalized matrix mean: {A_normalized.mean():.6f} (should be ~0)")
        print(f"   Normalized matrix std: {A_normalized.std():.6f} (should be ~1)")
        
        # Example 2: Apply custom function to each block
        print("\n2. Apply ReLU activation:")
        relu_path = os.path.join(tmpdir, "relu.bin")
        A_relu = A.blockwise_apply(
            lambda x: np.maximum(0, x - 0.5),
            output_path=relu_path
        )
        print(f"   ReLU applied successfully")
        
        A.close()
        A_normalized.close()
        A_relu.close()


def example_lazy_evaluation():
    """Demonstrate lazy evaluation and operator fusion."""
    print("\n" + "="*60)
    print("Example 3: Lazy Evaluation & Operator Fusion")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create matrices
        path_a = os.path.join(tmpdir, "A.bin")
        path_b = os.path.join(tmpdir, "B.bin")
        
        shape = (600, 400)
        create_large_matrix(path_a, shape, value_func=lambda i, j: 2.0)
        create_large_matrix(path_b, shape, value_func=lambda i, j: 3.0)
        
        A = OOCMatrix(path_a, shape, dtype=np.float32, mode='r')
        B = OOCMatrix(path_b, shape, dtype=np.float32, mode='r')
        
        # Build lazy expression: (A + B) * 2
        # This doesn't execute yet!
        print("\nBuilding lazy expression: (A + B) * 2")
        lazy_result = (A + B) * 2
        
        print(f"Lazy plan created: {lazy_result._plan}")
        
        # Now execute the plan
        print("\nExecuting the computation plan...")
        result_path = os.path.join(tmpdir, "result.bin")
        result = lazy_result.compute(result_path)
        
        print(f"Result computed: {result}")
        print(f"Expected value: (2 + 3) * 2 = 10")
        print(f"Actual mean: {result.mean():.2f}")
        
        A.close()
        B.close()
        result.close()


def example_matrix_multiplication():
    """Demonstrate matrix multiplication with custom operations."""
    print("\n" + "="*60)
    print("Example 4: Matrix Multiplication")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create matrices for multiplication
        path_a = os.path.join(tmpdir, "A.bin")
        path_b = os.path.join(tmpdir, "B.bin")
        
        shape_a = (500, 300)
        shape_b = (300, 200)
        
        create_large_matrix(path_a, shape_a)
        create_large_matrix(path_b, shape_b)
        
        A = OOCMatrix(path_a, shape_a, dtype=np.float32, mode='r')
        B = OOCMatrix(path_b, shape_b, dtype=np.float32, mode='r')
        
        print(f"\nA shape: {A.shape}")
        print(f"B shape: {B.shape}")
        
        # Method 1: Using matmul with custom operation (uses np.dot by default)
        print("\nMethod 1: Using matmul() with custom operation")
        result_path = os.path.join(tmpdir, "result_matmul.bin")
        C = A.matmul(B, op=np.dot, output_path=result_path)
        
        print(f"Result shape: {C.shape}")
        print(f"Result mean: {C.mean():.6f}")
        
        # Method 2: Using @ operator (lazy evaluation)
        print("\nMethod 2: Using @ operator (lazy)")
        lazy_product = A @ B
        
        result_path2 = os.path.join(tmpdir, "result_lazy.bin")
        C2 = lazy_product.compute(result_path2)
        
        print(f"Result shape: {C2.shape}")
        print(f"Result mean: {C2.mean():.6f}")
        
        A.close()
        B.close()
        C.close()
        C2.close()


def example_block_iteration():
    """Demonstrate iterating over matrix blocks."""
    print("\n" + "="*60)
    print("Example 5: Block Iteration")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a matrix
        path = os.path.join(tmpdir, "data.bin")
        shape = (400, 300)
        create_large_matrix(path, shape, value_func=lambda i, j: i + j)
        
        A = OOCMatrix(path, shape, dtype=np.float32, mode='r')
        
        print(f"\nIterating over blocks of {A}:")
        print(f"Block size: {TILE_SIZE}x{TILE_SIZE}")
        
        block_count = 0
        total_sum = 0.0
        
        # Process each block
        for block, (r_start, c_start) in A.iterate_blocks():
            block_count += 1
            block_sum = np.sum(block)
            total_sum += block_sum
            
            if block_count <= 3:  # Show first 3 blocks
                print(f"  Block {block_count}: position=({r_start}, {c_start}), "
                      f"shape={block.shape}, sum={block_sum:.2f}")
        
        print(f"\nTotal blocks processed: {block_count}")
        print(f"Total sum (via iteration): {total_sum:.2f}")
        print(f"Total sum (via A.sum()): {A.sum():.2f}")
        
        A.close()


def example_as_in_issue():
    """
    Reproduce the example from the GitHub issue.
    
    This demonstrates the API as specified in the issue description.
    """
    print("\n" + "="*60)
    print("Example 6: API from GitHub Issue")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create matrices (smaller for demo purposes)
        path_a = os.path.join(tmpdir, "fileA.bin")
        path_b = os.path.join(tmpdir, "fileB.bin")
        
        shape_a = (10000, 1000)  # Scaled down from 10M for demo
        shape_b = (1000, 1000)
        
        print("\nCreating large matrices...")
        create_large_matrix(path_a, shape_a)
        create_large_matrix(path_b, shape_b)
        
        # User applies existing NumPy/SciPy operations on each block automatically
        A = OOCMatrix(path_a, shape=shape_a, dtype=np.float32, mode='r')
        B = OOCMatrix(path_b, shape=shape_b, dtype=np.float32, mode='r')
        
        print(f"\nA: {A}")
        print(f"B: {B}")
        
        # This triggers block-wise multiplication, but each block operation is plain NumPy
        def matmul_blocks(A_block, B_block):
            return np.dot(A_block, B_block)
        
        # API exposes big operations -- no kernel rewrite required!
        result_path = os.path.join(tmpdir, "result.bin")
        C = A.matmul(B, op=matmul_blocks, output_path=result_path)
        
        print(f"\nResult C = A @ B: {C}")
        
        # Downstream systems can consume each result block
        print("\nProcessing result blocks:")
        block_count = 0
        for block, idx in C.iterate_blocks():
            # downstream systems can consume each result block
            block_count += 1
            if block_count <= 2:
                print(f"  Processing block at {idx}, shape: {block.shape}")
        
        print(f"Total result blocks: {block_count}")
        
        # Other ops, e.g., sum, mean, normalization:
        print("\nComputing statistics:")
        mean = A.mean()
        std_val = A.std()
        
        print(f"  Mean: {mean:.6f}")
        print(f"  Std: {std_val:.6f}")
        
        # Normalization using blockwise_apply
        norm_path = os.path.join(tmpdir, "normalized.bin")
        A_normalized = A.blockwise_apply(lambda x: (x - mean) / std_val, output_path=norm_path)
        
        print(f"\nNormalized A: {A_normalized}")
        print(f"  Normalized mean: {A_normalized.mean():.6f}")
        
        A.close()
        B.close()
        C.close()
        A_normalized.close()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  OOCMatrix API Examples - Out-of-Core Matrix Operations")
    print("="*70)
    print("\nThis demonstrates the high-level API for out-of-core operations,")
    print("wrapping NumPy/SciPy operations with lazy evaluation and block-wise")
    print("processing, without reimplementing mathematical operations.")
    
    example_basic_operations()
    example_blockwise_operations()
    example_lazy_evaluation()
    example_matrix_multiplication()
    example_block_iteration()
    example_as_in_issue()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
