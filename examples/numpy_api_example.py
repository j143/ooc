#!/usr/bin/env python3
"""
NumPy-Compatible API Example

This example demonstrates how to use the Paper framework's NumPy-compatible API
for out-of-core matrix operations. The API provides a familiar NumPy interface
while handling datasets larger than memory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Paper's NumPy-compatible API
from paper import numpy_api as pnp
import numpy as np

def example_basic_operations():
    """Demonstrate basic array creation and operations."""
    print("=" * 60)
    print("Example 1: Basic Array Creation and Operations")
    print("=" * 60)
    
    # Create arrays from data (similar to NumPy)
    a = pnp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = pnp.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
    
    print(f"Array a: shape={a.shape}, dtype={a.dtype}")
    print(f"Array b: shape={b.shape}, dtype={b.dtype}")
    
    # Element-wise addition (lazy evaluation)
    c = a + b
    print(f"\nLazy addition c = a + b: {c}")
    print(f"Is lazy? {c._is_lazy}")
    
    # Compute the result
    result = c.compute()
    print(f"\nComputed result:")
    print(result.to_numpy())
    
    print()

def example_scalar_multiplication():
    """Demonstrate scalar multiplication."""
    print("=" * 60)
    print("Example 2: Scalar Multiplication")
    print("=" * 60)
    
    a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
    
    # Scalar multiplication (works both ways)
    c1 = a * 2
    c2 = 2 * a
    
    print("Array a:")
    print(a.to_numpy())
    
    print("\nResult of a * 2:")
    print(c1.compute().to_numpy())
    
    print("\nResult of 2 * a:")
    print(c2.compute().to_numpy())
    
    print()

def example_matrix_multiplication():
    """Demonstrate matrix multiplication."""
    print("=" * 60)
    print("Example 3: Matrix Multiplication")
    print("=" * 60)
    
    # Create matrices for multiplication
    a = pnp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # 2x3
    b = pnp.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)  # 3x2
    
    print(f"Matrix a: shape={a.shape}")
    print(a.to_numpy())
    
    print(f"\nMatrix b: shape={b.shape}")
    print(b.to_numpy())
    
    # Matrix multiplication using @ operator
    c = a @ b
    print(f"\nResult of a @ b: shape={c.shape}")
    print(c.compute().to_numpy())
    
    print()

def example_chained_operations():
    """Demonstrate chaining operations with lazy evaluation."""
    print("=" * 60)
    print("Example 4: Chained Operations (Lazy Evaluation)")
    print("=" * 60)
    
    a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
    b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
    
    # Complex expression: (a + b) * 2
    # This creates a computation plan without executing
    c = (a + b) * 2
    
    print("Expression: (a + b) * 2")
    print(f"Is lazy? {c._is_lazy}")
    print(f"Result shape: {c.shape}")
    
    # Execute the entire computation plan
    result = c.compute()
    print("\nComputed result:")
    print(result.to_numpy())
    
    # Note: The framework automatically optimizes this!
    # The fused kernel performs addition and scalar multiplication in one pass
    
    print()

def example_array_creation_functions():
    """Demonstrate various array creation functions."""
    print("=" * 60)
    print("Example 5: Array Creation Functions")
    print("=" * 60)
    
    # Zeros array
    zeros = pnp.zeros((3, 4))
    print("Zeros array (3x4):")
    print(zeros.to_numpy())
    
    # Ones array
    ones = pnp.ones((2, 3))
    print("\nOnes array (2x3):")
    print(ones.to_numpy())
    
    # Identity matrix
    identity = pnp.eye(4)
    print("\nIdentity matrix (4x4):")
    print(identity.to_numpy())
    
    # Random array
    random = pnp.random_rand((2, 2))
    print("\nRandom array (2x2) - values in [0, 1):")
    print(random.to_numpy())
    
    print()

def example_transpose():
    """Demonstrate transpose operation."""
    print("=" * 60)
    print("Example 6: Transpose")
    print("=" * 60)
    
    a = pnp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    
    print("Original array (2x3):")
    print(a.to_numpy())
    
    # Transpose using .T property
    a_t = a.T
    
    print(f"\nTransposed array (3x2):")
    print(a_t.to_numpy())
    
    print()

def example_file_operations():
    """Demonstrate saving and loading arrays."""
    print("=" * 60)
    print("Example 7: File I/O Operations")
    print("=" * 60)
    
    # Create temporary directory for demo
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Create and save an array
    a = pnp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    save_path = os.path.join(temp_dir, "my_array.bin")
    
    print(f"Saving array to: {save_path}")
    pnp.save(save_path, a)
    print("Array saved successfully!")
    
    # Load the array
    print(f"\nLoading array from: {save_path}")
    loaded = pnp.load(save_path, shape=(2, 3))
    print("Loaded array:")
    print(loaded.to_numpy())
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print("\nCleanup completed.")
    
    print()

def example_large_arrays():
    """Demonstrate out-of-core capabilities with larger arrays."""
    print("=" * 60)
    print("Example 8: Large Arrays (Out-of-Core)")
    print("=" * 60)
    
    print("Creating large random arrays (1000x1000)...")
    a = pnp.random_rand((1000, 1000))
    b = pnp.random_rand((1000, 1000))
    
    print(f"Array a: shape={a.shape}, size={a.size} elements")
    print(f"Array b: shape={b.shape}, size={b.size} elements")
    
    # Perform operations without loading entire arrays into memory
    print("\nPerforming lazy operations...")
    c = (a + b) * 0.5
    
    print(f"Result shape: {c.shape}")
    print("Note: The operations are lazy - no computation has happened yet!")
    
    print("\nComputing result (this actually executes the operations)...")
    result = c.compute()
    print(f"Result computed successfully! Shape: {result.shape}")
    print("The framework handled this efficiently using disk-backed matrices.")
    
    print()

def example_numpy_comparison():
    """Compare Paper API with NumPy syntax."""
    print("=" * 60)
    print("Example 9: NumPy Compatibility Comparison")
    print("=" * 60)
    
    print("NumPy syntax:")
    print("  import numpy as np")
    print("  a = np.array([[1, 2], [3, 4]])")
    print("  b = np.array([[5, 6], [7, 8]])")
    print("  c = (a + b) * 2")
    print()
    
    print("Paper's NumPy-compatible API syntax:")
    print("  from paper import numpy_api as pnp")
    print("  a = pnp.array([[1, 2], [3, 4]])")
    print("  b = pnp.array([[5, 6], [7, 8]])")
    print("  c = (a + b) * 2")
    print("  result = c.compute()  # Execute the lazy computation")
    print()
    
    print("The API is nearly identical, making migration easy!")
    print("Benefits:")
    print("  - Familiar NumPy interface")
    print("  - Handles datasets larger than memory")
    print("  - Automatic optimization (fusion, caching)")
    print("  - Lazy evaluation for efficiency")
    
    print()

def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Paper Framework - NumPy-Compatible API Examples")
    print("=" * 60 + "\n")
    
    example_basic_operations()
    example_scalar_multiplication()
    example_matrix_multiplication()
    example_chained_operations()
    example_array_creation_functions()
    example_transpose()
    example_file_operations()
    example_large_arrays()
    example_numpy_comparison()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
