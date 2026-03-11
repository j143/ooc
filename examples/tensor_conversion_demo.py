"""
Demonstration of Direct Conversion to Device Tensors

This example demonstrates how to convert Paper's out-of-core arrays
to PyTorch and TensorFlow tensors with efficient memory handling.

Features demonstrated:
- Loading large arrays from disk
- Building lazy computation graphs
- Converting to PyTorch tensors (CPU and GPU)
- Converting to TensorFlow tensors
- Memory-efficient operations
"""

import numpy as np
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper import numpy_api as pnp


def demo_basic_conversion():
    """Demonstrate basic tensor conversion."""
    print("=" * 70)
    print("DEMO 1: Basic Tensor Conversion")
    print("=" * 70)
    
    # Create a simple array
    print("\n1. Creating a Paper array...")
    arr = pnp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(f"   Array shape: {arr.shape}")
    print(f"   Array dtype: {arr.dtype}")
    
    # Convert to PyTorch (if available)
    try:
        import torch
        print("\n2. Converting to PyTorch tensor...")
        torch_tensor = arr.to_torch()
        print(f"   PyTorch tensor shape: {torch_tensor.shape}")
        print(f"   PyTorch tensor dtype: {torch_tensor.dtype}")
        print(f"   PyTorch tensor device: {torch_tensor.device}")
        print(f"   Tensor data:\n{torch_tensor}")
    except ImportError:
        print("\n2. PyTorch not available (skipping PyTorch conversion)")
    
    # Convert to TensorFlow (if available)
    try:
        import tensorflow as tf
        print("\n3. Converting to TensorFlow tensor...")
        tf_tensor = arr.to_tensorflow()
        print(f"   TensorFlow tensor shape: {tf_tensor.shape}")
        print(f"   TensorFlow tensor dtype: {tf_tensor.dtype}")
        print(f"   Tensor data:\n{tf_tensor.numpy()}")
    except ImportError:
        print("\n3. TensorFlow not available (skipping TensorFlow conversion)")


def demo_lazy_computation():
    """Demonstrate tensor conversion with lazy computation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Lazy Computation with Tensor Conversion")
    print("=" * 70)
    
    # Create arrays
    print("\n1. Creating Paper arrays...")
    a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
    b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
    print(f"   Array A shape: {a.shape}")
    print(f"   Array B shape: {b.shape}")
    
    # Build lazy computation
    print("\n2. Building lazy computation: (A + B) * 2")
    c = (a + b) * 2
    print(f"   Result is lazy: {c._is_lazy}")
    
    # Execute computation
    print("\n3. Computing result...")
    result = c.compute()
    print(f"   Result computed, is lazy: {result._is_lazy}")
    
    # Convert to tensors
    try:
        import torch
        print("\n4. Converting to PyTorch tensor...")
        torch_tensor = result.to_torch()
        print(f"   PyTorch result:\n{torch_tensor}")
    except ImportError:
        print("\n4. PyTorch not available")
    
    try:
        import tensorflow as tf
        print("\n5. Converting to TensorFlow tensor...")
        tf_tensor = result.to_tensorflow()
        print(f"   TensorFlow result:\n{tf_tensor.numpy()}")
    except ImportError:
        print("\n5. TensorFlow not available")


def demo_out_of_core_workflow():
    """Demonstrate the complete out-of-core workflow from the issue."""
    print("\n" + "=" * 70)
    print("DEMO 3: Out-of-Core Array to Device Tensor Workflow")
    print("=" * 70)
    
    # Create a temporary directory for our data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a larger dataset
        print("\n1. Creating a large array on disk (1000x1000)...")
        large_matrix_path = os.path.join(temp_dir, "large_matrix.dat")
        test_data = np.random.rand(1000, 1000).astype(np.float32)
        test_data.tofile(large_matrix_path)
        print(f"   Created file: {large_matrix_path}")
        print(f"   File size: {os.path.getsize(large_matrix_path) / (1024*1024):.2f} MB")
        
        # Load as out-of-core array (memory-mapped)
        print("\n2. Loading array with memory mapping (no data loaded yet)...")
        arr = pnp.load(large_matrix_path, shape=(1000, 1000), dtype=np.float32)
        print(f"   Array shape: {arr.shape}")
        print(f"   Memory-mapped: Yes")
        
        # Build computation graph
        print("\n3. Building lazy computation graph: arr * 2")
        c = arr * 2
        print(f"   Computation graph built (lazy: {c._is_lazy})")
        
        # Execute computation
        print("\n4. Executing computation plan...")
        result = c.compute()
        print(f"   Computation complete")
        
        # Convert to device tensors
        try:
            import torch
            print("\n5. Converting to PyTorch tensor (efficient conversion)...")
            torch_tensor = result.to_torch()
            print(f"   PyTorch tensor created")
            print(f"   Shape: {torch_tensor.shape}")
            print(f"   Device: {torch_tensor.device}")
            print(f"   Sample values (first 3x3):\n{torch_tensor[:3, :3]}")
            
            # Verify computation correctness
            expected_sample = test_data[:3, :3] * 2
            actual_sample = torch_tensor[:3, :3].numpy()
            matches = np.allclose(actual_sample, expected_sample)
            print(f"   Computation verified: {'✓' if matches else '✗'}")
            
        except ImportError:
            print("\n5. PyTorch not available")
        
        try:
            import tensorflow as tf
            print("\n6. Converting to TensorFlow tensor (efficient conversion)...")
            tf_tensor = result.to_tensorflow()
            print(f"   TensorFlow tensor created")
            print(f"   Shape: {tf_tensor.shape}")
            print(f"   Sample values (first 3x3):\n{tf_tensor[:3, :3].numpy()}")
            
            # Verify computation correctness
            expected_sample = test_data[:3, :3] * 2
            actual_sample = tf_tensor[:3, :3].numpy()
            matches = np.allclose(actual_sample, expected_sample)
            print(f"   Computation verified: {'✓' if matches else '✗'}")
            
        except ImportError:
            print("\n6. TensorFlow not available")


def demo_gpu_conversion():
    """Demonstrate GPU tensor conversion if CUDA is available."""
    print("\n" + "=" * 70)
    print("DEMO 4: GPU Tensor Conversion")
    print("=" * 70)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("\nCUDA is not available. GPU conversion demo skipped.")
            return
        
        print("\n1. Creating Paper array...")
        arr = pnp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        print("\n2. Converting to CUDA tensor...")
        cuda_tensor = arr.to_torch(device='cuda')
        print(f"   CUDA tensor created")
        print(f"   Shape: {cuda_tensor.shape}")
        print(f"   Device: {cuda_tensor.device}")
        print(f"   Tensor data:\n{cuda_tensor}")
        
        print("\n3. Performing GPU computation...")
        result = cuda_tensor * 3 + 1
        print(f"   Result on GPU:\n{result}")
        
    except ImportError:
        print("\nPyTorch not available. GPU conversion demo skipped.")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("Paper Framework: Out-of-Core Arrays to Device Tensors")
    print("=" * 70)
    
    # Check which frameworks are available
    frameworks = []
    try:
        import torch
        frameworks.append(f"PyTorch {torch.__version__}")
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        frameworks.append(f"TensorFlow {tf.__version__}")
    except ImportError:
        pass
    
    if frameworks:
        print(f"\nAvailable frameworks: {', '.join(frameworks)}")
    else:
        print("\nNo deep learning frameworks detected.")
        print("Install PyTorch: pip install torch")
        print("Install TensorFlow: pip install tensorflow")
    
    # Run demonstrations
    demo_basic_conversion()
    demo_lazy_computation()
    demo_out_of_core_workflow()
    demo_gpu_conversion()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Benefits:")
    print("  • Efficient memory usage with memory-mapped arrays")
    print("  • Zero-copy conversion where possible")
    print("  • Seamless integration with PyTorch and TensorFlow")
    print("  • Support for both CPU and GPU devices")
    print("  • Lazy evaluation for optimized computation")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
