"""
Integration Example: Using Paper with PyTorch

This example demonstrates how Paper can preprocess large datasets
before feeding them to PyTorch DataLoaders and models.
"""

import numpy as np
from paper import numpy_api as pnp

def demonstrate_pytorch_integration():
    """
    Shows how Paper works with PyTorch for large-scale data handling.
    """
    print("=" * 70)
    print("Paper + PyTorch Integration Example")
    print("=" * 70)
    
    # 1. Create a large dataset using Paper
    print("\n1. Creating large dataset with Paper (out-of-core friendly)...")
    n_samples = 5000
    n_features = 100
    
    # In production, this would be: pnp.load('huge_features.bin', shape=...)
    np_data = np.random.randn(n_samples, n_features).astype(np.float32)
    paper_data = pnp.array(np_data)
    print(f"   Created Paper array: shape={paper_data.shape}")
    
    # 2. Preprocess with Paper (out-of-core operations)
    print("\n2. Preprocessing with Paper (I/O optimized)...")
    # Example: normalize the data
    scaled_data = paper_data * 0.01  # Lazy operation
    print(f"   Built lazy preprocessing plan: {scaled_data}")
    
    # Compute the preprocessed data
    preprocessed = scaled_data.compute()
    print(f"   ✓ Preprocessing complete: {preprocessed.shape}")
    
    # 3. Feed to PyTorch
    print("\n3. Converting to PyTorch tensors...")
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Convert Paper array to NumPy, then to PyTorch tensor
        X = preprocessed.to_numpy()
        y = np.random.randint(0, 2, size=n_samples)  # Dummy labels
        
        # Create PyTorch tensors
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y).long()
        
        print(f"   ✓ Created tensors: X={X_tensor.shape}, y={y_tensor.shape}")
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        print(f"   ✓ Created DataLoader with batch_size=32")
        print(f"   ✓ Number of batches: {len(loader)}")
        
        # Simulate training loop
        print("\n4. Simulating training loop...")
        batch = next(iter(loader))
        X_batch, y_batch = batch
        print(f"   First batch: X={X_batch.shape}, y={y_batch.shape}")
        print("   ✓ Ready for model.forward(X_batch)")
        
    except ImportError:
        print("   (PyTorch not installed, skipping PyTorch steps)")
    
    # 5. The workflow
    print("\n" + "=" * 70)
    print("WORKFLOW SUMMARY:")
    print("=" * 70)
    print("""
Step 1: Paper loads and preprocesses data (out-of-core)
        ↓
Step 2: Convert to PyTorch tensors (only when needed)
        ↓
Step 3: Feed to DataLoader
        ↓
Step 4: Train your model as usual

Key Benefits:
  ✓ Paper handles data that doesn't fit in RAM
  ✓ PyTorch handles model training
  ✓ No need to rewrite existing PyTorch code
  ✓ Seamless integration between both frameworks
""")

def demonstrate_batch_prediction():
    """
    Shows how Paper can handle batch prediction on large datasets.
    """
    print("\n" + "=" * 70)
    print("Use Case: Batch Prediction on Huge Datasets")
    print("=" * 70)
    
    print("\nScenario: Making predictions on 1M samples...")
    print("(Demo uses smaller size for speed)")
    
    n_samples = 10000  # In production: 1,000,000+
    n_features = 50
    
    # Load test data with Paper
    test_data = pnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    print(f"Test data loaded: {test_data.shape}")
    
    print("\nPaper's role:")
    print("  • Efficiently loads data in tiles")
    print("  • Minimizes memory footprint")
    print("  • Handles preprocessing at scale")
    
    print("\nPyTorch/Model's role:")
    print("  • Processes batches through trained model")
    print("  • Generates predictions")
    
    # Simulate preprocessing
    preprocessed = (test_data * 2.0).compute()
    print(f"\n✓ Preprocessed {preprocessed.shape[0]:,} samples")
    print("✓ Ready for batch prediction")
    print("  (In production: model.predict(preprocessed.to_numpy()))")

if __name__ == "__main__":
    demonstrate_pytorch_integration()
    demonstrate_batch_prediction()
    
    print("\n" + "=" * 70)
    print("Integration Complete!")
    print("=" * 70)
