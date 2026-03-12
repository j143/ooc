"""
Integration Example: Using Paper with scikit-learn

This example demonstrates how Paper transparently optimizes I/O 
for preprocessing large datasets before feeding them to sklearn pipelines.
"""

import numpy as np
from paper import numpy_api as pnp
import tempfile
import os

def demonstrate_sklearn_integration():
    """
    Shows how Paper works with sklearn for large-scale preprocessing.
    """
    print("=" * 70)
    print("Paper + scikit-learn Integration Example")
    print("=" * 70)
    
    # 1. Create a large dataset using Paper (out-of-core friendly)
    print("\n1. Creating large dataset with Paper...")
    n_samples = 10000
    n_features = 50
    
    # Generate data using NumPy (in practice, this would be loaded from disk)
    np_data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Convert to Paper array for out-of-core operations
    paper_data = pnp.array(np_data)
    print(f"   Created Paper array: shape={paper_data.shape}, dtype={paper_data.dtype}")
    
    # 2. Perform preprocessing using Paper's optimized operations
    print("\n2. Paper handles data loading and basic operations...")
    # This operation benefits from Paper's tiling and caching
    # In production, you'd do operations like: paper_data - paper_data.mean(axis=0)
    
    # For now, just show data is ready
    print(f"   ✓ Data ready for sklearn: {paper_data.shape}")
    
    # 3. Now use with sklearn
    print("\n3. Feeding to scikit-learn...")
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # Convert Paper array to NumPy for sklearn
        X = paper_data.to_numpy()
        
        # Standard sklearn pipeline
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=10)
        X_reduced = pca.fit_transform(X_scaled)
        
        print(f"   ✓ StandardScaler applied: {X_scaled.shape}")
        print(f"   ✓ PCA reduction: {X_reduced.shape}")
        print(f"   ✓ Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    except ImportError:
        print("   (sklearn not installed, skipping sklearn steps)")
    
    # 4. The key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("""
Paper optimizes the I/O-intensive preprocessing steps:
  • Loading large datasets from disk
  • Computing correlation matrices
  • Matrix multiplications for feature engineering

sklearn handles the ML algorithms:
  • Standardization, PCA, model training
  • All the sophisticated ML logic

This separation allows you to:
  ✓ Use Paper for out-of-core data that doesn't fit in RAM
  ✓ Use sklearn for the ML tasks it excels at
  ✓ No need to rewrite existing sklearn code
  ✓ Simply use Paper for data loading and preprocessing
""")

def demonstrate_feature_engineering():
    """
    Shows Paper's strength in large-scale feature engineering.
    """
    print("\n" + "=" * 70)
    print("Use Case: Feature Engineering at Scale")
    print("=" * 70)
    
    # Simulate a real-world scenario: computing features for a large dataset
    print("\nScenario: Large-scale data preprocessing...")
    
    n_samples = 1000  # Small demo, but imagine 100k+
    n_features = 20
    
    # Load features (in production, from disk)
    X = pnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    
    print(f"Original features: {X.shape}")
    
    # Demonstrate element-wise operations
    print("\nPaper handles operations like:")
    print("  • Element-wise addition/subtraction (normalization)")
    print("  • Scalar multiplication (scaling)")
    print("  • Transpose operations")
    
    # Example: scaling
    X_scaled = X * 2.0
    print(f"\n✓ Lazy scaling plan created: {X_scaled}")
    
    # Execute
    result = X_scaled.compute()
    print(f"✓ Computed scaled features: {result.shape}")
    print("  (All operations optimized with tile-based I/O)")

if __name__ == "__main__":
    demonstrate_sklearn_integration()
    demonstrate_feature_engineering()
    
    print("\n" + "=" * 70)
    print("Integration Complete!")
    print("=" * 70)
