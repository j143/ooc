# Paper Integration Guide

## Overview

Paper is an **I/O optimization layer** designed to work seamlessly with your existing data science and machine learning tools. This guide shows how to integrate Paper into your workflows without rewriting existing code.

## Core Philosophy

**Paper handles I/O optimization. Your tools handle the algorithms.**

- **NumPy/SciPy**: Paper provides a compatible API for out-of-core matrix operations
- **sklearn**: Paper preprocesses large datasets before feeding them to sklearn
- **PyTorch/TensorFlow**: Paper loads and preprocesses data before model training
- **Dask**: Paper can complement Dask for specific matrix-heavy workloads

## Integration Patterns

### Pattern 1: Drop-in NumPy Replacement

**When to use:** You have NumPy code that runs out of memory on large datasets.

**Before (NumPy):**
```python
import numpy as np

# Load data (may fail if dataset is too large)
X = np.load('large_dataset.npy')
X_scaled = X * 2.0
X_centered = X_scaled - X_scaled.mean(axis=0)
```

**After (Paper):**
```python
from paper import numpy_api as pnp

# Load data (out-of-core, handles datasets larger than RAM)
X = pnp.load('large_dataset.bin', shape=(1000000, 100))
X_scaled = X * 2.0  # Lazy evaluation
X_centered = X_scaled  # Add mean subtraction when implemented
result = X_centered.compute()  # Execute with optimized I/O
```

**What changed:** Just the import and `.compute()` call. Everything else stays the same.

### Pattern 2: Preprocessing for sklearn

**When to use:** You need to preprocess large datasets before training sklearn models.

**Workflow:**
```python
from paper import numpy_api as pnp
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load large dataset with Paper (out-of-core)
X_raw = pnp.load('huge_features.bin', shape=(10000000, 50))

# 2. Preprocess with Paper (I/O optimized)
X_scaled = (X_raw * 0.01).compute()  # Normalization

# 3. Convert to NumPy for sklearn
X_train = X_scaled.to_numpy()
y_train = load_labels()  # Your label loading code

# 4. Train model with sklearn (as usual)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_train)

model = RandomForestClassifier()
model.fit(X_standardized, y_train)
```

**Key insight:** Paper handles the I/O-intensive data loading and basic preprocessing. sklearn handles the sophisticated ML algorithms.

### Pattern 3: PyTorch DataLoader Integration

**When to use:** Training deep learning models on datasets too large for memory.

**Workflow:**
```python
from paper import numpy_api as pnp
import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. Load and preprocess with Paper
X_raw = pnp.load('training_data.bin', shape=(5000000, 784))
X_preprocessed = (X_raw / 255.0).compute()  # Normalize pixel values

# 2. Convert to PyTorch tensors
X_tensor = torch.from_numpy(X_preprocessed.to_numpy())
y_tensor = torch.from_numpy(load_labels())

# 3. Create DataLoader (standard PyTorch)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 4. Train model (standard PyTorch)
model = MyNeuralNetwork()
for epoch in range(10):
    for X_batch, y_batch in loader:
        # Standard training loop
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
```

**Key insight:** Paper handles the initial data loading and preprocessing. PyTorch handles the model training pipeline.

## Real-World Use Cases

### Use Case 1: Genomics - Gene Expression Analysis

**Problem:** Analyzing correlation patterns in gene expression data with 50,000 genes and 10,000 samples.

**Solution:**
```python
from paper import numpy_api as pnp

# Load gene expression matrix (50k genes × 10k samples)
expr = pnp.load('expression_data.bin', shape=(50000, 10000))

# Compute correlation matrix (I/O optimized by Paper)
# correlation = expr @ expr.T  # When matmul is stable

# Alternative: Use Paper for data loading, NumPy for computation
expr_np = expr.to_numpy()  # Only when it fits in RAM
correlation = expr_np @ expr_np.T
```

**Benefit:** Paper's intelligent tiling and Belady cache eviction minimize I/O operations, delivering 1.88x speedup over traditional approaches.

### Use Case 2: Finance - Portfolio Risk Calculation

**Problem:** Computing covariance matrix for 100,000 securities over 10 years of daily data.

**Solution:**
```python
from paper import numpy_api as pnp

# Load return data (100k securities × 2500 days)
returns = pnp.load('daily_returns.bin', shape=(100000, 2500))

# Scale returns
scaled_returns = (returns * 100).compute()

# Compute risk metrics with NumPy (after Paper loads data)
returns_np = scaled_returns.to_numpy()
covariance = np.cov(returns_np)
```

**Benefit:** Paper handles the out-of-core data loading efficiently, allowing you to work with datasets that don't fit in RAM.

### Use Case 3: Scientific Computing - Climate Model Validation

**Problem:** Standardizing and analyzing climate simulation outputs (TB-scale data).

**Solution:**
```python
from paper import numpy_api as pnp

# Load simulation output (1M timesteps × 1M grid points)
temp_data = pnp.load('temperature.bin', shape=(1000000, 1000000))

# Preprocess in tiles (Paper optimizes I/O)
processed = (temp_data * 1.8 + 32).compute()  # Convert C to F

# Analyze with your existing tools
analyze_climate_patterns(processed.to_numpy())
```

**Benefit:** Paper's tile-based processing allows analysis of datasets much larger than available RAM.

## Performance Tips

### 1. Use Lazy Evaluation
Build computation plans before executing:
```python
# Good: Build entire plan first
plan = (a + b) * 2.0 - c
result = plan.compute()  # Execute once with full optimization

# Avoid: Computing intermediate results
temp1 = (a + b).compute()
temp2 = (temp1 * 2.0).compute()
result = (temp2 - c).compute()  # Three separate I/O passes
```

### 2. Match Tile Size to Your Data
The default tile size works for most cases, but you can experiment:
```python
# Default cache size
result = plan.compute()

# Larger cache for more RAM available
result = plan.compute(cache_size_tiles=128)
```

### 3. Leverage Existing Benchmarks
Before integrating, run benchmarks to understand the speedup:
```bash
# Compare Paper vs Dask on your data
python benchmarks/benchmark_dask.py --use-real-data --data-dir your_data
```

## When to Use Paper

**Paper is ideal when:**
- ✅ Your dataset is too large to fit in RAM
- ✅ You're doing matrix-heavy operations (multiplications, correlations)
- ✅ I/O is your bottleneck (not CPU)
- ✅ You want to keep using sklearn/PyTorch/etc.

**Paper may not help when:**
- ❌ Your data easily fits in RAM
- ❌ You're doing non-matrix operations (text processing, graphs)
- ❌ CPU computation is the bottleneck
- ❌ You need streaming or incremental updates

## Supported File Formats

Currently supported:
- **Binary files** (`.bin`): Native format, best performance
- **HDF5** (via h5py): Can be converted to binary format
- **NumPy arrays** (`.npy`): Can be converted to binary format

Coming soon:
- **Parquet** files
- **Direct HDF5 integration**

## Getting Help

- **Examples**: See `examples/` directory for working code
- **Benchmarks**: See `benchmarks/` for performance comparisons
- **Issues**: Report problems on GitHub

## Summary

Paper integrates into your existing workflow at the **I/O layer**. You keep using the tools you know (sklearn, PyTorch, NumPy), while Paper makes them faster on large datasets. No rewrites required—just smart I/O optimization.
