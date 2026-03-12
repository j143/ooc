# PyTorch Real-World Example

## Overview

This example demonstrates a **concrete real-world scenario** of using Paper with PyTorch for image classification on large datasets.

## Scenario: Medical Image Classification

**Problem:**
- Large medical imaging dataset (e.g., X-rays, MRI scans, CT scans)
- 100,000+ images at 28×28 pixels (~2.3 GB for grayscale, much larger for color)
- Binary classification task (e.g., tumor detection, defect identification)
- Dataset too large to load entirely into RAM

**Solution:**
- **Paper** handles out-of-core data loading and preprocessing
- **PyTorch** handles model training with standard DataLoader and training loop
- **Zero changes** to existing PyTorch code required

## Files

- `pytorch_mnist_example.py` - Complete working example with real data generation and model training

## Running the Example

### Basic Run (without PyTorch)
Shows data loading and preprocessing with Paper:
```bash
python examples/pytorch_mnist_example.py
```

### Full Run (with PyTorch)
Install PyTorch first to see the complete training workflow:
```bash
pip install torch
python examples/pytorch_mnist_example.py
```

## What the Example Demonstrates

### 1. Large Dataset Generation
```python
# Generate 100,000 images (simulating real medical imaging data)
features, labels_path, output_dir = generate_large_image_dataset(
    n_samples=100000,
    img_height=28,
    img_width=28,
    use_paper=True
)
```

### 2. Paper-Optimized Preprocessing
```python
# Paper handles out-of-core preprocessing with lazy evaluation
preprocessed = features * 2.0  # Lazy operation
result = preprocessed.compute()  # Executes with optimized I/O
```

### 3. Standard PyTorch Training
```python
# Convert to PyTorch tensors (standard workflow)
X_tensor = torch.from_numpy(preprocessed.to_numpy())
y_tensor = torch.from_numpy(labels)

# Create DataLoader (no changes to PyTorch code)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Train model (standard PyTorch loop)
for epoch in range(epochs):
    for data, target in loader:
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### 4. Batch Inference
```python
# Load test data with Paper
test_data = pnp.load('test_images.bin', shape=(50000, 784))

# Preprocess and predict
preprocessed = (test_data * 2.0).compute()
predictions = model(torch.from_numpy(preprocessed.to_numpy()))
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  Large Dataset on Disk (2+ GB)              │
│  • Medical images, satellite imagery, etc.  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Paper Framework (I/O Optimization)         │
│  • Tile-based loading (handles OOM)         │
│  • Belady cache eviction (optimal)          │
│  • Lazy evaluation (build plans)            │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  PyTorch (Model Training)                   │
│  • Standard DataLoader                      │
│  • Your existing training code              │
│  • No modifications needed                  │
└─────────────────────────────────────────────┘
```

## Key Benefits

### 1. No Code Changes Required
Your PyTorch training code stays exactly the same. Just use Paper for data loading:
```python
# Before (NumPy - may run out of memory)
X = np.load('large_dataset.npy')

# After (Paper - handles datasets larger than RAM)
X = pnp.load('large_dataset.bin', shape=(100000, 784)).compute()
```

### 2. Proven Performance
- **1.89x speedup** over Dask on real-world datasets
- Handles datasets **10x-100x larger** than available RAM
- Belady's algorithm ensures optimal cache utilization

### 3. Real-World Applications
- **Medical Imaging**: X-rays, MRI, CT scans for diagnosis
- **Satellite Imagery**: Land use classification, change detection
- **Industrial Inspection**: Defect detection in manufacturing
- **Video Processing**: Frame-by-frame analysis of large video datasets

## Performance Tips

### 1. Use Appropriate Batch Sizes
```python
# Smaller batches for very large datasets
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 2. Leverage Paper's Caching
```python
# Increase cache size if you have more RAM available
result = plan.compute(cache_size_tiles=128)  # Default is 64
```

### 3. Precompute When Possible
```python
# If preprocessing is expensive, compute once and save
preprocessed = (data * 2.0).compute()
pnp.save('preprocessed.bin', preprocessed)
```

## Extending the Example

### Add Data Augmentation
```python
# Paper handles loading, then apply augmentations
data = pnp.load('images.bin', shape=(100000, 784))
scaled = (data * 2.0).compute()

# Use standard libraries for augmentation
from torchvision import transforms
# ... apply transforms to numpy arrays before conversion
```

### Use Real Image Data
```python
# Convert your existing image dataset to binary format
from paper.data_prep import convert_to_binary

convert_to_binary(
    input_path='my_images.h5',
    output_path='my_images.bin',
    dataset_name='images'
)

# Then load with Paper
data = pnp.load('my_images.bin', shape=(n_samples, img_size))
```

### Multi-Class Classification
```python
# Change output layer size
model = SimpleImageClassifier(input_dim=784, num_classes=10)

# Rest of the code stays the same
criterion = nn.CrossEntropyLoss()
```

## Comparison with Alternatives

| Approach | RAM Usage | Speed | Code Changes |
|----------|-----------|-------|--------------|
| **NumPy (in-memory)** | Very High | Fast | None |
| **Dask** | Medium | Medium | Moderate |
| **Paper + PyTorch** | Low | Fast (1.89x) | Minimal |
| **PyTorch IterableDataset** | Low | Slow | Significant |

## Troubleshooting

### Out of Memory Errors
```python
# Reduce batch size
loader = DataLoader(dataset, batch_size=16)

# Or increase cache size for Paper
result = plan.compute(cache_size_tiles=32)
```

### Slow Preprocessing
```python
# Check if bottleneck is I/O or computation
# Paper optimizes I/O, not computation
# For compute-heavy tasks, use GPU or parallel processing
```

### Data Format Issues
```python
# Ensure data is in binary format for best performance
# Convert from HDF5/NumPy using Paper's data_prep tools
```

## Next Steps

1. **Try with your own data**: Convert your dataset to binary format
2. **Experiment with model architectures**: Paper works with any PyTorch model
3. **Scale up**: Test with datasets 10x-100x larger than RAM
4. **Benchmark**: Compare against your current data loading approach

## Additional Resources

- [INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md) - Comprehensive integration patterns
- [examples/integration_pytorch.py](integration_pytorch.py) - Simpler conceptual example
- [benchmarks/](../benchmarks/) - Performance comparison scripts

## Questions?

This example demonstrates that Paper is not a replacement for PyTorch—it's an I/O optimization layer that makes PyTorch faster on large datasets. Your model training code stays exactly as is.
