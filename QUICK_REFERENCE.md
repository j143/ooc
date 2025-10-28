# Quick Reference Guide: Real Dataset Integration

## Installation & Setup

```bash
# Install dependencies
pip install numpy h5py psutil dask

# Clone repository
git clone https://github.com/j143/ooc
cd ooc
```

## Quick Start Examples

### 1. Generate a Real Dataset

```bash
# Small dataset (~95 MB) - for quick testing
python -m data_prep.download_dataset --output-dir real_data --size small

# Medium dataset (~381 MB) - standard benchmarking
python -m data_prep.download_dataset --output-dir real_data --size medium

# Large dataset (~763 MB) - comprehensive testing
python -m data_prep.download_dataset --output-dir real_data --size large
```

### 2. Run Benchmarks

#### With Real Data
```bash
python benchmarks/benchmark_dask.py --use-real-data --data-dir real_data
```

#### With Synthetic Data
```bash
python benchmarks/benchmark_dask.py --shape 8192 8192
```

#### With Custom Cache Size
```bash
python benchmarks/benchmark_dask.py --use-real-data --data-dir real_data --cache-size 256
```

### 3. Run the Complete Demo

```bash
# Quick demo with automatic cleanup
python demo_real_dataset.py --size small --cleanup

# Full demo without cleanup
python demo_real_dataset.py --size medium --output-dir my_data
```

## Converting Custom Datasets

### From NumPy
```bash
python -m data_prep.convert_to_binary input.npy output.bin --validate
```

### From HDF5
```bash
python -m data_prep.convert_to_binary input.h5 output.bin \
    --format hdf5 --dataset my_dataset --validate
```

### From CSV
```bash
python -m data_prep.convert_to_binary input.csv output.bin \
    --format csv --shape 10000 5000 --validate
```

## Python API Usage

### Generate Dataset Programmatically

```python
from data_prep import download_gene_expression_data

# Generate dataset
filepath, shape = download_gene_expression_data(
    output_dir="real_data",
    size="medium",
    random_seed=42
)

print(f"Dataset created: {filepath}")
print(f"Shape: {shape}")
```

### Validate Dataset

```python
from data_prep import validate_binary_file
import numpy as np

is_valid = validate_binary_file(
    filepath="real_data/gene_expression.dat",
    shape=(10000, 10000),
    dtype=np.float32
)

print(f"Valid: {is_valid}")
```

### Convert Data Format

```python
from data_prep import convert_to_paper_format

output_path, shape = convert_to_paper_format(
    input_path="data.npy",
    output_path="data.bin",
    input_format="npy"
)

print(f"Converted to: {output_path}")
```

## Common Use Cases

### 1. Quick Performance Test
```bash
# Generate small dataset and benchmark
python -m data_prep.download_dataset --output-dir test_data --size small
python benchmarks/benchmark_dask.py --use-real-data --data-dir test_data
```

### 2. Comprehensive Benchmark Suite
```bash
# Test multiple sizes
for size in small medium large; do
    echo "Testing size: $size"
    python -m data_prep.download_dataset --output-dir data_$size --size $size
    python benchmarks/benchmark_dask.py --use-real-data --data-dir data_$size
done
```

### 3. Compare Synthetic vs Real Data
```bash
# Real data benchmark
python -m data_prep.download_dataset --output-dir real_data --size medium
python benchmarks/benchmark_dask.py --use-real-data --data-dir real_data

# Synthetic data benchmark with same shape
python benchmarks/benchmark_dask.py --shape 10000 10000
```

## Dataset Size Reference

| Preset | Genes | Samples | File Size | Recommended RAM |
|--------|-------|---------|-----------|-----------------|
| small  | 5,000 | 5,000   | ~95 MB    | ≥ 512 MB        |
| medium | 10,000| 10,000  | ~381 MB   | ≥ 1 GB          |
| large  | 20,000| 10,000  | ~763 MB   | ≥ 2 GB          |
| xlarge | 30,000| 15,000  | ~1.7 GB   | ≥ 4 GB          |

## Troubleshooting

### Issue: Dataset not found
```bash
# Ensure you've generated the dataset first
python -m data_prep.download_dataset --output-dir real_data --size medium
```

### Issue: Shape mismatch
```bash
# Check actual dataset dimensions
ls -lh real_data/gene_expression.dat

# Validate the dataset
python -c "from data_prep import validate_binary_file; \
    validate_binary_file('real_data/gene_expression.dat', (10000, 10000))"
```

### Issue: Out of memory
```bash
# Use a smaller dataset
python -m data_prep.download_dataset --output-dir real_data --size small

# Or increase cache size
python benchmarks/benchmark_dask.py --use-real-data --data-dir real_data --cache-size 512
```

## Performance Tips

1. **Cache Size**: Increase `--cache-size` for better performance (at cost of memory)
2. **Dataset Size**: Start with `small` for testing, use `large` for real benchmarks
3. **Reproducibility**: Use the same `--seed` value for reproducible datasets
4. **Memory**: Ensure available RAM is 2-3x the dataset size for optimal performance

## File Locations

- **Data Preparation**: `data_prep/`
- **Benchmarks**: `benchmarks/benchmark_dask.py`
- **Tests**: `tests/test_data_prep.py`
- **Demo**: `demo_real_dataset.py`
- **Documentation**: `data_prep/README.md`, `REAL_DATASET_IMPLEMENTATION.md`

## Getting Help

```bash
# Data preparation help
python -m data_prep.download_dataset --help
python -m data_prep.convert_to_binary --help

# Benchmark help
python benchmarks/benchmark_dask.py --help

# Demo help
python demo_real_dataset.py --help

# Run tests
python run_tests.py
```

## Example Output

### Benchmark Results
```
======================================================================
      BENCHMARK COMPARISON: Paper vs. Dask
      Dataset: Real Gene Expression (5000 x 5000)
======================================================================
Metric                    | Paper (Optimal)      | Dask                
----------------------------------------------------------------------
Time (s)                  | 1.75               | 3.31
Peak Memory (MB)          | 361.17               | 259.72
Avg CPU Util.(%)          | 372.24               | 396.25
----------------------------------------------------------------------
Paper Speedup             | 1.89x
Paper Memory Saving       | -39.1%
======================================================================
```

Paper achieves **1.89x speedup** on real gene expression data! 🚀
