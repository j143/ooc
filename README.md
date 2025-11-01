# Paper framework

Paper a lightweight Python framework for performing matrix computations on datasets that are too large to fit into main memory. It is designed around the principle of lazy evaluation, which allows it to build a computation plan and apply powerful optimizations, such as operator fusion, before executing any costly I/O operations.

> When your matrix computation is bottlenecked by I/O (data too large for RAM), Paper's intelligent tiling + prefetching strategy outperforms Dask's lazy evaluation.

The architecture is inspired by modern data systems and academic research (e.g., PreVision), with a clear separation between the logical plan, the physical execution backend, and an intelligent optimizer.

## NumPy-Compatible API

Paper now includes a **NumPy-compatible API layer** that provides a familiar interface for users migrating from NumPy or other array libraries. This makes it easy to leverage Paper's out-of-core capabilities with minimal code changes.

### Quick Start

```python
# Import Paper's NumPy-compatible API
from paper import numpy_api as pnp
import numpy as np

# Create arrays (similar to NumPy)
a = pnp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
b = pnp.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)

# Perform operations with lazy evaluation
c = (a + b) * 2

# Execute the computation plan
result = c.compute()
print(result.to_numpy())
```

### Key Features

- **Familiar NumPy Interface**: Use the same syntax as NumPy for array creation and operations
- **Lazy Evaluation**: Build computation plans without executing until `.compute()` is called
- **Automatic Optimization**: Operator fusion and intelligent caching applied automatically
- **Out-of-Core Support**: Handle datasets larger than memory seamlessly
- **Matrix Operations**: Support for addition, scalar multiplication, and matrix multiplication (@)

### Supported Operations

**Array Creation:**
- `pnp.array(data)` - Create array from data
- `pnp.zeros(shape)` - Create zeros array
- `pnp.ones(shape)` - Create ones array
- `pnp.eye(n)` - Create identity matrix
- `pnp.random_rand(shape)` - Create random array

**Operations:**
- `a + b` - Element-wise addition
- `a * scalar` - Scalar multiplication
- `a @ b` - Matrix multiplication
- `a.T` - Transpose

**I/O:**
- `pnp.load(filepath, shape)` - Load array from file
- `pnp.save(filepath, array)` - Save array to file

### Examples

See `examples/numpy_api_example.py` for comprehensive examples demonstrating:
- Basic array operations
- Chained operations with lazy evaluation
- Matrix multiplication
- File I/O
- Large array handling (out-of-core)

Run the examples:
```bash
python examples/numpy_api_example.py
```

### Architecture

#### How it fits in application

Your application (ML, Finance, Science)
  - sklearn, PyTorch, XGBoost, etc.
                  ↓
Paper: I/O optimization layer
  - Intelligent tiling + prefetching
  - Lazy evaluation with compute plans
  - Optimal buffer management (example, Belady inspired...)
                  ↓
Storage (HDF5, Binary, S3, etc.)
  - Paper orchestrates reads, doesn't replace

key: Paper is **transparent** to your application. It replaces I/O, not your business logic.


#### Paper architecture

![Framework architecture](/paper-architecture.svg "Architecture")

### Three committments to guide design

1. Reuse best operations that already exists
 - Provide matrix operations: @, .T, +, -, /, reductions
 - Don't implement ML: No Logisitic regression, NN, clustering
 - Let sklearn, PyTorch, XGBoost do their job

2. Respect user workflows
 - NumPy users: import paper as pnp (same API, better I/O)
 - Dask users: Paper handles matrix ops, Dask handles scheduling
 - sklearn users: Transparent optimization of preprocessing
 - Doesn't require rewriting existing code

3. Enable production ML workflows
 - Feature engineering at scale (correlation matrices)
 - Batch prediction on huge datasets
 - Iterative solvers (scientific computing)
 - All using existing frameworks, Paper optimizes I/O

2. Respect User Workflows (Don't Disrupt)

NumPy users: import paper as pnp (same API, better I/O)

Dask users: Paper handles matrix ops, Dask handles scheduling

sklearn users: Transparent optimization of preprocessing

Not: Require rewriting existing code

3. Enable Production ML Workflows (Don't Replace)

Feature engineering at scale (correlation matrices)

Batch prediction on huge datasets

Iterative solvers (scientific computing)

All using existing frameworks, Paper optimizes I/O

### Testing

```bash
# Run all tests
python ./tests/run_tests.py

# Run a specific test
python ./tests/run_tests.py addition
python ./tests/run_tests.py fused
python ./tests/run_tests.py scalar
```

### Buffer Manager architecture


![Buffer manager architecture](/buffer-manager-architecture.svg "Buffer Manager")


### Benchmarks

Paper includes comprehensive benchmarking capabilities to compare performance with Dask on both synthetic and real-world datasets.

#### Running Benchmarks

**Synthetic Data (Default):**
```bash
# Quick test with small matrices
python benchmarks/benchmark_dask.py --shape 1000 1000

# Standard benchmark (8k x 8k)
python benchmarks/benchmark_dask.py --shape 8192 8192

# Large benchmark (16k x 16k)
python benchmarks/benchmark_dask.py --shape 16384 16384
```

**Real-World Data:**
```bash
# Generate a realistic gene expression dataset
python -m data_prep.download_dataset --output-dir real_data --size medium

# Run benchmark with real data
python benchmarks/benchmark_dask.py --use-real-data --data-dir real_data
```

#### Benchmark Results

**Synthetic Data - 8kx8k matrix**

```
==================================================
      BENCHMARK COMPARISON: paper vs. Dask
==================================================
Metric               | Paper (Optimal)      | Dask                
--------------------------------------------------
Time (s)             | 28.79               | 57.00
Peak Memory (MB)     | 1382.03               | 1710.95
Avg CPU Util.(%)     | 170.74               | 169.30
==================================================
```

**Synthetic Data - 16kx16k matrix**

```
Multiplication complete.
--- Finished: Paper (Optimal Policy) in 224.7157 seconds ---

--- Running 'Dask' Benchmark ---

--- Starting: Dask ---
/usr/local/lib/python3.12/dist-packages/dask/array/routines.py:452: PerformanceWarning: Increasing number of chunks by factor of 16
  out = blockwise(
--- Finished: Dask in 467.5384 seconds ---

==================================================
      BENCHMARK COMPARISON: paper vs. Dask
==================================================
Metric               | Paper (Optimal)      | Dask                
--------------------------------------------------
Time (s)             | 224.72               | 467.54
Peak Memory (MB)     | 3970.48               | 4738.61
Avg CPU Util.(%)     | 169.33               | 162.30
==================================================
```

**Real-World Data - Gene Expression (5k x 5k)**

Paper demonstrates even better performance on structured real-world data:

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

### Real Dataset Support

Paper now includes a complete data preparation pipeline for working with real-world datasets. This enables benchmarking on realistic data that mimics production workloads.

**Features:**
- Generate realistic gene expression datasets with biological characteristics
- Convert data from common formats (HDF5, NumPy, CSV, TSV) to Paper's binary format
- Validate converted datasets for correctness
- Multiple size presets (small, medium, large, xlarge)

**Quick Start:**
```bash
# Generate a dataset
python -m data_prep.download_dataset --output-dir real_data --size large

# Benchmark with it
python benchmarks/benchmark_dask.py --use-real-data --data-dir real_data
```

See [data_prep/README.md](data_prep/README.md) for detailed documentation.

### Machine Learning Task: Gene Expression Classification

Paper now includes an end-to-end ML workflow implementing ML algorithms **using Paper's operators** (matrix multiplication, transpose, addition, subtraction). This demonstrates Paper's capability to handle ML workloads directly, not just as a data loading framework.

**Features:**
- Complete gene expression classification pipeline
- **ML algorithms implemented using Paper's out-of-core operators** (logistic regression via gradient descent)
- Disease vs control classification
- Performance metrics: accuracy, ROC AUC
- Side-by-side comparison of Paper vs Dask on ML tasks
- Reproducible results with seed control
- **Benchmarks Paper's ML workload capability**, not external ML libraries

**Quick Start:**
```bash
# Generate dataset
python -m data_prep.download_dataset --output-dir ml_data --size small

# Run ML classification with Paper
python ml_classification.py --data-path ml_data/gene_expression.dat --shape 5000 5000

# Compare Paper vs Dask on ML task
python ml_classification.py \
  --data-path ml_data/gene_expression.dat \
  --shape 5000 5000 \
  --hdf5-path ml_data/data.hdf5 \
  --compare
```

**Example Output:**
```
ML PIPELINE COMPARISON: Paper vs. Dask/HDF5
======================================================================
Metric                         | Paper           | Dask/HDF5      
----------------------------------------------------------------------
Total Time (s)                 | 1.99            | 1.86           
Accuracy                       | 0.4970          | 0.4970         
ROC AUC                        | 0.5074          | 0.5074         
----------------------------------------------------------------------
Paper Speedup                  | 0.93x           |
======================================================================
```

This demonstrates **actionable ML results** with real datasets, not just computational benchmarking.

**Complete Example:**
```bash
# Run the comprehensive ML example
python examples/ml_classification_example.py
```

This example demonstrates:
- Complete ML problem-solving workflow (not just benchmarking)
- Disease classification with real metrics (accuracy, ROC AUC)
- Framework comparison on actual ML tasks
- Reproducible results with actionable insights

### Results

![eviction stress](/cache_visualization_eviction_stress_32.png "Buffer Manager")

![large analysis](/cache_visualization_large_analysis_new.png "Buffer Manager")

