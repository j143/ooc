# Paper framework

Paper a lightweight Python framework for performing matrix computations on datasets that are too large to fit into main memory. It is designed around the principle of lazy evaluation, which allows it to build a computation plan and apply powerful optimizations, such as operator fusion, before executing any costly I/O operations.

The architecture is inspired by modern data systems and academic research (e.g., PreVision), with a clear separation between the logical plan, the physical execution backend, and an intelligent optimizer.

## OOCMatrix API - High-Level NumPy-like Interface

The framework now includes **OOCMatrix**, a high-level wrapper that provides a NumPy-like API for out-of-core operations. This API focuses on **orchestration** rather than reimplementing mathematical operations, leveraging existing NumPy/SciPy/Pandas operations with lazy evaluation and block-wise processing.

### Key Features

- **NumPy-like API**: Familiar interface for matrix operations
- **Lazy Evaluation**: Build computation plans before execution
- **Block-wise Processing**: Automatically chunks large datasets
- **Operator Support**: Reuses existing optimized libraries (NumPy, SciPy) for in-block operations
- **Memory Efficient**: Smart block loading, buffer management, and streaming

### Quick Start

```python
from paper import OOCMatrix
import numpy as np

# Create or load large matrices
A = OOCMatrix('fileA.bin', shape=(10_000_000, 1000))
B = OOCMatrix('fileB.bin', shape=(1000, 1000))

# Perform operations using existing NumPy functions
C = A.matmul(B, op=np.dot)

# Iterate over result blocks (no full matrix in memory)
for block, (r, c) in C.iterate_blocks():
    process(block)  # Each block uses NumPy operations

# Common operations (computed block-wise)
mean = A.mean()
std = A.std()
total = A.sum()

# Apply custom transformations using NumPy
A_normalized = A.blockwise_apply(lambda x: (x - mean) / std)

# Lazy evaluation with operator fusion
result = (A + B) * 2  # Builds plan, doesn't execute
materialized = result.compute('output.bin')  # Executes optimized plan
```

### Examples

See `examples_oocmatrix.py` for comprehensive examples including:
- Basic statistics (sum, mean, std, min, max)
- Block-wise transformations
- Lazy evaluation and operator fusion
- Matrix multiplication with custom operations
- Block iteration for streaming workflows

### Architecture


![Framework architecture](/paper-architecture.svg "Architecture")


### Testing

```bash
# Run all tests (including OOCMatrix tests)
python run_tests.py

# Or run specific test modules
python -m unittest tests.test_operators
python -m unittest tests.test_core
python -m unittest tests.test_fusion_operations
```

### Examples

```bash
# Run OOCMatrix examples
python examples_oocmatrix.py
```

### Buffer Manager architecture


![Buffer manager architecture](/buffer-manager-architecture.svg "Buffer Manager")


### Benchmarks

with Dask

8kx8k matrix

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

16kx16k matrix

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

### Results

![eviction stress](/cache_visualization_eviction_stress_32.png "Buffer Manager")

![large analysis](/cache_visualization_large_analysis_new.png "Buffer Manager")

