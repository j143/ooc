# Paper framework

Paper a lightweight Python framework for performing matrix computations on datasets that are too large to fit into main memory. It is designed around the principle of lazy evaluation, which allows it to build a computation plan and apply powerful optimizations, such as operator fusion, before executing any costly I/O operations.

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


![Framework architecture](/paper-architecture.svg "Architecture")


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

