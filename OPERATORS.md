# OOCMatrix - Out-of-Core Matrix Operations

## Overview

The `OOCMatrix` class provides a high-level, NumPy-like API for out-of-core matrix operations. It wraps existing NumPy/SciPy operations with lazy evaluation, block-wise processing, and streaming constructs, focusing on **orchestration** rather than reimplementing mathematical operations.

## Design Philosophy

**Key Principle**: Don't reimplement optimized math libraries - orchestrate them intelligently.

- **Reuse Existing Libraries**: All mathematical operations (matrix multiply, element-wise ops, reductions) use NumPy/SciPy/Pandas for in-block computations
- **Focus on Orchestration**: The framework handles block loading, buffer management, lazy evaluation, and scheduling
- **Transparent Out-of-Core**: Provides a familiar API that doesn't require full matrices in RAM

## API Reference

### Creating OOCMatrix

```python
from paper import OOCMatrix

# Load existing matrix
A = OOCMatrix('data.bin', shape=(10_000_000, 1000), dtype=np.float32, mode='r')

# Create new matrix
B = OOCMatrix('new.bin', shape=(1000, 500), dtype=np.float32, mode='w+', create=True)
```

### Block-wise Operations

#### blockwise_apply(op, output_path=None)
Apply a function to each block independently.

```python
# Normalize using NumPy operations on each block
A_norm = A.blockwise_apply(lambda x: (x - x.mean()) / x.std())

# Apply ReLU activation
A_relu = A.blockwise_apply(lambda x: np.maximum(0, x))

# Custom transformation
A_transformed = A.blockwise_apply(lambda x: np.tanh(x * 2.0))
```

#### blockwise_reduce(op, combine_op=None)
Reduce across all blocks using a reduction operation.

```python
# Custom reduction
max_val = A.blockwise_reduce(np.max, combine_op=max)

# Sum with custom combining
total = A.blockwise_reduce(np.sum, combine_op=lambda x, y: x + y)
```

### Matrix Operations

#### matmul(other, op=None, output_path=None)
Matrix multiplication with custom operation.

```python
# Use NumPy dot product (default)
C = A.matmul(B, op=np.dot)

# Use custom operation
C = A.matmul(B, op=lambda a, b: np.dot(a, b))
```

### Statistical Operations

All operations are computed block-wise without loading the full matrix:

```python
total = A.sum()      # Sum of all elements
mean = A.mean()      # Mean value
std = A.std()        # Standard deviation
min_val = A.min()    # Minimum value
max_val = A.max()    # Maximum value
```

### Lazy Evaluation & Operators

Build computation plans that execute only when materialized:

```python
# Lazy operations (don't execute immediately)
result = (A + B) * 2
result = A @ B
result = A * scalar

# Execute the plan
materialized = result.compute('output.bin')
```

### Block Iteration

Stream through matrix blocks for custom processing:

```python
for block, (row_start, col_start) in A.iterate_blocks():
    # block is a NumPy array
    # Process each block independently
    process(block)
```

## Usage Examples

### Example 1: Normalization

```python
from paper import OOCMatrix
import numpy as np

A = OOCMatrix('data.bin', shape=(10_000_000, 1000))

# Compute statistics (block-wise)
mean = A.mean()
std = A.std()

# Normalize using existing NumPy operations
A_normalized = A.blockwise_apply(lambda x: (x - mean) / std)

print(f"Normalized mean: {A_normalized.mean()}")  # ~0
print(f"Normalized std: {A_normalized.std()}")    # ~1
```

### Example 2: Matrix Multiplication Pipeline

```python
A = OOCMatrix('A.bin', shape=(10_000_000, 1000))
B = OOCMatrix('B.bin', shape=(1000, 500))

# Matrix multiplication using NumPy dot
C = A.matmul(B, op=np.dot)

# Process results block-by-block
for block, (r, c) in C.iterate_blocks():
    # Send to downstream system
    send_to_database(block, position=(r, c))
```

### Example 3: Complex Expression with Fusion

```python
A = OOCMatrix('A.bin', shape=(1_000_000, 1000))
B = OOCMatrix('B.bin', shape=(1_000_000, 1000))

# Build lazy expression (operator fusion optimization)
result = (A + B) * 2

# Execute optimized plan
C = result.compute('result.bin')
```

### Example 4: Custom Block Processing

```python
A = OOCMatrix('data.bin', shape=(10_000_000, 1000))

# Apply custom function using SciPy
from scipy.special import expit

A_sigmoid = A.blockwise_apply(lambda x: expit(x))

# Chain operations
A_processed = A.blockwise_apply(
    lambda x: np.clip((x - x.mean()) / x.std(), -3, 3)
)
```

## Performance Characteristics

- **Memory Efficient**: Only loads blocks needed for current computation
- **Lazy Evaluation**: Builds DAG for optimization before execution
- **Operator Fusion**: Automatically fuses compatible operations (e.g., (A+B)*scalar)
- **Buffer Management**: Intelligent caching and eviction strategies
- **Parallelization**: Thread-based parallelism for fused operations

## Comparison with Similar Frameworks

| Feature | OOCMatrix | Dask | Vaex |
|---------|-----------|------|------|
| Lazy Evaluation | ✓ | ✓ | ✓ |
| Operator Fusion | ✓ | Limited | Limited |
| Block Iteration | ✓ | ✓ | ✓ |
| NumPy Backend | ✓ | ✓ | ✓ |
| Buffer Management | Advanced | Basic | Basic |
| Custom Operations | ✓ | ✓ | ✓ |

## Implementation Details

### No Kernel Rewrites
All mathematical operations use existing, optimized libraries:
- Matrix operations: `np.dot`, `np.matmul`
- Element-wise: `np.add`, `np.multiply`, etc.
- Reductions: `np.sum`, `np.mean`, `np.std`, etc.
- Custom: Any NumPy/SciPy function

### Focus Areas
The framework provides:
1. **Block Orchestration**: Managing how data flows through operations
2. **Lazy DAG Building**: Constructing computation plans
3. **Buffer Management**: Caching, eviction, prefetching
4. **Scheduling**: Optimal ordering of block operations
5. **Streaming**: Iterating over results without materialization

### What We Don't Do
- ❌ Reimplement matrix multiplication
- ❌ Rewrite NumPy/SciPy kernels
- ❌ Custom linear algebra code
- ❌ Low-level optimization (handled by NumPy/BLAS)

### What We Do
- ✓ Smart block loading and caching
- ✓ Lazy evaluation and operator fusion
- ✓ Memory-efficient streaming
- ✓ Computation plan optimization
- ✓ Transparent out-of-core orchestration

## Testing

The OOCMatrix implementation includes comprehensive tests:

```bash
# Run all operator tests
python -m unittest tests.test_operators

# Run specific test class
python -m unittest tests.test_operators.TestOOCMatrix
python -m unittest tests.test_operators.TestOOCMatrixIntegration
```

Test coverage includes:
- Block iteration
- Block-wise operations
- Lazy evaluation
- Operator overloading
- Statistical operations
- Context manager protocol
- Integration with existing Plan infrastructure

## Benchmarks

See main README for performance comparisons with Dask and other frameworks.

## Future Extensions

Potential areas for enhancement:
- GPU backend support (CuPy)
- Additional reduction operations
- More sophisticated operator fusion patterns
- Distributed computation support
- Integration with Pandas for heterogeneous data
