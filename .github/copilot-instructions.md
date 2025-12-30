# Copilot Instructions for Paper Framework

## Project Overview

Paper is a lightweight Python framework for performing **out-of-core matrix computations** on datasets too large to fit in memory. It uses lazy evaluation to build computation plans and apply optimizations (operator fusion) before executing I/O operations.

## Architecture

The codebase follows a clear separation of concerns:

```
paper/
├── core.py       # PaperMatrix - disk-backed matrix using memory-mapped files
├── plan.py       # Lazy evaluation plan tree (EagerNode, AddNode, MultiplyNode, etc.)
├── optimizer.py  # Plan inspection, I/O trace generation, and fusion rules
├── backend.py    # High-performance execution kernels for matrix operations
├── buffer.py     # BufferManager with LRU and Belady's optimal eviction
├── config.py     # Centralized configuration (TILE_SIZE, cache sizes)
└── numpy_api.py  # NumPy-compatible API layer (pnp.array, pnp.zeros, etc.)
```

## Key Design Patterns

### 1. Lazy Evaluation
- Operations return `Plan` objects, not computed results
- Computation happens only when `.compute()` is called
- Plans form a tree structure with operation nodes

### 2. Stateless Buffer Manager
- The `BufferManager` is **stateless** by design to avoid race conditions
- `trace_pos` (current position in I/O trace) is passed as a parameter, NOT stored as instance state
- This enables thread-safe parallel execution

### 3. Tiled Processing
- Matrices are processed in tiles of `TILE_SIZE` (default: 1024)
- All configuration values live in `config.py`

### 4. Memory-Mapped Files
- `PaperMatrix` uses `np.memmap` for out-of-core access
- Always use `np.array(data[slice], copy=True)` to get concrete in-memory tiles

## Coding Conventions

### Python Style
- Use type hints for function parameters: `def add(A: PaperMatrix, B: PaperMatrix, ...)`
- Use `np.float32` as the default dtype
- Document classes and public methods with docstrings

### File I/O Patterns
```python
# Creating empty files for output matrices
def _create_empty_file(filepath, shape, dtype):
    with open(filepath, "wb") as f:
        file_size = shape[0] * shape[1] * np.dtype(dtype).itemsize
        if file_size > 0:
            f.seek(file_size - 1)
            f.write(b'\0')

# Reading tiles from memmap (always copy!)
tile = np.array(self.data[r_start:r_end, c_start:c_end], copy=True)
```

### Buffer Manager Usage
```python
# Stateless pattern - pass trace_pos as parameter
tile_A = buffer_manager.get_tile(A, r_start, c_start, trace_pos)
trace_pos += 1  # Caller manages position
tile_B = buffer_manager.get_tile(B, r_start, c_start, trace_pos)
trace_pos += 1
```

### Plan Node Pattern
```python
class NewOperationNode:
    """Operation node for the computation plan."""
    def __init__(self, left: 'Plan', right: 'Plan'):
        self.left = left
        self.right = right
        self.shape = ...  # Compute output shape
    
    def execute(self, output_path, buffer_manager):
        """Execute this operation."""
        # 1. Execute children to get input matrices
        # 2. Perform tiled computation
        # 3. Return result PaperMatrix
```

## Testing

- Tests use Python's standard `unittest` framework
- Test files are in `tests/` directory
- Run all tests: `python run_tests.py`
- Test modules:
  - `test_core.py` - PaperMatrix class
  - `test_buffer.py` - BufferManager (LRU, Belady)
  - `test_plan_optimizer.py` - Plan tree and I/O trace
  - `test_backend_kernels.py` - Computation kernels
  - `test_fusion_operations.py` - Fused operations
  - `test_numpy_api.py` - NumPy-compatible API

## Important Implementation Details

### Cache Key Format
- Buffer manager cache keys: `(full_filepath, r_start, c_start)`
- I/O trace keys: `(os.path.basename(filepath), r_start, c_start)`
- Always convert between formats when comparing

### Fusion Rules
Defined in `optimizer.py`:
```python
FUSION_RULES = [
    ((MultiplyScalarNode, AddNode), backend.execute_fused_add_multiply),
    ((MultiplyScalarNode, MultiplyNode), backend.execute_fused_matmul_scalar),
    # ...
]
```

### Thread Safety
- Use `threading.Lock()` for cache access in BufferManager
- Never share mutable state across worker threads
- Pass all required context as function parameters

## NumPy API Compatibility

The `numpy_api.py` module provides familiar NumPy interface:
```python
from paper import numpy_api as pnp

a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
b = pnp.zeros((2, 2))
c = (a + b) * 2
result = c.compute()
```

## When Adding New Features

1. **New Operations**: Add a node class in `plan.py`, kernel in `backend.py`
2. **New Optimizations**: Add fusion rules in `optimizer.py`
3. **New API Methods**: Extend `numpy_api.py` following NumPy conventions
4. **Configuration**: Add constants to `config.py`
5. **Always**: Add corresponding tests in `tests/`

## Dependencies

Core dependencies (see `requirements.txt`):
- `numpy` - Array operations
- `h5py` - HDF5 file support (optional)
- Standard library: `collections`, `threading`, `tempfile`, `os`
