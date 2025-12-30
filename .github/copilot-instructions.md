# Copilot Instructions for Paper Framework

## Project Overview

Paper is a lightweight Python framework for performing **out-of-core matrix computations** on datasets too large to fit in memory. It uses lazy evaluation to build computation plans and apply optimizations (operator fusion) before executing I/O operations.

## Architecture

The codebase follows a clear separation of concerns:

```
paper/
├── core.py          # PaperMatrix - disk-backed matrix using memory-mapped files
├── plan.py          # Lazy evaluation plan tree (EagerNode, AddNode, MultiplyNode, etc.)
├── optimizer.py     # Three-stage optimizer: analyze(), rewrite(), execute_plan()
├── backend.py       # High-performance execution kernels for matrix operations
├── buffer.py        # BufferManager with LRU and Belady's optimal eviction
├── config.py        # Centralized configuration (TILE_SIZE, cache sizes)
├── numpy_api.py     # NumPy-compatible API layer (pnp.array, pnp.zeros, etc.)
└── observability.py # Logging, profiling, tracing utilities
```

## Core Principles

These principles guide all architectural decisions in the Paper framework:

1. **Make optimizer purely analytical**: Never materialize data during pattern matching.
2. **Separate concerns**: Analysis (trace + match) → IR rewrite (fuse) → Execution (apply kernels).
3. **Define small, stable contracts**: Each layer (Plan/Node metadata, BufferManager, Backend) has clear interfaces.
4. **Make fusion selection deterministic**: Testable and fast (no I/O during optimization).
5. **Treat heavy tests and benchmarks as gated**: Cached jobs in CI for efficiency.

## Priority Roadmap

### **Priority 1 (Critical)** - Stop side-effects in optimizer ✅ DONE
- Remove any calls to `node.execute()` inside optimizer/analysis
- Replace with metadata-only inspection APIs
- **Why**: Prevents data materialization during plan analysis, critical for performance
- **Status**: ✅ Pattern detection in `_detect_fusion_pattern()` uses only `isinstance()` checks and metadata access. No `execute()` calls during analysis phase

### **Priority 2 (Critical)** - Create two-stage optimizer ✅ DONE
- `analyze(plan)` → Trace + MatchResults
- `rewrite(plan, match)` → FusedPlan (new node types)
- `execute(fused_plan, backend, buffer_mgr)`
- **Why**: Clean separation enables testability and independent optimization
- **Status**: ✅ Three-stage pipeline implemented in `optimizer.py`: `analyze()` generates I/O trace and detects patterns without execution, `rewrite()` prepares optimized plan, `execute_plan()` runs with fusion. Legacy `execute()` maintained for backward compatibility

### **Priority 3 (High)** - Introduce immutable, hashable Plan representation ❌ NOT DONE
- Deterministic hashing for plan diffs, caching, and baseline comparisons
- **Why**: Enables plan comparison, caching, and regression detection
- **Status**: No `__hash__()` or `__eq__()` methods found in Plan or Node classes

### **Priority 4 (High)** - Add a small, fast unit test surface ✅ DONE
- Optimizer tests: trace generation, pattern detection, rewrite correctness (use mocked backend)
- **Why**: Fast feedback loop for optimizer development
- **Status**: `test_plan_optimizer.py` has tests for plan construction and I/O trace generation

### **Priority 5 (High)** - Add integration smoke tests ✅ DONE
- Run fused kernels on tiny matrices (CI quick job)
- **Why**: Catch integration bugs without heavy computation
- **Status**: `test_fusion_operations.py` tests fused kernels with small (128×128) matrices. CI runs all tests in `ci.yml`

### **Priority 6 (Medium)** - Add benchmarks & regression checks ⚠️ PARTIAL
- Baselines stored as CI artifacts
- Regress only if delta > threshold
- **Why**: Prevent performance regressions systematically
- **Status**: `benchmarks/benchmark_dask.py` exists with benchmarking framework. CI uploads test artifacts but no baseline comparison or regression checks

### **Priority 7 (Medium)** - Plugin backend API ✅ DONE
- Simple interface: `(inputs, params, output_path, buffer_mgr)`
- Swapping implementations is trivial
- **Why**: Enables experimentation with different execution strategies
- **Status**: All backend kernels follow consistent signature `(A, B, ..., output_path, buffer_manager)`. Interface is clean and swappable

### **Priority 8 (Low)** - Observability ✅ DONE
- Trace-level logging
- Cost model hooks
- Per-plan flame profiles
- **Why**: Debugging and performance analysis
- **Status**: ✅ Comprehensive observability in `observability.py`: structured logging with `configure_logging()`, `ExecutionProfiler` for timing/flame graphs, `TraceLogger` for execution traces, `CostEstimate` dataclass for I/O cost modeling in optimizer

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

### Three-Stage Optimizer Usage

The new three-stage optimizer provides clean separation between analysis, rewriting, and execution:

```python
from paper.optimizer import analyze, rewrite, execute_plan, estimate_cost
from paper.buffer import BufferManager

# Stage 1: Analyze (no execution - metadata only)
io_trace, match_result = analyze(plan)

# Check what pattern was detected
if match_result.is_fusable:
    print(f"Detected pattern: {match_result.pattern.value}")
    print(f"Parameters: {match_result.parameters}")

# Estimate execution cost
cost = estimate_cost(plan, match_result)
print(f"Predicted I/O ops: {cost.io_operations}")
print(f"Total cost: {cost.total_cost}")

# Stage 2: Rewrite (prepare optimized plan)
rewritten_plan = rewrite(plan, match_result)

# Stage 3: Execute
buffer_manager = BufferManager(max_cache_size_tiles=64, io_trace=io_trace)
result = execute_plan(rewritten_plan, match_result, output_path, buffer_manager)
```

### Observability Features

Enable structured logging, profiling, and tracing:

```python
from paper.observability import configure_logging, get_profiler, TraceLogger

# Configure logging
logger = configure_logging(level="DEBUG", log_file="paper.log")

# Use global profiler
profiler = get_profiler()

with profiler.profile("my_operation"):
    # ... do work ...
    pass

# Print profiling summary
profiler.print_summary()
profiler.save_json("profile.json")
profiler.save_flame_graph("flame.json")

# Execution tracing
trace = TraceLogger()
trace.begin("operation")
trace.log("step 1 complete")
trace.end()
trace.print()
```

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
