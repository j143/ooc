# ooc

Paper a lightweight Python framework for performing matrix computations on datasets that are too large to fit into main memory. It is designed around the principle of lazy evaluation, which allows it to build a computation plan and apply powerful optimizations—such as operator fusion—before executing any costly I/O operations.

The architecture is inspired by modern data systems and academic research (e.g., PreVision), with a clear separation between the logical plan, the physical execution backend, and an intelligent optimizer.

### Architecture

```
+-----------------------------------------------------------------+
|                          User Application                       |
|                 (e.g., result = (A + B) * 2)                    |
+-----------------------------------------------------------------+
                                │
                                ▼
+-----------------------------------------------------------------+
|    1. Plan Layer (paper/plan.py)                                 |
|       - User-facing `Plan` object.                              |
|       - Builds a graph of `PlanNode` objects (AST).             |
|       - Does NO computation.                                    |
+-----------------------------------------------------------------+
                                │
                                ▼ .compute()
+-----------------------------------------------------------------+
|    2. Optimizer Layer (paper/optimizer.py)                       |
|       - Inspects the entire `PlanNode` graph.                   |
|       - Matches patterns against a list of `FUSION_RULES`.      |
|       - Rewrites the plan or selects a specialized kernel.      |
+-----------------------------------------------------------------+
                                │
                                ▼
+-----------------------------------------------------------------+
|    3. Backend Layer (paper/backend.py)                           |
|       - A library of "eager" execution kernels.                 |
|       - Contains both standard and fused functions.             |
|       - Performs the actual tile-by-tile computation.           |
+-----------------------------------------------------------------+
                                │
                                ▼
+-----------------------------------------------------------------+
|    4. Core Layer (paper/core.py)                                 |
|       - `paperMatrix` class handles direct disk I/O.             |
|       - Uses `numpy.memmap` for efficient file access.          |
+-----------------------------------------------------------------+

```

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


### Results

![fast test](/cache_visualization_fast_test.png "Buffer Manager")

![Standard](/cache_visualization_standard.png "Buffer Manager")

