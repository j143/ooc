# Paper framework

Paper a lightweight Python framework for performing matrix computations on datasets that are too large to fit into main memory. It is designed around the principle of lazy evaluation, which allows it to build a computation plan and apply powerful optimizations, such as operator fusion, before executing any costly I/O operations.

The architecture is inspired by modern data systems and academic research (e.g., PreVision), with a clear separation between the logical plan, the physical execution backend, and an intelligent optimizer.

### Architecture


![Framework architecture](/paper-architecture.svg "Architecture")


### Hierarchical Memory Management

Paper now supports multi-tier caching (RAM → SSD → Network) for scaling beyond single-machine limits. The hierarchical buffer manager automatically manages data movement across tiers:

- **RAM Tier**: Fast in-memory cache (smallest capacity)
- **SSD Tier**: Local disk-based cache (medium capacity)
- **Network Tier**: Remote storage simulation (largest capacity)

Data is automatically promoted from slower tiers to faster tiers on access, and demoted when space is needed.

```python
from paper.hierarchical_buffer import HierarchicalBufferManager
from paper.core import PaperMatrix
from paper.backend import add

# Create hierarchical buffer manager
buffer_mgr = HierarchicalBufferManager(
    ram_capacity_tiles=64,      # 64 tiles in RAM
    ssd_capacity_tiles=256,     # 256 tiles on SSD
    network_capacity_tiles=1024, # 1024 tiles in network storage
    network_latency_ms=50.0     # Simulated network latency
)

# Use with matrix operations
A = PaperMatrix("A.bin", shape=(8192, 8192))
B = PaperMatrix("B.bin", shape=(8192, 8192))
C = add(A, B, "C.bin", buffer_mgr)

# View performance metrics
buffer_mgr.print_metrics()
```

Run the demo to see hierarchical caching in action:

```bash
python demo_hierarchical.py
```

### Testing

```bash
# Run all tests
python ./tests/run_tests.py

# Run a specific test
python ./tests/run_tests.py addition
python ./tests/run_tests.py fused
python ./tests/run_tests.py scalar

# Run hierarchical memory management tests
python -m unittest tests.test_storage_tier
python -m unittest tests.test_hierarchical_buffer
python -m unittest tests.test_hierarchical_system
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

