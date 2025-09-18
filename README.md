# Paper framework

Paper a lightweight Python framework for performing matrix computations on datasets that are too large to fit into main memory. It is designed around the principle of lazy evaluation, which allows it to build a computation plan and apply powerful optimizations, such as operator fusion, before executing any costly I/O operations.

The architecture is inspired by modern data systems and academic research (e.g., PreVision), with a clear separation between the logical plan, the physical execution backend, and an intelligent optimizer.

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

