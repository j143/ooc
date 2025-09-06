import os
import sys
import shutil
import time
import h5py
import numpy as np
import dask.array as da


# In Colab
# !git clone https://github.com/j143/ooc
# %cd /content/ooc
# !pip install .
# 
# Add the project root to the Python path
# In Colab notebooks, __file__ is not defined.
# We can use the current working directory instead, assuming it's the project root after the %cd command.
project_root = os.getcwd()
sys.path.insert(0, project_root)


from paper.core import PaperMatrix
from paper.plan import Plan, EagerNode
from paper.config import TILE_SIZE
from benchmarks.utils import Benchmark # Assuming your Benchmark class is in utils

# --- Configuration ---
BENCH_DATA_DIR = "dask_benchmark_data"
HDF5_FILE = os.path.join(BENCH_DATA_DIR, "data.hdf5")
SHAPE = (8192, 8192)  # Use a size that exceeds your RAM
CACHE_SIZE = 128

def setup_data():
    """Create a shared HDF5 file for both frameworks to use."""
    if os.path.exists(BENCH_DATA_DIR):
        shutil.rmtree(BENCH_DATA_DIR)
    os.makedirs(BENCH_DATA_DIR)

    print(f"Creating shared data file at {HDF5_FILE} with shape {SHAPE}...")
    with h5py.File(HDF5_FILE, 'w') as f:
        f.create_dataset('A', data=np.random.rand(*SHAPE))
        f.create_dataset('B', data=np.random.rand(*SHAPE))
    print("Data creation complete.")

def run_paper_benchmark():
    """Run the A @ B computation using the paper framework."""
    A_handle = PaperMatrix(HDF5_FILE, SHAPE, mode='r') # Paper can read HDF5 with a custom core class
    B_handle = PaperMatrix(HDF5_FILE, SHAPE, mode='r')

    plan_A = Plan(EagerNode(A_handle))
    plan_B = Plan(EagerNode(B_handle))
    matmul_plan = plan_A @ plan_B

    with Benchmark("Paper (Optimal Policy)") as b:
        matmul_plan.compute(
            os.path.join(BENCH_DATA_DIR, "C_paper.bin"),
            cache_size_tiles=CACHE_SIZE
        )
    return {'time': b.elapsed, 'memory': b.peak_mem, 'cpu': b.avg_cpu}


def run_dask_benchmark():
    """Run the A @ B computation using Dask."""
    # Dask reads from the same file, ensuring a fair comparison
    with h5py.File(HDF5_FILE, 'r') as f:
        a_dask = da.from_array(f['A'], chunks=(TILE_SIZE, TILE_SIZE))
        b_dask = da.from_array(f['B'], chunks=(TILE_SIZE, TILE_SIZE))

        computation = a_dask @ b_dask

        with Benchmark("Dask") as b:
            # Dask's .compute() triggers the execution
            result = computation.compute()
    return {'time': b.elapsed, 'memory': b.peak_mem, 'cpu': b.avg_cpu}


if __name__ == '__main__':
    setup_data()

    print("\n--- Running 'paper' Benchmark ---")
    paper_results = run_paper_benchmark()

    print("\n--- Running 'Dask' Benchmark ---")
    dask_results = run_dask_benchmark()

    # --- Print Summary Table ---
    print("\n" + "="*50)
    print("      BENCHMARK COMPARISON: paper vs. Dask")
    print("="*50)
    print(f"{'Metric':<20} | {'Paper (Optimal)':<20} | {'Dask':<20}")
    print("-"*50)
    print(f"{'Time (s)':<20} | {paper_results['time']:.2f}{'':<14} | {dask_results['time']:.2f}")
    print(f"{'Peak Memory (MB)':<20} | {paper_results['memory']:.2f}{'':<14} | {dask_results['memory']:.2f}")
    print(f"{'Avg CPU Util.(%)':<20} | {paper_results['cpu']:.2f}{'':<14} | {dask_results['cpu']:.2f}")
    print("="*50)
