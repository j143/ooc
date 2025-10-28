import os
import sys
import shutil
import time
import h5py
import numpy as np
import dask.array as da
import argparse


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
from benchmarks.utils import Benchmark, create_matrix_file

# --- Default Configuration ---
DEFAULT_BENCH_DATA_DIR = "dask_benchmark_data"
DEFAULT_SHAPE = (8192, 8192)
DEFAULT_CACHE_SIZE = 128

def setup_synthetic_data(bench_data_dir, shape):
    """Create synthetic data files for both frameworks to use."""
    if os.path.exists(bench_data_dir):
        shutil.rmtree(bench_data_dir)
    os.makedirs(bench_data_dir)

    print(f"\n{'='*60}")
    print("CREATING SYNTHETIC DATA")
    print(f"{'='*60}")
    print(f"Data directory: {bench_data_dir}")
    print(f"Shape: {shape}")
    print(f"Size per matrix: ~{(shape[0] * shape[1] * 4) / (1024**2):.2f} MB")
    
    # Create binary files for Paper
    A_path = os.path.join(bench_data_dir, "A.bin")
    B_path = os.path.join(bench_data_dir, "B.bin")
    
    create_matrix_file(A_path, shape)
    create_matrix_file(B_path, shape)
    
    # Also create HDF5 file for Dask
    hdf5_path = os.path.join(bench_data_dir, "data.hdf5")
    print(f"Creating HDF5 file for Dask at {hdf5_path}...")
    
    with h5py.File(hdf5_path, 'w') as f:
        # Read from binary files and write to HDF5
        A_data = np.memmap(A_path, dtype=np.float32, mode='r', shape=shape)
        B_data = np.memmap(B_path, dtype=np.float32, mode='r', shape=shape)
        
        f.create_dataset('A', data=A_data[:])
        f.create_dataset('B', data=B_data[:])
    
    print(f"✓ Synthetic data creation complete.")
    
    return A_path, B_path, hdf5_path


def setup_real_data(data_dir, shape):
    """
    Prepare real dataset for benchmarking.
    
    Args:
        data_dir: Directory containing real dataset
        shape: Expected shape of the data
        
    Returns:
        Tuple of (A_path, B_path, hdf5_path)
    """
    print(f"\n{'='*60}")
    print("PREPARING REAL DATA")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    
    # Look for gene expression data
    gene_expr_path = os.path.join(data_dir, "gene_expression.dat")
    
    if not os.path.exists(gene_expr_path):
        raise FileNotFoundError(
            f"Real dataset not found at {gene_expr_path}.\n"
            f"Please generate it first using:\n"
            f"  python -m data_prep.download_dataset --output-dir {data_dir}"
        )
    
    # Check shape
    actual_size = os.path.getsize(gene_expr_path)
    expected_size = shape[0] * shape[1] * 4  # float32
    
    if actual_size != expected_size:
        actual_shape = (int(np.sqrt(actual_size / 4)), int(np.sqrt(actual_size / 4)))
        print(f"Warning: Dataset shape mismatch!")
        print(f"  Expected: {shape}")
        print(f"  Found: approximately {actual_shape}")
        print(f"  Using actual dataset shape...")
        shape = actual_shape
    
    print(f"Dataset shape: {shape}")
    print(f"Size: ~{actual_size / (1024**2):.2f} MB")
    
    # For real data, we'll use the same file for both A and B matrices
    # This is reasonable for benchmarking purposes
    A_path = gene_expr_path
    B_path = gene_expr_path
    
    # Create HDF5 file for Dask
    hdf5_path = os.path.join(data_dir, "data.hdf5")
    
    if not os.path.exists(hdf5_path):
        print(f"Creating HDF5 file for Dask at {hdf5_path}...")
        data = np.memmap(gene_expr_path, dtype=np.float32, mode='r', shape=shape)
        
        with h5py.File(hdf5_path, 'w') as f:
            # Create datasets in chunks to avoid loading all in memory
            chunk_size = min(1000, shape[0])
            f.create_dataset('A', data=data[:], chunks=(chunk_size, chunk_size))
            f.create_dataset('B', data=data[:], chunks=(chunk_size, chunk_size))
        
        print(f"✓ HDF5 file created.")
    else:
        print(f"✓ Using existing HDF5 file: {hdf5_path}")
    
    print(f"✓ Real data preparation complete.")
    
    return A_path, B_path, hdf5_path, shape


def run_paper_benchmark(A_path, B_path, output_dir, shape, cache_size):
    """Run the A @ B computation using the paper framework."""
    A_handle = PaperMatrix(A_path, shape, mode='r')
    B_handle = PaperMatrix(B_path, shape, mode='r')

    plan_A = Plan(EagerNode(A_handle))
    plan_B = Plan(EagerNode(B_handle))
    matmul_plan = plan_A @ plan_B

    with Benchmark("Paper (Optimal Policy)") as b:
        matmul_plan.compute(
            os.path.join(output_dir, "C_paper.bin"),
            cache_size_tiles=cache_size
        )
    return {'time': b.elapsed, 'memory': b.peak_mem, 'cpu': b.avg_cpu}


def run_dask_benchmark(hdf5_path):
    """Run the A @ B computation using Dask."""
    # Dask reads from HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        a_dask = da.from_array(f['A'], chunks=(TILE_SIZE, TILE_SIZE))
        b_dask = da.from_array(f['B'], chunks=(TILE_SIZE, TILE_SIZE))

        computation = a_dask @ b_dask

        with Benchmark("Dask") as b:
            # Dask's .compute() triggers the execution
            result = computation.compute()
    return {'time': b.elapsed, 'memory': b.peak_mem, 'cpu': b.avg_cpu}


def print_results_table(paper_results, dask_results, dataset_info=""):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*70)
    print("      BENCHMARK COMPARISON: Paper vs. Dask")
    if dataset_info:
        print(f"      {dataset_info}")
    print("="*70)
    print(f"{'Metric':<25} | {'Paper (Optimal)':<20} | {'Dask':<20}")
    print("-"*70)
    print(f"{'Time (s)':<25} | {paper_results['time']:.2f}{'':<14} | {dask_results['time']:.2f}")
    print(f"{'Peak Memory (MB)':<25} | {paper_results['memory']:.2f}{'':<14} | {dask_results['memory']:.2f}")
    print(f"{'Avg CPU Util.(%)':<25} | {paper_results['cpu']:.2f}{'':<14} | {dask_results['cpu']:.2f}")
    print("-"*70)
    
    # Calculate speedup
    speedup = dask_results['time'] / paper_results['time']
    mem_saving = ((dask_results['memory'] - paper_results['memory']) / dask_results['memory']) * 100
    
    print(f"{'Paper Speedup':<25} | {speedup:.2f}x")
    print(f"{'Paper Memory Saving':<25} | {mem_saving:.1f}%")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark Paper vs Dask for matrix multiplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic data (default)
  python benchmarks/benchmark_dask.py
  
  # Run with real gene expression data
  python benchmarks/benchmark_dask.py --use-real-data --data-dir real_data
  
  # Custom shape for synthetic data
  python benchmarks/benchmark_dask.py --shape 10000 10000
  
  # Generate real data first, then benchmark
  python -m data_prep.download_dataset --output-dir real_data --size large
  python benchmarks/benchmark_dask.py --use-real-data --data-dir real_data
        """
    )
    
    parser.add_argument(
        '--use-real-data',
        action='store_true',
        help='Use real gene expression dataset instead of synthetic data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=DEFAULT_BENCH_DATA_DIR,
        help='Directory for data files (default: dask_benchmark_data)'
    )
    parser.add_argument(
        '--shape',
        type=int,
        nargs=2,
        default=DEFAULT_SHAPE,
        metavar=('ROWS', 'COLS'),
        help=f'Matrix shape for synthetic data (default: {DEFAULT_SHAPE[0]} {DEFAULT_SHAPE[1]})'
    )
    parser.add_argument(
        '--cache-size',
        type=int,
        default=DEFAULT_CACHE_SIZE,
        help=f'Cache size in tiles for Paper (default: {DEFAULT_CACHE_SIZE})'
    )
    parser.add_argument(
        '--skip-paper',
        action='store_true',
        help='Skip Paper benchmark (run only Dask)'
    )
    parser.add_argument(
        '--skip-dask',
        action='store_true',
        help='Skip Dask benchmark (run only Paper)'
    )
    
    args = parser.parse_args()
    
    # Convert shape to tuple
    shape = tuple(args.shape)
    
    # Setup data
    if args.use_real_data:
        A_path, B_path, hdf5_path, shape = setup_real_data(args.data_dir, shape)
        dataset_info = f"Dataset: Real Gene Expression ({shape[0]} x {shape[1]})"
        output_dir = args.data_dir
    else:
        A_path, B_path, hdf5_path = setup_synthetic_data(args.data_dir, shape)
        dataset_info = f"Dataset: Synthetic ({shape[0]} x {shape[1]})"
        output_dir = args.data_dir
    
    # Run benchmarks
    paper_results = None
    dask_results = None
    
    if not args.skip_paper:
        print("\n" + "="*60)
        print("Running Paper Benchmark")
        print("="*60)
        paper_results = run_paper_benchmark(A_path, B_path, output_dir, shape, args.cache_size)
    
    if not args.skip_dask:
        print("\n" + "="*60)
        print("Running Dask Benchmark")
        print("="*60)
        dask_results = run_dask_benchmark(hdf5_path)
    
    # Print results
    if paper_results and dask_results:
        print_results_table(paper_results, dask_results, dataset_info)
    elif paper_results:
        print(f"\nPaper Results: {paper_results}")
    elif dask_results:
        print(f"\nDask Results: {dask_results}")

