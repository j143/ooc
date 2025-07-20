# benchmarks/fusion.py
import os
import sys
import shutil
import psutil

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.core import PaperMatrix
from paper.plan import Plan, EagerNode
from paper import backend
from benchmarks.utils import create_matrix_file
from benchmarks.utils import Benchmark

# --- Benchmark Configuration ---
BENCHMARK_DATA_DIR = "benchmark_data"
SHAPE = (8000, 8000) 
BYTES_PER_ELEMENT = 4 # float32

def run_benchmark():
    """
    Runs a benchmark to compare fused vs. non-fused execution across
    time, memory, I/O, and CPU usage.
    """
    # --- 1. Setup Phase ---
    if os.path.exists(BENCHMARK_DATA_DIR):
        shutil.rmtree(BENCHMARK_DATA_DIR)
    os.makedirs(BENCHMARK_DATA_DIR)

    path_A = os.path.join(BENCHMARK_DATA_DIR, "A.bin")
    path_B = os.path.join(BENCHMARK_DATA_DIR, "B.bin")
    
    create_matrix_file(path_A, SHAPE)
    create_matrix_file(path_B, SHAPE)
    
    A_handle = PaperMatrix(path_A, SHAPE, mode='r')
    B_handle = PaperMatrix(path_B, SHAPE, mode='r')

    # --- 3. Run Fused Path ---
    fused_stats = {}
    with Benchmark("Fused Execution: (A + B) * 2") as b:
        plan_A = Plan(EagerNode(A_handle))
        plan_B = Plan(EagerNode(B_handle))
        fused_plan = (plan_A + plan_B) * 2
        result_fused = fused_plan.compute(os.path.join(BENCHMARK_DATA_DIR, "result_fused.bin"))
        result_fused.close()
    
    fused_stats['time'] = b.elapsed
    fused_stats['mem'] = b.peak_mem
    fused_stats['cpu'] = b.avg_cpu
    # Data written: just the final result matrix
    fused_stats['io_write_gb'] = (SHAPE[0] * SHAPE[1] * BYTES_PER_ELEMENT) / (1024**3)

    # --- 4. Run Unfused Path ---
    unfused_stats = {}
    with Benchmark("Unfused Execution: TMP = A + B; D = TMP * 2") as b:
        path_tmp = os.path.join(BENCHMARK_DATA_DIR, "tmp.bin")
        tmp_handle = backend.add(A_handle, B_handle, path_tmp)
        
        result_unfused = PaperMatrix(os.path.join(BENCHMARK_DATA_DIR, "result_unfused.bin"), SHAPE, mode='w+')
        for r in range(0, SHAPE[0], backend.TILE_SIZE):
            for c in range(0, SHAPE[1], backend.TILE_SIZE):
                tile = tmp_handle.data[r:r+backend.TILE_SIZE, c:c+backend.TILE_SIZE]
                result_unfused.data[r:r+backend.TILE_SIZE, c:c+backend.TILE_SIZE] = tile * 2
        result_unfused.data.flush()
        
        tmp_handle.close()
        result_unfused.close()
    
    unfused_stats['time'] = b.elapsed
    unfused_stats['mem'] = b.peak_mem
    unfused_stats['cpu'] = b.avg_cpu
    # Data written: the intermediate TMP matrix AND the final result matrix
    unfused_stats['io_write_gb'] = 2 * (SHAPE[0] * SHAPE[1] * BYTES_PER_ELEMENT) / (1024**3)

    # --- 5. Report Results ---
    print("\n" + "="*55)
    print("                 BENCHMARK RESULTS")
    print("="*55)
    print(f"{'Metric':<20} | {'Fused Path':<15} | {'Unfused Path':<15}")
    print("-"*55)
    print(f"{'Time (s)':<20} | {fused_stats['time']:<15.4f} | {unfused_stats['time']:<15.4f}")
    print(f"{'Peak Memory (MB)':<20} | {fused_stats['mem']:<15.2f} | {unfused_stats['mem']:<15.2f}")
    print(f"{'Avg CPU Util.(%)':<20} | {fused_stats['cpu']:<15.2f} | {unfused_stats['cpu']:<15.2f}")
    print(f"{'Total Write (GB)':<20} | {fused_stats['io_write_gb']:<15.3f} | {unfused_stats['io_write_gb']:<15.3f}")
    print("="*55)

    speedup = unfused_stats['time'] / fused_stats['time']
    io_reduction = unfused_stats['io_write_gb'] / fused_stats['io_write_gb']
    print(f"🎉 FUSION WIN: {speedup:.2f}x faster, with {io_reduction:.1f}x less disk I/O.")
    print("="*55)

    # --- 6. Final Cleanup ---
    A_handle.close()
    B_handle.close()

if __name__ == '__main__':
    run_benchmark()
