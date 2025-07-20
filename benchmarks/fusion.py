# benchmarks/benchmark_fusion.py
import os
import sys
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.core import PaperMatrix
from paper.plan import Plan, EagerNode
from paper import backend
# Import the new centralized utility
from benchmarks.utils import create_benchmark_data
from benchmarks.utils import Timer

# --- Benchmark Configuration ---
BENCHMARK_DATA_DIR = "benchmark_data"
SHAPE = (8000, 8000) 

def run_benchmark():
    """
    Runs a benchmark to compare fused vs. non-fused execution.
    """
    # --- 1. Setup Phase ---
    if os.path.exists(BENCHMARK_DATA_DIR):
        shutil.rmtree(BENCHMARK_DATA_DIR)
    os.makedirs(BENCHMARK_DATA_DIR)

    path_A = os.path.join(BENCHMARK_DATA_DIR, "A.bin")
    path_B = os.path.join(BENCHMARK_DATA_DIR, "B.bin")
    
    # Use the new centralized utility. It creates random data by default.
    create_benchmark_data(path_A, SHAPE)
    create_benchmark_data(path_B, SHAPE)
    
    # --- 2. Open Handles ---
    A_handle = PaperMatrix(path_A, SHAPE, mode='r')
    B_handle = PaperMatrix(path_B, SHAPE, mode='r')

    # --- 3. Run Fused Path ---
    fused_time = 0
    with Timer("Fused Execution: (A + B) * 2") as t:
        plan_A = Plan(EagerNode(A_handle))
        plan_B = Plan(EagerNode(B_handle))
        fused_plan = (plan_A + plan_B) * 2
        result_fused = fused_plan.compute(os.path.join(BENCHMARK_DATA_DIR, "result_fused.bin"))
        result_fused.close()
    fused_time = t.elapsed

    # --- 4. Run Unfused Path ---
    unfused_time = 0
    with Timer("Unfused Execution: TMP = A + B; D = TMP * 2") as t:
        path_tmp = os.path.join(BENCHMARK_DATA_DIR, "tmp.bin")
        tmp_handle = backend.add(A_handle, B_handle, path_tmp)
        
        # This part simulates the second, non-fused step
        result_unfused = PaperMatrix(os.path.join(BENCHMARK_DATA_DIR, "result_unfused.bin"), SHAPE, mode='w+')
        for r in range(0, SHAPE[0], backend.TILE_SIZE):
            for c in range(0, SHAPE[1], backend.TILE_SIZE):
                tile = tmp_handle.data[r:r+backend.TILE_SIZE, c:c+backend.TILE_SIZE]
                result_unfused.data[r:r+backend.TILE_SIZE, c:c+backend.TILE_SIZE] = tile * 2
        result_unfused.data.flush()
        
        tmp_handle.close()
        result_unfused.close()
    unfused_time = t.elapsed

    # --- 5. Report Results ---
    print("\n" + "="*40)
    print("           BENCHMARK RESULTS")
    print("="*40)
    print(f"Matrix Shape: {SHAPE}")
    print(f"Fused Path Time:      {fused_time:.4f} seconds")
    print(f"Unfused Path Time:    {unfused_time:.4f} seconds")
    print("-"*40)
    
    if fused_time < unfused_time:
        speedup = unfused_time / fused_time
        print(f"🎉 FUSION WIN: {speedup:.2f}x faster!")
    else:
        print("🤔 Unfused was faster. Check I/O patterns or caching.")
    print("="*40)

    # --- 6. Final Cleanup ---
    A_handle.close()
    B_handle.close()

if __name__ == '__main__':
    run_benchmark()
