
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.core import PaperMatrix
from paper.plan import Plan, EagerNode
from benchmarks.utils import create_matrix_file
from benchmarks.utils import Benchmark

# --- Visualization Configuration ---
VIS_DATA_DIR = "visualize_data"
# Use smaller matrices for a clearer, less cluttered visualization
SHAPE = (4096, 4096)
# Use a small cache size to force misses and evictions, making the graph interesting
CACHE_SIZE_TILES = 16

def visualize_matrix_multiplication():
    """
    Runs A @ B and generates a visualization of the cache access patterns.
    """
    # --- 1. Setup Phase ---
    if os.path.exists(VIS_DATA_DIR):
        shutil.rmtree(VIS_DATA_DIR)
    os.makedirs(VIS_DATA_DIR)

    path_A = os.path.join(VIS_DATA_DIR, "A.bin")
    path_B = os.path.join(VIS_DATA_DIR, "B.bin")
    
    create_matrix_file(path_A, SHAPE)
    create_matrix_file(path_B, SHAPE)
    
    A_handle = PaperMatrix(path_A, SHAPE, mode='r')
    B_handle = PaperMatrix(path_B, SHAPE, mode='r')

    # --- 2. Build and Execute Plan ---
    plan_A = Plan(EagerNode(A_handle))
    plan_B = Plan(EagerNode(B_handle))
    
    matmul_plan = plan_A @ plan_B
    
    print("Executing A @ B to generate cache event log...")
    # The compute method now returns the buffer manager instance
    result_matrix, buffer_manager = matmul_plan.compute(
        os.path.join(VIS_DATA_DIR, "C_result.bin")
    )
    print("Execution complete.")

    # --- 3. Process the Event Log ---
    log = buffer_manager.get_log()
    if not log:
        print("Event log is empty. Nothing to visualize.")
        return

    # Assign a unique integer ID (y-coordinate) to each tile key
    tile_keys = sorted(list(set([event[2] for event in log])))
    tile_to_y = {key: i for i, key in enumerate(tile_keys)}

    events = {'HIT': [], 'MISS': [], 'EVICT': []}
    for i, (timestamp, event_type, key) in enumerate(log):
        y = tile_to_y[key]
        events[event_type].append((i, y))

    # --- 4. Generate the Visualization ---
    print("Generating cache access pattern visualization...")
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot misses (large red dots)
    miss_x, miss_y = zip(*events['MISS']) if events['MISS'] else ([], [])
    ax.scatter(miss_x, miss_y, c='red', s=50, label='Cache Miss (Disk Read)', zorder=3)

    # Plot hits (small green dots)
    hit_x, hit_y = zip(*events['HIT']) if events['HIT'] else ([], [])
    ax.scatter(hit_x, hit_y, c='green', s=10, label='Cache Hit (RAM)', zorder=2)

    # Plot evictions (blue 'x' markers)
    evict_x, evict_y = zip(*events['EVICT']) if events['EVICT'] else ([], [])
    ax.scatter(evict_x, evict_y, c='blue', marker='x', s=30, label='Eviction', zorder=4)

    ax.set_title(f'Cache Access Pattern for Matrix Multiplication (Cache Size: {CACHE_SIZE_TILES} tiles)')
    ax.set_xlabel('Operation Index (Time)')
    ax.set_ylabel('Unique Tile ID')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Set y-ticks to be more readable if there are not too many tiles
    if len(tile_keys) < 30:
        ax.set_yticks(range(len(tile_keys)))
        ax.set_yticklabels([f"{name}[{r},{c}]" for name, r, c in tile_keys], rotation=0, fontsize=8)

    plt.tight_layout()
    output_image_path = os.path.join(VIS_DATA_DIR, "cache_visualization.png")
    plt.savefig(output_image_path)
    print(f"✅ Visualization saved to: {output_image_path}")

    # --- 5. Cleanup ---
    A_handle.close()
    B_handle.close()
    result_matrix.close()

if __name__ == '__main__':
    visualize_matrix_multiplication()

