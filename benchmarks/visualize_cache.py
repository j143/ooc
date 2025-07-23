
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import Counter

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.core import PaperMatrix
from paper.plan import Plan, EagerNode
from paper.buffer import TILE_SIZE
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

    # --- Generate the Multi-Plot visualization
    print("Generating advanced cache visualizations...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 24), gridspec_kw={'height_ratios': [3, 2, 1]})
    fig.suptitle('Advanced Cache Performance Analysis for Matrix Multiplication', fontsize=16)

    # -- Plot 1: Cache acces pattern timeline
    # Assign a unique integer ID (y-coordinate) to each tile key
    tile_keys = sorted(list(set([event[2] for event in log])))
    tile_to_y = {key: i for i, key in enumerate(tile_keys)}

    events = {'HIT': [], 'MISS': [], 'EVICT': []}
    for i, event in enumerate(log):
        _, event_type, key, _ = event
        if key in tile_to_y:
            events[event_type].append((i, tile_to_y[key]))

    ax1 = axes[0]
    miss_x, miss_y = zip(*events['MISS']) if events['MISS'] else ([], [])
    ax1.scatter(miss_x, miss_y, c='red', s=30, label='Cache Miss', zorder=3)
    hit_x, hit_y = zip(*events['HIT']) if events['HIT'] else ([], [])
    ax1.scatter(hit_x, hit_y, c='green', s=20, label='Cache Hit', zorder=3)
    evict_x, evict_y = zip(*events['EVICT']) if events['EVICT'] else ([], [])
    ax1.scatter(evict_x, evict_y, c='blue', marker='x', s=30, label='Eviction', zorder=4)
    ax1.set_title(f'1. Cache Access Pattern Timeline (Cache Size: {CACHE_SIZE_TILES} tiles)')
    ax1.set_xlabel('Operation Index (Time)')
    ax1.set_ylabel('Unique Tile ID')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Set y-ticks to be more readable if there are not too many tiles
    if len(tile_keys) < 30:
        ax.set_yticks(range(len(tile_keys)))
        ax.set_yticklabels([f"{name}[{r},{c}]" for name, r, c in tile_keys], rotation=0, fontsize=8)

    # # --- Plot 2: Static Tile Access Heatmap ---
    ax2 = axes[1]
    access_counts = Counter(event[2] for event in log if event[1] in ['HIT', 'MISS'])
    max_row_idx = SHAPE[0] // TILE_SIZE
    max_col_idx = SHAPE[1] // TILE_SIZE
    heatmap_A = np.zeros((max_row_idx, max_col_idx))
    heatmap_B = np.zeros((max_row_idx, max_col_idx))

    for (name, r, c), count in access_counts.items():
        r_idx, c_idx = r // TILE_SIZE, c // TILE_SIZE
        if name == 'A.bin':
            heatmap_A[r_idx, c_idx] = count
        elif name == 'B.bin':
            heatmap_B[r_idx, c_idx] = count
    
    im = ax2.imshow(heatmap_A + heatmap_B, cmap='hot', interpolation='nearest')
    ax2.set_title('2. Static Tile Access Heatmap (Total Hits + Misses)')
    ax2.set_xlabel('Tile Column Index')
    ax2.set_ylabel('Tile Row Index')
    fig.colorbar(im, ax=ax2, label='Access Count')

    # --- Plot 3: Cache Occupancy Over Time ---
    ax3 = axes[2]
    time_steps = range(len(log))
    cache_sizes = [event[3] for event in log]
    ax3.plot(time_steps, cache_sizes, label='Cache Occupancy', color='purple')
    ax3.axhline(y=CACHE_SIZE_TILES, color='r', linestyle='--', label='Max Cache Size')
    ax3.set_title('3. Cache Occupancy Over Time')
    ax3.set_xlabel('Operation Index (Time)')
    ax3.set_ylabel('Tiles in Cache')
    ax3.set_ylim(0, CACHE_SIZE_TILES + 2)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_image_path = os.path.join(VIS_DATA_DIR, "cache_visualization.png")
    plt.savefig(output_image_path)
    print(f"Visualization saved to: {output_image_path}")

    # --- 5. Cleanup ---
    A_handle.close()
    B_handle.close()
    result_matrix.close()

if __name__ == '__main__':
    visualize_matrix_multiplication()

