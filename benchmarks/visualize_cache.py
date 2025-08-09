
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

# Configuration presets for different analysis scenarios
SCENARIOS = {
    "fast_test": {"shape": (4096, 4096), "cache_size": 64},
    "standard": {"shape": (8192, 8192), "cache_size": 128}, 
    "large_analysis": {"shape": (16384, 16384), "cache_size": 256},
    "eviction_stress": {"shape": (8192, 8192), "cache_size": 32}  # Force evictions
}

# Choose scenario
SCENARIO = "eviction_stress"  # Change to "fast_test", "large_analysis", or "eviction_stress"
config = SCENARIOS[SCENARIO]
SHAPE = config["shape"]
CACHE_SIZE_TILES = config["cache_size"]

print(f"Using scenario '{SCENARIO}': {SHAPE} matrix, {CACHE_SIZE_TILES} tile cache")

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
    
    print(f"Matrix size: {SHAPE}")
    print(f"Estimated tiles per matrix: {(SHAPE[0]//TILE_SIZE) * (SHAPE[1]//TILE_SIZE)}")
    print(f"Cache size: {CACHE_SIZE_TILES} tiles")
    print("Executing A @ B to generate cache event log...")
    
    start_time = time.time()
    # The compute method now returns the buffer manager instance
    result_matrix, buffer_manager = matmul_plan.compute(
        os.path.join(VIS_DATA_DIR, "C_result.bin")
    )
    elapsed = time.time() - start_time
    print(f"Execution completed in {elapsed:.1f} seconds")

    # --- 3. Process the Event Log ---
    log = buffer_manager.get_log()
    if not log:
        print("Event log is empty. Nothing to visualize.")
        return

    # Analyze event types for debugging
    event_counts = {}
    for event in log:
        event_type = event[1]
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print(f"\nEvent Log Analysis:")
    print(f"Total events: {len(log)}")
    for event_type, count in event_counts.items():
        percentage = (count / len(log)) * 100
        print(f"  {event_type}: {count} ({percentage:.1f}%)")
    
    # For large logs, sample for visualization performance
    if len(log) > 10000:
        sample_size = 10000
        step = len(log) // sample_size
        log = log[::step]
        print(f"Sampled {len(log)} events for visualization")

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
    if evict_x:  # Only plot if evictions exist
        ax1.scatter(evict_x, evict_y, c='blue', marker='x', s=50, label='Eviction', zorder=4)
        print(f"Plotted {len(evict_x)} eviction events")
    else:
        print("No eviction events found - cache may be sized perfectly or eviction logic missing")
    
    ax1.set_title(f'1. Cache Access Pattern Timeline (Cache Size: {CACHE_SIZE_TILES} tiles)')
    ax1.set_xlabel('Operation Index (Time)')
    ax1.set_ylabel('Unique Tile ID')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Add rolling hit rate as secondary y-axis
    ax1_twin = ax1.twinx()
    window_size = max(50, len(log) // 100)  # Adaptive window size
    hit_rates = []
    for i in range(len(log)):
        window_start = max(0, i - window_size)
        window_events = log[window_start:i+1]
        hits = sum(1 for event in window_events if event[1] == 'HIT')
        hit_rate = hits / len(window_events) if window_events else 0
        hit_rates.append(hit_rate)
    
    ax1_twin.plot(range(len(log)), hit_rates, 'purple', alpha=0.7, linewidth=2, label='Rolling Hit Rate')
    ax1_twin.set_ylabel('Rolling Hit Rate', color='purple')
    ax1_twin.set_ylim(0, 1)
    ax1_twin.tick_params(axis='y', labelcolor='purple')

    # Set y-ticks to be more readable if there are not too many tiles
    if len(tile_keys) < 30:
        ax1.set_yticks(range(len(tile_keys)))
        ax1.set_yticklabels([f"{name}[{r},{c}]" for name, r, c in tile_keys], rotation=0, fontsize=8)

    # # --- Plot 2: Improved Tile Access Heatmap ---
    ax2 = axes[1]
    access_counts = Counter(event[2] for event in log if event[1] in ['HIT', 'MISS'])
    max_row_idx = SHAPE[0] // TILE_SIZE
    max_col_idx = SHAPE[1] // TILE_SIZE
    heatmap_combined = np.zeros((max_row_idx, max_col_idx))

    for (name, r, c), count in access_counts.items():
        r_idx, c_idx = r // TILE_SIZE, c // TILE_SIZE
        heatmap_combined[r_idx, c_idx] += count
    
    # Use viridis palette and add tile boundaries
    im = ax2.imshow(heatmap_combined, cmap='viridis', interpolation='nearest')
    ax2.set_title('2. Tile Access Frequency Heatmap (1024x1024 tiles)')
    ax2.set_xlabel('Tile Column Index')
    ax2.set_ylabel('Tile Row Index')
    
    # Add grid lines to show tile boundaries
    ax2.set_xticks(np.arange(-0.5, max_col_idx, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, max_row_idx, 1), minor=True)
    ax2.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.7)
    
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

