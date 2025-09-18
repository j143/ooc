
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
    "fast_test": {"shape": (4096, 4096), "cache_size": 32},
    "standard": {"shape": (8192, 8192), "cache_size": 128}, 
    "large_analysis": {"shape": (16384, 16384), "cache_size": 256},
    "eviction_stress": {"shape": (8192, 8192), "cache_size": 32}  # Force evictions
}

# Choose scenario
SCENARIO = "large_analysis"  # Change to "fast_test", "large_analysis", or "eviction_stress"
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
        os.path.join(VIS_DATA_DIR, "C_result.bin"), 
        cache_size_tiles=CACHE_SIZE_TILES
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

    # --- Generate the Enhanced Multi-Plot visualization for Buffer Architecture Analysis
    print("Generating enhanced cache visualizations for buffer architecture analysis...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Enhanced Cache Performance Analysis for Buffer Architecture Decisions', fontsize=16)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # -- Plot 1: Cache Access Pattern Timeline with Improved Detail
    tile_keys = sorted(list(set([event[2] for event in log])))
    tile_to_y = {key: i for i, key in enumerate(tile_keys)}

    events = {'HIT': [], 'MISS': [], 'EVICT': []}
    for i, event in enumerate(log):
        _, event_type, key, _ = event
        if key in tile_to_y:
            events[event_type].append((i, tile_to_y[key]))

    ax1 = axes[0]
    miss_x, miss_y = zip(*events['MISS']) if events['MISS'] else ([], [])
    ax1.scatter(miss_x, miss_y, c='red', s=15, label='Cache Miss', alpha=0.7)
    hit_x, hit_y = zip(*events['HIT']) if events['HIT'] else ([], [])
    ax1.scatter(hit_x, hit_y, c='green', s=10, label='Cache Hit', alpha=0.5)
    evict_x, evict_y = zip(*events['EVICT']) if events['EVICT'] else ([], [])
    if evict_x:
        ax1.scatter(evict_x, evict_y, c='blue', marker='x', s=40, label='Eviction', alpha=0.8)
    
    ax1.set_title(f'Access Timeline (Cache: {CACHE_SIZE_TILES} tiles)')
    ax1.set_xlabel('Operation Index')
    ax1.set_ylabel('Tile ID')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -- Plot 2: Matrix-Based Tile Access Pattern (A vs B matrices)
    ax2 = axes[1]
    access_counts_A = Counter()
    access_counts_B = Counter() 
    
    for event in log:
        if event[1] in ['HIT', 'MISS']:
            name, r, c = event[2]
            tile_coord = (r // TILE_SIZE, c // TILE_SIZE)
            if 'A' in name:
                access_counts_A[tile_coord] += 1
            elif 'B' in name:
                access_counts_B[tile_coord] += 1
    
    max_row_idx = SHAPE[0] // TILE_SIZE
    max_col_idx = SHAPE[1] // TILE_SIZE
    
    # Create combined heatmap showing A and B access patterns
    heatmap_diff = np.zeros((max_row_idx, max_col_idx))
    for (r_idx, c_idx), count_A in access_counts_A.items():
        count_B = access_counts_B.get((r_idx, c_idx), 0)
        heatmap_diff[r_idx, c_idx] = count_A - count_B  # Positive = more A, negative = more B
    
    im2 = ax2.imshow(heatmap_diff, cmap='RdBu_r', interpolation='nearest')
    ax2.set_title('A vs B Matrix Access Difference')
    ax2.set_xlabel('Tile Column')
    ax2.set_ylabel('Tile Row')
    fig.colorbar(im2, ax=ax2, label='A_accesses - B_accesses')

    # -- Plot 3: Cache Performance Metrics Over Time
    ax3 = axes[2]
    window_size = max(100, len(log) // 50)
    metrics = {'hit_rate': [], 'eviction_rate': [], 'cache_utilization': []}
    
    for i in range(0, len(log), window_size):
        window = log[i:i+window_size]
        hits = sum(1 for e in window if e[1] == 'HIT')
        evictions = sum(1 for e in window if e[1] == 'EVICT')
        avg_cache_size = sum(e[3] for e in window) / len(window) if window else 0
        
        metrics['hit_rate'].append(hits / len(window) if window else 0)
        metrics['eviction_rate'].append(evictions / len(window) if window else 0) 
        metrics['cache_utilization'].append(avg_cache_size / CACHE_SIZE_TILES if CACHE_SIZE_TILES > 0 else 0)
    
    x_windows = range(len(metrics['hit_rate']))
    ax3.plot(x_windows, metrics['hit_rate'], 'g-', label='Hit Rate', linewidth=2)
    ax3.plot(x_windows, metrics['eviction_rate'], 'r-', label='Eviction Rate', linewidth=2)
    ax3.plot(x_windows, metrics['cache_utilization'], 'b-', label='Cache Utilization', linewidth=2)
    ax3.set_title('Performance Metrics Over Time')
    ax3.set_xlabel('Time Window')
    ax3.set_ylabel('Rate/Utilization')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # -- Plot 4: Tile Reuse Distance Analysis
    ax4 = axes[3]
    tile_last_access = {}
    reuse_distances = []
    
    for i, event in enumerate(log):
        if event[1] in ['HIT', 'MISS']:
            tile_key = event[2]
            if tile_key in tile_last_access:
                distance = i - tile_last_access[tile_key]
                reuse_distances.append(distance)
            tile_last_access[tile_key] = i
    
    if reuse_distances:
        # Plot histogram of reuse distances
        bins = np.logspace(0, np.log10(max(reuse_distances)), 30)
        ax4.hist(reuse_distances, bins=bins, alpha=0.7, edgecolor='black')
        ax4.set_xscale('log')
        ax4.set_title('Tile Reuse Distance Distribution')
        ax4.set_xlabel('Distance (log scale)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Add cache size indicator
        ax4.axvline(x=CACHE_SIZE_TILES, color='red', linestyle='--', 
                   label=f'Cache Size ({CACHE_SIZE_TILES})', linewidth=2)
        ax4.legend()

    # -- Plot 5: Cache Efficiency Analysis
    ax5 = axes[4]
    
    # Calculate miss rate by cache size (theoretical analysis)
    cache_sizes = range(16, 513, 16)  # Test different cache sizes
    theoretical_miss_rates = []
    
    for cache_size in cache_sizes:
        if reuse_distances:
            # Estimate miss rate: tiles with reuse distance > cache_size will likely miss
            long_reuse = sum(1 for d in reuse_distances if d > cache_size)
            miss_rate = long_reuse / len(reuse_distances)
            theoretical_miss_rates.append(miss_rate)
        else:
            theoretical_miss_rates.append(0)
    
    ax5.plot(cache_sizes, theoretical_miss_rates, 'b-', linewidth=2, label='Estimated Miss Rate')
    ax5.axvline(x=CACHE_SIZE_TILES, color='red', linestyle='--', 
               label=f'Current Cache Size ({CACHE_SIZE_TILES})', linewidth=2)
    ax5.set_title('Cache Size vs Miss Rate Analysis')
    ax5.set_xlabel('Cache Size (tiles)')
    ax5.set_ylabel('Estimated Miss Rate')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)

    # -- Plot 6: Eviction Algorithm Effectiveness
    ax6 = axes[5]
    
    # Analyze eviction timing patterns
    eviction_intervals = []
    last_eviction = None
    
    for i, event in enumerate(log):
        if event[1] == 'EVICT':
            if last_eviction is not None:
                eviction_intervals.append(i - last_eviction)
            last_eviction = i
    
    if eviction_intervals:
        ax6.hist(eviction_intervals, bins=30, alpha=0.7, edgecolor='black')
        ax6.set_title('Eviction Interval Distribution')
        ax6.set_xlabel('Operations Between Evictions')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        # Add statistics text
        avg_interval = np.mean(eviction_intervals)
        ax6.axvline(x=avg_interval, color='red', linestyle='--', 
                   label=f'Avg Interval: {avg_interval:.1f}', linewidth=2)
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No Evictions Detected\n(Cache may be oversized)', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Eviction Analysis')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_image_path = os.path.join(VIS_DATA_DIR, "cache_visualization.png")
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
    print(f"Enhanced visualization saved to: {output_image_path}")

    # --- Generate Buffer Architecture Recommendations ---
    print("\n" + "="*60)
    print("BUFFER ARCHITECTURE ANALYSIS & RECOMMENDATIONS")
    print("="*60)
    
    current_hit_rate = event_counts.get('HIT', 0) / len(log) if log else 0
    current_miss_rate = event_counts.get('MISS', 0) / len(log) if log else 0
    current_eviction_rate = event_counts.get('EVICT', 0) / len(log) if log else 0
    
    print(f"Current Performance:")
    print(f"  Hit Rate: {current_hit_rate:.1%}")
    print(f"  Miss Rate: {current_miss_rate:.1%}")  
    print(f"  Eviction Rate: {current_eviction_rate:.1%}")
    
    # Cache sizing recommendations
    if reuse_distances:
        optimal_cache_size = np.percentile(reuse_distances, 80)  # 80th percentile
        print(f"\nCache Sizing Analysis:")
        print(f"  Current cache size: {CACHE_SIZE_TILES} tiles")
        print(f"  Recommended cache size: {int(optimal_cache_size)} tiles (80th percentile of reuse)")
        
        if optimal_cache_size > CACHE_SIZE_TILES * 1.5:
            print(f"  ⚠️  RECOMMENDATION: Increase cache size to ~{int(optimal_cache_size)} tiles")
        elif optimal_cache_size < CACHE_SIZE_TILES * 0.7:
            print(f"  💡 OPTIMIZATION: Cache could be reduced to ~{int(optimal_cache_size)} tiles")
        else:
            print(f"  ✅ Cache size is reasonably optimal")
    
    # Access pattern analysis
    total_unique_tiles = len(tile_keys)
    working_set_ratio = total_unique_tiles / CACHE_SIZE_TILES if CACHE_SIZE_TILES > 0 else float('inf')
    
    print(f"\nWorking Set Analysis:")
    print(f"  Total unique tiles accessed: {total_unique_tiles}")
    print(f"  Working set ratio: {working_set_ratio:.1f}x cache size")
    
    if working_set_ratio > 4:
        print(f"  ⚠️  Large working set - consider:")
        print(f"     • Larger cache size")
        print(f"     • Better tile scheduling")
        print(f"     • Hierarchical caching")
    elif working_set_ratio < 1.5:
        print(f"  💡 Small working set - cache may be oversized")
    
    # Eviction algorithm analysis
    if eviction_intervals:
        eviction_frequency = len(eviction_intervals) / len(log) if log else 0
        print(f"\nEviction Algorithm Analysis:")
        print(f"  Eviction frequency: {eviction_frequency:.3f} per operation")
        print(f"  Average interval: {np.mean(eviction_intervals):.1f} operations")
        
        if eviction_frequency > 0.1:
            print(f"  ⚠️  High eviction frequency - consider larger cache or better algorithm")
        elif eviction_frequency < 0.01:
            print(f"  💡 Low eviction frequency - cache may be oversized")
        else:
            print(f"  ✅ Reasonable eviction frequency")

    print(f"\nArchitectural Recommendations:")
    if current_hit_rate < 0.6:
        print(f"  🔧 Priority 1: Improve hit rate (currently {current_hit_rate:.1%})")
        print(f"     • Increase cache size")
        print(f"     • Improve prefetching")
        print(f"     • Better tile ordering")
    
    if current_eviction_rate > 0.15:
        print(f"  🔧 Priority 2: Reduce eviction overhead (currently {current_eviction_rate:.1%})")
        print(f"     • Larger cache to reduce pressure")
        print(f"     • More efficient eviction algorithm")
    
    if working_set_ratio > 3:
        print(f"  🔧 Priority 3: Address large working set")
        print(f"     • Implement tile clustering")
        print(f"     • Add second-level cache")
        print(f"     • Optimize computation order")

    # --- 5. Cleanup ---
    A_handle.close()
    B_handle.close()
    result_matrix.close()

if __name__ == '__main__':
    visualize_matrix_multiplication()

