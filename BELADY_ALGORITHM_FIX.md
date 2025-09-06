# Belady's Algorithm Fix

## Problem Statement
Belady's algorithm distances were not working correctly in the buffer manager, causing poor cache performance during matrix multiplication operations.

## Root Cause Analysis
The issue was a **key format mismatch** between the I/O trace and cache keys:

- **I/O Trace keys**: `('A.bin', r_start, c_start)` - using `os.path.basename()`
- **Cache keys**: `('/full/path/A.bin', r_start, c_start)` - using full file paths

This caused `future_trace.index(tile_key)` to always raise `ValueError`, assigning all cached tiles a distance of `float('inf')`, making eviction choices arbitrary instead of optimal.

## Solution
Modified `_evict_optimal()` in `paper/buffer.py` to convert cache keys to basename format before lookup:

```python
# Convert cache key to match io_trace format (basename instead of full path)
trace_key = (os.path.basename(tile_key[0]), tile_key[1], tile_key[2])
# Find the next distance when this tile is needed
distance = future_trace.index(trace_key)
```

## Performance Impact
**Test Scenario**: Matrix multiplication (8192×8192) with 32-tile cache (eviction stress scenario)

### Before Fix:
- Total events: 1,536
- Cache misses: 576 (37.5%)
- Cache hits: 448 (29.2%)
- Evictions: 512 (33.3%)

### After Fix:
- Total events: 1,137
- Cache misses: 177 (15.6%) ← **69% reduction**
- Cache hits: 847 (74.5%) ← **156% improvement**  
- Evictions: 113 (9.9%) ← **78% reduction**

## Validation
The fix ensures that Belady's algorithm correctly identifies the optimal tile to evict (the one with the furthest future use), dramatically improving cache hit rates and reducing unnecessary I/O operations.

Cache visualization saved in `cache_visualization_fixed.png` shows the improved access patterns.