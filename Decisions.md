
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

# Shared state vs stateless for `trace_pos`

Status: In review

![Buffer manager sharedstate vs stateless](/buffer-manager-shared-position.svg "Buffer Manager")

### Shared State vs. Stateless

In high-performance systems, one of the most critical design choices is how to manage state in a concurrent environment. The evolution of the paper framework's BufferManager provides a perfect case study, highlighting the pitfalls of a shared-state design and the robustness of a stateless approach.

#### The Flawed Approach: Shared State

Initially, the BufferManager was designed with a single, shared counter, self.trace_pos, to track its position in the future I/O trace. Multiple worker threads would read and update this central counter.

Analogy: A team of workers sharing a single whiteboard. The first worker reads an instruction, erases it, and writes the next one.

**The Problem:** This created a classic race condition. If Thread A read the counter and then paused, Thread B could read the same value. When Thread A finally updated the counter, Thread B would be left working with outdated information. This "corrupted future view" caused the optimal eviction policy to make nonsensical decisions, degrading its performance to that of the inefficient LRU policy.

This approach is simple to conceptualize but is fragile, unpredictable, and extremely difficult to debug in a parallel system.

#### The Robust Solution: A Stateless Design

The corrected architecture eliminates this shared state. The BufferManager no longer tracks its own position in the trace. Instead, it becomes a stateless service.

Analogy: A team of workers where each is given their own personal, pre-written instruction sheet.

**The Mechanism:** The responsibility of tracking progress is moved up to the orchestrator (the backend kernel). Before submitting any tasks, the backend calculates the correct trace_pos for each individual unit of work. This position is then passed as a parameter to the BufferManager.get_tile method.

Each call to the BufferManager is now a self-contained, independent request that provides all the context needed to make a perfect decision. By removing the shared counter, we completely eliminate the race condition.
