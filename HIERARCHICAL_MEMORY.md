# Hierarchical Memory Management Implementation

## Overview

This implementation adds multi-tier caching (RAM → SSD → Network) to the Paper matrix framework, enabling it to scale beyond single-machine memory limits. The system automatically manages data movement across storage tiers based on access patterns.

## Architecture

### Storage Tier Abstraction (`paper/storage_tier.py`)

The implementation uses an abstract `StorageTier` base class that defines the interface for all storage tiers:

- **`get(key)`**: Retrieve data from this tier or lower tiers, with automatic promotion
- **`put(key, data)`**: Store data in this tier, with automatic eviction and demotion
- **`_evict_one()`**: Evict least recently used item when capacity is reached
- **Metrics tracking**: Hits, misses, evictions, promotions, demotions

### Three Concrete Tier Implementations

#### 1. RAMTier (Fastest, Smallest)
- **Storage**: In-memory dictionary
- **Eviction Policy**: LRU (Least Recently Used)
- **Typical Capacity**: 64-256 tiles
- **Use Case**: Hot data that is frequently accessed

#### 2. SSDTier (Medium Speed, Medium Capacity)
- **Storage**: Local file system (NumPy .npy files)
- **Eviction Policy**: LRU
- **Typical Capacity**: 256-1024 tiles
- **Use Case**: Recently accessed data that doesn't fit in RAM

#### 3. NetworkTier (Slowest, Largest)
- **Storage**: Simulated network storage (local files with latency)
- **Latency Simulation**: Configurable delay (e.g., 50ms)
- **Typical Capacity**: 1024+ tiles
- **Use Case**: Cold data, archival storage, cloud-native deployments

### Hierarchical Buffer Manager (`paper/hierarchical_buffer.py`)

Orchestrates the three-tier hierarchy:

```python
RAM (64 tiles)
  ↓ demotion on eviction
  ↑ promotion on access
SSD (256 tiles)
  ↓ demotion on eviction
  ↑ promotion on access
Network (1024 tiles)
```

**Key Features:**
- **Automatic Promotion**: When data is accessed in a lower tier, it's automatically promoted to the higher tier
- **Automatic Demotion**: When a tier reaches capacity, LRU data is evicted and demoted to the next tier
- **Thread-Safe**: All tier operations are protected by locks
- **Metrics**: Comprehensive tracking of cache behavior

## Usage

### Basic Usage

```python
from paper.hierarchical_buffer import HierarchicalBufferManager
from paper.core import PaperMatrix
from paper.backend import add

# Create hierarchical buffer manager
buffer_mgr = HierarchicalBufferManager(
    ram_capacity_tiles=64,
    ssd_capacity_tiles=256,
    network_capacity_tiles=1024,
    network_latency_ms=50.0
)

# Load matrices
A = PaperMatrix("A.bin", shape=(8192, 8192))
B = PaperMatrix("B.bin", shape=(8192, 8192))

# Perform operation with hierarchical caching
C = add(A, B, "C.bin", buffer_mgr)

# View metrics
buffer_mgr.print_metrics()
```

### Configuration Options

```python
HierarchicalBufferManager(
    ram_capacity_tiles=64,        # Number of tiles in RAM cache
    ssd_capacity_tiles=256,       # Number of tiles in SSD cache
    network_capacity_tiles=1024,  # Number of tiles in network cache
    ssd_cache_dir="/path/to/ssd", # Optional: custom SSD cache location
    network_cache_dir="/path/to/network",  # Optional: custom network cache location
    network_latency_ms=50.0       # Simulated network latency in milliseconds
)
```

### Metrics

The hierarchical buffer manager provides detailed metrics:

```python
# Get metrics for all tiers
metrics = buffer_mgr.get_tier_metrics()

# Access individual tier metrics
ram_metrics = metrics['ram']
print(f"RAM Hit Rate: {ram_metrics['hit_rate']:.2%}")
print(f"RAM Utilization: {ram_metrics['utilization']:.2%}")

# Get summary metrics
summary = buffer_mgr.get_summary_metrics()
print(f"Overall Hit Rate: {summary['overall_hit_rate']:.2%}")
```

## Testing

The implementation includes comprehensive tests:

### Unit Tests (`tests/test_storage_tier.py`)
- **TestRAMTier**: 6 tests for in-memory cache
- **TestSSDTier**: 4 tests for disk-based cache
- **TestNetworkTier**: 3 tests for network storage
- **TestTierHierarchy**: 5 tests for multi-tier interactions

### Integration Tests (`tests/test_hierarchical_buffer.py`)
- Basic tile retrieval
- Cache hit/miss behavior
- Tier metrics tracking
- RAM to SSD demotion
- SSD to RAM promotion
- Summary metrics computation
- Multiple matrices support
- Large workload stress testing

### System Tests (`tests/test_hierarchical_system.py`)
- Matrix addition with hierarchical cache
- Matrix multiplication with hierarchical cache
- Cache promotion during computation
- Cache efficiency comparison
- Concurrent access testing
- Comprehensive metrics reporting

### Running Tests

```bash
# Run all hierarchical memory management tests
python -m unittest tests.test_storage_tier
python -m unittest tests.test_hierarchical_buffer
python -m unittest tests.test_hierarchical_system

# Run all tests including existing tests
python run_tests.py
```

## Demo

Run the interactive demo to see hierarchical caching in action:

```bash
python demo_hierarchical.py
```

The demo includes:
1. Basic hierarchical cache operations
2. Tier promotion and demotion
3. Matrix operations with hierarchical caching
4. Performance comparison across cache configurations

## Performance Characteristics

### Cache Hit Rates (Typical)
- **RAM Tier**: 60-80% hit rate for hot data
- **SSD Tier**: 10-20% hit rate for warm data
- **Overall**: 70-90% hit rate (avoiding disk I/O for most accesses)

### Latency (Approximate)
- **RAM Hit**: ~100ns
- **SSD Hit**: ~100μs (1000x slower than RAM)
- **Network Hit**: ~50ms (500,000x slower than RAM)
- **Disk I/O (Miss)**: ~10ms

### Scalability
- **Single-tier (RAM only)**: Limited by available memory (~64GB typical)
- **Two-tier (RAM + SSD)**: Limited by SSD capacity (~1TB typical)
- **Three-tier (RAM + SSD + Network)**: Virtually unlimited (cloud storage)

## Implementation Details

### Thread Safety
- Each tier has its own lock to prevent race conditions
- Locks are acquired when accessing cache state (get, put, evict)
- Deadlock prevention: `_evict_one()` called from within lock doesn't re-acquire lock

### Memory Management
- RAMTier keeps data in memory using NumPy arrays
- SSDTier and NetworkTier use NumPy's .npy format for efficient serialization
- Automatic cleanup of cache directories on tier destruction

### Eviction Policy
- LRU (Least Recently Used) for all tiers
- Could be extended to support other policies (LFU, ARC, etc.)
- Optimal eviction (Belady's algorithm) could be integrated with I/O trace

## Future Enhancements

### Potential Improvements
1. **Adaptive Tier Sizing**: Automatically adjust tier capacities based on workload
2. **Prefetching**: Anticipate future accesses and pre-load data
3. **Compression**: Compress data in lower tiers to increase effective capacity
4. **Cloud Integration**: Replace NetworkTier simulation with actual S3/Azure/GCS storage
5. **NUMA-aware RAM Tier**: Optimize for NUMA architectures
6. **Tiering Policies**: Support for different eviction policies per tier
7. **Background Migration**: Asynchronously move data between tiers during idle time

### Cloud-Native Deployment
The hierarchical memory management system is designed for cloud-native deployments:
- **Container-friendly**: Each tier can use separate volumes
- **Kubernetes-ready**: SSD tier can use persistent volumes, network tier can use object storage
- **Elastic**: Tier capacities can be adjusted based on instance type and workload

## Conclusion

The hierarchical memory management system enables the Paper framework to:
- **Scale beyond RAM**: Process datasets larger than available memory
- **Optimize performance**: Keep hot data in fast tiers
- **Support cloud deployments**: Three-tier architecture maps naturally to cloud infrastructure
- **Maintain compatibility**: Drop-in replacement for existing BufferManager

All tests pass, including both new hierarchical tests and existing framework tests, ensuring no regressions were introduced.
