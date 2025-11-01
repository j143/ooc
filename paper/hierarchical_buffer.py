"""
Hierarchical Buffer Manager integrating multi-tier storage.
Builds on the existing BufferManager with RAM → SSD → Network tiers.
"""

import threading
import time
from typing import Optional, List, Dict
import numpy as np

from .core import PaperMatrix
from .storage_tier import StorageTier, RAMTier, SSDTier, NetworkTier


class HierarchicalBufferManager:
    """
    Buffer manager with multi-tier caching support.
    Orchestrates data movement across RAM → SSD → Network tiers.
    """
    
    def __init__(self, 
                 ram_capacity_tiles: int = 64,
                 ssd_capacity_tiles: int = 256,
                 network_capacity_tiles: int = 1024,
                 ssd_cache_dir: Optional[str] = None,
                 network_cache_dir: Optional[str] = None,
                 network_latency_ms: float = 50.0,
                 io_trace: Optional[list] = None):
        """
        Initialize hierarchical buffer manager.
        
        Args:
            ram_capacity_tiles: Number of tiles to cache in RAM
            ssd_capacity_tiles: Number of tiles to cache on SSD
            network_capacity_tiles: Number of tiles to cache in network storage
            ssd_cache_dir: Directory for SSD cache (None = temp directory)
            network_cache_dir: Directory for network cache (None = temp directory)
            network_latency_ms: Simulated network latency in milliseconds
            io_trace: Optional I/O trace for optimal eviction policy
        """
        # Build the tier hierarchy: RAM → SSD → Network
        self.network_tier = NetworkTier(
            capacity_tiles=network_capacity_tiles,
            cache_dir=network_cache_dir,
            latency_ms=network_latency_ms
        )
        
        self.ssd_tier = SSDTier(
            capacity_tiles=ssd_capacity_tiles,
            cache_dir=ssd_cache_dir,
            next_tier=self.network_tier
        )
        
        self.ram_tier = RAMTier(
            capacity_tiles=ram_capacity_tiles,
            next_tier=self.ssd_tier
        )
        
        self.io_trace = io_trace
        self.lock = threading.Lock()
        
        # Event logging for compatibility with existing BufferManager
        self.event_log = []
    
    def get_tile(self, matrix: PaperMatrix, r_start: int, c_start: int, 
                 trace_pos: int = 0) -> np.ndarray:
        """
        Fetch a tile from the hierarchical cache or load from disk.
        
        Args:
            matrix: The matrix to fetch from
            r_start: Row start position for the tile
            c_start: Column start position for the tile
            trace_pos: Current position in the I/O trace (for future use)
            
        Returns:
            Tile data as numpy array
        """
        tile_key = (matrix.filepath, r_start, c_start)
        current_time = time.perf_counter()
        
        # Try to get from the tier hierarchy
        data = self.ram_tier.get(tile_key)
        
        if data is not None:
            # Cache hit (at some tier)
            with self.lock:
                self.event_log.append((current_time, 'HIT', tile_key, 
                                      self.ram_tier.size))
            return data
        
        # Cache miss - load from source file
        with self.lock:
            self.event_log.append((current_time, 'MISS', tile_key, 
                                  self.ram_tier.size))
        
        # Load from disk
        data = matrix.get_tile(r_start, c_start)
        
        # Store in the hierarchy (will cascade through tiers as needed)
        self.ram_tier.put(tile_key, data)
        
        return data
    
    def get_log(self) -> List:
        """Get event log for compatibility with existing tests."""
        with self.lock:
            return list(self.event_log)
    
    def get_tier_metrics(self) -> Dict[str, dict]:
        """
        Get performance metrics for all tiers.
        
        Returns:
            Dictionary mapping tier name to metrics
        """
        return {
            'ram': self.ram_tier.get_metrics(),
            'ssd': self.ssd_tier.get_metrics(),
            'network': self.network_tier.get_metrics()
        }
    
    def get_summary_metrics(self) -> dict:
        """
        Get aggregated metrics across all tiers.
        
        Returns:
            Dictionary with summary statistics
        """
        metrics = self.get_tier_metrics()
        
        total_hits = sum(m['hits'] for m in metrics.values())
        total_misses = sum(m['misses'] for m in metrics.values())
        total_evictions = sum(m['evictions'] for m in metrics.values())
        
        ram_hit_rate = metrics['ram']['hit_rate']
        
        # Overall hit rate (found in any tier vs loaded from source)
        total_requests = len(self.event_log)
        cache_hits = sum(1 for event in self.event_log if event[1] == 'HIT')
        overall_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_evictions': total_evictions,
            'ram_hit_rate': ram_hit_rate,
            'overall_hit_rate': overall_hit_rate,
            'total_requests': total_requests,
            'tiers': metrics
        }
    
    def clear(self) -> None:
        """Clear all tiers and reset metrics."""
        self.ram_tier.clear()
        self.ssd_tier.clear()
        self.network_tier.clear()
        
        with self.lock:
            self.event_log.clear()
    
    def print_metrics(self) -> None:
        """Print formatted metrics for all tiers."""
        print("\n" + "=" * 70)
        print("HIERARCHICAL BUFFER MANAGER METRICS")
        print("=" * 70)
        
        metrics = self.get_tier_metrics()
        
        for tier_name in ['ram', 'ssd', 'network']:
            m = metrics[tier_name]
            print(f"\n{m['name']} Tier:")
            print(f"  Hits: {m['hits']}")
            print(f"  Misses: {m['misses']}")
            print(f"  Hit Rate: {m['hit_rate']:.2%}")
            print(f"  Evictions: {m['evictions']}")
            print(f"  Promotions: {m['promotions']}")
            print(f"  Demotions: {m['demotions']}")
            print(f"  Utilization: {m['size']}/{m['capacity']} ({m['utilization']:.2%})")
        
        summary = self.get_summary_metrics()
        print(f"\nOverall:")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Overall Hit Rate: {summary['overall_hit_rate']:.2%}")
        print(f"  RAM Hit Rate: {summary['ram_hit_rate']:.2%}")
        print("=" * 70 + "\n")
