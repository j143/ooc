"""
Storage tier abstraction for hierarchical memory management.
Implements multi-tier caching: RAM → SSD → Network
"""

import os
import shutil
import tempfile
import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class StorageTier(ABC):
    """
    Abstract base class for storage tiers in hierarchical memory management.
    Each tier can store and retrieve tiles, with different performance characteristics.
    """
    
    def __init__(self, name: str, capacity_tiles: int, next_tier: Optional['StorageTier'] = None):
        """
        Initialize a storage tier.
        
        Args:
            name: Human-readable name for this tier (e.g., "RAM", "SSD", "Network")
            capacity_tiles: Maximum number of tiles this tier can store
            next_tier: The next tier in the hierarchy (slower, larger capacity)
        """
        self.name = name
        self.capacity = capacity_tiles
        self.next_tier = next_tier
        self.size = 0  # Current number of tiles stored
        self.lock = threading.Lock()
        
        # Metrics tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.promotions = 0  # Tiles promoted from lower tier
        self.demotions = 0   # Tiles demoted to lower tier
    
    @abstractmethod
    def _get_internal(self, key: Tuple) -> Optional[np.ndarray]:
        """
        Internal method to retrieve a tile from this tier only.
        Returns None if not found.
        """
        pass
    
    @abstractmethod
    def _put_internal(self, key: Tuple, data: np.ndarray) -> None:
        """
        Internal method to store a tile in this tier only.
        """
        pass
    
    @abstractmethod
    def _evict_one(self) -> None:
        """
        Internal method to evict one tile from this tier.
        Should demote to next tier if available.
        """
        pass
    
    @abstractmethod
    def _contains(self, key: Tuple) -> bool:
        """Check if this tier contains the given key."""
        pass
    
    def get(self, key: Tuple) -> Optional[np.ndarray]:
        """
        Retrieve a tile from this tier or any lower tier.
        Promotes tiles from lower tiers to this tier on access.
        
        Args:
            key: Tile key (filepath, row_start, col_start)
            
        Returns:
            Tile data if found, None otherwise
        """
        with self.lock:
            # Try to get from this tier
            data = self._get_internal(key)
            
            if data is not None:
                self.hits += 1
                return data
            
            self.misses += 1
        
        # Not in this tier, try next tier
        if self.next_tier is not None:
            data = self.next_tier.get(key)
            
            if data is not None:
                # Promote to this tier
                self.put(key, data)
                with self.lock:
                    self.promotions += 1
            
            return data
        
        # Not found in any tier
        return None
    
    def put(self, key: Tuple, data: np.ndarray) -> None:
        """
        Store a tile in this tier.
        Evicts if necessary and demotes to next tier.
        
        Args:
            key: Tile key (filepath, row_start, col_start)
            data: Tile data to store
        """
        with self.lock:
            # Check if we need to evict
            if self.size >= self.capacity and not self._contains(key):
                self._evict_one()
                self.evictions += 1
            
            # Store the tile
            self._put_internal(key, data)
    
    def get_metrics(self) -> dict:
        """Get performance metrics for this tier."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'name': self.name,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'promotions': self.promotions,
                'demotions': self.demotions,
                'hit_rate': hit_rate,
                'size': self.size,
                'capacity': self.capacity,
                'utilization': self.size / self.capacity if self.capacity > 0 else 0.0
            }
    
    def clear(self) -> None:
        """Clear all data from this tier."""
        with self.lock:
            self.size = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.promotions = 0
            self.demotions = 0


class RAMTier(StorageTier):
    """
    RAM tier - fastest, smallest capacity.
    Stores tiles in memory using a dictionary.
    """
    
    def __init__(self, capacity_tiles: int, next_tier: Optional[StorageTier] = None):
        super().__init__("RAM", capacity_tiles, next_tier)
        self.cache = {}  # key -> (data, lru_order)
        self.lru_counter = 0
    
    def _get_internal(self, key: Tuple) -> Optional[np.ndarray]:
        if key in self.cache:
            data, _ = self.cache[key]
            # Update LRU order
            self.lru_counter += 1
            self.cache[key] = (data, self.lru_counter)
            return data
        return None
    
    def _put_internal(self, key: Tuple, data: np.ndarray) -> None:
        if key not in self.cache:
            self.size += 1
        
        self.lru_counter += 1
        self.cache[key] = (data.copy(), self.lru_counter)
    
    def _evict_one(self) -> None:
        if not self.cache:
            return
        
        # Find LRU item
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
        data, _ = self.cache[lru_key]
        
        # Demote to next tier if available
        if self.next_tier is not None:
            self.next_tier.put(lru_key, data)
            # Don't acquire lock here - called from within lock
            self.demotions += 1
        
        # Remove from this tier
        del self.cache[lru_key]
        self.size -= 1
    
    def _contains(self, key: Tuple) -> bool:
        return key in self.cache
    
    def clear(self) -> None:
        super().clear()
        with self.lock:
            self.cache.clear()
            self.lru_counter = 0


class SSDTier(StorageTier):
    """
    SSD tier - medium speed, medium capacity.
    Stores tiles as files in a local directory.
    """
    
    def __init__(self, capacity_tiles: int, cache_dir: Optional[str] = None, 
                 next_tier: Optional[StorageTier] = None):
        super().__init__("SSD", capacity_tiles, next_tier)
        
        # Create cache directory
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="ssd_tier_")
            self.owns_cache_dir = True
        else:
            self.cache_dir = cache_dir
            self.owns_cache_dir = False
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.index = {}  # key -> (filename, lru_order)
        self.lru_counter = 0
    
    def _key_to_filename(self, key: Tuple) -> str:
        """Convert a tile key to a unique filename."""
        filepath, r_start, c_start = key
        base = os.path.basename(filepath)
        # Remove extension and create unique name
        name = os.path.splitext(base)[0]
        return f"{name}_r{r_start}_c{c_start}.npy"
    
    def _get_internal(self, key: Tuple) -> Optional[np.ndarray]:
        if key in self.index:
            filename, _ = self.index[key]
            filepath = os.path.join(self.cache_dir, filename)
            
            if os.path.exists(filepath):
                data = np.load(filepath)
                # Update LRU order
                self.lru_counter += 1
                self.index[key] = (filename, self.lru_counter)
                return data
            else:
                # File was deleted externally, clean up index
                del self.index[key]
                self.size -= 1
        
        return None
    
    def _put_internal(self, key: Tuple, data: np.ndarray) -> None:
        filename = self._key_to_filename(key)
        filepath = os.path.join(self.cache_dir, filename)
        
        # Save to disk
        np.save(filepath, data)
        
        if key not in self.index:
            self.size += 1
        
        self.lru_counter += 1
        self.index[key] = (filename, self.lru_counter)
    
    def _evict_one(self) -> None:
        if not self.index:
            return
        
        # Find LRU item
        lru_key = min(self.index.keys(), key=lambda k: self.index[k][1])
        filename, _ = self.index[lru_key]
        filepath = os.path.join(self.cache_dir, filename)
        
        # Load data for demotion
        if os.path.exists(filepath):
            data = np.load(filepath)
            
            # Demote to next tier if available
            if self.next_tier is not None:
                self.next_tier.put(lru_key, data)
                # Don't acquire lock here - called from within lock
                self.demotions += 1
            
            # Remove file
            os.remove(filepath)
        
        # Remove from index
        del self.index[lru_key]
        self.size -= 1
    
    def _contains(self, key: Tuple) -> bool:
        return key in self.index
    
    def clear(self) -> None:
        super().clear()
        with self.lock:
            # Clean up files
            for key in list(self.index.keys()):
                filename, _ = self.index[key]
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            self.index.clear()
            self.lru_counter = 0
    
    def __del__(self):
        """Clean up cache directory if we created it."""
        if self.owns_cache_dir and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)


class NetworkTier(StorageTier):
    """
    Network tier - slowest, largest capacity.
    Simulates network storage with configurable latency.
    In production, this would connect to cloud storage (S3, etc.)
    """
    
    def __init__(self, capacity_tiles: int, cache_dir: Optional[str] = None,
                 latency_ms: float = 50.0):
        super().__init__("Network", capacity_tiles, None)  # No next tier
        
        # Create cache directory
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="network_tier_")
            self.owns_cache_dir = True
        else:
            self.cache_dir = cache_dir
            self.owns_cache_dir = False
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.index = {}  # key -> filename
        self.latency_seconds = latency_ms / 1000.0
    
    def _key_to_filename(self, key: Tuple) -> str:
        """Convert a tile key to a unique filename."""
        filepath, r_start, c_start = key
        base = os.path.basename(filepath)
        name = os.path.splitext(base)[0]
        return f"{name}_r{r_start}_c{c_start}.npy"
    
    def _get_internal(self, key: Tuple) -> Optional[np.ndarray]:
        if key in self.index:
            # Simulate network latency
            time.sleep(self.latency_seconds)
            
            filename = self.index[key]
            filepath = os.path.join(self.cache_dir, filename)
            
            if os.path.exists(filepath):
                data = np.load(filepath)
                return data
            else:
                # File was deleted externally, clean up index
                del self.index[key]
                self.size -= 1
        
        return None
    
    def _put_internal(self, key: Tuple, data: np.ndarray) -> None:
        # Simulate network latency
        time.sleep(self.latency_seconds)
        
        filename = self._key_to_filename(key)
        filepath = os.path.join(self.cache_dir, filename)
        
        # Save to disk (simulating network upload)
        np.save(filepath, data)
        
        if key not in self.index:
            self.size += 1
        
        self.index[key] = filename
    
    def _evict_one(self) -> None:
        if not self.index:
            return
        
        # Simple eviction - remove first item
        # In production, could use more sophisticated policy
        evict_key = next(iter(self.index.keys()))
        filename = self.index[evict_key]
        filepath = os.path.join(self.cache_dir, filename)
        
        # Remove file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Remove from index
        del self.index[evict_key]
        self.size -= 1
    
    def _contains(self, key: Tuple) -> bool:
        return key in self.index
    
    def clear(self) -> None:
        super().clear()
        with self.lock:
            # Clean up files
            for key in list(self.index.keys()):
                filename = self.index[key]
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            self.index.clear()
    
    def __del__(self):
        """Clean up cache directory if we created it."""
        if self.owns_cache_dir and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
