import collections
import threading
import os
from .core import PaperMatrix
import time

import numpy as np
from .config import TILE_SIZE, DEFAULT_CACHE_SIZE_TILES

class BufferManager:
    """
    Manages on in-memory cache of matrix tiles to reduce disk I/O.
    Thread-safe LRU policy
    """
    def __init__(self, max_cache_size_tiles: int = DEFAULT_CACHE_SIZE_TILES, io_trace: list = None):
        self.max_size = max_cache_size_tiles
        self.io_trace = io_trace
        # Remove self.trace_pos - now passed as parameter for stateless operation
        self.cache = {} # Stores tile data: (matrix_path, r, c) -> numpy_tile
        self.lru_tracker = collections.OrderedDict() # Keeps track of usage order
        self.lock = threading.Lock() # Ensure thread-safe access to the cache

        # -- New Logging Attribute --
        # (timestamp, event_type, tile_key)
        self.event_log = []
    
    def _evict_lru(self):
        """
        Evicts the least recently used tile
        """
        if not self.lru_tracker:
            return
            
        lru_key, _ = self.lru_tracker.popitem(last=False)
        if lru_key in self.cache:
            del self.cache[lru_key]
        # log the eviction event
        self.event_log.append((time.perf_counter(), 'EVICT', lru_key, len(self.cache)))
    
    def _evict_optimal(self, trace_pos: int):
        """
        Belady's algorithm
        Evicts the tile whose next use is furthest in the future according to the I/O trace.
        Args:
            trace_pos: Current position in the I/O trace (stateless)
        """
        if not self.cache:
            return
            
        # Bounds check for trace position
        if trace_pos >= len(self.io_trace):
            # If trace is exhausted, fall back to LRU
            self._evict_lru()
            return
            
        future_trace = self.io_trace[trace_pos:]
        distances = {}
        for tile_key in self.cache:
            try:
                # Convert cache key to match io_trace format (basename instead of full path)
                trace_key = (os.path.basename(tile_key[0]), tile_key[1], tile_key[2])
                # Find the next distance when this tile is needed
                distance = future_trace.index(trace_key)
                distances[tile_key] = distance
            except ValueError:
                # This tile is never used again, candidate for eviction!
                distances[tile_key] = float('inf')
        
        # Evict the tile with the largest distance to its next use
        evict_key = max(distances, key=distances.get)
        del self.cache[evict_key]
        # Fix: Remove from LRU tracker consistently
        if evict_key in self.lru_tracker:
            del self.lru_tracker[evict_key]
        self.event_log.append((time.perf_counter(), 'EVICT', evict_key, len(self.cache)))

    def get_tile(self, matrix: PaperMatrix, r_start: int, c_start: int, trace_pos: int = 0):
        """
        Fetches a tile, use the cache if possible, or loading from disk.
        Args:
            matrix: The matrix to fetch from
            r_start: Row start position for the tile
            c_start: Column start position for the tile
            trace_pos: Current position in the I/O trace (stateless)
        """
        tile_key = (matrix.filepath, r_start, c_start)
        current_time = time.perf_counter()

        with self.lock:
            if tile_key in self.cache:
                # --- cache hit ---
                # Mark as most recently used
                self.lru_tracker.move_to_end(tile_key)
                self.event_log.append((current_time, 'HIT', tile_key, len(self.cache)))
                
                # Note: trace_pos is not incremented here as it's managed by caller
                return self.cache[tile_key]
        
        # --- Cache miss ---
        # Load from disk (outside the lock to allow concurrent I/O)
        tile_data = matrix.get_tile(r_start, c_start)

        with self.lock:
            # Double-check if tile was added by another thread while we were loading
            if tile_key in self.cache:
                self.lru_tracker.move_to_end(tile_key)
                return self.cache[tile_key]

            # Log the cache miss
            self.event_log.append((current_time, 'MISS', tile_key, len(self.cache)))

            # Check if we need to evict before adding (fix: check before, not after)
            if len(self.cache) >= self.max_size:
                if self.io_trace:
                    self._evict_optimal(trace_pos)
                else:
                    self._evict_lru()

            # Add the new tile to the cache
            self.cache[tile_key] = tile_data
            self.lru_tracker[tile_key] = None
            self.lru_tracker.move_to_end(tile_key)
            
            # Note: trace_pos is not incremented here as it's managed by caller
            
        return tile_data

    def get_log(self):
        """
        Returns the event log of cache accesses.
        """
        return self.event_log
