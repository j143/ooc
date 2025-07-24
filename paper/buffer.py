import collections
import threading
from .core import PaperMatrix
import time

import numpy as np

TILE_SIZE = 1024

class BufferManager:
    """
    Manages on in-memory cache of matrix tiles to reduce disk I/O.
    Thread-safe LRU policy
    """
    def __init__(self, max_cache_size_tiles: int = 64, io_trace: list = None):
        self.max_size = max_cache_size_tiles
        self.io_trace = io_trace
        self.trace_pos = 0
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
        lru_key, _ = self.lru_tracker.popitem(last=False)
        del self.cache[lru_key]
        # log the eviction event
        self.event_log.append((time.perf_counter(), 'EVICT', lru_key, len(self.cache)))
    
    def _evict_optimal(self):
        """
        Belady's algorithm
        Evicts the tile whose next use is furthest in the future according to the I/O trace.
        """
        future_trace = self.io_trace[self.trace_pos:]
        distances = {}
        for tile_key in self.cache:
            try:
                # Find the next distance when this tile is needed
                distance = future_trace.index(tile_key)
                distances[tile_key] = distance
            except ValueError:
                # This tail is never used again, candidate for eviction!
                distances[tile_key] = float('inf')
        
        # Evict the tile with the largest distance to its next use
        evict_key = max(distances, key=distances.get)
        del self.cache[evict_key]
        del self.lru_tracker[evict_key]
        self.event_log.append((time.perf_counter(), 'EVICT', evict_key, len(self.cache)))

    def get_tile(self, matrix: PaperMatrix, r_start: int, c_start: int):
        """
        Fetches a tile, use the cache if possible, or loading from disk.
        """
        tile_key = (matrix.filepath, r_start, c_start)
        current_time = time.perf_counter()

        with self.lock:
            # Increment trace position regardless of hit / miss
            if self.io_trace:
                self.trace_pos += 1

            if tile_key in self.cache:
                # --- cache hit ---
                # Mark as most recently used
                self.lru_tracker.move_to_end(tile_key)
                self.event_log.append((current_time, 'HIT', tile_key, len(self.cache)))
                return self.cache[tile_key]
        
        # --- Cache miss ---
        with self.lock:
            self.event_log.append((current_time, 'MISS', tile_key, len(self.cache)))
        
        # load from disk (outside the lock to allow concurrent I/O);
        # r_end = min(r_start + TILE_SIZE, matrix.shape[0])
        # c_end = min(c_start + TILE_SIZE, matrix.shape[1])

        # Perform slicing in two steps as multi-dimensional slicing on memmap is not implemented
        # tile_data = matrix.data[r_start:r_end, c_start:c_end]
        tile_data = matrix.get_tile(r_start, c_start)
        # row_slice = matrix.data[r_start:r_end]
        # tile_data = row_slice[:, 0:(c_end - c_start)] # is this 0:(c_end - c_start) or c_start:c_end?

        with self.lock:
            if tile_key in self.cache:
                self.lru_tracker.move_to_end(tile_key)
                return self.cache[tile_key]

            # Add the new tile to the cache
            self.cache[tile_key] = tile_data
            self.lru_tracker[tile_key] = None
            self.lru_tracker.move_to_end(tile_key)

            # If cache is over capacity, evict the least recently used tile
            if len(self.cache) > self.max_size:
                if self.io_trace:
                    self._evict_optimal()
                else:
                    self._evict_lru()
            
        return tile_data

    def get_log(self):
        """
        Returns the event log of cache accesses.
        """
        return self.event_log
