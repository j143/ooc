import collections
import threading
from .core import PaperMatrix

TILE_SIZE = 1000

class BufferManager:
    """
    Manages on in-memory cache of matrix tiles to reduce disk I/O.
    Thread-safe LRU policy
    """
    def __init__(self, max_cache_size_tiles: int = 64):
        self.max_size = max_cache_size_tiles
        self.cache = {} # Stores tile data: (matrix_path, r, c) -> numpy_tile
        self.lru_tracker = collections.OrderedDict() # Keeps track of usage order
        self.lock = threading.Lock() # Ensure thread-safe access to the cache

    def get_tile(self, matrix: PaperMatrix, r_start: int, c_start: int):
        """
        Fetches a tile, use the cache if possible, or loading from disk.
        """
        tile_key = (matrix.filename, r_start, c_start)

        with self.lock:
            if tile_key in self.cache:
                # --- cache hit ---
                # Mark as most recently used
                self.lru_tracker.move_to_end(tile_key)
                return self.cache[tile_key]
        
        # --- Cache miss ---
        # load from disk (outside the lock to allow concurrent I/O);
        r_end = min(r_start + TILE_SIZE, matrix.shape[0])
        c_end = min(c_start + TILE_SIZE, matrix.shape[1])
        tile_data = matrix.data[r_start:r_end, c_start:c_end]

        with self.lock:
            # Add the new tile to the cache
            self.cache[tile_key] = tile_data
            self.lru_tracker[tile_key] = None
            self.lru_tracker.move_to_end(tile_key)

            # If cache is over capacity, evict the least recently used tile
            if len(self.cache) > self.max_size:
                lru_key, _ = self.lru_tracker.popitem(last=False)
                del self.cache[lru_key]
            
        return tile_data
