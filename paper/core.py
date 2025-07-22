
# --- Purpose: Handles the physical on-disk representation of a matrix. ---

import numpy as np
import os

TILE_SIZE = 1000  # Size of the tile for out-of-core operations

class PaperMatrix:
    """
    Represents a matrix stored on disk using a memory-mapped file.
    It holds metadata but doesn't load the whole file into RAM.
    """
    def __init__(self, filepath, shape, dtype=np.float32, mode='r'):
        self.filepath = filepath
        self.shape = shape
        self.dtype = np.dtype(dtype)
        
        # This is the core of our out-of-core access.
        # It links a NumPy array interface to the file on disk.
        # Create a memory-map to an array stored in a binary file on disk.
        # Memory-mapped files are used for accessing small segments of large files on disk,
        # without reading the entire file into memory. NumPy’s memmap’s are array-like objects.
        self.data = np.memmap(self.filepath, dtype=self.dtype, mode=mode, shape=self.shape)
        
    def get_tile(self, r_start, c_start):
        """
        Safely extracts a tile from the memory-mapped file.
        This method contains the workaround for the memmap slicing issue.
        """
        r_end = min(r_start + TILE_SIZE, self.shape[0])
        c_end = min(c_start + TILE_SIZE, self.shape[1])

        # 1. Slice the rows, which returns a view
        row_view = self.data[r_start:r_end]
        # 2. Immediately copy this view into a new, concrete in-memory array
        in_memory_rows = np.array(row_view, copy=True)  # Ensure we have a copy in memory
        # 3. Now, slice the columns from the clean, in-memory array
        tile = in_memory_rows[:, 0:(c_end - c_start)]
        return tile


    def close(self):
        """Explicitly closes the underlying memory-mapped file handle."""
        
        if self.data is not None and self.data._mmap is not None:
            self.data._mmap.close()
    
    def __repr__(self):
        return f"PaperMatrix(path='{self.filepath}', shape={self.shape}, dtype={self.dtype.name})"
