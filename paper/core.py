
# --- Purpose: Handles the physical on-disk representation of a matrix. ---

import numpy as np
import os
from .config import TILE_SIZE

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

        # Approach 1
        # The original slicing approach was problematic due to how memmap handles multi-dimensional slicing.
        # # 1. Slice the rows, which returns a view
        # row_view = self.data[r_start:r_end]
        # # 2. Immediately copy this view into a new, concrete in-memory array
        # in_memory_rows = np.array(row_view, copy=True)  # Ensure we have a copy in memory
        # # 3. Now, slice the columns from the clean, in-memory array
        # tile = in_memory_rows[:, 0:(c_end - c_start)]
        # return tile
        # r_end = min(r_start + TILE_SIZE, self.shape[0])
        # c_end = min(c_start + TILE_SIZE, self.shape[1])

        # Approach 2
        # This approach avoids the multi-dimensional slicing issue by reading the tile row by row.
        # # Create the destination tile array with the final, correct shape.
        # tile_shape = (r_end - r_start, c_end - c_start)
        # tile = np.empty(tile_shape, dtype=self.dtype)

        # # Read the data row by row from the memmap object into the tile.
        # # This avoids creating a large intermediate copy of all rows.
        # for i in range(tile_shape[0]):
        #     # Get a view of a single row from the memmap
        #     row_view = self.data[r_start + i]
        #     # Slice the required columns from that row and place it in our tile
        #     tile[i, :] = row_view[c_start:c_end]

        # Approach 3
        # This line performs the 2D slice and immediately copies the data into a 
        # new concrete in-memory array within Numpy's optimized C code.
        tile = np.array(self.data[r_start:r_end, c_start:c_end], copy=True)
            
        return tile


    def close(self):
        """Explicitly closes the underlying memory-mapped file handle."""
        
        if self.data is not None and self.data._mmap is not None:
            self.data._mmap.close()
    
    def __repr__(self):
        return f"PaperMatrix(path='{self.filepath}', shape={self.shape}, dtype={self.dtype.name})"
