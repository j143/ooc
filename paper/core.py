
# --- Purpose: Handles the physical on-disk representation of a matrix. ---

import numpy as np
import os
import time


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
        
    def close(self):
        """Explicitly closes the underlying memory-mapped file handle."""
        
        if self.data is not None and self.data._mmap is not None:
            self.data._mmap.close()
    
    def __repr__(self):
        return f"PaperMatrix(path='{self.filepath}', shape={self.shape}, dtype={self.dtype.name})"
