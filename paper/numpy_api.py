"""
NumPy-compatible API Layer for Paper Framework

This module provides a NumPy-compatible interface for the Paper framework,
allowing users to leverage familiar NumPy operations while benefiting from
out-of-core matrix operations for datasets larger than memory.

The API mimics NumPy's interface but operates on disk-backed matrices through
the Paper framework's lazy evaluation and optimization capabilities.
"""

import numpy as np
import os
import tempfile
from typing import Union, Tuple, Optional
from .core import PaperMatrix
from .plan import Plan, EagerNode
from .config import TILE_SIZE


class ndarray:
    """
    NumPy-compatible array class for out-of-core matrix operations.
    
    This class provides a familiar NumPy-like interface while delegating
    operations to the Paper framework's lazy evaluation system.
    
    Attributes:
        shape: Tuple representing the dimensions of the array
        dtype: NumPy data type of the array elements
        ndim: Number of dimensions (always 2 for matrices)
        size: Total number of elements in the array
    """
    
    def __init__(self, data=None, filepath=None, shape=None, dtype=np.float32, mode='r'):
        """
        Initialize a Paper ndarray.
        
        Args:
            data: Initial data (numpy array or nested lists)
            filepath: Path to existing matrix file on disk
            shape: Shape tuple for the array
            dtype: Data type for array elements
            mode: File access mode ('r', 'w+', 'r+')
        """
        self.dtype = np.dtype(dtype)
        
        if filepath is not None:
            # Load from existing file
            if shape is None:
                raise ValueError("Shape must be provided when loading from filepath")
            self._matrix = PaperMatrix(filepath, shape, dtype=self.dtype, mode=mode)
            self._plan = Plan(EagerNode(self._matrix))
            self._filepath = filepath
            self._is_lazy = False
        elif data is not None:
            # Create from data
            data_array = np.asarray(data, dtype=self.dtype)
            if data_array.ndim != 2:
                raise ValueError("Paper arrays must be 2-dimensional matrices")
            
            # Create a temporary file to store the data
            fd, self._filepath = tempfile.mkstemp(suffix='.bin', prefix='paper_array_')
            os.close(fd)
            
            # Write data to file
            self._matrix = PaperMatrix(self._filepath, data_array.shape, dtype=self.dtype, mode='w+')
            self._matrix.data[:] = data_array
            self._matrix.data.flush()
            self._plan = Plan(EagerNode(self._matrix))
            self._is_lazy = False
        elif shape is not None:
            # Create empty array
            fd, self._filepath = tempfile.mkstemp(suffix='.bin', prefix='paper_array_')
            os.close(fd)
            
            self._matrix = PaperMatrix(self._filepath, shape, dtype=self.dtype, mode='w+')
            self._plan = Plan(EagerNode(self._matrix))
            self._is_lazy = False
        else:
            raise ValueError("Must provide either data, filepath, or shape")
    
    @classmethod
    def _from_plan(cls, plan: Plan, shape: Tuple[int, int], dtype=np.float32):
        """Internal constructor for lazy arrays from computation plans."""
        obj = cls.__new__(cls)
        obj._plan = plan
        obj._matrix = None
        obj._filepath = None
        obj.dtype = np.dtype(dtype)
        obj._is_lazy = True
        obj._shape = shape
        return obj
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the array."""
        if self._is_lazy:
            return self._shape
        return self._matrix.shape if self._matrix else self._plan.shape
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions (always 2 for matrices)."""
        return 2
    
    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return self.shape[0] * self.shape[1]
    
    @property
    def T(self) -> 'ndarray':
        """
        Return the transpose of the array.
        Note: Currently creates a copy. Future optimization: lazy transpose.
        """
        # For now, materialize and transpose
        result = self._materialize()
        transposed = result.T
        return array(transposed, dtype=self.dtype)
    
    def _materialize(self) -> np.ndarray:
        """
        Materialize the lazy computation into an actual NumPy array.
        Warning: This loads the entire array into memory.
        
        Internal method. Use to_numpy() for public API.
        """
        if self._is_lazy:
            # Compute the plan
            fd, temp_path = tempfile.mkstemp(suffix='.bin', prefix='paper_materialized_')
            os.close(fd)
            
            result_matrix, _ = self._plan.compute(temp_path)
            data = np.array(result_matrix.data, copy=True)
            result_matrix.close()
            os.unlink(temp_path)
            return data
        else:
            # Already materialized
            return np.array(self._matrix.data, copy=True)
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert the array to a NumPy array.
        
        This method materializes the array, loading it into memory.
        For lazy arrays, this executes the computation plan first.
        
        Returns:
            np.ndarray: A NumPy array containing the data
            
        Warning:
            This loads the entire array into memory and may fail for
            very large arrays that don't fit in RAM.
        
        Examples:
            >>> import paper.numpy_api as pnp
            >>> a = pnp.array([[1, 2], [3, 4]])
            >>> numpy_arr = a.to_numpy()
            >>> print(numpy_arr)
            [[1. 2.]
             [3. 4.]]
        """
        return self._materialize()
    
    def compute(self, output_path: Optional[str] = None, cache_size_tiles: Optional[int] = None):
        """
        Execute the lazy computation plan and return a materialized ndarray.
        
        Args:
            output_path: Optional path to save the result
            cache_size_tiles: Optional cache size for buffer manager
            
        Returns:
            ndarray: Materialized result
        """
        if not self._is_lazy:
            return self
        
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.bin', prefix='paper_computed_')
            os.close(fd)
        
        result_matrix, _ = self._plan.compute(output_path, cache_size_tiles)
        
        # Create a new ndarray from the result
        result = ndarray.__new__(ndarray)
        result._matrix = result_matrix
        result._filepath = output_path
        result._plan = Plan(EagerNode(result_matrix))
        result.dtype = self.dtype
        result._is_lazy = False
        return result
    
    def __add__(self, other: Union['ndarray', int, float]) -> 'ndarray':
        """Element-wise addition."""
        if isinstance(other, (int, float)):
            # Scalar addition - not yet optimized
            raise NotImplementedError("Scalar addition not yet implemented")
        
        if not isinstance(other, ndarray):
            raise TypeError(f"Unsupported operand type for +: 'ndarray' and '{type(other).__name__}'")
        
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        
        # Create lazy addition plan
        new_plan = self._plan + other._plan
        return ndarray._from_plan(new_plan, self.shape, self.dtype)
    
    def __mul__(self, other: Union['ndarray', int, float]) -> 'ndarray':
        """Element-wise multiplication or scalar multiplication."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            new_plan = self._plan * other
            return ndarray._from_plan(new_plan, self.shape, self.dtype)
        
        # Element-wise multiplication not yet implemented
        raise NotImplementedError("Element-wise multiplication with arrays not yet implemented")
    
    def __rmul__(self, other: Union[int, float]) -> 'ndarray':
        """Right scalar multiplication."""
        return self.__mul__(other)
    
    def __matmul__(self, other: 'ndarray') -> 'ndarray':
        """Matrix multiplication."""
        if not isinstance(other, ndarray):
            raise TypeError(f"Unsupported operand type for @: 'ndarray' and '{type(other).__name__}'")
        
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Inner dimensions must match: {self.shape} @ {other.shape}")
        
        # Create lazy matmul plan
        new_plan = self._plan @ other._plan
        result_shape = (self.shape[0], other.shape[1])
        return ndarray._from_plan(new_plan, result_shape, self.dtype)
    
    def __repr__(self) -> str:
        """String representation."""
        if self._is_lazy:
            return f"ndarray(shape={self.shape}, dtype={self.dtype.name}, lazy=True)"
        return f"ndarray(shape={self.shape}, dtype={self.dtype.name})"
    
    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, '_matrix') and self._matrix is not None:
            try:
                self._matrix.close()
            except:
                pass
        
        # Clean up temporary files
        if hasattr(self, '_filepath') and self._filepath and os.path.exists(self._filepath):
            if 'paper_array_' in self._filepath or 'paper_computed_' in self._filepath:
                try:
                    os.unlink(self._filepath)
                except:
                    pass


def array(data, dtype=np.float32) -> ndarray:
    """
    Create a Paper array from existing data.
    
    Args:
        data: Array-like data (list, tuple, or numpy array)
        dtype: Data type for the array elements
        
    Returns:
        ndarray: A Paper ndarray object
        
    Examples:
        >>> import paper.numpy_api as pnp
        >>> a = pnp.array([[1, 2], [3, 4]])
        >>> b = pnp.array([[5, 6], [7, 8]])
        >>> c = a + b  # Lazy operation
        >>> result = c.compute()  # Execute
    """
    return ndarray(data=data, dtype=dtype)


def zeros(shape: Tuple[int, int], dtype=np.float32) -> ndarray:
    """
    Create an array filled with zeros.
    
    Args:
        shape: Shape of the array (rows, cols)
        dtype: Data type
        
    Returns:
        ndarray: Array filled with zeros
    """
    arr = ndarray(shape=shape, dtype=dtype)
    arr._matrix.data[:] = 0
    arr._matrix.data.flush()
    return arr


def ones(shape: Tuple[int, int], dtype=np.float32) -> ndarray:
    """
    Create an array filled with ones.
    
    Args:
        shape: Shape of the array (rows, cols)
        dtype: Data type
        
    Returns:
        ndarray: Array filled with ones
    """
    arr = ndarray(shape=shape, dtype=dtype)
    arr._matrix.data[:] = 1
    arr._matrix.data.flush()
    return arr


def random_rand(shape: Tuple[int, int], dtype=np.float32) -> ndarray:
    """
    Create an array with random values from [0, 1).
    
    Args:
        shape: Shape of the array (rows, cols)
        dtype: Data type
        
    Returns:
        ndarray: Array with random values
    """
    arr = ndarray(shape=shape, dtype=dtype)
    
    # Fill tile by tile to avoid loading entire matrix into memory
    for r_start in range(0, shape[0], TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, shape[0])
        for c_start in range(0, shape[1], TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, shape[1])
            tile_shape = (r_end - r_start, c_end - c_start)
            random_tile = np.random.rand(*tile_shape).astype(dtype)
            arr._matrix.data[r_start:r_end, c_start:c_end] = random_tile
    
    arr._matrix.data.flush()
    return arr


def eye(n: int, dtype=np.float32) -> ndarray:
    """
    Create a 2-D identity matrix.
    
    Args:
        n: Number of rows and columns
        dtype: Data type
        
    Returns:
        ndarray: Identity matrix
    """
    arr = zeros((n, n), dtype=dtype)
    
    # Set diagonal to 1
    for i in range(n):
        arr._matrix.data[i, i] = 1
    
    arr._matrix.data.flush()
    return arr


def load(filepath: str, shape: Tuple[int, int], dtype=np.float32) -> ndarray:
    """
    Load a matrix from a binary file.
    
    Args:
        filepath: Path to the binary matrix file
        shape: Shape of the matrix
        dtype: Data type
        
    Returns:
        ndarray: Loaded array
    """
    return ndarray(filepath=filepath, shape=shape, dtype=dtype, mode='r')


def save(filepath: str, arr: ndarray):
    """
    Save an array to a binary file.
    
    Args:
        filepath: Path to save the file
        arr: Array to save
    """
    if arr._is_lazy:
        # Compute first if lazy
        arr = arr.compute(output_path=filepath)
    else:
        # Copy to new location if needed
        if arr._filepath != filepath:
            # Simple file copy
            import shutil
            shutil.copy(arr._filepath, filepath)


# Expose common NumPy functions
dot = lambda a, b: a @ b  # Matrix multiplication
add = lambda a, b: a + b  # Element-wise addition
multiply = lambda a, b: a * b  # Scalar or element-wise multiplication
