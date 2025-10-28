"""
Operators module - provides high-level API for out-of-core operations.

This module implements the OOCMatrix wrapper that provides a NumPy-like API
for out-of-core matrix operations, focusing on orchestration rather than
reimplementing mathematical operations.
"""

import numpy as np
import os
from typing import Callable, Iterator, Tuple, Optional, Any
from .core import PaperMatrix
from .plan import Plan, EagerNode
from .config import TILE_SIZE


class OOCMatrix:
    """
    Out-of-core matrix wrapper that provides a NumPy-like API.
    
    This class wraps existing NumPy/SciPy operations with lazy evaluation,
    block-wise processing, and streaming constructs. It focuses on smart
    block loading, buffer management, and iteration control while delegating
    actual mathematical operations to optimized libraries.
    
    Example:
        >>> A = OOCMatrix('fileA.h5', shape=(10_000_000, 1000))
        >>> B = OOCMatrix('fileB.h5', shape=(1000, 1000))
        >>> C = A.matmul(B, op=lambda a, b: np.dot(a, b))
        >>> for block, idx in C.iterate_blocks():
        ...     process(block)
    """
    
    def __init__(self, filepath: str, shape: Tuple[int, int], 
                 dtype=np.float32, mode='r', create: bool = False):
        """
        Initialize an OOCMatrix.
        
        Args:
            filepath: Path to the matrix file on disk
            shape: Shape of the matrix (rows, cols)
            dtype: Data type of the matrix elements
            mode: File access mode ('r', 'w+', etc.)
            create: If True, create a new empty file
        """
        self.filepath = filepath
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.mode = mode
        
        # Create file if requested
        if create and not os.path.exists(filepath):
            self._create_empty_file()
        
        # Wrap the underlying PaperMatrix
        self._matrix = PaperMatrix(filepath, shape, dtype=dtype, mode=mode)
        
        # Create a lazy Plan wrapper for this matrix
        self._plan = Plan(EagerNode(self._matrix))
    
    def _create_empty_file(self):
        """Create an empty file of the correct size."""
        with open(self.filepath, "wb") as f:
            file_size = self.shape[0] * self.shape[1] * self.dtype.itemsize
            if file_size > 0:
                f.seek(file_size - 1)
                f.write(b'\0')
    
    @property
    def plan(self) -> Plan:
        """Get the underlying lazy evaluation plan."""
        return self._plan
    
    @property
    def matrix(self) -> PaperMatrix:
        """Get the underlying PaperMatrix."""
        return self._matrix
    
    def iterate_blocks(self, block_size: Optional[int] = None) -> Iterator[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Iterate over blocks of the matrix.
        
        This allows downstream systems to consume result blocks without
        loading the entire matrix into memory.
        
        Args:
            block_size: Size of each block (defaults to TILE_SIZE)
        
        Yields:
            Tuple of (block_data, (row_start, col_start))
        """
        if block_size is None:
            block_size = TILE_SIZE
        
        rows, cols = self.shape
        
        for r_start in range(0, rows, block_size):
            r_end = min(r_start + block_size, rows)
            for c_start in range(0, cols, block_size):
                c_end = min(c_start + block_size, cols)
                
                # Load the block from disk
                block = self._matrix.get_tile(r_start, c_start)
                
                yield block, (r_start, c_start)
    
    def blockwise_apply(self, op: Callable[[np.ndarray], np.ndarray], 
                       output_path: Optional[str] = None) -> 'OOCMatrix':
        """
        Apply a function to each block of the matrix.
        
        This method enables element-wise transformations using existing NumPy
        operations without loading the entire matrix into memory.
        
        Args:
            op: Function to apply to each block (takes ndarray, returns ndarray)
            output_path: Path for the output matrix (if None, creates temp file)
        
        Returns:
            New OOCMatrix with the operation applied
        
        Example:
            >>> A = OOCMatrix('input.bin', shape=(10000, 1000))
            >>> # Normalize using NumPy operations on each block
            >>> A_norm = A.blockwise_apply(lambda x: (x - x.mean()) / x.std())
        """
        if output_path is None:
            output_path = self.filepath + ".blockwise_apply.tmp"
        
        # Create output matrix
        result = OOCMatrix(output_path, self.shape, dtype=self.dtype, 
                          mode='w+', create=True)
        
        # Process each block
        for block, (r_start, c_start) in self.iterate_blocks():
            # Apply the operation using existing NumPy/library code
            transformed_block = op(block)
            
            # Write the result
            r_end = r_start + block.shape[0]
            c_end = c_start + block.shape[1]
            result._matrix.data[r_start:r_end, c_start:c_end] = transformed_block
        
        result._matrix.data.flush()
        return result
    
    def blockwise_reduce(self, op: Callable[[np.ndarray], Any], 
                        combine_op: Optional[Callable[[Any, Any], Any]] = None) -> Any:
        """
        Apply a reduction operation across all blocks.
        
        This enables operations like sum, mean, max, etc. using existing
        NumPy reduction functions on a per-block basis.
        
        Args:
            op: Reduction function to apply to each block
            combine_op: Function to combine results from different blocks
                       (if None, uses addition for combining)
        
        Returns:
            Reduced value
        
        Example:
            >>> A = OOCMatrix('input.bin', shape=(10000, 1000))
            >>> mean = A.blockwise_reduce(np.mean)
            >>> total_sum = A.blockwise_reduce(np.sum)
        """
        if combine_op is None:
            # Default to addition for combining results
            combine_op = lambda x, y: x + y
        
        result = None
        count = 0
        
        for block, _ in self.iterate_blocks():
            block_result = op(block)
            
            if result is None:
                result = block_result
            else:
                result = combine_op(result, block_result)
            
            count += 1
        
        return result
    
    def matmul(self, other: 'OOCMatrix', 
               op: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
               output_path: Optional[str] = None) -> 'OOCMatrix':
        """
        Perform matrix multiplication using block-wise processing.
        
        This method leverages existing optimized libraries (NumPy, etc.) for
        in-block operations while handling the orchestration of large matrices.
        
        Args:
            other: The other matrix to multiply with
            op: Operation to use for block multiplication (defaults to np.dot)
            output_path: Path for output matrix
        
        Returns:
            Result matrix
        
        Example:
            >>> A = OOCMatrix('A.bin', shape=(10000, 1000))
            >>> B = OOCMatrix('B.bin', shape=(1000, 500))
            >>> C = A.matmul(B, op=np.dot)
        """
        if op is None:
            op = np.dot
        
        # Use the underlying Plan infrastructure for lazy evaluation
        result_plan = self._plan @ other._plan
        
        if output_path is None:
            output_path = self.filepath + ".matmul.result.tmp"
        
        # Execute the plan
        result_matrix, buffer_manager = result_plan.compute(output_path)
        
        # Wrap in OOCMatrix
        return OOCMatrix(output_path, result_matrix.shape, 
                        dtype=result_matrix.dtype, mode='r')
    
    def sum(self) -> float:
        """
        Compute the sum of all elements.
        
        Returns:
            Sum of all matrix elements
        """
        return self.blockwise_reduce(np.sum)
    
    def mean(self) -> float:
        """
        Compute the mean of all elements.
        
        Returns:
            Mean of all matrix elements
        """
        total_sum = self.sum()
        total_elements = self.shape[0] * self.shape[1]
        return total_sum / total_elements
    
    def std(self) -> float:
        """
        Compute the standard deviation of all elements.
        
        Returns:
            Standard deviation of all matrix elements
        """
        mean_val = self.mean()
        
        # Compute variance using blockwise operations
        def variance_block(block):
            return np.sum((block - mean_val) ** 2)
        
        total_var = self.blockwise_reduce(variance_block)
        total_elements = self.shape[0] * self.shape[1]
        
        return np.sqrt(total_var / total_elements)
    
    def max(self) -> float:
        """
        Compute the maximum element.
        
        Returns:
            Maximum element value
        """
        return self.blockwise_reduce(np.max, combine_op=max)
    
    def min(self) -> float:
        """
        Compute the minimum element.
        
        Returns:
            Minimum element value
        """
        return self.blockwise_reduce(np.min, combine_op=min)
    
    def __add__(self, other: 'OOCMatrix') -> 'OOCMatrix':
        """
        Add two matrices using lazy evaluation.
        
        Args:
            other: Matrix to add
        
        Returns:
            New OOCMatrix representing the lazy addition
        """
        # Use the underlying Plan infrastructure
        result_plan = self._plan + other._plan
        
        # Create a new OOCMatrix wrapper with the combined plan
        # Note: This doesn't execute yet, maintaining lazy evaluation
        result = OOCMatrix.__new__(OOCMatrix)
        result.filepath = self.filepath + ".add.lazy"
        result.shape = self.shape
        result.dtype = self.dtype
        result.mode = 'r'
        result._plan = result_plan
        result._matrix = None  # Will be created on compute
        
        return result
    
    def __mul__(self, scalar: float) -> 'OOCMatrix':
        """
        Multiply matrix by a scalar using lazy evaluation.
        
        Args:
            scalar: Scalar value to multiply by
        
        Returns:
            New OOCMatrix representing the lazy multiplication
        """
        result_plan = self._plan * scalar
        
        result = OOCMatrix.__new__(OOCMatrix)
        result.filepath = self.filepath + ".mul.lazy"
        result.shape = self.shape
        result.dtype = self.dtype
        result.mode = 'r'
        result._plan = result_plan
        result._matrix = None
        
        return result
    
    def __matmul__(self, other: 'OOCMatrix') -> 'OOCMatrix':
        """
        Matrix multiply using lazy evaluation (@ operator).
        
        Args:
            other: Matrix to multiply with
        
        Returns:
            New OOCMatrix representing the lazy multiplication
        """
        result_plan = self._plan @ other._plan
        
        result = OOCMatrix.__new__(OOCMatrix)
        result.filepath = self.filepath + ".matmul.lazy"
        result.shape = (self.shape[0], other.shape[1])
        result.dtype = self.dtype
        result.mode = 'r'
        result._plan = result_plan
        result._matrix = None
        
        return result
    
    def compute(self, output_path: str, cache_size_tiles: Optional[int] = None) -> 'OOCMatrix':
        """
        Execute the lazy computation plan and materialize the result.
        
        Args:
            output_path: Path where the result should be stored
            cache_size_tiles: Size of cache in tiles
        
        Returns:
            OOCMatrix with the computed result
        """
        result_matrix, buffer_manager = self._plan.compute(output_path, cache_size_tiles)
        
        return OOCMatrix(output_path, result_matrix.shape, 
                        dtype=result_matrix.dtype, mode='r')
    
    def close(self):
        """Close the underlying matrix file."""
        if self._matrix is not None:
            self._matrix.close()
    
    def __repr__(self):
        return f"OOCMatrix(path='{self.filepath}', shape={self.shape}, dtype={self.dtype.name})"
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
