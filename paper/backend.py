
# --- Purpose: Contains the high-performance execution kernels. ---

import numpy as np
import os
from .core import PaperMatrix
from concurrent.futures import ThreadPoolExecutor

from .buffer import BufferManager, TILE_SIZE

buffer_manager = BufferManager(max_cache_size_tiles=64)

def _create_empty_file(filepath, shape, dtype):
    """Helper to create an empty file of the correct size for writing."""
    with open(filepath, "wb") as f:
        file_size = shape[0] * shape[1] * np.dtype(dtype).itemsize
        if file_size > 0:
            f.seek(file_size - 1)
            f.write(b'\0')

def add(A: PaperMatrix, B: PaperMatrix, output_path: str, buffer_manager: BufferManager | None) -> PaperMatrix:
    """
    Performs out-of-core matrix addition: C = A + B using the provided 
    """
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape for addition.")
    
    # C = PaperMatrix(output_path, A.shape, mode='w+')
    _create_empty_file(output_path, A.shape, A.dtype)
    
    # loop now uses the buffer manager for all data reads
    rows, cols = A.shape
    # Iterate through the matrices tile by tile
    with open(output_path, "r+b") as f_out:
        for r_start in range(0, rows, TILE_SIZE):
            r_end = min(r_start + TILE_SIZE, rows)
            for c_start in range(0, cols, TILE_SIZE):
                c_end = min(c_start + TILE_SIZE, cols)
                
                # 1. Read tiles from A and B into memory
                # tile_A = A.data[r_start:r_end, c_start:c_end]
                # tile_B = B.data[r_start:r_end, c_start:c_end]
                if buffer_manager:
                    print(" - Backend: Executing buffered 'add' kernel" )
                    tile_A = buffer_manager.get_tile(A, r_start, c_start)
                    tile_B = buffer_manager.get_tile(B, r_start, c_start)
                else:
                    tile_A = A.data[r_start:r_end, c_start:c_end]
                    tile_B = B.data[r_start:r_end, c_start:c_end]
                
                # 2. Compute the result in memory
                # tile_C = tile_A + tile_B
                
                # 3. Write the resulting tile to C's file
                # C.data[r_start:r_end, c_start:c_end] = tile_C
        
        
                result_tile = (tile_A + tile_B).astype(A.dtype)

                # Write the tile row by row to handle non-contiguous writes
                for i in range(result_tile.shape[0]):
                    row_offset = ((r_start + i) * A.shape[1] + c_start) * A.dtype.itemsize
                    f_out.seek(row_offset)
                    f_out.write(result_tile[i, :].tobytes())
    
    return PaperMatrix(output_path, A.shape, mode='r')

def multiply(A: PaperMatrix, B: PaperMatrix, output_path: str, buffer_manager: BufferManager | None) -> PaperMatrix:
    """Performs out-of-core tiled matrix multiplication: C = A @ B."""
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must match for multiplication.")
    
    C_shape = (A.shape[0], B.shape[1])
    C = PaperMatrix(output_path, C_shape, mode='w+')

    print("Performing eager multiplication...")
    rows_A, K, cols_B = A.shape[0], A.shape[1], B.shape[1]

    # Loop over the tiles of the output matrix C
    for r_start in range(0, rows_A, TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, rows_A)
        for c_start in range(0, cols_B, TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, cols_B)

            # 1. Initialize an in-memory tile for accumulating the result
            tile_C = np.zeros((r_end - r_start, c_end - c_start), dtype=C.dtype)

            # 2. Loop through the inner dimension k
            for k_start in range(0, K, TILE_SIZE):
                k_end = min(k_start + TILE_SIZE, K)

                # load the corresponding tiles from A and B
                # tile_A = A.data[r_start:r_end, k_start:k_end]
                # tile_B = B.data[k_start:k_end, c_start:c_end]
                if buffer_manager:
                    print(" - Backend: Executing buffered 'multiply' kernel" )
                    tile_A = buffer_manager.get_tile(A, r_start, k_start)
                    tile_B = buffer_manager.get_tile(B, k_start, c_start)
                else:
                    tile_A = A.data[r_start:r_end, k_start:k_end]
                    tile_B = B.data[k_start:k_end, c_start:c_end]

                #3. Perform in-memory multiplication and accumulate
                tile_C += tile_A @ tile_B
            
            # 4. write the final computed tile to disk
            C.data[r_start:r_end, c_start:c_end] = tile_C
    
    C.data.flush()
    print("Multiplication complete.")
    return C

# New parallel kernel

def _process_fused_tile(A, B, scalar, r_start, r_end, c_start, c_end, buffer_manager: BufferManager):
    """Helper function to process a single tile. This is what each thread runs."""
    
    if buffer_manager:
        tile_A = buffer_manager.get_tile(A, r_start, c_start)
        tile_B = buffer_manager.get_tile(B, r_start, c_start)
    else:
        tile_A = A.data[r_start:r_end, c_start:c_end]
        tile_B = B.data[r_start:r_end, c_start:c_end]

    fused_result_tile = (tile_A + tile_B) * scalar
    return r_start, c_start, fused_result_tile

def execute_fused_add_multiply(A: PaperMatrix, B: PaperMatrix, scalar: float, output_path: str, buffer_manager: BufferManager | None) -> PaperMatrix:
    """
    Performs the fused (A + B) * scalar operation in parallel.
    """
    C = PaperMatrix(output_path, A.shape, mode='w+')
    rows, cols = A.shape
    
    # Use a ThreadPoolExecutor to manage a pool of worker threads.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        # Submit all tile-processing tasks to the thread pool.
        for r_start in range(0, rows, TILE_SIZE):
            r_end = min(r_start + TILE_SIZE, rows)
            for c_start in range(0, cols, TILE_SIZE):
                c_end = min(c_start + TILE_SIZE, cols)
                # submit() schedules the function to run and returns a Future object.
                future = executor.submit(
                    _process_fused_tile,
                    A, B, scalar,
                    r_start, r_end, c_start, c_end,
                    buffer_manager
                )
                futures.append(future)

        # Retrieve results as they complete.
        for future in futures:
            # future.result() waits for the task to finish and gets its return value.
            r_start, c_start, result_tile = future.result()
            r_end = r_start + result_tile.shape[0]
            c_end = c_start + result_tile.shape[1]
            # Write the computed tile to the correct location in the output file.
            C.data[r_start:r_end, c_start:c_end] = result_tile
    
    C.data.flush()
    return C

def execute_fused_matmul_scalar(A: PaperMatrix, B: PaperMatrix, scalar: float, output_path: str) -> PaperMatrix:
    """
    Performs the fused (A @ B) * scalar operation in a single pass
    without creating a temporary matrix for (A @ B).
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must match for multiplication.")
    
    C_shape = (A.shape[0], B.shape[1])
    C = PaperMatrix(output_path, C_shape, mode='w+')
    
    rows_A, K, cols_B = A.shape[0], A.shape[1], B.shape[1]
    
    # Loop over the tiles of the output matrix C
    for r_start in range(0, rows_A, TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, rows_A)
        for c_start in range(0, cols_B, TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, cols_B)
            
            # Initialize an in-memory tile for accumulating the result
            tile_C = np.zeros((r_end - r_start, c_end - c_start), dtype=C.dtype)
            
            # Loop through the inner dimension k
            for k_start in range(0, K, TILE_SIZE):
                k_end = min(k_start + TILE_SIZE, K)
                
                # Load the corresponding tiles from A and B
                tile_A = A.data[r_start:r_end, k_start:k_end]
                tile_B = B.data[k_start:k_end, c_start:c_end]
                
                # Perform in-memory multiplication and accumulate
                tile_C += tile_A @ tile_B
            
            # Apply scalar multiplication directly and write to disk
            C.data[r_start:r_end, c_start:c_end] = tile_C * scalar
    
    C.data.flush()
    return C

def execute_fused_add_matmul(A: PaperMatrix, B: PaperMatrix, C: PaperMatrix, output_path: str) -> PaperMatrix:
    """
    Performs the fused (A + B) @ C operation in a single pass
    without creating a temporary matrix for (A + B).
    """
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape for addition.")
    if A.shape[1] != C.shape[0]:
        raise ValueError("Inner dimensions must match for multiplication.")
    
    result_shape = (A.shape[0], C.shape[1])
    result = PaperMatrix(output_path, result_shape, mode='w+')
    
    rows_A, K, cols_C = A.shape[0], A.shape[1], C.shape[1]
    
    # Loop over the tiles of the output matrix
    for r_start in range(0, rows_A, TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, rows_A)
        for c_start in range(0, cols_C, TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, cols_C)
            
            # Initialize an in-memory tile for accumulating the result
            result_tile = np.zeros((r_end - r_start, c_end - c_start), dtype=result.dtype)
            
            # Loop through the inner dimension k
            for k_start in range(0, K, TILE_SIZE):
                k_end = min(k_start + TILE_SIZE, K)
                
                # Load the corresponding tiles from A and B
                tile_A = A.data[r_start:r_end, k_start:k_end]
                tile_B = B.data[r_start:r_end, k_start:k_end]
                tile_C = C.data[k_start:k_end, c_start:c_end]
                
                # Perform fused operation: (A + B) @ C for this tile
                sum_tile = tile_A + tile_B
                result_tile += sum_tile @ tile_C
            
            # Write the final computed tile to disk
            result.data[r_start:r_end, c_start:c_end] = result_tile
    
    result.data.flush()
    return result

def execute_fused_double_scalar(A: PaperMatrix, scalar1: float, scalar2: float, output_path: str) -> PaperMatrix:
    """
    Performs the fused (A * scalar1) * scalar2 operation in a single pass,
    optimizing by directly multiplying A by (scalar1 * scalar2).
    """
    C = PaperMatrix(output_path, A.shape, mode='w+')
    
    # Pre-compute the combined scalar
    combined_scalar = scalar1 * scalar2
    
    rows, cols = A.shape
    for r_start in range(0, rows, TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, rows)
        for c_start in range(0, cols, TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, cols)
            
            # Read input tile
            tile_A = A.data[r_start:r_end, c_start:c_end]
            
            # Apply combined scalar multiplication
            result_tile = tile_A * combined_scalar
            
            # Write the result
            C.data[r_start:r_end, c_start:c_end] = result_tile
    
    C.data.flush()
    return C
