def add_eager(A: MiniMatrix, B: MiniMatrix, output_path: str):
    """Performs out-of-core matrix addition: C = A + B."""
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape for addition.")

    C = MiniMatrix(output_path, A.shape, mode='w+')
    
    print("Performing eager addition...")
    rows, cols = A.shape
    # Iterate through the matrices tile by tile
    for r_start in range(0, rows, TILE_SIZE):
        r_end = min(r_start + TILE_SIZE, rows)
        for c_start in range(0, cols, TILE_SIZE):
            c_end = min(c_start + TILE_SIZE, cols)
            
            # 1. Read tiles from A and B into memory
            tile_A = A.data[r_start:r_end, c_start:c_end]
            tile_B = B.data[r_start:r_end, c_start:c_end]
            
            # 2. Compute the result in memory
            tile_C = tile_A + tile_B
            
            # 3. Write the resulting tile to C's file
            C.data[r_start:r_end, c_start:c_end] = tile_C
    
    C.data.flush() # Ensure all data is saved to disk
    print("Addition complete.")
    return C

def multiply_eager(A: MiniMatrix, B: MiniMatrix):
    """Performs out-of-core tiled matrix multiplication: C = A @ B."""
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must match for multiplication.")
    
    C_shape = (A.shape[0], B.shape[1])
    C_path = os.path.join(DATA_DIR, "C_mul.bin")
    C = MiniMatrix(C_path, C_shape, mode='w+')

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
                tile_A = A.data[r_start:r_end, k_start:k_end]
                tile_B = B.data[k_start:k_end, c_start:c_end]

                #3. Perform in-memory multiplication and accumulate
                tile_C += tile_A @ tile_B
            
            # 4. write the final computed tile to disk
            C.data[r_start:r_end, c_start:c_end] = tile_C
        
        C.data.flush()
        print("Multiplication complete.")
        return C
