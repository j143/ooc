"""
Convert various data formats to Paper-compatible binary format.

This module provides utilities to convert data from common formats
(CSV, TSV, HDF5, NumPy) to the simple binary format that PaperMatrix uses.
"""

import os
import numpy as np
import h5py
from typing import Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paper.core import PaperMatrix
from paper.config import TILE_SIZE


def convert_to_paper_format(
    input_path: str,
    output_path: str,
    input_format: str = "auto",
    shape: Optional[Tuple[int, int]] = None,
    dtype=np.float32,
    dataset_name: Optional[str] = None
) -> Tuple[str, Tuple[int, int]]:
    """
    Convert data from various formats to Paper-compatible binary format.
    
    Args:
        input_path: Path to input data file
        output_path: Path for output binary file
        input_format: Input format - "auto", "npy", "hdf5", "csv", "tsv", "binary"
        shape: Shape of the data (required for binary and csv/tsv)
        dtype: Data type for output
        dataset_name: For HDF5 files, name of the dataset to read
        
    Returns:
        Tuple of (output_path, shape)
    """
    # Auto-detect format from file extension
    if input_format == "auto":
        ext = os.path.splitext(input_path)[1].lower()
        format_map = {
            ".npy": "npy",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".csv": "csv",
            ".tsv": "tsv",
            ".txt": "tsv",
            ".dat": "binary",
            ".bin": "binary"
        }
        input_format = format_map.get(ext, "binary")
    
    print(f"\nConverting data to Paper format:")
    print(f"  Input: {input_path} (format: {input_format})")
    print(f"  Output: {output_path}")
    
    # Load data based on format
    if input_format == "npy":
        data = np.load(input_path)
        if data.dtype != dtype:
            data = data.astype(dtype)
        shape = data.shape
        
    elif input_format == "hdf5":
        with h5py.File(input_path, 'r') as f:
            if dataset_name is None:
                # Use first dataset found
                dataset_name = list(f.keys())[0]
            print(f"  Reading HDF5 dataset: {dataset_name}")
            data = f[dataset_name][:]
            if data.dtype != dtype:
                data = data.astype(dtype)
            shape = data.shape
            
    elif input_format in ["csv", "tsv"]:
        delimiter = ',' if input_format == "csv" else '\t'
        data = np.loadtxt(input_path, delimiter=delimiter, dtype=dtype)
        shape = data.shape
        
    elif input_format == "binary":
        if shape is None:
            raise ValueError("Shape must be provided for binary input format")
        # Read binary file and reshape
        data = np.fromfile(input_path, dtype=dtype).reshape(shape)
        
    else:
        raise ValueError(f"Unsupported format: {input_format}")
    
    print(f"  Data shape: {shape}")
    print(f"  Data dtype: {dtype}")
    print(f"  Size: {data.nbytes / (1024**2):.2f} MB")
    
    # Write to Paper format using memory-mapped file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    output_data = np.memmap(output_path, dtype=dtype, mode='w+', shape=shape)
    
    # Copy data in tiles to avoid memory issues with large arrays
    chunk_size = 1000
    for i in range(0, shape[0], chunk_size):
        i_end = min(i + chunk_size, shape[0])
        output_data[i:i_end] = data[i:i_end]
        if i % (chunk_size * 10) == 0:
            progress = (i / shape[0]) * 100
            print(f"  Progress: {progress:.1f}%")
    
    output_data.flush()
    del output_data
    
    print(f"✓ Conversion complete: {output_path}")
    
    return output_path, shape


def validate_binary_file(
    filepath: str,
    shape: Tuple[int, int],
    dtype=np.float32,
    n_samples: int = 10
) -> bool:
    """
    Validate a Paper-compatible binary file.
    
    Args:
        filepath: Path to binary file
        shape: Expected shape
        dtype: Expected data type
        n_samples: Number of random samples to check
        
    Returns:
        True if validation passes
    """
    print(f"\nValidating binary file: {filepath}")
    print(f"  Expected shape: {shape}")
    print(f"  Expected dtype: {dtype}")
    
    if not os.path.exists(filepath):
        print("  ✗ File does not exist")
        return False
    
    # Check file size
    expected_size = shape[0] * shape[1] * np.dtype(dtype).itemsize
    actual_size = os.path.getsize(filepath)
    
    print(f"  Expected size: {expected_size / (1024**2):.2f} MB")
    print(f"  Actual size: {actual_size / (1024**2):.2f} MB")
    
    if actual_size != expected_size:
        print(f"  ✗ Size mismatch!")
        return False
    
    # Try to read using PaperMatrix
    try:
        matrix = PaperMatrix(filepath, shape, dtype=dtype, mode='r')
        
        # Sample some random tiles
        print(f"  Sampling {n_samples} random tiles...")
        for i in range(n_samples):
            r_start = np.random.randint(0, max(1, shape[0] - TILE_SIZE))
            c_start = np.random.randint(0, max(1, shape[1] - TILE_SIZE))
            tile = matrix.get_tile(r_start, c_start)
            
            # Check for invalid values
            if np.any(np.isnan(tile)) or np.any(np.isinf(tile)):
                print(f"  ✗ Found NaN or Inf values in tile at ({r_start}, {c_start})")
                return False
        
        matrix.close()
        print("  ✓ Validation passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
        return False


def create_metadata_file(
    data_dir: str,
    dataset_name: str,
    shape: Tuple[int, int],
    dtype: str,
    description: str = ""
) -> str:
    """
    Create a metadata file for the dataset.
    
    Args:
        data_dir: Directory containing the dataset
        dataset_name: Name of the dataset
        shape: Shape of the data
        dtype: Data type
        description: Optional description
        
    Returns:
        Path to metadata file
    """
    metadata_path = os.path.join(data_dir, "metadata.txt")
    
    with open(metadata_path, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Shape: {shape[0]} x {shape[1]}\n")
        f.write(f"Dtype: {dtype}\n")
        f.write(f"Size: {(shape[0] * shape[1] * np.dtype(dtype).itemsize) / (1024**3):.2f} GB\n")
        if description:
            f.write(f"\nDescription:\n{description}\n")
    
    print(f"✓ Metadata saved to: {metadata_path}")
    return metadata_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert data to Paper-compatible binary format"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input data file"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path for output binary file"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "npy", "hdf5", "csv", "tsv", "binary"],
        help="Input format (default: auto-detect)"
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        help="Shape as two integers (rows cols) - required for binary/csv/tsv"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data type (default: float32)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="For HDF5: name of dataset to read"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the output file after conversion"
    )
    
    args = parser.parse_args()
    
    # Convert dtype string to numpy dtype
    dtype = getattr(np, args.dtype)
    shape_tuple = tuple(args.shape) if args.shape else None
    
    output_path, shape = convert_to_paper_format(
        input_path=args.input_path,
        output_path=args.output_path,
        input_format=args.format,
        shape=shape_tuple,
        dtype=dtype,
        dataset_name=args.dataset
    )
    
    if args.validate:
        validate_binary_file(output_path, shape, dtype)
