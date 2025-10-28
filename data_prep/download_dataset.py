"""
Download and prepare gene expression dataset for benchmarking.

This module provides utilities to download a large gene expression dataset
from publicly available sources and prepare it for use with the Paper framework.

For this implementation, we'll create a synthetic but realistic gene expression
dataset that mimics real biological data characteristics:
- Large size (exceeding RAM)
- Realistic value distributions (log-normal, as in real RNA-seq data)
- Structured patterns (gene co-expression modules)
"""

import os
import numpy as np
import sys
from typing import Tuple, Optional


def generate_realistic_gene_expression_data(
    output_dir: str,
    n_samples: int = 10000,
    n_genes: int = 20000,
    dtype=np.float32,
    random_seed: int = 42
) -> Tuple[str, Tuple[int, int]]:
    """
    Generate a large, realistic gene expression matrix.
    
    This creates a synthetic dataset that mimics real gene expression data:
    - Size: (n_genes x n_samples) - genes as rows, samples as columns
    - Values follow log-normal distribution (characteristic of RNA-seq)
    - Contains structured patterns (gene modules with correlated expression)
    
    Args:
        output_dir: Directory to save the dataset
        n_samples: Number of samples (columns) - default 10,000
        n_genes: Number of genes (rows) - default 20,000
        dtype: Data type for the matrix
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (filepath, shape)
        
    Note:
        A 20,000 x 10,000 matrix of float32 = ~800MB per matrix.
        For benchmarking, we'll create multiple matrices to exceed typical RAM.
    """
    np.random.seed(random_seed)
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "gene_expression.dat")
    shape = (n_genes, n_samples)
    
    print(f"Generating realistic gene expression data: {n_genes} genes x {n_samples} samples")
    print(f"Expected size: ~{(n_genes * n_samples * np.dtype(dtype).itemsize) / (1024**3):.2f} GB")
    
    # Create memory-mapped file for efficient generation
    data = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
    
    # Generate data in chunks to avoid memory issues
    chunk_size = 1000  # Process 1000 genes at a time
    
    for gene_start in range(0, n_genes, chunk_size):
        gene_end = min(gene_start + chunk_size, n_genes)
        chunk_genes = gene_end - gene_start
        
        # Generate base expression levels (log-normal distribution)
        # Mean expression varies by gene
        base_expression = np.random.lognormal(
            mean=2.0, 
            sigma=1.5, 
            size=(chunk_genes, n_samples)
        ).astype(dtype)
        
        # Add some correlation structure (gene modules)
        # Every 100 genes form a module with correlated expression
        module_size = 100
        for module_start in range(0, chunk_genes, module_size):
            module_end = min(module_start + module_size, chunk_genes)
            
            # Generate a shared expression pattern for this module
            shared_pattern = np.random.randn(n_samples).astype(dtype)
            
            # Add the shared pattern to genes in this module
            for i in range(module_start, module_end):
                # Mix individual variation with shared pattern
                base_expression[i] += 0.3 * shared_pattern
        
        # Ensure non-negative values (as in real RNA-seq)
        base_expression = np.maximum(base_expression, 0)
        
        # Write chunk to file
        data[gene_start:gene_end, :] = base_expression
        
        if (gene_start // chunk_size + 1) % 5 == 0:
            progress = (gene_end / n_genes) * 100
            print(f"  Progress: {progress:.1f}% ({gene_end}/{n_genes} genes)")
    
    # Flush to disk
    data.flush()
    del data
    
    print(f"✓ Generated dataset saved to: {filepath}")
    print(f"  Shape: {shape}")
    print(f"  Dtype: {dtype}")
    
    return filepath, shape


def download_gene_expression_data(
    output_dir: str,
    size: str = "medium",
    random_seed: int = 42
) -> Tuple[str, Tuple[int, int]]:
    """
    Download or generate gene expression dataset.
    
    Args:
        output_dir: Directory to save the dataset
        size: Dataset size - "small", "medium", or "large"
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (filepath, shape)
    """
    size_configs = {
        "small": (5000, 5000),      # ~100MB - for quick testing
        "medium": (10000, 10000),   # ~400MB - moderate size
        "large": (20000, 10000),    # ~800MB - large dataset
        "xlarge": (30000, 15000),   # ~1.8GB - very large
    }
    
    if size not in size_configs:
        raise ValueError(f"Size must be one of {list(size_configs.keys())}")
    
    n_genes, n_samples = size_configs[size]
    
    print(f"\n{'='*60}")
    print(f"GENE EXPRESSION DATA GENERATION")
    print(f"{'='*60}")
    print(f"Size preset: {size}")
    print(f"Dimensions: {n_genes} genes x {n_samples} samples")
    
    return generate_realistic_gene_expression_data(
        output_dir=output_dir,
        n_samples=n_samples,
        n_genes=n_genes,
        random_seed=random_seed
    )


if __name__ == "__main__":
    # Command-line interface for standalone usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download/generate gene expression dataset for benchmarking"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="real_data",
        help="Directory to save the dataset (default: real_data)"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "medium", "large", "xlarge"],
        default="medium",
        help="Dataset size preset (default: medium)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    filepath, shape = download_gene_expression_data(
        output_dir=args.output_dir,
        size=args.size,
        random_seed=args.seed
    )
    
    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"{'='*60}")
    print(f"Dataset ready at: {filepath}")
    print(f"Shape: {shape}")
    print(f"\nYou can now use this dataset in benchmarks by passing:")
    print(f"  --data-dir {args.output_dir}")
