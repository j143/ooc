# Data Preparation for Real-World Benchmarks

This directory contains utilities for preparing real-world datasets for use with the Paper framework benchmarks.

## Overview

The data preparation pipeline consists of two main steps:

1. **Download/Generate Dataset**: Obtain a large dataset suitable for benchmarking
2. **Convert to Binary Format**: Convert the dataset to Paper's binary format for efficient out-of-core processing

## Quick Start

### Generate a Gene Expression Dataset

```bash
# Generate a medium-sized dataset (~400MB)
python -m data_prep.download_dataset --output-dir real_data --size medium

# Generate a large dataset (~800MB)
python -m data_prep.download_dataset --output-dir real_data --size large

# Generate an extra-large dataset (~1.8GB)
python -m data_prep.download_dataset --output-dir real_data --size xlarge
```

### Convert Existing Data

If you have data in other formats (HDF5, NumPy, CSV), you can convert it:

```bash
# Convert HDF5 file
python -m data_prep.convert_to_binary input.h5 output.bin --format hdf5 --dataset mydata

# Convert NumPy file
python -m data_prep.convert_to_binary input.npy output.bin --format npy

# Convert CSV file
python -m data_prep.convert_to_binary input.csv output.bin --format csv --shape 10000 5000
```

## Dataset Characteristics

The generated gene expression dataset mimics real biological data:

- **Structure**: Genes (rows) x Samples (columns)
- **Value Distribution**: Log-normal (characteristic of RNA-seq data)
- **Patterns**: Gene co-expression modules (correlated groups of genes)
- **Non-negative**: All values ≥ 0 (as in real expression data)

### Size Presets

| Preset  | Dimensions        | Size    | Use Case           |
|---------|-------------------|---------|--------------------|
| small   | 5,000 x 5,000     | ~100MB  | Quick testing      |
| medium  | 10,000 x 10,000   | ~400MB  | Moderate benchmark |
| large   | 20,000 x 10,000   | ~800MB  | Large benchmark    |
| xlarge  | 30,000 x 15,000   | ~1.8GB  | Very large dataset |

## Usage in Benchmarks

After generating the dataset, you can use it in benchmarks:

```bash
# Run benchmark with real data
python benchmarks/benchmark_dask.py --use-real-data --data-dir real_data

# Compare synthetic vs real data
python benchmarks/benchmark_dask.py --use-real-data --data-dir real_data --compare-synthetic
```

## File Formats

### Input Formats Supported

- **HDF5** (`.h5`, `.hdf5`): Hierarchical data format
- **NumPy** (`.npy`): NumPy binary format
- **CSV** (`.csv`): Comma-separated values
- **TSV** (`.tsv`, `.txt`): Tab-separated values
- **Binary** (`.dat`, `.bin`): Raw binary data

### Output Format

All datasets are converted to Paper's binary format:
- Raw binary file (memory-mappable)
- Row-major layout
- Specified dtype (default: float32)
- No compression (for maximum I/O performance)

## Validation

The conversion utilities include validation to ensure data integrity:

```bash
# Validate a converted file
python -m data_prep.convert_to_binary input.npy output.bin --validate
```

Validation checks:
- File size matches expected dimensions
- Data is readable via PaperMatrix
- No NaN or Inf values
- Random tile sampling succeeds

## API Reference

### download_dataset.py

```python
from data_prep import download_gene_expression_data

filepath, shape = download_gene_expression_data(
    output_dir="real_data",
    size="medium",  # "small", "medium", "large", "xlarge"
    random_seed=42
)
```

### convert_to_binary.py

```python
from data_prep import convert_to_paper_format, validate_binary_file

# Convert data
output_path, shape = convert_to_paper_format(
    input_path="data.h5",
    output_path="data.bin",
    input_format="hdf5",
    dataset_name="expression_matrix"
)

# Validate
is_valid = validate_binary_file(output_path, shape)
```

## Notes

- Generated datasets are reproducible (same random seed = same data)
- Large datasets are generated in chunks to avoid memory issues
- Memory-mapped files are used throughout for efficiency
- All utilities support progress reporting for long operations
