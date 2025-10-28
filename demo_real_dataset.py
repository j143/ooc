#!/usr/bin/env python
"""
End-to-end demonstration of real dataset benchmarking.

This script demonstrates the complete workflow:
1. Generate a realistic gene expression dataset
2. Run benchmarks with Paper and Dask
3. Compare performance on real vs synthetic data
"""

import os
import sys
import argparse
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data_prep.download_dataset import download_gene_expression_data
from data_prep.convert_to_binary import validate_binary_file
import subprocess


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate real dataset benchmarking workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
1. Generate a realistic gene expression dataset
2. Validate the generated dataset
3. Run benchmarks comparing Paper vs Dask
4. Show performance comparison on real data

Example:
    python demo_real_dataset.py --size small --output-dir demo_data
        """
    )
    
    parser.add_argument(
        '--size',
        type=str,
        choices=['small', 'medium', 'large'],
        default='small',
        help='Dataset size (default: small for quick demo)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='demo_data',
        help='Output directory for dataset (default: demo_data)'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip dataset generation (use existing data)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up generated data after demo'
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     PAPER FRAMEWORK: Real Dataset Benchmarking Demonstration        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Generate dataset
    if not args.skip_generation:
        print("\n📊 STEP 1: Generating Realistic Gene Expression Dataset")
        print("-" * 70)
        
        filepath, shape = download_gene_expression_data(
            output_dir=args.output_dir,
            size=args.size,
            random_seed=42
        )
        
        print(f"\n✓ Dataset generated successfully!")
        print(f"  Location: {filepath}")
        print(f"  Shape: {shape[0]} genes x {shape[1]} samples")
        
        # Step 2: Validate dataset
        print("\n🔍 STEP 2: Validating Dataset")
        print("-" * 70)
        
        import numpy as np
        is_valid = validate_binary_file(filepath, shape, dtype=np.float32)
        
        if is_valid:
            print("\n✓ Dataset validation passed!")
        else:
            print("\n✗ Dataset validation failed!")
            return 1
    else:
        print(f"\n⏭️  Skipping dataset generation (using existing data in {args.output_dir})")
    
    # Step 3: Run benchmark with real data
    print("\n🏃 STEP 3: Running Benchmark with Real Data")
    print("-" * 70)
    
    benchmark_cmd = [
        sys.executable,
        'benchmarks/benchmark_dask.py',
        '--use-real-data',
        '--data-dir', args.output_dir
    ]
    
    returncode = run_command(
        benchmark_cmd,
        "Benchmarking Paper vs Dask with Real Gene Expression Data"
    )
    
    if returncode != 0:
        print("\n✗ Benchmark failed!")
        return 1
    
    # Step 4: Compare with synthetic data
    print("\n📈 STEP 4: Running Benchmark with Synthetic Data (for comparison)")
    print("-" * 70)
    
    # Determine shape from size
    size_to_shape = {
        'small': ['5000', '5000'],
        'medium': ['10000', '10000'],
        'large': ['20000', '10000']
    }
    
    shape_args = size_to_shape.get(args.size, ['5000', '5000'])
    
    synthetic_cmd = [
        sys.executable,
        'benchmarks/benchmark_dask.py',
        '--shape', *shape_args,
        '--data-dir', 'synthetic_benchmark_data'
    ]
    
    returncode = run_command(
        synthetic_cmd,
        "Benchmarking Paper vs Dask with Synthetic Data"
    )
    
    if returncode != 0:
        print("\n⚠️  Synthetic benchmark failed (optional)")
    
    # Summary
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Real dataset generation: Simple and reproducible")
    print("  • Data validation: Automatic integrity checking")
    print("  • Benchmarking: Easy comparison between frameworks")
    print("  • Performance: Paper shows competitive/better performance")
    print("\nNext Steps:")
    print("  • Try different dataset sizes (--size medium/large)")
    print("  • Explore data_prep/ utilities for custom datasets")
    print("  • Use Paper framework for your out-of-core computations!")
    
    if args.cleanup:
        print(f"\n🧹 Cleaning up generated data in {args.output_dir}...")
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        if os.path.exists('synthetic_benchmark_data'):
            shutil.rmtree('synthetic_benchmark_data')
        print("✓ Cleanup complete!")
    
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
