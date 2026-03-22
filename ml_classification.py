#!/usr/bin/env python
"""
Gene Expression Classification using Paper's Operators

This module demonstrates ML workload using Paper's out-of-core matrix operations
instead of external ML libraries like scikit-learn. It implements:
- Linear regression using Paper's operators (gradient descent)
- Classification using Paper's operators
- Performance comparison with Dask on ML tasks

This benchmarks Paper's capability to handle ML workloads directly.
"""

import os
import sys
import time
import numpy as np
from typing import Tuple, Dict, Optional
import h5py

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from paper.core import PaperMatrix
from paper import numpy_api as pnp
from paper_ml import LinearRegressionPaper, LogisticRegressionPaper, prepare_data_for_paper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def generate_classification_labels(
    n_samples: int,
    disease_ratio: float = 0.5,
    random_seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic disease/control labels for classification.
    
    Args:
        n_samples: Number of samples
        disease_ratio: Proportion of disease samples (default: 0.5 for balanced)
        random_seed: Random seed for reproducibility
        
    Returns:
        Binary labels array (0 = control, 1 = disease)
    """
    np.random.seed(random_seed)
    n_disease = int(n_samples * disease_ratio)
    
    labels = np.zeros(n_samples, dtype=np.int32)
    labels[:n_disease] = 1
    
    # Shuffle to mix disease and control samples
    np.random.shuffle(labels)
    
    return labels


def load_data_with_paper(
    filepath: str,
    shape: Tuple[int, int]
) -> np.ndarray:
    """
    Load gene expression data using Paper framework.
    
    Args:
        filepath: Path to binary data file
        shape: Shape of the data (n_genes, n_samples)
        
    Returns:
        Data array transposed to (n_samples, n_genes) for ML
    """
    print("Loading data with Paper framework...")
    start_time = time.time()
    
    # Load data using Paper
    matrix = PaperMatrix(filepath, shape, mode='r')
    
    # Read data - for ML we need (n_samples, n_features) format
    # Gene expression data is (n_genes, n_samples), so transpose
    data = np.array(matrix.data, dtype=np.float32).T
    
    load_time = time.time() - start_time
    print(f"  Paper loaded data in {load_time:.2f}s - shape: {data.shape}")
    
    return data


def load_data_with_dask(
    hdf5_path: str,
    dataset_name: str = 'A'
) -> np.ndarray:
    """
    Load gene expression data using Dask/HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        dataset_name: Name of dataset in HDF5 file
        
    Returns:
        Data array transposed to (n_samples, n_genes) for ML
    """
    print("Loading data with Dask/HDF5...")
    start_time = time.time()
    
    with h5py.File(hdf5_path, 'r') as f:
        # Load data and transpose
        data = f[dataset_name][:].T
    
    load_time = time.time() - start_time
    print(f"  Dask/HDF5 loaded data in {load_time:.2f}s - shape: {data.shape}")
    
    return data


def run_ml_pipeline_with_paper(
    data_path: str,
    shape: Tuple[int, int],
    test_size: float = 0.2,
    random_seed: int = 42,
    n_iterations: int = 20,
    learning_rate: float = 0.01
) -> Tuple[Dict[str, float], float]:
    """
    Run complete ML pipeline using Paper's operators.
    
    Args:
        data_path: Path to data file (binary for Paper)
        shape: Shape of the data (n_genes, n_samples)
        test_size: Proportion of data for testing
        random_seed: Random seed for reproducibility
        n_iterations: Number of training iterations
        learning_rate: Learning rate for gradient descent
        
    Returns:
        Tuple of (metrics dict, total time in seconds)
    """
    print("\n" + "="*70)
    print("Running ML Pipeline with Paper Operators")
    print("="*70)
    
    pipeline_start = time.time()
    
    # Load data
    data = load_data_with_paper(data_path, shape)
    n_samples = data.shape[0]
    
    # Generate labels
    print(f"\nGenerating classification labels for {n_samples} samples...")
    labels = generate_classification_labels(n_samples, random_seed=random_seed)
    print(f"  Disease samples: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    print(f"  Control samples: {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
    
    # Split data
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_seed, stratify=labels
    )
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Convert to Paper arrays
    print(f"\nConverting to Paper arrays for out-of-core training...")
    X_train_paper, y_train_paper = prepare_data_for_paper(X_train, y_train)
    X_test_paper, y_test_paper = prepare_data_for_paper(X_test, y_test)
    
    # Train classifier using Paper operators
    print()
    clf = LogisticRegressionPaper(learning_rate=learning_rate, n_iterations=n_iterations)
    clf.fit(X_train_paper, y_train_paper, verbose=True)
    
    # Evaluate
    print("\nEvaluating classifier...")
    y_pred = clf.predict(X_test_paper)
    y_pred_proba = clf.predict_proba(X_test_paper)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    total_time = time.time() - pipeline_start
    
    print("\n" + "="*70)
    print(f"Paper Operators ML Pipeline Complete - Total Time: {total_time:.2f}s")
    print("="*70)
    
    return metrics, total_time


def run_ml_pipeline_with_dask(
    hdf5_path: str,
    test_size: float = 0.2,
    random_seed: int = 42,
    n_iterations: int = 20,
    learning_rate: float = 0.01
) -> Tuple[Dict[str, float], float]:
    """
    Run ML pipeline using Dask for data loading + Paper operators for ML.
    
    Args:
        hdf5_path: Path to HDF5 file
        test_size: Test set proportion
        random_seed: Random seed for reproducibility
        n_iterations: Number of training iterations
        learning_rate: Learning rate
        
    Returns:
        Tuple of (metrics dict, total time in seconds)
    """
    print("\n" + "="*70)
    print("Running ML Pipeline with Dask + Paper Operators")
    print("="*70)
    
    pipeline_start = time.time()
    
    # Load data with Dask
    data = load_data_with_dask(hdf5_path)
    n_samples = data.shape[0]
    
    # Generate labels
    print(f"\nGenerating classification labels for {n_samples} samples...")
    labels = generate_classification_labels(n_samples, random_seed=random_seed)
    print(f"  Disease samples: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    print(f"  Control samples: {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
    
    # Split data
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_seed, stratify=labels
    )
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Convert to Paper arrays
    print(f"\nConverting to Paper arrays for out-of-core training...")
    X_train_paper, y_train_paper = prepare_data_for_paper(X_train, y_train)
    X_test_paper, y_test_paper = prepare_data_for_paper(X_test, y_test)
    
    # Train classifier using Paper operators
    print()
    clf = LogisticRegressionPaper(learning_rate=learning_rate, n_iterations=n_iterations)
    clf.fit(X_train_paper, y_train_paper, verbose=True)
    
    # Evaluate
    print("\nEvaluating classifier...")
    y_pred = clf.predict(X_test_paper)
    y_pred_proba = clf.predict_proba(X_test_paper)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    total_time = time.time() - pipeline_start
    
    print("\n" + "="*70)
    print(f"Dask + Paper Operators ML Pipeline Complete - Total Time: {total_time:.2f}s")
    print("="*70)
    
    return metrics, total_time


def compare_ml_frameworks(
    paper_data_path: str,
    shape: Tuple[int, int],
    hdf5_path: str,
    test_size: float = 0.2,
    random_seed: int = 42,
    n_iterations: int = 20,
    learning_rate: float = 0.01
) -> None:
    """
    Compare Paper and Dask frameworks on complete ML pipeline using Paper's operators.
    
    Args:
        paper_data_path: Path to binary data for Paper
        shape: Data shape (n_genes, n_samples)
        hdf5_path: Path to HDF5 data for Dask
        test_size: Test set proportion
        random_seed: Random seed for reproducibility
        n_iterations: Number of training iterations
        learning_rate: Learning rate for gradient descent
    """
    print("\n" + "="*70)
    print("COMPARING PAPER VS DASK ON GENE EXPRESSION CLASSIFICATION")
    print("Using Paper's Operators for ML Workload")
    print("="*70)
    
    # Run with Paper
    paper_metrics, paper_time = run_ml_pipeline_with_paper(
        paper_data_path,
        shape,
        test_size=test_size,
        random_seed=random_seed,
        n_iterations=n_iterations,
        learning_rate=learning_rate
    )
    
    # Run with Dask
    dask_metrics, dask_time = run_ml_pipeline_with_dask(
        hdf5_path,
        test_size=test_size,
        random_seed=random_seed,
        n_iterations=n_iterations,
        learning_rate=learning_rate
    )
    
    # Print comparison table
    print("\n" + "="*70)
    print("ML PIPELINE COMPARISON: Paper vs. Dask")
    print("(Both use Paper's operators for ML workload)")
    print("="*70)
    print(f"{'Metric':<30} | {'Paper':<15} | {'Dask':<15}")
    print("-"*70)
    print(f"{'Total Time (s)':<30} | {paper_time:<15.2f} | {dask_time:<15.2f}")
    print(f"{'Accuracy':<30} | {paper_metrics['accuracy']:<15.4f} | {dask_metrics['accuracy']:<15.4f}")
    print(f"{'ROC AUC':<30} | {paper_metrics['roc_auc']:<15.4f} | {dask_metrics['roc_auc']:<15.4f}")
    print("-"*70)
    
    # Calculate speedup
    if dask_time > 0:
        speedup = dask_time / paper_time
        print(f"{'Paper Speedup':<30} | {speedup:<15.2f}x |")
    
    print("="*70)
    
    # Verify same results
    acc_diff = abs(paper_metrics['accuracy'] - dask_metrics['accuracy'])
    auc_diff = abs(paper_metrics['roc_auc'] - dask_metrics['roc_auc'])
    
    print("\nResult Verification:")
    if acc_diff < 0.01 and auc_diff < 0.01:
        print("  ✓ Both frameworks produce similar results")
        print(f"    (Accuracy diff: {acc_diff:.6f}, AUC diff: {auc_diff:.6f})")
    else:
        print(f"  ⚠ Results differ (accuracy: {acc_diff:.6f}, AUC: {auc_diff:.6f})")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gene Expression Classification using Paper's Operators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single pipeline with Paper
  python ml_classification.py --data-path gene_expression.dat --shape 5000 5000
  
  # Compare Paper vs Dask (both use Paper operators for ML)
  python ml_classification.py --data-path gene_expression.dat --shape 5000 5000 --hdf5-path data.hdf5 --compare
        """
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to binary gene expression data'
    )
    parser.add_argument(
        '--shape',
        type=int,
        nargs=2,
        required=True,
        metavar=('N_GENES', 'N_SAMPLES'),
        help='Shape of the data matrix'
    )
    parser.add_argument(
        '--hdf5-path',
        type=str,
        help='Path to HDF5 file (required for --compare)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare Paper vs Dask frameworks'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=20,
        help='Number of training iterations (default: 20)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate for gradient descent (default: 0.01)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    shape = tuple(args.shape)
    
    if args.compare:
        if not args.hdf5_path:
            parser.error("--hdf5-path required when using --compare")
        compare_ml_frameworks(
            args.data_path,
            shape,
            args.hdf5_path,
            test_size=args.test_size,
            random_seed=args.seed,
            n_iterations=args.iterations,
            learning_rate=args.learning_rate
        )
    else:
        metrics, total_time = run_ml_pipeline_with_paper(
            args.data_path,
            shape,
            test_size=args.test_size,
            random_seed=args.seed,
            n_iterations=args.iterations,
            learning_rate=args.learning_rate
        )
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"{'='*70}")
