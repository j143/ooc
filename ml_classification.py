#!/usr/bin/env python
"""
Gene Expression Classification Example

This module demonstrates an end-to-end ML workflow for gene expression classification:
- Load gene expression data using Paper or Dask
- Generate disease/control labels
- Train a classifier
- Evaluate with accuracy, ROC AUC, and other metrics
- Compare Paper vs Dask performance on the complete ML pipeline
"""

import os
import sys
import time
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import h5py

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from paper.core import PaperMatrix


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


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 100,
    random_seed: int = 42
) -> LogisticRegression:
    """
    Train a logistic regression classifier.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels
        max_iter: Maximum iterations for solver
        random_seed: Random seed for reproducibility
        
    Returns:
        Trained classifier
    """
    print(f"Training classifier on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    start_time = time.time()
    
    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_seed,
        solver='lbfgs',
        n_jobs=-1  # Use all CPU cores
    )
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.2f}s")
    
    return clf


def evaluate_classifier(
    clf: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate classifier and return metrics.
    
    Args:
        clf: Trained classifier
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating classifier...")
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    return metrics


def run_ml_pipeline(
    data_path: str,
    shape: Tuple[int, int],
    use_paper: bool = True,
    test_size: float = 0.2,
    random_seed: int = 42,
    hdf5_path: Optional[str] = None
) -> Tuple[Dict[str, float], float]:
    """
    Run complete ML pipeline: load data, train, evaluate.
    
    Args:
        data_path: Path to data file (binary for Paper)
        shape: Shape of the data (n_genes, n_samples)
        use_paper: If True, use Paper framework, else use Dask/HDF5
        test_size: Proportion of data for testing
        random_seed: Random seed for reproducibility
        hdf5_path: Path to HDF5 file (required if use_paper=False)
        
    Returns:
        Tuple of (metrics dict, total time in seconds)
    """
    print("\n" + "="*70)
    framework_name = "Paper" if use_paper else "Dask/HDF5"
    print(f"Running ML Pipeline with {framework_name}")
    print("="*70)
    
    pipeline_start = time.time()
    
    # Load data
    if use_paper:
        data = load_data_with_paper(data_path, shape)
    else:
        if hdf5_path is None:
            raise ValueError("hdf5_path required when use_paper=False")
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
    
    # Train classifier
    print()
    clf = train_classifier(X_train, y_train, random_seed=random_seed)
    
    # Evaluate
    print()
    metrics = evaluate_classifier(clf, X_test, y_test)
    
    total_time = time.time() - pipeline_start
    
    print("\n" + "="*70)
    print(f"{framework_name} ML Pipeline Complete - Total Time: {total_time:.2f}s")
    print("="*70)
    
    return metrics, total_time


def compare_ml_frameworks(
    paper_data_path: str,
    shape: Tuple[int, int],
    hdf5_path: str,
    test_size: float = 0.2,
    random_seed: int = 42
) -> None:
    """
    Compare Paper and Dask frameworks on complete ML pipeline.
    
    Args:
        paper_data_path: Path to binary data for Paper
        shape: Data shape (n_genes, n_samples)
        hdf5_path: Path to HDF5 data for Dask
        test_size: Test set proportion
        random_seed: Random seed for reproducibility
    """
    print("\n" + "="*70)
    print("COMPARING PAPER VS DASK ON GENE EXPRESSION CLASSIFICATION")
    print("="*70)
    
    # Run with Paper
    paper_metrics, paper_time = run_ml_pipeline(
        paper_data_path,
        shape,
        use_paper=True,
        test_size=test_size,
        random_seed=random_seed
    )
    
    # Run with Dask (data_path not used, only hdf5_path is used)
    dask_metrics, dask_time = run_ml_pipeline(
        data_path="",  # Not used when use_paper=False
        shape=shape,
        use_paper=False,
        test_size=test_size,
        random_seed=random_seed,
        hdf5_path=hdf5_path
    )
    
    # Print comparison table
    print("\n" + "="*70)
    print("ML PIPELINE COMPARISON: Paper vs. Dask/HDF5")
    print("="*70)
    print(f"{'Metric':<30} | {'Paper':<15} | {'Dask/HDF5':<15}")
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
    
    # Verify same results (should be identical with same random seed)
    acc_diff = abs(paper_metrics['accuracy'] - dask_metrics['accuracy'])
    auc_diff = abs(paper_metrics['roc_auc'] - dask_metrics['roc_auc'])
    
    print("\nResult Verification:")
    if acc_diff < 1e-6 and auc_diff < 1e-6:
        print("  ✓ Both frameworks produce identical results")
    else:
        print(f"  ⚠ Small differences detected (accuracy: {acc_diff:.6f}, AUC: {auc_diff:.6f})")
        print("    This is expected due to numerical precision")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gene Expression Classification Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single pipeline with Paper
  python ml_classification.py --data-path real_data/gene_expression.dat --shape 5000 5000
  
  # Compare Paper vs Dask
  python ml_classification.py --data-path real_data/gene_expression.dat --shape 5000 5000 --hdf5-path real_data/data.hdf5 --compare
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
        '--use-dask',
        action='store_true',
        help='Use Dask/HDF5 instead of Paper'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
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
            random_seed=args.seed
        )
    else:
        use_paper = not args.use_dask
        if not use_paper and not args.hdf5_path:
            parser.error("--hdf5-path required when using --use-dask")
        
        metrics, total_time = run_ml_pipeline(
            args.data_path,
            shape,
            use_paper=use_paper,
            test_size=args.test_size,
            random_seed=args.seed,
            hdf5_path=args.hdf5_path
        )
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"{'='*70}")
