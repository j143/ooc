#!/usr/bin/env python
"""
Machine Learning algorithms implemented using Paper's operators.

This module implements ML algorithms (linear regression, logistic regression)
using only Paper's out-of-core matrix operations, demonstrating the framework's
capability to handle ML workloads without relying on external ML libraries.
"""

import os
import sys
import numpy as np
from typing import Tuple, Optional
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from paper import numpy_api as pnp


class LinearRegressionPaper:
    """
    Linear Regression implemented using Paper's matrix operations.
    
    Uses gradient descent optimization with Paper's out-of-core operators:
    - Matrix multiplication (@)
    - Matrix transpose (.T)
    - Scalar multiplication (*)
    - Matrix addition (+)
    - Matrix subtraction (-)
    
    This demonstrates ML workload on Paper without external ML libraries.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 100):
        """
        Initialize linear regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X: pnp.ndarray, y: pnp.ndarray, verbose: bool = True) -> 'LinearRegressionPaper':
        """
        Train linear regression model using gradient descent.
        
        Args:
            X: Feature matrix (n_samples, n_features) as Paper ndarray
            y: Target vector (n_samples, 1) as Paper ndarray
            verbose: Print training progress
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        if verbose:
            print(f"Training Linear Regression with Paper operators...")
            print(f"  Features: {n_features}, Samples: {n_samples}")
            print(f"  Learning rate: {self.learning_rate}, Iterations: {self.n_iterations}")
        
        # Initialize weights with zeros
        # For out-of-core, we'll use small initialization in memory then convert
        weights_init = np.zeros((n_features, 1), dtype=np.float32)
        self.weights = pnp.array(weights_init)
        self.bias = 0.0
        
        # Gradient descent training
        # To avoid deep nesting issues, we materialize weights at each iteration
        for iteration in range(self.n_iterations):
            # Forward pass: y_pred = X @ weights
            y_pred = X @ self.weights
            
            # Compute error: error = y_pred - y
            error = y_pred - y
            
            # Compute gradients using Paper operations:
            # dw = (1/n) * X.T @ error
            X_T = X.T
            gradient = X_T @ error
            
            # Materialize gradient to prevent deep nesting
            gradient_np = gradient.to_numpy()
            
            # Update weights in NumPy to avoid nesting
            scale_factor = self.learning_rate / n_samples
            weights_np = self.weights.to_numpy()
            weights_np = weights_np - scale_factor * gradient_np
            
            # Convert back to Paper array
            self.weights = pnp.array(weights_np)
            
            # Print progress
            if verbose and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                # Compute loss for monitoring (MSE)
                error_np = error.to_numpy()
                loss = np.mean(error_np ** 2)
                print(f"  Iteration {iteration}: MSE = {loss:.6f}")
        
        if verbose:
            print("  Training complete!")
        
        return self
    
    def predict(self, X: pnp.ndarray) -> pnp.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples, 1)
        """
        if self.weights is None:
            raise RuntimeError("Model must be trained before prediction")
        
        # y_pred = X @ weights + bias
        y_pred = X @ self.weights
        return y_pred
    
    def score(self, X: pnp.ndarray, y: pnp.ndarray) -> float:
        """
        Compute R² score.
        
        Args:
            X: Feature matrix
            y: True values
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        
        # Materialize for scoring
        y_true_np = y.to_numpy()
        y_pred_np = y_pred.to_numpy()
        
        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y_true_np - y_pred_np) ** 2)
        ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
        
        r2 = 1 - (ss_res / ss_tot)
        return r2


class LogisticRegressionPaper:
    """
    Logistic Regression implemented using Paper's matrix operations.
    
    Uses gradient descent with sigmoid activation.
    Simplified version for binary classification.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 100):
        """
        Initialize logistic regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: pnp.ndarray, y: pnp.ndarray, verbose: bool = True) -> 'LogisticRegressionPaper':
        """
        Train logistic regression model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples, 1) with values 0 or 1
            verbose: Print training progress
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        if verbose:
            print(f"Training Logistic Regression with Paper operators...")
            print(f"  Features: {n_features}, Samples: {n_samples}")
            print(f"  Learning rate: {self.learning_rate}, Iterations: {self.n_iterations}")
        
        # Initialize weights
        weights_init = np.zeros((n_features, 1), dtype=np.float32)
        self.weights = pnp.array(weights_init)
        self.bias = 0.0
        
        # Training loop - materialize weights at each iteration to avoid nesting
        for iteration in range(self.n_iterations):
            # Compute linear combination: z = X @ w
            z = X @ self.weights
            
            # Apply sigmoid (need to materialize for non-linear operation)
            z_np = z.to_numpy()
            y_pred_np = self._sigmoid(z_np)
            y_pred = pnp.array(y_pred_np)
            
            # Compute error
            error = y_pred - y
            
            # Compute gradient: X.T @ error
            X_T = X.T
            gradient = X_T @ error
            
            # Materialize gradient to avoid deep nesting
            gradient_np = gradient.to_numpy()
            
            # Update weights in NumPy
            scale_factor = self.learning_rate / n_samples
            weights_np = self.weights.to_numpy()
            weights_np = weights_np - scale_factor * gradient_np
            
            # Convert back to Paper array
            self.weights = pnp.array(weights_np)
            
            # Print progress
            if verbose and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                # Binary cross-entropy loss
                error_np = error.to_numpy()
                loss = np.mean(error_np ** 2)  # Using MSE for simplicity
                print(f"  Iteration {iteration}: Loss = {loss:.6f}")
        
        if verbose:
            print("  Training complete!")
        
        return self
    
    def predict_proba(self, X: pnp.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probabilities (n_samples, 1)
        """
        if self.weights is None:
            raise RuntimeError("Model must be trained before prediction")
        
        z = X @ self.weights
        z_np = z.to_numpy()
        proba = self._sigmoid(z_np)
        return proba
    
    def predict(self, X: pnp.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            threshold: Decision threshold
            
        Returns:
            Class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.int32)
    
    def score(self, X: pnp.ndarray, y: pnp.ndarray) -> float:
        """
        Compute accuracy score.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Accuracy
        """
        y_pred = self.predict(X)
        y_true = y.to_numpy()
        accuracy = np.mean(y_pred == y_true)
        return accuracy


def prepare_data_for_paper(X_np: np.ndarray, y_np: np.ndarray) -> Tuple[pnp.ndarray, pnp.ndarray]:
    """
    Convert NumPy arrays to Paper arrays for out-of-core processing.
    
    Args:
        X_np: Feature matrix as NumPy array
        y_np: Target vector as NumPy array
        
    Returns:
        Tuple of (X_paper, y_paper)
    """
    # Ensure 2D shapes
    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)
    
    # Convert to Paper arrays
    X_paper = pnp.array(X_np.astype(np.float32))
    y_paper = pnp.array(y_np.astype(np.float32))
    
    return X_paper, y_paper


if __name__ == '__main__':
    # Simple test
    print("Testing Paper ML implementations...")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Linear regression test
    print("\n1. Linear Regression Test")
    print("-"*60)
    true_weights = np.random.randn(n_features, 1).astype(np.float32)
    y_linear_np = X_np @ true_weights + np.random.randn(n_samples, 1).astype(np.float32) * 0.1
    
    X_paper, y_paper = prepare_data_for_paper(X_np, y_linear_np)
    
    lr_model = LinearRegressionPaper(learning_rate=0.1, n_iterations=50)
    lr_model.fit(X_paper, y_paper, verbose=True)
    
    r2_score = lr_model.score(X_paper, y_paper)
    print(f"\nR² Score: {r2_score:.4f}")
    
    # Logistic regression test
    print("\n2. Logistic Regression Test")
    print("-"*60)
    y_binary_np = (X_np[:, 0] > 0).astype(np.float32).reshape(-1, 1)
    
    X_paper, y_paper = prepare_data_for_paper(X_np, y_binary_np)
    
    log_model = LogisticRegressionPaper(learning_rate=0.1, n_iterations=50)
    log_model.fit(X_paper, y_paper, verbose=True)
    
    accuracy = log_model.score(X_paper, y_paper)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\n" + "="*60)
    print("Tests complete!")
