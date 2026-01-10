"""
Tests for ML classification functionality using Paper's operators.
"""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_classification import generate_classification_labels
from paper_ml import LinearRegressionPaper, LogisticRegressionPaper, prepare_data_for_paper


class TestMLClassification(unittest.TestCase):
    """Test ML classification utilities with Paper operators."""
    
    def test_generate_labels_shape(self):
        """Test that generated labels have correct shape."""
        n_samples = 100
        labels = generate_classification_labels(n_samples)
        
        self.assertEqual(len(labels), n_samples)
        self.assertEqual(labels.dtype, np.int32)
    
    def test_generate_labels_values(self):
        """Test that labels are binary (0 and 1)."""
        labels = generate_classification_labels(100)
        
        unique_values = np.unique(labels)
        self.assertTrue(np.array_equal(unique_values, [0, 1]))
    
    def test_generate_labels_ratio(self):
        """Test that disease ratio is respected."""
        n_samples = 1000
        disease_ratio = 0.3
        
        labels = generate_classification_labels(n_samples, disease_ratio=disease_ratio)
        
        actual_ratio = np.mean(labels)
        # Should be close to specified ratio (within 5% tolerance)
        self.assertAlmostEqual(actual_ratio, disease_ratio, delta=0.05)
    
    def test_generate_labels_reproducibility(self):
        """Test that same seed produces same labels."""
        labels1 = generate_classification_labels(100, random_seed=42)
        labels2 = generate_classification_labels(100, random_seed=42)
        
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_generate_labels_different_seeds(self):
        """Test that different seeds produce different labels."""
        labels1 = generate_classification_labels(100, random_seed=42)
        labels2 = generate_classification_labels(100, random_seed=43)
        
        # Should not be identical
        self.assertFalse(np.array_equal(labels1, labels2))
    
    def test_linear_regression_training(self):
        """Test that linear regression can be trained with Paper operators."""
        # Create simple synthetic data
        np.random.seed(42)
        n_samples = 50
        n_features = 3
        
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        true_weights = np.array([[1.0], [2.0], [-0.5]], dtype=np.float32)
        y_np = X_np @ true_weights + np.random.randn(n_samples, 1).astype(np.float32) * 0.1
        
        X_paper, y_paper = prepare_data_for_paper(X_np, y_np)
        
        # Train model
        model = LinearRegressionPaper(learning_rate=0.1, n_iterations=20)
        model.fit(X_paper, y_paper, verbose=False)
        
        # Check that model has weights
        self.assertIsNotNone(model.weights)
        
        # Check that predictions work
        y_pred = model.predict(X_paper)
        self.assertIsNotNone(y_pred)
    
    def test_linear_regression_score(self):
        """Test that linear regression scoring works."""
        np.random.seed(42)
        n_samples = 50
        n_features = 3
        
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        true_weights = np.array([[1.0], [2.0], [-0.5]], dtype=np.float32)
        y_np = X_np @ true_weights + np.random.randn(n_samples, 1).astype(np.float32) * 0.1
        
        X_paper, y_paper = prepare_data_for_paper(X_np, y_np)
        
        # Train model
        model = LinearRegressionPaper(learning_rate=0.1, n_iterations=30)
        model.fit(X_paper, y_paper, verbose=False)
        
        # Compute R² score
        r2 = model.score(X_paper, y_paper)
        
        # R² should be reasonable for this linearly generated data
        self.assertGreater(r2, 0.5)
    
    def test_logistic_regression_training(self):
        """Test that logistic regression can be trained with Paper operators."""
        np.random.seed(42)
        n_samples = 50
        n_features = 3
        
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        # Create separable classes
        y_np = (X_np[:, 0] > 0).astype(np.float32).reshape(-1, 1)
        
        X_paper, y_paper = prepare_data_for_paper(X_np, y_np)
        
        # Train model
        model = LogisticRegressionPaper(learning_rate=0.1, n_iterations=20)
        model.fit(X_paper, y_paper, verbose=False)
        
        # Check that model has weights
        self.assertIsNotNone(model.weights)
    
    def test_logistic_regression_predict(self):
        """Test that logistic regression predictions work."""
        np.random.seed(42)
        n_samples = 50
        n_features = 3
        
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        y_np = (X_np[:, 0] > 0).astype(np.float32).reshape(-1, 1)
        
        X_paper, y_paper = prepare_data_for_paper(X_np, y_np)
        
        # Train model
        model = LogisticRegressionPaper(learning_rate=0.1, n_iterations=30)
        model.fit(X_paper, y_paper, verbose=False)
        
        # Make predictions
        y_pred = model.predict(X_paper)
        
        # Check prediction shape
        self.assertEqual(y_pred.shape, y_np.shape)
        
        # Check that predictions are binary
        unique_preds = np.unique(y_pred)
        self.assertTrue(np.all(np.isin(unique_preds, [0, 1])))
    
    def test_logistic_regression_accuracy(self):
        """Test that logistic regression achieves reasonable accuracy."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        # Create clearly separable classes
        y_np = (X_np[:, 0] + X_np[:, 1] > 0).astype(np.float32).reshape(-1, 1)
        
        X_paper, y_paper = prepare_data_for_paper(X_np, y_np)
        
        # Train model
        model = LogisticRegressionPaper(learning_rate=0.1, n_iterations=50)
        model.fit(X_paper, y_paper, verbose=False)
        
        # Compute accuracy
        accuracy = model.score(X_paper, y_paper)
        
        # Should achieve good accuracy on separable data
        self.assertGreater(accuracy, 0.7)


def run_tests():
    """Run all tests in this module."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMLClassification)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
