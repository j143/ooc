"""
Tests for ML classification functionality.
"""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_classification import (
    generate_classification_labels,
    train_classifier,
    evaluate_classifier
)


class TestMLClassification(unittest.TestCase):
    """Test ML classification utilities."""
    
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
    
    def test_train_classifier_basic(self):
        """Test that classifier can be trained."""
        # Create simple synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(np.int32)  # Simple linear separable
        
        clf = train_classifier(X_train, y_train, max_iter=100)
        
        # Check that classifier was trained
        self.assertTrue(hasattr(clf, 'coef_'))
        self.assertTrue(hasattr(clf, 'intercept_'))
    
    def test_train_classifier_predictions(self):
        """Test that trained classifier can make predictions."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(np.int32)
        
        clf = train_classifier(X_train, y_train, max_iter=100)
        
        # Make predictions
        X_test = np.random.randn(20, n_features).astype(np.float32)
        predictions = clf.predict(X_test)
        
        # Check predictions are valid
        self.assertEqual(len(predictions), 20)
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))
    
    def test_evaluate_classifier_metrics(self):
        """Test that evaluation returns expected metrics."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Create linearly separable data
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(np.int32)
        
        X_test = np.random.randn(50, n_features).astype(np.float32)
        y_test = (X_test[:, 0] > 0).astype(np.int32)
        
        clf = train_classifier(X_train, y_train, max_iter=100)
        metrics = evaluate_classifier(clf, X_test, y_test)
        
        # Check that metrics are returned
        self.assertIn('accuracy', metrics)
        self.assertIn('roc_auc', metrics)
        
        # Check that metrics are in valid range
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreaterEqual(metrics['roc_auc'], 0.0)
        self.assertLessEqual(metrics['roc_auc'], 1.0)
        
        # For linearly separable data, accuracy should be high
        self.assertGreater(metrics['accuracy'], 0.7)
        self.assertGreater(metrics['roc_auc'], 0.7)
    
    def test_evaluate_classifier_perfect_prediction(self):
        """Test evaluation with perfect predictions."""
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        
        # Create perfectly separable data
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        X_train[:25, 0] = 10  # Class 1
        X_train[25:, 0] = -10  # Class 0
        y_train = np.array([1] * 25 + [0] * 25, dtype=np.int32)
        
        X_test = np.random.randn(20, n_features).astype(np.float32)
        X_test[:10, 0] = 10  # Class 1
        X_test[10:, 0] = -10  # Class 0
        y_test = np.array([1] * 10 + [0] * 10, dtype=np.int32)
        
        clf = train_classifier(X_train, y_train, max_iter=100)
        metrics = evaluate_classifier(clf, X_test, y_test)
        
        # Should have perfect or near-perfect accuracy
        self.assertGreater(metrics['accuracy'], 0.95)
        self.assertGreater(metrics['roc_auc'], 0.95)


def run_tests():
    """Run all tests in this module."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMLClassification)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
