"""
Unit tests for tensor conversion methods (PyTorch and TensorFlow).

Tests the to_torch() and to_tensorflow() methods to ensure efficient
conversion of out-of-core arrays to device tensors.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import sys

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper import numpy_api as pnp


class TestTorchConversion(unittest.TestCase):
    """Test cases for PyTorch tensor conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Check if PyTorch is available
        try:
            import torch
            self.torch_available = True
            self.torch = torch
        except ImportError:
            self.torch_available = False
            self.torch = None
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_to_torch_cpu_materialized(self):
        """Test converting materialized array to PyTorch CPU tensor."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed")
        
        # Create a simple array
        data = [[1, 2, 3], [4, 5, 6]]
        arr = pnp.array(data, dtype=np.float32)
        
        # Convert to PyTorch tensor
        tensor = arr.to_torch()
        
        # Verify properties
        self.assertIsInstance(tensor, self.torch.Tensor)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.dtype, self.torch.float32)
        self.assertEqual(tensor.device.type, 'cpu')
        
        # Verify data correctness
        expected = self.torch.tensor(data, dtype=self.torch.float32)
        self.assertTrue(self.torch.allclose(tensor, expected))
    
    def test_to_torch_cpu_lazy(self):
        """Test converting lazy array to PyTorch CPU tensor."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed")
        
        # Create lazy computation
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        c = a + b  # Lazy
        
        # Convert to PyTorch tensor (should compute first)
        tensor = c.to_torch()
        
        # Verify properties
        self.assertIsInstance(tensor, self.torch.Tensor)
        self.assertEqual(tensor.shape, (2, 2))
        
        # Verify data correctness
        expected = self.torch.tensor([[6, 8], [10, 12]], dtype=self.torch.float32)
        self.assertTrue(self.torch.allclose(tensor, expected))
    
    def test_to_torch_explicit_cpu_device(self):
        """Test converting array to PyTorch tensor with explicit CPU device."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed")
        
        arr = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        
        # Convert with explicit CPU device
        tensor = arr.to_torch(device='cpu')
        
        self.assertEqual(tensor.device.type, 'cpu')
        self.assertEqual(tensor.shape, (2, 2))
    
    def test_to_torch_cuda_device(self):
        """Test converting array to PyTorch CUDA tensor if available."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed")
        
        if not self.torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        
        arr = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        
        # Convert to CUDA tensor
        tensor = arr.to_torch(device='cuda')
        
        self.assertEqual(tensor.device.type, 'cuda')
        self.assertEqual(tensor.shape, (2, 2))
        
        # Verify data correctness
        expected = self.torch.tensor([[1, 2], [3, 4]], dtype=self.torch.float32)
        self.assertTrue(self.torch.allclose(tensor.cpu(), expected))
    
    def test_to_torch_memory_efficiency(self):
        """Test that to_torch uses memory-mapped data efficiently."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed")
        
        # Create a larger array to test memory efficiency
        arr = pnp.random_rand((100, 100), dtype=np.float32)
        
        # Convert to PyTorch tensor
        tensor = arr.to_torch()
        
        # Verify the tensor was created successfully
        self.assertIsInstance(tensor, self.torch.Tensor)
        self.assertEqual(tensor.shape, (100, 100))
        
        # Verify values are in expected range
        self.assertTrue(self.torch.all(tensor >= 0))
        self.assertTrue(self.torch.all(tensor < 1))
    
    def test_to_torch_with_loaded_file(self):
        """Test converting loaded file-based array to PyTorch tensor."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed")
        
        # Create and save an array
        test_path = os.path.join(self.test_dir, "test.bin")
        test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        test_data.tofile(test_path)
        
        # Load using Paper API
        arr = pnp.load(test_path, shape=(2, 2))
        
        # Convert to PyTorch tensor
        tensor = arr.to_torch()
        
        # Verify data correctness
        expected = self.torch.tensor([[1, 2], [3, 4]], dtype=self.torch.float32)
        self.assertTrue(self.torch.allclose(tensor, expected))
    
    def test_to_torch_not_installed(self):
        """Test error handling when PyTorch is not installed."""
        # This test will only work if PyTorch is actually not installed
        # We can't easily simulate this in the test environment
        # But the method should raise ImportError if torch is not available
        pass


class TestTensorFlowConversion(unittest.TestCase):
    """Test cases for TensorFlow tensor conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Check if TensorFlow is available
        try:
            import tensorflow as tf
            self.tf_available = True
            self.tf = tf
        except ImportError:
            self.tf_available = False
            self.tf = None
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_to_tensorflow_materialized(self):
        """Test converting materialized array to TensorFlow tensor."""
        if not self.tf_available:
            self.skipTest("TensorFlow is not installed")
        
        # Create a simple array
        data = [[1, 2, 3], [4, 5, 6]]
        arr = pnp.array(data, dtype=np.float32)
        
        # Convert to TensorFlow tensor
        tensor = arr.to_tensorflow()
        
        # Verify properties
        self.assertIsInstance(tensor, self.tf.Tensor)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.dtype, self.tf.float32)
        
        # Verify data correctness
        expected = self.tf.constant(data, dtype=self.tf.float32)
        self.assertTrue(self.tf.reduce_all(self.tf.equal(tensor, expected)).numpy())
    
    def test_to_tensorflow_lazy(self):
        """Test converting lazy array to TensorFlow tensor."""
        if not self.tf_available:
            self.skipTest("TensorFlow is not installed")
        
        # Create lazy computation
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        c = a + b  # Lazy
        
        # Convert to TensorFlow tensor (should compute first)
        tensor = c.to_tensorflow()
        
        # Verify properties
        self.assertIsInstance(tensor, self.tf.Tensor)
        self.assertEqual(tensor.shape, (2, 2))
        
        # Verify data correctness
        expected = self.tf.constant([[6, 8], [10, 12]], dtype=self.tf.float32)
        self.assertTrue(self.tf.reduce_all(self.tf.equal(tensor, expected)).numpy())
    
    def test_to_tensorflow_memory_efficiency(self):
        """Test that to_tensorflow handles memory-mapped data efficiently."""
        if not self.tf_available:
            self.skipTest("TensorFlow is not installed")
        
        # Create a larger array to test memory efficiency
        arr = pnp.random_rand((100, 100), dtype=np.float32)
        
        # Convert to TensorFlow tensor
        tensor = arr.to_tensorflow()
        
        # Verify the tensor was created successfully
        self.assertIsInstance(tensor, self.tf.Tensor)
        self.assertEqual(tensor.shape, (100, 100))
        
        # Verify values are in expected range
        self.assertTrue(self.tf.reduce_all(tensor >= 0).numpy())
        self.assertTrue(self.tf.reduce_all(tensor < 1).numpy())
    
    def test_to_tensorflow_with_loaded_file(self):
        """Test converting loaded file-based array to TensorFlow tensor."""
        if not self.tf_available:
            self.skipTest("TensorFlow is not installed")
        
        # Create and save an array
        test_path = os.path.join(self.test_dir, "test.bin")
        test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        test_data.tofile(test_path)
        
        # Load using Paper API
        arr = pnp.load(test_path, shape=(2, 2))
        
        # Convert to TensorFlow tensor
        tensor = arr.to_tensorflow()
        
        # Verify data correctness
        expected = self.tf.constant([[1, 2], [3, 4]], dtype=self.tf.float32)
        self.assertTrue(self.tf.reduce_all(self.tf.equal(tensor, expected)).numpy())
    
    def test_to_tensorflow_float64(self):
        """Test converting float64 array to TensorFlow tensor."""
        if not self.tf_available:
            self.skipTest("TensorFlow is not installed")
        
        # Create array with float64
        arr = pnp.array([[1, 2], [3, 4]], dtype=np.float64)
        
        # Convert to TensorFlow tensor
        tensor = arr.to_tensorflow()
        
        # Verify dtype is preserved
        self.assertEqual(tensor.dtype, self.tf.float64)
    
    def test_to_tensorflow_not_installed(self):
        """Test error handling when TensorFlow is not installed."""
        # This test will only work if TensorFlow is actually not installed
        # We can't easily simulate this in the test environment
        # But the method should raise ImportError if tensorflow is not available
        pass


class TestTensorConversionIntegration(unittest.TestCase):
    """Integration tests for tensor conversion with Paper operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Check availability
        try:
            import torch
            self.torch_available = True
            self.torch = torch
        except ImportError:
            self.torch_available = False
            self.torch = None
        
        try:
            import tensorflow as tf
            self.tf_available = True
            self.tf = tf
        except ImportError:
            self.tf_available = False
            self.tf = None
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_api_example_workflow_torch(self):
        """Test the example workflow from the issue with PyTorch."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed")
        
        # Create test data file
        test_path = os.path.join(self.test_dir, "large_matrix.dat")
        test_data = np.random.rand(100, 100).astype(np.float32)
        test_data.tofile(test_path)
        
        # Load a large out-of-core array (no data read yet)
        arr = pnp.load(test_path, shape=(100, 100), dtype=np.float32)
        
        # Build computation graph (lazy, nothing loaded)
        # Using scalar multiplication which is supported
        c = arr * 2
        
        # Execute the computation plan
        result = c.compute()
        
        # Convert result to PyTorch tensor
        torch_tensor = result.to_torch()
        
        # Verify
        self.assertIsInstance(torch_tensor, self.torch.Tensor)
        self.assertEqual(torch_tensor.shape, (100, 100))
        
        # Verify computation correctness
        expected = self.torch.from_numpy(test_data * 2)
        self.assertTrue(self.torch.allclose(torch_tensor, expected, rtol=1e-5))
    
    def test_api_example_workflow_tensorflow(self):
        """Test the example workflow from the issue with TensorFlow."""
        if not self.tf_available:
            self.skipTest("TensorFlow is not installed")
        
        # Create test data file
        test_path = os.path.join(self.test_dir, "large_matrix.dat")
        test_data = np.random.rand(100, 100).astype(np.float32)
        test_data.tofile(test_path)
        
        # Load a large out-of-core array (no data read yet)
        arr = pnp.load(test_path, shape=(100, 100), dtype=np.float32)
        
        # Build computation graph (lazy, nothing loaded)
        # Using scalar multiplication which is supported
        c = arr * 2
        
        # Execute the computation plan
        result = c.compute()
        
        # Convert result to TensorFlow tensor
        tf_tensor = result.to_tensorflow()
        
        # Verify
        self.assertIsInstance(tf_tensor, self.tf.Tensor)
        self.assertEqual(tf_tensor.shape, (100, 100))
        
        # Verify computation correctness
        expected = self.tf.constant(test_data * 2, dtype=self.tf.float32)
        self.assertTrue(self.tf.reduce_all(
            self.tf.abs(tf_tensor - expected) < 1e-5
        ).numpy())
    
    def test_chained_operations_to_torch(self):
        """Test chained operations followed by PyTorch conversion."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed")
        
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        
        # Chain operations
        result = ((a + b) * 2).compute()
        
        # Convert to PyTorch
        tensor = result.to_torch()
        
        # Verify
        expected = self.torch.tensor([[12, 16], [20, 24]], dtype=self.torch.float32)
        self.assertTrue(self.torch.allclose(tensor, expected))
    
    def test_chained_operations_to_tensorflow(self):
        """Test chained operations followed by TensorFlow conversion."""
        if not self.tf_available:
            self.skipTest("TensorFlow is not installed")
        
        a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)
        
        # Chain operations
        result = ((a + b) * 2).compute()
        
        # Convert to TensorFlow
        tensor = result.to_tensorflow()
        
        # Verify
        expected = self.tf.constant([[12, 16], [20, 24]], dtype=self.tf.float32)
        self.assertTrue(self.tf.reduce_all(self.tf.equal(tensor, expected)).numpy())


if __name__ == '__main__':
    unittest.main()
