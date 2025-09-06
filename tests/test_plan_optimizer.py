"""
Unit tests for the Plan class and optimizer functionality.
Tests plan construction, PlanNode tree building, and I/O trace generation.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import sys

# Add the parent directory to the path so we can import the paper module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper import PaperMatrix, Plan, EagerNode
from paper.plan import AddNode, MultiplyNode, MultiplyScalarNode
from paper.optimizer import generate_io_trace
from paper.config import TILE_SIZE


class TestPlanAndOptimizer(unittest.TestCase):
    """Test cases for Plan class and optimizer functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.test_shape = (128, 128)  # Small test matrices for fast execution
        self.test_dtype = np.float32
        
        # Create test matrix files
        self.matrix_paths = {}
        for name in ['A', 'B', 'C']:
            filepath = os.path.join(self.test_dir, f"{name}.bin")
            self._create_test_matrix_file(filepath, fill_value=float(ord(name) - ord('A') + 1))
            self.matrix_paths[name] = filepath
    
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_test_matrix_file(self, filepath, fill_value=1.0):
        """Create a test matrix file with specified fill value."""
        data = np.full(self.test_shape, fill_value, dtype=self.test_dtype)
        data.tofile(filepath)
    
    def test_plan_object_construction_addition(self):
        """Test that Plan objects correctly construct PlanNode tree for addition (+)."""
        # Create PaperMatrix handles
        matrix_A = PaperMatrix(self.matrix_paths['A'], self.test_shape, dtype=self.test_dtype, mode='r')
        matrix_B = PaperMatrix(self.matrix_paths['B'], self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Create Plan objects
        plan_A = Plan(EagerNode(matrix_A))
        plan_B = Plan(EagerNode(matrix_B))
        
        # Test addition operation
        add_plan = plan_A + plan_B
        
        # Verify the plan structure
        self.assertIsInstance(add_plan, Plan)
        self.assertIsInstance(add_plan.op, AddNode)
        self.assertIsInstance(add_plan.op.left, EagerNode)
        self.assertIsInstance(add_plan.op.right, EagerNode)
        
        # Verify shapes are propagated correctly
        self.assertEqual(add_plan.shape, self.test_shape)
        
        matrix_A.close()
        matrix_B.close()
    
    def test_plan_object_construction_matrix_multiplication(self):
        """Test that Plan objects correctly construct PlanNode tree for matrix multiplication (@)."""
        # Create compatible matrix shapes for multiplication
        shape_A = (64, 32)
        shape_B = (32, 48)
        
        # Create matrix files with compatible shapes
        path_A = os.path.join(self.test_dir, "matmul_A.bin")
        path_B = os.path.join(self.test_dir, "matmul_B.bin")
        
        data_A = np.full(shape_A, 2.0, dtype=self.test_dtype)
        data_B = np.full(shape_B, 3.0, dtype=self.test_dtype)
        data_A.tofile(path_A)
        data_B.tofile(path_B)
        
        # Create PaperMatrix handles
        matrix_A = PaperMatrix(path_A, shape_A, dtype=self.test_dtype, mode='r')
        matrix_B = PaperMatrix(path_B, shape_B, dtype=self.test_dtype, mode='r')
        
        # Create Plan objects
        plan_A = Plan(EagerNode(matrix_A))
        plan_B = Plan(EagerNode(matrix_B))
        
        # Test matrix multiplication operation
        matmul_plan = plan_A @ plan_B
        
        # Verify the plan structure
        self.assertIsInstance(matmul_plan, Plan)
        self.assertIsInstance(matmul_plan.op, MultiplyNode)
        self.assertIsInstance(matmul_plan.op.left, EagerNode)
        self.assertIsInstance(matmul_plan.op.right, EagerNode)
        
        # Verify result shape is correct for matrix multiplication
        expected_shape = (shape_A[0], shape_B[1])
        self.assertEqual(matmul_plan.shape, expected_shape)
        
        matrix_A.close()
        matrix_B.close()
    
    def test_plan_object_construction_scalar_multiplication(self):
        """Test that Plan objects correctly construct PlanNode tree for scalar multiplication."""
        # Create PaperMatrix handle
        matrix_A = PaperMatrix(self.matrix_paths['A'], self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Create Plan object
        plan_A = Plan(EagerNode(matrix_A))
        
        # Test scalar multiplication
        scalar_plan = plan_A * 2.5
        
        # Verify the plan structure
        self.assertIsInstance(scalar_plan, Plan)
        self.assertIsInstance(scalar_plan.op, MultiplyScalarNode)
        self.assertIsInstance(scalar_plan.op.left, EagerNode)
        self.assertEqual(scalar_plan.op.right, 2.5)
        
        # Verify shape is preserved
        self.assertEqual(scalar_plan.shape, self.test_shape)
        
        matrix_A.close()
    
    def test_plan_object_construction_complex_expression(self):
        """Test complex expression: (A + B) * 2."""
        # Create PaperMatrix handles
        matrix_A = PaperMatrix(self.matrix_paths['A'], self.test_shape, dtype=self.test_dtype, mode='r')
        matrix_B = PaperMatrix(self.matrix_paths['B'], self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Create Plan objects
        plan_A = Plan(EagerNode(matrix_A))
        plan_B = Plan(EagerNode(matrix_B))
        
        # Create complex plan: (A + B) * 2
        complex_plan = (plan_A + plan_B) * 2
        
        # Verify the outer structure
        self.assertIsInstance(complex_plan, Plan)
        self.assertIsInstance(complex_plan.op, MultiplyScalarNode)
        self.assertEqual(complex_plan.op.right, 2)
        
        # Verify the inner structure (addition)
        add_node = complex_plan.op.left
        self.assertIsInstance(add_node, AddNode)
        self.assertIsInstance(add_node.left, EagerNode)
        self.assertIsInstance(add_node.right, EagerNode)
        
        matrix_A.close()
        matrix_B.close()
    
    def test_generate_io_trace_simple_addition(self):
        """Test that optimizer.generate_io_trace produces correct trace for A + B."""
        # Create PaperMatrix handles
        matrix_A = PaperMatrix(self.matrix_paths['A'], self.test_shape, dtype=self.test_dtype, mode='r')
        matrix_B = PaperMatrix(self.matrix_paths['B'], self.test_shape, dtype=self.test_dtype, mode='r')
        
        # Create Plan objects
        plan_A = Plan(EagerNode(matrix_A))
        plan_B = Plan(EagerNode(matrix_B))
        
        # Create addition plan
        add_plan = plan_A + plan_B
        
        # Generate I/O trace
        io_trace = generate_io_trace(add_plan)
        
        # Verify trace is not empty
        self.assertGreater(len(io_trace), 0)
        
        # Verify trace contains expected files
        files_in_trace = set(item[0] for item in io_trace)
        self.assertIn('A.bin', files_in_trace)
        self.assertIn('B.bin', files_in_trace)
        
        # For our small test matrices (128x128), we expect 1 tile per matrix
        # Each tile access should appear in the trace
        expected_tiles = []
        for r in range(0, self.test_shape[0], TILE_SIZE):
            for c in range(0, self.test_shape[1], TILE_SIZE):
                expected_tiles.append(('A.bin', r, c))
                expected_tiles.append(('B.bin', r, c))
        
        # Verify all expected tiles are in the trace
        for tile in expected_tiles:
            self.assertIn(tile, io_trace)
        
        matrix_A.close()
        matrix_B.close()
    
    def test_generate_io_trace_matrix_multiplication(self):
        """Test that optimizer.generate_io_trace produces correct trace for A @ B."""
        # Create compatible matrix shapes for multiplication
        shape_A = (64, 32)
        shape_B = (32, 64)
        
        # Create matrix files
        path_A = os.path.join(self.test_dir, "trace_A.bin")
        path_B = os.path.join(self.test_dir, "trace_B.bin")
        
        data_A = np.full(shape_A, 2.0, dtype=self.test_dtype)
        data_B = np.full(shape_B, 3.0, dtype=self.test_dtype)
        data_A.tofile(path_A)
        data_B.tofile(path_B)
        
        # Create PaperMatrix handles
        matrix_A = PaperMatrix(path_A, shape_A, dtype=self.test_dtype, mode='r')
        matrix_B = PaperMatrix(path_B, shape_B, dtype=self.test_dtype, mode='r')
        
        # Create Plan objects
        plan_A = Plan(EagerNode(matrix_A))
        plan_B = Plan(EagerNode(matrix_B))
        
        # Create matrix multiplication plan
        matmul_plan = plan_A @ plan_B
        
        # Generate I/O trace
        io_trace = generate_io_trace(matmul_plan)
        
        # Verify trace is not empty and contains both matrices
        self.assertGreater(len(io_trace), 0)
        
        files_in_trace = set(item[0] for item in io_trace)
        self.assertIn('trace_A.bin', files_in_trace)
        self.assertIn('trace_B.bin', files_in_trace)
        
        matrix_A.close()
        matrix_B.close()
    
    def test_generate_io_trace_non_empty_for_simple_plan(self):
        """Test that optimizer.generate_io_trace produces a non-empty trace for simple A @ B plan."""
        # Create simple matrix multiplication that should definitely produce a trace
        shape_A = (32, 32)
        shape_B = (32, 32)
        
        # Create matrix files
        path_A = os.path.join(self.test_dir, "simple_A.bin")
        path_B = os.path.join(self.test_dir, "simple_B.bin")
        
        data_A = np.ones(shape_A, dtype=self.test_dtype)
        data_B = np.ones(shape_B, dtype=self.test_dtype)
        data_A.tofile(path_A)
        data_B.tofile(path_B)
        
        # Create PaperMatrix handles
        matrix_A = PaperMatrix(path_A, shape_A, dtype=self.test_dtype, mode='r')
        matrix_B = PaperMatrix(path_B, shape_B, dtype=self.test_dtype, mode='r')
        
        # Create Plan objects
        plan_A = Plan(EagerNode(matrix_A))
        plan_B = Plan(EagerNode(matrix_B))
        
        # Create simple plan
        simple_plan = plan_A @ plan_B
        
        # Generate I/O trace
        io_trace = generate_io_trace(simple_plan)
        
        # The most basic requirement: trace should not be empty
        self.assertGreater(len(io_trace), 0, "I/O trace should not be empty for A @ B plan")
        
        # Verify trace entries have correct format (filename, r_start, c_start)
        for entry in io_trace:
            self.assertEqual(len(entry), 3, "Each trace entry should have 3 elements")
            filename, r_start, c_start = entry
            self.assertIsInstance(filename, str, "Filename should be string")
            self.assertIsInstance(r_start, int, "r_start should be integer")
            self.assertIsInstance(c_start, int, "c_start should be integer")
        
        matrix_A.close()
        matrix_B.close()


if __name__ == '__main__':
    unittest.main()