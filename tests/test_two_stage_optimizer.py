"""
Unit tests for the two-stage optimizer (Priority 2).

Tests the analyze → rewrite → execute pipeline.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.core import PaperMatrix
from paper.plan import Plan, EagerNode
from paper import optimizer
from paper.optimizer import (
    analyze, rewrite, execute_plan, estimate_cost,
    FusionPattern, MatchResult
)
from paper.buffer import BufferManager


class TestTwoStageOptimizer(unittest.TestCase):
    """Test cases for the two-stage optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_shape = (128, 128)
        self.test_dtype = np.float32
        
        # Create test matrices
        self.matrix_A_path = os.path.join(self.test_dir, "A.bin")
        self.matrix_B_path = os.path.join(self.test_dir, "B.bin")
        
        np.random.seed(42)
        self.data_A = np.random.rand(*self.test_shape).astype(self.test_dtype)
        self.data_B = np.random.rand(*self.test_shape).astype(self.test_dtype)
        
        self.data_A.tofile(self.matrix_A_path)
        self.data_B.tofile(self.matrix_B_path)
        
        self.A = PaperMatrix(self.matrix_A_path, self.test_shape, dtype=self.test_dtype, mode='r')
        self.B = PaperMatrix(self.matrix_B_path, self.test_shape, dtype=self.test_dtype, mode='r')
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'A'):
            self.A.close()
        if hasattr(self, 'B'):
            self.B.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_analyze_detects_fused_add_multiply_pattern(self):
        """Test that analyze() detects (A + B) * scalar pattern without executing."""
        # Build plan: (A + B) * 2.5
        plan_A = Plan(EagerNode(self.A))
        plan_B = Plan(EagerNode(self.B))
        plan = (plan_A + plan_B) * 2.5
        
        # Analyze (should not execute anything)
        io_trace, match_result = analyze(plan)
        
        # Verify pattern was detected
        self.assertTrue(match_result.is_fusable)
        self.assertEqual(match_result.pattern, FusionPattern.FUSED_ADD_MULTIPLY)
        self.assertEqual(match_result.parameters['scalar'], 2.5)
        self.assertEqual(len(match_result.input_shapes), 2)
        
        # Verify I/O trace was generated
        self.assertIsInstance(io_trace, list)
        self.assertGreater(len(io_trace), 0)
    
    def test_analyze_no_pattern_detection(self):
        """Test that analyze() returns NONE for non-fusable patterns."""
        # Simple addition (not fusable)
        plan_A = Plan(EagerNode(self.A))
        plan_B = Plan(EagerNode(self.B))
        plan = plan_A + plan_B
        
        # Analyze
        io_trace, match_result = analyze(plan)
        
        # Verify no pattern
        self.assertFalse(match_result.is_fusable)
        self.assertEqual(match_result.pattern, FusionPattern.NONE)
    
    def test_cost_estimate_for_fused_operation(self):
        """Test that cost estimation works for fused operations."""
        plan_A = Plan(EagerNode(self.A))
        plan_B = Plan(EagerNode(self.B))
        plan = (plan_A + plan_B) * 2.5
        
        _, match_result = analyze(plan)
        cost = estimate_cost(plan, match_result)
        
        # Verify cost estimate structure
        self.assertGreater(cost.io_operations, 0)
        self.assertGreater(cost.compute_operations, 0)
        self.assertGreater(cost.estimated_io_bytes, 0)
        self.assertGreater(cost.cache_benefit, 0)  # Fusion provides benefit
        self.assertGreater(cost.total_cost, 0)
    
    def test_cost_estimate_for_unfused_operation(self):
        """Test that cost estimation works for unfused operations."""
        plan_A = Plan(EagerNode(self.A))
        plan_B = Plan(EagerNode(self.B))
        plan = plan_A + plan_B
        
        _, match_result = analyze(plan)
        cost = estimate_cost(plan, match_result)
        
        # Verify cost estimate
        self.assertGreater(cost.io_operations, 0)
        self.assertEqual(cost.cache_benefit, 0.0)  # No fusion benefit
    
    def test_three_stage_pipeline_execution(self):
        """Test the full three-stage pipeline: analyze → rewrite → execute."""
        plan_A = Plan(EagerNode(self.A))
        plan_B = Plan(EagerNode(self.B))
        plan = (plan_A + plan_B) * 2.5
        
        output_path = os.path.join(self.test_dir, "result.bin")
        
        # Stage 1: Analyze
        io_trace, match_result = analyze(plan)
        self.assertTrue(match_result.is_fusable)
        
        # Stage 2: Rewrite
        rewritten_plan = rewrite(plan, match_result)
        self.assertIsNotNone(rewritten_plan)
        
        # Stage 3: Execute
        buffer_manager = BufferManager(max_cache_size_tiles=8, io_trace=io_trace)
        result = execute_plan(rewritten_plan, match_result, output_path, buffer_manager)
        
        # Verify result
        self.assertIsInstance(result, PaperMatrix)
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        expected = (self.data_A + self.data_B) * 2.5
        np.testing.assert_allclose(result_data, expected, rtol=1e-5)
    
    def test_legacy_execute_backward_compatibility(self):
        """Test that legacy execute() still works via three-stage pipeline."""
        plan_A = Plan(EagerNode(self.A))
        plan_B = Plan(EagerNode(self.B))
        plan = (plan_A + plan_B) * 2.5
        
        output_path = os.path.join(self.test_dir, "result_legacy.bin")
        
        # Use legacy execute()
        result = optimizer.execute(plan, output_path, None)
        
        # Verify it produces correct results
        self.assertIsInstance(result, PaperMatrix)
        result_data = np.fromfile(output_path, dtype=self.test_dtype).reshape(self.test_shape)
        
        expected = (self.data_A + self.data_B) * 2.5
        np.testing.assert_allclose(result_data, expected, rtol=1e-5)
    
    def test_analyze_double_scalar_pattern(self):
        """Test detection of (A * scalar1) * scalar2 pattern."""
        plan_A = Plan(EagerNode(self.A))
        plan = (plan_A * 2.0) * 3.0
        
        _, match_result = analyze(plan)
        
        self.assertTrue(match_result.is_fusable)
        self.assertEqual(match_result.pattern, FusionPattern.FUSED_DOUBLE_SCALAR)
        self.assertEqual(match_result.parameters['scalar1'], 2.0)
        self.assertEqual(match_result.parameters['scalar2'], 3.0)
    
    def test_no_execute_during_analyze(self):
        """Critical test: analyze() must not call execute() on nodes."""
        plan_A = Plan(EagerNode(self.A))
        plan_B = Plan(EagerNode(self.B))
        plan = (plan_A + plan_B) * 2.5
        
        # Track if execute was called
        original_execute = self.A.get_tile
        execute_called = []
        
        def tracked_get_tile(*args, **kwargs):
            execute_called.append(True)
            return original_execute(*args, **kwargs)
        
        self.A.get_tile = tracked_get_tile
        
        # Analyze should not trigger any get_tile calls
        _, match_result = analyze(plan)
        
        # Verify analyze did not execute
        self.assertEqual(len(execute_called), 0, 
                        "analyze() should not call execute() or get_tile()")
        self.assertTrue(match_result.is_fusable)


if __name__ == '__main__':
    unittest.main()
