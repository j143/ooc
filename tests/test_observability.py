"""
Unit tests for observability features (Priority 8).
"""

import unittest
import os
import tempfile
import shutil
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.observability import (
    configure_logging, ExecutionProfiler, TraceLogger,
    get_profiler
)


class TestObservability(unittest.TestCase):
    """Test cases for observability utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_logging_configuration(self):
        """Test that logging configuration works."""
        log_file = os.path.join(self.test_dir, "test.log")
        logger = configure_logging(level="DEBUG", log_file=log_file)
        
        self.assertIsNotNone(logger)
        logger.info("Test message")
        
        # Verify log file was created
        self.assertTrue(os.path.exists(log_file))
    
    def test_execution_profiler_context_manager(self):
        """Test profiler context manager."""
        profiler = ExecutionProfiler()
        
        with profiler.profile("operation1"):
            import time
            time.sleep(0.01)
        
        with profiler.profile("operation2"):
            time.sleep(0.02)
        
        # Verify entries were recorded
        self.assertEqual(len(profiler.entries), 2)
        self.assertEqual(profiler.entries[0].name, "operation1")
        self.assertEqual(profiler.entries[1].name, "operation2")
        
        # Verify durations
        self.assertGreater(profiler.entries[0].duration, 0.01)
        self.assertGreater(profiler.entries[1].duration, 0.02)
    
    def test_profiler_decorator(self):
        """Test profiler function decorator."""
        profiler = ExecutionProfiler()
        
        @profiler.profile_decorator("test_func")
        def my_function(x):
            import time
            time.sleep(0.01)
            return x * 2
        
        result = my_function(5)
        
        self.assertEqual(result, 10)
        self.assertEqual(len(profiler.entries), 1)
        self.assertEqual(profiler.entries[0].name, "test_func")
        self.assertGreater(profiler.entries[0].duration, 0.01)
    
    def test_profiler_summary_statistics(self):
        """Test profiler summary generation."""
        profiler = ExecutionProfiler()
        
        # Profile same operation multiple times
        for i in range(5):
            with profiler.profile("repeated_op"):
                pass
        
        summary = profiler.get_summary()
        
        self.assertIn("repeated_op", summary)
        self.assertEqual(summary["repeated_op"]["count"], 5)
        self.assertGreater(summary["repeated_op"]["total"], 0)
        self.assertGreater(summary["repeated_op"]["mean"], 0)
    
    def test_profiler_json_export(self):
        """Test exporting profiler results to JSON."""
        profiler = ExecutionProfiler()
        
        with profiler.profile("op1", param=42):
            pass
        
        json_file = os.path.join(self.test_dir, "profile.json")
        profiler.save_json(json_file)
        
        # Verify file was created and contains valid JSON
        self.assertTrue(os.path.exists(json_file))
        with open(json_file) as f:
            data = json.load(f)
        
        self.assertIn("summary", data)
        self.assertIn("entries", data)
        self.assertEqual(len(data["entries"]), 1)
    
    def test_profiler_flame_graph_export(self):
        """Test exporting flame graph data."""
        profiler = ExecutionProfiler()
        
        with profiler.profile("outer"):
            with profiler.profile("inner"):
                pass
        
        flame_file = os.path.join(self.test_dir, "flame.json")
        profiler.save_flame_graph(flame_file)
        
        # Verify file was created
        self.assertTrue(os.path.exists(flame_file))
        with open(flame_file) as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 2)  # outer and inner
    
    def test_profiler_enable_disable(self):
        """Test enabling/disabling profiler."""
        profiler = ExecutionProfiler()
        
        # Profile while enabled
        with profiler.profile("op1"):
            pass
        
        # Disable and profile
        profiler.disable()
        with profiler.profile("op2"):
            pass
        
        # Re-enable and profile
        profiler.enable()
        with profiler.profile("op3"):
            pass
        
        # Should only have recorded op1 and op3
        self.assertEqual(len(profiler.entries), 2)
        self.assertEqual(profiler.entries[0].name, "op1")
        self.assertEqual(profiler.entries[1].name, "op3")
    
    def test_trace_logger_basic(self):
        """Test basic trace logging."""
        trace = TraceLogger()
        
        trace.begin("operation")
        trace.log("step1")
        trace.log("step2")
        trace.end()
        
        # Verify events were recorded
        self.assertEqual(len(trace.events), 4)
        self.assertEqual(trace.events[0]['type'], 'begin')
        self.assertEqual(trace.events[1]['type'], 'event')
        self.assertEqual(trace.events[3]['type'], 'end')
    
    def test_trace_logger_nested(self):
        """Test nested trace logging."""
        trace = TraceLogger()
        
        trace.begin("outer")
        trace.log("outer_step")
        trace.begin("inner")
        trace.log("inner_step")
        trace.end()
        trace.end()
        
        # Verify depth tracking
        self.assertEqual(trace.events[0]['depth'], 0)  # outer begin
        self.assertEqual(trace.events[1]['depth'], 1)  # outer_step
        self.assertEqual(trace.events[2]['depth'], 1)  # inner begin
        self.assertEqual(trace.events[3]['depth'], 2)  # inner_step
    
    def test_trace_logger_save(self):
        """Test saving trace to JSON."""
        trace = TraceLogger()
        
        trace.begin("op")
        trace.log("step")
        trace.end()
        
        trace_file = os.path.join(self.test_dir, "trace.json")
        trace.save(trace_file)
        
        # Verify file
        self.assertTrue(os.path.exists(trace_file))
        with open(trace_file) as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 3)
    
    def test_global_profiler_singleton(self):
        """Test that get_profiler() returns singleton instance."""
        profiler1 = get_profiler()
        profiler2 = get_profiler()
        
        self.assertIs(profiler1, profiler2)


if __name__ == '__main__':
    unittest.main()
