import unittest
import os
import sys
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.core import PaperMatrix
from paper.plan import Plan, EagerNode
from benchmarks.utils import create_matrix_file
from paper.config import TILE_SIZE

class TestDeterminism(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and matrices for testing."""
        self.test_dir = "test_determinism_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Use a shape that creates multiple tiles to allow for varied execution order
        self.shape = (TILE_SIZE * 4, TILE_SIZE * 4)
        self.path_A = os.path.join(self.test_dir, "A.bin")
        self.path_B = os.path.join(self.test_dir, "B.bin")
        
        create_matrix_file(self.path_A, self.shape)
        create_matrix_file(self.path_B, self.shape)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def _get_execution_signature(self):
        """
        Runs the A @ B computation and returns a "signature" of the cache events.
        The signature is a list of (event_type, tile_key) tuples, ignoring timestamps.
        """
        A_handle = PaperMatrix(self.path_A, self.shape, mode='r')
        B_handle = PaperMatrix(self.path_B, self.shape, mode='r')

        plan_A = Plan(EagerNode(A_handle))
        plan_B = Plan(EagerNode(B_handle))
        matmul_plan = plan_A @ plan_B

        # Use a small cache to force I/O and interaction
        _, buffer_manager = matmul_plan.compute(
            os.path.join(self.test_dir, "C.bin"),
            cache_size_tiles=16
        )
        
        log = buffer_manager.get_log()
        
        A_handle.close()
        B_handle.close()
        
        # The signature ignores timestamps and cache size, focusing only on the sequence of events
        return [(event[1], event[2]) for event in log]

    def test_parallel_execution_is_non_deterministic(self):
        """
        Tests that two identical runs produce different event traces
        due to the non-deterministic nature of `as_completed`.
        """
        print("\n--- Testing for Non-Determinism ---")
        print("Running computation for the first time...")
        signature_1 = self._get_execution_signature()
        
        print("Running computation for the second time...")
        signature_2 = self._get_execution_signature()
        
        # The core of the test: we expect the two runs to have different event orders.
        # NOTE: There is a small chance this test could fail if, by coincidence,
        # the OS schedules the threads in the exact same order twice. A consistent
        # pass proves that the execution order is variable.
        self.assertNotEqual(
            signature_1, 
            signature_2,
            "Execution was deterministic, which is unexpected with the current parallel backend."
        )
        print("✓ Test passed: The two runs produced different event traces, confirming non-determinism.")

if __name__ == '__main__':
    unittest.main()

