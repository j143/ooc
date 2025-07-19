#!/usr/bin/env python
"""
Test runner for Paper matrix operations.
Usage:
    python run_tests.py [test_name]
    
If test_name is provided, only that test will be run.
Available tests:
    - addition: Tests matrix addition (A + B)
    - fused: Tests fused operation ((A + B) * 2)
    - scalar: Tests scalar multiplication (A * 2)
    - all: Runs all tests (default)
"""

import sys
import os
from tests.test_matrix_operations import (
    test_matrix_addition,
    test_fused_add_multiply,
    test_scalar_multiply,
    run_all_tests
)

def print_usage():
    print(__doc__)

if __name__ == "__main__":
    # Check if a specific test was requested
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "addition":
            test_matrix_addition()
        elif test_name == "fused":
            test_fused_add_multiply()
        elif test_name == "scalar":
            test_scalar_multiply()
        elif test_name == "all":
            run_all_tests()
        else:
            print(f"Unknown test: {test_name}")
            print_usage()
    else:
        # Default: run all tests
        run_all_tests()
