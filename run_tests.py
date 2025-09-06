#!/usr/bin/env python
"""
Test runner for Paper matrix operations using Python's unittest framework.

This script uses unittest's TestLoader to automatically discover and execute 
all tests in the tests/ directory. It provides a professional, structured 
testing workflow suitable for CI/CD integration.

Usage:
    python run_tests.py

Exit Status:
    0: All tests passed
    Non-zero: One or more tests failed
"""

import sys
import unittest
import os

def main():
    """Main entry point for the test runner."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up test discovery
    loader = unittest.TestLoader()
    start_dir = os.path.join(script_dir, 'tests')
    
    # Discover all test files in the tests directory
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create a test runner with verbose output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    # Run the tests
    print("Paper Matrix Framework - Test Suite")
    print("=" * 50)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Return appropriate exit code for CI/CD integration
    if result.wasSuccessful():
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())