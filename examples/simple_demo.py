#!/usr/bin/env python3
"""
Simple NumPy API Demo

A minimal example showing how to use Paper's NumPy-compatible API.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper import numpy_api as pnp
import numpy as np

# Create arrays using familiar NumPy syntax
a = pnp.array([[1, 2], [3, 4]], dtype=np.float32)
b = pnp.array([[5, 6], [7, 8]], dtype=np.float32)

print("Array a:")
print(a.to_numpy())

print("\nArray b:")
print(b.to_numpy())

# Build a computation plan (lazy evaluation)
result = (a + b) * 2

print(f"\nComputation plan built: {result}")
print(f"Shape: {result.shape}")

# Execute the plan
computed = result.compute()

print("\nComputed result:")
print(computed.to_numpy())

print("\n✓ Success! The NumPy-compatible API works seamlessly.")
