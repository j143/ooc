# paper/config.py
"""
Centralized configuration for Paper matrix library.
This module provides a single source of truth for all configurable parameters.
"""

# Core computation parameters
TILE_SIZE = 1024  # Size of matrix tiles for out-of-core operations

# Buffer manager configuration
DEFAULT_CACHE_SIZE_TILES = 64  # Default number of tiles to cache in memory

# Performance tuning parameters
DEFAULT_BLOCK_SIZE = TILE_SIZE  # Alias for backward compatibility