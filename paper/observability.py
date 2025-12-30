"""
Observability utilities for the Paper framework.

This module provides:
- Structured logging configuration
- Performance profiling decorators and context managers
- Execution tracing and visualization
"""

import logging
import time
import functools
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict
import json


# ============================================================================
# Logging Configuration
# ============================================================================

def configure_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure structured logging for the Paper framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
    """
    log_level = getattr(logging, level.upper())
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger for paper module
    paper_logger = logging.getLogger('paper')
    paper_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    paper_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(detailed_formatter)
        paper_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    paper_logger.propagate = False
    
    return paper_logger


# ============================================================================
# Performance Profiling
# ============================================================================

@dataclass
class ProfileEntry:
    """Single profile measurement."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self):
        """Mark this entry as complete and calculate duration."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'duration': self.duration,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'metadata': self.metadata
        }


class ExecutionProfiler:
    """
    Profiler for tracking execution performance.
    
    Example:
        profiler = ExecutionProfiler()
        
        with profiler.profile("operation"):
            # ... do work ...
            pass
        
        profiler.print_summary()
        profiler.save_flame_graph("profile.json")
    """
    
    def __init__(self):
        self.entries: List[ProfileEntry] = []
        self.current_stack: List[ProfileEntry] = []
        self.aggregated: Dict[str, List[float]] = defaultdict(list)
        self._enabled = True
    
    @contextmanager
    def profile(self, name: str, **metadata):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name of the operation being profiled
            **metadata: Additional metadata to attach
        """
        if not self._enabled:
            yield
            return
        
        entry = ProfileEntry(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata
        )
        
        self.current_stack.append(entry)
        
        try:
            yield entry
        finally:
            entry.complete()
            self.entries.append(entry)
            self.aggregated[name].append(entry.duration)
            self.current_stack.pop()
    
    def profile_decorator(self, name: Optional[str] = None):
        """
        Decorator for profiling function calls.
        
        Example:
            @profiler.profile_decorator("my_function")
            def my_function(x, y):
                return x + y
        """
        def decorator(func):
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile(profile_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated statistics for all profiled operations.
        
        Returns:
            Dictionary mapping operation names to statistics
        """
        summary = {}
        for name, durations in self.aggregated.items():
            if durations:
                summary[name] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'mean': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        return summary
    
    def print_summary(self):
        """Print a formatted summary of profiling results."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("EXECUTION PROFILE SUMMARY")
        print("="*80)
        print(f"{'Operation':<40} {'Count':>8} {'Total (s)':>12} {'Mean (s)':>12}")
        print("-"*80)
        
        # Sort by total time descending
        sorted_ops = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for name, stats in sorted_ops:
            print(f"{name:<40} {stats['count']:>8} {stats['total']:>12.4f} {stats['mean']:>12.6f}")
        
        print("="*80)
        
        # Overall statistics
        total_time = sum(stats['total'] for stats in summary.values())
        total_count = sum(stats['count'] for stats in summary.values())
        print(f"{'TOTAL':<40} {total_count:>8} {total_time:>12.4f}")
        print("="*80 + "\n")
    
    def save_json(self, filepath: str):
        """
        Save profiling results to JSON file.
        
        Args:
            filepath: Path to save JSON data
        """
        data = {
            'summary': self.get_summary(),
            'entries': [entry.to_dict() for entry in self.entries]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_flame_graph(self, filepath: str):
        """
        Save data suitable for flame graph visualization.
        
        The output format is compatible with tools like speedscope.
        
        Args:
            filepath: Path to save flame graph data
        """
        # Build a simple stack trace format
        flame_data = []
        
        for entry in self.entries:
            flame_data.append({
                'name': entry.name,
                'value': entry.duration * 1000,  # Convert to milliseconds
                'start': entry.start_time * 1000,
                'end': entry.end_time * 1000 if entry.end_time else None,
                'metadata': entry.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(flame_data, f, indent=2)
    
    def reset(self):
        """Clear all profiling data."""
        self.entries.clear()
        self.current_stack.clear()
        self.aggregated.clear()
    
    def enable(self):
        """Enable profiling."""
        self._enabled = True
    
    def disable(self):
        """Disable profiling."""
        self._enabled = False


# Global profiler instance
_global_profiler = ExecutionProfiler()

def get_profiler() -> ExecutionProfiler:
    """Get the global profiler instance."""
    return _global_profiler


# ============================================================================
# Trace Visualization
# ============================================================================

class TraceLogger:
    """
    Logger for execution traces with hierarchical formatting.
    
    Example:
        trace = TraceLogger()
        trace.begin("Operation A")
        trace.log("Step 1")
        trace.begin("Sub-operation")
        trace.log("Sub-step")
        trace.end()
        trace.end()
        trace.print()
    """
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.depth = 0
        self.logger = logging.getLogger('paper.trace')
    
    def begin(self, name: str, **metadata):
        """Begin a new operation scope."""
        event = {
            'type': 'begin',
            'name': name,
            'depth': self.depth,
            'timestamp': time.perf_counter(),
            'metadata': metadata
        }
        self.events.append(event)
        self.logger.debug("  " * self.depth + f"▶ {name}")
        self.depth += 1
    
    def end(self, **metadata):
        """End the current operation scope."""
        self.depth = max(0, self.depth - 1)
        event = {
            'type': 'end',
            'depth': self.depth,
            'timestamp': time.perf_counter(),
            'metadata': metadata
        }
        self.events.append(event)
    
    def log(self, message: str, **metadata):
        """Log an event within the current scope."""
        event = {
            'type': 'event',
            'message': message,
            'depth': self.depth,
            'timestamp': time.perf_counter(),
            'metadata': metadata
        }
        self.events.append(event)
        self.logger.debug("  " * self.depth + f"• {message}")
    
    def print(self):
        """Print formatted trace."""
        print("\n" + "="*80)
        print("EXECUTION TRACE")
        print("="*80)
        
        start_time = self.events[0]['timestamp'] if self.events else 0
        
        for event in self.events:
            indent = "  " * event['depth']
            elapsed = event['timestamp'] - start_time
            
            if event['type'] == 'begin':
                print(f"{elapsed:8.3f}s {indent}▶ {event['name']}")
            elif event['type'] == 'event':
                print(f"{elapsed:8.3f}s {indent}• {event['message']}")
        
        print("="*80 + "\n")
    
    def save(self, filepath: str):
        """Save trace to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.events, f, indent=2)
    
    def reset(self):
        """Clear all trace data."""
        self.events.clear()
        self.depth = 0
