# benchmarks/utils.py
import numpy as np
import os
import time
import sys
import threading
import psutil

# Add the project root to the Python path to allow importing 'paper'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paper.core import PaperMatrix

# --- Configuration ---
TILE_SIZE = 1024

def create_matrix_file(filepath, shape, fill_value=None, tile_size=TILE_SIZE):
    """Creates and saves a matrix file, centralizing data creation."""
    # (Implementation from previous step is unchanged)
    print(f"Creating data file at '{filepath}' with shape {shape}...")
    matrix = PaperMatrix(filepath, shape, mode='w+')
    for r_start in range(0, shape[0], tile_size):
        r_end = min(r_start + tile_size, shape[0])
        for c_start in range(0, shape[1], tile_size):
            c_end = min(c_start + tile_size, shape[1])
            tile_shape = (r_end - r_start, c_end - c_start)
            if fill_value is not None:
                tile_data = np.full(tile_shape, fill_value, dtype=matrix.dtype)
            else:
                tile_data = np.random.rand(*tile_shape).astype(matrix.dtype)
            matrix.data[r_start:r_end, c_start:c_end] = tile_data
    matrix.data.flush()
    matrix.close()
    print(f"Creation of '{filepath}' complete.")


class ResourceMonitor(threading.Thread):
    """A thread that monitors CPU and memory usage of a process."""
    def __init__(self, process, interval=0.1):
        super().__init__()
        self.process = process
        self.interval = interval
        self.running = True
        self.peak_memory_mb = 0
        self.cpu_percents = []

    def run(self):
        while self.running:
            try:
                # Record memory usage (Resident Set Size)
                mem_info = self.process.memory_info()
                memory_mb = mem_info.rss / (1024 * 1024)
                if memory_mb > self.peak_memory_mb:
                    self.peak_memory_mb = memory_mb

                # Record CPU usage
                self.cpu_percents.append(self.process.cpu_percent(interval=self.interval))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            # No sleep needed as cpu_percent has an interval

    def stop(self):
        self.running = False
        self.join()
        avg_cpu = sum(self.cpu_percents) / len(self.cpu_percents) if self.cpu_percents else 0
        return self.peak_memory_mb, avg_cpu

class Benchmark:
    """A context manager to handle timing and resource monitoring for a benchmark run."""
    def __init__(self, description):
        self.description = description
        self.monitor = None
        self.start_time = 0
        self.elapsed = 0
        self.peak_mem = 0
        self.avg_cpu = 0

    def __enter__(self):
        print(f"\n--- Starting: {self.description} ---")
        process = psutil.Process(os.getpid())
        self.monitor = ResourceMonitor(process)
        self.monitor.start()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = time.perf_counter() - self.start_time
        self.peak_mem, self.avg_cpu = self.monitor.stop()
        print(f"--- Finished: {self.description} in {self.elapsed:.4f} seconds ---")

