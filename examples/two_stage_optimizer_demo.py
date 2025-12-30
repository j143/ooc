"""
Example demonstrating the new two-stage optimizer and observability features.

This example shows:
1. Three-stage optimizer pipeline (analyze → rewrite → execute)
2. Structured logging
3. Performance profiling
4. Cost estimation
5. Execution tracing
"""

import numpy as np
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.core import PaperMatrix
from paper.plan import Plan, EagerNode
from paper.optimizer import analyze, rewrite, execute_plan, estimate_cost
from paper.buffer import BufferManager
from paper.observability import (
    configure_logging, get_profiler, TraceLogger
)


def main():
    """Run the example with observability features."""
    
    # ========================================================================
    # Setup: Configure logging
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE: Two-Stage Optimizer with Observability")
    print("="*80 + "\n")
    
    # Configure structured logging
    logger = configure_logging(level="INFO", log_file="paper_example.log")
    logger.info("Starting two-stage optimizer example")
    
    # Get global profiler
    profiler = get_profiler()
    
    # Create trace logger
    trace = TraceLogger()
    
    # ========================================================================
    # Step 1: Create test data
    # ========================================================================
    print("Step 1: Creating test matrices...")
    trace.begin("Data Creation")
    
    with profiler.profile("create_test_data"):
        test_dir = tempfile.mkdtemp()
        shape = (2048, 2048)
        dtype = np.float32
        
        # Create matrices A and B
        A_path = os.path.join(test_dir, "A.bin")
        B_path = os.path.join(test_dir, "B.bin")
        
        np.random.seed(42)
        data_A = np.random.rand(*shape).astype(dtype)
        data_B = np.random.rand(*shape).astype(dtype)
        
        data_A.tofile(A_path)
        data_B.tofile(B_path)
        
        A = PaperMatrix(A_path, shape, dtype=dtype, mode='r')
        B = PaperMatrix(B_path, shape, dtype=dtype, mode='r')
        
        trace.log(f"Created matrices: {shape}")
    
    trace.end()
    
    # ========================================================================
    # Step 2: Build computation plan
    # ========================================================================
    print("\nStep 2: Building computation plan...")
    trace.begin("Plan Construction")
    
    with profiler.profile("build_plan"):
        plan_A = Plan(EagerNode(A))
        plan_B = Plan(EagerNode(B))
        
        # Build a complex plan: (A + B) * 2.5
        plan = (plan_A + plan_B) * 2.5
        
        trace.log(f"Plan: (A + B) * 2.5")
    
    trace.end()
    
    # ========================================================================
    # Step 3: STAGE 1 - Analyze (no execution)
    # ========================================================================
    print("\nStep 3: STAGE 1 - Analyzing plan (no execution)...")
    trace.begin("Stage 1: Analyze")
    
    with profiler.profile("analyze_plan"):
        io_trace, match_result = analyze(plan)
        
        print(f"\n  ✓ Pattern detected: {match_result.pattern.value}")
        print(f"  ✓ Fusion available: {match_result.is_fusable}")
        print(f"  ✓ I/O trace length: {len(io_trace)} tile accesses")
        print(f"  ✓ Input shapes: {match_result.input_shapes}")
        print(f"  ✓ Parameters: {match_result.parameters}")
        
        trace.log(f"Pattern: {match_result.pattern.value}")
        trace.log(f"I/O trace: {len(io_trace)} accesses")
    
    trace.end()
    
    # ========================================================================
    # Step 4: Cost Estimation
    # ========================================================================
    print("\nStep 4: Estimating execution cost...")
    trace.begin("Cost Estimation")
    
    with profiler.profile("estimate_cost"):
        cost = estimate_cost(plan, match_result)
        
        print(f"\n  ✓ I/O operations: {cost.io_operations}")
        print(f"  ✓ Compute operations: {cost.compute_operations}")
        print(f"  ✓ Estimated I/O bytes: {cost.estimated_io_bytes:,}")
        print(f"  ✓ Cache benefit: {cost.cache_benefit:.1%}")
        print(f"  ✓ Total cost: {cost.total_cost:.0f}")
        
        trace.log(f"Total cost: {cost.total_cost:.0f}")
    
    trace.end()
    
    # ========================================================================
    # Step 5: STAGE 2 - Rewrite plan
    # ========================================================================
    print("\nStep 5: STAGE 2 - Rewriting plan...")
    trace.begin("Stage 2: Rewrite")
    
    with profiler.profile("rewrite_plan"):
        rewritten_plan = rewrite(plan, match_result)
        trace.log("Plan rewritten for fusion")
    
    trace.end()
    
    # ========================================================================
    # Step 6: STAGE 3 - Execute
    # ========================================================================
    print("\nStep 6: STAGE 3 - Executing plan...")
    trace.begin("Stage 3: Execute")
    
    output_path = os.path.join(test_dir, "result.bin")
    
    with profiler.profile("execute_plan"):
        buffer_manager = BufferManager(max_cache_size_tiles=32, io_trace=io_trace)
        result = execute_plan(rewritten_plan, match_result, output_path, buffer_manager)
        
        trace.log("Execution complete")
    
    trace.end()
    
    # ========================================================================
    # Step 7: Verify results
    # ========================================================================
    print("\nStep 7: Verifying results...")
    trace.begin("Verification")
    
    with profiler.profile("verify_result"):
        result_data = np.fromfile(output_path, dtype=dtype).reshape(shape)
        expected = (data_A + data_B) * 2.5
        
        max_diff = np.max(np.abs(result_data - expected))
        print(f"\n  ✓ Max difference from expected: {max_diff:.2e}")
        
        trace.log(f"Verification: max_diff={max_diff:.2e}")
    
    trace.end()
    
    # ========================================================================
    # Step 8: Print profiling results
    # ========================================================================
    print("\n" + "="*80)
    print("PROFILING RESULTS")
    print("="*80)
    
    profiler.print_summary()
    
    # Save profiling data
    profiler.save_json("profile_results.json")
    profiler.save_flame_graph("flame_graph.json")
    print("✓ Profiling data saved to profile_results.json and flame_graph.json")
    
    # ========================================================================
    # Step 9: Print execution trace
    # ========================================================================
    trace.print()
    
    # Save trace
    trace.save("execution_trace.json")
    print("✓ Execution trace saved to execution_trace.json")
    
    # ========================================================================
    # Step 10: Show cache statistics
    # ========================================================================
    print("\n" + "="*80)
    print("CACHE STATISTICS")
    print("="*80)
    
    cache_log = buffer_manager.get_log()
    
    hits = sum(1 for event in cache_log if event[1] == 'HIT')
    misses = sum(1 for event in cache_log if event[1] == 'MISS')
    evictions = sum(1 for event in cache_log if event[1] == 'EVICT')
    total = hits + misses
    
    print(f"Cache hits:      {hits:6d} ({100*hits/total if total > 0 else 0:.1f}%)")
    print(f"Cache misses:    {misses:6d} ({100*misses/total if total > 0 else 0:.1f}%)")
    print(f"Evictions:       {evictions:6d}")
    print(f"Total accesses:  {total:6d}")
    print("="*80 + "\n")
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    A.close()
    B.close()
    result.close()
    
    import shutil
    shutil.rmtree(test_dir)
    
    print("✓ Example complete!")
    print(f"✓ Log file: paper_example.log")
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
