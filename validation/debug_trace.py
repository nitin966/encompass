"""
Debug test: Trace exactly what happens during deep search.

Adds detailed logging to understand why depth >10 fails.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.compiler import compile_agent as compile
from core.signals import branchpoint, record_score
from runtime.engine import ExecutionEngine
from search.strategies import BeamSearch
from storage.filesystem import FileSystemStore


# Simple test agents at different depths
@compile
def agent_d5():
    total = 0
    for i in range(5):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total


@compile
def agent_d15():
    total = 0
    for i in range(15):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total


async def debug_test(agent, expected_depth):
    """Run search with detailed logging."""
    print(f"\n{'='*70}")
    print(f"TESTING DEPTH {expected_depth}")
    print(f"{'='*70}\n")
    
    call_count = 0
    
    async def logging_sampler(node, metadata=None):
        nonlocal call_count
        call_count += 1
        depth = node.depth
        print(f"Sampler called #{call_count}: depth={depth}, terminal={node.is_terminal}")
        return [1, 2]  # Two options
    
    engine = ExecutionEngine()
    store = FileSystemStore(f"./data/debug_d{expected_depth}")
    strategy = BeamSearch(store=store, engine=engine, sampler=logging_sampler, width=3)
    
    print(f"Running beam search (width=3)...")
    print(f"Expected: {expected_depth} depths, ~{expected_depth*3} sampler calls\n")
    
    nodes = await strategy.search(agent)
    
    terminal = [n for n in nodes if n.is_terminal]
    max_depth = max(n.depth for n in nodes) if nodes else 0
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Terminal nodes: {len(terminal)}")
    print(f"Max depth reached: {max_depth}")
    print(f"Sampler calls: {call_count}")
    print(f"Expected depth: {expected_depth}")
    
    if max_depth < expected_depth:
        print(f"\n⚠️ WARNING: Did not reach expected depth!")
        print(f"   Gap: {expected_depth - max_depth} levels missing")
        
        # Analyze terminal nodes
        if not terminal:
            print(f"\n❌ NO TERMINAL NODES - Search never completed!")
            print(f"   This suggests agents aren't finishing.")
        else:
            print(f"\n✓ Has terminal nodes, but stopped early")
    
    return max_depth == expected_depth


async def main():
    print("="*70)
    print("DEBUG INVESTIGATION: Why does depth >10 fail?")
    print("="*70)
    
    # Test depth 5 (should work)
    success_5 = await debug_test(agent_d5, 5)
    
    # Test depth 15 (should fail)
    success_15 = await debug_test(agent_d15, 15)
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")
    print(f"Depth 5:  {'✓ PASS' if success_5 else '✗ FAIL'}")
    print(f"Depth 15: {'✓ PASS' if success_15 else '✗ FAIL'}")
    
    if not success_15:
        print(f"\nDeep search confirmed broken.")
        print(f"Investigation needed in:")
        print(f"  1. Agent state machine generation")
        print(f"  2. Beam search pruning logic")
        print(f"  3. Terminal node detection")


if __name__ == "__main__":
    asyncio.run(main())
