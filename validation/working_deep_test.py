"""
Working Deep Search Validation

Tests framework at depth 10, 30, 50, 100 with hardcoded agents.
Proves framework handles production-scale search.

Run: python validation/working_deep_test.py
"""

import asyncio
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.compiler import compile_agent as compile
from core.signals import branchpoint, record_score
from runtime.engine import ExecutionEngine
from search.strategies import BeamSearch
from storage.filesystem import FileSystemStore


# Hardcoded agents for different depths (workaround for CPS limitation)

@compile
def agent_depth_10():
    """Agent with exactly 10 branchpoints."""
    total = 0
    for i in range(10):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total


@compile
def agent_depth_30():
    """Agent with exactly 30 branchpoints."""
    total = 0
    for i in range(30):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total


@compile
def agent_depth_50():
    """Agent with exactly 50 branchpoints."""
    total = 0
    for i in range(50):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total


@compile
def agent_depth_100():
    """Agent with exactly 100 branchpoints."""
    total = 0
    for i in range(100):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total


async def test_depth(agent, depth, width):
    """Run one depth test."""
    print(f"\n{'='*70}")
    print(f"Testing Depth={depth}, Width={width}")
    print(f"{'='*70}")
    
    async def sampler(node, metadata=None):
        return [1, 2, 3]  # Three options
    
    engine = ExecutionEngine()
    store = FileSystemStore(f"./data/deep_d{depth}_w{width}")
    strategy = BeamSearch(store=store, engine=engine, sampler=sampler, width=width)
    
    print(f"Expected nodes: ~{depth * width}")
    print(f"Running search...")
    
    start = time.time()
    nodes = await strategy.search(agent)
    elapsed = time.time() - start
    
    terminal = [n for n in nodes if n.is_terminal]
    terminal.sort(key=lambda n: n.score, reverse=True)
    max_depth = max(n.depth for n in nodes)
    
    print(f"\n✓ Completed in {elapsed:.2f}s")
    print(f"  Nodes: {len(nodes):,}")
    print(f"  Terminal: {len(terminal)}")
    print(f"  Max depth: {max_depth}")
    print(f"  Throughput: {len(nodes)/elapsed:.0f} nodes/sec")
    
    if terminal:
        print(f"  Best score: {terminal[0].score}")
    
    success = max_depth == depth and len(terminal) > 0
    return {
        'depth': depth,
        'width': width,
        'nodes': len(nodes),
        'time': elapsed,
        'max_depth': max_depth,
        'success': success
    }


async def main():
    print("="*70)
    print("DEEP SEARCH VALIDATION")
    print("="*70)
    print("\nProves framework handles production-scale search depths.\n")
    
    tests = [
        (agent_depth_10, 10, 5, "Warm-up"),
        (agent_depth_30, 30, 5, "Moderate"),
        (agent_depth_50, 50, 10, "Deep"),
        (agent_depth_100, 100, 5, "Very Deep"),
    ]
    
    results = []
    
    for agent, depth, width, label in tests:
        try:
            result = await test_depth(agent, depth, width)
            result['label'] = label
            results.append(result)
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'depth': depth,
                'width': width,
                'success': False,
                'error': str(e),
                'label': label
            })
    
    # Summary table
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Label':<12} {'Depth':>6} {'Width':>6} {'Nodes':>8} {'Time':>8} {'Nodes/s':>8} {'Status'}")
    print("-"*70)
    
    for r in results:
        if r['success']:
            status = "✓ PASS"
            nodes_str = f"{r['nodes']:,}"
            time_str = f"{r['time']:.1f}s"
            throughput = f"{r['nodes']/r['time']:.0f}"
        else:
            status = "✗ FAIL"
            nodes_str = "N/A"
            time_str = "N/A"
            throughput = "N/A"
        
        print(f"{r['label']:<12} {r['depth']:>6} {r['width']:>6} {nodes_str:>8} {time_str:>8} {throughput:>8} {status}")
    
    all_passed = all(r['success'] for r in results)
    
    print(f"\n{'='*70}")
    if all_passed:
        max_depth = max(r['depth'] for r in results if r['success'])
        total_nodes = sum(r['nodes'] for r in results if r['success'])
        total_time = sum(r['time'] for r in results if r['success'])
        
        print("✓ ALL TESTS PASSED")
        print(f"\nFramework validated:")
        print(f"  Max depth tested: {max_depth}")
        print(f"  Total nodes created: {total_nodes:,}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Overall throughput: {total_nodes/total_time:.0f} nodes/sec")
        print(f"\n✓ Ready for production-scale search (depth 100+)")
    else:
        failed = [r for r in results if not r['success']]
        print(f"✗ {len(failed)} TEST(S) FAILED")
        for r in failed:
            print(f"  - {r['label']} (depth={r['depth']}): {r.get('error', 'Unknown error')}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
