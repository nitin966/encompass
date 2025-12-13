"""
Simple Deep Search Test

Proves framework handles deep search (depth 30-50).

Run: python validation/simple_deep_test.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.compiler import compile_agent as compile
from core.signals import branchpoint, record_score
from runtime.engine import ExecutionEngine
from search.strategies import BeamSearch
from storage.filesystem import FileSystemStore


def create_counter_agent(target_depth):
    """Create agent with specified depth using closure."""

    @compile
    def counter_agent(depth):
        """Count up through target_depth decisions."""
        total = 0
        for i in range(depth):
            choice = branchpoint(f"step_{i}")
            total += choice

        record_score(total)
        return total

    return lambda: counter_agent(target_depth)


async def test_deep_search(depth, width):
    """Test search at specified depth and width."""

    print(f"\n{'=' * 60}")
    print(f"Testing Depth={depth}, Width={width}")
    print(f"{'=' * 60}\n")

    # Create agent
    agent = create_counter_agent(depth)
    print(f"✓ Created agent with {depth} branchpoints")

    # Sampler
    async def sampler(node, metadata=None):
        return [1, 2, 3]  # Three options at each branch

    # Setup
    engine = ExecutionEngine()
    store = FileSystemStore(f"./data/deep_test_d{depth}_w{width}")
    strategy = BeamSearch(store=store, engine=engine, sampler=sampler, width=width)

    print(f"✓ Created beam search (width={width})")
    print(f"  Expected nodes: ~{depth * width}")
    print("\nRunning search...")

    # Run
    start = time.time()
    nodes = await strategy.search(agent)
    elapsed = time.time() - start

    # Analyze
    terminal = [n for n in nodes if n.is_terminal]
    terminal.sort(key=lambda n: n.score, reverse=True)
    max_depth = max(n.depth for n in nodes)

    # Report
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print("✓ Search completed successfully")
    print(f"  Nodes created: {len(nodes)}")
    print(f"  Terminal nodes: {len(terminal)}")
    print(f"  Max depth: {max_depth}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {len(nodes) / elapsed:.0f} nodes/sec")

    if terminal:
        print(f"\n  Best score: {terminal[0].score}")
        print(f"  Top 3 scores: {[n.score for n in terminal[:3]]}")

    # Validation
    success = (
        len(nodes) > 0
        and max_depth == depth
        and len(terminal) > 0
        and elapsed < depth * 2  # Should be fast
    )

    return success, len(nodes), max_depth, elapsed


async def main():
    print("=" * 60)
    print("SIMPLE DEEP SEARCH TEST")
    print("=" * 60)
    print("\nValidates framework handles deep search efficiently.\n")

    tests = [
        (10, 3, "Warm-up"),
        (20, 5, "Moderate"),
        (30, 5, "Deep"),
        (50, 3, "Very Deep"),
    ]

    results = []

    for depth, width, label in tests:
        try:
            success, nodes, max_depth, time_taken = await test_deep_search(depth, width)
            results.append(
                {
                    "depth": depth,
                    "width": width,
                    "success": success,
                    "nodes": nodes,
                    "time": time_taken,
                    "label": label,
                }
            )
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            results.append(
                {"depth": depth, "width": width, "success": False, "error": str(e), "label": label}
            )

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}\n")

    print(f"{'Label':<15} {'Depth':<8} {'Width':<8} {'Nodes':<8} {'Time':<10} {'Status'}")
    print(f"{'-' * 60}")

    for r in results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        nodes_str = f"{r.get('nodes', 0):,}" if "nodes" in r else "N/A"
        time_str = f"{r.get('time', 0):.2f}s" if "time" in r else "N/A"
        print(
            f"{r['label']:<15} {r['depth']:<8} {r['width']:<8} {nodes_str:<8} {time_str:<10} {status}"
        )

    all_passed = all(r["success"] for r in results)

    print(f"\n{'=' * 60}")
    if all_passed:
        max_depth_tested = max(r["depth"] for r in results)
        print("✓ ALL TESTS PASSED")
        print(f"\nFramework validated at depth={max_depth_tested}!")
        print("Ready for production-scale search.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nCheck errors above for details.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
