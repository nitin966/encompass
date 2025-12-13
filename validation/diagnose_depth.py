"""
Diagnostic: Find exact failure point for deep loops.

Tests agent at increasing depths to identify where it breaks.
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


# Test at multiple depths
for test_depth in [5, 10, 15, 20, 25, 30]:
    print(f"\n{'='*60}")
    print(f"Testing depth={test_depth}")
    print(f"{'='*60}")
    
    # Create agent with exact depth
    code = f"""
@compile
def test_agent():
    total = 0
    for i in range({test_depth}):
        choice = branchpoint(f"step_{{i}}")
        total += choice
    record_score(total)
    return total
"""
    
    exec(code, globals())
    
    async def test():
        async def sampler(node, metadata=None):
            return [1, 2]
        
        engine = ExecutionEngine()
        store = FileSystemStore(f"./data/diag_d{test_depth}")
        strategy = BeamSearch(store=store, engine=engine, sampler=sampler, width=2)
        
        try:
            nodes = await strategy.search(test_agent)
            terminal = [n for n in nodes if n.is_terminal]
            max_depth = max(n.depth for n in nodes) if nodes else 0
            
            print(f"✓ Success")
            print(f"  Nodes: {len(nodes)}")
            print(f"  Terminal: {len(terminal)}")
            print(f"  Max depth: {max_depth}")
            
            if max_depth < test_depth:
                print(f"  ⚠️ WARNING: Didn't reach target depth!")
                print(f"     Expected: {test_depth}, Got: {max_depth}")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test())
