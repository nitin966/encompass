import unittest
import asyncio
from runtime.engine import ExecutionEngine
from runtime.node import SearchNode
from encompass.std import action
from encompass import encompass_agent, branchpoint, record_score
from search.strategies import BeamSearch
from storage.filesystem import FileSystemStore

class TestSafety(unittest.IsolatedAsyncioTestCase):
    async def test_side_effect_replay(self):
        """
        Verify that side effects are executed exactly once per unique path,
        even when replayed multiple times.
        """
        # Global counter to track side effects
        execution_count = 0
        
        @action
        def increment_counter(val):
            nonlocal execution_count
            execution_count += val
            return execution_count

        @encompass_agent
        def risky_agent():
            # This side effect should only happen ONCE per path
            curr = yield increment_counter(1)
            
            # Branching
            choice = yield branchpoint("decision")
            
            if choice == "A":
                yield record_score(10)
            else:
                yield record_score(5)
                
            return "Done"

        engine = ExecutionEngine()
        store = FileSystemStore("test_trace")
        
        async def sampler(node, metadata=None):
            return ["A", "B"]

        # Run Search
        # This will explore both branches.
        # The 'increment_counter' happens BEFORE the branch.
        # It should be executed ONCE for the root, and then replayed for the children.
        beam = BeamSearch(store, engine, sampler, width=2)
        results = await beam.search(risky_agent)
        
        # Verification
        # execution_count should be 1.
        # If replay was unsafe, it would be called every time we re-ran the generator for the children.
        # We have 2 children (A and B).
        # 1. Root -> Effect (Executes, count=1) -> BranchPoint.
        # 2. Child A -> Replay Root -> Replay Effect (SKIP) -> Branch A.
        # 3. Child B -> Replay Root -> Replay Effect (SKIP) -> Branch B.
        
        self.assertEqual(execution_count, 1, f"Side effect executed {execution_count} times instead of 1!")
        self.assertEqual(len(results), 3) # Root, A, B (visited)

if __name__ == '__main__':
    unittest.main()
