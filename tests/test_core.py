import unittest
import shutil
import os
from core.decorators import encompass_agent
from core.signals import branchpoint, record_score
from runtime.engine import ExecutionEngine
from search.strategies import MCTS, BeamSearch
from storage.filesystem import FileSystemStore

@encompass_agent
def simple_agent():
    # A simple agent for testing
    # Depth 0: 0 or 1
    choice = branchpoint("choice")
    if choice == 1:
        record_score(10.0)
        return "Win"
    else:
        record_score(0.0)
        return "Lose"

async def simple_sampler(node):
    return [0, 1]

class TestEncompassCore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        if os.path.exists("encompass_trace_test"):
            shutil.rmtree("encompass_trace_test")
        self.engine = ExecutionEngine()
        self.store = FileSystemStore(base_path="encompass_trace_test")

    def tearDown(self):
        if os.path.exists("encompass_trace_test"):
            shutil.rmtree("encompass_trace_test")

    async def test_engine_step(self):
        root = self.engine.create_root()
        # Step with input 1 (Win)
        child, signal = await self.engine.step(simple_agent, root, 1)
        self.assertTrue(child.is_terminal)
        self.assertEqual(child.score, 10.0)
        self.assertEqual(child.metadata['result'], "Win")
        
        # Step with input 0 (Lose)
        child_lose, _ = await self.engine.step(simple_agent, root, 0)
        self.assertTrue(child_lose.is_terminal)
        self.assertEqual(child_lose.score, 0.0)

    async def test_beam_search(self):
        # The original simple_sampler was global.
        # The change moves it inside the test and updates its signature.
        async def simple_sampler(node, metadata=None):
            if node.depth == 0:
                return [0, 1]
            return []

        beam = BeamSearch(self.store, self.engine, simple_sampler, width=2)
        results = await beam.search(simple_agent)
        
        # Check that we found the winning node
        winning_nodes = [n for n in results if n.is_terminal and n.score == 10.0]
        self.assertTrue(len(winning_nodes) > 0)

    async def test_mcts_search(self):
        # MCTS also needs a sampler, but the diff didn't specify changing this one.
        # For consistency, I'll define a local one here too, matching the new signature.
        async def simple_sampler(node, metadata=None):
            return [0, 1]

        mcts = MCTS(self.store, self.engine, simple_sampler, iterations=50)
        results = await mcts.search(simple_agent)
        
        # Should find the winning node
        winning_nodes = [n for n in results if n.is_terminal and n.score == 10.0]
        self.assertTrue(len(winning_nodes) > 0)

    async def test_deep_branching(self):
        """Test that the engine and search can handle non-trivial depth."""
        
        @encompass_agent
        def deep_agent():
            # Go deep: 20 steps
            # Always pick 1 to survive
            for i in range(20):
                choice = branchpoint(f"step_{i}")
                if choice != 1:
                    record_score(-1.0)
                    return "Dead"
                record_score(1.0)
            return "Alive"

        async def deep_sampler(node, metadata=None):
            return [0, 1]

        # Use Beam Search to find the single valid path
        beam = BeamSearch(self.store, self.engine, deep_sampler, width=2, max_depth=25)
        results = await beam.search(deep_agent)
        
        survivors = [n for n in results if n.is_terminal and n.metadata.get('result') == "Alive"]
        self.assertEqual(len(survivors), 1)
        self.assertEqual(survivors[0].score, 20.0)

    async def test_mcts_noisy_preference(self):
        """
        Test that MCTS prefers a higher-reward branch in a noisy environment.
        
        Scenario:
        - Branch A: 50% chance of 100, 50% chance of 0 (Avg 50)
        - Branch B: Always 10 (Avg 10)
        
        MCTS should eventually prefer Branch A despite the risk.
        """
        import random
        
        @encompass_agent
        def noisy_agent():
            choice = branchpoint("root")
            
            if choice == "A":
                # Simulate noise via a second hidden branchpoint or just stochastic score?
                # Since our engine is deterministic replay, we must use an input to represent the "noise outcome".
                # Let's say the environment (sampler) determines the outcome.
                outcome = branchpoint("chance_node")
                if outcome == "win":
                    record_score(100.0)
                else:
                    record_score(0.0)
            else:
                # Branch B
                record_score(10.0)
            return "Done"

        async def noisy_sampler(node, metadata=None):
            if node.depth == 0:
                return ["A", "B"]
            if node.depth == 1:
                # If we are in Branch A (action_taken="A"), we have a chance node
                # To simulate 50/50, the sampler returns both, and MCTS rollouts will pick randomly.
                # NOTE: In our deterministic engine, "A" leads to a specific node.
                # The "chance" happens at the NEXT step.
                return ["win", "lose"]
            return []

        # We need enough iterations for MCTS to realize A is better on average
        # Increased iterations and exploration to ensure convergence despite variance
        mcts = MCTS(self.store, self.engine, noisy_sampler, iterations=1000, exploration_weight=20.0)
        results = await mcts.search(noisy_agent)
        
        # Analyze the root's children visits
        # Root is the first node. Its children are the nodes for "A" and "B".
        # We can inspect the MCTS internal state (visits)
        
        # Find root
        root = [n for n in results if n.depth == 0][0]
        
        # Find children
        children = mcts.children.get(root.node_id, [])
        node_a = next((c for c in children if c.action_taken == "A"), None)
        node_b = next((c for c in children if c.action_taken == "B"), None)
        
        if node_a and node_b:
            visits_a = mcts.visits.get(node_a.node_id, 0)
            visits_b = mcts.visits.get(node_b.node_id, 0)
            
            # A (Avg 50) should be visited more than B (Avg 10)
            # print(f"Visits A: {visits_a}, Visits B: {visits_b}")
            self.assertGreater(visits_a, visits_b)

if __name__ == '__main__':
    unittest.main()
