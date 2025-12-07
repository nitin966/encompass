import unittest
import shutil
import os
from examples.translation_agent import create_translation_agent
from core.llm import MockLLM
from runtime.engine import ExecutionEngine
from search.strategies import MCTS
from storage.filesystem import FileSystemStore

# Sampler from run_translation.py
async def translation_sampler(node):
    if node.depth == 0: return [0, 1, 2]
    elif node.depth == 1: return [0, 1]
    return []

class TestTranslationE2E(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        if os.path.exists("encompass_trace_e2e"):
            shutil.rmtree("encompass_trace_e2e")
        self.store = FileSystemStore(base_path="encompass_trace_e2e")
        self.engine = ExecutionEngine()
        self.llm = MockLLM()
        self.agent = create_translation_agent(self.llm)

    def tearDown(self):
        if os.path.exists("encompass_trace_e2e"):
            shutil.rmtree("encompass_trace_e2e")

    async def test_translation_flow_mcts(self):
        """
        Runs the full translation agent with MCTS and verifies it finds the optimal solution.
        """
        mcts = MCTS(
            store=self.store,
            engine=self.engine,
            sampler=translation_sampler,
            iterations=30,
            exploration_weight=1.0
        )
        
        results = await mcts.search(self.agent)
        results.sort(key=lambda n: n.score, reverse=True)
        
        self.assertTrue(len(results) > 0)
        best_node = results[0]
        
        # Verify score is high (our mock logic gives > 100 for best path)
        self.assertTrue(best_node.score > 100.0)
        
        # Verify content contains expected C++ code
        code = best_node.metadata.get('result', '')
        self.assertIn("template <typename T>", code)
        self.assertIn("return a + b;", code)

if __name__ == '__main__':
    unittest.main()
