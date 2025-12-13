import unittest
from core.decorators import encompass_agent
from core.signals import branchpoint
from runtime.engine import ExecutionEngine

execution_count = 0

@encompass_agent
def counting_agent():
    global execution_count
    execution_count += 1
    branchpoint("start")
    return "done"

class TestCaching(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        global execution_count
        execution_count = 0
        self.engine = ExecutionEngine()

    async def test_caching(self):
        root = self.engine.create_root()
        
        # First execution: Should run
        child1, _ = await self.engine.step(counting_agent, root, "A")
        self.assertEqual(execution_count, 1)
        
        # Second execution (Same input): Should hit cache, count should NOT increase
        child2, _ = await self.engine.step(counting_agent, root, "A")
        self.assertEqual(execution_count, 1)
        
        # Third execution (Different input): Should run
        child3, _ = await self.engine.step(counting_agent, root, "B")
        self.assertEqual(execution_count, 2)

if __name__ == '__main__':
    unittest.main()
