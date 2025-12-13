import unittest
from runtime.engine import ExecutionEngine
from encompass import compile, branchpoint

@compile
def sub_agent():
    choice = branchpoint("sub_choice")
    return choice * 2
    
@compile
def main_agent():
    val = sub_agent()
    return val + 1

class TestNestedSearch(unittest.IsolatedAsyncioTestCase):
    async def test_nested_agent(self):
        engine = ExecutionEngine()
            
        # Run step 1: Start main, enter sub, hit branchpoint
        root = engine.create_root()
        node1, signal1 = await engine.step(main_agent, root)
        
        self.assertEqual(signal1.name, "sub_choice")
        self.assertEqual(node1.depth, 0) # Depth 0 because no choice made yet? Or 0 because root.
        
        # Run step 2: Provide choice to sub
        # Choice = 5. Sub returns 10. Main returns 11.
        node2, signal2 = await engine.step(main_agent, node1, 5)
        
        self.assertTrue(node2.is_terminal)
        self.assertEqual(node2.metadata['result'], 11)

if __name__ == "__main__":
    unittest.main()
