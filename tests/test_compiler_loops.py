import unittest
from core.compiler import compile_agent
from runtime.engine import ExecutionEngine
from encompass import branchpoint

class TestCompilerLoops(unittest.IsolatedAsyncioTestCase):
    async def test_for_loop(self):
        def agent():
            total = 0
            for i in [1, 2, 3]:
                x = branchpoint("choice", options=[i])
                total = total + x
            return total
            
        AgentClass = compile_agent(agent)
        engine = ExecutionEngine()
        
        root = engine.create_root()
        # Loop 1 (i=1)
        node1, sig1 = await engine.step(AgentClass, root)
        self.assertEqual(sig1.metadata['options'], [1])
        
        # Loop 2 (i=2)
        node2, sig2 = await engine.step(AgentClass, node1, 1)
        self.assertEqual(sig2.metadata['options'], [2])
        
        # Loop 3 (i=3)
        node3, sig3 = await engine.step(AgentClass, node2, 2)
        self.assertEqual(sig3.metadata['options'], [3])
        
        # Finish
        node4, sig4 = await engine.step(AgentClass, node3, 3)
        self.assertTrue(node4.is_terminal)
        self.assertEqual(node4.metadata['result'], 6) # 1+2+3

    async def test_while_loop(self):
        def agent():
            count = 0
            while count < 3:
                x = branchpoint("choice", options=[1])
                count = count + 1
            return count
            
        AgentClass = compile_agent(agent)
        engine = ExecutionEngine()
        
        root = engine.create_root()
        
        # Get first branchpoint (count=0)
        node, sig = await engine.step(AgentClass, root)
        self.assertFalse(node.is_terminal)
        
        # Loop iterations
        for i in range(3):
            node, sig = await engine.step(AgentClass, node, 1)
            if i < 2:  # First 2 iterations should not be terminal
                self.assertFalse(node.is_terminal)
            else:  # Third iteration consumes input and exits loop
                self.assertTrue(node.is_terminal)
                self.assertEqual(node.metadata['result'], 3)

if __name__ == "__main__":
    unittest.main()
