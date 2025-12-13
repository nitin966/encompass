import unittest
from core.compiler import compile_agent
from runtime.engine import ExecutionEngine
from encompass import branchpoint

class TestCompilerControlFlow(unittest.IsolatedAsyncioTestCase):
    async def test_if_statement(self):
        def agent():
            x = branchpoint("choice", options=[1, 2])
            if x == 1:
                y = "one"
            else:
                y = "two"
            return y
            
        AgentClass = compile_agent(agent)
        engine = ExecutionEngine()
        
        # Test Path 1
        root = engine.create_root()
        node, sig = await engine.step(AgentClass, root) # Yields branchpoint
        
        # Resume with 1
        child, sig = await engine.step(AgentClass, node, 1)
        self.assertTrue(child.is_terminal)
        self.assertEqual(child.metadata['result'], "one")
        
        # Test Path 2
        # Resume with 2
        child2, sig = await engine.step(AgentClass, node, 2)
        self.assertTrue(child2.is_terminal)
        self.assertEqual(child2.metadata['result'], "two")

    async def test_argument_capture(self):
        def agent(start_val):
            x = branchpoint("choice", options=[1])
            return start_val + x
            
        AgentClass = compile_agent(agent)
        engine = ExecutionEngine()
        
        # Instantiate with argument
        machine = AgentClass(10)
        
        # Run manually since engine.step usually takes a factory
        # We can pass a lambda factory
        agent_factory = lambda: AgentClass(10)
        
        root = engine.create_root()
        node, sig = await engine.step(agent_factory, root)
        
        # Resume
        child, sig = await engine.step(agent_factory, node, 5)
        self.assertTrue(child.is_terminal)
        self.assertEqual(child.metadata['result'], 15) # 10 + 5

if __name__ == "__main__":
    unittest.main()
