import unittest
from runtime.engine import ExecutionEngine
from core.compiler import compile_agent
from encompass import branchpoint
from encompass.std import local_search
from search.strategies import BeamSearch

# Define agents at module level to ensure visibility for compiled code
def sub_agent():
    x = branchpoint("sub_choice")
    return x * 2

SubAgentClass = compile_agent(sub_agent)

async def dummy_sampler(node, metadata=None):
    return [5, 10]

def main_agent_compiled():
     # Use SubAgentClass which is now global
     result = local_search(BeamSearch, SubAgentClass, sampler=dummy_sampler, width=2)
     return result

AgentClass = compile_agent(main_agent_compiled)

class TestLocalSearch(unittest.IsolatedAsyncioTestCase):
    async def test_local_search(self):
        engine = ExecutionEngine()
        root = engine.create_root()
        
        # Run
        node, sig = await engine.step(AgentClass, root)
        
        self.assertTrue(node.is_terminal)
        self.assertIn(node.metadata['result'], [10, 20])

if __name__ == "__main__":
    unittest.main()
