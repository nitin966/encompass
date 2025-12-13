print("DEBUG: Loading test file")
import unittest
from core.compiler import compile_agent
from core.signals import branchpoint, ControlSignal

class TestImplicitYield(unittest.TestCase):
    def test_implicit_branchpoint(self):
        def agent():
            # No yield!
            x = branchpoint("choice")
            return x

        AgentClass = compile_agent(agent)
        machine = AgentClass()
        
        # Step 1: Run to branchpoint
        sig = machine.run()
        self.assertIsInstance(sig, ControlSignal)
        self.assertEqual(sig.name, "choice")
        self.assertFalse(machine._done, "Machine should not be done after implicit yield")
        
        res = machine.run(10)
        self.assertIsNone(res)
        self.assertTrue(machine._done)
        self.assertEqual(machine._result, 10)



if __name__ == "__main__":
    unittest.main()
