import unittest
from core.compiler import compile_agent, AgentMachine
from encompass import branchpoint

class TestCPSCompiler(unittest.TestCase):
    def test_linear_agent(self):
        def my_agent():
            x = branchpoint("step1")
            y = branchpoint("step2")
            return x + y
            
        MachineClass = compile_agent(my_agent)
        machine = MachineClass()
        
        # Step 1
        sig1 = machine.run(None)
        self.assertEqual(sig1.name, "step1")
        self.assertEqual(machine._state, 1)
        
        # Step 2
        sig2 = machine.run(10) # x = 10
        self.assertEqual(sig2.name, "step2")
        self.assertEqual(machine._state, 2)
        self.assertEqual(machine._ctx['x'], 10)
        
        # Step 3
        machine.run(20) # y = 20
        self.assertTrue(machine._done)
        self.assertEqual(machine._result, 30)

if __name__ == "__main__":
    unittest.main()
