"""
Test cost tracking functionality.
"""

import unittest
import asyncio
from encompass import compile, branchpoint
from core.signals import record_costs
from runtime.engine import ExecutionEngine


class TestCostTracking(unittest.IsolatedAsyncioTestCase):
    
    async def test_cost_tracking_cps(self):
        """Test that RecordCosts signals are tracked in CPS agents."""
        
        @compile
        def agent():
            # Yield a cost record
            record_costs(tokens=100, dollars=0.01)
            
            # Yield a branchpoint
            x = branchpoint("choice")
            
            # Another cost
            record_costs(tokens=50, dollars=0.005)
            
            return x
        
        engine = ExecutionEngine()
        root = engine.create_root()
        
        # First step - should record first cost
        node1, sig1 = await engine.step(agent, root)
        
        # Check cost was recorded
        self.assertEqual(len(engine.cost_aggregator.node_costs), 1)
        total_cost = engine.cost_aggregator.get_total_cost()
        self.assertGreater(total_cost, 0)
        
        # Get summary
        summary = engine.cost_aggregator.get_summary()
        self.assertEqual(summary['node_count'], 1)
        self.assertGreater(summary['total_cost_usd'], 0)
        
    def test_cost_summary(self):
        """Test cost summary reporting."""
        from runtime.costs import CostAggregator
        
        agg = CostAggregator()
        
        # Record some costs
        agg.record("node1", tokens_in=100, tokens_out=50, cost_usd=0.01, model="gpt-4")
        agg.record("node2", tokens_in=200, tokens_out=100, cost_usd=0.02, model="gpt-4")
        agg.record("node3", tokens_in=50, tokens_out=25, cost_usd=0.005, model="gpt-3.5")
        
        # Check totals
        self.assertAlmostEqual(agg.get_total_cost(), 0.035)
        
        tokens = agg.get_total_tokens()
        self.assertEqual(tokens['tokens_in'], 350)
        self.assertEqual(tokens['tokens_out'], 175)
        self.assertEqual(tokens['tokens_total'], 525)
        
        # Check summary
        summary = agg.get_summary()
        self.assertEqual(summary['node_count'], 3)
        self.assertEqual(len(summary['models']), 2)
        self.assertEqual(summary['models']['gpt-4']['count'], 2)
        self.assertEqual(summary['models']['gpt-3.5']['count'], 1)


if __name__ == "__main__":
    # Run async tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCostTracking)
    for test in suite:
        if asyncio.iscoroutinefunction(test._testMethodName):
            asyncio.run(test.debug())
        else:
            test.debug()
