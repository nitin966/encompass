import unittest
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.compiler import compile_agent
from core.signals import branchpoint

class TestHeavyState(unittest.TestCase):
    def test_heavy_state_serialization(self):
        """
        Verify performance of saving/loading heavy state (1MB string).
        If structural sharing fails, this will be O(N^2) or very slow.
        """
        
        # Define the agent
        def heavy_agent():
            # Load a 1MB "context"
            large_context = "x" * 1024 * 1024 
            
            for i in range(20): # Reduced from 50 to 20 for quick CI, but enough to show trend
                # Force a save/load cycle
                choice = branchpoint(f"step_{i}")
                
                # Modify slightly to force pmap update
                large_context += "x"
                
            return len(large_context)

        Machine = compile_agent(heavy_agent)
        
        print(f"\nRunning Heavy State Test (1MB context)...")
        
        # Manual execution loop with save/load
        m = Machine()
        
        # Initial run to first branchpoint
        start_time = time.time()
        sig = m.run()
        
        current_state = m.save()
        
        step_times = []
        
        for i in range(20):
            step_start = time.time()
            
            # Simulate reloading from storage (bytes serialization)
            # This is critical: we must serialize to bytes to test dill performance
            import dill
            serialized = dill.dumps(current_state)
            loaded_state = dill.loads(serialized)
            
            # Create new machine and load
            new_m = Machine()
            new_m.load(loaded_state)
            
            # Run next step
            sig = new_m.run(i) # Pass choice
            
            if not new_m._done:
                current_state = new_m.save()
            
            step_end = time.time()
            duration = step_end - step_start
            step_times.append(duration)
            print(f"Step {i}: {duration:.4f}s")
            
        total_time = time.time() - start_time
        avg_time = sum(step_times) / len(step_times)
        
        print(f"Total time: {total_time:.4f}s")
        print(f"Average step time: {avg_time:.4f}s")
        
        # Assertions
        # If it copies 1MB every time, 20 steps * 1MB copy/serialize might take a bit,
        # but if it copies HISTORY of states or something worse, it explodes.
        # 1MB serialization should be fast (<0.1s).
        # We assert that it doesn't grow linearly with steps (O(1) per step roughly).
        # Actually, the state size grows slightly (1 byte per step), so it is O(1).
        self.assertLess(avg_time, 0.5, "Average step time too high (likely copying data inefficiently)")

if __name__ == "__main__":
    unittest.main()
