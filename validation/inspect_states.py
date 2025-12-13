"""
Inspect compiled state machine to see what states are generated.
"""

import sys
from pathlib import Path
import ast
import inspect

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.compiler import compile_agent as compile
from core.signals import branchpoint, record_score


# Create agents at different depths
@compile
def agent_d5():
    total = 0
    for i in range(5):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total


@compile
def agent_d15():
    total = 0
    for i in range(15):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total


def inspect_agent(agent_class, name):
    """Inspect the generated state machine."""
    print(f"\n{'='*70}")
    print(f"INSPECTING: {name}")
    print(f"{'='*70}\n")
    
    # Create instance
    machine = agent_class()
    
    # Check if it has states attribute
    if hasattr(machine, 'states') or hasattr(agent_class, 'states'):
        print("❌ No 'states' attribute found")
        print("   State machine structure unclear")
    
    # Try to get the run method source
    try:
        source = inspect.getsource(agent_class.run)
        lines = source.split('\n')
        
        # Count state branches
        state_checks = [l for l in lines if 'self._state ==' in l]
        print(f"State branches in run(): {len(state_checks)}")
        
        if len(state_checks) < 20:
            print(f"\nState checks:")
            for i, line in enumerate(state_checks[:15]):
                print(f"  {line.strip()}")
            if len(state_checks) > 15:
                print(f"  ... and {len(state_checks) - 15} more")
        else:
            print(f"  (Too many to display)")
            
        # Look for max state number
        max_state = 0
        for line in lines:
            if 'self._state ==' in line:
                try:
                    num = int(line.split('==')[1].split(':')[0].strip())
                    max_state = max(max_state, num)
                except:
                    pass
        
        print(f"\nMax state number: {max_state}")
        
    except Exception as e:
        print(f"Could not inspect source: {e}")
    
    # Try running it manually to see states
    print(f"\nManual execution test:")
    machine = agent_class()
    for step in range(min(20, 100)):  # Try up to 20 steps
        try:
            sig = machine.run(1)  # Always choose 1
            if sig is None:
                print(f"  Step {step}: Returned None (done)")
                break
            print(f"  Step {step}: {type(sig).__name__}, state={machine._state}")
        except Exception as e:
            print(f"  Step {step}: ERROR - {e}")
            break


# Inspect both
inspect_agent(agent_d5, "agent_d5 (should work)")
inspect_agent(agent_d15, "agent_d15 (fails at depth 10)")

print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")
print("If agent_d15 stops at step ~10-12, the state machine")
print("generation is limited, not the search strategy.")
