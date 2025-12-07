import asyncio
import time
import re
from encompass import compile, branchpoint, record_score
from encompass.std import action, early_stop
from runtime.engine import ExecutionEngine
from storage.filesystem import FileSystemStore
from search.strategies import BeamSearch, BestOfNSearch

# Mock GSM8K Problem
PROBLEM = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
ANSWER = 72

@action
def calculator(expression):
    """Safe calculator tool."""
    try:
        # Very simple eval
        return eval(expression, {"__builtins__": {}}, {})
    except:
        return None

@compile
def math_solver():
    # Step 1: Plan
    plan = yield branchpoint("plan", metadata={"options": [
        "Calculate May sales first",
        "Calculate total directly",
        "Guess 100"
    ]})
    
    total = 0
    
    if plan == "Calculate May sales first":
        # Step 2: Calc May
        may_expr = "48 / 2"
        may = yield calculator(may_expr)
        if may is None: return "Error"
        
        # Step 3: Calc Total
        total_expr = f"48 + {may}"
        total = yield calculator(total_expr)
        
    elif plan == "Calculate total directly":
        total_expr = "48 + (48 / 2)"
        total = yield calculator(total_expr)
        
    else:
        total = 100
        
    # Check answer (Oracle for benchmark)
    if total == ANSWER:
        yield record_score(100)
        yield early_stop()
        return f"Solved: {total}"
    else:
        yield record_score(0)
        return f"Wrong: {total}"

async def solver_sampler(node, metadata=None):
    if "options" in metadata:
        return metadata["options"]
    return []

async def run_benchmark():
    store = FileSystemStore("gsm8k_trace")
    engine = ExecutionEngine()
    
    print(f"--- GSM8K Benchmark: {PROBLEM[:50]}... ---")
    
    # 1. Beam Search (Diversity=0.0)
    start = time.time()
    beam = BeamSearch(store, engine, solver_sampler, width=2, diversity_penalty=0.0)
    results = await beam.search(math_solver)
    best = max(results, key=lambda n: n.score)
    print(f"Beam (No Div): Score={best.score}, Solved={'Solved' in str(best.metadata)}")
    
    # 2. Beam Search (Diversity=10.0)
    # Should explore different plans more aggressively?
    start = time.time()
    beam_div = BeamSearch(store, engine, solver_sampler, width=2, diversity_penalty=10.0)
    results = await beam_div.search(math_solver)
    best = max(results, key=lambda n: n.score)
    print(f"Beam (Div):    Score={best.score}, Solved={'Solved' in str(best.metadata)}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
