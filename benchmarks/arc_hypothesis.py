import asyncio
import time
import random
from encompass import compile, branchpoint, record_score
from encompass.std import action, early_stop
from runtime.engine import ExecutionEngine
from storage.filesystem import FileSystemStore
from search.strategies import BeamSearch, BestOfNSearch, BestFirstSearch

# Mock ARC Task: Input [1, 2] -> Output [2, 4] (Multiply by 2)
TASK_INPUT = [1, 2, 3]
TASK_OUTPUT = [2, 4, 6]

@action
def validate_hypothesis(func_code):
    """Simulates running the generated code against the task."""
    try:
        # Unsafe exec for demo purposes (in sandbox ideally)
        local_scope = {}
        exec(func_code, {}, local_scope)
        transform = local_scope.get('transform')
        if not transform: return 0.0
        
        score = 0
        for i, o in zip(TASK_INPUT, TASK_OUTPUT):
            if transform(i) == o:
                score += 1
        return score / len(TASK_INPUT)
    except Exception:
        return 0.0

@compile
def arc_agent():
    # 1. Hypothesize Rule
    rule = yield branchpoint("hypothesis", metadata={"options": ["add_1", "mul_2", "square"]})
    
    # 2. Generate Code based on rule
    if rule == "add_1":
        code = "def transform(x): return x + 1"
    elif rule == "mul_2":
        code = "def transform(x): return x * 2"
    elif rule == "square":
        code = "def transform(x): return x * x"
    else:
        code = "def transform(x): return x"
        
    # 3. Validate
    accuracy = yield validate_hypothesis(code)
    yield record_score(accuracy * 100)
    
    if accuracy == 1.0:
        yield early_stop()
        return "Solved"
        
    return "Failed"

async def arc_sampler(node, metadata=None):
    if "options" in metadata:
        return metadata["options"]
    return []

async def run_benchmark():
    store = FileSystemStore("arc_trace")
    engine = ExecutionEngine()
    
    print(f"--- ARC Benchmark: Hypothesis Search ---")
    
    # 1. Best-of-N (Random Baseline)
    start = time.time()
    bon = BestOfNSearch(store, engine, arc_sampler, n=10)
    results = await bon.search(arc_agent)
    duration = time.time() - start
    best = max(results, key=lambda n: n.score)
    print(f"Best-of-N:   Time={duration:.4f}s, Best Score={best.score}, Solved={'Solved' in str(best.metadata)}")
    
    # 2. Beam Search (Guided)
    start = time.time()
    beam = BeamSearch(store, engine, arc_sampler, width=2)
    results = await beam.search(arc_agent)
    duration = time.time() - start
    best = max(results, key=lambda n: n.score)
    print(f"Beam Search: Time={duration:.4f}s, Best Score={best.score}, Solved={'Solved' in str(best.metadata)}")

    # 3. Best First Search
    start = time.time()
    befs = BestFirstSearch(store, engine, arc_sampler)
    results = await befs.search(arc_agent)
    duration = time.time() - start
    best = max(results, key=lambda n: n.score)
    print(f"Best-First:  Time={duration:.4f}s, Best Score={best.score}, Solved={'Solved' in str(best.metadata)}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
