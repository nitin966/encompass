import asyncio
import time
import random
from encompass import compile, branchpoint, record_score
from encompass.std import action
from runtime.engine import ExecutionEngine
from storage.filesystem import FileSystemStore
from search.strategies import BeamSearch, MCTS, BestOfNSearch

# Benchmark Task: Find a target number in a grid
TARGET = 42

@compile
def math_agent():
    current = 0
    steps = 0
    path = []
    
    for _ in range(5):
        op = yield branchpoint("op", metadata={"options": ["+1", "+2", "*2"]})
        path.append(op)
        
        if op == "+1":
            current += 1
        elif op == "+2":
            current += 2
        elif op == "*2":
            current *= 2
            
        # Guide search
        dist = abs(TARGET - current)
        yield record_score(-dist) # Higher score = closer to target
        
        if current == TARGET:
            yield record_score(100)
            return path
            
    return path

async def benchmark_sampler(node, metadata=None):
    return ["+1", "+2", "*2"]

async def run_benchmark():
    store = FileSystemStore("benchmark_trace")
    engine = ExecutionEngine()
    
    print(f"--- Benchmark: Target {TARGET} ---")
    
    # 1. Beam Search
    start = time.time()
    beam = BeamSearch(store, engine, benchmark_sampler, width=3, max_depth=10)
    results = await beam.search(math_agent)
    duration = time.time() - start
    best = max(results, key=lambda n: n.score)
    print(f"Beam Search: Time={duration:.4f}s, Best Score={best.score}, Nodes={len(results)}")
    
    # 2. MCTS
    start = time.time()
    mcts = MCTS(store, engine, benchmark_sampler, iterations=50)
    results = await mcts.search(math_agent)
    duration = time.time() - start
    best = max(results, key=lambda n: n.score)
    print(f"MCTS:        Time={duration:.4f}s, Best Score={best.score}, Nodes={len(results)}")
    
    # 3. Best of N
    start = time.time()
    bon = BestOfNSearch(store, engine, benchmark_sampler, n=20)
    results = await bon.search(math_agent)
    duration = time.time() - start
    best = max(results, key=lambda n: n.score)
    print(f"Best-of-N:   Time={duration:.4f}s, Best Score={best.score}, Nodes={len(results)}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
