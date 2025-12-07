import argparse
import asyncio
import json
import time
import random
from runtime.engine import ExecutionEngine
from storage.filesystem import FileSystemStore
from search.strategies import BeamSearch, MCTS, BestOfNSearch, BestFirstSearch, BFS, DFS

# Import benchmarks
from benchmarks.gsm8k import create_math_agent, solver_sampler, DATASET as GSM8K_DATASET
from benchmarks.arc_hypothesis import arc_agent, arc_sampler
from benchmarks.reflexion import reflexion_agent, reflexion_sampler

def get_strategy(name, store, engine, sampler, **kwargs):
    if name == "beam":
        return BeamSearch(store, engine, sampler, width=kwargs.get("width", 3), diversity_penalty=kwargs.get("diversity", 0.0))
    elif name == "mcts":
        return MCTS(store, engine, sampler, iterations=kwargs.get("iterations", 100))
    elif name == "best_of_n":
        return BestOfNSearch(store, engine, sampler, n=kwargs.get("n", 10))
    elif name == "bfs":
        return BFS(store, engine, sampler)
    elif name == "dfs":
        return DFS(store, engine, sampler)
    elif name == "befs":
        return BestFirstSearch(store, engine, sampler)
    else:
        raise ValueError(f"Unknown strategy: {name}")

async def run(args):
    # Setup
    random.seed(args.seed)
    
    # Determine Dataset and Agent Factory
    tasks = []
    if args.benchmark == "gsm8k":
        for i, item in enumerate(GSM8K_DATASET):
            agent_factory = create_math_agent(item["question"], item["answer"])
            tasks.append((f"gsm8k_{i}", agent_factory, solver_sampler))
    elif args.benchmark == "arc":
        # TODO: Add real ARC dataset support
        tasks.append(("arc_mock", arc_agent, arc_sampler))
    elif args.benchmark == "reflexion":
        tasks.append(("reflexion_mock", reflexion_agent, reflexion_sampler))
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
        
    print(f"--- Running {args.benchmark} ({len(tasks)} tasks) with {args.strategy} (Seed={args.seed}) ---")
    
    total_solved = 0
    total_nodes = 0
    start_time = time.time()
    
    for task_id, agent_factory, sampler in tasks:
        store = FileSystemStore(f"traces/{args.benchmark}_{args.strategy}_{args.seed}/{task_id}")
        engine = ExecutionEngine()
        engine.set_scope(f"{task_id}_{args.seed}")
        
        strategy = get_strategy(args.strategy, store, engine, sampler, 
                                width=args.width, 
                                n=args.n, 
                                iterations=args.iterations,
                                diversity=args.diversity)
        
        results = await strategy.search(agent_factory)
        
        best_node = max(results, key=lambda n: n.score) if results else None
        solved = "Solved" in str(best_node.metadata) if best_node else False
        
        if solved:
            total_solved += 1
        total_nodes += len(results)
        
        print(f"Task {task_id}: Solved={solved}, Best Score={best_node.score if best_node else 0}")

    duration = time.time() - start_time
    accuracy = total_solved / len(tasks) if tasks else 0.0
    
    metrics = {
        "benchmark": args.benchmark,
        "strategy": args.strategy,
        "seed": args.seed,
        "tasks": len(tasks),
        "accuracy": accuracy,
        "total_nodes": total_nodes,
        "duration": duration
    }
    
    print(json.dumps(metrics, indent=2))
    
    # Save metrics
    if args.output:
        with open(args.output, "a") as f:
            f.write(json.dumps(metrics) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnCompass Benchmark Runner")
    parser.add_argument("--benchmark", type=str, required=True, choices=["gsm8k", "arc", "reflexion"])
    parser.add_argument("--strategy", type=str, required=True, choices=["beam", "mcts", "best_of_n", "bfs", "dfs", "befs"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, help="Output JSONL file")
    
    # Strategy params
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--diversity", type=float, default=0.0)
    
    args = parser.parse_args()
    asyncio.run(run(args))
