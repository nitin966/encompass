import os
import asyncio
import shutil
from examples.translation_agent import create_translation_agent
from core.llm import MockLLM
from runtime.engine import ExecutionEngine
from search.strategies import BeamSearch, MCTS
from storage.filesystem import FileSystemStore
from visualization.exporter import export_to_dot

# 1. Define a Sampler (Async)
async def translation_sampler(node, metadata=None):
    """
    A simple sampler that returns valid indices for the translation agent's choices.
    In a real scenario, this would use an LLM to generate options.
    """
    # Depth 0: Signature Style (3 options)
    if node.depth == 0:
        return [0, 1, 2]
    # Depth 1: Body Style (2 options)
    elif node.depth == 1:
        return [0, 1]
    return []

async def main():
    print("--- Running EnCompass Translation Demo ---\n")

    # Clean up previous trace
    if os.path.exists("encompass_trace"):
        shutil.rmtree("encompass_trace")

    # 2. Setup Components
    llm = MockLLM()
    agent = create_translation_agent(llm)
    engine = ExecutionEngine()
    store = FileSystemStore()

    # 3. Run Beam Search
    print("[Beam Search]")
    beam = BeamSearch(store, engine, translation_sampler, width=3)
    results_beam = await beam.search(agent)
    
    if results_beam:
        # Sort by score
        results_beam.sort(key=lambda n: n.score, reverse=True)
        top_node = results_beam[0]
        print(f"Top Result (Score {top_node.score}):")
        print(top_node.metadata.get('result'))
    else:
        print("No results found.")
    print()

    # 4. Run MCTS
    print("[MCTS]")
    mcts = MCTS(store, engine, translation_sampler, iterations=50)
    results_mcts = await mcts.search(agent)
    
    if results_mcts:
        results_mcts.sort(key=lambda n: n.score, reverse=True)
        top_node = results_mcts[0]
        print(f"Top Result (Score {top_node.score}):")
        print(top_node.metadata.get('result'))
    else:
        print("No results found.")

    # 5. Visualize
    print("\nGenerating Visualization...")
    # Combine nodes from both searches for visualization
    all_nodes = {n.node_id: n for n in results_beam + results_mcts}
    export_to_dot(list(all_nodes.values()), "translation_search_tree")

if __name__ == "__main__":
    asyncio.run(main())
