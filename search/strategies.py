import logging
import asyncio
from typing import List, Callable, Any
from runtime.engine import ExecutionEngine
from storage.base import StateStore
from core.signals import BranchPoint

logger = logging.getLogger(__name__)

class BeamSearch:
    def __init__(self, store: StateStore, engine: ExecutionEngine, sampler: Callable, width=3, max_depth=10):
        self.store = store
        self.engine = engine
        self.sampler = sampler
        self.width = width
        self.max_depth = max_depth

    async def search(self, agent_factory: Callable) -> List:
        # Create root (empty history)
        root = self.engine.create_root()
        
        # Prime the root
        root, _ = await self.engine.step(agent_factory, root, None)
        self.store.save_node(root)
        
        frontier = [root]
        self.visited = {root}
        completed = []

        for d in range(self.max_depth):
            if not frontier: break
            
            candidates = []
            
            # Parallel Sampling: Get inputs for all nodes in frontier at once
            # We assume sampler is async
            tasks = [self.sampler(node) for node in frontier if not node.is_terminal]
            
            # Map back results to nodes
            # Note: We need to handle terminal nodes carefully.
            # Let's filter frontier to only non-terminal first
            active_nodes = [n for n in frontier if not n.is_terminal]
            terminal_nodes = [n for n in frontier if n.is_terminal]
            completed.extend(terminal_nodes)
            
            if not active_nodes:
                break

            inputs_list = await asyncio.gather(*[self.sampler(n) for n in active_nodes])
            
            # Now expand
            for node, inputs in zip(active_nodes, inputs_list):
                for inp in inputs:
                    child, signal = await self.engine.step(agent_factory, node, inp)
                    self.store.save_node(child)
                    candidates.append(child)
                    self.visited.add(child)

            candidates.sort(key=lambda n: n.score, reverse=True)
            frontier = candidates[:self.width]

        completed.extend(frontier)
        return list(self.visited)

class MCTS:
    """
    Monte Carlo Tree Search (MCTS) strategy.
    
    Uses UCT (Upper Confidence Bound for Trees) for selection and performs
    random rollouts to estimate the value of new nodes.
    
    Attributes:
        store: StateStore to save nodes.
        engine: ExecutionEngine to run the agent.
        sampler: Function to generate possible actions from a node.
        iterations: Number of MCTS iterations to run.
        exploration_weight: UCT exploration constant (Cp).
    """
    def __init__(self, store: StateStore, engine: ExecutionEngine, sampler: Callable, iterations=100, exploration_weight=1.41):
        self.store = store
        self.engine = engine
        self.sampler = sampler
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.nodes = {} # Map id -> node
        self.children = {} # Map id -> list of children nodes
        self.visits = {} # Map id -> int
        self.values = {} # Map id -> float (total score)

    async def search(self, agent_factory: Callable) -> List:
        import math
        import random
        
        # Initialize Root
        root = self.engine.create_root()
        root, _ = await self.engine.step(agent_factory, root, None)
        self.store.save_node(root)
        
        self.nodes[root.node_id] = root
        self.visits[root.node_id] = 0
        self.values[root.node_id] = 0.0
        
        for _ in range(self.iterations):
            node = root
            path = [node]
            
            # Selection
            while node.node_id in self.children and self.children[node.node_id]:
                # UCT Selection
                best_child = None
                best_uct = -float('inf')
                
                for child in self.children[node.node_id]:
                    if child.node_id not in self.visits:
                        best_child = child
                        break
                        
                    if self.visits[child.node_id] == 0:
                        uct = float('inf')
                    else:
                        q = self.values[child.node_id] / self.visits[child.node_id]
                        u = self.exploration_weight * math.sqrt(math.log(self.visits[node.node_id]) / self.visits[child.node_id])
                        uct = q + u
                    
                    if uct > best_uct:
                        best_uct = uct
                        best_child = child
                
                node = best_child
                path.append(node)
                
            # Expansion
            if not node.is_terminal:
                inputs = await self.sampler(node)
                new_children = []
                for inp in inputs:
                    child, _ = await self.engine.step(agent_factory, node, inp)
                    self.store.save_node(child)
                    new_children.append(child)
                    
                    self.nodes[child.node_id] = child
                    self.visits[child.node_id] = 0
                    self.values[child.node_id] = 0.0
                
                self.children[node.node_id] = new_children
                
                if new_children:
                    node = random.choice(new_children)
                    path.append(node)

            # Simulation (Rollout)
            rollout_node = node
            while not rollout_node.is_terminal:
                possible_inputs = await self.sampler(rollout_node)
                if not possible_inputs:
                    break 
                action = random.choice(possible_inputs)
                rollout_node, _ = await self.engine.step(agent_factory, rollout_node, action)
            
            rollout_score = rollout_node.score

            # Backpropagation
            for n in path:
                self.visits[n.node_id] += 1
                self.values[n.node_id] += rollout_score

        return list(self.nodes.values())
