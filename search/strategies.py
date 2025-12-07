import logging
import asyncio
from typing import List, Callable, Any
from runtime.engine import ExecutionEngine
from runtime.node import SearchNode
from storage.base import StateStore
from core.signals import BranchPoint, ScoreSignal

logger = logging.getLogger(__name__)

class BeamSearch:
    """
    Implements Beam Search strategy for exploring the agent's execution tree.
    
    Attributes:
        store: StateStore for saving visited nodes.
        engine: ExecutionEngine for running the agent.
        sampler: Async function to generate possible actions for a node.
        width: The beam width (number of top candidates to keep at each depth).
        max_depth: Maximum depth to search.
    """
    def __init__(self, store: StateStore, engine: ExecutionEngine, sampler: Callable, width: int = 3, max_depth: int = 10, diversity_penalty: float = 0.0):
        self.store = store
        self.engine = engine
        self.sampler = sampler
        self.width = width
        self.max_depth = max_depth
        self.diversity_penalty = diversity_penalty

    async def search(self, agent_factory: Callable) -> List[SearchNode]:
        root = self.engine.create_root()
        # Initial step to get first signal
        root, signal = await self.engine.step(agent_factory, root, None)
        self.store.save_node(root)
        
        # Frontier is list of (node, signal) tuples
        frontier = [(root, signal)]
        self.visited = {root}
        completed = []
        
        for d in range(self.max_depth):
            if not frontier:
                break
                
            candidates = []
            active_items = [item for item in frontier if not item[0].is_terminal]
            terminal_items = [item for item in frontier if item[0].is_terminal]
            
            # Add terminal nodes to completed
            completed.extend([n for n, s in terminal_items])
            
            if not active_items:
                break
            
            # Parallel Sampling: Pass metadata from the signal
            # signal is the LAST signal received. If it's a BranchPoint, it has metadata.
            tasks = []
            for node, signal in active_items:
                meta = signal.metadata if isinstance(signal, BranchPoint) else {}
                tasks.append(self.sampler(node, meta))
            
            inputs_list = await asyncio.gather(*tasks)
            
            for (node, _), inputs in zip(active_items, inputs_list):
                for inp in inputs:
                    child, new_signal = await self.engine.step(agent_factory, node, inp)
                    self.store.save_node(child)
                    candidates.append((child, new_signal))
                    self.visited.add(child)
            
            # Sort candidates by score
            # Diversity: Penalize nodes that share the same parent to encourage breadth
            # Adjusted Score = Score - (diversity_penalty * sibling_count)
            if self.diversity_penalty > 0:
                parent_counts = {}
                for node, _ in candidates:
                    parent_counts[node.parent_id] = parent_counts.get(node.parent_id, 0) + 1
                
                # Re-sort with penalty
                # We want to pick the best, but if we pick many from same parent, their effective score drops?
                # Actually, we should select iteratively.
                selected = []
                temp_candidates = list(candidates)
                temp_candidates.sort(key=lambda x: x[0].score, reverse=True)
                
                counts = {}
                while len(selected) < self.width and temp_candidates:
                    # Find best adjusted score
                    best_idx = -1
                    best_val = float('-inf')
                    
                    for i, (node, _) in enumerate(temp_candidates):
                        cnt = counts.get(node.parent_id, 0)
                        adj_score = node.score - (self.diversity_penalty * cnt)
                        if adj_score > best_val:
                            best_val = adj_score
                            best_idx = i
                    
                    if best_idx != -1:
                        item = temp_candidates.pop(best_idx)
                        selected.append(item)
                        counts[item[0].parent_id] = counts.get(item[0].parent_id, 0) + 1
                    else:
                        break
                frontier = selected
            else:
                candidates.sort(key=lambda item: item[0].score, reverse=True)
                frontier = candidates[:self.width]
            
        # Add any remaining frontier nodes to completed if they are terminal or just return all visited
        # Return all visited nodes, sorted by score
        all_nodes = list(self.visited)
        all_nodes.sort(key=lambda n: n.score, reverse=True)
        return all_nodes

class MCTS:
    """
    Implements Monte Carlo Tree Search (MCTS) with UCT selection.
    """
    def __init__(self, store: StateStore, engine: ExecutionEngine, sampler: Callable, iterations: int = 100, exploration_weight: float = 1.414):
        self.store = store
        self.engine = engine
        self.sampler = sampler
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        
        self.nodes = {}  # node_id -> SearchNode
        self.children = {} # node_id -> List[SearchNode]
        self.visits = {} # node_id -> int
        self.values = {} # node_id -> float (total value)
        self.signals = {} # node_id -> ControlSignal (to store metadata)
        
        # Min-Max tracking for normalization
        self.min_score = float('inf')
        self.max_score = float('-inf')

    async def search(self, agent_factory: Callable) -> List[SearchNode]:
        import math
        import random
        
        # Initialize Root
        root = self.engine.create_root()
        root, signal = await self.engine.step(agent_factory, root, None)
        self.store.save_node(root)
        
        self.nodes[root.node_id] = root
        self.signals[root.node_id] = signal
        self.visits[root.node_id] = 0
        self.values[root.node_id] = 0.0
        
        for _ in range(self.iterations):
            node = root
            path = [node]
            
            # Selection
            while node.node_id in self.children and self.children[node.node_id]:
                # UCT Selection with Normalization
                best_child = None
                best_uct = -float('inf')
                
                for child in self.children[node.node_id]:
                    if child.node_id not in self.visits:
                        best_child = child
                        break
                        
                    if self.visits[child.node_id] == 0:
                        uct = float('inf')
                    else:
                        # Normalize Q value
                        q_val = self.values[child.node_id] / self.visits[child.node_id]
                        if self.max_score > self.min_score:
                            q_normalized = (q_val - self.min_score) / (self.max_score - self.min_score)
                        else:
                            q_normalized = q_val # Fallback if no range yet
                            
                        u = self.exploration_weight * math.sqrt(math.log(self.visits[node.node_id]) / self.visits[child.node_id])
                        uct = q_normalized + u
                    
                    if uct > best_uct:
                        best_uct = uct
                        best_child = child
                
                node = best_child
                path.append(node)
            
            # Expansion
            if not node.is_terminal:
                # Get metadata from stored signal
                signal = self.signals.get(node.node_id)
                meta = signal.metadata if isinstance(signal, BranchPoint) else {}
                
                inputs = await self.sampler(node, meta)
                new_children = []
                for inp in inputs:
                    child, new_signal = await self.engine.step(agent_factory, node, inp)
                    self.store.save_node(child)
                    new_children.append(child)
                    
                    self.nodes[child.node_id] = child
                    self.signals[child.node_id] = new_signal
                    self.visits[child.node_id] = 0
                    self.values[child.node_id] = 0.0
                
                self.children[node.node_id] = new_children
                
                if new_children:
                    node = random.choice(new_children)
                    path.append(node)
            
            # Simulation (Rollout)
            # Simple random rollout until terminal
            current_rollout_node = node
            rollout_signal = self.signals.get(current_rollout_node.node_id)
            
            while not current_rollout_node.is_terminal:
                meta = rollout_signal.metadata if isinstance(rollout_signal, BranchPoint) else {}
                possible_inputs = await self.sampler(current_rollout_node, meta)
                if not possible_inputs:
                    break
                action = random.choice(possible_inputs)
                current_rollout_node, rollout_signal = await self.engine.step(agent_factory, current_rollout_node, action)
            
            rollout_score = current_rollout_node.score
            
            # Update Min-Max for normalization
            self.min_score = min(self.min_score, rollout_score)
            self.max_score = max(self.max_score, rollout_score)
            
            # Backpropagation
            for n in path:
                if n.node_id not in self.visits: self.visits[n.node_id] = 0
                if n.node_id not in self.values: self.values[n.node_id] = 0.0
                
                self.visits[n.node_id] += 1
                self.values[n.node_id] += rollout_score

        # Return all visited nodes, sorted by visit count (most robust metric for MCTS)
        all_nodes = list(self.nodes.values())
        all_nodes.sort(key=lambda n: self.visits.get(n.node_id, 0), reverse=True)
        return all_nodes

class BestOfNSearch:
    """
    Global Best-of-N Strategy.
    Runs the agent N times (using the sampler to make choices) and picks the best result.
    """
    def __init__(self, store: StateStore, engine: ExecutionEngine, sampler: Callable, n: int = 10):
        self.store = store
        self.engine = engine
        self.sampler = sampler
        self.n = n

    async def search(self, agent_factory: Callable) -> List[SearchNode]:
        import random
        
        results = []
        
        for _ in range(self.n):
            # Run one full trajectory
            node = self.engine.create_root()
            node, signal = await self.engine.step(agent_factory, node, None)
            self.store.save_node(node)
            
            while not node.is_terminal:
                meta = signal.metadata if isinstance(signal, BranchPoint) else {}
                possible_inputs = await self.sampler(node, meta)
                if not possible_inputs:
                    break
                action = random.choice(possible_inputs)
                node, signal = await self.engine.step(agent_factory, node, action)
                self.store.save_node(node)
            
            results.append(node)
            
        # Sort by score
        results.sort(key=lambda n: n.score, reverse=True)
        return results

class BFS:
    """Breadth-First Search."""
    def __init__(self, store: StateStore, engine: ExecutionEngine, sampler: Callable, max_depth: int = 10):
        self.store = store
        self.engine = engine
        self.sampler = sampler
        self.max_depth = max_depth

    async def search(self, agent_factory: Callable) -> List[SearchNode]:
        root = self.engine.create_root()
        root, signal = await self.engine.step(agent_factory, root, None)
        self.store.save_node(root)
        
        queue = [(root, signal)]
        visited = {root}
        
        while queue:
            node, signal = queue.pop(0)
            
            if node.is_terminal or node.depth >= self.max_depth:
                continue
                
            meta = signal.metadata if isinstance(signal, BranchPoint) else {}
            inputs = await self.sampler(node, meta)
            
            for inp in inputs:
                child, new_signal = await self.engine.step(agent_factory, node, inp)
                self.store.save_node(child)
                visited.add(child)
                queue.append((child, new_signal))
                
        return list(visited)

class DFS:
    """Depth-First Search."""
    def __init__(self, store: StateStore, engine: ExecutionEngine, sampler: Callable, max_depth: int = 10):
        self.store = store
        self.engine = engine
        self.sampler = sampler
        self.max_depth = max_depth

    async def search(self, agent_factory: Callable) -> List[SearchNode]:
        root = self.engine.create_root()
        root, signal = await self.engine.step(agent_factory, root, None)
        self.store.save_node(root)
        
        stack = [(root, signal)]
        visited = {root}
        
        while stack:
            node, signal = stack.pop()
            
            if node.is_terminal or node.depth >= self.max_depth:
                continue
                
            meta = signal.metadata if isinstance(signal, BranchPoint) else {}
            inputs = await self.sampler(node, meta)
            
            # Reverse inputs to preserve order when popping from stack
            for inp in reversed(inputs):
                child, new_signal = await self.engine.step(agent_factory, node, inp)
                self.store.save_node(child)
                visited.add(child)
                stack.append((child, new_signal))
                
        return list(visited)

class BestFirstSearch:
    """Best-First Search using a Priority Queue on scores."""
    def __init__(self, store: StateStore, engine: ExecutionEngine, sampler: Callable, max_depth: int = 10, beam_width: int = 1000):
        self.store = store
        self.engine = engine
        self.sampler = sampler
        self.max_depth = max_depth
        self.beam_width = beam_width # Soft limit to prevent explosion

    async def search(self, agent_factory: Callable) -> List[SearchNode]:
        import heapq
        
        root = self.engine.create_root()
        root, signal = await self.engine.step(agent_factory, root, None)
        self.store.save_node(root)
        
        # Min-heap, so store negative score
        pq = [(-root.score, 0, root, signal)] # (neg_score, tie_breaker, node, signal)
        visited = {root}
        count = 0
        
        while pq:
            neg_score, _, node, signal = heapq.heappop(pq)
            
            if node.is_terminal or node.depth >= self.max_depth:
                continue
                
            meta = signal.metadata if isinstance(signal, BranchPoint) else {}
            inputs = await self.sampler(node, meta)
            
            for inp in inputs:
                child, new_signal = await self.engine.step(agent_factory, node, inp)
                self.store.save_node(child)
                visited.add(child)
                
                count += 1
                heapq.heappush(pq, (-child.score, count, child, new_signal))
                
            # Prune if too large
            if len(pq) > self.beam_width:
                pq = heapq.nsmallest(self.beam_width, pq)
                heapq.heapify(pq)
                
        return list(visited)
