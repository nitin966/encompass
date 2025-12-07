import logging
import asyncio
from typing import Any, Tuple, Union, Generator, Callable, List
from runtime.node import SearchNode
from core.signals import BranchPoint, ScoreSignal, Effect, Protect, KillBranch, EarlyStop, RecordCosts, ControlSignal

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    The core runtime that executes agent programs using a Replay Architecture.
    
    It manages the execution of generator-based agents by replaying their history
    of inputs to reach a specific state, and then injecting new inputs to proceed.
    """
    @staticmethod
    def create_root() -> SearchNode:
        """Creates the initial root node with an empty history."""
        # Root has empty history
        return SearchNode(trace_history=[], depth=0, action_taken="<init>")

    def __init__(self):
        # Cache: history_hash -> (score, last_signal, is_terminal, final_result)
        self._cache = {}
        self._cache = {} # history_hash -> (score, last_signal, is_terminal, final_result)
        self._effect_cache = {} # scope -> effect_key -> result (Scoped Memoization)
        self._current_scope = "global"

    def set_scope(self, scope: str):
        """Sets the current caching scope."""
        self._current_scope = scope
        if scope not in self._effect_cache:
            self._effect_cache[scope] = {}

    def clear_scope(self, scope: str):
        """Clears the cache for a specific scope."""
        if scope in self._effect_cache:
            del self._effect_cache[scope]

    def _compute_history_hash(self, history: List[Any]) -> str:
        """
        Computes a robust hash for the history.
        Tries to use JSON for stability, falls back to string representation.
        """
        import json
        import hashlib
        
        try:
            # Try JSON serialization with sorted keys for determinism
            serialized = json.dumps(history, sort_keys=True)
        except (TypeError, ValueError):
            # Fallback: String representation of the tuple
            # This is less safe for objects with identical __str__ but better than crashing
            serialized = str(tuple(history))
            
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()

    async def step(self, agent_factory: Callable[[], Generator], node: SearchNode, input_value: Any = None) -> Tuple[SearchNode, Union[ControlSignal, None]]:
        """
        Executes the agent.
        1. Replays history from 'node'.
        2. Injects 'input_value' (if provided) as the decision for the current BranchPoint.
        3. Continues execution, handling any 'Effect' signals automatically.
        4. Stops at the next 'BranchPoint' or termination.
        """
        from core.signals import Effect, BranchPoint, ScoreSignal
        
        # 1. Construct the history to replay
        # The node.trace_history contains [choice_0, effect_res_1, choice_2, ...]
        replay_history = list(node.trace_history)
        if input_value is not None:
            replay_history.append(input_value)
            
        # We will build the *new* history for the child node as we go
        # It starts as a copy of replay_history, but might grow if we encounter NEW effects
        current_history = list(replay_history)
        
        # Check Cache (Optimization)
        history_key = self._compute_history_hash(current_history)
        # Note: The cache might return a state that is "in the middle" of effects if we cached poorly.
        # But our protocol says we only stop at BranchPoints. So cached states are always at BranchPoints.
        if history_key in self._cache:
             return self._reconstruct_from_cache(history_key, current_history, node)

        # 2. Start Execution
        gen = agent_factory()
        current_score = 0.0
        last_signal = None
        is_done = False
        final_result = None
        
        try:
            signal = next(gen)
            
            # --- REPLAY PHASE ---
            # We consume the replay_history items one by one.
            # Each item corresponds to a yield in the generator (BranchPoint or Effect).
            
            for stored_input in replay_history:
                # 1. Handle Scores (they don't consume history)
                while isinstance(signal, ScoreSignal):
                    current_score += signal.value
                    signal = next(gen)
                
                # 2. Inject stored input
                # Whether it's a BranchPoint (choice) or Effect (result), we just send it back.
                if isinstance(signal, (BranchPoint, Effect)):
                    signal = gen.send(stored_input)
                else:
                    raise TypeError(f"During replay, expected BranchPoint or Effect, got {type(signal)}")

            # --- FRONTIER PHASE ---
            # We have exhausted the history. Now we run forward.
            # We auto-execute Effects until we hit a BranchPoint or Finish.
            
            while True:
                # Consume scores
                while isinstance(signal, ScoreSignal):
                    current_score += signal.value
                    signal = next(gen)
                
                if isinstance(signal, Effect):
                    # ... (Effect handling code is same, just need to preserve it or re-write it)
                    # I will rewrite it to be safe since I am replacing a block
                    effect_key = signal.key
                    if effect_key is None:
                        try:
                            import json
                            import hashlib
                            payload = str(signal.args) + str(signal.kwargs)
                            arg_hash = hashlib.md5(payload.encode('utf-8')).hexdigest()
                            effect_key = f"{signal.func.__module__}.{signal.func.__name__}:{arg_hash}"
                        except Exception:
                            effect_key = None
                    
                    
                    # 2. Check Scoped Cache (Memoization)
                    # Ensure scope dict exists
                    if self._current_scope not in self._effect_cache:
                        self._effect_cache[self._current_scope] = {}
                        
                    scope_cache = self._effect_cache[self._current_scope]
                    
                    if effect_key and effect_key in scope_cache:
                        result = scope_cache[effect_key]
                    else:
                        if asyncio.iscoroutinefunction(signal.func):
                             result = await signal.func(*signal.args, **signal.kwargs)
                        else:
                             result = signal.func(*signal.args, **signal.kwargs)
                        if effect_key:
                            scope_cache[effect_key] = result
                    
                    current_history.append(result)
                    signal = gen.send(result)
                
                elif isinstance(signal, Protect):
                    # Handle Protect: Retry logic
                    attempts = signal.attempts
                    last_exc = None
                    result = None
                    success = False
                    
                    for _ in range(attempts):
                        try:
                            if asyncio.iscoroutinefunction(signal.func):
                                result = await signal.func(*signal.args, **signal.kwargs)
                            else:
                                result = signal.func(*signal.args, **signal.kwargs)
                            success = True
                            break
                        except signal.exceptions as e:
                            last_exc = e
                            continue
                    
                    if not success:
                        raise last_exc
                        
                    current_history.append(result)
                    signal = gen.send(result)

                elif isinstance(signal, KillBranch):
                    # Prune branch
                    # We can raise a special exception or just return None
                    # Returning None (or special status) stops the loop
                    # Let's set score to -inf and break
                    current_score = -1e9
                    is_done = True
                    break
                    
                elif isinstance(signal, EarlyStop):
                    # Success
                    current_score = 1e9
                    is_done = True
                    break
                    
                elif isinstance(signal, RecordCosts):
                    # Just record and continue
                    # In a real system, we'd aggregate this in the Node
                    # For now, we ignore or log
                    current_history.append(None) # Ack
                    signal = gen.send(None)
                    
                    
                elif isinstance(signal, BranchPoint):
                    # PAUSE! We need an external decision.
                    last_signal = signal
                    break
                    
                else:
                    # Should be impossible if typed correctly, but safety
                    raise TypeError(f"Unexpected signal type: {type(signal)}")

        except StopIteration as e:
            is_done = True
            final_result = e.value
            
        # Update Cache
        # We cache the state at this specific history point (which corresponds to a BranchPoint or End)
        history_key = self._compute_history_hash(current_history)
        self._cache[history_key] = (current_score, last_signal, is_done, final_result)
        
        # Create Child Node
        child = SearchNode(
            trace_history=current_history,
            score=current_score,
            depth=node.depth + 1 if input_value is not None else node.depth, # Depth increments on *choices*? Or steps? Let's say choices.
            parent_id=node.node_id,
            is_terminal=is_done,
            action_taken=str(input_value) if input_value is not None else "<auto>",
            metadata={'result': final_result} if is_done else {}
        )
        
        return child, last_signal

    def _reconstruct_from_cache(self, key: str, history: List[Any], parent: SearchNode) -> Tuple[SearchNode, Any]:
        """Helper to reconstruct node from cache."""
        score, signal, is_done, result = self._cache[key]
        child = SearchNode(
            trace_history=history,
            score=score,
            depth=parent.depth + 1, # Approx
            parent_id=parent.node_id,
            is_terminal=is_done,
            action_taken="<cached>",
            metadata={'result': result} if is_done else {}
        )
        return child, signal
