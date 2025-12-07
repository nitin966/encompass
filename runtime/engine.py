import logging
from typing import Any, Tuple, Union, Generator, Callable, List
from runtime.node import SearchNode
from core.signals import BranchPoint, ScoreSignal, ControlSignal

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Core engine for executing EnCompass agents.
    
    Handles the replay-based execution model, injecting history into the agent
    generator and capturing the next signal (BranchPoint or ScoreSignal).
    """
    @staticmethod
    def create_root() -> SearchNode:
        # Root has empty history
        return SearchNode(trace_history=[], depth=0, action_taken="<init>")

    def __init__(self):
        # Cache: tuple(history) -> (score, last_signal, is_terminal, final_result)
        self._cache = {}

    def step(self, agent_factory: Callable[[], Generator], node: SearchNode, input_value: Any = None) -> Tuple[SearchNode, Union[ControlSignal, None]]:
        """
        Executes one step of the agent by replaying history and injecting the new input.
        
        Args:
            agent_factory: Function that returns a new agent generator.
            node: Current search node (contains history).
            input_value: New input to inject (action taken).
            
        Returns:
            Tuple of (new_child_node, last_signal_received).
        """
        # Construct new history
        new_history = list(node.trace_history)
        if input_value is not None:
            new_history.append(input_value)
            
        # Check Cache
        history_key = tuple(str(x) for x in new_history) # Simple string serialization for key
        if history_key in self._cache:
            current_score, last_signal, is_done, final_result = self._cache[history_key]
        else:
            # Replay Phase
            gen = agent_factory()
            
            current_score = 0.0
            last_signal = None
            is_done = False
            final_result = None
            
            try:
                signal = next(gen)
                
                # Replay history
                for stored_input in new_history:
                    # Consume score signals
                    while isinstance(signal, ScoreSignal):
                        current_score += signal.value
                        signal = next(gen)
                    
                    # Inject stored input at BranchPoint
                    if isinstance(signal, BranchPoint):
                        signal = gen.send(stored_input)

                # Frontier Phase: Consume trailing scores
                while isinstance(signal, ScoreSignal):
                    current_score += signal.value
                    signal = next(gen)
                    
                last_signal = signal

            except StopIteration as e:
                is_done = True
                final_result = e.value
            
            # Update Cache
            self._cache[history_key] = (current_score, last_signal, is_done, final_result)
        
        # Create Child Node
        child_node = SearchNode(
            trace_history=new_history,
            score=current_score,
            depth=len(new_history),
            parent_id=node.node_id,
            is_terminal=is_done,
            action_taken=str(input_value)
        )
        
        if is_done:
            child_node.metadata['result'] = final_result
            
        return child_node, last_signal
