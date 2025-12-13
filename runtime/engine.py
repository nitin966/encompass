import asyncio
import logging
from collections.abc import Callable, Generator
from typing import Any

from core.signals import (
    BranchPoint,
    ControlSignal,
    EarlyStop,
    Effect,
    KillBranch,
    LocalSearch,
    Protect,
    RecordCosts,
    ScoreSignal,
)
from runtime.costs import CostAggregator
from runtime.node import SearchNode

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Runtime engine for executing AgentMachine instances.

    Features:
    - O(1) Replay: Uses machine.save()/load() to resume execution efficiently.
    - Signal Handling: Processes BranchPoint, Effect, LocalSearch, etc.
    - Caching: Caches results of deterministic steps (optional).
    - Safety: Ensures idempotent execution of effects.
    """

    @staticmethod
    def create_root() -> SearchNode:
        """Creates the initial root node with an empty history."""
        # Root has empty history
        return SearchNode(trace_history=[], depth=0, action_taken="<init>")

    def __init__(self, store=None):
        """
        Initialize execution engine.

        Args:
            store: Optional StateStore for cache persistence
        """
        self.store = store

        # Load cache from store if available
        if store and hasattr(store, "load_cache"):
            self._cache = store.load_cache()
        else:
            self._cache = {}

        self._effect_cache = {}  # scope -> effect_key -> result (Scoped Memoization)
        self._current_scope = "global"

        # Cost tracking
        self.cost_aggregator = CostAggregator()

    def set_scope(self, scope: str):
        """Sets the current caching scope."""
        self._current_scope = scope
        if scope not in self._effect_cache:
            self._effect_cache[scope] = {}

    def clear_scope(self, scope: str):
        """Clears the cache for a specific scope."""
        if scope in self._effect_cache:
            del self._effect_cache[scope]

    def save_cache(self):
        """
        Persist execution cache to disk (if store supports it).
        Call this periodically during long searches to enable resumability.
        """
        if self.store and hasattr(self.store, "save_cache"):
            self.store.save_cache(self._cache)

    def _compute_history_hash(self, history: list[Any]) -> str:
        """
        Computes a robust hash for the history.
        Tries to use JSON for stability, falls back to string representation.
        """
        import hashlib
        import json

        try:
            # Try JSON serialization with sorted keys for determinism
            serialized = json.dumps(history, sort_keys=True)
        except (TypeError, ValueError):
            # Fallback: String representation of the tuple
            # This is less safe for objects with identical __str__ but better than crashing
            serialized = str(tuple(history))

        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    async def step(
        self, agent_factory: Callable[[], Generator], node: SearchNode, input_value: Any = None
    ) -> tuple[SearchNode, ControlSignal | None]:
        """
        Executes the agent.
        1. Replays history from 'node'.
        2. Injects 'input_value' (if provided) as the decision for the current BranchPoint.
        3. Continues execution, handling any 'Effect' signals automatically.
        4. Stops at the next 'BranchPoint' or termination.
        """

        # 1. Construct the history to replay
        # The node.trace_history contains [choice_0, effect_res_1, choice_2, ...]
        # For CPS path, we DON'T add input_value to replay_history because we apply it separately
        # For legacy generator path, we DO add it
        replay_history = list(node.trace_history)

        # We will build the *new* history for the child node as we go
        # It starts as a copy of replay_history, but might grow if we encounter NEW effects
        current_history = list(replay_history)
        if input_value is not None:
            current_history.append(input_value)

        # Check Cache (Optimization)
        history_key = self._compute_history_hash(current_history)
        # Note: The cache might return a state that is "in the middle" of effects if we cached poorly.
        # But our protocol says we only stop at BranchPoints. So cached states are always at BranchPoints.
        if history_key in self._cache:
            return self._reconstruct_from_cache(history_key, current_history, node)

        # 2. Start Execution
        # Check if agent_factory returns an AgentMachine class or instance
        # We need to inspect what agent_factory() returns.
        # But agent_factory is usually a function that returns a generator.
        # If we use the new compiler, agent_factory might be the Machine Class itself?

        # Let's assume agent_factory returns an AgentMachine INSTANCE or a Generator.
        agent_instance = agent_factory()

        from core.compiler import AgentMachine

        if isinstance(agent_instance, AgentMachine):
            # --- O(1) REPLAY PATH ---
            machine = agent_instance

            # Initialize execution state
            current_score = node.score

            # Helper to run machine until next BranchPoint or Done
            async def advance_machine(val=None):
                nonlocal current_score
                sig = machine.run(val)
                while True:
                    if machine._done:
                        return None  # Done

                    if isinstance(sig, ScoreSignal):
                        current_score += sig.value
                        sig = machine.run(None)

                    elif isinstance(sig, Effect):
                        # ... (Effect handling logic) ...
                        effect_key = sig.key
                        if effect_key is None:
                            try:
                                import hashlib

                                payload = str(sig.args) + str(sig.kwargs)
                                arg_hash = hashlib.md5(payload.encode("utf-8")).hexdigest()
                                effect_key = f"{sig.func.__module__}.{sig.func.__name__}:{arg_hash}"
                            except Exception:
                                effect_key = None

                        if self._current_scope not in self._effect_cache:
                            self._effect_cache[self._current_scope] = {}
                        scope_cache = self._effect_cache[self._current_scope]

                        if effect_key and effect_key in scope_cache:
                            result = scope_cache[effect_key]
                        else:
                            if asyncio.iscoroutinefunction(sig.func):
                                result = await sig.func(*sig.args, **sig.kwargs)
                            else:
                                result = sig.func(*sig.args, **sig.kwargs)
                            if effect_key:
                                scope_cache[effect_key] = result

                        sig = machine.run(result)

                    elif isinstance(sig, Protect):
                        attempts = sig.attempts
                        last_exc = None
                        result = None
                        success = False
                        for _ in range(attempts):
                            try:
                                if asyncio.iscoroutinefunction(sig.func):
                                    result = await sig.func(*sig.args, **sig.kwargs)
                                else:
                                    result = sig.func(*sig.args, **sig.kwargs)
                                success = True
                                break
                            except sig.exceptions as e:
                                last_exc = e
                                continue
                        if not success:
                            raise last_exc
                        sig = machine.run(result)

                    elif isinstance(sig, KillBranch):
                        current_score = -1e9
                        # Force done
                        return sig  # Treat as done (will check machine._done? No, KillBranch stops execution)
                        # We should probably return a special signal or set flag?
                        # Let's set machine._done = True manually? No.
                        # We break and return KillBranch signal?

                    elif isinstance(sig, EarlyStop):
                        current_score = 1e9
                        return sig

                    elif isinstance(sig, RecordCosts):
                        # Record cost for this node
                        self.cost_aggregator.record(
                            node_id=str(node.node_id),
                            tokens_in=sig.tokens if hasattr(sig, "tokens") else 0,
                            tokens_out=0,  # RecordCosts doesn't split in/out
                            cost_usd=sig.dollars if hasattr(sig, "dollars") else 0.0,
                        )
                        sig = machine.run(None)

                    elif isinstance(sig, BranchPoint):
                        return sig

                    elif isinstance(sig, LocalSearch):
                        # ... (LocalSearch logic) ...
                        strategy_cls = sig.strategy_factory
                        sub_agent = sig.agent_factory
                        kwargs = sig.kwargs
                        strategy = strategy_cls(None, self, **kwargs)
                        results = await strategy.search(sub_agent)
                        if results:
                            terminals = [n for n in results if n.is_terminal]
                            if terminals:
                                best_result = terminals[0].metadata.get("result")
                            else:
                                best_result = results[0].metadata.get("result")
                            sig = machine.run(best_result)
                        else:
                            sig = machine.run(None)

                    elif isinstance(sig, AgentMachine):
                        if hasattr(machine, "_stack"):
                            machine._stack.append(sig)
                            sig = machine.run(None)
                        else:
                            raise TypeError(
                                "AgentMachine does not support nesting (missing _stack)."
                            )

                    else:
                        raise TypeError(f"Unexpected signal type: {type(sig)}")

            # --- EXECUTION LOGIC ---
            is_done = False
            final_result = None
            last_signal = None

            # 1. Restore State
            if node.machine_state:
                machine.load(node.machine_state)
                # We are at the BranchPoint.
                # The machine is loaded to the state *before* the next BranchPoint is yielded.
                # So we need to run it once to get the BranchPoint signal.
                # However, if input_value is None, it means we are just continuing from a loaded state
                # and expect the next signal to be a BranchPoint.
                # If input_value is not None, we are making a choice at the loaded BranchPoint.
                # Let's assume machine.load() puts us *at* the BranchPoint, ready for input.
                # So, if input_value is None, we are just getting the BranchPoint signal.
                # If input_value is not None, we are providing the input to the BranchPoint.
                # This means the machine.run(input_value) will be the first call.
                # If input_value is None, we need to get the signal.
                if input_value is None:
                    last_signal = await advance_machine(None)  # Get the BranchPoint signal
            else:
                # Replay from start
                # Run to first BranchPoint
                last_signal = await advance_machine(None)

                # Replay history
                for stored_input in replay_history:
                    if isinstance(last_signal, BranchPoint):
                        last_signal = await advance_machine(stored_input)
                    else:
                        # Should not happen if history matches execution
                        # This indicates a mismatch between expected BranchPoint and actual signal
                        # Or that replay_history contains non-BranchPoint items that were not handled
                        # by advance_machine (e.g., Effects).
                        # The advance_machine handles effects internally, so replay_history should only
                        # contain inputs for BranchPoints.
                        pass

            # 2. Apply Input (if provided)
            if input_value is not None:
                # We expect to be at a BranchPoint (last_signal)
                last_signal = await advance_machine(input_value)

            # Check completion
            if last_signal is None and machine._done:
                is_done = True
                final_result = machine._result
            elif isinstance(last_signal, (KillBranch, EarlyStop)):
                is_done = True
                final_result = None  # Or some status?

            # Update Cache
            history_key = self._compute_history_hash(
                node.trace_history + ([input_value] if input_value is not None else [])
            )
            self._cache[history_key] = (current_score, last_signal, is_done, final_result)

            # Create Child Node
            child = SearchNode(
                trace_history=node.trace_history + [input_value] if input_value is not None else [],
                score=current_score,
                depth=node.depth + 1 if input_value is not None else node.depth,
                parent_id=node.node_id,
                is_terminal=is_done,
                action_taken=str(input_value) if input_value is not None else "<auto>",
                metadata={"result": final_result} if is_done else {},
                machine_state=machine.save()
                if not is_done
                else None,  # Only save state if not terminal
            )
            return child, last_signal

        # --- LEGACY GENERATOR PATH (O(N) Replay) ---
        # We maintain a stack of generators to support nested agents
        # For this path, we need input_value in replay_history
        if input_value is not None:
            replay_history.append(input_value)

        gen_stack = [agent_instance]
        current_gen = gen_stack[-1]

        current_score = 0.0
        last_signal = None
        is_done = False
        final_result = None

        # Helper to advance the current generator stack
        def advance_generator(val=None):
            nonlocal current_gen
            try:
                sig = current_gen.send(val)

                # Handle nested generator yield
                while isinstance(sig, Generator):
                    # Push new generator
                    gen_stack.append(sig)
                    current_gen = sig
                    # Start it
                    sig = current_gen.send(None)

                return sig
            except StopIteration as e:
                # Current generator finished
                if len(gen_stack) > 1:
                    # Pop and return result to parent
                    gen_stack.pop()
                    current_gen = gen_stack[-1]
                    # Pass the result of the sub-agent back to the parent
                    return advance_generator(e.value)
                else:
                    # Root agent finished
                    raise e

        try:
            # Start the root generator
            # We use send(None) initially
            signal = advance_generator(None)

            # --- REPLAY PHASE ---
            for stored_input in replay_history:
                # 1. Handle Scores
                while isinstance(signal, ScoreSignal):
                    current_score += signal.value
                    signal = advance_generator(None)

                # 2. Inject stored input
                if isinstance(signal, (BranchPoint, Effect)):
                    signal = advance_generator(stored_input)
                else:
                    raise TypeError(
                        f"During replay, expected BranchPoint or Effect, got {type(signal)}"
                    )

            # --- FRONTIER PHASE ---
            while True:
                # Consume scores
                while isinstance(signal, ScoreSignal):
                    current_score += signal.value
                    signal = advance_generator(None)

                if isinstance(signal, Effect):
                    # ... (Effect handling logic) ...
                    effect_key = signal.key
                    if effect_key is None:
                        try:
                            import hashlib

                            payload = str(signal.args) + str(signal.kwargs)
                            arg_hash = hashlib.md5(payload.encode("utf-8")).hexdigest()
                            effect_key = (
                                f"{signal.func.__module__}.{signal.func.__name__}:{arg_hash}"
                            )
                        except Exception:
                            effect_key = None

                    # Check Scoped Cache
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
                    signal = advance_generator(result)

                elif isinstance(signal, Protect):
                    # Handle Protect
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
                    signal = advance_generator(result)

                elif isinstance(signal, KillBranch):
                    current_score = -1e9
                    is_done = True
                    break

                elif isinstance(signal, EarlyStop):
                    current_score = 1e9
                    is_done = True
                    break

                elif isinstance(signal, RecordCosts):
                    # Record cost for this node
                    self.cost_aggregator.record(
                        node_id=str(node.node_id),
                        tokens_in=signal.tokens if hasattr(signal, "tokens") else 0,
                        tokens_out=0,
                        cost_usd=signal.dollars if hasattr(signal, "dollars") else 0.0,
                    )
                    current_history.append(None)
                    signal = advance_generator(None)

                elif isinstance(signal, BranchPoint):
                    last_signal = signal
                    break

                else:
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
            depth=node.depth + 1
            if input_value is not None
            else node.depth,  # Depth increments on *choices*? Or steps? Let's say choices.
            parent_id=node.node_id,
            is_terminal=is_done,
            action_taken=str(input_value) if input_value is not None else "<auto>",
            metadata={"result": final_result} if is_done else {},
        )

        return child, last_signal

    def _reconstruct_from_cache(
        self, key: str, history: list[Any], parent: SearchNode
    ) -> tuple[SearchNode, Any]:
        """Helper to reconstruct node from cache."""
        score, signal, is_done, result = self._cache[key]
        child = SearchNode(
            trace_history=history,
            score=score,
            depth=parent.depth + 1,  # Approx
            parent_id=parent.node_id,
            is_terminal=is_done,
            action_taken="<cached>",
            metadata={"result": result} if is_done else {},
        )
        return child, signal
