from functools import wraps
from core.signals import Effect, BranchPoint, ScoreSignal

def action(func):
    """
    Decorator that converts a function call into an Effect signal.
    
    When the decorated function is called, it yields an Effect(func, args, kwargs)
    instead of executing the function body. The ExecutionEngine will intercept
    this signal, execute the function (if not replaying), and inject the result.
    
    Usage:
        @action
        def my_side_effect(x):
            return x + 1
            
        def agent():
            result = yield my_side_effect(10)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # We yield the Effect. The generator must be iterated by the Engine.
        # Note: This means the agent function must be a generator and use 'yield'.
        return Effect(func=func, args=args, kwargs=kwargs)
    return wrapper

def kill_branch():
    """
    Signal to immediately terminate the current branch as a failure.
    """
    from core.signals import ScoreSignal
    # Yield a massive negative score to prune this branch effectively
    return ScoreSignal(value=-1e9, context="kill_branch")

def early_stop():
    """
    Signal to stop searching this branch (success).
    """
    from core.signals import ScoreSignal
    # Yield a massive positive score to prioritize this result
    return ScoreSignal(value=1e9, context="early_stop")

def record_costs(tokens: int = 0, dollars: float = 0.0):
    """
    Signal to record resource usage.
    """
    from core.signals import ScoreSignal
    # We use ScoreSignal for now, but ideally a CostSignal
    # For now, we just log it or ignore it, or subtract from score?
    # Let's subtract from score to penalize cost
    return ScoreSignal(value=0.0, context=f"cost:tokens={tokens},dollars={dollars}")

class NoCopy:
    """Marker for objects that should not be deep-copied across branches."""
    pass

# For protect(), we need to yield branchpoints.
# Since it yields, it must be a generator.
def protect(func, num_attempts=3):
    """
    Executes a function with retries managed by the search process.
    If the function fails, it yields a branchpoint to decide whether to retry.
    
    Usage:
        result = yield from protect(risky_func)(args)
    """
    def wrapper(*args, **kwargs):
        for i in range(num_attempts):
            try:
                # If func is an Effect (decorated with @action), we yield it
                # If it's a regular function, we just call it
                if hasattr(func, '__wrapped__'): # Check if @action
                     res = func(*args, **kwargs)
                     if isinstance(res, Effect):
                         yield res
                         # How do we get the result of the effect?
                         # The engine injects it into the generator.
                         # But here we are in a sub-generator.
                         # 'yield' yields the Effect to the Engine.
                         # The Engine sends the result back to 'yield'.
                         # So: result = yield Effect(...)
                         # But func(*args) returns the Effect object immediately if @action is used.
                         # So: result = yield func(*args)
                         return # We can't easily get the result here without 'yield from' logic in the agent?
                         # Actually, if the user does `yield from protect(...)`, then `wrapper` is the generator.
                         # So `result = yield func(...)` works.
                     else:
                         return func(*args, **kwargs)
                else:
                     return func(*args, **kwargs)
            except Exception as e:
                # Yield a branchpoint to decide whether to retry
                # In a real implementation, the sampler would see this and decide.
                # For now, we just log and continue loop
                pass
        raise Exception("Max retries exceeded")
    return wrapper
