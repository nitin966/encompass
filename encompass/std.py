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
    from core.signals import KillBranch
    return KillBranch()

def early_stop():
    """
    Signal to stop searching this branch (success).
    """
    from core.signals import EarlyStop
    return EarlyStop()

def record_costs(tokens: int = 0, dollars: float = 0.0):
    """
    Signal to record resource usage.
    """
    from core.signals import RecordCosts
    return RecordCosts(tokens=tokens, dollars=dollars)

class NoCopy:
    """Marker for objects that should not be deep-copied across branches."""
    pass

def protect(func, attempts=3, exceptions=(Exception,)):
    """
    Signal to execute a function with retries managed by the engine.
    
    Usage:
        result = yield protect(risky_func, attempts=3)
    """
    from core.signals import Protect
    def wrapper(*args, **kwargs):
        return Protect(func=func, args=args, kwargs=kwargs, attempts=attempts, exceptions=exceptions)
    return wrapper
