from functools import wraps
from core.signals import Effect

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
