import functools
import hashlib
import json
import inspect
from typing import Any, Callable, Dict, Optional

# Global registry of side-effect results
# Key: (function_name, args_hash) -> Result
_SIDE_EFFECT_CACHE: Dict[str, Any] = {}

def _compute_args_hash(args, kwargs) -> str:
    """Computes a stable hash for function arguments."""
    try:
        # Try JSON first
        payload = {'args': args, 'kwargs': kwargs}
        serialized = json.dumps(payload, sort_keys=True)
    except (TypeError, ValueError):
        # Fallback to string representation
        serialized = str(args) + str(kwargs)
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()

def idempotent(func: Callable) -> Callable:
    """
    Decorator to mark a function as having side effects.
    
    During replay, if the function has been called with the same arguments
    in the same 'context' (not fully enforced here, but assumed via cache),
    it returns the cached result instead of re-executing.
    
    In a full implementation, this would be tied to the ExecutionEngine's
    current trace ID to ensure path-specificity.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # In a real system, we'd check if we are in "Replay Mode" vs "Frontier Mode"
        # For now, we implement a simple global cache to demonstrate the concept.
        
        key = f"{func.__module__}.{func.__name__}:{_compute_args_hash(args, kwargs)}"
        
        if key in _SIDE_EFFECT_CACHE:
            return _SIDE_EFFECT_CACHE[key]
        
        result = func(*args, **kwargs)
        _SIDE_EFFECT_CACHE[key] = result
        return result
        
    return wrapper

class SideEffectGuard:
    """
    Context manager to control side effects.
    """
    def __enter__(self):
        # Could set a global flag here
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def clear_side_effect_cache():
    """Clears the global side effect cache."""
    _SIDE_EFFECT_CACHE.clear()
