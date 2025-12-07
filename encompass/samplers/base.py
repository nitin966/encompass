from typing import Protocol, List, Any, Dict
from runtime.node import SearchNode

class Sampler(Protocol):
    async def __call__(self, node: SearchNode, metadata: Dict[str, Any] = None) -> List[Any]:
        """
        Given a search node, return a list of possible inputs (actions) to take.
        """
        ...
