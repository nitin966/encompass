from typing import Protocol, List, Any
from runtime.node import SearchNode

class Sampler(Protocol):
    async def __call__(self, node: SearchNode) -> List[Any]:
        """
        Given a search node, return a list of possible inputs (actions) to take.
        """
        ...
