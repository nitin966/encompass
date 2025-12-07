from typing import List, Any
from encompass.samplers.base import Sampler
from runtime.node import SearchNode
from core.llm import LanguageModel

class LLMSampler:
    """
    A sampler that uses an LLM to generate options for a branch point.
    It expects the node metadata to contain a 'prompt' or 'options' key.
    """
    def __init__(self, llm: LanguageModel):
        self.llm = llm

    async def __call__(self, node: SearchNode, metadata: Dict[str, Any] = None) -> List[Any]:
        # If the node represents a BranchPoint, we might have metadata
        # In this simple implementation, we assume the agent logic defines the options
        # and we are just selecting them. 
        
        # However, for a true "AI" sampler, we might want to GENERATE options.
        # For this demo parity, we'll stick to the logic used in the translation agent
        # where options are fixed, but we could extend this to ask the LLM.
        
        # For now, let's implement a "Smart Sampler" that filters options based on LLM score
        # or simply passes through options if they are defined in the agent.
        
        # NOTE: In the current architecture, the 'options' are defined inside the generator
        # and not easily accessible unless passed to branchpoint().
        # Let's assume branchpoint("name", options=[...]) was called.
        
        options = node.metadata.get('options')
        if options:
            return options
            
        # Fallback: If no options are explicit, maybe we generate them?
        # This requires the agent to support open-ended input.
        return []
