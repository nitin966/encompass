from typing import List, Any, Dict
from runtime.node import SearchNode
from encompass.samplers.base import Sampler

class OpenAIFunctionSampler:
    """
    A sampler that uses OpenAI's Function Calling API to generate structured inputs.
    
    It expects the 'BranchPoint' metadata to contain a JSON schema defining the 
    expected input structure.
    """
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.api_key = api_key
        self.model = model

    async def __call__(self, node: SearchNode) -> List[Any]:
        # 1. Extract Schema from Node Metadata
        # In a real app, the BranchPoint would carry the schema
        schema = node.metadata.get('schema')
        prompt = node.metadata.get('prompt', 'Select the next best action.')
        
        if not schema:
            # Fallback to simple options if no schema
            return node.metadata.get('options', [])

        # 2. Construct OpenAI Request
        # We simulate the API call here for the library demo
        # In production, this would use `openai.AsyncOpenAI().chat.completions.create(...)`
        
        # Mocking the response based on the prompt/schema
        # If we are selecting a style, return valid styles
        if "signature_style" in str(node.metadata):
             return [0, 1, 2] # Indices for the styles
             
        return []
