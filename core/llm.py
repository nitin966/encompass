from typing import Protocol, List, Any

class LanguageModel(Protocol):
    async def generate(self, prompt: str, options: List[str] = None) -> str:
        """
        Generate text based on prompt. 
        If options are provided, return one of the options.
        """
        ...
        
    async def score(self, text: str, criteria: str) -> float:
        """
        Score the text based on criteria. Returns 0.0 to 1.0.
        """
        ...

class MockLLM:
    async def generate(self, prompt: str, options: List[str] = None) -> str:
        # Simple heuristic or random choice for mock
        if options:
            # Deterministic mock: pick the longest option just to have logic
            # or just pick the first one.
            # Let's pick based on a hash of the prompt to be deterministic but varied
            import hashlib
            idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(options)
            return options[idx]
        return "Mock response"

    async def score(self, text: str, criteria: str) -> float:
        # Mock scoring: length based or random
        return 0.8
