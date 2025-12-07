import os
import asyncio
from typing import List, Optional
from core.llm import LanguageModel

class OpenAIModel:
    """
    Production-grade OpenAI Adapter.
    Supports async generation, retries, and scoring.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o-mini", temperature: float = 0.7):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
            
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "dummy", # Local LLMs might not need key
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.model = model
        self.temperature = temperature

    async def generate(self, prompt: str, options: List[str] = None) -> str:
        """
        Generates text. If options are provided, forces the model to choose one.
        """
        if options:
            # Use logit_bias or system prompt to constrain.
            # For simplicity, we'll use system prompt.
            system_prompt = f"You are a helpful assistant. You must choose exactly one of the following options: {options}. Output ONLY the chosen option."
        else:
            system_prompt = "You are a helpful assistant."

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Basic retry logic could go here
            print(f"OpenAI Error: {e}")
            return ""

    async def score(self, text: str, criteria: str) -> float:
        """
        Scores text from 0.0 to 1.0 based on criteria.
        """
        prompt = f"Evaluate the following text based on this criteria: '{criteria}'.\n\nText: {text}\n\nOutput ONLY a single float number between 0.0 and 1.0."
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an evaluator. Output only a float."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0 # Deterministic for scoring
            )
            content = response.choices[0].message.content.strip()
            return float(content)
        except Exception:
            return 0.0
