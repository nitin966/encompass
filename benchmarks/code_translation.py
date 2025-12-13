import sys
import os

# Add project root to sys.path to allow imports from core, encompass, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import random
from typing import List, Dict, Any
from encompass import branchpoint, effect, record_score
from encompass.std import local_search, calculator, early_stop
from core.compiler import compile_agent
from search.strategies import BeamSearch, BestOfNSearch
from runtime.engine import ExecutionEngine

# Mock Repository
REPO = {
    "utils.py": ["def add(a, b): pass", "def sub(a, b): pass"],
    "main.py": ["def main(): pass"]
}

# Mock LLM
async def translate_function(code: str, style: str) -> str:
    # Simulate translation
    return f"{code.replace('pass', 'return 0')} # Style: {style}"

async def run_tests(code: str) -> bool:
    # Simulate testing
    return "return 0" in code

# File 1: utils.py
# Func 1: add
def add_translator():
    style = branchpoint("style_choice_add", options=["concise", "verbose"])
    new_code = effect(translate_function, "def add(a, b): pass", style)
    record_score(1.0)
    return new_code

AddTranslator = compile_agent(add_translator)
async def add_sampler(node, metadata=None): return ["concise", "verbose"]

# Func 2: sub
def sub_translator():
    style = branchpoint("style_choice_sub", options=["concise", "verbose"])
    new_code = effect(translate_function, "def sub(a, b): pass", style)
    record_score(1.0)
    return new_code
    
SubTranslator = compile_agent(sub_translator)
async def sub_sampler(node, metadata=None): return ["concise", "verbose"]

@compile_agent
def translation_agent():
    total_score = 0
    translated_funcs = []
    
    ls_add = local_search(BestOfNSearch, AddTranslator, sampler=add_sampler, n=2)
    res_add = yield ls_add
    translated_funcs.append(res_add)
    
    res_sub = yield local_search(BestOfNSearch, SubTranslator, sampler=sub_sampler, n=2)
    translated_funcs.append(res_sub)
    
    return "Translation Complete"

async def main():
    engine = ExecutionEngine()
    root = engine.create_root()
    
    node, signal = await engine.step(translation_agent, root)
    
    print(f"Final Node: {node}")
    print(f"Result: {node.metadata.get('result')}")

if __name__ == "__main__":
    asyncio.run(main())
