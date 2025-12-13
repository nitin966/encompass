import asyncio
import time
from encompass import compile, branchpoint, record_score
from encompass.std import action, early_stop, kill_branch
from runtime.engine import ExecutionEngine
from storage.filesystem import FileSystemStore
from search.strategies import BestFirstSearch, BeamSearch

# Mock Two Sum Problem
TEST_CASES = [([2, 7, 11, 15], 9, [0, 1]), ([3, 2, 4], 6, [1, 2])]

@action
def run_tests(code_str):
    """Runs unit tests on the generated code."""
    try:
        local_scope = {}
        exec(code_str, {}, local_scope)
        two_sum = local_scope.get('twoSum')
        if not two_sum: return 0.0
        
        passed = 0
        for nums, target, expected in TEST_CASES:
            if two_sum(nums, target) == expected:
                passed += 1
        return passed / len(TEST_CASES)
    except Exception:
        return 0.0

@compile
def reflexion_agent():
    # Attempt 1
    code = branchpoint("generate_code", metadata={"attempt": 1})
    score = run_tests(code)
    
    if score == 1.0:
        record_score(100)
        early_stop()
        return "Solved"
        
    # Attempt 2 (Reflexion)
    reflection = branchpoint("reflect", metadata={"prev_score": score})
    code_v2 = branchpoint("generate_code_v2", metadata={"reflection": reflection})
    score_v2 = run_tests(code_v2)
    
    if score_v2 == 1.0:
        record_score(100)
        early_stop()
        return "Solved"
        
    record_score(score_v2 * 100)
    return "Failed"

async def reflexion_sampler(node, metadata=None):
    # Mock sampler that improves code if reflection is good
    if "attempt" in metadata:
        return ["def twoSum(nums, target): return []", "def twoSum(nums, target): return [0, 1]"]
    if "reflection" in metadata:
        return ["Fixed Code"]
    if "prev_score" in metadata:
        return ["I need to fix the logic"]
    return []

def clean_code(text):
    """Extracts code from markdown blocks or returns raw text."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()

async def create_reflexion_llm_sampler(llm):
    """Creates a sampler that uses a real LLM for Reflexion."""
    async def sampler(node, metadata=None):
        # Unwrap metadata if nested
        if metadata and "metadata" in metadata:
            metadata = metadata["metadata"]
            
        if "attempt" in metadata:
            prompt = (
                "Write a Python function `twoSum(nums, target)` that returns indices of the two numbers such that they add up to target.\n"
                "You may assume that each input would have exactly one solution, and you may not use the same element twice.\n"
                "Return ONLY the Python code inside a markdown block."
            )
            response = await llm.generate(prompt)
            return [clean_code(response)]
            
        if "prev_score" in metadata:
            prompt = (
                f"You wrote code for Two Sum that achieved a score of {metadata['prev_score']} (0.0 to 1.0).\n"
                "Reflect on what might be wrong with the logic. Be concise."
            )
            response = await llm.generate(prompt)
            return [response]
            
        if "reflection" in metadata:
            prompt = (
                f"Your previous attempt failed. Reflection: {metadata['reflection']}\n"
                "Write FIXED Python code for `twoSum(nums, target)` based on this reflection.\n"
                "Return ONLY the Python code inside a markdown block."
            )
            response = await llm.generate(prompt)
            return [clean_code(response)]
            
        return []
    return sampler

async def run_benchmark():
    store = FileSystemStore("reflexion_trace")
    engine = ExecutionEngine()
    
    print(f"--- Reflexion Benchmark ---")
    
    # Best First Search (simulates Re-expansion by prioritizing promising nodes)
    start = time.time()
    befs = BestFirstSearch(store, engine, reflexion_sampler)
    results = await befs.search(reflexion_agent)
    best = max(results, key=lambda n: n.score) if results else None
    print(f"Best-First: Score={best.score if best else 0}, Solved={'Solved' in str(best.metadata) if best else False}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
