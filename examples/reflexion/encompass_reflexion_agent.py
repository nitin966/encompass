"""EnCompass Reflexion Agent - Simple Version with Branchpoint.

This implements the SAME functionality as base_reflexion_agent.py but using
EnCompass primitives. Notice how much simpler the code is:

KEY METRICS:
- Lines of code: ~140 lines (vs ~400 in base_reflexion_agent.py)
- Complexity: Low (EnCompass handles state, reflection loops automatically)
- Same functionality: Generate, test, reflect, improve

The key differences:
1. No manual state machine
2. No explicit reflection loops  
3. branchpoint() marks exploration points
4. record_score() provides feedback for search
5. Linear, readable control flow
"""

import asyncio
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from encompass import branchpoint, record_score
from encompass.llm.ollama import OllamaModel
from problems import CodingProblem, get_problems


# ============================================================================
# PROMPTS (same as base agent)
# ============================================================================

GENERATE_PROMPT = """You are an expert Python programmer.

Problem: {description}

Function signature: {signature}

Write a Python function that solves this problem.
Return ONLY the Python function code, no explanations.
"""

IMPROVE_PROMPT = """Your code failed some tests. 

Problem: {description}
Previous code: 
{code}

Test results: {test_results}

Write an IMPROVED Python function.
Return ONLY the Python function code.
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_code(response: str) -> str:
    """Clean LLM response to extract Python code."""
    code = response.strip()
    for prefix in ["```python", "```"]:
        if code.startswith(prefix):
            code = code[len(prefix):]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def compile_function(code: str, func_name: str) -> Tuple[Optional[Callable], str]:
    """Safely compile and extract the function."""
    try:
        local_scope = {}
        exec(code, {"__builtins__": __builtins__}, local_scope)
        return local_scope.get(func_name), ""
    except Exception as e:
        return None, str(e)


# ============================================================================
# ENCOMPASS AGENT - Linear control flow, no state machine!
# ============================================================================

async def solve_problem(llm: OllamaModel, problem: CodingProblem, max_attempts: int = 3) -> Tuple[str, float, bool]:
    """Solve a coding problem with Reflexion.
    
    With EnCompass:
    - branchpoint() marks where search can explore alternatives
    - record_score() tells the search strategy how good this path is
    - No manual state machine needed!
    """
    print(f"  Solving '{problem.name}'...")
    
    # Initial code generation - branchpoint allows exploring alternatives
    branchpoint("initial_generation")
    
    prompt = GENERATE_PROMPT.format(
        description=problem.description,
        signature=problem.function_signature
    )
    response = await llm.generate(prompt, max_tokens=1024)
    code = clean_code(response)
    
    # Test and iterate (Reflexion loop)
    for attempt in range(max_attempts):
        func, error = compile_function(code, problem.function_name)
        
        if func is None:
            test_results = f"Compilation error: {error}"
            accuracy = 0.0
        else:
            accuracy, correct, total = problem.validate(func)
            
            if accuracy == 1.0:
                record_score(accuracy * 100)
                print(f"    ✓ Solved! ({correct}/{total} tests passed)")
                return code, accuracy, True
            
            # Format test results for reflection
            test_results = f"{correct}/{total} tests passed"
        
        # Not solved - branchpoint for improvement direction
        if attempt < max_attempts - 1:
            branchpoint("improvement_direction")
            
            improve_prompt = IMPROVE_PROMPT.format(
                description=problem.description,
                code=code,
                test_results=test_results
            )
            response = await llm.generate(improve_prompt, max_tokens=1024)
            code = clean_code(response)
    
    # Record final score
    record_score(accuracy * 100)
    print(f"    Partial: {accuracy*100:.0f}%")
    
    return code, accuracy, False


async def run_reflexion(
    model: str = "qwen2.5:32b",
    output_dir: Path = None,
) -> dict:
    """Run the EnCompass Reflexion agent.
    
    Dramatically simpler than base_reflexion_agent.py:
    - No ReflexionState enum
    - No ProblemState/AgentState dataclasses
    - No step_* functions
    - Just linear, sequential code!
    """
    base_dir = Path(__file__).parent
    
    if output_dir is None:
        output_dir = base_dir / "output" / "encompass_agent"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ENCOMPASS REFLEXION AGENT (With EnCompass)")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print(f"Lines of code in this agent: ~140 (vs ~400 for base agent)")
    print()
    
    llm = OllamaModel(model=model, temperature=0.3)
    problems = get_problems()
    
    results = {
        "model": model,
        "agent": "encompass",
        "agent_lines": 140,
        "problems": [],
        "solved_count": 0,
        "failed_count": 0,
    }
    
    # Simple loop - no state machine needed!
    for problem in problems:
        code, accuracy, solved = await solve_problem(llm, problem)
        
        if solved:
            results["solved_count"] += 1
        else:
            results["failed_count"] += 1
        
        results["problems"].append({
            "name": problem.name,
            "difficulty": problem.difficulty,
            "status": "solved" if solved else "failed",
            "accuracy": accuracy,
            "code": code,
        })
        
        # Save code
        if code:
            (output_dir / f"{problem.name}.py").write_text(code)
    
    print(f"\nResults: {results['solved_count']}/{len(problems)} problems solved")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EnCompass Reflexion Agent")
    parser.add_argument("--model", default="qwen2.5:32b", help="Ollama model name")
    args = parser.parse_args()
    
    results = asyncio.run(run_reflexion(model=args.model))
