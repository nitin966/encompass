"""Base Reflexion Agent (WITHOUT EnCompass) - State Machine Version.

This implements the Reflexion pattern with explicit state management,
retry loops, and reflection - all the complexity that EnCompass abstracts away.

KEY METRICS:
- Lines of code: ~400 lines
- Complexity: High (manual state, reflection loops, test management)
- Compare to: encompass_reflexion_agent.py (~130 lines with same functionality)
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from encompass.llm.ollama import OllamaModel
from problems import CodingProblem, get_problems


# ============================================================================
# STATE DEFINITIONS - Manual state management required without EnCompass
# ============================================================================

class ReflexionState(Enum):
    """States in the Reflexion state machine."""
    INIT = "init"
    GENERATING = "generating"
    TESTING = "testing"
    REFLECTING = "reflecting"
    IMPROVING = "improving"
    SOLVED = "solved"
    FAILED = "failed"


@dataclass
class ProblemState:
    """State for a single problem - manual bookkeeping."""
    problem: CodingProblem
    current_code: str = ""
    current_func: Optional[Callable] = None
    state: ReflexionState = ReflexionState.INIT
    attempts: int = 0
    max_attempts: int = 3
    accuracy: float = 0.0
    reflection: str = ""
    test_results: str = ""
    error_message: str = ""


@dataclass
class AgentState:
    """Global agent state - must track all variables explicitly."""
    model_name: str
    problem_states: Dict[str, ProblemState] = field(default_factory=dict)
    current_problem_idx: int = 0
    total_problems: int = 0
    solved_count: int = 0
    failed_count: int = 0
    total_attempts: int = 0


# ============================================================================
# PROMPTS
# ============================================================================

GENERATE_PROMPT = """You are an expert Python programmer.

Problem: {description}

Function signature: {signature}

Write a Python function that solves this problem.
Return ONLY the Python function code, no explanations or markdown.
"""

REFLECT_PROMPT = """Your code failed some test cases.

Problem: {description}
Your code:
```python
{code}
```

Test results:
{test_results}

Reflect on why your code failed. What is wrong with the logic?
Be specific and concise.
"""

IMPROVE_PROMPT = """Your previous code had issues.

Problem: {description}
Your previous code:
```python
{code}
```

Your reflection: {reflection}

Write an IMPROVED Python function that fixes the issues.
Return ONLY the Python function code, no explanations.
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
        func = local_scope.get(func_name)
        if func is None:
            return None, f"No '{func_name}' function found"
        return func, ""
    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except Exception as e:
        return None, f"Execution error: {e}"


def format_test_results(problem: CodingProblem, func: Callable) -> str:
    """Format test results showing expected vs actual."""
    lines = []
    for args, expected in problem.test_cases:
        try:
            if isinstance(args, tuple):
                result = func(*args)
            else:
                result = func(args)
            status = "✓" if result == expected else "✗"
            lines.append(f"  {status} {problem.function_name}{args} = {result} (expected: {expected})")
        except Exception as e:
            lines.append(f"  ✗ {problem.function_name}{args} raised {type(e).__name__}: {e}")
    return "\n".join(lines)


# ============================================================================
# STATE MACHINE STEP FUNCTIONS - The complexity EnCompass eliminates
# ============================================================================

async def step_init(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, ReflexionState]:
    """Initialize problem states."""
    for problem in get_problems():
        problem_state = ProblemState(problem=problem)
        state.problem_states[problem.name] = problem_state
        state.total_problems += 1
    
    if state.total_problems == 0:
        return state, ReflexionState.FAILED
    
    return state, ReflexionState.GENERATING


async def step_generate(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, ReflexionState]:
    """Generate initial code for current problem."""
    problem_names = list(state.problem_states.keys())
    if state.current_problem_idx >= len(problem_names):
        return state, ReflexionState.SOLVED
    
    current_name = problem_names[state.current_problem_idx]
    problem_state = state.problem_states[current_name]
    
    # Skip if already solved
    if problem_state.state == ReflexionState.SOLVED:
        state.current_problem_idx += 1
        return state, ReflexionState.GENERATING
    
    # Check attempt limit
    if problem_state.attempts >= problem_state.max_attempts:
        problem_state.state = ReflexionState.FAILED
        state.failed_count += 1
        state.current_problem_idx += 1
        return state, ReflexionState.GENERATING
    
    problem_state.attempts += 1
    state.total_attempts += 1
    print(f"  Generating code for '{problem_state.problem.name}' (attempt {problem_state.attempts}/{problem_state.max_attempts})...")
    
    try:
        prompt = GENERATE_PROMPT.format(
            description=problem_state.problem.description,
            signature=problem_state.problem.function_signature
        )
        response = await llm.generate(prompt, max_tokens=1024)
        code = clean_code(response)
        problem_state.current_code = code
        problem_state.state = ReflexionState.TESTING
        
        return state, ReflexionState.TESTING
        
    except Exception as e:
        problem_state.error_message = str(e)
        return state, ReflexionState.GENERATING


async def step_test(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, ReflexionState]:
    """Test the generated code."""
    problem_names = list(state.problem_states.keys())
    current_name = problem_names[state.current_problem_idx]
    problem_state = state.problem_states[current_name]
    
    # Compile the function
    func, error = compile_function(problem_state.current_code, problem_state.problem.function_name)
    
    if func is None:
        print(f"    Compilation failed: {error}")
        problem_state.error_message = error
        problem_state.test_results = f"Compilation error: {error}"
        problem_state.state = ReflexionState.REFLECTING
        return state, ReflexionState.REFLECTING
    
    problem_state.current_func = func
    
    # Run tests
    accuracy, correct, total = problem_state.problem.validate(func)
    problem_state.accuracy = accuracy
    problem_state.test_results = format_test_results(problem_state.problem, func)
    
    if accuracy == 1.0:
        print(f"    ✓ Solved! ({correct}/{total} tests passed)")
        problem_state.state = ReflexionState.SOLVED
        state.solved_count += 1
        state.current_problem_idx += 1
        return state, ReflexionState.GENERATING
    else:
        print(f"    Partial: {correct}/{total} tests passed ({accuracy*100:.0f}%)")
        problem_state.state = ReflexionState.REFLECTING
        return state, ReflexionState.REFLECTING


async def step_reflect(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, ReflexionState]:
    """Reflect on why the code failed."""
    problem_names = list(state.problem_states.keys())
    current_name = problem_names[state.current_problem_idx]
    problem_state = state.problem_states[current_name]
    
    print(f"    Reflecting on failure...")
    
    try:
        prompt = REFLECT_PROMPT.format(
            description=problem_state.problem.description,
            code=problem_state.current_code,
            test_results=problem_state.test_results
        )
        reflection = await llm.generate(prompt, max_tokens=512)
        problem_state.reflection = reflection.strip()
        problem_state.state = ReflexionState.IMPROVING
        
        return state, ReflexionState.IMPROVING
        
    except Exception as e:
        problem_state.error_message = str(e)
        return state, ReflexionState.GENERATING


async def step_improve(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, ReflexionState]:
    """Generate improved code based on reflection."""
    problem_names = list(state.problem_states.keys())
    current_name = problem_names[state.current_problem_idx]
    problem_state = state.problem_states[current_name]
    
    # Check attempt limit
    if problem_state.attempts >= problem_state.max_attempts:
        problem_state.state = ReflexionState.FAILED
        state.failed_count += 1
        state.current_problem_idx += 1
        return state, ReflexionState.GENERATING
    
    problem_state.attempts += 1
    state.total_attempts += 1
    print(f"    Improving code (attempt {problem_state.attempts}/{problem_state.max_attempts})...")
    
    try:
        prompt = IMPROVE_PROMPT.format(
            description=problem_state.problem.description,
            code=problem_state.current_code,
            reflection=problem_state.reflection
        )
        response = await llm.generate(prompt, max_tokens=1024)
        code = clean_code(response)
        problem_state.current_code = code
        problem_state.state = ReflexionState.TESTING
        
        return state, ReflexionState.TESTING
        
    except Exception as e:
        problem_state.error_message = str(e)
        return state, ReflexionState.GENERATING


# ============================================================================
# MAIN STATE MACHINE DRIVER
# ============================================================================

async def run_state_machine(state: AgentState, llm: OllamaModel) -> AgentState:
    """Run the Reflexion state machine."""
    current_state = ReflexionState.INIT
    
    while current_state not in (ReflexionState.SOLVED, ReflexionState.FAILED):
        if current_state == ReflexionState.INIT:
            state, current_state = await step_init(state, llm)
            
        elif current_state == ReflexionState.GENERATING:
            state, current_state = await step_generate(state, llm)
            if state.current_problem_idx >= state.total_problems:
                break
                
        elif current_state == ReflexionState.TESTING:
            state, current_state = await step_test(state, llm)
            
        elif current_state == ReflexionState.REFLECTING:
            state, current_state = await step_reflect(state, llm)
            
        elif current_state == ReflexionState.IMPROVING:
            state, current_state = await step_improve(state, llm)
    
    return state


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def run_reflexion(
    model: str = "qwen2.5:32b",
    output_dir: Path = None,
) -> dict:
    """Run the base Reflexion agent."""
    base_dir = Path(__file__).parent
    
    if output_dir is None:
        output_dir = base_dir / "output" / "base_agent"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("BASE REFLEXION AGENT (Without EnCompass)")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print(f"Lines of code in this agent: ~400")
    print()
    
    llm = OllamaModel(model=model, temperature=0.3)
    
    state = AgentState(model_name=model)
    final_state = await run_state_machine(state, llm)
    
    # Build results
    results = {
        "model": model,
        "agent": "base",
        "agent_lines": 400,
        "problems": [],
        "solved_count": final_state.solved_count,
        "failed_count": final_state.failed_count,
        "total_attempts": final_state.total_attempts,
    }
    
    for name, problem_state in final_state.problem_states.items():
        results["problems"].append({
            "name": name,
            "difficulty": problem_state.problem.difficulty,
            "status": "solved" if problem_state.state == ReflexionState.SOLVED else "failed",
            "accuracy": problem_state.accuracy,
            "attempts": problem_state.attempts,
            "code": problem_state.current_code,
        })
        
        # Save code to file
        if problem_state.current_code:
            (output_dir / f"{name}.py").write_text(problem_state.current_code)
    
    print(f"\nResults: {results['solved_count']}/{final_state.total_problems} problems solved")
    print(f"Total LLM calls: {results['total_attempts']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Base Reflexion Agent")
    parser.add_argument("--model", default="qwen2.5:32b", help="Ollama model name")
    args = parser.parse_args()
    
    results = asyncio.run(run_reflexion(model=args.model))
