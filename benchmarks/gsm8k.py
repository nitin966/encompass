import asyncio
import time
import re
from encompass import compile, branchpoint, record_score
from encompass.std import action, early_stop
from runtime.engine import ExecutionEngine
from storage.filesystem import FileSystemStore
from search.strategies import BeamSearch, BestOfNSearch

# Mini-Eval Dataset (5 Real GSM8K Problems) with Oracle Steps
DATASET = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": 72,
        "steps": ["Calculate May sales: 48 / 2", "Calculate total: 48 + 24"]
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": 10,
        "steps": ["Convert minutes to hours: 50 / 60", "Calculate earnings: 12 * (50/60)"]
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": 5,
        "steps": ["Calculate current money: 100 / 2", "Calculate parents gift: 15", "Calculate grandparents gift: 15 * 2", "Calculate total money: 50 + 15 + 30", "Calculate needed: 100 - 95"]
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to finish the book tomorrow, how many pages should she read?",
        "answer": 84,
        "steps": ["Calculate today pages: 12 * 2", "Calculate total read: 12 + 24", "Calculate remaining: 120 - 36"]
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": 624,
        "steps": ["Calculate pages per week: 3 * 2 * 2", "Calculate pages per year: 12 * 52"]
    }
]

@action
def calculator(expression):
    """Safe calculator tool."""
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except:
        return None

def create_math_agent(problem_text, expected_answer):
    @compile
    def math_solver():
        # Step 1: Plan
        # We ask for a plan, but for the Oracle demo, we just proceed to steps.
        # The sampler will guide us.
        
        # We loop until we find the answer or hit a limit
        current_val = 0
        for i in range(5):
            # BranchPoint for next step
            # Metadata includes the problem to help the Oracle Sampler
            step = yield branchpoint(f"step_{i}", 
                problem=problem_text,
                context=f"Current value: {current_val}"
            )
            
            if step == "Final Answer":
                break
                
            # Extract expression from step (e.g. "Calculate: 1+1")
            if ":" in step:
                _, expr = step.split(":", 1)
                val = yield calculator(expr.strip())
                if val is not None:
                    current_val = val
            
            # Check if we hit the answer
            if current_val == expected_answer:
                yield record_score(1e9)
                return f"Solved: {current_val}"
        
        yield record_score(0)
        return f"Failed: {current_val}"
    return math_solver

async def solver_sampler(node, metadata=None):
    # Oracle Sampler: Looks up the problem in DATASET and returns the correct next step
    problem_text = metadata.get("problem")
    if not problem_text:
        return ["Wait"]
        
    # Find problem
    problem_data = next((p for p in DATASET if p["question"] == problem_text), None)
    if not problem_data:
        return ["Unknown"]
        
    steps = problem_data["steps"]
    
    # Determine which step we are at based on node depth or history
    step_idx = node.depth
    
    options = []
    if step_idx < len(steps):
        options.append(steps[step_idx])
        # Add distractors
        options.append(f"Wrong step {step_idx}")
    else:
        options.append("Final Answer")
        
    return options

# We don't run_benchmark here anymore, we use the CLI
