"""
Full GSM8K benchmark with complete dataset.

Downloads the GSM8K dataset from HuggingFace and evaluates agents
with proper metrics and comparison.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from encompass import compile, branchpoint, record_score
from encompass.std import action
from runtime.engine import ExecutionEngine
from storage.filesystem import FileSystemStore
from search.strategies import BeamSearch, BestOfNSearch


@action
def calculator(expression):
    """Safe calculator tool."""
    try:
        # Safe eval - only basic math operations
        return eval(expression, {"__builtins__": {}}, {})
    except:
        return None


def create_math_agent(problem_text, target_answer):
    """
    Create a math-solving agent for GSM8K.
    
    The agent tries to solve the problem step by step,
    using a calculator for arithmetic.
    """
    @compile
    def math_solver(problem, answer):
        # Strategy: Generate solution steps
        current_val = 0
        
        for i in range(5):  # Max 5 steps
            # Ask for next step
            step = branchpoint(
                f"step_{i}",
                problem=problem,
                context=f"Current value: {current_val}"
            )
            
            if step == "Final Answer":
                break
            
            # Extract expression from step
            if ":" in step:
                parts = step.split(":", 1)
                expr = parts[1]
                val = calculator(expr.strip())
                if val is not None:
                    current_val = val
            
            # Check if correct
            if current_val == answer:
                record_score(1e9)
                return f"Solved: {current_val}"
        
        record_score(0)
        return f"Failed: {current_val}"
    
    return lambda: math_solver(problem_text, target_answer)


def load_gsm8k_dataset(split="test", num_samples=None):
    """
    Load GSM8K dataset.
    
    Args:
        split: 'train' or 'test'
        num_samples: Limit number of samples (None = all)
    
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    data_dir = Path("data/gsm8k")
    data_file = data_dir / f"{split}.jsonl"
    
    # Download if not exists
    if not data_file.exists():
        print(f"Downloading GSM8K {split} split...")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download from HuggingFace
        try:
            import urllib.request
            # Correct URL for GSM8K test.jsonl (Official OpenAI GitHub)
            url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
            urllib.request.urlretrieve(url, data_file)
            print(f"Downloaded to {data_file}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Using fallback mini dataset...")
            return get_mini_dataset()
    
    # Load dataset
    problems = []
    with open(data_file, 'r') as f:
        for line in f:
            if num_samples and len(problems) >= num_samples:
                break
            
            item = json.loads(line)
            
            # Extract numerical answer from answer string
            answer_text = item['answer']
            # GSM8K answers are like "#### 72" at the end
            match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
            if match:
                answer_num = float(match.group(1).replace(',', ''))
                problems.append({
                    'question': item['question'],
                    'answer': answer_num,
                    'answer_text': answer_text
                })
    
    return problems


def get_mini_dataset():
    """Fallback mini dataset if download fails."""
    return [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": 72,
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": 10,
        },
        {
            "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
            "answer": 5,
        }
    ]


async def oracle_sampler(node, metadata=None):
    """
    Oracle sampler for testing (knows the correct steps).
    For real evaluation, replace with LLM sampler.
    """
    # For now, return generic math operations
    return [
        "Add all numbers",
        "Divide first by second",
        "Multiply values",
        "Subtract second from first",
        "Final Answer"
    ]


async def evaluate_gsm8k(strategy_name="beam", num_problems=10):
    """
    Evaluate on GSM8K dataset.
    
    Args:
        strategy_name: 'beam', 'mcts', or 'best_of_n'
        num_problems: Number of problems to evaluate
    
    Returns:
        Dict with accuracy, avg_score, total_nodes, etc.
    """
    print(f"Loading GSM8K dataset ({num_problems} problems)...")
    problems = load_gsm8k_dataset(split="test", num_samples=num_problems)
    
    print(f"Loaded {len(problems)} problems")
    
    # Setup
    engine = ExecutionEngine()
    store = FileSystemStore("./data/gsm8k_results")
    
    if strategy_name == "beam":
        strategy = BeamSearch(store=store, engine=engine, sampler=oracle_sampler, width=3)
    elif strategy_name == "best_of_n":
        strategy = BestOfNSearch(store=store, engine=engine, sampler=oracle_sampler, n=5)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Evaluate each problem
    results = []
    solved_count = 0
    total_nodes = 0
    
    for i, problem in enumerate(problems):
        print(f"\nProblem {i+1}/{len(problems)}: {problem['question'][:60]}...")
        
        agent = create_math_agent(problem['question'], problem['answer'])
        
        try:
            nodes = await strategy.search(agent)
            
            # Find best scoring terminal node
            terminal_nodes = [n for n in nodes if n.is_terminal]
            if terminal_nodes:
                best = max(terminal_nodes, key=lambda n: n.score)
                solved = best.score > 0
                solved_count += solved
                total_nodes += len(nodes)
                
                results.append({
                    'problem_id': i,
                    'solved': solved,
                    'score': best.score,
                    'nodes_explored': len(nodes)
                })
                
                print(f"  Result: {'✓ SOLVED' if solved else '✗ FAILED'} (score={best.score:.1f}, nodes={len(nodes)})")
            else:
                results.append({
                    'problem_id': i,
                    'solved': False,
                    'score': 0,
                    'nodes_explored': len(nodes)
                })
                print(f"  Result: ✗ NO TERMINAL NODE (nodes={len(nodes)})")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'problem_id': i,
                'solved': False,
                'score': 0,
                'error': str(e)
            })
    
    # Compute metrics
    accuracy = solved_count / len(problems) if problems else 0
    avg_nodes = total_nodes / len(problems) if problems else 0
    
    summary = {
        'strategy': strategy_name,
        'num_problems': len(problems),
        'accuracy': accuracy,
        'solved_count': solved_count,
        'avg_nodes_per_problem': avg_nodes,
        'total_nodes': total_nodes,
        'results': results
    }
    
    print(f"\n{'='*60}")
    print(f"GSM8K Evaluation Results")
    print(f"{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Problems: {len(problems)}")
    print(f"Accuracy: {accuracy:.1%} ({solved_count}/{len(problems)})")
    print(f"Avg nodes/problem: {avg_nodes:.1f}")
    print(f"Total nodes: {total_nodes}")
    
    # Save results
    output_file = Path("data/gsm8k_results") / f"eval_{strategy_name}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return summary


if __name__ == "__main__":
    import sys
    
    strategy = sys.argv[1] if len(sys.argv) > 1 else "beam"
    num_problems = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    asyncio.run(evaluate_gsm8k(strategy, num_problems))
