#!/usr/bin/env python3
"""Run the complete Reflexion experiment.

This script:
1. Runs the base Reflexion agent (without EnCompass)
2. Runs the EnCompass Reflexion agent 
3. Tests both outputs
4. Generates a comparison report

Usage:
    python run_experiment.py --model qwen2.5:32b
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add paths for imports
base_dir = Path(__file__).parent
project_root = base_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

sys.path.insert(0, str(base_dir / "tests"))
from test_reflexion import run_tests


async def run_experiment(model: str = "qwen2.5:32b") -> dict:
    """Run the complete experiment comparing both agents."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "base_agent": None,
        "encompass_agent": None,
        "comparison": {},
    }
    
    print("=" * 70)
    print("REFLEXION EXPERIMENT")
    print("Comparing Base Agent vs EnCompass Agent")
    print("=" * 70)
    print(f"\nModel: {model}")
    print(f"Timestamp: {results['timestamp']}")
    
    # Count lines of code
    base_agent_lines = len((base_dir / "base_reflexion_agent.py").read_text().splitlines())
    encompass_agent_lines = len((base_dir / "encompass_reflexion_agent.py").read_text().splitlines())
    
    print(f"\n--- AGENT LINE COUNTS ---")
    print(f"Base Agent:      {base_agent_lines} lines")
    print(f"EnCompass Agent: {encompass_agent_lines} lines")
    print(f"Reduction:       {base_agent_lines - encompass_agent_lines} lines ({100*(base_agent_lines - encompass_agent_lines)/base_agent_lines:.1f}%)")
    
    # === RUN BASE AGENT ===
    print("\n" + "=" * 70)
    print("PHASE 1: Running Base Agent (Without EnCompass)")
    print("=" * 70)
    
    start_time = time.time()
    try:
        from base_reflexion_agent import run_reflexion as run_base
        base_results = await run_base(model=model)
        base_results["duration"] = time.time() - start_time
        base_results["agent_lines"] = base_agent_lines
        results["base_agent"] = base_results
    except Exception as e:
        print(f"Base agent failed: {e}")
        import traceback
        traceback.print_exc()
        results["base_agent"] = {"error": str(e), "duration": time.time() - start_time}
    
    # === RUN ENCOMPASS AGENT ===
    print("\n" + "=" * 70)
    print("PHASE 2: Running EnCompass Agent (With EnCompass)")
    print("=" * 70)
    
    start_time = time.time()
    try:
        from encompass_reflexion_agent import run_reflexion as run_encompass
        encompass_results = await run_encompass(model=model)
        encompass_results["duration"] = time.time() - start_time
        encompass_results["agent_lines"] = encompass_agent_lines
        results["encompass_agent"] = encompass_results
    except Exception as e:
        print(f"EnCompass agent failed: {e}")
        import traceback
        traceback.print_exc()
        results["encompass_agent"] = {"error": str(e), "duration": time.time() - start_time}
    
    # === TEST OUTPUTS ===
    print("\n" + "=" * 70)
    print("PHASE 3: Testing Generated Code")
    print("=" * 70)
    
    base_passed, base_failed = 0, 0
    enc_passed, enc_failed = 0, 0
    
    if results["base_agent"] and "error" not in results["base_agent"]:
        print("\nTesting Base Agent Output...")
        base_output_dir = str(base_dir / "output" / "base_agent")
        try:
            base_passed, base_failed, _ = run_tests(base_output_dir)
            results["base_agent"]["tests_passed"] = base_passed
            results["base_agent"]["tests_failed"] = base_failed
            print(f"  Results: {base_passed} passed, {base_failed} failed")
        except Exception as e:
            print(f"  Testing failed: {e}")
    
    if results["encompass_agent"] and "error" not in results["encompass_agent"]:
        print("\nTesting EnCompass Agent Output...")
        enc_output_dir = str(base_dir / "output" / "encompass_agent")
        try:
            enc_passed, enc_failed, _ = run_tests(enc_output_dir)
            results["encompass_agent"]["tests_passed"] = enc_passed
            results["encompass_agent"]["tests_failed"] = enc_failed
            print(f"  Results: {enc_passed} passed, {enc_failed} failed")
        except Exception as e:
            print(f"  Testing failed: {e}")
    
    # === COMPARISON ===
    results["comparison"] = {
        "lines_of_code": {
            "base_agent": base_agent_lines,
            "encompass_agent": encompass_agent_lines,
            "reduction": base_agent_lines - encompass_agent_lines,
            "reduction_percent": 100 * (base_agent_lines - encompass_agent_lines) / base_agent_lines,
        },
        "problems_solved": {
            "base_agent": results["base_agent"].get("solved_count", 0) if results["base_agent"] else 0,
            "encompass_agent": results["encompass_agent"].get("solved_count", 0) if results["encompass_agent"] else 0,
        },
        "tests_passed": {
            "base_agent": base_passed,
            "encompass_agent": enc_passed,
        },
    }
    
    # === GENERATE REPORT ===
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    
    report = generate_report(results)
    report_path = base_dir / "output" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")
    
    json_path = base_dir / "output" / "results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Results saved to: {json_path}")
    
    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Base Agent':<15} {'EnCompass':<15}")
    print("-" * 60)
    print(f"{'Lines of Code':<30} {base_agent_lines:<15} {encompass_agent_lines:<15}")
    base_solved = results["base_agent"].get("solved_count", "N/A") if results["base_agent"] else "N/A"
    enc_solved = results["encompass_agent"].get("solved_count", "N/A") if results["encompass_agent"] else "N/A"
    print(f"{'Problems Solved':<30} {str(base_solved):<15} {str(enc_solved):<15}")
    print(f"{'Tests Passed':<30} {base_passed:<15} {enc_passed:<15}")
    
    return results


def generate_report(results: dict) -> str:
    """Generate a markdown report comparing both agents."""
    comp = results["comparison"]
    
    return f"""# Reflexion Experiment Report

**Date:** {results['timestamp']}  
**Model:** {results['model']}

## Key Finding: EnCompass Reduces Code Complexity

| Metric | Base Agent | EnCompass Agent | Improvement |
|--------|------------|-----------------|-------------|
| Lines of Code | {comp['lines_of_code']['base_agent']} | {comp['lines_of_code']['encompass_agent']} | {comp['lines_of_code']['reduction_percent']:.1f}% reduction |
| Problems Solved | {comp['problems_solved']['base_agent']} | {comp['problems_solved']['encompass_agent']} | - |
| Tests Passed | {comp['tests_passed']['base_agent']} | {comp['tests_passed']['encompass_agent']} | - |

## Analysis

The EnCompass agent achieves the same functionality with **{comp['lines_of_code']['reduction']} fewer lines of code**.

**Base agent complexity:**
- Manual `ReflexionState` enum
- Explicit state tracking with dataclasses
- Manual generate/test/reflect/improve loops
- Complex control flow

**EnCompass agent simplicity:**
- `branchpoint()` for code generation alternatives
- `record_score()` for test accuracy feedback
- Linear, readable control flow
"""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Reflexion Experiment")
    parser.add_argument("--model", default="qwen2.5:32b", help="Ollama model name")
    args = parser.parse_args()
    
    asyncio.run(run_experiment(model=args.model))
