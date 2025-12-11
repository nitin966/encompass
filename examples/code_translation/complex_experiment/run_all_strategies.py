"""Run all 6 strategies and compare."""
import asyncio
from pathlib import Path
from examples.code_translation.complex_experiment.encompass_agent import STRATEGIES, run_with_strategy

async def main():
    src = Path("examples/code_translation/input/jMinBpe/src/com/minbpe")
    files = sorted(src.rglob("*.java"))[:1]  # 1 file for speed
    
    print("Strategy Comparison")
    print("-" * 40)
    
    for config in STRATEGIES:
        print(f"\n{config.name} (file={config.file_strategy}, method={config.method_strategy})")
        for f in files:
            code, score = await run_with_strategy(f, config)
            lines = len(code.splitlines()) if code else 0
            print(f"  {f.name}: {lines} lines, score={score:.1f}")

if __name__ == "__main__": asyncio.run(main())
