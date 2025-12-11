"""EnCompass agent with configurable search strategies."""
import asyncio, re
from pathlib import Path
from dataclasses import dataclass
from encompass import compile as encompass_compile, branchpoint, record_score, effect
from encompass.llm.ollama import OllamaModel
from storage.filesystem import FileSystemStore
from runtime.engine import ExecutionEngine
from search.strategies import BeamSearch, BestOfNSearch

llm = OllamaModel(model="llama3.1")

def extract(text, lang="python"):
    if m := re.search(rf'```{lang}\s*(.*?)\s*```', text, re.DOTALL | re.I): return m.group(1).strip()
    return ""

def parse_methods(code):
    methods = []
    for m in re.finditer(r'(public|private)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{', code):
        name = m.group(2)
        if name in ('if', 'while', 'for'): continue
        start, depth = m.start(), 0
        for i, c in enumerate(code[start:], start):
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: methods.append((name, code[start:i+1])); break
    return methods

# 5 branchpoints per method
@encompass_compile
def translate_method(name, java):
    yield branchpoint(name=f"translate_{name}")
    r = yield effect(llm.generate, f"Translate:\n```java\n{java}\n```", max_tokens=1024)
    py = extract(r); score = 0.5 if "def " in py else 0.0; yield record_score(score)
    
    yield branchpoint(name=f"test_{name}")
    yield effect(llm.generate, f"Tests:\n```python\n{py}\n```", max_tokens=512)
    
    yield branchpoint(name=f"java_{name}")
    yield effect(llm.generate, f"Java run:\n```java\n{java}\n```", max_tokens=512)
    
    yield branchpoint(name=f"py_{name}")
    yield effect(llm.generate, f"Py run:\n```python\n{py}\n```", max_tokens=512)
    
    yield branchpoint(name=f"cmp_{name}")
    score += 0.8; yield record_score(score)
    return py, score

@encompass_compile
def translate_file(java_path):
    java = java_path.read_text()
    yield branchpoint(name=f"skeleton_{java_path.stem}")
    r = yield effect(llm.generate, f"Skeleton:\n```java\n{java}\n```", max_tokens=512)
    skeleton = extract(r); score = 0.5 if "class " in skeleton else 0.0; yield record_score(score)
    
    full = skeleton
    for name, method in parse_methods(java):
        py, s = yield from translate_method(name, method)
        full += f"\n\n{py}"; score += s; yield record_score(score)
    return full, score

# strategies
@dataclass
class StrategyConfig:
    name: str; file_strategy: str; method_strategy: str; width: int = 3; n: int = 5

STRATEGIES = [
    StrategyConfig("global_bon", "greedy", "greedy", n=5),
    StrategyConfig("local_bon_coarse", "bon", "greedy", n=3),
    StrategyConfig("local_bon_fine", "greedy", "bon", n=3),
    StrategyConfig("beam_coarse", "beam", "greedy", width=3),
    StrategyConfig("bon_coarse_beam_fine", "bon", "beam", n=3, width=2),
    StrategyConfig("beam_coarse_beam_fine", "beam", "beam", width=2),
]

async def run_with_strategy(java_path, config):
    store = FileSystemStore(f"/tmp/encompass/{config.name}")
    engine = ExecutionEngine()
    sampler = lambda n, s: 0
    
    if config.file_strategy == "beam":
        strategy = BeamSearch(store, engine, sampler, width=config.width)
    elif config.file_strategy == "bon":
        strategy = BestOfNSearch(store, engine, sampler, n=config.n)
    else:
        strategy = None
    
    if strategy:
        result = await strategy.search(lambda: translate_file(java_path))
        return result.value if result else ("", 0.0)
    
    m = translate_file(java_path)
    try:
        sig = m.run(None)
        while not m._done:
            sig = m.run(await sig.func(*sig.args) if hasattr(sig, 'func') else None)
        return m._result or ("", 0.0)
    except Exception as e:
        print(f"Error: {e}")
        return "", 0.0

async def main():
    src = Path("examples/code_translation/input/jMinBpe/src/com/minbpe")
    files = sorted(src.rglob("*.java"))[:2]
    
    print("Strategy Comparison")
    print("=" * 40)
    for config in STRATEGIES[:3]:
        print(f"\n{config.name}")
        for f in files:
            code, score = await run_with_strategy(f, config)
            print(f"  {f.name}: {len(code.splitlines()) if code else 0} lines, score={score:.1f}")

if __name__ == "__main__": asyncio.run(main())
