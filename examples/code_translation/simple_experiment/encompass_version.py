"""EnCompass Java->Python translation using @compile."""
import asyncio, re
from pathlib import Path
from encompass import compile as encompass_compile, branchpoint, record_score, effect
from encompass.llm.ollama import OllamaModel

llm = OllamaModel(model="llama3.1")

def extract(text):
    if m := re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL | re.I): return m.group(1).strip()
    if m := re.search(r'```\s*(.*?)\s*```', text, re.DOTALL): return m.group(1).strip()
    return text.strip() if text.strip().startswith(('class ', 'def ', 'import ')) else ""

def valid(code):
    try: compile(code, '<s>', 'exec'); return bool(code)
    except: return False

@encompass_compile
def translate(java, name):
    yield branchpoint(name=f"translate-{name}")
    r = yield effect(llm.generate, f"Translate to Python:\n```java\n{java}\n```", max_tokens=2048)
    code = extract(r)
    yield record_score(1.0 if valid(code) else 0.0)
    return code

async def run(java, name, retries=3):
    for _ in range(retries):
        m = translate(java, name)
        try:
            sig = m.run(None)
            while not m._done:
                sig = m.run(await sig.func(*sig.args) if hasattr(sig, 'func') else None)
            if valid(m._result or ""): return m._result
        except: pass
    return ""

async def main():
    src = Path("examples/code_translation/input/jMinBpe/src/com/minbpe")
    out = Path("examples/code_translation/simple_experiment/output/encompass")
    out.exists() and __import__('shutil').rmtree(out); out.mkdir(parents=True, exist_ok=True)
    
    for f in sorted(src.rglob("*.java")):
        print(f"{f.name}...", end=" ", flush=True)
        code = await run(f.read_text(), f.stem)
        if code: (out / f"{f.stem}.py").write_text(code); print(f"✓ {len(code.splitlines())} lines")
        else: print("✗")

if __name__ == "__main__": asyncio.run(main())
