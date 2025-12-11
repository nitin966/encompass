"""Vanilla Java->Python translation using state machine."""
import asyncio, re
from pathlib import Path
from dataclasses import dataclass
from encompass.llm.ollama import OllamaModel

llm = OllamaModel(model="llama3.1")

def extract(text):
    if m := re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL | re.I): return m.group(1).strip()
    if m := re.search(r'```\s*(.*?)\s*```', text, re.DOTALL): return m.group(1).strip()
    return text.strip() if text.strip().startswith(('class ', 'def ', 'import ')) else ""

def valid(code): 
    try: compile(code, '<s>', 'exec'); return bool(code)
    except: return False

@dataclass
class State:
    java: str; name: str; step: str = "translate"; result: str = ""; tries: int = 0

class Machine:
    def __init__(self, java, name, max_tries=3):
        self.s = State(java=java, name=name)
        self.max = max_tries
    
    async def run(self):
        while True:
            if self.s.step == "translate":
                r = await llm.generate(f"Translate to Python:\n```java\n{self.s.java}\n```", max_tokens=2048)
                self.s.result = extract(r)
                self.s.step = "validate"
            elif self.s.step == "validate":
                if valid(self.s.result): return self.s.result
                self.s.tries += 1
                self.s.step = "translate" if self.s.tries < self.max else "done"
            else:
                return self.s.result if valid(self.s.result) else ""

async def main():
    src = Path("examples/code_translation/input/jMinBpe/src/com/minbpe")
    out = Path("examples/code_translation/simple_experiment/output/vanilla")
    out.exists() and __import__('shutil').rmtree(out); out.mkdir(parents=True, exist_ok=True)
    
    for f in sorted(src.rglob("*.java")):
        print(f"{f.name}...", end=" ", flush=True)
        code = await Machine(f.read_text(), f.stem).run()
        if code: (out / f"{f.stem}.py").write_text(code); print(f"✓ {len(code.splitlines())} lines")
        else: print("✗")

if __name__ == "__main__": asyncio.run(main())
