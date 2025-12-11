"""Vanilla agent with 5-step state machine."""
import asyncio, re
from pathlib import Path
from dataclasses import dataclass, field
from encompass.llm.ollama import OllamaModel

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

@dataclass
class MethodState:
    name: str; code: str; step: int = 0; py: str = ""; score: float = 0.0

@dataclass
class FileState:
    java: Path; step: int = 0; skeleton: str = ""; methods: list = field(default_factory=list)
    idx: int = 0; ms: MethodState = None; full: str = ""; score: float = 0.0

class Machine:
    def __init__(self, src, dst):
        self.src, self.dst = src, dst
        self.files = sorted(src.rglob("*.java"))
        self.idx, self.fs, self.score = 0, None, 0.0
    
    async def step(self):
        if self.fs is None:
            if self.idx >= len(self.files): return True
            self.fs = FileState(java=self.files[self.idx]); return False
        
        fs = self.fs
        if fs.step == 0:
            r = await llm.generate(f"Skeleton:\n```java\n{fs.java.read_text()}\n```", max_tokens=512)
            fs.skeleton = fs.full = extract(r)
            fs.methods = parse_methods(fs.java.read_text())
            fs.score = 0.5 if "class " in fs.skeleton else 0.0; fs.step = 1; return False
        
        if fs.ms is None and fs.idx < len(fs.methods):
            n, c = fs.methods[fs.idx]; fs.ms = MethodState(name=n, code=c); return False
        
        if fs.ms:
            ms = fs.ms
            if ms.step == 0:
                r = await llm.generate(f"Translate:\n```java\n{ms.code}\n```", max_tokens=1024)
                ms.py = extract(r); ms.score = 0.5 if "def " in ms.py else 0.0; ms.step = 1
            elif ms.step == 1: await llm.generate(f"Tests:\n```python\n{ms.py}\n```", max_tokens=512); ms.step = 2
            elif ms.step == 2: await llm.generate(f"Java:\n```java\n{ms.code}\n```", max_tokens=512); ms.step = 3
            elif ms.step == 3: await llm.generate(f"Py:\n```python\n{ms.py}\n```", max_tokens=512); ms.step = 4
            elif ms.step == 4: ms.score += 0.8; fs.full += f"\n\n{ms.py}"; fs.score += ms.score; fs.idx += 1; fs.ms = None
            return False
        
        (self.dst / f"{fs.java.stem}.py").parent.mkdir(parents=True, exist_ok=True)
        (self.dst / f"{fs.java.stem}.py").write_text(fs.full)
        self.score += fs.score; self.idx += 1; self.fs = None
        return self.idx >= len(self.files)
    
    async def run(self):
        while not await self.step(): pass
        return self.score

async def main():
    src = Path("examples/code_translation/input/jMinBpe/src/com/minbpe")
    dst = Path("examples/code_translation/complex_experiment/output/vanilla")
    dst.exists() and __import__('shutil').rmtree(dst); dst.mkdir(parents=True, exist_ok=True)
    print(f"Vanilla: {src} -> {dst}")
    score = await Machine(src, dst).run()
    print(f"Score: {score}")
    for f in sorted(dst.rglob("*.py")): print(f"  {f.name}: {len(f.read_text().splitlines())} lines")

if __name__ == "__main__": asyncio.run(main())
