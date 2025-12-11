"""Compare vanilla vs encompass."""
from pathlib import Path

def lines(p): return len([l for l in p.read_text().splitlines() if l.strip() and not l.strip().startswith('#')])

def valid(d):
    if not d.exists(): return 0, 0
    files = list(d.rglob("*.py"))
    v = sum(1 for f in files if (lambda c: not any(True for _ in [compile(c, f.name, 'exec')] if False))(f.read_text()) or True)
    v = 0
    for f in files:
        try: compile(f.read_text(), f.name, 'exec'); v += 1
        except: pass
    return len(files), v

base = Path("examples/code_translation/simple_experiment")
vl, el = lines(base/"baseline_vanilla.py"), lines(base/"encompass_version.py")
vf, vv = valid(base/"output/vanilla")
ef, ev = valid(base/"output/encompass")
print(f"Vanilla:   {vl} lines, {vv}/{vf} valid")
print(f"EnCompass: {el} lines, {ev}/{ef} valid")
print(f"Reduction: {(vl-el)*100//vl}%")
