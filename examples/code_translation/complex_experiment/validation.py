"""Validation for translated code."""
from pathlib import Path

def valid(code):
    try: compile(code, '<s>', 'exec'); return bool(code)
    except: return False

def validate_dir(d):
    if not d.exists(): return {"error": "not found"}
    files = list(d.rglob("*.py"))
    v = sum(1 for f in files if valid(f.read_text()))
    return {"total": len(files), "valid": v}

if __name__ == "__main__":
    for n in ["vanilla", "encompass"]:
        r = validate_dir(Path(f"examples/code_translation/simple_experiment/output/{n}"))
        print(f"{n}: {r.get('valid', 0)}/{r.get('total', 0)} valid")
