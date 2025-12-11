"""Test tokenizers."""
import importlib.util

def test(path):
    spec = importlib.util.spec_from_file_location("tok", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    t = m.BasicTokenizer(); t.vocab = {i: bytes([i]) for i in range(256)}; t.merges = {}
    return t.decode(t.encode("hello")) == "hello"

for n, p in [("Vanilla", "simple_experiment/output/vanilla/BasicTokenizer.py"),
             ("EnCompass", "simple_experiment/output/encompass/BasicTokenizer.py")]:
    base = "examples/code_translation/"
    try: print(f"{n}: {'PASS' if test(base+p) else 'FAIL'}")
    except Exception as e: print(f"{n}: ERROR - {e}")
