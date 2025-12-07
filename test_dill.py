import dill
def gen():
    yield 1
    yield 2
    yield 3

g = gen()
print(f"First: {next(g)}") # 1

try:
    g2 = dill.copy(g)
    print(f"Cloned: {next(g2)}") # Should be 2
    print(f"Original: {next(g)}") # Should be 2
except Exception as e:
    print(f"Error: {e}")
