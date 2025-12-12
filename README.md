
# EnCompass

EnCompass is a Python-to-CPS (Continuation-Passing Style) compiler.

It transforms standard Python generator functions into resumable state machines. This allows the program state (local variables, instruction pointer) to be serialized, cloned, and restored.

By treating the program execution as a tree of states, we can apply search algorithms (Beam Search, MCTS) to guide LLM agents, rather than relying on a single greedy sample.

**Status**: 100% pass rate (5/5) on GSM8K mini-eval using `qwen2.5:32b` with Beam Search (width=8).

## Quickstart

Requires `ollama` running locally.

```bash
# 1. Install
pip install -r requirements.txt

# 2. Pull Model (Qwen 2.5 32B recommended for math/code)
ollama pull qwen2.5:32b

# 3. Run
python run_benchmark.py --benchmark gsm8k --strategy beam --real-llm --model qwen2.5:32b --width 8
```

## Mechanism

Standard LLM agents execute linearly:
`State_t -> LLM -> Action -> State_{t+1}`.
If the LLM errs, the trajectory fails.

EnCompass compiles the agent code:
```python
@encompass.compile
def agent():
    # 'yield' becomes a checkpoint.
    # The searcher can fork execution here 8 times (width=8).
    plan = yield branchpoint(options=["Plan A", "Plan B", ...])
    ...
```

Into a state machine:
```python
class AgentMachine:
    def run(self, state):
        if state.pc == 0:
            return BranchPoint(...)
        if state.pc == 1:
            # Restore variables, continue execution
            ...
```

This enables **O(1) relative branching**: forking a process is just copying the state object, avoiding re-computation of the history.

## Results

**GSM8K Mini-Eval (5 problems)**
- **Model**: `qwen2.5:32b`
- **Strategy**: Beam Search (k=8)
- **Accuracy**: 100% (5/5)
- **Baseline**: ~40% (Greedy/Beam k=3 with smaller models)

## Project Layout

- `encompass/core`: AST transformation logic (the compiler).
- `encompass/search`: Search implementations (Beam, MCTS, BFS).
- `encompass/llm`: Interfaces for OpenAI and Ollama.
- `benchmarks`: GSM8K evaluation scripts.

## License

MIT
