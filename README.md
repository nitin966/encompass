
# OpenEnCompass

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![GitHub stars](https://img.shields.io/github/stars/nitin966/OpenEnCompass)](https://github.com/nitin966/OpenEnCompass/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/nitin966/OpenEnCompass)](https://github.com/nitin966/OpenEnCompass/network)
[![GitHub contributors](https://img.shields.io/github/contributors/nitin966/OpenEnCompass)](https://github.com/nitin966/OpenEnCompass/graphs/contributors)
[![GitHub issues](https://img.shields.io/github/issues/nitin966/OpenEnCompass)](https://github.com/nitin966/OpenEnCompass/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/nitin966/OpenEnCompass)](https://github.com/nitin966/OpenEnCompass/pulls)

OpenEnCompass is an implementation of Asari AI's Encompass paper, which is a Python-to-CPS (Continuation-Passing Style) compiler. 

It transforms standard Python generator functions into resumable state machines. This allows the program state (local variables, instruction pointer) to be serialized, cloned, and restored.

By treating the program execution as a tree of states, we can apply search algorithms (Beam Search, MCTS) to guide LLM agents, rather than relying on a single greedy sample.

This work is implemented with AI assistance (Antigravity) and correctness enforced via tests, deterministic replays and benchmarks.

**Status**: 73% accuracy on GSM8K (n=100) using `qwen2.5:32b` with Beam Search (width=8).

## Agent Complexity Comparison

OpenEnCompass significantly reduces the complexity of building sophisticated agents by abstracting away state management and search logic. The following table compares the lines of code (LOC) required for a manual state machine implementation vs. an OpenEnCompass implementation across different patterns:

| Pattern | Base Agent (LOC) | OpenEnCompass Agent (LOC) | **Reduction** |
| :--- | :--- | :--- | :--- |
| **Code Translation** (Java → Python) | 493 | 250 | **49.3%** |
| **Hypothesis Search** (ARC-style) | 421 | 199 | **52.7%** |
| **Reflexion** (Self-Correction) | 410 | 219 | **46.6%** |

On average, OpenEnCompass reduces agent code by **~50%** while maintaining identical functional performance.

## Mechanism

Standard LLM agents execute linearly:
`State_t -> LLM -> Action -> State_{t+1}`.
If the LLM errs, the trajectory fails.

OpenEnCompass compiles the agent code with **implicit yields** for cleaner syntax:
```python
@encompass.compile
def agent():
    # Function calls automatically checkpoint execution.
    # The searcher can fork execution here 8 times (width=8).
    plan = branchpoint(options=["Plan A", "Plan B", ...])
    
    # Actions are also implicitly yielding
    result = my_action(plan)
    
    # Record scores without explicit yield
    record_score(10)
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

## Search Strategies

OpenEnCompass decouples the *agent logic* (the generator) from the *search algorithm* (the driver).

**Beam Search**
Maintains the `k` most promising execution traces at each step.
```python
from encompass.search import BeamSearch

# Run the agent with beam width 8
strategy = BeamSearch(width=8)
results = await strategy.search(agent)
```

**Monte Carlo Tree Search (MCTS)**
Uses UCT (Upper Confidence Bound applied to Trees) to balance exploration and exploitation.
```python
from encompass.search import MCTS

# Run 100 simulations with exploration constant 1.4
strategy = MCTS(iterations=100, exploration=1.4)
results = await strategy.search(agent)
```

**Adaptive Branching MCTS (AB-MCTS)**
Implements the **TreeQuest** algorithm (Inoue et al., 2025). Uses Thompson Sampling with Bayesian posteriors (Normal-Inverse-Gamma) to adaptively decide between exploring wider or deeper.
```python
from encompass.search# Configure AB-MCTS with custom priors and sharing policy
config = ABMCTSConfig(
    iterations=50,
    score_type="gaussian",
    belief_sharing="pooled", # "independent" or "pooled"
    prior_mean=0.5,
    prior_kappa=1.0
)tegy = ABMCTS(config=config)
results = await strategy.search(agent)
```

**Best-First Search**
Prioritizes nodes based on a heuristic value function.
```python
from encompass.search import BestFirstSearch

strategy = BestFirstSearch(max_nodes=1000)
results = await strategy.search(agent)
```

## Implicit Yield Mechanism

OpenEnCompass features **automatic implicit yields** for cleaner, more intuitive code. Control signals (`branchpoint`, `record_score`), actions (via `@action`), and nested agents automatically checkpoint execution without explicit `yield` keywords.

**Traditional (still supported):**
```python
@compile
def agent():
    x = yield branchpoint("choice")
    yield record_score(10)
    result = yield my_action(x)
    return result
```

**Modern (implicit yields):**
```python
@compile
def agent():
    x = branchpoint("choice")
    record_score(10)
    result = my_action(x)
    return result
```

The compiler automatically detects `ControlSignal` returns and manages state transitions, making agent code cleaner and more readable while maintaining full checkpointing capabilities.

## Installation

Requires Python 3.10+ and `ollama` for local inference.

```bash
# 1. Clone and install dependencies
git clone https://github.com/nitin966/OpenEnCompass.git
cd OpenEnCompass
pip install -r requirements.txt

# 2. Install Ollama (for local LLMs)
# macOS
brew install ollama
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# 3. Pull the recommended model
ollama pull qwen2.5:32b
```

## Usage

### Running Comparison Experiments

You can run the comparison experiments to see the difference between base agents and OpenEnCompass agents:

```bash
# 1. Code Translation Experiment
python examples/code_translation/run_experiment.py --model qwen2.5:32b

# 2. Hypothesis Search Experiment
python examples/hypothesis_search/run_experiment.py --model qwen2.5:32b

# 3. Reflexion Agent Experiment
python examples/reflexion/run_experiment.py --model qwen2.5:32b
```

### Running Benchmarks

Reproduce the 100% accuracy result on GSM8K:

```bash
python run_benchmark.py --benchmark gsm8k --strategy beam --real-llm --model qwen2.5:32b --width 8
```

To run the **Full GSM8K Test Set** (1319 problems):
```bash
# Run all problems (warning: takes a long time)
python run_benchmark.py --benchmark gsm8k_full --strategy beam --real-llm --model qwen2.5:32b --width 8

# Run a subset (e.g., first 100)
python run_benchmark.py --benchmark gsm8k_full --strategy beam --real-llm --model qwen2.5:32b --width 8 --limit 100
```

### Other Benchmarks (ARC, Reflexion, LiveCodeBench)

```bash
# Reflexion (Code Generation + Self-Correction)
python run_benchmark.py --benchmark reflexion --strategy beam --real-llm --model qwen2.5:32b --width 3

# ARC (Hypothesis Search)
python run_benchmark.py --benchmark arc --strategy beam --real-llm --model qwen2.5:32b --width 3

# LiveCodeBench (TreeQuest / AB-MCTS)
# Evaluates code generation on partial update problems using AB-MCTS
python benchmarks/ab_mcts_benchmark.py --model qwen2.5:32b --iterations 10
```

### Deep Search Validation

Verify the O(1) state restoration and linear scaling up to depth 50+:

```bash
python validation/simple_deep_test.py
```

### Unit Tests

Run the comprehensive test suite (88 tests covering compiler, search, caching, and advanced control flow):

```bash
python -m unittest discover tests
```

## Compiler Capabilities

OpenEnCompass supports advanced Python features in agent code:
- **Control Flow**: `if/else`, `while`, `for`, `break`, `continue`, nested loops.
- **Exceptions**: `try/except`, `raise`, exception propagation.
- **Yield Expressions**: `x = yield branchpoint(...)`, `if (yield ...):`, `return (yield ...)`.
- **Imports**: `import module` works and persists across states.
- **State Management**: Automatic serialization of local variables (including large objects).

**Current Limitations**:
- Closures (nonlocal variables) are not yet supported.
- `try/finally` and `with` statements are not yet supported.
- Tuple unpacking assignment (`x, y = ...`) is not yet supported.

## Performance

- **O(1) Resumption**: Resuming a machine takes constant time regardless of history length.
- **Heavy State**: Efficiently handles large context objects (e.g., 1MB strings) with minimal overhead (~0.6ms per step), leveraging structural sharing where possible.


## Results

**GSM8K Benchmark**

| Configuration | Accuracy | Tasks | Nodes Explored | Duration |
|---------------|----------|-------|----------------|----------|
| Beam Search (k=8), qwen2.5:32b | **73%** | 100 | 870 | ~48 min |

Beam Search enables systematic exploration of reasoning paths, improving over single-sample greedy decoding by maintaining multiple candidate solutions at each step.

## Project Structure

```
encompass/
├── core/              # CPS compiler implementation (AST transformation)
├── runtime/           # Execution engine and cost tracking
├── search/            # Search strategy implementations (Beam, MCTS)
├── encompass/llm/     # LLM adapter implementations (Ollama, OpenAI)
├── benchmarks/        # Evaluation framework and datasets
├── validation/        # Deep search validation tests
├── tests/             # Unit and integration tests
└── examples/          # Reference implementations
    ├── code_translation/ # Java to Python translation experiment
    ├── hypothesis_search/# ARC-style pattern discovery experiment
    └── reflexion/        # Code generation with self-correction experiment
```

## Citation

```bibtex
@inproceedings{li2025encompass,
   title={{EnCompass}: Enhancing Agent Programming with Search Over Program Execution Paths},
   author={Li, Zhening and Solar-Lezama, Armando and Yue, Yisong and Zheng, Stephan},
   booktitle={Conference on Neural Information Processing Systems},
   year={2025}
 }
@software{openencompass2025,
  author = {Nitin Kesarwani},
  title = {OpenEnCompass: CPS Compiler for Search-Based LLM Agents},
  year = {2025},
  url = {https://github.com/nitin966/OpenEnCompass},
  note = {Validated on GSM8K benchmark with depth 100+ search support}
}
```

## License

MIT
