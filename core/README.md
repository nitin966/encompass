# EnCompass Core Runtime System

The `core/` directory implements the compilation toolchain and runtime environment for the EnCompass framework.

This system addresses the fundamental limitation of standard Python generators in search-based AI, ie. the inability to efficiently fork and serialize execution state. To enable algorithms such as Monte Carlo Tree Search (MCTS) or Beam Search over arbitrary code, the runtime effectively implements a userspace virtual machine that reifies the Python stack into explicit, persistent data structures.

## 1. System Architecture

The core relies on a source-to-source transformation that converts imperative Python generator functions into a `Continuation-Passing Style (CPS)` state machine.

### 1.1 The Virtual Machine (`AgentMachine`)

The `AgentMachine` class (defined in `compiler.py`) serves as the execution substrate. It replaces the opaque CPython stack frame with an explicit object model designed for serialization and structural sharing.

* **Instruction Pointer (`_state`)**: An integer index representing the current basic block of execution. State `0` is the entry point.
* **Memory (`_ctx`)**: A `pyrsistent.pmap` (Persistent Hash Array Mapped Trie). This provides an immutable dictionary for local variables.
    * **Structural Sharing**: `machine.snapshot()` is an $O(1)$ operation. Forked machines share the same underlying memory nodes until a write occurs, minimizing memory pressure during wide tree searches.
* **Call Stack (`_stack`)**: A list managing nested agent invocations, manually implementing the semantics of `yield from`.

### 1.2 Serialization
Because the entire state is reified into standard Python primitives (int, list, pmap), the machine can be serialized via `dill`. This enables "suspend-to-disk" capabilities essential for long-running agentic workflows.

---

## 2. The CPS Compiler (`compiler.py`)

The `CPSCompiler` is an `ast.NodeTransformer` that linearizes hierarchical control flow into a flat graph of state transitions.

### 2.1 Compilation Strategy: Buffer and Flush

The compiler maintains a linear buffer of AST statements (`self.current_stmts`). Code generation follows a strict pattern:
1.  **Accumulate**: Visitor methods append standard AST nodes to the buffer.
2.  **Flush**: Upon encountering a control flow boundary (Yield, If, While, Return), the current buffer is flushed.
    * The buffer is assigned a **State ID**.
    * The buffer is cleared, and `self.current_state` increments.
    * Transitions are emitted as assignments to `self._state`.

### 2.2 Variable Rewriting

To enable persistence, the compiler intercepts all variable access:
* **Store**: `x = val` $\rightarrow$ `self._ctx = self._ctx.set('x', val)`
* **Load**: `x` $\rightarrow$ `self._ctx['x']`

This ensures that the `_ctx` map remains the single source of truth for the local scope.

### 2.3 Control Flow Flattening

The compiler transforms structured programming constructs into a Control Flow Graph (CFG) using explicit jumps.

#### Conditionals (`visit_If`)
Implemented as a split-merge graph.
1.  **Placeholders**: The compiler emits temporary negative integers (`THEN_PH`, `ELSE_PH`, `JOIN_PH`) for jump targets that do not yet exist.
2.  **Backpatching**: After recursively visiting the `body` and `orelse` blocks, the compiler walks the generated nodes and updates the `_state` assignments to point to the resolved State IDs.

#### Loops (`visit_While`, `visit_For`)
Loops are compiled into cyclic graphs with three distinct regions:
* **HEAD**: Evaluates the condition. Jumps to `BODY` or `AFTER`.
* **BODY**: The loop content. The final statement of this block is an unconditional jump back to `HEAD`.
* **AFTER**: The continuation point following the loop.

A `loop_stack` tracks the `HEAD` and `AFTER` IDs to correctly resolve `break` and `continue` statements during deep traversal.

### 2.4 Exception Handling Implementation

Because a single Python `try` block may be fragmented into dozens of compiled states, the `try/except` context must be preserved across state boundaries.

**Mechanism:**
1.  **Try Stack**: The compiler maintains a stack of active exception handlers.
2.  **State Wrapping**: During `_flush_state()`, if the `try_stack` is non-empty, the flushed block is wrapped in an `ast.Try` node.
3.  **Handler Routing**: The generated `except` handlers do not contain user logic. Instead, they:
    * Capture the exception.
    * Set `self._state` to the ID of the compiled user handler code.
    * Return immediately to the run loop (trampoline).

This ensures that if an exception occurs in *any* fragmented state corresponding to the user's `try` block, execution correctly transitions to the handler state.

### 2.5 Implicit Continuations (`visit_Call`)

To improve developer ergonomics, the compiler supports "Implicit Yields." When visiting an `ast.Call`:
1.  The return value is assigned to a temporary.
2.  An `isinstance(val, ControlSignal)` check is injected.
3.  If true, the machine automatically yields the signal and suspends, treating the function call as a suspension point.

---

## 3. Signal Protocol (`signals.py`)

The system enforces a strict boundary between Agent Logic (User Space) and Execution Engine (Kernel Space) via `ControlSignal` objects.

* **`BranchPoint`**: A request for non-deterministic input. The search engine (Kernel) uses a sampler or policy to select a value and resumes the machine.
* **`Effect`**: A request for side-effect execution. The Kernel executes the payload and caches the result.
* **`ScoreSignal`**: Emits a scalar reward for the current trajectory.

---

## 4. Determinism and Safety (`safety.py`)

Tree search algorithms require re-executing prefixes of the agent's path (Replay). To ensure correctness, side effects must be idempotent within the context of a trace.

* **`@idempotent`**: A decorator that enforces distinct-execution caching.
* **Hashing**: Arguments are hashed via MD5 (`_compute_args_hash`) to generate a cache key.
* **Replay**: If a function is called with identical arguments during a replay phase, the cached result is returned immediately, bypassing the actual execution.

---

## 5. Limitations

The compiler implements a strict subset of Python semantics to guarantee serializability:

* **Closure Capture**: Unsupported. The `AgentMachine` is a class, not a closure. All dependencies must be passed via `__init__`.
* **Async/Await**: Unsupported. The runtime is synchronous.
* **Tuple Unpacking**: Supported only for named variables (`a, b = x`). Anonymous unpacking (`_, b = x`) is rejected due to the requirement for explicit variable tracking in `_ctx`.
