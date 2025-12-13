"""
CPS Compiler - Transforms generator functions into resumable state machines.

This module converts Python generator functions into continuation-passing style (CPS)
state machines that can be:
- Pickled and resumed from any yield point in O(1) time
- Branched to explore multiple execution paths
- Replayed efficiently for tree search and MCTS

Example:
    @compile
    def agent():
        x = yield branchpoint("choice")
        return x + 1

    # Creates an AgentMachine subclass that can be:
    machine = agent()
    sig = machine.run(None)  # Returns branchpoint signal
    sig = machine.run(5)      # Consumes input, returns result
    state = machine.save()    # Serialize to bytes
    machine.load(state)       # Restore from bytes

Key Limitations:
    - No closure variable capture (pass as arguments instead)
    - No tuple unpacking with _ (use named variables)
    - No await or yield from (not yet supported)

See docs/CPS_LIMITATIONS.md for details and workarounds.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable

import dill
from pyrsistent import pmap

from core.signals import ControlSignal


class AgentMachine(ControlSignal):
    """
    Base class for compiled state machines.

    All compiled agents inherit from this class, which provides:
    - State management (_state, _ctx, _done, _result)
    - Save/load for checkpointing
    - Stack for nested agent execution
    """

    def __init__(self):
        self._state = 0  # Current state number (which yield point)
        self._ctx = pmap()  # Local variables (Immutable Persistent Map)
        self._done = False  # Whether execution completed
        self._result = None  # Final return value
        self._stack = []  # Call stack for nested agents

        # Exception handling state
        self._exception = None  # Current exception being handled
        self._exception_type = None  # Type of current exception
        self._in_handler = False  # Whether we're in an except handler

    def run(self, _input=None):
        """
        Execute one step of the machine.

        Args:
            _input: Value to send to the current yield point

        Returns:
            Signal object (BranchPoint, ScoreSignal, etc.) or None if done
        """
        raise NotImplementedError

    def save(self):
        """
        Serialize machine state to bytes.

        Returns:
            bytes: Pickled state that can be loaded later
        """
        return dill.dumps(self)

    def snapshot(self):
        """
        Create a fast in-memory snapshot of the machine state.
        
        Returns:
            AgentMachine: A copy of the machine with shared immutable context.
        """
        import copy
        # Shallow copy the machine itself
        snap = copy.copy(self)
        # Since _ctx is immutable (pmap), we can safely share it!
        # Any update will create a new pmap, leaving this one untouched.
        snap._ctx = self._ctx
        snap._stack = self._stack[:]
        return snap

    def load(self, state):
        """
        Restore machine state.

        Args:
            state: bytes, dict (legacy), or AgentMachine instance to restore from
        """
        if isinstance(state, bytes):
            loaded = dill.loads(state)
            self.__dict__.update(loaded.__dict__)
        elif isinstance(state, AgentMachine):
            # When loading from a snapshot (which is already a valid machine state),
            # we want to adopt its state.
            
            # 1. Copy scalar fields (state, done, result, etc.)
            self.__dict__.update(state.__dict__)
            
            # 2. Context is immutable, so we can just point to it.
            # Future updates will replace self._ctx with a new pmap.
            self._ctx = state._ctx
            self._stack = state._stack[:]
        else:
            # Legacy dict support
            self._state = state["_state"]
            self._ctx = pmap(state["_ctx"]) # Convert dict to pmap
            self._done = state["_done"]
            self._result = state["_result"]
            self._stack = state.get("_stack", [])


class CPSCompiler(ast.NodeTransformer):
    """
    AST transformer that compiles generator functions to state machines.

    Transforms each yield statement into:
    1. Save current state
    2. Return the yielded value
    3. On next run(), resume from saved state

    Control flow (if/for/while) is handled by emitting explicit state transitions.
    """

    def __init__(self, varnames):
        self.states = {}  # state_id -> list of statements
        self.current_state = 0
        self.current_stmts = []
        self.varnames = set(varnames)
        self.loop_stack = []  # List of dicts: {'head': int, 'after': int (placeholder)}
        self.placeholder_counter = -100
        self.try_stack = []  # List of dicts: {'handlers': [], 'placeholders': []}

    def _flush_state(self, next_state=None):
        # If next_state is provided, we are transitioning.
        if next_state is not None:
            # Find the return statement
            if self.current_stmts and isinstance(self.current_stmts[-1], ast.Return):
                ret = self.current_stmts.pop()
                # Add transition
                self.current_stmts.append(ast.parse(f"self._state = {next_state}").body[0])
                # Add return back
                self.current_stmts.append(ret)
            else:
                # Just append
                self.current_stmts.append(ast.parse(f"self._state = {next_state}").body[0])

        # Wrap in try/except if needed
        if self.try_stack and self.current_stmts:
            # Wrap from inside out (top of stack is innermost)
            for context in reversed(self.try_stack):
                handlers = []
                for i, orig_handler in enumerate(context["handlers"]):
                    target_ph = context["placeholders"][i]

                    # Create handler body
                    h_body = []

                    # Capture exception if named
                    if orig_handler.name:
                        h_body.append(
                            ast.Assign(
                                targets=[
                                    ast.Attribute(
                                        value=ast.Name(id="self", ctx=ast.Load()),
                                        attr="_ctx",
                                        ctx=ast.Store(),
                                    )
                                ],
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Attribute(
                                            value=ast.Name(id="self", ctx=ast.Load()),
                                            attr="_ctx",
                                            ctx=ast.Load(),
                                        ),
                                        attr="set",
                                        ctx=ast.Load(),
                                    ),
                                    args=[
                                        ast.Constant(value=orig_handler.name),
                                        ast.Name(id=orig_handler.name, ctx=ast.Load())
                                    ],
                                    keywords=[],
                                )
                            )
                        )

                    # Transition to handler state
                    h_body.append(
                        ast.Assign(
                            targets=[
                                ast.Attribute(
                                    value=ast.Name(id="self", ctx=ast.Load()),
                                    attr="_state",
                                    ctx=ast.Store(),
                                )
                            ],
                            value=ast.Constant(value=target_ph),
                        )
                    )

                    # Return run()
                    h_body.append(
                        ast.Return(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="self", ctx=ast.Load()),
                                    attr="run",
                                    ctx=ast.Load(),
                                ),
                                args=[ast.Constant(value=None)],
                                keywords=[],
                            )
                        )
                    )

                    handlers.append(
                        ast.ExceptHandler(
                            type=orig_handler.type, name=orig_handler.name, body=h_body
                        )
                    )

                # Create Try node
                self.current_stmts = [
                    ast.Try(body=self.current_stmts, handlers=handlers, orelse=[], finalbody=[])
                ]

        self.states[self.current_state] = self.current_stmts
        self.current_state = next_state if next_state is not None else self.current_state + 1
        self.current_stmts = []

    def visit_Assign(self, node):
        # Rewrite assignments to use self._ctx.set()
        # x = ... -> self._ctx = self._ctx.set('x', ...)
        
        # Evaluate value (handles yields recursively)
        value = self.visit(node.value)
        
        # Use a temp variable to hold the value (handles multiple targets safely)
        temp_name = f"_assign_tmp_{self.current_state}_{len(self.current_stmts)}"
        self.current_stmts.append(
            ast.Assign(targets=[ast.Name(id=temp_name, ctx=ast.Store())], value=value)
        )
        temp_node = ast.Name(id=temp_name, ctx=ast.Load())
        
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in self.varnames:
                # self._ctx = self._ctx.set('x', temp)
                self.current_stmts.append(
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_ctx",
                                ctx=ast.Store()
                            )
                        ],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id="self", ctx=ast.Load()),
                                    attr="_ctx",
                                    ctx=ast.Load()
                                ),
                                attr="set",
                                ctx=ast.Load()
                            ),
                            args=[ast.Constant(value=t.id), temp_node],
                            keywords=[]
                        )
                    )
                )
            else:
                # Normal assignment: t = temp
                self.current_stmts.append(ast.Assign(targets=[t], value=temp_node))

    def visit_Global(self, node):
        # Preserve global declarations
        self.current_stmts.append(node)

    def visit_Nonlocal(self, node):
        # Preserve nonlocal declarations
        self.current_stmts.append(node)

    def visit_AugAssign(self, node):
        # Handle augmented assignments like x += 1
        # If x is a local variable (in varnames), rewrite to self._ctx = self._ctx.set('x', self._ctx['x'] + value)
        # Otherwise, keep as is (for globals)
        
        # Evaluate value (handles yields recursively)
        value = self.visit(node.value)

        if isinstance(node.target, ast.Name) and node.target.id in self.varnames:
            # Rewrite to self._ctx = self._ctx.set('x', self._ctx['x'] op value)
            
            # 1. Read current value: self._ctx['x']
            current_val = ast.Subscript(
                value=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr="_ctx", ctx=ast.Load()
                ),
                slice=ast.Constant(value=node.target.id),
                ctx=ast.Load(),
            )
            
            # 2. Compute new value: current op value
            new_val = ast.BinOp(left=current_val, op=node.op, right=value)
            
            # 3. Update context: self._ctx = self._ctx.set('x', new_val)
            self.current_stmts.append(
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_ctx",
                            ctx=ast.Store()
                        )
                    ],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_ctx",
                                ctx=ast.Load()
                            ),
                            attr="set",
                            ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=node.target.id), new_val],
                        keywords=[]
                    )
                )
            )
        else:
            # Keep as is (for globals or attributes)
            self.current_stmts.append(ast.AugAssign(target=node.target, op=node.op, value=value))

    def visit_Yield(self, node):
        # Handle yield expression
        # 1. Evaluate value to yield
        yield_val = self.visit(node.value) if node.value else ast.Constant(value=None)
        
        # 2. Set next state
        next_state = self.current_state + 1
        self.current_stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="_state", ctx=ast.Store()
                    )
                ],
                value=ast.Constant(value=next_state),
            )
        )
        
        # 3. Return yielded value
        self.current_stmts.append(ast.Return(value=yield_val))
        
        # 4. Flush state (start next block)
        self._flush_state(next_state)
        
        # 5. Return expression for received value (_input)
        return ast.Name(id="_input", ctx=ast.Load())

    def visit_Call(self, node):
        # Implicit Yield Logic
        # 1. Emit call and assign to temp
        # 2. Check if signal
        # 3. If signal, yield (return)
        # 4. If not, continue

        # We need to visit args first!
        # But generic_visit doesn't work for Call because we need to restructure.
        # We must manually visit args and keywords to ensure they are processed (e.g. nested calls)

        new_args = [self.visit(arg) for arg in node.args]
        new_keywords = [ast.keyword(arg=k.arg, value=self.visit(k.value)) for k in node.keywords]
        new_func = self.visit(node.func)

        # Reconstruct call with visited args
        call_node = ast.Call(func=new_func, args=new_args, keywords=new_keywords)

        # Generate temp variable name
        temp_name = f"_tmp_{self.current_state}_{len(self.current_stmts)}"

        # 1. Assign result to temp
        self.current_stmts.append(
            ast.Assign(targets=[ast.Name(id=temp_name, ctx=ast.Store())], value=call_node)
        )

        # 2. Import ControlSignal (if not already imported? We can just assume it's available or import it locally)
        # Better to import it once at module level, but we are inside a function.
        # We'll import it inside the check to be safe and local.
        self.current_stmts.append(
            ast.ImportFrom(
                module="core.signals", names=[ast.alias(name="ControlSignal", asname=None)], level=0
            )
        )

        # 3. Check and Split
        next_state = self.current_state + 1

        # if isinstance(temp, ControlSignal):
        if_stmt = ast.If(
            test=ast.Call(
                func=ast.Name(id="isinstance", ctx=ast.Load()),
                args=[
                    ast.Name(id=temp_name, ctx=ast.Load()),
                    ast.Name(id="ControlSignal", ctx=ast.Load()),
                ],
                keywords=[],
            ),
            body=[
                # self._state = next_state
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_state",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=next_state),
                ),
                # return temp
                ast.Return(value=ast.Name(id=temp_name, ctx=ast.Load())),
            ],
            orelse=[
                # Not a signal.
                # Save temp to _ctx so next state can retrieve it (simulating continuation)
                # self._ctx = self._ctx.set('_saved_temp', temp)
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_ctx",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_ctx",
                                ctx=ast.Load(),
                            ),
                            attr="set",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Constant(value=f"_saved_{temp_name}"),
                            ast.Name(id=temp_name, ctx=ast.Load())
                        ],
                        keywords=[],
                    )
                ),
                # self._state = next_state
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_state",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=next_state),
                ),
                # continue (loop in run)
                ast.Continue(),
            ],
        )
        self.current_stmts.append(if_stmt)

        self._flush_state(next_state)

        # In NEXT STATE:
        # Retrieve value
        # if '_saved_temp' in self._ctx:
        #     val = self._ctx.pop('_saved_temp')
        # else:
        #     val = _input

        val_name = f"_val_{temp_name}"
        saved_key = f"_saved_{temp_name}"

        check_saved = ast.If(
            test=ast.Compare(
                left=ast.Constant(value=saved_key),
                ops=[ast.In()],
                comparators=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="_ctx", ctx=ast.Load()
                    )
                ],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id=val_name, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_ctx",
                            ctx=ast.Load(),
                        ),
                        slice=ast.Constant(value=saved_key),
                        ctx=ast.Load(),
                    )
                ),
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_ctx",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_ctx",
                                ctx=ast.Load(),
                            ),
                            attr="discard",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Constant(value=saved_key)],
                        keywords=[],
                    )
                )
            ],
            orelse=[
                ast.Assign(
                    targets=[ast.Name(id=val_name, ctx=ast.Store())],
                    value=ast.Name(id="_input", ctx=ast.Load()),
                )
            ],
        )
        self.current_stmts.append(check_saved)

        # Return the variable name node
        return ast.Name(id=val_name, ctx=ast.Load())

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Yield):
            yield_val = self.visit(node.value.value) if node.value.value else None
            self.current_stmts.append(ast.Return(value=yield_val))

            next_state = self.current_state + 1
            self._flush_state(next_state)
        else:
            res = self.generic_visit(node)
            if res is None:
                print(f"WARNING: generic_visit returned None for {node} with value {node.value}")
            else:
                self.current_stmts.append(res)

    def visit_If(self, node):
        # 1. Visit test
        test = self.visit(node.test)

        # 2. Emit conditional jump with placeholders

        # Placeholder values
        THEN_PH = -1
        ELSE_PH = -2
        JOIN_PH = -3

        # Construct the If statement for the state machine
        if_stmt = ast.If(
            test=test,
            body=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_state",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=THEN_PH),
                ),
                ast.Return(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=None)],
                        keywords=[],
                    )
                ),
            ],
            orelse=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_state",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=ELSE_PH),
                ),
                ast.Return(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=None)],
                        keywords=[],
                    )
                ),
            ],
        )
        self.current_stmts.append(if_stmt)
        self._flush_state(None)  # Flush entry block

        # 3. Visit THEN block
        then_start = self.current_state
        for stmt in node.body:
            self.visit(stmt)

        # Check if we need to jump to join
        if not (self.current_stmts and isinstance(self.current_stmts[-1], ast.Return)):
            self.current_stmts.append(
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_state",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=JOIN_PH),
                )
            )
            self.current_stmts.append(
                ast.Return(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=None)],
                        keywords=[],
                    )
                )
            )

        then_end = self.current_state
        self._flush_state(None)

        # 4. Visit ELSE block
        else_start = self.current_state
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

            if not (self.current_stmts and isinstance(self.current_stmts[-1], ast.Return)):
                self.current_stmts.append(
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_state",
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Constant(value=JOIN_PH),
                    )
                )
                self.current_stmts.append(
                    ast.Return(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="run",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Constant(value=None)],
                            keywords=[],
                        )
                    )
                )
        else:
            # Empty else means jump directly to join
            # We can handle this by setting else_start = join_start later?
            # Or we can emit a jump block?
            # Better: Set else_start to JOIN_PH (resolved later)
            pass

        else_end = self.current_state
        self._flush_state(None)

        # 5. Join State
        join_start = self.current_state

        # 6. Backpatching
        # Fix Entry
        # Use if_stmt directly (it's the same object)

        # Fix THEN
        if_stmt.body[0].value.value = then_start
        # Fix ELSE
        if node.orelse:
            if_stmt.orelse[0].value.value = else_start
        else:
            if_stmt.orelse[0].value.value = join_start

        # Fix THEN End (if it has placeholder)
        def patch_jump(state_id, target):
            stmts = self.states[state_id]
            if not stmts:
                return
            # Look for assignment to _state with placeholder
            for stmt in stmts:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Assign) and len(node.targets) == 1:
                        t = node.targets[0]
                        if isinstance(t, ast.Attribute) and t.attr == "_state":
                            if isinstance(node.value, ast.Constant) and node.value.value == JOIN_PH:
                                node.value.value = target

        patch_jump(then_end, join_start)
        patch_jump(else_end, join_start)

    def visit_Try(self, node):
        # 1. Setup Placeholders for handlers
        handler_placeholders = []
        for _ in node.handlers:
            handler_placeholders.append(self.placeholder_counter)
            self.placeholder_counter -= 1

        JOIN_PH = self.placeholder_counter
        self.placeholder_counter -= 1

        # 2. Push context
        self.try_stack.append({"handlers": node.handlers, "placeholders": handler_placeholders})

        # 3. Visit Body
        # Note: _flush_state will automatically wrap generated states with try/except
        # that jumps to the handler placeholders.
        for stmt in node.body:
            self.visit(stmt)

        # Flush body state while stack is still active to ensure wrapping
        self._flush_state(None)

        # 4. Pop context
        self.try_stack.pop()

        # 5. Handle Orelse (executed if no exception)
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

        # 6. Jump to JOIN
        # If we fall through body/orelse, we go to JOIN
        if not (self.current_stmts and isinstance(self.current_stmts[-1], ast.Return)):
            self.current_stmts.append(
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_state",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=JOIN_PH),
                )
            )
            self.current_stmts.append(
                ast.Return(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=None)],
                        keywords=[],
                    )
                )
            )

        self._flush_state(None)

        # 7. Compile Handlers
        handler_starts = []
        for _i, handler in enumerate(node.handlers):
            start_state = self.current_state
            handler_starts.append(start_state)

            # Visit handler body
            for stmt in handler.body:
                self.visit(stmt)

            # Jump to JOIN
            if not (self.current_stmts and isinstance(self.current_stmts[-1], ast.Return)):
                self.current_stmts.append(
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_state",
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Constant(value=JOIN_PH),
                    )
                )
                self.current_stmts.append(
                    ast.Return(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="run",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Constant(value=None)],
                            keywords=[],
                        )
                    )
                )

            self._flush_state(None)

        # 8. JOIN State
        join_start = self.current_state

        # 9. Backpatching
        def patch_placeholders(ph, target):
            for _sid, stmts in self.states.items():
                if not stmts:
                    continue
                # We need to look DEEP into the statements because they might be wrapped in Try/Except
                for stmt in stmts:
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Assign) and len(node.targets) == 1:
                            t = node.targets[0]
                            if isinstance(t, ast.Attribute) and t.attr == "_state":
                                if isinstance(node.value, ast.Constant) and node.value.value == ph:
                                    node.value.value = target

        # Patch JOIN
        patch_placeholders(JOIN_PH, join_start)

        # Patch Handlers
        for i, ph in enumerate(handler_placeholders):
            patch_placeholders(ph, handler_starts[i])

    def visit_While(self, node):
        # 1. Emit jump to HEAD
        head_start = self.current_state + 1
        self.current_stmts.append(
            ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=None)],
                    keywords=[],
                )
            )
        )
        self._flush_state(head_start)

        # 2. HEAD State: Test
        test = self.visit(node.test)

        BODY_PH = self.placeholder_counter
        self.placeholder_counter -= 1
        AFTER_PH = self.placeholder_counter
        self.placeholder_counter -= 1

        self.loop_stack.append({"head": head_start, "after": AFTER_PH})

        # if test: goto BODY; else: goto AFTER
        if_stmt = ast.If(
            test=test,
            body=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_state",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=BODY_PH),
                ),
                ast.Return(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=None)],
                        keywords=[],
                    )
                ),
            ],
            orelse=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_state",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=AFTER_PH),
                ),
                ast.Return(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=None)],
                        keywords=[],
                    )
                ),
            ],
        )
        self.current_stmts.append(if_stmt)
        self._flush_state(None)  # Flush HEAD

        # 3. BODY State
        body_start = self.current_state
        for stmt in node.body:
            self.visit(stmt)

        # Jump back to HEAD
        # Always add this jump, even if there was a yield (which creates a Return in an earlier state)
        # The jump needs to be in the CURRENT state (after all yields have been processed)
        self.current_stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="_state", ctx=ast.Store()
                    )
                ],
                value=ast.Constant(value=head_start),
            )
        )
        self.current_stmts.append(
            ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=None)],
                    keywords=[],
                )
            )
        )

        self._flush_state(None)

        self.loop_stack.pop()

        # 4. AFTER State
        after_start = self.current_state

        # 5. Backpatching
        # Fix HEAD
        # Use if_stmt directly
        if_stmt.body[0].value.value = body_start
        if_stmt.orelse[0].value.value = after_start

        # Global backpatch for breaks/after
        def patch_placeholders(ph, target):
            for _sid, stmts in self.states.items():
                if not stmts:
                    continue
                for stmt in stmts:
                    if stmt is None:
                        continue
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Assign) and len(node.targets) == 1:
                            t = node.targets[0]
                            if isinstance(t, ast.Attribute) and t.attr == "_state":
                                if isinstance(node.value, ast.Constant) and node.value.value == ph:
                                    node.value.value = target

        patch_placeholders(AFTER_PH, after_start)

    def visit_For(self, node):
        # 1. Init Iterator
        iter_name = f"_iter_{self.current_state}"
        iter_expr = self.visit(node.iter)

        # self._ctx[iter_name] = iter(expr)
        # self._ctx[iter_name] = iter(expr)
        # self._ctx = self._ctx.set(iter_name, iter(expr))
        self.current_stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="_ctx",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_ctx",
                            ctx=ast.Load(),
                        ),
                        attr="set",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=iter_name),
                        ast.Call(
                            func=ast.Name(id="iter", ctx=ast.Load()), args=[iter_expr], keywords=[]
                        )
                    ],
                    keywords=[],
                )
            )
        )

        head_start = self.current_state + 1
        self.current_stmts.append(
            ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=None)],
                    keywords=[],
                )
            )
        )
        self._flush_state(head_start)

        # 2. HEAD State: Next
        BODY_PH = self.placeholder_counter
        self.placeholder_counter -= 1
        AFTER_PH = self.placeholder_counter
        self.placeholder_counter -= 1

        self.loop_stack.append({"head": head_start, "after": AFTER_PH})

        val_name = f"_val_{head_start}"

        try_body = [
            ast.Assign(
                targets=[ast.Name(id=val_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="next", ctx=ast.Load()),
                    args=[
                        ast.Subscript(
                            value=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_ctx",
                                ctx=ast.Load(),
                            ),
                            slice=ast.Constant(value=iter_name),
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[],
                ),
            )
        ]

        # Assign to target
        saved_stmts = self.current_stmts
        self.current_stmts = []

        # Create Assign node: target = val
        assign_node = ast.Assign(targets=[node.target], value=ast.Name(id=val_name, ctx=ast.Load()))
        self.visit_Assign(assign_node)

        assign_stmts = self.current_stmts
        self.current_stmts = saved_stmts

        try_body.extend(assign_stmts)

        # Jump to BODY
        try_body.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="_state", ctx=ast.Store()
                    )
                ],
                value=ast.Constant(value=BODY_PH),
            )
        )
        try_body.append(
            ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=None)],
                    keywords=[],
                )
            )
        )

        # Handler
        handler = ast.ExceptHandler(
            type=ast.Name(id="StopIteration", ctx=ast.Load()),
            name=None,
            body=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_state",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=AFTER_PH),
                ),
                ast.Return(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=None)],
                        keywords=[],
                    )
                ),
            ],
        )

        try_stmt = ast.Try(body=try_body, handlers=[handler], orelse=[], finalbody=[])
        self.current_stmts.append(try_stmt)

        self._flush_state(None)  # Flush HEAD

        # 3. BODY State
        body_start = self.current_state
        for stmt in node.body:
            self.visit(stmt)

        # Jump back to HEAD
        if not (self.current_stmts and isinstance(self.current_stmts[-1], ast.Return)):
            self.current_stmts.append(
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_state",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=head_start),
                )
            )
            self.current_stmts.append(
                ast.Return(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=None)],
                        keywords=[],
                    )
                )
            )

        self._flush_state(None)

        self.loop_stack.pop()

        # 4. AFTER State
        after_start = self.current_state

        # 5. Backpatching
        # Fix HEAD (BODY_PH)
        # Use try_stmt directly
        for stmt in try_stmt.body:
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value == BODY_PH
            ):
                stmt.value.value = body_start

        # Fix AFTER_PH (in handler)
        for handler in try_stmt.handlers:
            for stmt in handler.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value == AFTER_PH
                ):
                    stmt.value.value = after_start

        # Global backpatch
        def patch_placeholders(ph, target):
            for _sid, stmts in self.states.items():
                if not stmts:
                    continue
                for stmt in stmts:
                    if stmt is None:
                        continue
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Assign) and len(node.targets) == 1:
                            t = node.targets[0]
                            if isinstance(t, ast.Attribute) and t.attr == "_state":
                                if isinstance(node.value, ast.Constant) and node.value.value == ph:
                                    node.value.value = target

        patch_placeholders(AFTER_PH, after_start)

    def visit_Break(self, node):
        if not self.loop_stack:
            return
        after_ph = self.loop_stack[-1]["after"]
        self.current_stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="_state", ctx=ast.Store()
                    )
                ],
                value=ast.Constant(value=after_ph),
            )
        )
        self.current_stmts.append(
            ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=None)],
                    keywords=[],
                )
            )
        )

    def visit_Continue(self, node):
        if not self.loop_stack:
            return
        head = self.loop_stack[-1]["head"]
        self.current_stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="_state", ctx=ast.Store()
                    )
                ],
                value=ast.Constant(value=head),
            )
        )
        self.current_stmts.append(
            ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="run", ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=None)],
                    keywords=[],
                )
            )
        )

    def visit_Name(self, node):
        # Rewrite variable reads: x -> self._ctx['x']
        if isinstance(node.ctx, ast.Load) and node.id in self.varnames:
            return ast.Subscript(
                value=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr="_ctx", ctx=ast.Load()
                ),
                slice=ast.Constant(value=node.id),
                ctx=ast.Load(),
            )
        return node

    def visit_FunctionDef(self, node):
        # We do NOT visit the body of the inner function, as it's a separate scope.
        # But we must ensure the function is stored in _ctx.
        self.current_stmts.append(node)

        if node.name in self.varnames:
            assign = ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="_ctx",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_ctx",
                            ctx=ast.Load(),
                        ),
                        attr="set",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=node.name),
                        ast.Name(id=node.name, ctx=ast.Load())
                    ],
                    keywords=[],
                )
            )
            self.current_stmts.append(assign)

    def visit_AsyncFunctionDef(self, node):
        # Same as FunctionDef
        self.current_stmts.append(node)

        if node.name in self.varnames:
            assign = ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="_ctx",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_ctx",
                            ctx=ast.Load(),
                        ),
                        attr="set",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=node.name),
                        ast.Name(id=node.name, ctx=ast.Load())
                    ],
                    keywords=[],
                )
            )
            self.current_stmts.append(assign)

    def visit_Return(self, node):
        # Handle return statement
        # 1. Evaluate value
        value = self.visit(node.value) if node.value else ast.Constant(value=None)
        
        # 2. Set _result = value
        self.current_stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="_result",
                        ctx=ast.Store(),
                    )
                ],
                value=value,
            )
        )
        
        # 3. Set _done = True
        self.current_stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="_done",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Constant(value=True),
            )
        )
        
        # 4. Return value (so run() returns it too)
        # We need to return the value so the caller gets it immediately.
        # But we also set _result for inspection.
        self.current_stmts.append(ast.Return(value=value))

    def visit_Import(self, node):
        # Handle import statement
        # import math -> self._ctx = self._ctx.set('math', __import__('math'))
        # But import is complex.
        # Simplest way: execute import (assigns to locals), then copy to _ctx.
        
        # 1. Emit original import
        self.current_stmts.append(node)
        
        # 2. Copy imported names to _ctx
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0] # Top level name
            if name in self.varnames:
                self.current_stmts.append(
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_ctx",
                                ctx=ast.Store()
                            )
                        ],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id="self", ctx=ast.Load()),
                                    attr="_ctx",
                                    ctx=ast.Load()
                                ),
                                attr="set",
                                ctx=ast.Load()
                            ),
                            args=[ast.Constant(value=name), ast.Name(id=name, ctx=ast.Load())],
                            keywords=[]
                        )
                    )
                )

    def visit_ImportFrom(self, node):
        # Handle from ... import ...
        self.current_stmts.append(node)
        for alias in node.names:
            name = alias.asname or alias.name
            if name in self.varnames:
                self.current_stmts.append(
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_ctx",
                                ctx=ast.Store()
                            )
                        ],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id="self", ctx=ast.Load()),
                                    attr="_ctx",
                                    ctx=ast.Load()
                                ),
                                attr="set",
                                ctx=ast.Load()
                            ),
                            args=[ast.Constant(value=name), ast.Name(id=name, ctx=ast.Load())],
                            keywords=[]
                        )
                    )
                )

    def visit_Raise(self, node):
        new_exc = self.visit(node.exc) if node.exc else None
        new_cause = self.visit(node.cause) if node.cause else None
        self.current_stmts.append(ast.Raise(exc=new_exc, cause=new_cause))


def compile_agent(func: Callable) -> type[AgentMachine]:
    """
    Compiles a generator function into an AgentMachine class.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]

    if isinstance(func_def, ast.AsyncFunctionDef):
        raise TypeError("Async functions are not supported. Use a standard generator function.")

    compiler = CPSCompiler(func.__code__.co_varnames)

    # Process body
    for stmt in func_def.body:
        compiler.visit(stmt)
        
    # Implicit return None at the end
    compiler.visit(ast.Return(value=None))

    compiler._flush_state(None)  # Flush last block

    # Generate run method
    # def run(self, _input=None):
    #     if self._state == 0: ...
    #     elif self._state == 1: ...

    cases = []
    for state_id, stmts in compiler.states.items():
        if not stmts:
            continue
        # Filter None
        if any(s is None for s in stmts):
            print(f"DEBUG: Found None in state {state_id}: {stmts}")
        stmts = [s for s in stmts if s is not None]
        compiler.states[state_id] = stmts
        
        cases.append(
            ast.If(
                test=ast.Compare(
                    left=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()), attr="_state", ctx=ast.Load()
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=state_id)],
                ),
                body=stmts,
                orelse=[],
            )
        )

    # Chain ifs (elif)
    run_body = []
    if cases:
        current = cases[0]
        run_body.append(current)
        for case in cases[1:]:
            current.orelse = [case]
            current = case

        # Wrap in while True loop to allow internal state transitions
        run_body = [ast.While(test=ast.Constant(value=True), body=[cases[0]], orelse=[])]
    else:
        run_body = [ast.Pass()]

    # Create class
    class_name = f"{func.__name__}_Machine"

    # Generate __init__
    # def __init__(self, arg1, arg2, ...):
    #     self._state = 0
    #     self._ctx = {'arg1': arg1, 'arg2': arg2, ...}
    #     self._done = False
    #     self._result = None

    # Extract args
    sig = inspect.signature(func)
    init_args = [ast.arg(arg="self")]
    ctx_keys = []

    for param in sig.parameters.values():
        init_args.append(ast.arg(arg=param.name))
        ctx_keys.append(param.name)

    # Build _ctx dict
    ctx_dict = ast.Dict(
        keys=[ast.Constant(value=k) for k in ctx_keys],
        values=[ast.Name(id=k, ctx=ast.Load()) for k in ctx_keys],
    )

    init_body = [
        ast.ImportFrom(module="pyrsistent", names=[ast.alias(name="pmap", asname=None)], level=0),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr="_state", ctx=ast.Store()
                )
            ],
            value=ast.Constant(value=0),
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr="_ctx", ctx=ast.Store()
                )
            ],
            value=ast.Call(
                func=ast.Name(id="pmap", ctx=ast.Load()),
                args=[ctx_dict],
                keywords=[],
            ),
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr="_done", ctx=ast.Store()
                )
            ],
            value=ast.Constant(value=False),
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr="_result", ctx=ast.Store()
                )
            ],
            value=ast.Constant(value=None),
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr="_stack", ctx=ast.Store()
                )
            ],
            value=ast.List(elts=[], ctx=ast.Load()),
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr="_exception", ctx=ast.Store()
                )
            ],
            value=ast.Constant(value=None),
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="_exception_type",
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=None),
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr="_in_handler", ctx=ast.Store()
                )
            ],
            value=ast.Constant(value=False),
        ),
    ]

    # We need to update run() to handle stack
    # But run() is generated from the function body.
    # We can wrap the generated run body?
    # Or we can add a preamble to run().

    # Actually, if we use a stack, the `run` method needs to delegate to the top of the stack.
    # But `run` IS the code for this machine.
    # So we need a `dispatch` method?
    # Or `run` checks stack first.

    # If stack is not empty:
    #   sub = stack[-1]
    #   sig = sub.run(_input)
    #   if sub._done:
    #       stack.pop()
    #       _input = sub._result
    #       # Fall through to execute self's logic with result
    #   else:
    #       return sig

    # We need to inject this logic at the start of `run`.

    stack_logic = [
        ast.If(
            test=ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()), attr="_stack", ctx=ast.Load()
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="sub", ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()), attr="_stack", ctx=ast.Load()
                        ),
                        slice=ast.Constant(value=-1),
                        ctx=ast.Load(),
                    ),
                ),
                ast.Assign(
                    targets=[ast.Name(id="sig", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="sub", ctx=ast.Load()), attr="run", ctx=ast.Load()
                        ),
                        args=[ast.Name(id="_input", ctx=ast.Load())],
                        keywords=[],
                    ),
                ),
                ast.If(
                    test=ast.Attribute(
                        value=ast.Name(id="sub", ctx=ast.Load()), attr="_done", ctx=ast.Load()
                    ),
                    body=[
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Attribute(
                                        value=ast.Name(id="self", ctx=ast.Load()),
                                        attr="_stack",
                                        ctx=ast.Load(),
                                    ),
                                    attr="pop",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[],
                            )
                        ),
                        ast.Assign(
                            targets=[ast.Name(id="_input", ctx=ast.Store())],
                            value=ast.Attribute(
                                value=ast.Name(id="sub", ctx=ast.Load()),
                                attr="_result",
                                ctx=ast.Load(),
                            ),
                        ),
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id="print", ctx=ast.Load()),
                                args=[
                                    ast.Constant(value="DEBUG: Stack pop. Result:"),
                                    ast.Name(id="_input", ctx=ast.Load()),
                                ],
                                keywords=[],
                            )
                        ),
                    ],
                    orelse=[ast.Return(value=ast.Name(id="sig", ctx=ast.Load()))],
                ),
            ],
            orelse=[],
        )
    ]

    run_body = stack_logic + run_body

    class_def = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id="AgentMachine", ctx=ast.Load())],
        keywords=[],
        body=[
            ast.FunctionDef(
                name="__init__",
                args=ast.arguments(
                    posonlyargs=[],
                    args=init_args,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],  # We should handle defaults too, but for now strict
                ),
                body=init_body,
                decorator_list=[],
            ),
            ast.FunctionDef(
                name="run",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[
                        ast.arg(arg="self"),
                        ast.arg(arg="_input", default=ast.Constant(value=None)),
                    ],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[ast.Constant(value=None)],
                ),
                body=run_body,
                decorator_list=[],
            ),
        ],
        decorator_list=[],
    )

    # Fix locations
    ast.fix_missing_locations(class_def)

    # Compile and execute definition
    module_ast = ast.Module(body=[class_def], type_ignores=[])

    # Execute in the function's globals to support shared state
    # We use a separate locals dict to capture the generated class without polluting globals
    exec_locals = {"AgentMachine": AgentMachine, "pmap": pmap}

    exec(compile(module_ast, filename="<string>", mode="exec"), func.__globals__, exec_locals)
    return exec_locals[class_name]