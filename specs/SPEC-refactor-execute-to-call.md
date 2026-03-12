# Refactor: `execute` → `__call__` Impact Analysis

## Motivation

Replace the `.execute()` method convention with `__call__()` so that all executable components are plain callables. This allows:

- **Plain functions as handlers** — `async def my_handler(event, data, configs): ...` works directly, no class needed.
- **Uniform protocol** — anything callable satisfies the interface; no need to remember a method name.
- **Composability** — `functools.partial`, decorators, and lambdas plug in natively.
- **Discoverability** — `callable(obj)` is a universal Python idiom; consumers don't need to know the API.

---

## Scope of Change

### A. Definitions to Rename (`def execute` → `def __call__`)

| File | Class(es) | Count |
|------|-----------|:-----:|
| `protocols.py` | `EventHandlerProtocol.execute`, `PolicyChainProtocol.execute` | 2 |
| `handlers.py` | `TextHandler`, `DataHandler`, `FunctionHandler`, `ToolHandler` | 4 |
| `policies.py` | `PolicyChain.execute` | 1 |
| **Total definitions** | | **7** |

### B. Internal Call Sites (`.execute(` → `(`)

| File | Context | Count |
|------|---------|:-----:|
| `policies.py` L77 | `await handler.execute(event, data, configs)` inside `PolicyChain` | 1 |
| `policies.py` L154 | `await handler.execute(...)` inside `RetryPolicy.execute_with_retry` | 1 |
| `policies.py` L219 | `handler.execute(...)` inside `TimeoutPolicy.execute_with_timeout` | 1 |
| `handlers.py` L68 | `await handler.execute(...)` inside `HandlerRegistry.handle` | 1 |
| **Total internal call sites** | | **4** |

### C. Test Call Sites

| File | Count |
|------|:-----:|
| `test_handlers.py` | 10 |
| `test_policies.py` | 9 (`.execute(`) + 9 (`execute_with_*`) + 6 (stub defs) |
| `test_integration.py` | 10 (`.execute(`) + 1 (`execute_with_*`) + 2 (stub defs) |
| **Total test references** | **~47** |

### D. Related Methods — PolicyProtocol Hooks

These contain "execute" in their *name* but serve a different role (lifecycle hooks, not the callable entry point):

| Method | Where |
|--------|-------|
| `before_execute` | `PolicyProtocol` (protocol + 4 impls) |
| `after_execute` | `PolicyProtocol` (protocol + 4 impls) |
| `execute_with_retry` | `RetryPolicy` (1 def, ~6 call sites in tests) |
| `execute_with_timeout` | `TimeoutPolicy` (1 def, ~4 call sites in tests) |

---

## Decision Matrix — What Changes vs. What Stays

| Symbol | Change? | Rationale |
|--------|:-------:|-----------|
| `EventHandlerProtocol.execute` | **YES → `__call__`** | Core target. Enables plain functions as handlers. |
| `PolicyChainProtocol.execute` | **YES → `__call__`** | Same reason — `chain(handler, event, data, configs)` is natural. |
| `PolicyChain.execute` | **YES → `__call__`** | Implements `PolicyChainProtocol`. |
| `TextHandler.execute` (and 3 siblings) | **YES → `__call__`** | Implement `EventHandlerProtocol`. |
| `HandlerRegistry.handle` | **NO** | Already named `.handle()`, not `.execute()`. Dispatches to `handler(...)`. |
| `before_execute` / `after_execute` | **NO** | These are *hook names* describing *when* they fire relative to execution. Renaming to `before_call`/`after_call` adds churn with no clarity gain — "before execution" is clearer than "before call". |
| `execute_with_retry` | **NO** | Convenience method on `RetryPolicy`. Not the protocol entry point. Renaming to `retry(...)` or `__call__` would be ambiguous since `RetryPolicy` also implements `PolicyProtocol` (with hooks). |
| `execute_with_timeout` | **NO** | Same reasoning as retry. |
| `on_error` | **NO** | Already a hook name, unrelated to `execute`. |

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Breaking downstream code** | **Low** | No external consumers yet — L3 is brand new. Only our own tests call `.execute()`. |
| **Protocol compatibility with plain functions** | **Medium** | `EventHandlerProtocol` is currently an ABC with `async def execute`. To accept plain functions, `HandlerRegistry` and `PolicyChain` need to call `handler(event, data, configs)` instead of `handler.execute(...)`. This works for *both* `__call__` classes and plain async functions. The ABC becomes a structural suggestion (or we switch to `Protocol` typing). |
| **Type checking** | **Low** | `Callable[[Any, Any, Any], Awaitable[ExecutionResult[T]]]` is the effective type. `typing.Protocol` with `__call__` supports this natively. |
| **Test churn** | **Medium** | ~47 references in tests, but all are mechanical find-and-replace. Zero logic changes. |
| **Docstring/comment drift** | **Low** | A handful of docstrings mention `execute`. Quick grep-and-fix. |

---

## Execution Plan

### Phase 1 — Protocol Layer (protocols.py)
1. `EventHandlerProtocol.execute` → `__call__`
2. `PolicyChainProtocol.execute` → `__call__`
3. Keep `PolicyProtocol.before_execute`, `after_execute`, `on_error` unchanged

### Phase 2 — Implementations (handlers.py, policies.py)
1. `TextHandler.execute` → `__call__` (and 3 siblings)
2. `PolicyChain.execute` → `__call__`
3. Update internal call sites: `handler.execute(...)` → `handler(...)` and `chain.execute(...)` → `chain(...)` in:
   - `PolicyChain.__call__` (calls handler)
   - `RetryPolicy.execute_with_retry` (calls handler)
   - `TimeoutPolicy.execute_with_timeout` (calls handler)
   - `HandlerRegistry.handle` (calls handler)

### Phase 3 — Tests (mechanical)
1. All test stub handlers: `async def execute(...)` → `async def __call__(...)`
2. All test call sites: `.execute(...)` → `(...)`
3. Verify `execute_with_retry` / `execute_with_timeout` call sites unchanged (they keep their names)

### Phase 4 — Validate
1. Run full suite (expect 450 green)
2. Verify no remaining references to `.execute(` in src (except `before_execute`, `after_execute`, `execute_with_*`)

---

## Estimated Diff Size

| Scope | Files | Lines changed |
|-------|:-----:|:-------------:|
| Protocols | 1 | ~6 |
| Implementations | 2 | ~15 |
| Tests | 3 | ~55 |
| **Total** | **6** | **~76** |

---

## Recommendation

**GO.** The change is safe, mechanical, and improves the API surface significantly. There are no external consumers, the test suite fully covers all paths, and the entire refactor is a rename with zero logic changes. The biggest win — plain async functions as handlers — unlocks a much more ergonomic API for L2 orchestrator integration.
