# Layer 2 — Orchestrator: Development Specification

> **Parent**: [SPEC-overview.md](SPEC-overview.md)  
> **Layer**: 2 (Middle)  
> **Status**: Ready for Implementation — All decisions finalized  
> **Last Updated**: 2026-03-12  
> **Depends On**: Layer 3 (Execution Platform) — via protocol abstractions only

---

## 1. Purpose

The Orchestrator Layer is the **strategy brain** of rh_cognitiv. It receives high-level intent (from the Cognitive Layer or directly from user code) and translates it into concrete execution plans expressed as DAGs. It:

- Builds **Plan DAGs** — the declared intent of what should happen
- Produces **Execution DAGs** — the runtime graph that records what actually happened
- Manages **node traversal**, including branching, loops, and parallel fan-out
- Enforces **validation** before each node executes
- Coordinates with the Execution Platform through **adapters** (never directly)
- Integrates with **ExecutionState** for time-travel (undo/redo) at the orchestration level

This layer **never** performs actual execution (no LLM calls, no function calls). It delegates everything downward through the L2→L3 adapter.

---

## 2. Component Architecture

```
orchestrators/
├── protocols.py          # ABCs — OrchestratorProtocol, NodeProtocol, DAGProtocol
├── models.py             # Pydantic models — NodeResult, ExecutionDAGEntry, OrchestratorConfig, etc.
│
├── nodes.py              # BaseNode, ExecutionNode, FlowNode hierarchies
├── flow_nodes.py         # ForEach, Filter, Switch, Get, IfNotOk + FlowHandlerRegistry
├── plan_dag.py           # PlanDAG — static intent graph + DAGBuilder
├── execution_dag.py      # ExecutionDAG — runtime graph (what happened)
│
├── dag_orchestrator.py   # DAGOrchestrator — core traversal engine
├── validation.py         # ValidationPipeline — composable pre-flight checks
└── adapters.py           # L2→L3 adapter registry — translates nodes into L3 calls
```

### Relationship to Layer 3

```
Orchestrator Layer                         Execution Platform (Layer 3)
──────────────────                         ──────────────────────────────

DAGOrchestrator.run_node(node, data)
        │
        ├── validation.validate(node, data)       (L2 ValidationPipeline)
        │
        ├── adapter.execute(node, data)  ─────► ExecutionEvent
        │                                        │
        │                                   PolicyChain(
        │                                     HandlerRegistry.handle()
        │                                   )  → BudgetPolicy, TimeoutPolicy,
        │                                        RetryPolicy wrapped
        │                                        │
        │                                   EventBus.emit(lifecycle)   ← observability only
        │                                   State.snapshot()
        │                                        │
        ├── ◄──────────────────────────── ExecutionResult[T]
        │                                  (T = LLMResultData | FunctionResultData
        │                                       | ToolResultData)
        │
        ├── execution_dag.record(node, result)
        │
        └── ready_queue.advance(result)
```

**Key**: `HandlerRegistry.handle()` dispatches execution directly. `PolicyChain` wraps it. `EventBus` is for lifecycle/observability events only — **not** for triggering execution.

**Key L3 integration points** (from the implemented execution_platform):

| L3 Component | How L2 Uses It |
|---|---|
| `HandlerRegistry` | Adapter dispatches `ExecutionEvent` via `registry.handle(event, data, configs)` |
| `PolicyChain` | Wraps handler calls with retry, timeout, budget — L2 does **not** re-implement these |
| `BudgetTracker` | Shared across all nodes in a run; L2 checks `can_proceed()` in validation, L3 enforces |
| `ExecutionState` | L2 calls `snapshot()` per node, uses `undo()`/`redo()` for time-travel, `add_level()`/`remove_level()` for nesting |
| `EventBus` | L2 emits lifecycle events for observability; uses `wait_for()` for escalation round-trips |
| `ExecutionResult[T]` | Generic typed results; adapter normalizes `T` into `NodeResult` for L2 consumption |
| `EscalationRequested/Resolved` | L2 pauses DAG traversal on escalation, resumes on resolution |

### L3 Types L2 Depends On

```python
# From execution_platform.types
ID, Timestamp, Ext, EntryRef[T], generate_ulid, now_timestamp

# From execution_platform.protocols (ABCs only — never concrete classes)
EventBusProtocol, HandlerRegistryProtocol, ExecutionStateProtocol,
BudgetTrackerProtocol, PolicyChainProtocol, ContextStoreProtocol

# From execution_platform.events
ExecutionEvent, TextPayload, DataPayload, FunctionPayload, ToolPayload,
EscalationRequested, EscalationResolved

# From execution_platform.models
EventKind, EventStatus, ExecutionResult, TokenUsage, ResultMetadata,
LLMResultData, FunctionResultData, ToolResultData

# From execution_platform.errors
CognitivError, BudgetError, InterruptError, EscalationError, ErrorCategory
```

---

## 3. The Dual DAG Model

### Plan DAG — What Should Happen
- **Static** — defined before execution begins (by the Cognitive Layer or user code)
- Represents the **intended** flow: "do A, then B, then if X do C else D"
- Nodes are **abstract** — they describe what to do, not how
- **Frozen** once submitted to the orchestrator (DI-L2-03)
- Can be serialized, versioned, and compared
- Supports **sub-graphs** (composite nodes) for nesting (DI-L2-05)

### Execution DAG — What Actually Happened
- **Dynamic** — built at runtime as nodes execute
- **Append-only** — entries are never removed, only marked `rolled_back` on undo (DI-L2-01)
- Records **real** execution order, including retries, fallbacks, and loop iterations
- Each entry links to: the Plan DAG node it came from, the `NodeResult`, timing, and `state_version`
- This is what powers time-travel: "show me what happened at step 5 and let me go back to step 3"

### Why Two DAGs?

| Concern | Plan DAG | Execution DAG |
|---------|----------|---------------|
| When built | Before execution | During execution |
| Mutability | Immutable once submitted | Append-only |
| Purpose | Intent, scheduling, optimization | Audit, replay, undo/redo |
| Branching meaning | "these could happen in parallel" | "these DID happen in parallel" |
| Loop representation | Single node with loop config | N entries (one per iteration) |

---

## 4. Design Decisions

### DD-L2-01: Node Type Hierarchy — A+C Hybrid

Two base types (`ExecutionNode` + `FlowNode`) as the runtime hierarchy for behavior and shared logic, combined with a `kind` discriminator for Pydantic tagged-union serialization.

```python
class BaseNode(BaseModel):
    id: ID
    kind: str                              # discriminator for Pydantic union + adapter dispatch
    label: str | None = None
    timeout_seconds: float | None = None   # per-node override (DI-L2-07)
    max_retries: int | None = None         # per-node override (DI-L2-07)
    ext: Ext = {}

class ExecutionNode(BaseNode): ...   # TextNode, DataNode, FunctionNode, ToolNode
class FlowNode(BaseNode): ...       # ForEach, Filter, Switch, Get, IfNotOk

# Pydantic discriminated union for serialization:
Node = Annotated[
    TextNode | DataNode | FunctionNode | ToolNode |
    ForEachNode | FilterNode | SwitchNode | GetNode | IfNotOkNode,
    Field(discriminator="kind")
]
```

ExecutionNode subtypes map 1:1 to L3's `EventKind`: `TextNode→TEXT`, `DataNode→DATA`, `FunctionNode→FUNCTION`, `ToolNode→TOOL`. FlowNodes carry their own `kind` values (`foreach`, `filter`, `switch`, `get`, `if_not_ok`) that never reach L3. The `kind` field drives both Pydantic union parsing AND adapter registry dispatch (DD-L2-04).

---

### DD-L2-02: DAG Representation — Custom Lightweight DAG Class

A custom lightweight `DAG` class. Edge list for serialization, adjacency dict internally. No `networkx` dependency.

```python
class DAG:
    _nodes: dict[str, Node]
    _forward: dict[str, list[str]]   # adjacency: node → successors
    _reverse: dict[str, list[str]]   # reverse adjacency: node → predecessors

    def add_node(self, node_id: str, node: Node) -> None: ...
    def add_edge(self, from_id: str, to_id: str) -> None: ...
    def get_node(self, node_id: str) -> Node: ...
    def successors(self, node_id: str) -> list[str]: ...
    def predecessors(self, node_id: str) -> list[str]: ...
    def topological_order(self) -> list[str]: ...
    def get_initial_nodes(self) -> list[str]: ...
    def get_newly_ready_nodes(self, completed: set[str]) -> list[str]: ...
    def validate(self) -> None: ...    # acyclic, connected, all edge refs valid
    def to_edge_list(self) -> list[dict]: ...  # for serialization
    def node_count(self) -> int: ...
```

---

### DD-L2-03: Orchestrator Traversal — Topological Sort + Ready Queue

Ready-queue traversal. Maintain a "ready queue" of nodes whose dependencies are all satisfied. Execute from the queue.

```python
class DAGOrchestrator:
    async def run(self, dag: PlanDAG, data: Any) -> ExecutionDAG:
        self.state.add_level()                    # L3 nesting tracking
        try:
            ready = dag.get_initial_nodes()
            completed: set[str] = set()
            while ready:
                # Stage 1: sequential
                for node_id in ready:
                    if self._interrupted:          # interrupt check (DI-L2-06)
                        raise InterruptError(...)
                    result = await self.run_node(dag.get_node(node_id), data)
                    completed.add(node_id)
                ready = dag.get_newly_ready_nodes(completed)
            return self.execution_dag
        finally:
            self.state.remove_level()
```

Stage 1: `await self.run_node(n)` sequentially. Stage 2: `await asyncio.gather(*[self.run_node(n) for n in ready])` for parallel branches. The data structure supports both without changes.

---

### DD-L2-04: Adapter — Registry (Strategy per Node Kind)

Adapter Registry. The orchestrator depends on the `NodeAdapter` protocol, not on Layer 3 directly. Each concrete adapter is the **full L2→L3 bridge**: convert, execute, normalize.

```python
class NodeAdapter(Protocol):
    async def execute(self, node: BaseNode, data: Any, configs: Any, platform: PlatformRef) -> NodeResult: ...

class AdapterRegistry:
    _adapters: dict[str, NodeAdapter] = {}

    def register(self, kind: str, adapter: NodeAdapter) -> None: ...
    async def execute(self, node: BaseNode, data: Any, configs: Any, platform: PlatformRef) -> NodeResult:
        return await self._adapters[node.kind].execute(node, data, configs, platform)
```

Each concrete adapter:

1. **Converts** node → `ExecutionEvent` (with correct `EventKind` + payload: `TextPayload`, `DataPayload`, `FunctionPayload`, `ToolPayload`)
2. **Builds** a per-node `PolicyChain` using node's timeout/retry overrides merged with orchestrator defaults (DI-L2-07)
3. **Executes** via `PolicyChain(HandlerRegistry.handle, event, data, configs)` — NOT via `EventBus.emit()`
4. **Normalizes** the `ExecutionResult[T]` into a uniform `NodeResult` for L2 consumption

```python
class TextNodeAdapter(NodeAdapter):
    async def execute(self, node: TextNode, data, configs, platform) -> NodeResult:
        event = ExecutionEvent(kind=EventKind.TEXT, payload=TextPayload(
            prompt=node.prompt, system_prompt=node.system_prompt,
            model=node.model, temperature=node.temperature, max_tokens=node.max_tokens
        ))
        chain = platform.build_policy_chain(node)  # per-node timeout/retry
        result = await chain(platform.registry.handle, event, data, configs)
        return NodeResult.from_execution_result(result)
```

---

### DD-L2-05: FlowNode Execution — Orchestrator Middleware

FlowNodes are modular handlers that the orchestrator invokes before deciding the next node. They modify the DAG traversal state (expand nodes, skip branches, etc.). L3 is never involved — FlowNodes are a pure L2 concern.

```python
class FlowHandler(Protocol):
    async def handle(self, node: FlowNode, data: Any, dag_state: DAGTraversalState) -> FlowResult: ...

class FlowHandlerRegistry:
    _handlers: dict[str, FlowHandler] = {}

    def register(self, kind: str, handler: FlowHandler) -> None: ...
    async def handle(self, node: FlowNode, data: Any, dag_state: DAGTraversalState) -> FlowResult: ...

# Concrete handlers:
class ForEachHandler(FlowHandler): ...   # expand inner node for each data item
class FilterHandler(FlowHandler): ...    # filter data, pass to successor
class SwitchHandler(FlowHandler): ...    # pick branch based on condition
class GetHandler(FlowHandler): ...       # retrieve data from context
class IfNotOkHandler(FlowHandler): ...   # skip/redirect on error
```

FlowNode executions don't touch `BudgetTracker` — no tokens consumed. Results are recorded in the ExecutionDAG as expanded entries.

---

### DD-L2-06: Time-Travel — Hybrid, Staged

Per-node snapshots (Stage 1) + named checkpoints (Stage 2+).

**Stage 1**: Call `state.snapshot()` after each node execution. The returned version number is stored in the `ExecutionDAGEntry.state_version` for random-access restore. L3 uses full deep-copy snapshots.

**Stage 2+**: Replace with delta chains (L3's evolution path via jsonpatch). Add named checkpoints as an L2-level abstraction: a dict mapping `checkpoint_name → version` maintained by the orchestrator.

```python
# In DAGOrchestrator.run_node():
result = await self.adapter.execute(node, data, configs, platform)
version = self.state.snapshot()  # L3 returns version: int
self.execution_dag.record(node, result, state_version=version)

# Undo to a specific step:
entry = self.execution_dag.get_entry(target_node_id)
self.state.restore(entry.state_version)  # L3 restores full state
self.execution_dag.mark_rolled_back(from_entry=entry)
# Resume traversal from that point
```

L3's `gc_collect(keep_first, keep_last)` is available for long-running DAGs.

---

### DD-L2-07: ForEach Parallel Execution — Configurable Failure Strategy

Each ForEach node specifies its failure strategy. Default: `fail_fast`.

```python
class ForEachNode(FlowNode):
    kind: Literal["foreach"] = "foreach"
    inner_node_id: str
    failure_strategy: Literal["fail_fast", "collect_all"] = "fail_fast"
    # Stage 2: add "fail_after_n" with max_failures: int | None
```

- `fail_fast`: first failure cancels all remaining branches
- `collect_all`: all branches run to completion, mixed ok/fail results

L3's `PolicyChain` handles retries *within* a single branch. ForEach's `failure_strategy` handles failures *across* branches. All branches share the same `BudgetTracker` instance (safe under single-threaded asyncio).

---

## 5. Design Insights

### DI-L2-01: ExecutionDAG — Append-Only

Append-only. Undo doesn't delete entries — it marks them as `rolled_back` and creates a new branch. Preserves full history. Aligns with L3's append-only snapshot chain.

```python
class NodeExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"
    WAITING = "waiting"          # escalation in progress

class ExecutionDAGEntry(BaseModel):
    id: ID
    node_id: str
    plan_node_ref: str                # which PlanDAG node this came from
    status: NodeExecutionStatus
    result: NodeResult | None = None
    started_at: Timestamp
    completed_at: Timestamp | None = None
    parent_entry_id: ID | None = None  # for branching (undo creates a new branch)
    state_version: int | None = None   # L3 ExecutionState snapshot version
```

---

### DI-L2-02: Validation — Composable Pipeline (Separate from L3 PolicyChain)

Pre-flight validation pipeline. **Separate from** L3's `PolicyChain` — different scope, different timing.

| Concern | L2 ValidationPipeline | L3 PolicyChain |
|---|---|---|
| When | Before adapter call | During execution |
| Scope | Orchestration (graph, dependencies, auth) | Execution (retry, timeout, budget enforcement) |
| On failure | Skip node, fail DAG, or escalate | Retry, timeout error, budget error |

```python
class NodeValidator(Protocol):
    async def validate(self, node: BaseNode, data: Any, context: ValidationContext) -> ValidationResult: ...

class ValidationPipeline:
    validators: list[NodeValidator]

    async def validate(self, node, data, context) -> ValidationResult:
        for v in self.validators:
            result = await v.validate(node, data, context)
            if not result.ok:
                return result
        return ValidationResult.ok()
```

Built-in validators:
- **InputSchemaValidator**: checks data matches node's expected input (pure L2)
- **DependencyValidator**: checks all upstream nodes completed in ExecutionDAG (pure L2)
- **BudgetValidator**: calls L3's `BudgetTracker.can_proceed()` as pre-flight check (L3 still enforces)

---

### DI-L2-03: PlanDAG Immutability — Frozen on Submission

PlanDAG is **frozen** once submitted to the orchestrator. No modifications during execution. Replan protocol (full replacement or incremental patch) is deferred to Stage 2. For Stage 1, PlanDAGs are fully immutable.

---

### DI-L2-04: DAGBuilder — Ergonomic Fluent API

Priority P2. Builder validates graph (acyclic, connected, all edge refs valid) and returns a frozen `PlanDAG`.

```python
dag = (
    DAGBuilder("my-pipeline")
    .add_node("extract", DataNode(prompt="Extract entities"))
    .add_node("transform", FunctionNode(function_name="clean_data"))
    .add_node("summarize", TextNode(prompt="Summarize results"))
    .edge("extract", "transform")
    .edge("transform", "summarize")
    .build()  # returns a frozen PlanDAG
)
```

---

### DI-L2-05: Sub-Graph Support (Composite Nodes)

PlanDAG supports sub-graphs via composite nodes. A composite node references another `PlanDAG` that is inlined during traversal. L3's `ExecutionState.add_level()` / `remove_level()` tracks nesting depth.

```python
class CompositeNode(BaseNode):
    kind: Literal["composite"] = "composite"
    sub_dag: PlanDAG    # nested DAG to execute as a single unit
```

Priority P3 — implement after core traversal works.

---

### DI-L2-06: Interrupt Handling

The traversal loop checks for interrupts before each node execution. L3 provides `InterruptError` (category=INTERRUPT). L2 uses `EventBus.wait_for()` for async external callbacks.

```python
# In the ready-queue loop:
for node_id in ready:
    if self._interrupted:
        raise InterruptError("DAG execution interrupted by user")
    result = await self.run_node(...)
```

The orchestrator exposes `interrupt()` for external callers (API, UI).

---

### DI-L2-07: Per-Node Timeout/Retry Config with Orchestrator Defaults

Nodes carry optional `timeout_seconds` and `max_retries` fields. The adapter merges these with orchestrator-wide defaults to build a per-node `PolicyChain`.

```python
# Orchestrator defaults
class OrchestratorConfig(BaseModel):
    default_timeout_seconds: float = 30.0
    default_max_retries: int = 3
    default_retry_base_delay: float = 0.1

# Per-node override (on BaseNode)
class BaseNode(BaseModel):
    timeout_seconds: float | None = None   # None = use orchestrator default
    max_retries: int | None = None         # None = use orchestrator default

# Adapter builds PolicyChain per-node:
def build_policy_chain(self, node: BaseNode, config: OrchestratorConfig) -> PolicyChain:
    timeout = node.timeout_seconds or config.default_timeout_seconds
    retries = node.max_retries or config.default_max_retries
    return PolicyChain([
        BudgetPolicy(tracker=self.budget_tracker),
        TimeoutPolicy(seconds=timeout),
        RetryPolicy(max_attempts=retries),
    ])
```

---

### DI-L2-08: Node Inputs/Outputs — Schema-Per-Kind

Node inputs mirror L3's typed payload pattern. Each ExecutionNode subtype has typed fields matching the L3 payload it will produce.

```python
class TextNode(ExecutionNode):
    kind: Literal["text"] = "text"
    prompt: str
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

class DataNode(ExecutionNode):
    kind: Literal["data"] = "data"
    prompt: str
    output_schema: dict[str, Any] | None = None
    model: str | None = None

class FunctionNode(ExecutionNode):
    kind: Literal["function"] = "function"
    function_name: str
    args: list[Any] = []
    kwargs: dict[str, Any] = {}

class ToolNode(ExecutionNode):
    kind: Literal["tool"] = "tool"
    prompt: str
    tools: list[dict[str, Any]] = []
    model: str | None = None
```

`NodeResult` is the uniform output wrapper:

```python
class NodeResult(BaseModel):
    ok: bool
    value: Any = None                    # the primary output (text, return_value, etc.)
    error_message: str | None = None
    error_category: str | None = None    # maps to L3's ErrorCategory
    token_usage: TokenUsage | None = None
    metadata: ResultMetadata | None = None
```

---

### DI-L2-09: Escalation Handling at DAG Level

When a handler escalates, the orchestrator pauses the DAG and waits for resolution.

```python
# In adapter, after catching EscalationError:
async def handle_escalation(self, node, event, platform):
    # 1. Emit escalation request
    await platform.bus.emit(EscalationRequested(
        event_id=event.id, question=..., options=..., node_id=node.id
    ))
    # 2. Persist escalation state in L3
    platform.state.set_escalated(event.id, question, options, node.id, resume_data)
    # 3. Mark execution DAG entry as waiting
    self.execution_dag.mark_waiting(node.id)
    # 4. Wait for human decision
    resolution = await platform.bus.wait_for(
        EscalationResolved, filter=lambda e: e.event_id == event.id
    )
    # 5. Clear escalation, resume
    platform.state.clear_escalation()
    return resolution.decision
```

---

## 6. Implementation Phases

### Phase 1 — Foundation: Protocols, Models, Node Types
> **Goal**: Establish all contracts and types. No runtime behavior yet — just shapes.

| Step | File | Deliverable |
|------|------|-------------|
| 1.1 | `protocols.py` | ABCs: `OrchestratorProtocol`, `DAGProtocol`, `NodeProtocol`, `NodeAdapterProtocol`, `FlowHandlerProtocol`, `NodeValidatorProtocol`, `ValidationPipelineProtocol` |
| 1.2 | `models.py` | Pydantic models: `NodeResult`, `ValidationResult`, `FlowResult`, `OrchestratorConfig`, `NodeExecutionStatus` enum, `ExecutionDAGEntry`, `DAGRunStatus` enum |
| 1.3 | `nodes.py` | `BaseNode`, `ExecutionNode`, `FlowNode`, concrete execution nodes: `TextNode`, `DataNode`, `FunctionNode`, `ToolNode`. Each with typed fields matching L3 payloads (DI-L2-08) |
| 1.4 | `flow_nodes.py` | `ForEachNode` (with `failure_strategy`), `FilterNode`, `SwitchNode`, `GetNode`, `IfNotOkNode`, `CompositeNode` (DI-L2-05, stub) |
| 1.5 | Tests | Node construction, Pydantic serialization/deserialization roundtrip, discriminated union parsing, field validation |

**L3 imports needed**: `ID`, `Timestamp`, `Ext`, `generate_ulid`, `now_timestamp`, `TokenUsage`, `ResultMetadata`, `ErrorCategory`

### Phase 2 — DAG Data Structures
> **Goal**: PlanDAG (static graph) + ExecutionDAG (runtime recording). The core data structures the orchestrator operates on.

| Step | File | Deliverable |
|------|------|-------------|
| 2.1 | `plan_dag.py` | `DAG` class with `add_node()`, `add_edge()`, `successors()`, `predecessors()`, `topological_order()`, `get_initial_nodes()`, `get_newly_ready_nodes(completed)`, `validate()`, `to_edge_list()`. `PlanDAG` frozen wrapper around `DAG`. `DAGBuilder` fluent API |
| 2.2 | `execution_dag.py` | `ExecutionDAG` — append-only recording. `record(node, result, state_version)`, `mark_rolled_back(from_entry)`, `mark_waiting(node_id)`, `get_entry(node_id)`, `entries()`, `get_by_status(status)` |
| 2.3 | Tests | DAG construction, topological sort correctness, cycle detection, initial nodes, ready-queue advancement, ExecutionDAG append-only invariant, rollback marking |

**L3 imports needed**: None (pure L2 data structures)

### Phase 3 — Adapter Registry + Validation Pipeline
> **Goal**: The bridge to L3 and the pre-flight safety rails. These are stateless components that can be tested with mocks.

| Step | File | Deliverable |
|------|------|-------------|
| 3.1 | `adapters.py` | `AdapterRegistry`, `NodeAdapter` protocol, concrete adapters: `TextNodeAdapter`, `DataNodeAdapter`, `FunctionNodeAdapter`, `ToolNodeAdapter`. Each builds `ExecutionEvent` + `PolicyChain`, executes, normalizes to `NodeResult`. `PlatformRef` dataclass holding L3 references |
| 3.2 | `validation.py` | `ValidationPipeline`, concrete validators: `InputSchemaValidator`, `DependencyValidator`, `BudgetValidator` |
| 3.3 | Tests | Adapter dispatch for all 4 event kinds (mock `HandlerRegistry` + `PolicyChain`), `NodeResult` normalization from each `ExecutionResult[T]`, ValidationPipeline ordering + short-circuit, BudgetValidator integration with mock `BudgetTracker` |

**L3 imports needed**: `ExecutionEvent`, `EventKind`, `TextPayload`, `DataPayload`, `FunctionPayload`, `ToolPayload`, `ExecutionResult`, `LLMResultData`, `FunctionResultData`, `ToolResultData`, `HandlerRegistryProtocol`, `PolicyChainProtocol`, `BudgetTrackerProtocol`, `PolicyChain`, `RetryPolicy`, `TimeoutPolicy`, `BudgetPolicy`

### Phase 4 — Core Orchestrator (Linear Traversal)
> **Goal**: DAGOrchestrator — the main loop. Linear execution (sequential ready queue). No parallelism yet.

| Step | File | Deliverable |
|------|------|-------------|
| 4.1 | `dag_orchestrator.py` | `DAGOrchestrator` with `async run(dag, data) -> ExecutionDAG`. Sequential ready-queue loop. Calls `state.add_level()` / `remove_level()`. Calls `validation.validate()` → `adapter.execute()` → `state.snapshot()` → `execution_dag.record()`. Interrupt check per-node. `OrchestratorConfig` for defaults |
| 4.2 | Integration tests | Linear DAG (A→B→C), branching DAG (A→B, A→C run sequentially), failing node (verify ExecutionDAG records failure), interrupt mid-execution. Uses real L3 components (in-memory) |

**L3 imports needed**: `ExecutionStateProtocol`, `EventBusProtocol`, `InterruptError`

### Phase 5 — FlowNode Handlers
> **Goal**: ForEach, Filter, Switch, Get, IfNotOk as orchestrator middleware. Modify DAG traversal.

| Step | File | Deliverable |
|------|------|-------------|
| 5.1 | `flow_nodes.py` (update) | `FlowHandlerRegistry`, `DAGTraversalState` class. Concrete handlers: `ForEachHandler`, `FilterHandler`, `SwitchHandler`, `GetHandler`, `IfNotOkHandler` |
| 5.2 | `dag_orchestrator.py` (update) | Integrate FlowHandlerRegistry into traversal loop. Detect FlowNodes, dispatch to FlowHandler instead of adapter. Record expanded entries in ExecutionDAG |
| 5.3 | Tests | ForEach expansion (N items → N executions), Filter (subset), Switch (branch selection), IfNotOk (error redirect). ExecutionDAG shape verification for each |

**L3 imports needed**: None (FlowNodes are pure L2)

### Phase 6 — Parallel Execution + ForEach Strategies
> **Goal**: `asyncio.gather` for parallel branches in ready queue. ForEach failure strategies.

| Step | File | Deliverable |
|------|------|-------------|
| 6.1 | `dag_orchestrator.py` (update) | Parallel ready-queue: `asyncio.gather(*[self.run_node(n) for n in ready])`. Partial failure handling |
| 6.2 | `flow_nodes.py` (update) | ForEach `fail_fast` and `collect_all` strategies implemented |
| 6.3 | Tests | Parallel branch execution (verify speedup with async sleeps), ForEach fail_fast (one failure kills all), ForEach collect_all (partial results), shared BudgetTracker under parallel load |

### Phase 7 — Time-Travel + Escalation
> **Goal**: Undo/redo via L3's ExecutionState. Escalation pause/resume.

| Step | File | Deliverable |
|------|------|-------------|
| 7.1 | `dag_orchestrator.py` (update) | `undo(target_node_id)` — mark entries as rolled_back, `state.restore(version)`. `redo()` — replay from restored point. Named checkpoints dict (`checkpoint_name → version`) |
| 7.2 | `adapters.py` (update) | Escalation handling: catch `EscalationError`, emit `EscalationRequested`, `await bus.wait_for(EscalationResolved)`, resume |
| 7.3 | Tests | Execute 5 nodes, undo to step 3 (verify state matches), redo. Escalation: pause DAG, resolve, verify resume. ExecutionDAG shape after undo (rolled_back entries preserved) |

### Phase 8 — Integration & Smoke Tests
> **Goal**: End-to-end validation with full L3 stack.

| Step | Deliverable |
|------|-------------|
| 8.1 | Full lifecycle: build PlanDAG → run DAGOrchestrator → adapters → L3 handlers → ExecutionDAG complete |
| 8.2 | Time-travel: execute 5 nodes, undo 2, verify state + ExecutionDAG |
| 8.3 | Budget exhaustion: run nodes until budget exceeded, verify graceful stop at validation |
| 8.4 | ForEach parallel: fan-out 3 branches, collect results |
| 8.5 | Mixed DAG: linear + branch + ForEach + Switch in one DAG |
| 8.6 | Escalation roundtrip: handler escalates, human resolves, DAG resumes |

---

## 7. Testing Strategy

| Level | What | How |
|-------|------|-----|
| **Unit** | Node types, DAG construction, topological sort | No L3 dependency |
| **Unit** | FlowNode handlers (ForEach, Switch, Filter) | Verify expansion/routing logic |
| **Unit** | Adapter registry + adapters | Mock `HandlerRegistryProtocol`, `PolicyChainProtocol` |
| **Unit** | ValidationPipeline | Mock validators, verify ordering + short-circuit |
| **Integration** | DAGOrchestrator → Adapter → L3 handlers | Real L3 with in-memory store |
| **Integration** | ExecutionDAG recording | Run a DAG, verify the execution trace shape |
| **Integration** | Time-travel: undo/redo | Real L3 ExecutionState, verify snapshot restore |
| **Scenario** | Linear, branching, ForEach, failing, escalation DAGs | Full stack assertions on results + ExecutionDAG |
| **Property** | DAG validity | Hypothesis: random DAG generation, verify topological properties |

---

## 8. Readiness Checklist

| Concern | Status | Notes |
|---------|--------|-------|
| All design decisions made | **Done** | 7 DDs + 9 DIs finalized |
| L3 alignment verified | **Done** | Spec corrected for actual L3 behavior |
| Node type hierarchy | **Done** | A+C hybrid, maps to L3 EventKind |
| DAG representation | **Done** | Custom lightweight class |
| Traversal strategy | **Done** | Ready queue, sequential Stage 1 → parallel Stage 2 |
| Adapter design | **Done** | Registry, full L2→L3 bridge via PolicyChain |
| FlowNode model | **Done** | Orchestrator middleware, pure L2 |
| Time-travel | **Done** | Per-node snapshots via L3 ExecutionState |
| ForEach parallelism | **Done** | Configurable failure strategy |
| Validation pipeline | **Done** | Composable, separate from L3 PolicyChain |
| Node inputs/outputs | **Done** | Schema-per-kind, NodeResult wrapper |
| Per-node config | **Done** | Orchestrator defaults + per-node overrides |
| Interrupt handling | **Done** | Check per-node in traversal loop |
| Escalation handling | **Done** | Pause DAG, wait_for resolution, resume |
| Sub-graph support | **Done** | CompositeNode, P3 |
| Implementation phases | **Done** | 8 phases, dependency-ordered, L3 imports per phase |
