# Layer 2 — Orchestrator: Development Specification

> **Parent**: [SPEC-overview.md](SPEC-overview.md)  
> **Layer**: 2 (Middle)  
> **Status**: Design Phase  
> **Last Updated**: 2026-03-10  
> **Depends On**: Layer 3 (Execution Platform) — via protocol abstractions only

---

## 1. Purpose

The Orchestrator Layer is the **strategy brain** of rh_cognitiv. It receives high-level intent (from the Cognitive Layer or directly from user code) and translates it into concrete execution plans expressed as DAGs. It:

- Builds **Plan DAGs** — the declared intent of what should happen
- Produces **Execution DAGs** — the runtime graph that records what actually happened
- Manages **node traversal**, including branching, loops, and parallel fan-out
- Enforces **policies**, **authorization**, and **validation** before each node executes
- Coordinates with the Execution Platform through **adapters** (never directly)
- Integrates with **ExecutionState** for time-travel (undo/redo) at the orchestration level

This layer **never** performs actual execution (no LLM calls, no function calls). It delegates everything downward through the L2→L3 adapter.

---

## 2. Component Architecture

```
orchestrators/
├── protocols.py          # ABCs — OrchestratorProtocol, NodeProtocol, DAGProtocol
├── models.py             # Pydantic models — DagNode, Edge, DAGConfig, etc.
│
├── nodes.py              # BaseNode, ExecutionNode, FlowNode hierarchies
├── flow_nodes.py         # ForEach, Filter, Switch, Get, IfNotOk
├── plan_dag.py           # PlanDAG — static intent graph
├── execution_dag.py      # ExecutionDAG — runtime graph (what happened)
│
├── dag_orchestrator.py   # DAGOrchestrator — core traversal engine
├── validation.py         # Policy enforcement, authorization, input validation
└── adapters.py           # L2→L3 adapter — translates nodes into ExecutionEvents
```

### Relationship to Layer 3

```
Orchestrator Layer                         Execution Platform (Layer 3)
──────────────────                         ──────────────────────────────

DAGOrchestrator.run_node(node, data)
        │
        ├── validation.validate(node, data)
        │
        ├── adapter.to_event(node)  ──────► ExecutionEvent
        │                                        │
        │                                   EventBus.emit()
        │                                   Handler.execute()
        │                                   State.snapshot()
        │                                        │
        ├── ◄──────────────────────────── ExecutionResult
        │
        ├── execution_dag.record(node, result)
        │
        └── run_next(result)
```

---

## 3. The Dual DAG Model

### Plan DAG — What Should Happen
- **Static** — defined before execution begins (by the Cognitive Layer or user code)
- Represents the **intended** flow: "do A, then B, then if X do C else D"
- Nodes are **abstract** — they describe what to do, not how
- Can be serialized, versioned, and compared

### Execution DAG — What Actually Happened
- **Dynamic** — built at runtime as nodes execute
- Records **real** execution order, including retries, fallbacks, and loop iterations
- Each node links to: the Plan DAG node it came from, the ExecutionResult, timing, and trace
- This is what powers time-travel: "show me what happened at step 5 and let me go back to step 3"

### Why Two DAGs?

| Concern | Plan DAG | Execution DAG |
|---------|----------|---------------|
| When built | Before execution | During execution |
| Mutability | Immutable once submitted | Append-only |
| Purpose | Intent, scheduling, optimization | Audit, replay, undo/redo |
| Branching meaning | "these could happen in parallel" | "these DID happen in parallel" |
| Loop representation | Single node with loop config | N nodes (one per iteration) |

---

## 4. Design Decisions

### DD-L2-01: Node Type Hierarchy

**Issue**: Nodes represent units of work in the DAG. They span very different behaviors: LLM calls, data transformations, loops, conditionals. How should the type hierarchy be structured?

**Options**:

#### Option A — Two Base Types: ExecutionNode + FlowNode
As sketched in the original SPEC. ExecutionNode produces results (LLM, function, tool). FlowNode controls flow (ForEach, Filter, Switch).

```python
class BaseNode: ...
class ExecutionNode(BaseNode): ...  # TextNode, DataNode, FunctionNode, ToolNode
class FlowNode(BaseNode): ...      # ForEach, Filter, Switch, Get, IfNotOk
```

| Pros | Cons |
|------|------|
| Clear semantic split (do vs. control) | Some nodes might be both (a "retry node"?) |
| Easy to validate (execution nodes need configs, flow nodes need children) | Two parallel hierarchies to maintain |
| Matches the mental model | |

#### Option B — Single Hierarchy with Capabilities
One `Node` base with capability flags or mixins.

```python
class Node:
    can_execute: bool
    can_branch: bool
    can_loop: bool
```

| Pros | Cons |
|------|------|
| Flat hierarchy | Capability combinations can be ambiguous |
| Flexible | Validation is harder (which combos are valid?) |
| | Doesn't convey intent as clearly |

#### Option C — Tagged Union with Discriminator
Node is a union type discriminated by `kind`, with Pydantic's discriminated unions.

```python
Node = Annotated[
    ExecutionNode | ForEachNode | FilterNode | SwitchNode | ...,
    Field(discriminator="kind")
]
```

| Pros | Cons |
|------|------|
| Type-safe dispatch in Pydantic | Every node type is a separate class |
| Serialization-friendly | No shared behavior via inheritance |
| Exhaustive matching via kind | Need a common protocol for shared operations |
| Natural for declarative definitions (future) | |

**Suggestion**: **Option A** as the runtime hierarchy (for behavior and shared logic) + **Option C** for serialization/deserialization (so DAGs can be saved/loaded declaratively). The `kind` discriminator lives on `BaseNode` and drives Pydantic union parsing.

> **DECISION**: `___________`  
> **Comments**: `___________`

---

### DD-L2-02: DAG Representation

**Issue**: How is the DAG data structure represented? Adjacency list, edge list, matrix? What traversal model?

**Options**:

#### Option A — Edge List (as in original SPEC)
```python
dag = {
    "nodes": {"n1": Node(...), "n2": Node(...)},
    "edges": [{"from": "n1", "to": "n2"}]
}
```

| Pros | Cons |
|------|------|
| Simple, serializable | Traversal requires scanning all edges |
| Matches the SPEC sketch | No built-in adjacency lookup |
| Easy to understand | Need to build adjacency index for performance |

#### Option B — Adjacency List (Dict of Node → List[Node])
```python
dag = {
    "n1": Node(..., successors=["n2", "n3"]),
    "n2": Node(..., successors=["n4"]),
}
```

| Pros | Cons |
|------|------|
| O(1) successor lookup | Harder to serialize neatly |
| Natural for traversal | Node definition mixed with graph structure |
| Efficient for large DAGs | Predecessor lookup needs reverse index |

#### Option C — Graph Library (networkx or custom lightweight)
Use a proper graph class with typed node/edge data.

```python
class DAG:
    _graph: dict[str, list[str]]  # adjacency
    _nodes: dict[str, Node]       # node data

    def successors(self, node_id: str) -> list[str]: ...
    def predecessors(self, node_id: str) -> list[str]: ...
    def topological_order(self) -> list[str]: ...
    def has_next(self) -> bool: ...
```

| Pros | Cons |
|------|------|
| Clean API, encapsulated graph logic | More code upfront |
| Can validate DAG properties (acyclic, connected) | |
| Supports topological sort for scheduling | |
| Serializable (export to edge list or adjacency) | |

**Suggestion**: **Option C** — a custom lightweight `DAG` class. It wraps adjacency data internally (edge list for serialization, adjacency for traversal) and exposes clean methods. Don't take `networkx` as a dependency — the DAG operations needed are small enough to implement directly.

> **DECISION**: `___________`  
> **Comments**: `___________`

---

### DD-L2-03: Orchestrator Traversal Strategy

**Issue**: How does `DAGOrchestrator` decide which node to execute next? The SPEC shows a linear `run_next` recursive loop, but real DAGs can have parallel branches, conditional paths, and dynamic modifications.

**Options**:

#### Option A — Linear Recursive (as in SPEC)
Follow topological order. One node at a time. `run_next()` calls itself recursively.

| Pros | Cons |
|------|------|
| Simplest to implement | No parallelism |
| Easy to reason about | Stack depth limits on large DAGs |
| Deterministic ordering | Doesn't exploit independence between branches |

#### Option B — Topological Sort + Ready Queue
Compute topological order. Maintain a "ready queue" of nodes whose dependencies are satisfied. Execute from the queue (parallel when multiple are ready).

```python
async def run(self):
    ready = self.dag.get_initial_nodes()
    while ready:
        results = await asyncio.gather(*[self.run_node(n) for n in ready])
        ready = self.dag.get_newly_ready_nodes(results)
```

| Pros | Cons |
|------|------|
| Natural parallelism for independent branches | More complex state tracking |
| Efficient use of async | Need to manage partial failure in parallel batch |
| Matches how real DAG schedulers work | Ordering within a batch is non-deterministic |
| Scales to complex DAGs | |

#### Option C — Event-Driven Traversal
Each completed node emits an event. A scheduler listens and queues the next nodes.

| Pros | Cons |
|------|------|
| Fully decoupled | Harder to reason about completion |
| Natural fit with EventBus | Debugging event-driven flows is tricky |
| Supports dynamic DAG modification | Over-engineered for straightforward DAGs |

**Suggestion**: **Option B** — Ready Queue. Start with sequential execution within the queue (process one at a time) for Stage 1. Add `asyncio.gather` for parallel branches in Stage 2. The data structure supports both without changes.

> **DECISION**: `___________`  
> **Comments**: `___________`

---

### DD-L2-04: Adapter L2→L3 Design

**Issue**: The orchestrator must translate its node types into Execution Platform events. How is this adapter structured?

**Options**:

#### Option A — Single Adapter Class with Method per Node Type
```python
class ExecutionAdapter:
    def to_event(self, node: BaseNode) -> ExecutionEvent:
        if isinstance(node, TextNode): return TextExecutionEvent(...)
        elif isinstance(node, DataNode): return DataExecutionEvent(...)
```

| Pros | Cons |
|------|------|
| Simple, centralized | Violates OCP — every new node type means editing this class |
| Easy to find all translations | Can become a large if/elif chain |

#### Option B — Adapter Registry (Strategy per Node Kind)
```python
class AdapterRegistry:
    _adapters: dict[str, NodeAdapter] = {}

    def register(self, kind: str, adapter: NodeAdapter): ...
    def to_event(self, node: BaseNode) -> ExecutionEvent:
        return self._adapters[node.kind].adapt(node)
```

| Pros | Cons |
|------|------|
| OCP-compliant — new types register without editing existing code | Indirection |
| Each adapter is independently testable | Must ensure all types are registered |
| Supports runtime registration (plugins) | |

#### Option C — Adapter Method on Node (node.to_event())
Each node knows how to convert itself to an event.

| Pros | Cons |
|------|------|
| Co-located logic | Node now depends on Layer 3 types (violates DI!) |
| | Mixes orchestrator and execution concerns |

**Suggestion**: **Option B** — Adapter Registry. It respects OCP and DIP. The orchestrator depends on the `NodeAdapter` protocol, not on Layer 3 directly. Each concrete adapter is a small class that knows both the node type and the event type.

> **DECISION**: `___________`  
> **Comments**: `___________`

---

### DD-L2-05: FlowNode Execution Model

**Issue**: FlowNodes (ForEach, Filter, Switch, Get) don't execute work — they control flow. How do they interact with the orchestrator and the execution platform?

**Options**:

#### Option A — FlowNodes Handled Entirely in the Orchestrator
The orchestrator recognizes FlowNodes and handles them with special logic before/instead of sending to L3.

```python
if isinstance(node, ForEachNode):
    results = []
    for item in data:
        results.append(self.run_node(node.inner_node, item))
    return results
```

| Pros | Cons |
|------|------|
| No L3 involvement for flow control | Orchestrator code grows complex |
| Clear separation — L3 only does real work | Flow logic is mixed into the traversal loop |
| Simpler L3 | ForEach within ForEach gets recursive and messy |

#### Option B — FlowNodes are L3 Events Too
FlowNodes are translated into special execution events that L3 handles.

| Pros | Cons |
|------|------|
| Uniform model — everything is an event | L3 now knows about flow control (leaking L2 concepts) |
| Tracing captures flow decisions | Blurs the layer boundary |
| | Over-engineering — flow control isn't "execution" |

#### Option C — FlowNodes as Orchestrator Middleware
FlowNodes are modular handlers that the orchestrator invokes before deciding the next node. They modify the DAG traversal state (expand nodes, skip branches, etc.).

```python
class ForEachHandler(FlowHandler):
    def handle(self, node: ForEachNode, data, dag_state) -> list[NodeRef]:
        """Expand the inner node for each data item and return new node refs."""
        expanded = [node.inner_node.with_input(item) for item in data]
        dag_state.expand_parallel(expanded)
        return expanded
```

| Pros | Cons |
|------|------|
| Clean separation — flow handlers modify traversal, not execution | Requires a `DAGState` concept for dynamic traversal |
| Each FlowNode type has its own handler | More abstraction |
| Orchestrator loop stays simple | |
| ExecutionDAG captures the expansion | |

**Suggestion**: **Option C** — FlowNodes as orchestrator middleware. This keeps L3 clean (it only executes real work), keeps the orchestrator loop simple (it delegates flow decisions to handlers), and produces a clean ExecutionDAG (expanded nodes are recorded).

> **DECISION**: `___________`  
> **Comments**: `___________`

---

### DD-L2-06: Time-Travel Undo/Redo Granularity

**Issue**: At what level does undo/redo operate in the orchestrator? Per-node? Per-branch? Per-DAG?

**Options**:

#### Option A — Per-Node Granularity
Each node execution is an undoable step. Undo rolls back to before that node ran.

| Pros | Cons |
|------|------|
| Fine-grained control | Expensive — snapshot per node |
| Users can precisely target what to undo | Parallel branches make ordering ambiguous |
| Natural mapping to ExecutionDAG nodes | "Undo node 5" in a parallel branch — what happens to node 6? |

#### Option B — Per-Checkpoint Granularity
The orchestrator defines explicit checkpoints (e.g., after each "phase" or user-defined points). Undo returns to the previous checkpoint.

| Pros | Cons |
|------|------|
| Controlled snapshot cost | Less precise than per-node |
| User chooses meaningful rollback points | Users must define checkpoints |
| Clear semantics for parallel branches | |

#### Option C — Hybrid: Auto-Checkpoint Per Node + Named Checkpoints
Every node creates a lightweight snapshot (delta). Users can also create named checkpoints (full snapshot). Undo navigates either.

| Pros | Cons |
|------|------|
| Fine-grained when needed, efficient when not | Most complex to implement |
| Named checkpoints for meaningful rollback points | Two snapshot mechanisms |
| Deltas keep per-node cost low | |

**Suggestion**: **Option C** — but staged. Stage 1: per-node full snapshots (simple, correct). Stage 2: replace with deltas. Stage 3: add named checkpoints. This follows the snapshot strategy from DD-L3-02.

> **DECISION**: `___________`  
> **Comments**: `___________`

---

### DD-L2-07: Parallel Execution in ForEach

**Issue**: When a ForEach node fans out over N items, how does parallel execution work? What happens when one branch fails?

**Options**:

#### Option A — All-or-Nothing (gather with fail-fast)
Use `asyncio.gather(return_exceptions=False)`. If any branch fails, cancel all others and fail the ForEach.

| Pros | Cons |
|------|------|
| Simple error semantics | One failure kills all branches (wasteful) |
| Clean rollback | Partial results are lost |
| Deterministic | |

#### Option B — Collect All Results (gather with return_exceptions)
Use `asyncio.gather(return_exceptions=True)`. All branches run to completion. Failed branches return error results.

| Pros | Cons |
|------|------|
| Maximum work completed | Must handle partial success/failure |
| No wasted computation | More complex error handling |
| Failed branches can be retried individually | What does "ForEach result" mean with mixed ok/fail? |

#### Option C — Configurable per ForEach Node
Each ForEach node specifies its failure strategy: fail-fast, collect-all, or fail-after-N.

```python
class ForEachNode(FlowNode):
    failure_strategy: Literal["fail_fast", "collect_all", "fail_after_n"]
    max_failures: int | None = None
```

| Pros | Cons |
|------|------|
| Flexible — right strategy for the use case | More config surface |
| Users choose the behavior | Must implement all strategies |
| | "fail_after_n" adds a counter |

**Suggestion**: **Option C** — configurable, with `fail_fast` as default (safest). Implement `fail_fast` and `collect_all` for Stage 1. Add `fail_after_n` later.

> **DECISION**: `___________`  
> **Comments**: `___________`

---

## 5. General Design Insights

### DI-L2-01: ExecutionDAG Should Be Append-Only

**Issue**: If the Execution DAG can be modified (nodes removed, reordered), time-travel breaks. It must be a faithful record of what happened.

**Suggestion**: Make ExecutionDAG append-only. New nodes can be added. Undo doesn't delete nodes — it marks them as `rolled_back` and creates a new branch. This preserves the full history.

```python
class ExecutionDAGEntry:
    node_id: str
    plan_node_ref: str          # which PlanDAG node this came from
    status: NodeExecutionStatus  # success, failed, rolled_back, skipped
    result_ref: EntryRef | None
    started_at: Timestamp
    completed_at: Timestamp | None
    parent_entry_id: str | None  # for branching (undo creates a new branch)
```

> **DECISION**: `___________`  
> **Comments**: `___________`

---

### DI-L2-02: Validation Should Be a Pipeline, Not a Gate

**Issue**: The SPEC mentions `validate_node_step(node, main_input)` as a pre-execution check. If validation is a single function, it becomes a monolith.

**Suggestion**: Make validation a pipeline of composable validators, similar to the policy chain in L3.

```python
class ValidationPipeline:
    validators: list[NodeValidator]

    async def validate(self, node, data, state) -> ValidationResult:
        for v in self.validators:
            result = await v.validate(node, data, state)
            if not result.ok:
                return result
        return ValidationResult.ok()

# Built-in validators:
# - InputSchemaValidator: checks data matches node's expected input
# - AuthorizationValidator: checks policy/permissions
# - DependencyValidator: checks all upstream nodes completed
# - BudgetValidator: checks remaining budget is sufficient
```

> **DECISION**: `___________`  
> **Comments**: `___________`

---

### DI-L2-03: PlanDAG Immutability Contract

**Issue**: If the Cognitive Layer can modify the PlanDAG while the Orchestrator is executing it, you get race conditions and broken invariants.

**Suggestion**: PlanDAG is **frozen** once submitted to the orchestrator. If the Cognitive Layer wants to change the plan mid-execution, it submits a **new** PlanDAG (or a patch) through a defined "replan" protocol. The orchestrator decides whether to accept the replan at the next safe checkpoint.

> **DECISION**: `___________`  
> **Comments**: `___________`

---

### DI-L2-04: Consider a DAGBuilder for Ergonomics

**Issue**: Constructing DAGs manually (nodes dict + edges list) is verbose and error-prone. For V1 (imperative) and especially V2 (declarative), a builder pattern improves usability.

**Suggestion**: Provide a `DAGBuilder` fluent API alongside the raw data structure.

```python
dag = (
    DAGBuilder("my-pipeline")
    .add_node("extract", DataNode(prompt="Extract entities"))
    .add_node("transform", FunctionNode(fn=clean_data))
    .add_node("summarize", TextNode(prompt="Summarize results"))
    .edge("extract", "transform")
    .edge("transform", "summarize")
    .build()  # returns a frozen PlanDAG
)
```

> **DECISION**: `___________`  
> **Comments**: `___________`

---

## 6. Implementation Priority

| Priority | Component | Rationale |
|----------|-----------|-----------|
| **P0** | `protocols.py` + `models.py` | Contracts for L1 and adapter |
| **P0** | `nodes.py` + `flow_nodes.py` | The vocabulary of the DAG |
| **P0** | `plan_dag.py` | The core data structure |
| **P1** | `dag_orchestrator.py` (linear traversal) | Core loop, even before parallelism |
| **P1** | `adapters.py` (L2→L3) | The bridge to the Execution Platform |
| **P1** | `execution_dag.py` | Runtime recording |
| **P2** | `validation.py` | Safety rails |
| **P2** | Parallel execution (ForEach) | After linear works correctly |
| **P3** | Time-travel / undo-redo integration | After snapshot system is proven |

---

## 7. Testing Strategy

| Level | What | How |
|-------|------|-----|
| **Unit** | Node types, DAG construction, traversal | Mock L3 via protocols |
| **Unit** | FlowNode handlers (ForEach, Switch) | Verify expansion logic |
| **Unit** | Adapter registry | Mock handlers, verify dispatch |
| **Integration** | DAGOrchestrator → Adapter → L3 Events | Real L3 with in-memory store |
| **Integration** | ExecutionDAG recording | Run a DAG, verify the execution trace |
| **Scenario** | Linear DAG, branching DAG, ForEach DAG, failing DAG | Full stack with assertions on results + ExecutionDAG shape |
| **Property** | DAG validity | Hypothesis: random DAG generation, verify topological properties |

---

## 8. Open Questions

- [ ] Should the PlanDAG support sub-graphs (nested DAGs / composite nodes)?
- [ ] How does the orchestrator handle external events (user interrupts, external API callbacks)?
- [ ] Should nodes carry their own timeout/retry config, or inherit from the orchestrator?
- [ ] What's the replan protocol — full DAG replacement, or incremental patch?
- [ ] How are node inputs/outputs typed? Generic `BaseModel`, or schema-per-node?
