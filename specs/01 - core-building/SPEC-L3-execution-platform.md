# Layer 3 — Execution Platform: Development Specification

> **Parent**: [SPEC-overview.md](SPEC-overview.md)  
> **Layer**: 3 (Foundation)  
> **Status**: Ready for Implementation — All decisions finalized  
> **Last Updated**: 2026-03-11

---

## 1. Purpose

The Execution Platform is the **bottom layer** of rh_cognitiv. It is a **self-contained**, **reusable** runtime engine that knows nothing about orchestration strategy or cognitive reasoning. It provides:

- A **unified event system** for all types of execution (LLM calls, function calls, tool invocations)
- **Immutable state snapshots** enabling time-travel, undo/redo
- A **context store** for memories and artifacts (with separate serialization strategies)
- **Cross-cutting concerns**: logging, tracing, retry, budget, timeout, interrupts
- **Error handling** with a categorized hierarchy (retryable vs fatal)
- **Budget tracking** as a first-class resource

Upper layers depend on this layer's **abstractions** (protocols/ABCs), never on concrete implementations.

---

## 2. Component Architecture

```
execution_platform/
├── protocols.py          # All ABCs — the contracts upper layers depend on
├── models.py             # Pydantic models — BaseEntry, Memory, Artifact, etc.
├── errors.py             # Error hierarchy (categorized with recoverability traits)
├── types.py              # ID, Timestamp, Ext, EntryRef[T]
│
├── event_bus.py          # EventBus — hybrid sync middleware + async subscribers
├── events.py             # ExecutionEvent base + Text, Data, Function, Tool
├── handlers.py           # Handler registry + concrete handlers per event kind
├── state.py              # ExecutionState — full snapshot chain (V1)
├── context_store.py      # ContextStore — file-based backend (V1)
├── memory_store.py       # Memory-specific logic and serialization
├── artifact_store.py     # Artifact-specific logic and format-diverse serialization
├── policies.py           # PolicyChain + Retry, Budget, Timeout policies
├── budget.py             # BudgetTracker — first-class resource
├── log_collector.py      # Structured log collection (EventBus subscriber)
└── trace_collector.py    # Execution trace recording (EventBus subscriber)
```

### Dependency Graph (Internal)

```
protocols.py ◄─── everything depends on this
     │
     ├── models.py (implements protocol shapes)
     ├── errors.py (standalone)
     ├── types.py  (standalone)
     │
     ├── event_bus.py      ← uses protocols, types
     ├── events.py         ← uses protocols, types, errors
     ├── handlers.py       ← uses protocols, types, errors, events
     ├── state.py          ← uses protocols, types, models
     ├── context_store.py  ← uses protocols, types, models
     ├── memory_store.py   ← uses protocols, models, context_store
     ├── artifact_store.py ← uses protocols, models, context_store
     ├── budget.py         ← uses types (standalone logic)
     ├── policies.py       ← uses protocols, errors, budget
     ├── log_collector.py  ← uses protocols, event_bus
     └── trace_collector.py← uses protocols, event_bus
```

---

## 3. Design Decisions

### DD-L3-01: Event System — Hybrid Sync Middleware + Async Subscribers

**Decision**: Hybrid model — sync middleware pipeline + async fan-out subscribers.

State management and logging run as **synchronous, ordered middleware** (deterministic, required for replay). Non-critical consumers (metrics, front-end SSE push) run as **async subscribers** that receive events as they are cast.

```python
bus.use(logging_middleware)      # sync, runs in order
bus.use(trace_middleware)        # sync, runs in order
bus.on_async(MetricsCollector)   # async, fire-and-forget
bus.on_async(SSEFrontendPush)    # async, real-time streaming to UI
bus.emit(event)                  # middleware runs sync, then fans out async
```

**Key constraint**: Async subscribers receive events in real-time as they are emitted. This enables use cases like Server-Sent Events (SSE) to a front-end, allowing users to watch execution progress live.

---

### DD-L3-02: Immutable State — Full Snapshot (V1)

**Decision**: Full deep-copy snapshot on every state transition.

Each state change produces a complete deep copy of the entire state. Simple, correct, random-access to any point in time.

**Evolution path**: V1 (full snapshots) → V2 (delta chains via jsonpatch) → V3 (hybrid: periodic full + deltas). jsonpatch is already a dependency, making the evolution natural.

---

### DD-L3-03: Error Hierarchy — Categorized with Recoverability Traits

**Decision**: Categorized exception hierarchy with `retryable` flag and `category` metadata.

```python
class ErrorCategory(str, Enum):
    TRANSIENT = "transient"       # network blip, rate limit
    PERMANENT = "permanent"       # invalid input, auth failure
    BUDGET = "budget"             # token/call/time budget exceeded
    INTERRUPT = "interrupt"       # user cancellation, human-in-the-loop
    ESCALATION = "escalation"     # needs human decision

class CognitivError(Exception):
    retryable: bool
    category: ErrorCategory
    attempt: int = 0             # which retry attempt produced this
    original: Exception | None   # wrapped root cause

class TransientError(CognitivError):     # retryable=True, category=TRANSIENT
class PermanentError(CognitivError):     # retryable=False, category=PERMANENT
class BudgetError(PermanentError):       # category=BUDGET
class InterruptError(PermanentError):    # category=INTERRUPT
class EscalationError(CognitivError):    # category=ESCALATION, retryable=False
class LLMTransientError(TransientError): ...
class TimeoutError(TransientError): ...
class ValidationError(PermanentError): ...
```

The `retryable` flag and `category` drive the PolicyChain's decision logic: retry, cancel, escalate, etc. `EscalationError` signals that a human decision is needed and the task is waiting.

**Future note**: `Result[T, E]` types will be introduced at internal boundaries (e.g., `ExecutionResult`) when the pattern is needed, but exceptions remain the primary error mechanism.

---

### DD-L3-04: ContextStore Backend — File-Based (V1)

**Decision**: File-based store (JSON files on disk) for V1.

Each entry is a JSON file. Directory structure mirrors the store layout. Human-readable, debuggable, no external dependencies.

**Evolution path**: V1 (file-based) → V2 (SQLite for structured queries and concurrency) → V3+ (vector store for semantic recall).

The `ContextStore` protocol abstraction means backends are swappable without touching consumers.

---

### DD-L3-05: ExecutionEvent Handlers — Strategy Pattern with Generic Results

**Decision**: Events are data-only. External handlers are registered in a handler registry (Strategy pattern). 

```python
class ExecutionEvent:
    kind: EventKind        # TEXT, DATA, FUNCTION, TOOL
    payload: EventPayload  # kind-specific data

class HandlerRegistry:
    def register(self, kind: EventKind, handler: EventHandler[T]): ...
    async def handle(self, event: ExecutionEvent, data, configs) -> ExecutionResult[T]: ...
```

`ExecutionResult[T]` is generic, where `T` carries kind-specific result data:

```python
class ExecutionResult(Generic[T]):
    ok: bool
    value: T | None           # kind-specific payload
    error: CognitivError | None
    metadata: ResultMetadata   # timing, attempt count, etc.

# Kind-specific payloads:
class LLMResultData:
    text: str
    thinking: str | None
    token_usage: TokenUsage    # prompt_tokens, completion_tokens, total
    model: str
    finish_reason: str

class FunctionResultData:
    return_value: Any
    duration_ms: float

class ToolResultData:
    llm_result: LLMResultData       # the LLM call that chose the tool
    function_result: FunctionResultData  # the tool execution
```

Handlers are independently testable, swappable at runtime, and composable with the PolicyChain.

---

### DD-L3-06: Logging & Tracing — Separate EventBus Subscribers

**Decision**: Logging and tracing are **separate EventBus subscribers** that produce different output formats.

- **LogCollector**: human-readable structured logs (JSON lines), optimized for debugging
- **TraceCollector**: machine-readable spans (execution ID, span ID, parent span, timing), optimized for performance analysis and replay

Both subscribe to the same EventBus events. Both carry the same execution context (execution ID, node ID). They are decoupled from each other and from the event system — adding a new observer (e.g., metrics exporter) means registering another subscriber.

---

### DD-L3-07: Retry & Policy — Middleware Chain

**Decision**: Policies are chainable middleware (PolicyChain) that wrap handler execution with `before_execute` / `after_execute` / `on_error` hooks.

```python
chain = PolicyChain([
    BudgetPolicy(tracker=budget_tracker),  # check budget before, consume after
    TimeoutPolicy(seconds=30),
    RetryPolicy(max_attempts=3, backoff=exponential),
])
result = await chain.execute(handler, event, data, configs)
```

Policies are configurable **per-node** via `execution_configs()`, dynamic at runtime, and composable. Each policy is a standalone component with self-contained logic.

The `BudgetTracker` (see DI-L3-03) is a standalone resource that policies query for decisions and executors update with consumption data. It is **not** embedded in configs — configs hold a reference to it.

---

### DD-L3-08: EntryRef — Typed Lazy Reference (V1)

**Decision**: `EntryRef[T]` carries type information and resolves lazily.

```python
class EntryRef(Generic[T]):
    id: ID
    entry_type: type[T]
    _resolved: T | None = PrivateAttr(default=None)

    def resolve(self, store: ContextStore) -> T: ...
```

**Evolution path**: V1 (typed lazy ref) → V2 (add `snapshot_version: int` for time-travel-bound resolution). The upgrade is additive — no breaking changes.

---

## 4. Design Decisions — Architecture

### DI-L3-01: ContextStore and ExecutionState Are Peer Services

**Decision**: `ContextStore` and `ExecutionState` are **independent systems**. ExecutionState holds `EntryRef`s to memories/artifacts, not the entries themselves.

- **ContextStore** manages long-lived entities (memories, artifacts) that survive across execution runs and crashes
- **ExecutionState** manages ephemeral execution snapshots scoped to a single run
- They are coordinated through explicit sync points, not containment

Snapshots stay lightweight (refs only). Context can be shared across multiple execution runs.

---

### DI-L3-02: Event Lifecycle Protocol

**Decision**: Every event follows an explicit lifecycle expressed as `EventStatus`. Lifecycle transitions are emitted through the EventBus, giving tracing and logging their hook points.

```python
class EventStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ESCALATED = "escalated"     # human-in-the-loop: awaiting user decision
    WAITING = "waiting"         # paused, waiting for external input
```

`ESCALATED` indicates that a decision from the user is needed and the task is paused until it arrives. This is the hook for human-in-the-loop patterns.

---

### DI-L3-03: Budget as a First-Class Standalone Resource

**Decision**: Budget is a standalone component with self-contained logic. Executors consume it for decisions and write to it for updates. It is NOT embedded in configs — configs hold a reference to the tracker.

```python
class BudgetTracker:
    token_budget: int
    token_used: int
    call_budget: int
    calls_made: int
    time_budget_seconds: float
    elapsed_seconds: float

    def can_proceed(self) -> bool: ...
    def consume(self, tokens: int = 0, calls: int = 0) -> None: ...
    def remaining(self) -> BudgetSnapshot: ...
    def is_exceeded(self) -> bool: ...
```

The `BudgetPolicy` in the PolicyChain queries `BudgetTracker.can_proceed()` before execution and calls `BudgetTracker.consume()` after. Handlers never touch the tracker directly — all budget logic lives inside the tracker and the policy.

---

### DI-L3-04: Memory and Artifact — Separate Logic, Shared Store

**Decision**: Memory and Artifact have **separate logic layers** even though both are backed by the same `ContextStore` for unified reference and querying.

**Why**: Artifacts can be any format (code files, binary data, structured JSON, images) and need format-diverse serialization. Memories are primarily text-based and serialize in a more uniform, LLM-friendly way.

```
ContextStore (unified interface for reference, query, lifecycle)
     │
     ├── MemoryStore (memory-specific logic)
     │   ├── Serialization: text-optimized (Markdown, structured text)
     │   ├── Retrieval: role-based, tag-based, semantic (future)
     │   └── Lifecycle: decay, consolidation, working-memory expiry
     │
     └── ArtifactStore (artifact-specific logic)
         ├── Serialization: format-diverse (JSON, code, binary, Markdown)
         │   Uses content.format field to select serializer
         ├── Retrieval: slug-based, version-based, type-based
         └── Lifecycle: versioning (supersedes chain), status transitions
```

Both stores register with the `ContextStore` protocol. Upper layers can query across both (via `MemoryQuery.kind`) or target one specifically. The format field (`content.format`: `text/plain`, `text/markdown`, `code/python`, `data/json`, `binary/...`) drives serializer selection for artifacts.

---

## 5. Implementation Phases

Development is organized into phases based on dependency order and importance.  
Each phase produces **testable, working code** before the next begins.

### Phase 1 — Foundation Types & Protocols
> **Goal**: Establish all contracts and types that everything else depends on.  
> **No runtime behavior yet — just shapes.**

| Step | File | Deliverable |
|------|------|-------------|
| 1.1 | `types.py` | `ID` (ULID), `Timestamp` (ISO-8601), `Ext` (dict), type aliases |
| 1.2 | `models.py` | Pydantic models: `EntryContent`, `BaseEntry`, `Memory`, `Artifact`, `MemoryRole`, `MemoryShape`, `MemoryOrigin`, `ArtifactType`, `ArtifactStatus`, `TokenUsage`, `ResultMetadata`, `LLMResultData`, `FunctionResultData`, `ToolResultData` |
| 1.3 | `errors.py` | `ErrorCategory` enum, `CognitivError` base, `TransientError`, `PermanentError`, `BudgetError`, `InterruptError`, `EscalationError`, `LLMTransientError`, `TimeoutError`, `ValidationError` |
| 1.4 | `protocols.py` | ABCs: `EventBusProtocol`, `EventHandlerProtocol[T]`, `HandlerRegistryProtocol`, `ExecutionStateProtocol`, `ContextStoreProtocol`, `PolicyProtocol`, `PolicyChainProtocol`, `LogCollectorProtocol`, `TraceCollectorProtocol`, `BudgetTrackerProtocol`, `SnapshotSerializerProtocol` |
| 1.5 | Tests | Unit tests for all models (Pydantic validation, serialization roundtrip), error hierarchy (`isinstance` checks, retryable flag) |

### Phase 2 — Event System & Budget
> **Goal**: EventBus (the backbone) + BudgetTracker (standalone resource). These have no dependencies beyond Phase 1.

| Step | File | Deliverable |
|------|------|-------------|
| 2.1 | `event_bus.py` | `EventBus` — sync middleware pipeline + async subscriber fan-out. `EventStatus` enum. `use()`, `on_async()`, `emit()`, `wait_for()` methods |
| 2.2 | `budget.py` | `BudgetTracker` — `can_proceed()`, `consume()`, `remaining()`, `is_exceeded()`. Self-contained, no EventBus dependency |
| 2.3 | Tests | EventBus: middleware ordering, async subscriber delivery, lifecycle event emission, `wait_for()` blocking subscribe. BudgetTracker: consumption tracking, overflow detection |

### Phase 3 — Events, Handlers & Policies
> **Goal**: The execution engine — events as data, handlers via registry, policies as middleware chain.

| Step | File | Deliverable |
|------|------|-------------|
| 3.1 | `events.py` | `ExecutionEvent` (data-only, `kind` + `payload`), `EventKind` enum (TEXT, DATA, FUNCTION, TOOL), kind-specific payloads, `EscalationRequested` / `EscalationResolved` event types |
| 3.2 | `handlers.py` | `HandlerRegistry`, `EventHandler[T]` ABC, concrete handlers: `TextHandler`, `DataHandler`, `FunctionHandler`, `ToolHandler`. Each returns `ExecutionResult[T]` with kind-specific `T` |
| 3.3 | `types.py` (update) | `EntryRef[T]` — typed lazy reference with `resolve()` |
| 3.4 | `policies.py` | `PolicyChain`, `RetryPolicy`, `TimeoutPolicy`, `BudgetPolicy`. Each with `before_execute()` / `after_execute()` / `on_error()` hooks |
| 3.5 | Tests | Handler registry dispatch, ExecutionResult generics, PolicyChain composition (retry + timeout + budget), policy interaction with BudgetTracker |

### Phase 4 — State Management
> **Goal**: ExecutionState with full-snapshot time-travel.

| Step | File | Deliverable |
|------|------|-------------|
| 4.1 | `state.py` | `ExecutionState` — snapshot chain via deep copy, `add_level()` / `remove_level()`, `snapshot()`, `restore(version)`, `get_current()`, `undo()`, `redo()`, `gc_collect()`. Accepts `serializer: SnapshotSerializer` (default: `JsonSnapshotSerializer`). Snapshots capture `ESCALATED` status + escalation context for cloud recovery |
| 4.2 | Integration | Wire ExecutionState into event lifecycle: snapshot on each state transition, emit lifecycle events through EventBus |
| 4.3 | Tests | Snapshot creation/restore roundtrip, undo/redo sequences, level management, state isolation between snapshots, `gc_collect()` with keep_first/keep_last, JSON serialization roundtrip via `SnapshotSerializer`, ESCALATED state persistence and recovery |

### Phase 5 — Context Store (Memory & Artifact)
> **Goal**: File-based storage with separate memory/artifact logic.

| Step | File | Deliverable |
|------|------|-------------|
| 5.1 | `context_store.py` | `ContextStore` — unified interface (`remember`, `store`, `recall`, `get`, `getArtifact`, `forget`, `consolidate`). File-based backend: one JSON file per entry |
| 5.2 | `memory_store.py` | `MemoryStore` — memory-specific logic: text-optimized serialization, role/tag-based retrieval, working-memory lifecycle |
| 5.3 | `artifact_store.py` | `ArtifactStore` — artifact-specific logic: format-diverse serialization (driven by `content.format`), slug-based retrieval, versioning (supersedes chain) |
| 5.4 | Tests | CRUD operations, recall with filters (kind, role, tags, topK), artifact versioning, format-specific serialization roundtrips |

### Phase 6 — Observability
> **Goal**: Logging and tracing as EventBus subscribers.

| Step | File | Deliverable |
|------|------|-------------|
| 6.1 | `log_collector.py` | `LogCollector` — EventBus subscriber, produces structured JSON log lines, carries execution context (execution ID, node ID) |
| 6.2 | `trace_collector.py` | `TraceCollector` — **opt-in** EventBus subscriber, produces machine-readable spans with OTel-compatible schema (trace_id, span_id, parent_span_id, attributes). Registered only when explicitly configured — zero overhead if unused |
| 6.3 | Tests | Verify log output on event lifecycle transitions, trace span creation/nesting, execution context propagation |

### Phase 7 — Integration & Smoke Tests
> **Goal**: End-to-end validation that all components work together.

| Step | Deliverable |
|------|-------------|
| 7.1 | Full lifecycle test: create event → PolicyChain (budget + retry) → handler → ExecutionResult → state snapshot → log + trace output |
| 7.2 | Time-travel test: execute 5 events, undo 2, verify state matches snapshot at step 3 |
| 7.3 | Budget exhaustion test: run events until budget exceeded, verify BudgetError and graceful stop |
| 7.4 | Escalation test: handler returns ESCALATED status, verify event lifecycle pauses |
| 7.5 | Parallel event test: multiple concurrent events via asyncio, verify no state corruption |

---

## 6. Readiness Checklist

Everything needed to start implementation:

| Concern | Status | Notes |
|---------|--------|-------|
| All design decisions made | **Done** | 8 DDs + 4 DIs + 5 OQs resolved |
| Technology stack | **Done** | Python 3.10+, Pydantic v2, jsonpatch, asyncio |
| Component list complete | **Done** | 13 files in execution_platform/ |
| Dependency order clear | **Done** | 7-phase plan with explicit dependencies |
| Error taxonomy defined | **Done** | Categorized hierarchy with recoverability traits |
| State strategy defined | **Done** | Full snapshot V1, evolution path to delta + hybrid |
| Store backend chosen | **Done** | File-based V1, evolution path to SQLite |
| Event model defined | **Done** | Hybrid sync/async EventBus, strategy-pattern handlers |
| Result model defined | **Done** | `ExecutionResult[T]` with kind-specific generics |
| Budget model defined | **Done** | Standalone `BudgetTracker` |
| Memory/Artifact separation | **Done** | Shared ContextStore, separate logic layers |
| Human-in-the-loop | **Done** | `ESCALATED` / `WAITING` event statuses |
| Open questions resolved | **Done** | All 5 OQs decided — routing, GC, serialization, tracing, escalation |
| Serialization extensibility | **Done** | `SnapshotSerializer` protocol — JSON default, pluggable |
| Cloud-safe escalation | **Done** | State persistence on ESCALATED for stateless recovery |

---

## 7. Testing Strategy

| Level | What | How |
|-------|------|-----|
| **Unit** | Each component in isolation | Mock protocols, test against ABCs |
| **Integration** | EventBus → Events → Handlers → State → Policies | Real components, file-based store |
| **Property** | Snapshot chain integrity | Hypothesis: random operations + verify undo/redo roundtrip |
| **Stress** | Parallel event execution | asyncio + many concurrent events, verify no state corruption |
| **Lifecycle** | Event status transitions | Verify CREATED→RUNNING→SUCCESS/FAILED/ESCALATED paths |

---

## 8. Resolved Open Questions

All open questions have been resolved. Decisions are incorporated into the implementation plan.

---

### OQ-L3-01: EventBus Routing — Type-Based Only (V1)

**Decision**: Type-based dispatch only. Subscribers register for event classes; any finer-grained filtering happens inside the handler.

```python
bus.on(ExecutionEvent, handler)  # receives ALL execution events
# handler filters internally: if event.kind == TEXT: ...
```

Simple, sufficient while event type count is small. No topic conventions or filter predicates needed yet.

**Evolution path**: Re-evaluate at Phase 6 (Observability). If SSE/streaming demands granular subscriptions, evolve to type + topic hybrid routing.

---

### OQ-L3-02: Snapshot GC — Keep All + Explicit GC Method

**Decision**: Keep all snapshots during execution (no automatic GC). Cleanup happens on session end. Provide an **explicit `gc_collect()` method** on `ExecutionState` for callers who need manual control.

```python
class ExecutionState:
    def gc_collect(self, keep_first: int | None = None, keep_last: int | None = None) -> int:
        """Manually collect old snapshots. Returns count of removed snapshots.
        
        Args:
            keep_first: Keep the first N snapshots (preserves early history).
            keep_last: Keep the last N snapshots (preserves recent undo ability).
        """
        ...
```

No automatic GC logic, no thinning tiers, no complexity. The caller decides when and how aggressively to prune. This keeps the state manager simple while giving long-running agents an escape hatch.

**Relevant phase**: Phase 4 (State Management) — add `gc_collect()` to `ExecutionState` and `ExecutionStateProtocol`.

---

### OQ-L3-03: Snapshot Serialization — JSON via SnapshotSerializer Protocol

**Decision**: JSON for V1, implemented behind a `SnapshotSerializer` protocol. JSON is the default implementation; the protocol is the **source of truth** — anyone can implement additional serializers (MessagePack, CBOR, custom binary, etc.).

```python
class SnapshotSerializer(Protocol):
    """Source of truth for snapshot serialization. Implement to add formats."""
    def serialize(self, state: dict) -> bytes: ...
    def deserialize(self, data: bytes) -> dict: ...

class JsonSnapshotSerializer:
    """Default implementation — JSON via Pydantic."""
    def serialize(self, state: dict) -> bytes:
        return json.dumps(state).encode()
    def deserialize(self, data: bytes) -> dict:
        return json.loads(data)
```

The `ExecutionState` accepts a `SnapshotSerializer` at construction. New backends are pluggable without touching state logic.

**Relevant phases**:
- Phase 1: Add `SnapshotSerializerProtocol` to `protocols.py`
- Phase 4: `ExecutionState` accepts `serializer: SnapshotSerializer` parameter. Ship `JsonSnapshotSerializer` as default.

---

### OQ-L3-04: Trace Format — Custom OTel-Compatible Schema (Optional)

**Decision**: Custom span model with OpenTelemetry-compatible field names. No OTel SDK dependency. Tracing is **optional** — the `TraceCollector` is an opt-in EventBus subscriber, not a required component.

```python
class Span:
    trace_id: str         # maps to OTel trace_id
    span_id: str          # maps to OTel span_id
    parent_span_id: str | None
    name: str             # "TextHandler.execute"
    status: SpanStatus    # OK, ERROR
    start_time: Timestamp
    end_time: Timestamp | None
    attributes: dict[str, str | int | float | bool]  # OTel-compatible types
    events: list[SpanEvent]  # OTel-compatible span events
```

Use OTel attribute naming conventions (`execution.node_id`, `event.kind`) from the start. When Jaeger/Grafana integration is needed, writing an OTel exporter is a trivial mapping — the data shape already matches.

**Optional by design**: The execution platform works without tracing enabled. `TraceCollector` is registered via `bus.on_async(TraceCollector)` only when explicitly configured. No performance overhead if not used.

**Relevant phase**: Phase 6 (Observability) — implement as opt-in subscriber.

---

### OQ-L3-05: ESCALATED Communication — Event-Based + Persisted State for Cloud

**Decision**: Event-based round-trip via the EventBus, **with mandatory state persistence** for cloud-safe recovery.

**Local execution (in-process)**:
```python
# Handler escalates:
bus.emit(EscalationRequested(event_id, question, options))

# Front-end receives via SSE, user decides, API emits:
bus.emit(EscalationResolved(event_id, decision="approved"))

# Paused handler resumes:
decision = await bus.wait_for(EscalationResolved, filter=lambda e: e.event_id == event_id)
```

**Cloud deployment (stateless processes)**:
When an `EscalationRequested` event is emitted, the execution state is persisted with `ESCALATED` status, including the escalation context (question, options, event_id). The process can terminate. When the human decision arrives (via API), a new process recovers the persisted state, sees the `ESCALATED` status, injects the decision, and resumes the flow.

```python
# On escalation:
state.snapshot()  # includes ESCALATED status + escalation context
await state_store.persist(execution_id, state)  # durable storage

# On recovery (new process):
state = await state_store.load(execution_id)
if state.status == EventStatus.ESCALATED:
    # Orchestrator knows to inject the human decision and resume
    orchestrator.resume(execution_id, decision=human_input)
```

**Key requirement**: The `EscalationRequested` event payload must include enough context for recovery — the question, options, originating node_id, and any data the handler needs to resume. This is stored as part of the state snapshot.

This adds a `wait_for()` method to the EventBus (targeted extension, not a redesign).

**Relevant phases**:
- Phase 2: Add `wait_for()` to EventBus
- Phase 3: Define `EscalationRequested` / `EscalationResolved` event types
- Phase 4: Ensure state snapshots capture `ESCALATED` status + escalation context
