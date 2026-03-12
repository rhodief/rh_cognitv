# rh_cognitiv — Framework Development Specification

> **Version**: 0.1.0-draft  
> **Status**: Design Phase  
> **Last Updated**: 2026-03-10

---

## 1. Vision & Motivation

**rh_cognitiv** is a cognitive-driven agentic framework built around three strict, independently replaceable layers. The core "why" is **full orchestration control** — enabling harness-ready, reliable agents with rich logging, trace, and **time-travel** (undo/redo) capabilities across every execution path.

### Core Principles

| Principle | Description |
|---|---|
| **Layer Independence** | Each layer (Cognitive → Orchestrator → Execution Platform) can be swapped without breaking the others |
| **Dependency Inversion** | Upper layers depend on abstractions, never on concrete implementations of lower layers |
| **Adapter Pattern** | Cross-layer communication happens exclusively through typed adapters |
| **Immutable State Snapshots** | Every state transition is recorded as an immutable entry, enabling time-travel |
| **Declarative Future** | V1 is imperative skills + DAGs; future versions will support fully declarative orchestration via "metaskills" |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────┐
│         Layer 1: Cognitive Layer            │  ← LLM reasoning, planning, review
│         (Skills, MetaSkills)                │     Communicates ONLY with Layer 2
├────────────── Adapter L1→L2 ────────────────┤
│         Layer 2: Orchestrator Layer         │  ← Strategy, scheduling, policy,
│         (DAG planning + execution flow)     │     DAG building, authorization
├────────────── Adapter L2→L3 ────────────────┤
│         Layer 3: Execution Platform         │  ← Shared, reusable, standalone
│  ┌──────────┬──────────┬─────────────────┐  │
│  │ EventBus │ Memory & │ Log & Traces    │  │
│  │ Retry /  │Artifacts │ Timeouts        │  │
│  │ Budget   │          │ Interrupts      │  │
│  ├──────────┼──────────┼─────────────────┤  │
│  │ Event    │ Context  │ Execution       │  │
│  │ Executors│ Store    │ State           │  │
│  └──────────┴──────────┴─────────────────┘  │
└─────────────────────────────────────────────┘
```

### Data Flow

```
Cognitive Layer                 Orchestrator Layer              Execution Platform
─────────────                   ──────────────────              ──────────────────
Skill(input)                         │                               │
    │                                │                               │
    ├──[Adapter L1→L2]──────► DAGOrchestrator.__call__()             │
    │                                │                               │
    │                         run_node(node)                         │
    │                                │                               │
    │                                ├──[Adapter L2→L3]──────► ExecutionEvent()
    │                                │                               │
    │                                │                         EventBus.emit()
    │                                │                         State.snapshot()
    │                                │                         Handler.execute()
    │                                │                               │
    │                                │◄────────────────────── ExecutionResult
    │                                │                               │
    │◄──────────────────────── OrchestratorResult                    │
```

### Strict Layer Rules

1. **Layer 1 (Cognitive) NEVER touches Layer 3 (Execution Platform)** — all interactions go through Layer 2
2. **Layer 2 (Orchestrator) accesses Layer 3 ONLY through defined interfaces** — via adapters and protocol abstractions
3. **Layer 3 (Execution Platform) knows nothing about Layers 1 or 2** — fully standalone, dependency-free downward

---

## 3. Strengths Analysis

| # | Strength | Impact |
|---|----------|--------|
| S1 | **Strict layer separation** — each layer is independently testable, replaceable, and evolvable | High — enables parallel development and clean boundaries |
| S2 | **Immutable state + time-travel** — UNDO/REDO via snapshot chain | High — differentiator for debugging and reliability |
| S3 | **Dual DAG model** (Plan DAG + Execution DAG) — intent vs. reality | High — allows rehydration, replay, and divergence tracking |
| S4 | **Rich type system** (Memory/Artifact/StoreEntry) — well-designed from day one | Medium — reduces ambiguity and enables structured retrieval |
| S5 | **Staged evolution** (ext field promotion) — forward-compatible without breaking changes | Medium — prevents premature abstraction |
| S6 | **Event-driven architecture** — unified event model for all execution types | High — enables cross-cutting concerns (logging, retry, budget) cleanly |
| S7 | **Adapter pattern for cross-layer communication** — SOLID-compliant | Medium — enables future layer rewrites without cascade |

## 4. Weaknesses & Warning Points

| # | Weakness | Severity | Mitigation |
|---|----------|----------|------------|
| W1 | **Complexity of immutable state + time-travel** — building a reliable snapshot chain with memory/artifact dependencies is extremely hard to get right | **Critical** | Start with simple linear snapshots before branching; define clear garbage-collection policy |
| W2 | **Over-abstraction risk** — 3 layers + adapters + dual DAGs + event system + state management is a lot of moving parts for V1 | **High** | Implement the minimal viable version of each component; defer advanced features to later stages |
| W3 | **Async execution model** — ForEach parallel branches, event gathering, and error rollback across parallel paths are concurrency-hard | **High** | Use structured concurrency (TaskGroup/asyncio); avoid custom async primitives |
| W4 | **Memory/Artifact store mixed with execution state** — coupling context (memories) with execution snapshots (state) can create circular dependencies | **Medium** | Keep ContextStore and ExecutionState as separate systems with explicit sync points |
| W5 | **TypeScript type definitions for a Python project** — the SPEC defines Memory/Artifact types in TypeScript but the codebase is Python | **Medium** | Translate to Pydantic models immediately; use the TS types as reference spec only |
| W6 | **Plan DAG ↔ Execution DAG synchronization** — keeping them in sync during retries, loops, and dynamic changes is non-trivial | **High** | Execution DAG should reference Plan DAG nodes but own its structure independently |
| W7 | **No clear error taxonomy** — retry, budget, interrupt, and failure modes are mentioned but not specced | **Medium** | Define an error hierarchy early (recoverable vs. fatal, transient vs. permanent) |
| W8 | **"Metaskill" declarative future** — designing V1 imperatively while planning for V2 declarative risks either premature abstraction or costly rewrites | **Medium** | Design interfaces now that can accept both imperative and declarative definitions later |

---

## 5. Fundamental Elements Checklist

These are non-negotiable for a framework of this nature:

### Layer 3 — Execution Platform (Foundation)
- [ ] `EventBus` — publish/subscribe with typed events
- [ ] `ExecutionEvent` hierarchy — Text, Data, Function, Tool
- [ ] `ExecutionState` — immutable snapshot chain
- [ ] `ContextStore` — Memory + Artifact unified store (Pydantic models)
- [ ] `LogCollector` / `TraceCollector` — structured execution traces
- [ ] Error hierarchy — recoverable, fatal, budget-exceeded, timeout, interrupt
- [ ] Retry / Budget / Timeout policies

### Layer 2 — Orchestrator
- [ ] `PlanDAG` — the intent graph (what should happen)
- [ ] `ExecutionDAG` — the reality graph (what is happening)
- [ ] `DAGOrchestrator` — traverses PlanDAG, produces ExecutionDAG
- [ ] Node types — `ExecutionNode`, `FlowNode` (ForEach, Filter, Switch, etc.)
- [ ] Adapter L2→L3 — translates nodes into ExecutionEvents
- [ ] Validation / Policy enforcement — authorization checks per node
- [ ] Snapshot integration — coordinate with ExecutionState for time-travel

### Layer 1 — Cognitive
- [ ] `Skill` — atomic cognitive capability (LLM call, reasoning step)
- [ ] `MetaSkill` — skill that produces/modifies other skills (future, but interface now)
- [ ] Adapter L1→L2 — translates cognitive intent into PlanDAG nodes
- [ ] Prompt management — template, context injection, serialization
- [ ] LLM abstraction — provider-agnostic interface

---

## 6. Execution Schedule

Development proceeds **bottom-up**: Execution Platform → Orchestrator → Cognitive.  
Each phase produces a usable, independently testable deliverable.

### Phase 1 — Execution Platform Foundation
> **Goal**: A standalone event-execution engine with state management

| Step | Deliverable | Depends On |
|------|-------------|------------|
| 1.1 | Core types & protocols (`BaseEntry`, `EntryContent`, Pydantic models) | — |
| 1.2 | `EventBus` — typed publish/subscribe | 1.1 |
| 1.3 | `ExecutionEvent` base + concrete handlers (Text, Data, Function, Tool) | 1.1, 1.2 |
| 1.4 | `ExecutionState` — snapshot chain, level management | 1.1 |
| 1.5 | `ContextStore` — Memory + Artifact store (Stage 1: file-based) + separate MemoryStore / ArtifactStore logic | 1.1 |
| 1.6 | Error hierarchy + Retry/Budget/Timeout policies | 1.3 |
| 1.7 | `LogCollector` + `TraceCollector` | 1.2, 1.3 |
| 1.8 | Integration tests — event lifecycle, state snapshots, store CRUD | All above |

### Phase 2 — Orchestrator Layer
> **Goal**: DAG-based orchestration that drives the Execution Platform

| Step | Deliverable | Depends On |
|------|-------------|------------|
| 2.1 | `BaseNode`, `ExecutionNode`, `FlowNode` type hierarchy | Phase 1 |
| 2.2 | `PlanDAG` data structure + traversal | 2.1 |
| 2.3 | `ExecutionDAG` — runtime DAG produced from plan execution | 2.1, 1.4 |
| 2.4 | Adapter L2→L3 — node-to-event translation | 2.1, 1.3 |
| 2.5 | `DAGOrchestrator` — core `run_next`/`run_node` loop | 2.2, 2.3, 2.4 |
| 2.6 | FlowNode implementations — ForEach, Filter, Switch, Get | 2.5 |
| 2.7 | Validation + policy enforcement | 2.5 |
| 2.8 | Time-travel integration — undo/redo via state snapshots | 2.5, 1.4 |
| 2.9 | Integration tests — DAG execution, parallel branches, rollback | All above |

### Phase 3 — Cognitive Layer
> **Goal**: LLM-powered skills that drive orchestration

| Step | Deliverable | Depends On |
|------|-------------|------------|
| 3.1 | LLM abstraction — provider-agnostic interface | — |
| 3.2 | `Skill` base class — atomic cognitive unit | 3.1 |
| 3.3 | Prompt management — template engine, context injection | 3.2, 1.5 |
| 3.4 | `ContextSerializer` — render memories into LLM prompts | 1.5, 3.3 |
| 3.5 | Adapter L1→L2 — skill intent → PlanDAG nodes | 3.2, Phase 2 |
| 3.6 | Built-in skills — TextGeneration, DataExtraction, CodeGeneration | 3.2, 3.3 |
| 3.7 | `MetaSkill` interface (stub for V2) | 3.2 |
| 3.8 | End-to-end integration tests — skill → orchestration → execution | All |

---

## 7. Layer-Specific Development Specs

The design decisions, detailed architecture, and implementation guidance for each layer are in dedicated documents:

| Layer | Document |
|-------|----------|
| **Layer 3 — Execution Platform** | [SPEC-L3-execution-platform.md](SPEC-L3-execution-platform.md) |
| **Layer 2 — Orchestrator** | [SPEC-L2-orchestrator.md](SPEC-L2-orchestrator.md) |
| **Layer 1 — Cognitive** | [SPEC-L1-cognitive.md](SPEC-L1-cognitive.md) |

---

## 8. Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.10+ | Async support, ML/LLM ecosystem |
| Type system | Pydantic v2 | Runtime validation, serialization, JSON Schema |
| State patches | jsonpatch | Efficient immutable state diffs |
| Async | asyncio + TaskGroup | Structured concurrency for parallel branches |
| LLM client | openai SDK (abstracted) | Provider-agnostic behind interface |
| API (optional) | FastAPI | For future agent-as-a-service exposure |
| Testing | pytest + pytest-asyncio | Standard, well-supported |

---

## 9. Package Structure

```
rh_cognitiv/
├── execution_platform/          # Layer 3 — standalone
│   ├── __init__.py
│   ├── event_bus.py             # EventBus, typed pub/sub
│   ├── events.py                # ExecutionEvent hierarchy
│   ├── state.py                 # ExecutionState, snapshots
│   ├── context_store.py         # ContextStore unified interface
│   ├── memory_store.py          # Memory-specific logic + serialization
│   ├── artifact_store.py        # Artifact-specific logic + format-diverse serialization
│   ├── models.py                # Pydantic models (BaseEntry, etc.)
│   ├── errors.py                # Error hierarchy
│   ├── policies.py              # Retry, Budget, Timeout
│   ├── log_collector.py         # Structured logging
│   ├── trace_collector.py       # Execution traces
│   └── protocols.py             # Abstract interfaces (ABCs)
│
├── orchestrators/               # Layer 2
│   ├── __init__.py
│   ├── nodes.py                 # BaseNode, ExecutionNode, FlowNode
│   ├── plan_dag.py              # PlanDAG structure
│   ├── execution_dag.py         # ExecutionDAG (runtime)
│   ├── dag_orchestrator.py      # Core orchestrator loop
│   ├── flow_nodes.py            # ForEach, Filter, Switch, Get
│   ├── validation.py            # Policy enforcement
│   ├── adapters.py              # L2→L3 adapter (node → event)
│   └── protocols.py             # Abstract interfaces
│
├── cognitive/                   # Layer 1
│   ├── __init__.py
│   ├── skill.py                 # Skill base class
│   ├── meta_skill.py            # MetaSkill interface (V2 stub)
│   ├── llm.py                   # LLM provider abstraction
│   ├── prompt.py                # Prompt templates + management
│   ├── serializer.py            # ContextSerializer
│   ├── adapters.py              # L1→L2 adapter (skill → DAG)
│   └── protocols.py             # Abstract interfaces
│
└── shared/                      # Cross-cutting utilities
    ├── __init__.py
    ├── types.py                 # ID, Timestamp, Ext
    └── ulid.py                  # ULID generation
```
