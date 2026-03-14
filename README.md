# RH COGNITV

**Cognitive Skill-driven Orchestration Framework for Python**

A three-layer framework for building agentic AI systems where **skills** define *what* to do, **orchestration** decides *when* and *in what order*, and an **execution platform** handles *how* — with full observability, time-travel state, and policy-governed execution.

```
pip install rh_cognitv
```

> Python 3.12+ · Pydantic v2 · Fully async · Type-safe · 1 295 tests

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Layer 3 — Execution Platform](#layer-3--execution-platform)
- [Layer 2 — Orchestrator](#layer-2--orchestrator)
- [Layer 1 — Cognitive](#layer-1--cognitive)
- [Quick Start](#quick-start)
- [Examples](#examples)
  - [Minimal: One‑Step Skill](#minimal-one-step-skill)
  - [Multi‑Step Agent with Tools](#multi-step-agent-with-tools)
  - [Zero‑Code Declarative Skill](#zero-code-declarative-skill)
  - [OpenAI Integration](#openai-integration)
  - [EventBus Observability](#eventbus-observability)
- [Design Concepts](#design-concepts)
- [API Reference Highlights](#api-reference-highlights)
- [Installation & Development](#installation--development)
- [License](#license)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  L1 — Cognitive Layer                               │
│  Skills · LLM Protocol · Prompt Engine · Serializer │
│  "What to do"                                       │
├──────────────────────┬──────────────────────────────┤
│  L1→L2 Adapter       │  SkillPlan ↔ PlanDAG         │
│                      │  ExecutionDAG ↔ SkillResult   │
├──────────────────────┴──────────────────────────────┤
│  L2 — Orchestrator Layer                            │
│  PlanDAG · DAGOrchestrator · Adapters · Flow Nodes  │
│  "When and in what order"                           │
├──────────────────────┬──────────────────────────────┤
│  L2→L3 Bridge        │  Node → ExecutionEvent        │
│                      │  ExecutionResult → NodeResult  │
├──────────────────────┴──────────────────────────────┤
│  L3 — Execution Platform                            │
│  EventBus · Handlers · Policies · State · Budget    │
│  Context Store · Observability                      │
│  "How to execute"                                   │
└─────────────────────────────────────────────────────┘
```

Each layer depends only on the one below it. L3 has zero upward dependencies and can be used standalone. All boundaries are defined by abstract protocols (ABCs), making every component independently testable and replaceable.

---

## Layer 3 — Execution Platform

The self-contained runtime engine. Manages the actual execution of events with full policy governance and observability.

### Core Components

| Component | Purpose |
|---|---|
| **EventBus** | Hybrid sync middleware + async subscriber fan-out. Type-based dispatch. |
| **HandlerRegistry** | Strategy pattern — maps `EventKind` → `EventHandler[T]`. |
| **TextHandler / DataHandler / FunctionHandler / ToolHandler** | Base handler implementations (subclass for real LLM/tool calls). |
| **PolicyChain** | Composable middleware wrapping handler execution. |
| **RetryPolicy** | Exponential backoff with configurable max attempts. |
| **TimeoutPolicy** | Per-node async timeout enforcement. |
| **BudgetPolicy** | Token / call / time budget enforcement via `BudgetTracker`. |
| **ExecutionState** | Immutable snapshot chain for time-travel state management. |
| **ContextStore** | Unified memory + artifact storage with query interface. |
| **LogCollector / TraceCollector** | Structured logging and execution tracing (EventBus subscribers). |

### Event Kinds

Every execution request is modeled as an `ExecutionEvent` with a typed payload:

| Kind | Payload | Handler Returns |
|---|---|---|
| `TEXT` | `TextPayload(prompt, system_prompt, model, temperature, max_tokens)` | `LLMResultData` |
| `DATA` | `DataPayload(prompt, output_schema, model)` | `LLMResultData` |
| `FUNCTION` | `FunctionPayload(function_name, args, kwargs)` | `FunctionResultData` |
| `TOOL` | `ToolPayload(prompt, tools, model)` | `ToolResultData` |

### Cognitive Memory Model

Memories carry rich metadata for trust reasoning:

- **Roles**: episodic, semantic, procedural, working
- **Shapes**: atom, sequence, summary, narrative
- **Origins**: observed, told, inferred, consolidated
- **Provenance**: full lineage tracking for every memory and artifact

### Escalation (Human-in-the-Loop)

When a handler needs human input, it emits `EscalationRequested`. The EventBus freezes the node (status `ESCALATED`), and `wait_for(EscalationResolved)` resumes execution when the decision arrives — cloud-safe with serializable resume data.

---

## Layer 2 — Orchestrator

The strategy brain — translates high-level plans into directed acyclic graphs, executes them with parallel scheduling, and records what happened.

### Core Components

| Component | Purpose |
|---|---|
| **PlanDAG** | Frozen, immutable intent graph built via `DAGBuilder`. |
| **ExecutionDAG** | Append-only runtime log of what actually happened. |
| **DAGOrchestrator** | Topological-sort ready-queue engine with parallel `asyncio.gather`. |
| **AdapterRegistry** | Strategy pattern — maps node `kind` → `NodeAdapter` (L2→L3 bridge). |
| **PlatformRef** | Bundle of L3 references (handler registry, config, budget). |
| **ValidationPipeline** | Pre-flight checks (dependency, budget, input schema). |
| **FlowHandlerRegistry** | Pure-L2 control flow nodes that never reach L3. |

### Node Types

**Execution Nodes** (map 1:1 to L3 `EventKind`):

| Node | L3 Event | Purpose |
|---|---|---|
| `TextNode` | `TEXT` | LLM text generation |
| `DataNode` | `DATA` | Structured data extraction |
| `FunctionNode` | `FUNCTION` | Direct function invocation |
| `ToolNode` | `TOOL` | LLM-driven tool use |

**Flow Nodes** (pure L2 control flow, never reach L3):

| Node | Purpose |
|---|---|
| `ForEachNode` | Iterate over a collection (parallel or sequential) |
| `FilterNode` | Conditional branch on a predicate |
| `SwitchNode` | Multi-way branch on a value |
| `GetNode` | Extract data from prior node results |
| `IfNotOkNode` | Error-handling branch |
| `CompositeNode` | Nested sub-DAG execution |

### DAG Builder

```python
from rh_cognitv.orchestrator import DAGBuilder, TextNode, FunctionNode

dag = (
    DAGBuilder("my_plan")
    .add_node("research", TextNode(prompt="Research the topic"))
    .add_node("calculate", FunctionNode(function_name="calc", kwargs={"x": 42}))
    .add_node("draft", TextNode(prompt="Write the answer"))
    .edge("research", "calculate")
    .edge("calculate", "draft")
    .build()
)
```

### Parallel Execution

Nodes whose predecessors are all complete execute concurrently via `asyncio.gather`. The orchestrator automatically discovers parallelism from the DAG topology — no manual configuration needed.

---

## Layer 1 — Cognitive

The intelligence layer — defines skills, abstracts LLM providers, builds prompts, serializes context, and bridges to L2.

### Core Components

| Component | Purpose |
|---|---|
| **SkillProtocol** | `plan()` / `interpret()` split — stateless atomic cognitive unit. |
| **Skill** | Abstract base class implementing `SkillProtocol` with `validate_output()`. |
| **ConfigSkill** | Zero-code declarative skill from `SkillConfig` — auto-generates plan/interpret. |
| **LLMProtocol** | Provider-agnostic: `complete()`, `complete_structured()`, `complete_with_tools()`. |
| **MockLLM** | Full-featured test double with response queuing, call recording, token tracking. |
| **PromptBuilder** | Chainable programmatic API: `.system()`, `.user()`, `.context()`, `.build()`. |
| **TemplateRenderer** | `str.format()`-based rendering for `ConfigSkill` templates. |
| **NaiveSerializer / SectionSerializer** | Context → prompt text (memories + artifacts). |
| **SkillToDAGAdapter** | Translates `SkillPlan` → `PlanDAG` (L1→L2 boundary). |
| **ResultAdapter** | Translates `ExecutionDAG` → `OrchestratorResult` (L2→L1 boundary). |
| **MetaSkill** | V2 stub for skills that generate other skills or DAGs from natural language. |

### Built-in Skills

| Skill | Description |
|---|---|
| `TextGenerationSkill` | Single text-generation step with configurable system prompt, model, and temperature. |
| `DataExtractionSkill` | Structured data extraction with Pydantic output schema validation. |
| `CodeGenerationSkill` | Code generation with language-aware system prompts. |
| `ReviewSkill` | Pass/fail review with `validate_output()` triggering `RetryableValidationError`. |

### Skill Lifecycle

```
1. Framework builds SkillContext (memories, artifacts, budget)
2. skill.plan(input, context) → SkillPlan (list of typed steps)
3. SkillToDAGAdapter converts SkillPlan → PlanDAG
4. DAGOrchestrator.run(dag) → ExecutionDAG
5. ResultAdapter converts ExecutionDAG → OrchestratorResult
6. skill.interpret(result) → SkillResult (typed output + provenance)
```

---

## Quick Start

```python
import asyncio
from pydantic import BaseModel

from rh_cognitv.cognitive import (
    Skill, SkillContext, SkillPlan, SkillStep, SkillResult,
    SkillProvenance, TextStepConfig,
    SkillToDAGAdapter, ResultAdapter, OrchestratorResult,
)
from rh_cognitv.orchestrator import (
    AdapterRegistry, PlatformRef, DAGOrchestrator, OrchestratorConfig,
)
from rh_cognitv.execution_platform import (
    HandlerRegistry, EventKind, ExecutionState, TextHandler,
)


# 1. Define your skill
class GreetingSkill(Skill):
    @property
    def name(self) -> str:
        return "greeting"

    @property
    def description(self) -> str:
        return "Generates a greeting"

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        return SkillPlan(
            name=self.name,
            steps=[
                SkillStep(
                    id="greet",
                    kind="text",
                    config=TextStepConfig(prompt=f"Say hello to {input.name}"),
                ),
            ],
        )

    async def interpret(self, result) -> SkillResult:
        r = result.step_results.get("greet")
        return SkillResult(
            output=r.value if r and r.ok else None,
            success=r.ok if r else False,
            provenance=SkillProvenance(skill_name=self.name),
        )


class UserInput(BaseModel):
    name: str


# 2. Wire and run
async def main():
    # L3: Handlers (use base TextHandler or plug in OpenAI, Anthropic, etc.)
    registry = HandlerRegistry()
    registry.register(EventKind.TEXT, TextHandler())

    # L2: Orchestrator
    platform = PlatformRef(registry=registry)
    orchestrator = DAGOrchestrator(
        adapter_registry=AdapterRegistry.with_defaults(),
        platform=platform,
        state=ExecutionState(),
    )

    # L1: Skill → Plan → DAG → Execute → Interpret
    skill = GreetingSkill()
    plan = await skill.plan(UserInput(name="World"), SkillContext())
    dag = SkillToDAGAdapter().to_dag(plan)
    exec_dag = await orchestrator.run(dag)
    result = await skill.interpret(ResultAdapter().from_result(exec_dag))

    print(result.output)  # LLM response text

asyncio.run(main())
```

---

## Examples

### Minimal: One‑Step Skill

```python
from rh_cognitv.cognitive import ConfigSkill, SkillConfig

skill = ConfigSkill(SkillConfig(
    name="summarizer",
    description="Summarize text",
    prompt_template="Summarize the following:\n\n{text}",
    system_prompt="You are a concise summarizer.",
    model="gpt-4o-mini",
    temperature=0.3,
))
```

`ConfigSkill` auto-generates `plan()` and `interpret()` — zero code needed for simple single-step skills.

### Multi‑Step Agent with Tools

```python
plan = SkillPlan(
    name="research_agent",
    steps=[
        SkillStep(id="lookup", kind="function",
                  config=FunctionStepConfig(function_name="search", kwargs={"q": "AI"})),
        SkillStep(id="analyze", kind="text",
                  config=TextStepConfig(prompt="Analyze the search results...")),
        SkillStep(id="review", kind="text",
                  config=TextStepConfig(prompt="Review the analysis for accuracy")),
    ],
)
```

Steps without explicit `depends_on` are wired sequentially. Add `depends_on` to declare parallel branches — the orchestrator discovers and exploits all available parallelism automatically.

### Zero‑Code Declarative Skill

```python
from pydantic import BaseModel

class SentimentOutput(BaseModel):
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float

skill = ConfigSkill(SkillConfig(
    name="sentiment",
    description="Classify text sentiment",
    prompt_template="Classify the sentiment of: {text}",
    output_schema=SentimentOutput,
    model="gpt-4o-mini",
))
```

When `output_schema` is a Pydantic model, `ConfigSkill` uses a `DATA` step with JSON mode and validates the output automatically — raising `RetryableValidationError` on schema mismatch so L3's `RetryPolicy` can retry.

### OpenAI Integration

The package includes a ready-to-use OpenAI adapter:

```python
from rh_cognitv.execution_platform.openai_handler import (
    OpenAITextHandler,
    OpenAIDataHandler,
)

# Register OpenAI-backed handlers
registry = HandlerRegistry()
registry.register(EventKind.TEXT, OpenAITextHandler(api_key="sk-...", model="gpt-4o-mini"))
registry.register(EventKind.DATA, OpenAIDataHandler(api_key="sk-...", model="gpt-4o-mini"))
```

Or implement your own by subclassing `EventHandlerProtocol[LLMResultData]` for any provider (Anthropic, local models, etc.).

### EventBus Observability

```python
from dataclasses import dataclass
from rh_cognitv.execution_platform import EventBus, ExecutionEvent

@dataclass
class StepStarted:
    node_id: str

@dataclass
class StepCompleted:
    node_id: str
    success: bool

bus = EventBus()

# Sync handler — runs in deterministic order
bus.on(StepStarted, lambda e: print(f"  ▶ {e.node_id}"))
bus.on(StepCompleted, lambda e: print(f"  {'✓' if e.success else '✗'} {e.node_id}"))

# Async subscriber — fire-and-forget fan-out
bus.on_async(StepCompleted, my_async_metrics_reporter)

# Middleware — intercept all events
bus.use(my_logging_middleware)
```

The `EventBus` supports sync middleware (deterministic, replayable), sync handlers, and async subscribers. Use `wait_for()` for escalation round-trips.

---

## Design Concepts

### Plan / Interpret Split

Skills never execute directly. `plan()` produces a declarative `SkillPlan` describing *what* work to do. The framework converts it to a DAG and executes it. `interpret()` receives the raw results and produces a typed `SkillResult` with provenance.

This separation means skills are **stateless**, **testable** (just call `plan()` and inspect the steps), and **composable**.

### Strategy Pattern Everywhere

- `HandlerRegistry` maps `EventKind` → handler
- `AdapterRegistry` maps node `kind` → adapter
- `FlowHandlerRegistry` maps flow node types → flow handlers
- `PolicyChain` wraps execution with composable policies

Every dispatch point uses the same pattern: a registry of named strategies. Swap any component without touching the rest.

### Protocol-Driven Boundaries

Every cross-layer boundary and major component is defined by an `ABC` protocol:

```
SkillProtocol          — L1 skill contract
LLMProtocol            — LLM provider contract
EventHandlerProtocol   — L3 handler contract
HandlerRegistryProtocol — L3 registry contract
ExecutionStateProtocol — L3 state contract
BudgetTrackerProtocol  — L3 budget contract
ContextStoreProtocol   — L3 storage contract
OrchestratorProtocol   — L2 orchestrator contract
NodeAdapterProtocol    — L2→L3 bridge contract
```

### Immutable State with Time-Travel

`ExecutionState` takes a full deep-copy snapshot after every node execution. You can `restore(version)` to any point, `undo()` / `redo()` through the chain, and `gc_collect()` to prune old snapshots. DAG entries are append-only — undo marks entries as `ROLLED_BACK`, never deletes.

### Budget as a First-Class Resource

`BudgetTracker` enforces token, call, and time limits. `BudgetPolicy` integrates with the `PolicyChain` to abort execution before it exceeds budget. Skills can declare `SkillConstraints` for per-plan limits.

### Context References

Skills declare execution-time context dependencies via `ContextRef` on each step:

```python
SkillStep(
    id="analyze",
    kind="text",
    config=TextStepConfig(prompt="Analyze this..."),
    context_refs=[
        ContextRef(kind="memory", id="mem-123", key="background"),
        ContextRef(kind="previous_result", from_step="research", key="data"),
        ContextRef(kind="query", query=MemoryQuery(text="relevant facts"), key="facts"),
    ],
)
```

The orchestrator resolves these before executing each node — injecting memories, artifacts, and prior results into the data payload.

### Composable Policies

Policies wrap handler execution in a chain:

```python
chain = PolicyChain([
    BudgetPolicy(tracker=budget),    # Check budget first
    TimeoutPolicy(seconds=30),       # Enforce timeout
    RetryPolicy(max_attempts=3),     # Retry on transient errors
])
result = await chain(handler.handle, event, data, configs)
```

Per-node overrides merge with orchestrator defaults automatically.

---

## API Reference Highlights

### Execution Platform (L3)

```python
from rh_cognitv.execution_platform import (
    # Core
    EventBus, HandlerRegistry, ExecutionState, BudgetTracker,
    # Events
    ExecutionEvent, TextPayload, DataPayload, FunctionPayload, ToolPayload,
    # Handlers
    TextHandler, DataHandler, FunctionHandler, ToolHandler,
    # Policies
    PolicyChain, RetryPolicy, TimeoutPolicy, BudgetPolicy,
    # Storage
    ContextStore, MemoryStore, ArtifactStore,
    # Models
    EventKind, ExecutionResult, LLMResultData, FunctionResultData,
    Memory, Artifact, MemoryRole, MemoryShape, TokenUsage,
    # Observability
    LogCollector, TraceCollector,
    # Escalation
    EscalationRequested, EscalationResolved,
    # Errors
    TransientError, PermanentError, BudgetError, InterruptError,
)
```

### Orchestrator (L2)

```python
from rh_cognitv.orchestrator import (
    # Core
    DAGOrchestrator, PlanDAG, ExecutionDAG, DAGBuilder,
    # Nodes
    TextNode, DataNode, FunctionNode, ToolNode,
    ForEachNode, FilterNode, SwitchNode, GetNode, IfNotOkNode,
    # Adapters
    AdapterRegistry, PlatformRef,
    # Validation
    ValidationPipeline, DependencyValidator, BudgetValidator,
    # Config
    OrchestratorConfig, NodeResult,
)
```

### Cognitive (L1)

```python
from rh_cognitv.cognitive import (
    # Skills
    Skill, ConfigSkill, SkillProtocol,
    TextGenerationSkill, DataExtractionSkill, CodeGenerationSkill, ReviewSkill,
    MetaSkill,
    # Models
    SkillPlan, SkillStep, SkillResult, SkillContext, SkillConfig,
    Message, MessageRole, CompletionResult, ToolResult,
    TextStepConfig, DataStepConfig, FunctionStepConfig, ToolStepConfig,
    # LLM
    LLMProtocol, MockLLM,
    # Prompt
    PromptBuilder, TemplateRenderer,
    # Serializer
    NaiveSerializer, SectionSerializer,
    # Adapters
    SkillToDAGAdapter, ResultAdapter, OrchestratorResult,
)
```

---

## Installation & Development

### Install from PyPI

```bash
pip install rh_cognitv
```

### Install from GitHub

```bash
pip install git+https://github.com/rhodief/rh-cognitv.git
```

### Local development

```bash
git clone https://github.com/rhodief/rh-cognitv.git
cd rh-cognitv
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
```

### Build the package

```bash
bash package.sh          # build wheel + sdist
bash package.sh --help   # full publishing guide
```

### Optional dependencies

```bash
pip install openai       # For OpenAI handler adapter
pip install python-dotenv # For .env file loading in examples
```

---

## License

MIT — see [pyproject.toml](pyproject.toml) for full metadata.