# Layer 1 — Cognitive: Development Specification

> **Parent**: [SPEC-overview.md](SPEC-overview.md)  
> **Layer**: 1 (Top)  
> **Status**: Ready for Development  
> **Last Updated**: 2026-03-14  
> **Depends On**: Layer 2 (Orchestrator) — via protocol abstractions only

---

## 1. Purpose

The Cognitive Layer is the **intelligence** of rh_cognitiv. It is where LLM reasoning, planning, and review happen. It provides:

- **Skills** — atomic cognitive capabilities (generate text, extract data, review output, call tools)
- **MetaSkills** (future V2) — skills that create/compose other skills and orchestrations declaratively
- **Prompt management** — template engine, context injection, memory serialization
- **LLM abstraction** — provider-agnostic interface to language models
- **Context serialization** — render memories and artifacts into LLM-consumable prompts

This layer **never** touches the Execution Platform (Layer 3) directly. All execution happens through the Orchestrator (Layer 2) via the L1→L2 adapter. The Cognitive Layer expresses **what** it wants; the Orchestrator decides **how** and **when**; the Execution Platform **does it**.

---

## 2. Component Architecture

```
cognitive/
├── protocols.py          # ABCs — SkillProtocol, LLMProtocol, PromptProtocol
├── models.py             # Pydantic models — SkillConfig, SkillResult, PromptTemplate
│
├── skill.py              # Skill base class — atomic cognitive unit
├── meta_skill.py         # MetaSkill interface (V2 stub)
├── builtin_skills.py     # TextGeneration, DataExtraction, CodeGeneration, Review
│
├── llm.py                # LLM provider abstraction
├── prompt.py             # Prompt template engine + context injection
├── serializer.py         # ContextSerializer — render memories into prompts
└── adapters.py           # L1→L2 adapter — skill intent → PlanDAG nodes
```

### Relationship to Layer 2

```
Cognitive Layer                              Orchestrator Layer (L2)
───────────────                              ──────────────────────

Skill.plan(input, context)
        │
        ├── serializer.render(memories)   ← pulls from ContextStore (via L2)
        │
        ├── prompt.build(template, context, input)
        │
        ├── Returns SkillPlan:
        │     - nodes to execute
        │     - expected outputs
        │     - constraints
        │
        └── adapter.to_dag(skill_plan) ──────► PlanDAG
                                                  │
                                            DAGOrchestrator runs it
                                                  │
                                            ◄──── OrchestratorResult
                                                  │
        Skill.interpret(result)  ◄────────── adapter.from_result(result)
        │
        └── Returns SkillResult (typed, validated)
```

---

## 3. Core Concepts

### 3.1 Skill

A **Skill** is the atomic unit of cognitive work. It encapsulates:
- **What** LLM call(s) to make (prompt template + input schema)
- **How** to interpret the result (output schema + validation)
- **What** context it needs (memory query + relevant artifacts)
- **What** plan to ask the orchestrator to execute (via adapter)

A Skill does NOT call the LLM directly. It produces a **SkillPlan** that describes the work, and the orchestrator takes it from there.

```
Skill lifecycle:
  1. plan(input, context)     → SkillPlan (what nodes to execute)
  2. [Orchestrator executes]  → OrchestratorResult
  3. interpret(result)        → SkillResult (typed output)
```

### 3.2 MetaSkill (V2)

A **MetaSkill** is a skill that produces other skills or orchestration graphs. It enables the declarative future: users describe what they want, and the MetaSkill generates the PlanDAG.

For V1: define the interface only. The MetaSkill protocol should accept a natural language description and return either a `Skill` or a `PlanDAG`.

### 3.3 ContextSerializer

Converts retrieved memories and artifacts into the text string injected into LLM prompts. This is the bridge between the ContextStore (Layer 3) and the Cognitive Layer.

- Stage 1: Naive concatenation (join text fields with headers)
- Stage 2: Section-based (group by memory role with markdown headers)
- Stage 3: Budget-aware (rank, truncate, fit within token limits)

### 3.4 LLM Abstraction

A provider-agnostic interface for making LLM calls. The Cognitive Layer defines **what** it needs (completion, structured output, embedding); the concrete implementation talks to OpenAI, Anthropic, local models, etc.

---

## 4. Design Decisions

### DD-L1-01: Skill Interface Design

**Issue**: Skills need a clear contract that separates reasoning (Cognitive) from execution (Orchestrator). The interface must support simple single-step skills AND complex multi-step skills, while remaining testable.

**Options**:

#### Option A — Plan/Interpret Split (Two-Phase Skill)
Skills have two methods: `plan()` produces the execution plan, `interpret()` processes the result.

```python
class Skill(ABC):
    @abstractmethod
    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan: ...

    @abstractmethod
    async def interpret(self, result: OrchestratorResult) -> SkillResult: ...
```

| Pros | Cons |
|------|------|
| Clear separation of planning and interpretation | Two methods to implement per skill |
| Plan is inspectable/modifiable before execution | SkillContext must carry enough info for planning |
| Testable: test plan() and interpret() independently | Simple skills feel over-structured |
| Natural fit for the architecture | |

#### Option B — Single `execute()` that Receives a Runner
Skills get a callback to invoke the orchestrator.

```python
class Skill(ABC):
    @abstractmethod
    async def execute(self, input: BaseModel, runner: OrchestratorRunner) -> SkillResult:
        plan = self.build_plan(input)
        result = await runner.run(plan)
        return self.interpret(result)
```

| Pros | Cons |
|------|------|
| Single entry point | Skill now orchestrates its own execution |
| Flexible — skill controls the flow | Harder to inspect plan before execution |
| Can do multi-round (plan, execute, replan, execute) | Testing requires mocking the runner |
| | Blurs the boundary with the orchestrator |

#### Option C — Declarative Skill (Config-Only, No Code)
Skills are defined as data (prompt template, input schema, output schema). The framework handles plan/interpret automatically.

```python
text_skill = SkillConfig(
    name="summarize",
    prompt_template="Summarize the following: {input.text}",
    input_schema=SummarizeInput,
    output_schema=SummarizeOutput,
    memory_query=MemoryQuery(role="semantic", topK=3),
)
```

| Pros | Cons |
|------|------|
| Zero code for simple skills | Can't handle complex multi-step logic |
| Easy to create, share, and version | Less flexible than code-based skills |
| Natural fit for V2 declarative vision | Some skills need custom plan/interpret logic |
| Can be generated by MetaSkills | |

**Suggestion**: **Option A** as the base protocol + **Option C** as a convenience layer for simple skills. `Skill` is the ABC (plan/interpret). `ConfigSkill` is a concrete implementation that auto-generates plan/interpret from a config object. This gives power users full control and simple cases zero boilerplate.

> **DECISION**: **Option A + C** — Plan/Interpret split as the base, ConfigSkill as convenience  
> **Comments**: This maps cleanly to the existing L2 architecture. `SkillPlan` steps translate directly to L2 `TextNode`, `DataNode`, `FunctionNode`, and `ToolNode` via the adapter. The two-phase design mirrors L2's own separation: `PlanDAG` (intent) vs `ExecutionDAG` (reality). `plan()` produces the equivalent of a `PlanDAG`, and `interpret()` consumes the `OrchestratorResult` derived from the `ExecutionDAG`. The L2 `DAGBuilder` fluent API is a natural target for the adapter to translate `SkillPlan` steps into. `ConfigSkill` should auto-generate `TextNode`-based plans from template + schema config — matching L2's discriminated `Node` union.

---

### DD-L1-02: LLM Abstraction Depth

**Issue**: How deep should the LLM abstraction go? Just text completion? Structured output? Tool calling? Streaming? Embeddings?

**Options**:

#### Option A — Minimal: Text In, Text Out
```python
class LLMProtocol(Protocol):
    async def complete(self, messages: list[Message]) -> str: ...
```

| Pros | Cons |
|------|------|
| Simplest possible interface | Structured output must be parsed by the skill |
| Works with any LLM provider | No native tool calling support |
| Easy to mock | No streaming |

#### Option B — Structured: Text + Function Calling + Structured Output
```python
class LLMProtocol(Protocol):
    async def complete(self, messages: list[Message]) -> CompletionResult: ...
    async def complete_structured(self, messages: list[Message], schema: type[T]) -> T: ...
    async def complete_with_tools(self, messages: list[Message], tools: list[Tool]) -> ToolResult: ...
```

| Pros | Cons |
|------|------|
| Covers the main LLM interaction patterns | Larger interface to implement per provider |
| Structured output is first-class | Not all providers support all methods |
| Tool calling is native | |

#### Option C — Full: B + Streaming + Embeddings + Token Counting
```python
class LLMProtocol(Protocol):
    async def complete(...) -> CompletionResult: ...
    async def complete_structured(...) -> T: ...
    async def complete_with_tools(...) -> ToolResult: ...
    async def stream(...) -> AsyncIterator[str]: ...
    async def embed(text: str) -> list[float]: ...
    def count_tokens(text: str) -> int: ...
```

| Pros | Cons |
|------|------|
| Complete feature coverage | Large interface — ISP violation risk |
| Streaming for real-time UX | Embeddings may belong in ContextStore, not here |
| Embedding for future vector search | Not all providers support everything |
| Token counting for budget management | |

**Suggestion**: **Option B** for Stage 1. The three methods (complete, structured, tools) map directly to the three ExecutionEvent types (TextEvent, DataEvent, ToolEvent). Add streaming and embeddings as **separate protocols** (ISP — Interface Segregation Principle) when needed.

```python
class LLMProtocol(Protocol):
    async def complete(self, ...) -> CompletionResult: ...
    async def complete_structured(self, ...) -> T: ...
    async def complete_with_tools(self, ...) -> ToolResult: ...

class StreamingLLMProtocol(LLMProtocol, Protocol):
    async def stream(self, ...) -> AsyncIterator[str]: ...

class EmbeddingProtocol(Protocol):
    async def embed(self, text: str) -> list[float]: ...
```

> **DECISION**: **Option B** — Structured: Text + Function Calling + Structured Output  
> **Comments**: The L3 layer already defines exactly four `EventKind` values: `TEXT`, `DATA`, `FUNCTION`, `TOOL` — with matching payloads (`TextPayload`, `DataPayload`, `FunctionPayload`, `ToolPayload`) and handlers (`TextHandler`, `DataHandler`, `FunctionHandler`, `ToolHandler`). The L3 result types `LLMResultData`, `FunctionResultData`, and `ToolResultData` are the return types these LLM protocol methods should produce. `complete()` → `TextPayload`/`LLMResultData`, `complete_structured()` → `DataPayload`/`LLMResultData`, `complete_with_tools()` → `ToolPayload`/`ToolResultData`. The ISP split for streaming/embedding is correct — L3 has no streaming or embedding facilities yet, so adding those as separate protocols avoids forcing implementation on Day 1. Note: `LLMResultData` already includes `token_usage: TokenUsage` and `thinking: str | None` (extended thinking) — the protocol's `CompletionResult` should align with or wrap this model.

---

### DD-L1-03: Prompt Template Engine

**Issue**: Skills need to build prompts from templates, injecting context (memories, artifacts, input data). What template engine?

**Options**:

#### Option A — Python f-strings / str.format()
```python
prompt = template.format(input=input_data, context=context_text)
```

| Pros | Cons |
|------|------|
| Zero dependencies | No conditionals, loops, or filters |
| Familiar to all Python devs | No escaping or safety |
| Fast | Can't handle complex prompts |

#### Option B — Jinja2
```python
prompt = jinja_env.from_string(template).render(input=input_data, context=context_text)
```

| Pros | Cons |
|------|------|
| Full template language (if/for/filters) | New dependency |
| Well-tested, battle-hardened | Template injection risk if user-supplied templates |
| Industry standard for text templating | Overkill for simple prompts |
| Supports template inheritance | |

#### Option C — Custom Lightweight Template Engine
Build a minimal template system with variable substitution + optional sections.

```python
# {{variable}} for substitution
# {{#if has_context}}...{{/if}} for conditionals
prompt = engine.render("Summarize: {{input.text}}{{#if context}}\nContext: {{context}}{{/if}}")
```

| Pros | Cons |
|------|------|
| Tailored to prompt needs | Must build and maintain it |
| Lightweight | Won't match Jinja2's robustness |
| No external dependency | Team must learn custom syntax |

#### Option D — Structured Prompt Builder (Programmatic, No Templates)
```python
prompt = (
    PromptBuilder()
    .system("You are a summarization assistant.")
    .context(serialized_memories)
    .user(f"Summarize: {input.text}")
    .build()
)
```

| Pros | Cons |
|------|------|
| Type-safe, IDE-friendly | Can't be defined declaratively (templates can) |
| No template injection risk | More verbose for simple cases |
| Composable | Harder to share/version prompts as files |
| Natural for multi-message prompts | |

**Suggestion**: **Option D** for the programmatic API (used in code-based Skills) + **Option A** (str.format) for simple variable substitution in templates. Add Jinja2 support later only if declarative skills need complex templates. ConfigSkill uses str.format templates; code-based Skills use PromptBuilder.

> **DECISION**: **Option D + A** — PromptBuilder (programmatic) + str.format (templates)  
> **Comments**: L3's `TextPayload` expects `prompt: str` and `system_prompt: str | None` — these are flat strings, not structured message arrays. `PromptBuilder` should produce exactly these two strings. The builder's `.system()`, `.context()`, `.user()` methods compose into the final `prompt` + `system_prompt` fields that map 1:1 to `TextPayload`. For `ConfigSkill`, `str.format` templating is sufficient — the template fills `{input.field}` and `{context}` and produces a plain string. No need for Jinja2 complexity since L3 payloads are flat text. `DataPayload` also takes `prompt: str` + `output_schema: dict` — same pattern. The `PromptBuilder.build()` return type should be a simple `BuiltPrompt(system_prompt: str | None, prompt: str)` that maps directly to payload fields.

---

### DD-L1-04: Adapter L1→L2 Design

**Issue**: The Cognitive Layer must translate skill plans into PlanDAGs for the orchestrator. How does this adapter work?

**Options**:

#### Option A — Skill Produces PlanDAG Directly
Skills return a PlanDAG from their `plan()` method. The adapter is trivial (pass-through).

```python
class Skill(ABC):
    async def plan(self, input, context) -> PlanDAG: ...
```

| Pros | Cons |
|------|------|
| Simple, no adapter needed | Skill now depends on orchestrator types |
| Direct control over the DAG | Violates layer isolation |
| | Skills must understand DAG construction |

#### Option B — Skill Produces a SkillPlan, Adapter Translates
Skills return a SkillPlan (cognitive-layer type). The adapter converts it to a PlanDAG.

```python
class SkillPlan:
    steps: list[SkillStep]  # sequential or parallel groups
    constraints: SkillConstraints

class SkillStep:
    kind: Literal["text", "data", "function", "tool"]
    config: StepConfig

# Adapter:
class SkillToDAGAdapter:
    def to_dag(self, plan: SkillPlan) -> PlanDAG: ...
```

| Pros | Cons |
|------|------|
| Layer isolation preserved | Extra type translations |
| Skills don't know about DAGs | Adapter must handle all step combinations |
| SkillPlan is a simpler API for skill authors | |
| Adapter can optimize the DAG (merge steps, reorder) | |

#### Option C — Skill Returns a Declarative Description, Smart Adapter Builds the DAG
Skills return a high-level description. A "compiler" adapter builds the optimal DAG.

```python
class SkillIntent:
    """What the skill wants to accomplish, not how."""
    goal: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    required_capabilities: list[str]  # ["text_generation", "data_extraction"]
```

| Pros | Cons |
|------|------|
| Maximum decoupling | "Compiler" is extremely complex |
| Skills are pure intent | Unpredictable DAGs (hard to debug) |
| Enables MetaSkill-style generation | Too ambitious for V1 |

**Suggestion**: **Option B** — the SkillPlan intermediate representation. Skills express work as a list of typed steps with constraints. The adapter converts steps into DAG nodes and edges. This keeps skills simple, preserves isolation, and the adapter logic is straightforward for V1 (sequence of steps → linear DAG).

> **DECISION**: **Option B** — SkillPlan intermediate representation, adapter translates to PlanDAG  
> **Comments**: The L2 layer provides `DAGBuilder` with a fluent `.add_node()` / `.edge()` / `.build() -> PlanDAG` API. The adapter's job is mechanical: iterate `SkillPlan.steps`, map each `SkillStep` to the matching L2 node type (`kind="text"` → `TextNode`, `kind="data"` → `DataNode`, `kind="function"` → `FunctionNode`, `kind="tool"` → `ToolNode`), add them to a `DAGBuilder`, wire edges (sequential steps get linear edges), and `.build()`. The `SkillStep.config` fields should align with L2 node constructor params — e.g. a text step config carries `prompt`, `system_prompt`, `model`, `temperature`, `max_tokens` (matching `TextNode` fields). Constraints from `SkillPlan` (timeout, retries) map to `BaseNode.timeout_seconds` and `BaseNode.max_retries`. The adapter also translates `OrchestratorResult` (an `ExecutionDAG` with `NodeResult` entries) back into a format `interpret()` can consume. This keeps L1 completely free of L2 imports — only the adapter knows both type systems.

---

### DD-L1-05: Context Injection Strategy

**Issue**: Skills need relevant context (memories, artifacts) to produce good prompts. How is context retrieved and injected?

**Options**:

#### Option A — Skill Defines a Static MemoryQuery
Each skill declares what context it needs at definition time.

```python
class SummarizeSkill(Skill):
    memory_query = MemoryQuery(role="semantic", tags=["user-preference"], topK=5)
```

| Pros | Cons |
|------|------|
| Predictable, inspectable | Can't adapt query based on input |
| Easy to optimize (pre-fetch) | Rigid — same query for all inputs |
| | May retrieve irrelevant context |

#### Option B — Skill Builds Query Dynamically from Input
Skill's `plan()` builds the memory query based on the actual input.

```python
async def plan(self, input, context):
    query = MemoryQuery(text=input.topic, role="semantic", topK=5)
    memories = await context.recall(query)
    ...
```

| Pros | Cons |
|------|------|
| Query is input-aware | Skill now calls the store (coupling) |
| Better relevance | Non-deterministic (different inputs → different context) |
| Flexible | Harder to test |

#### Option C — SkillContext Pre-Loads Based on Skill Metadata + Input
A SkillContext object is assembled before `plan()` is called. It uses skill metadata (declared queries) + input data to retrieve relevant context.

```python
class SkillContext:
    memories: list[Memory]
    artifacts: list[Artifact]
    budget: BudgetSnapshot

# Built by the framework:
context = await build_skill_context(skill.metadata, input_data, store)
result = await skill.plan(input_data, context)
```

| Pros | Cons |
|------|------|
| Skill doesn't touch the store directly | Framework must infer what's relevant |
| Context is pre-loaded and inspectable | May over-fetch or under-fetch |
| Testable — inject mock context | Skill metadata must be expressive enough |
| Clean separation | |

**Suggestion**: **Option C** — pre-loaded SkillContext. The framework builds context from skill metadata (base queries) enriched with input data (dynamic query terms). The skill receives a ready-to-use `SkillContext` and never touches the store. This preserves layer isolation and makes skills pure functions of (input, context) → plan.

> **DECISION**: **Option C** — Pre-loaded SkillContext  
> **Comments**: The L3 `ContextStoreProtocol` provides `recall(query: MemoryQuery) -> list[QueryResult]` where `MemoryQuery` supports `text`, `role`, `artifact_type`, `tags`, and `top_k` filters. The framework's context builder can combine skill metadata (static `MemoryQuery` per skill — role, tags, top_k) with input-derived terms (inject `input.topic` into `MemoryQuery.text`). Each `QueryResult` carries `entry: Memory | Artifact` and `score: float`. The `SkillContext` should expose the resolved `list[Memory]`, `list[Artifact]`, and a `BudgetSnapshot` (from L3's `BudgetTrackerProtocol.remaining()`). The `serialized_context: str` field (pre-rendered by `ContextSerializer`) maps directly to what gets injected into `TextPayload.prompt` or `TextPayload.system_prompt`. This approach means skills never import from `execution_platform` — perfect layer isolation.

---

### DD-L1-06: Skill Composition Model

**Issue**: Complex cognitive tasks require multiple skills working together (e.g., "research → outline → write → review"). How do skills compose?

**Options**:

#### Option A — Skill Returns Multi-Step SkillPlan
A single skill can return a plan with multiple steps, some of which invoke other skills.

```python
async def plan(self, input, context):
    return SkillPlan(steps=[
        SkillStep(kind="text", config=research_config),
        SkillStep(kind="data", config=extract_config),
        SkillStep(kind="text", config=synthesis_config),
    ])
```

| Pros | Cons |
|------|------|
| Self-contained — one skill, one plan | Large skills become monolithic |
| Clear execution flow | Can't reuse individual steps across skills |
| | Not composable |

#### Option B — Skills Reference Other Skills (Skill Graph)
Skills can declare dependencies on other skills. The adapter resolves the graph.

```python
class WriteArticleSkill(Skill):
    dependencies = [ResearchSkill, OutlineSkill, ReviewSkill]

    async def plan(self, input, context):
        return SkillPlan(steps=[
            SkillRef("research", input=input),
            SkillRef("outline", input=FromPrevious()),
            SkillRef("write", input=FromPrevious()),
            SkillRef("review", input=FromPrevious()),
        ])
```

| Pros | Cons |
|------|------|
| Compositional — reuse skills | Dependency resolution complexity |
| Each skill is atomic and testable | Circular dependency risk |
| Natural for complex workflows | How does data flow between skills? |

#### Option C — Orchestrator Composes Skills (Layer 2 Concern)
Skills are always atomic. Composition happens at the orchestrator level — the PlanDAG wires skills together.

```python
dag = (
    DAGBuilder("article-pipeline")
    .add_skill_node("research", ResearchSkill, input_schema)
    .add_skill_node("outline", OutlineSkill)
    .add_skill_node("write", WriteSkill)
    .add_skill_node("review", ReviewSkill)
    .edge("research", "outline")
    .edge("outline", "write")
    .edge("write", "review")
    .build()
)
```

| Pros | Cons |
|------|------|
| Clean SRP — skills are atomic, orchestrator composes | Skills can't express their own composition preferences |
| DAG is inspectable and modifiable | Composition logic lives in a different layer |
| Reuse by wiring differently | Need a convenience layer for common patterns |
| Natural fit for the architecture | |

**Suggestion**: **Option C** — skill composition is an orchestrator concern. Skills stay atomic (single responsibility). The PlanDAG wires them together. Provide convenience builders (like `SkillPipeline`) that auto-generate linear or branching DAGs from skill lists. This is the cleanest separation and sets up perfectly for MetaSkills (which generate DAGs from skill descriptions).

> **DECISION**: **Option C** — Composition is an L2 orchestrator concern  
> **Comments**: L2 already has exactly the right primitives: `DAGBuilder` for constructing `PlanDAG`s with arbitrary node wiring, `FlowNode` subtypes (`ForEachNode`, `SwitchNode`, `FilterNode`, `IfNotOkNode`) for branching/looping, and `CompositeNode` (stub) for sub-DAG nesting. A `SkillPipeline` convenience in L1 would just be sugar that calls `DAGBuilder.add_node().edge()...build()` under the hood — mapping `[SkillA, SkillB, SkillC]` to a linear `PlanDAG`. For branching, `SwitchNode` already handles conditional routing; `ForEachNode` handles fan-out. The L2 `DAGOrchestrator.run(dag, data)` handles the full execution. Skills remain pure `(input, context) → plan` units. MetaSkills (V2) would generate `PlanDAG`s via `DAGBuilder` — the composition vocabulary is already in L2, no need to duplicate it in L1.

---

### DD-L1-07: Execution-Time Context References on SkillSteps

**Issue**: DD-L1-05 gives skills pre-loaded context at *planning* time via `SkillContext`. But individual steps within a `SkillPlan` may need *different* context at *execution* time — a research step needs background memories, a synthesis step needs the research step's output, a review step needs the original artifact. How do steps declare their execution-time context dependencies?

**Design**:

Each `SkillStep` carries an explicit `context_refs` list that declares what context the orchestrator should resolve and inject *before executing that step*.

```python
class ContextRef(BaseModel):
    """Declarative reference to context needed at execution time."""
    kind: Literal["memory", "artifact", "query", "previous_result"]
    id: str | None = None              # Direct memory/artifact ID
    slug: str | None = None            # Artifact by slug (+version)
    version: int | None = None         # Specific artifact version
    query: MemoryQuery | None = None   # Dynamic query
    from_step: str | None = None       # Result from a prior step's ID
    key: str = "context"               # Injection key in the prompt template

class SkillStep(BaseModel):
    id: str                            # Unique step ID (within the plan)
    kind: Literal["text", "data", "function", "tool"]
    config: StepConfig
    context_refs: list[ContextRef] = []

class SkillPlan(BaseModel):
    name: str
    steps: list[SkillStep]
    constraints: SkillConstraints | None = None
```

**ContextRef kinds**:

| Kind | Resolves Via | Use Case |
|------|-------------|----------|
| `memory` | `ContextStore.get(id)` | Pin a specific memory by ID |
| `artifact` | `ContextStore.get_artifact(slug, version)` | Pin an artifact (e.g., current draft) |
| `query` | `ContextStore.recall(query)` | Dynamic search (e.g., "related memories") |
| `previous_result` | `DAGTraversalState.node_results[from_step]` | Output of a prior step in this plan |

**How the adapter maps it**:

```
SkillStep.context_refs  →  BaseNode.ext["context_refs"]  (serialized)
                                   │
                        Orchestrator resolves before node execution:
                          1. memory/artifact → ContextStore.get() / get_artifact()
                          2. query → ContextStore.recall()
                          3. previous_result → node_results[from_step].value
                                   │
                        ContextSerializer renders resolved context
                                   │
                        Injected into prompt via ContextRef.key
```

The orchestrator resolves refs inside `DAGOrchestrator._run_node()` — read `node.ext["context_refs"]`, resolve each, serialize via `ContextSerializer`, and merge into the `data` dict under each ref's `key`. The node's prompt template uses `{context}` (or custom keys) to inject the resolved content.

**Relationship to DD-L1-05 (Pre-loaded SkillContext)**:

| | SkillContext (DD-L1-05) | ContextRef (DD-L1-07) |
|---|---|---|
| **When** | Plan time | Execution time |
| **Who builds it** | Framework, before `plan()` | Orchestrator, before each node |
| **What it carries** | Skill-level context (overview) | Step-level context (precise) |
| **Purpose** | Help skill decide *what* to plan | Give each step *exactly* what it needs |

These are complementary, not competing: the skill uses `SkillContext` to reason about what steps to create, then each step declares its own `context_refs` for execution.

> **DECISION**: **Adopted** — SkillSteps carry explicit `context_refs` for execution-time context  
> **Comments**: Maps cleanly to existing L2/L3 infrastructure. L2's `BaseNode.ext` dict carries the refs as metadata. L3's `ContextStore` provides all three resolution methods: `get(id)`, `get_artifact(slug, version)`, `recall(query)`. The `previous_result` kind leverages `DAGTraversalState.node_results` which L2 already maintains. The adapter translation is mechanical — serialize `context_refs` into `BaseNode.ext["context_refs"]`. The orchestrator needs one small addition: a context resolution hook in `_run_node()` that reads the refs, resolves them, and merges serialized context into `data`. This is the key V1 mechanism for data-passing in multi-step skills — making the plan self-describing and inspectable.

---

## 5. General Design Insights

### DI-L1-01: Skills Should Be Stateless

**Issue**: If skills hold state between invocations, they become hard to test, can't be reused concurrently, and break the time-travel model.

**Suggestion**: Skills are stateless. All state flows through input (SkillContext + input data) and output (SkillPlan/SkillResult). Configuration is immutable and set at construction time. Any persistent state goes into the ContextStore as memories/artifacts.

> **DECISION**: **Agreed** — Skills must be stateless  
> **Comments**: Consistent with both L3 and L2 patterns. L3's `ExecutionState` owns all mutable state centrally — nothing else holds state. L3 handlers (`TextHandler`, etc.) are stateless dispatchers. L2's `DAGOrchestrator` tracks execution state in `ExecutionDAG` (append-only) and delegates to `ExecutionState` for snapshots — nodes themselves are immutable Pydantic `BaseModel`s. Skills should follow the same pattern: immutable config at construction, all dynamic state flows through `SkillContext` (in) and `SkillResult` (out). The `SkillResult.suggested_memories` and `suggested_artifacts` fields are the correct way to persist state — routed through `ContextStore.remember()` / `store()` by the framework, never by the skill directly.

---

### DI-L1-02: SkillResult Should Carry Provenance

**Issue**: When a skill produces a result, that result may become a memory or artifact. It needs provenance (which skill, which input, which context was used) to be traceable.

**Suggestion**: SkillResult includes provenance metadata that maps directly to the Memory/Artifact `provenance` field.

```python
class SkillResult(BaseModel):
    output: BaseModel                # The actual result
    success: bool
    provenance: SkillProvenance      # Which skill, input hash, context snapshot
    suggested_memories: list[CreateMemory]   # Memories to store from this result
    suggested_artifacts: list[CreateArtifact] # Artifacts to store
```

This lets the system automatically build the memory graph: "this artifact was produced by skill X with these input memories."

> **DECISION**: **Agreed** — SkillResult carries provenance  
> **Comments**: Maps directly to existing L3 models. `Memory` has `provenance: Provenance` (with `origin: MemoryOrigin` and `source: str`). `Artifact` has `provenance: ArtifactProvenance` (with `input_memory_ids: list[ID]` and `intent: str`). `SkillResult.suggested_memories` should produce `Memory` objects with `provenance.origin = INFERRED` and `provenance.source` set to the skill name. `SkillResult.suggested_artifacts` should produce `Artifact` objects with `provenance.input_memory_ids` populated from the `SkillContext.memories` that were used, and `provenance.intent` describing the skill's goal. The `SkillProvenance` type on `SkillResult` should include `skill_name: str`, `input_hash: str`, and `context_memory_ids: list[ID]` — enabling full traceability through the L3 store. The framework stores these via `ContextStore.remember()` / `store()` after `interpret()` returns.

---

### DI-L1-03: MetaSkill — Define the Interface Now

**Issue**: MetaSkills are V2, but the interface should be designed now to avoid breaking changes.

**Suggestion**: MetaSkill extends Skill with additional methods for generating skills/DAGs from descriptions.

```python
class MetaSkillProtocol(SkillProtocol, Protocol):
    async def generate_skill(self, description: str, context: SkillContext) -> SkillConfig: ...
    async def generate_dag(self, description: str, context: SkillContext) -> PlanDAG: ...
```

V1 implementation: raise `NotImplementedError`. The interface exists so that V2 can implement it without changing the protocol.

> **DECISION**: **Agreed** — Define MetaSkill interface now, stub for V1  
> **Comments**: The `generate_dag()` method should return a `PlanDAG` (L2 type) — this is the one place where L1 legitimately references L2 types, since MetaSkills are a bridge between cognitive intent and orchestration structure. However, for V1, keep it behind a `NotImplementedError` as suggested. The `generate_skill()` method returning `SkillConfig` is clean — `ConfigSkill` (from DD-L1-01) would consume it. L2's `CompositeNode` (currently a stub with `sub_dag: Any`) is the natural L2-side counterpart for MetaSkill-generated sub-DAGs in V2.

---

### DI-L1-04: Token Budget Awareness

**Issue**: LLM calls have token limits. The Cognitive Layer must be budget-aware: how much context to inject, how long a prompt can be, how much output to request.

**Suggestion**: `SkillContext` includes a `TokenBudget` (from the SPEC's type system). The `ContextSerializer` respects this budget when rendering. Skills can query `context.budget.remaining` to adjust their strategy.

```python
class SkillContext:
    memories: list[Memory]
    artifacts: list[Artifact]
    budget: TokenBudget
    serialized_context: str       # Pre-rendered by ContextSerializer within budget
    remaining_prompt_tokens: int  # Budget minus serialized_context
```

> **DECISION**: **Agreed** — SkillContext includes TokenBudget and pre-serialized context  
> **Comments**: L3 provides `TokenBudget` (with `total`, `working`, `episodic`, `semantic`, `procedural`, `artifacts` fields) and `BudgetSnapshot` (with `tokens_remaining`, `calls_remaining`, `time_remaining_seconds`). `SkillContext` should carry both: `TokenBudget` for prompt-construction budgeting (how much context to serialize) and `BudgetSnapshot` for execution-awareness (can we afford more LLM calls). The `ContextSerializer` should respect `TokenBudget` category limits when rendering — e.g. episodic memories get at most `budget.episodic` tokens, semantic memories get `budget.semantic` tokens. The `remaining_prompt_tokens` field is derived: `budget.total - count_tokens(serialized_context)`. This requires a `count_tokens` utility — which belongs on the `LLMProtocol` or as a standalone helper (aligns with Option C's `count_tokens` method from DD-L1-02, to add later).

---

## 6. Implementation Phases

Development follows the overview spec's Phase 3 (Cognitive Layer), broken into explicit sub-phases with precise deliverables and test gates. Each phase produces independently testable output. **Do not start a phase until the previous phase's tests pass.**

---

### Phase 3.1 — Protocols & Models (Foundation)
> **Goal**: All type contracts — every other file imports from these

| Step | File | Deliverable | Depends On |
|------|------|-------------|------------|
| 3.1.1 | `cognitive/protocols.py` | `SkillProtocol` ABC (`plan`, `interpret`, `validate_output`), `LLMProtocol` (3 methods: `complete`, `complete_structured`, `complete_with_tools`), `PromptProtocol`, `ContextSerializerProtocol`, `MetaSkillProtocol` (V2 stub) | L3 types, L2 types (for adapter protocol only) |
| 3.1.2 | `cognitive/models.py` | `ContextRef`, `StepConfig` (text/data/function/tool configs), `SkillStep`, `SkillConstraints`, `SkillPlan`, `SkillContext`, `SkillProvenance`, `SkillResult`, `ReplanRequest`, `CreateMemory`, `CreateArtifact`, `SkillConfig`, `BuiltPrompt`, `CompletionResult`, `ToolResult`, `Message` | 3.1.1 |

**Test gate**: Unit tests for all models (Pydantic validation, serialization round-trips, defaults). Unit tests for protocol shape (structural subtyping checks).

```
tests/cognitive/
├── test_models.py          # Model construction, validation, defaults, serialization
└── test_protocols.py       # Protocol structural checks, ABC enforcement
```

---

### Phase 3.2 — LLM Abstraction
> **Goal**: Provider-agnostic LLM interface + mock implementation for testing

| Step | File | Deliverable | Depends On |
|------|------|-------------|------------|
| 3.2.1 | `cognitive/llm.py` | `LLMProtocol` concrete reference: `MockLLM` (returns canned responses, records calls). Optional: `StreamingLLMProtocol`, `EmbeddingProtocol` as separate ABCs (empty stubs). | 3.1.1, 3.1.2 |

**Test gate**: `MockLLM` satisfies `LLMProtocol`. `complete()` returns `CompletionResult` aligned with L3's `LLMResultData`. `complete_structured()` returns typed output. `complete_with_tools()` returns `ToolResult`. Token usage tracking works.

```
tests/cognitive/
└── test_llm.py             # MockLLM protocol compliance, call recording, token usage
```

---

### Phase 3.3 — Skill Base Class
> **Goal**: Core Skill ABC — plan/interpret split with optional validation hook

| Step | File | Deliverable | Depends On |
|------|------|-------------|------------|
| 3.3.1 | `cognitive/skill.py` | `Skill` ABC: `plan(input, context) -> SkillPlan`, `interpret(result) -> SkillResult`, optional `validate_output(output) -> ValidationResult`. Skill metadata: `name`, `description`, `memory_query` (for SkillContext building). `RetryableValidationError(TransientError)` for output validation retries. | 3.1.1, 3.1.2 |

**Test gate**: Concrete test skill implementing `Skill`. Test `plan()` returns valid `SkillPlan` with `SkillStep`s and `context_refs`. Test `interpret()` returns valid `SkillResult` with provenance. Test `validate_output()` raises `RetryableValidationError` on bad output. Test statelessness — skill produces consistent output with same input.

```
tests/cognitive/
└── test_skill.py           # ABC enforcement, test skill implementation, validation hook
```

---

### Phase 3.4 — Prompt Engine
> **Goal**: PromptBuilder (programmatic) + str.format (templates) → `BuiltPrompt`

| Step | File | Deliverable | Depends On |
|------|------|-------------|------------|
| 3.4.1 | `cognitive/prompt.py` | `PromptBuilder`: chainable `.system()`, `.context()`, `.user()`, `.build() -> BuiltPrompt`. `TemplateRenderer`: `str.format`-based renderer for `ConfigSkill` templates. Both produce `BuiltPrompt(system_prompt: str | None, prompt: str)` matching L3's `TextPayload` fields. | 3.1.2 |

**Test gate**: Builder produces correct `system_prompt` + `prompt` strings. Context injection places serialized context in the right position. Template renderer fills `{input.field}` and `{context}` placeholders. Edge cases: empty context, no system prompt, multiple `.user()` calls.

```
tests/cognitive/
└── test_prompt.py          # PromptBuilder, TemplateRenderer, BuiltPrompt shape
```

---

### Phase 3.5 — Context Serializer
> **Goal**: Render memories/artifacts into text for prompt injection

| Step | File | Deliverable | Depends On |
|------|------|-------------|------------|
| 3.5.1 | `cognitive/serializer.py` | `NaiveSerializer` (Stage 1): concatenate `Memory.content.text` and `Artifact.content.text` with headers. `SectionSerializer` (Stage 2): group by `MemoryRole` with markdown headers. Both implement `ContextSerializerProtocol`. Input: `list[Memory]`, `list[Artifact]`. Output: `str`. | 3.1.1, 3.1.2 |

**Test gate**: Naive serializer joins entries with headers. Section serializer groups by role (`EPISODIC`, `SEMANTIC`, `PROCEDURAL`, `WORKING`). Empty inputs produce empty string. Artifact formatting includes `type` and `slug`. Round-trip: serialized output can be injected into `PromptBuilder.context()` and appears in `BuiltPrompt`.

```
tests/cognitive/
└── test_serializer.py      # NaiveSerializer, SectionSerializer, edge cases
```

---

### Phase 3.6 — L1→L2 Adapter
> **Goal**: Translate SkillPlan → PlanDAG and OrchestratorResult → interpret-ready format

| Step | File | Deliverable | Depends On |
|------|------|-------------|------------|
| 3.6.1 | `cognitive/adapters.py` | `SkillToDAGAdapter.to_dag(plan: SkillPlan) -> PlanDAG`: maps `SkillStep` → L2 node types (`TextNode`, `DataNode`, `FunctionNode`, `ToolNode`), wires sequential edges via `DAGBuilder`, serializes `context_refs` into `BaseNode.ext["context_refs"]`, maps constraints to `timeout_seconds`/`max_retries`. `ResultAdapter.from_result(execution_dag: ExecutionDAG) -> OrchestratorResult`: extracts `NodeResult` values by step ID for `interpret()`. | 3.1.2, L2 (`DAGBuilder`, node types, `ExecutionDAG`) |

**Test gate**: Single-step plan produces single-node PlanDAG. Multi-step plan produces linear DAG with correct edges. `context_refs` appear in `BaseNode.ext`. Step kinds map to correct node types. Constraints propagate. `from_result()` correctly extracts node results.

```
tests/cognitive/
└── test_adapters.py        # to_dag mapping, edge wiring, context_refs, from_result
```

---

### Phase 3.7 — L2 Context Resolution Hook
> **Goal**: Enable `DAGOrchestrator` to resolve `context_refs` before node execution

| Step | File | Deliverable | Depends On |
|------|------|-------------|------------|
| 3.7.1 | `orchestrator/dag_orchestrator.py` (extension) | Add context resolution between validation (step 3) and execution (step 4) in `_run_node()`: read `node.ext.get("context_refs")`, resolve each `ContextRef` via `ContextStore.get()` / `get_artifact()` / `recall()` / `node_results[]`, serialize via `ContextSerializer`, merge into `data` dict under each ref's `key`. Requires `ContextStoreProtocol` and `ContextSerializerProtocol` as optional constructor args. | Phase 3.5, Phase 3.6, L2 `_run_node()`, L3 `ContextStore` |

**Test gate**: Node with `context_refs` in `ext` receives resolved context in `data`. `memory` refs resolve via store. `artifact` refs resolve by slug/version. `query` refs resolve via `recall()`. `previous_result` refs resolve from `node_results`. Missing refs produce clear errors. Nodes without `context_refs` work unchanged (backward compatible). **All existing 830 L2/L3 tests still pass.**

```
tests/orchestrator/
└── test_context_resolution.py  # ContextRef resolution in _run_node, backward compat
```

---

### Phase 3.8 — Built-in Skills
> **Goal**: Concrete skill implementations for common LLM patterns

| Step | File | Deliverable | Depends On |
|------|------|-------------|------------|
| 3.8.1 | `cognitive/builtin_skills.py` | `TextGenerationSkill`: single `text` step, takes prompt + system prompt. `DataExtractionSkill`: single `data` step, takes prompt + output schema. `CodeGenerationSkill`: text step with code-oriented system prompt. `ReviewSkill`: text step that produces pass/fail + feedback, uses `validate_output()`. | 3.3.1, 3.4.1, 3.5.1 |
| 3.8.2 | `cognitive/skill.py` (addition) | `ConfigSkill(Skill)`: auto-generates `plan()` and `interpret()` from `SkillConfig` data. `plan()` builds prompt via `TemplateRenderer`, creates single `SkillStep`. `interpret()` validates output against `output_schema` via Pydantic. | 3.8.1 |

**Test gate**: Each built-in skill produces correct `SkillPlan` from `plan()`. `interpret()` returns typed `SkillResult`. `ReviewSkill.validate_output()` catches bad output. `ConfigSkill` auto-generates from `SkillConfig` — zero custom code. ConfigSkill Pydantic output validation works.

```
tests/cognitive/
├── test_builtin_skills.py  # Each skill's plan/interpret, validate_output
└── test_config_skill.py    # ConfigSkill from SkillConfig, auto plan/interpret
```

---

### Phase 3.9 — MetaSkill Stub
> **Goal**: V2 interface — define now, implement later

| Step | File | Deliverable | Depends On |
|------|------|-------------|------------|
| 3.9.1 | `cognitive/meta_skill.py` | `MetaSkill(Skill)`: extends `Skill` with `generate_skill(description, context) -> SkillConfig` and `generate_dag(description, context) -> PlanDAG`. V1: both raise `NotImplementedError`. | 3.3.1 |

**Test gate**: `MetaSkill` is importable, satisfies `MetaSkillProtocol`. Both methods raise `NotImplementedError`. Interface is structurally compatible with `SkillProtocol`.

```
tests/cognitive/
└── test_meta_skill.py      # Interface check, NotImplementedError
```

---

### Phase 3.10 — Integration Tests
> **Goal**: Full pipeline — Skill → Adapter → DAGOrchestrator → Execution Platform

| Step | File | Deliverable | Depends On |
|------|------|-------------|------------|
| 3.10.1 | Integration: Skill → Adapter → PlanDAG | Verify a skill's `plan()` output translates into a valid `PlanDAG` that `DAGOrchestrator` accepts. Test with `TextGenerationSkill` and multi-step custom skill. | All above |
| 3.10.2 | Integration: Context resolution round-trip | Skill declares `context_refs` → adapter maps to `ext` → orchestrator resolves from `ContextStore` → handler receives resolved data. | Phase 3.7 |
| 3.10.3 | Integration: Full pipeline with MockLLM | Skill → Adapter → DAGOrchestrator → L3 handlers (with MockLLM wired into `TextHandler`) → `ExecutionDAG` → `ResultAdapter.from_result()` → Skill.`interpret()` → `SkillResult`. | All above |
| 3.10.4 | Integration: Replan flow | Skill returns `SkillResult` with `replan` set → framework builds new `PlanDAG` → re-executes. | All above |
| 3.10.5 | Integration: Output validation retry | Skill's `validate_output()` rejects output → `RetryableValidationError` → `RetryPolicy` retries → passes on second attempt. | All above |

**Test gate**: All integration tests pass. All 830+ existing L2/L3 tests still pass. No regressions.

```
tests/cognitive/
└── test_integration.py     # Full pipeline, context resolution, replan, validation retry
```

---

### Phase Summary

| Phase | What | Files | Test Files |
|-------|------|-------|------------|
| **3.1** | Protocols + Models | `protocols.py`, `models.py` | `test_models.py`, `test_protocols.py` |
| **3.2** | LLM Abstraction | `llm.py` | `test_llm.py` |
| **3.3** | Skill Base Class | `skill.py` | `test_skill.py` |
| **3.4** | Prompt Engine | `prompt.py` | `test_prompt.py` |
| **3.5** | Context Serializer | `serializer.py` | `test_serializer.py` |
| **3.6** | L1→L2 Adapter | `adapters.py` | `test_adapters.py` |
| **3.7** | L2 Context Hook | `dag_orchestrator.py` (ext) | `test_context_resolution.py` |
| **3.8** | Built-in + Config Skills | `builtin_skills.py`, `skill.py` (add) | `test_builtin_skills.py`, `test_config_skill.py` |
| **3.9** | MetaSkill Stub | `meta_skill.py` | `test_meta_skill.py` |
| **3.10** | Integration | — | `test_integration.py` |

---

## 7. Testing Strategy

| Level | What | How |
|-------|------|-----|
| **Unit** | Skill plan/interpret methods | Mock SkillContext, verify SkillPlan/SkillResult |
| **Unit** | PromptBuilder | Verify prompt construction |
| **Unit** | ContextSerializer | Mock memories/artifacts, verify rendered output fits budget |
| **Unit** | LLM abstraction | Mock provider, verify call translation |
| **Integration** | Skill → Adapter → PlanDAG | Real adapter, verify DAG structure |
| **Integration** | Full pipeline: Skill → Orchestrator → Execution | Real components, mock LLM |
| **E2E** | Full pipeline with real LLM | Smoke tests only (expensive) |

---

## 8. Open Questions

- [x] **Should skills have a `validate_output()` hook for output quality checks?**
  > **RESOLVED: YES.** Add an optional `validate_output(result) -> ValidationResult` hook on `Skill`. When validation fails (`ok=False`), the framework re-emits the underlying event — which L3's `RetryPolicy` already handles (exponential backoff, `max_attempts` cap). The hook runs inside `interpret()`: parse result → validate → if invalid, raise a `ValidationError` (L3's existing error type, `retryable=False` by default). To make it retryable, introduce a `RetryableValidationError(TransientError)` so `RetryPolicy` picks it up. `ConfigSkill` gets a built-in validator that checks the output against `output_schema` via Pydantic — free for all declarative skills.

- [x] **How does multi-turn conversation fit into the skill model?**
  > **RESOLVED: PlanDAG chain with data-passing.** Agreed — the "chain of skills" maps to a `PlanDAG` with linear edges. Each node's result flows to the next via `DAGOrchestrator`'s `data` dict (L2 already passes `data` to each node). For the "plan → execute → review → pass to next" pattern: the review node is an `IfNotOkNode` — if the review skill flags issues, `IfNotOkNode.redirect_to` points back to the generation node (retry loop within the DAG). Data flows between nodes through `NodeResult.value` stored in the `ExecutionDAG`, accessible to downstream nodes via `DAGTraversalState.node_results[node_id]`. No new mechanism needed — this is pure L2 wiring.

- [x] **Should ConfigSkill support conditional steps (if/else in the template)?**
  > **RESOLVED: NO conditionals in templates — delegate to L2.** Agreed with your annotation. `ConfigSkill` templates stay simple (`str.format` only). If a skill needs branching, it produces a `SkillPlan` with multiple steps, and the adapter maps them to L2 `SwitchNode` or `IfNotOkNode`. This keeps ConfigSkill trivial and leverages L2's existing flow control vocabulary. Adding template conditionals would duplicate `SwitchNode` logic inside L1 — violating layer separation.

- [x] **How are skill versions managed? Can you pin a skill version in a DAG?**
  > **RESOLVED: Use L3's Artifact versioning model.** L3's `Artifact` already supports versioning: `slug` (unique name), `version` (auto-incremented), `supersedes` (previous version ID), and `ArtifactStatus` (ACTIVE/DEPRECATED). A skill's `SkillConfig` can be stored as an `Artifact(type=ArtifactType.SKILL, slug="summarize-v1")`. Pinning in a DAG: the `BaseNode.ext` field (L2) can carry `{"skill_slug": "summarize", "skill_version": 2}` — the adapter resolves the exact skill config via `ArtifactStore.get_by_slug(slug, version)` before building the plan. Latest active version is the default (matching `ArtifactStore` behavior). No new versioning system needed — L3 already has it.

- [x] **Should the LLM abstraction support batched calls?**
  > **RESOLVED: NO.** Not for V1. L2's `DAGOrchestrator` already runs independent nodes in parallel via `asyncio.gather` — natural batching without a batched API. If a provider supports native batching later, add it as a separate `BatchLLMProtocol` (ISP, same pattern as `StreamingLLMProtocol`).

- [x] **How do skills signal to the orchestrator that they want to replan?**
  > **RESOLVED: `SkillResult` carries a `replan` signal.** Add an optional `replan: ReplanRequest | None` field to `SkillResult`. When `interpret()` determines the approach needs to change (e.g., research revealed a different angle), it returns a `ReplanRequest(reason: str, suggested_steps: list[SkillStep] | None)`. The framework — not the skill — handles it: if `replan` is set, the adapter builds a new `PlanDAG` from the suggested steps and the `DAGOrchestrator` runs it. This keeps skills pure (they just return data) while enabling replanning. The L2 `DAGOrchestrator` already supports interruption via `interrupt()` — replan means: interrupt current DAG, build new DAG, run it. For V1, keep this simple: replan = "abort and re-run with new plan". V2 can support partial replanning (modify remaining nodes in-flight).
