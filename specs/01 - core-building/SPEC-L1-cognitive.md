# Layer 1 — Cognitive: Development Specification

> **Parent**: [SPEC-overview.md](SPEC-overview.md)  
> **Layer**: 1 (Top)  
> **Status**: Design Phase  
> **Last Updated**: 2026-03-10  
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

> **DECISION**: `___________`  
> **Comments**: `___________`

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

> **DECISION**: `___________`  
> **Comments**: `___________`

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

> **DECISION**: `___________`  
> **Comments**: `___________`

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

> **DECISION**: `___________`  
> **Comments**: `___________`

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

> **DECISION**: `___________`  
> **Comments**: `___________`

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

> **DECISION**: `___________`  
> **Comments**: `___________`

---

## 5. General Design Insights

### DI-L1-01: Skills Should Be Stateless

**Issue**: If skills hold state between invocations, they become hard to test, can't be reused concurrently, and break the time-travel model.

**Suggestion**: Skills are stateless. All state flows through input (SkillContext + input data) and output (SkillPlan/SkillResult). Configuration is immutable and set at construction time. Any persistent state goes into the ContextStore as memories/artifacts.

> **DECISION**: `___________`  
> **Comments**: `___________`

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

> **DECISION**: `___________`  
> **Comments**: `___________`

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

> **DECISION**: `___________`  
> **Comments**: `___________`

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

> **DECISION**: `___________`  
> **Comments**: `___________`

---

## 6. Implementation Priority

| Priority | Component | Rationale |
|----------|-----------|-----------|
| **P0** | `protocols.py` + `models.py` | Contracts — everything depends on these |
| **P0** | `llm.py` | LLM abstraction — skills need this to express work |
| **P0** | `skill.py` | Core Skill ABC |
| **P1** | `prompt.py` (PromptBuilder + str.format templates) | Prompt construction |
| **P1** | `serializer.py` (NaiveSerializer) | Render context for prompts |
| **P1** | `adapters.py` (L1→L2) | Connect to orchestrator |
| **P2** | `builtin_skills.py` | TextGeneration, DataExtraction, CodeGeneration |
| **P2** | ConfigSkill (declarative skill-from-config) | Convenience layer |
| **P3** | `meta_skill.py` | V2 stub — interface only |

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

- [ ] Should skills have a `validate_output()` hook for output quality checks?
- [ ] How does multi-turn conversation fit into the skill model? (skill per turn? Long-running skill?)
- [ ] Should ConfigSkill support conditional steps (if/else in the template)?
- [ ] How are skill versions managed? Can you pin a skill version in a DAG?
- [ ] Should the LLM abstraction support batched calls (multiple prompts in one API call)?
- [ ] How do skills signal to the orchestrator that they want to replan? (e.g., "the research step revealed we need a different approach")
