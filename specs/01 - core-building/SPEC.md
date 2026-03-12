# rh Congnitv

## Layers

1. Congnitive
    skills, metaskills etc..

2. Orchestrator
    - Stratgegy, scheduling, policies, authorizations
    - DAG builder
    - ExecutionDAG uses ExecutionPlatform. 

3. Execution Platform
    - EventBus: unified event for emit and broadCast events
    - EventExecutor: class which execute the event, they cast events        
    - event: ExecutionEvent[status: running, success, failed], InterrupEvent, NotificationEvent[wanings, infra, llm_context... usefull metrics... para as 3 Camadas...]
    - Logs and Traces: Special eventHandle which records logs somewhere
    - Timeouts / Retry / Budget / Interrupt: EventSuff...
    - Memory and Artifacts

```
┌─────────────────────────────────────────┐
│           Cognitive Layer               │  ← LLM reasoning, planning, review
├─────────────────────────────────────────┤
│         Orchestrator Layer              │  ← Strategy, scheduling, policy nuance
├─────────────────────────────────────────┤
│         Execution Platform              │  ← Shared, reusable across all orchestrators
│  ┌──────────┬──────────┬─────────────┐  │
│  │  Retry / │ Memory & │  Log &      │  │
│  │  Budget  │Artifacts │  Traces     │  │
│  ├──────────┼──────────┼─────────────┤  │
│  │Execution │ Skills & │  Other...   │  │
│  │  Nodes   │Capabilities             │  │
│  └──────────┴──────────┴─────────────┘  │
└─────────────────────────────────────────┘
```

ORC
```python
# this is the plan DAG. The idea is to have a plan dag, which the orchestrator will follow as guideline, but the execution plataform will create its on DAG
# called ExecutionDAG which represents the order of execution (in the view of execution), so branches here might mean parallel execution, but not all parallel execution need to leave in a branch (about loop branches, it might live on single Node)
DAG = {
    'nodes': {
        'node1': DagNode('1'),
        'node2': DagNode('2')
    },
    edges: [
        {from: 'node1', 'to': 'node2'}
    ]
}

orch = DAGOrchestor(DAG)

class BaseNode:
    ...

class ExecutionNode(BaseNode):
    ... # types or extend -> TextNode, DataNode, FunctionNode, ToolNode

class FlowNode(BaseNode):
    ... # types or extend -> ForEach(data_list, node) -> list_data, Filter(data_list, mapper) -> list_data, get(data, key) -> data, switch(data, Node) -> Node IfNotOkResult

class DAGOrchestrator:
    def __init__(self, dag, state):
        ...
        

    def __call__(self, input_data: BaseModel):
        self.state.load_snapshot()
        initial_data = self.state.get_current_input(input_data) # EntryRef | BaseModel
        return self.run_next(initial_data)
        
    def run_next(input_data: EntryRef | BaseModel)
        if not self.DAG.has_next():
            return input_data
        node = self.DAG.get_next_node()
        result = self.run_node(node, input_data)
        self.run_next(result)

    def run_node(self, node, main_input: BaseEntryRef):
        validate_node_step(node, main_input) # policy, dependencies, authorizations, input etc...
        event = self.node_to_event(node) # talvez node.event
        # Isso tem que ir para dentro do handler the ForEachFlowNode.... 
        if isinstance(event, FlowNode[ForEach]):
            async with self.state.parallel(name='ddd') as p:
                for idx, data in enumerate(main_input):
                    execution_event = ExecutionEvent(event, self.state, tag={idx})
                    evx = execution_event(main_input, node.execution_configs() or self.execution_configs())
                    p.add(evx)
                return p.gather()
        execution_event = ExecutionEvent(event, self.state)
        result = execution_event(main_input, node.execution_configs() or self.execution_configs())
        return result.result


class TextExecutionEvent:
    # TextExecutionEvent::LLM Call [content]

class DataExecutionEvent:
    # EDataExecutionEvent::LLM Call [function_call]

class FunctionExecutionEvent:
    # FunctionEvecutionEvent::Funcion Call

class ToolExecutionEvent:
    # ToolExecutionEvent::LLM Call
    #                   ::Function Call
    # DataExecutionEvent + FunctionEvecutionEvent

class ExecutionState:
    state {
        execution_dag: ExecutionDag
        execution_history[executionRef] -> referência para UNDO REDO... 
        current_node: NodeRef
        last_executed_node: NodeRef
        memory_store: MemoryStore
        artifact_store: ArtifactStore
    } # para cada... snapshot com banco de elementos imutáveis, somente uma árvore de dependências
    # a ideia é poder navegar no tempo e ter o exato snapshot. Isso permite UNDO, REDO stuff
    # de todo o estado, inclusive das memórias e artefatos. 
    eventBus: EventBus
    


class ExecutionEvent:
    def __init__(self, event_element, state):
        ...

    def __call__(self, data: entry_ref | baseModel, confgs) ExecutionResult[EntryRef | BaseModel]:
        self.state.add_level()            
        execution_dependencies_entry = self.state.get_dependencies_entry_ref(data if entry_ref else base_model)
        try:
            result = self.handler(execution_dependencies_entry, confgs)
        except:
            #retry stuff
            ...

        # castEvent
        state_keys_from_result = self.get_state_keys_from_result(result)
        self.state.set_dependencis(state_keys_from_result)
        if not result.ok:
            raise... checkers...
        state.remove_level()
        return result
        # It migth return
        return BaseEntryRef | BaseRef

    def group(self, configs: exec_configs): AsyncEventWIthEsqueciONome
        # Se um dos eventos resultar em erro, ele aplica o retry do início. 
        async with AsyncEventWIthEsqueciONome as ev:
            ev.add(event)
            ev.gather()


## Memory and Artifact
```
/**
 * CONTEXT SYSTEM — Core Types
 * 
 * Design principles:
 *  1. Every field present at Stage 1 is useful at Stage 1 — no premature fields
 *  2. `ext` is the forward-compat escape hatch — promotes to first-class over time
 *  3. Text is always the ground truth — `content.text` is never optional
 *  4. IDs are stable ULIDs — survive backend migrations
 *  5. Discriminated union on `kind` — Memory and Artifact share the same store
 */

// ─────────────────────────────────────────────
// PRIMITIVES
// ─────────────────────────────────────────────

/** Stable ULID string — e.g. "01HXYZ3K..." */
type ID = string

/** ISO-8601 timestamp string — serializes cleanly to JSON */
type Timestamp = string

/**
 * The escape hatch that makes Stage 1 forward-compatible with Stage 5.
 * Unknown fields live here until they're used enough to be promoted
 * to first-class typed fields.
 * 
 * Rule: if you write to the same ext key 3+ times, promote it.
 */
type Ext = Record<string, unknown>


// ─────────────────────────────────────────────
// SHARED FOUNDATIONS
// ─────────────────────────────────────────────

/**
 * Every entry in the store — Memory or Artifact — shares this base.
 * The discriminated `kind` field drives type narrowing everywhere.
 */
interface BaseEntry {
  /** Stable identity — never changes after creation */
  id: ID

  /** Discriminator — drives all type narrowing */
  kind: 'memory' | 'artifact'

  /**
   * Content — text is ALWAYS present.
   * This is what gets serialized into the LLM prompt.
   * Everything else is retrieval/filtering infrastructure.
   */
  content: EntryContent

  /** When this entry was created in the store */
  createdAt: Timestamp

  /** When this entry was last modified */
  updatedAt: Timestamp

  /**
   * Free-form tags for filtering and grouping.
   * Stage 1 retrieval runs on this before you have a vector index.
   */
  tags: string[]

  /**
   * Forward-compatibility bag.
   * Anything that doesn't fit the typed fields yet goes here.
   * Promoted to first-class when usage patterns stabilize.
   * 
   * @example
   * ext: { importance: 0.9, project: "auth-module", threadId: "t_01HX" }
   */
  ext: Ext
}

interface EntryContent {
  /** Ground truth — always present, serializes to LLM prompt */
  text: string

  /**
   * Optional short description for budget-constrained contexts.
   * If absent, consumers truncate `text` themselves.
   * Promote from ext when you implement the serialization layer.
   */
  summary?: string

  /**
   * MIME-like format hint for the serializer.
   * 
   * @example 'text/plain' | 'text/markdown' | 'code/typescript' | 'data/json'
   */
  format?: string
}


// ─────────────────────────────────────────────
// MEMORY
// ─────────────────────────────────────────────

/**
 * What cognitive role does this memory play?
 * 
 * episodic   — something that happened ("user rejected verbose output at 14:03")
 * semantic   — something believed to be true ("user prefers TypeScript")
 * procedural — how to do something (a skill, a pattern)
 * working    — active right now, ephemeral (survives the task, not the session)
 */
type MemoryRole = 'episodic' | 'semantic' | 'procedural' | 'working'

/**
 * What is the shape of the content?
 * Orthogonal to role — any role can have any shape.
 * 
 * atom      — one indivisible claim or event
 * sequence  — ordered list of atoms
 * summary   — lossy compression of multiple memories
 * narrative — atoms connected by causality ("because", "therefore")
 */
type MemoryShape = 'atom' | 'sequence' | 'summary' | 'narrative'

/**
 * Where did this memory come from, and how much should we trust it?
 * 
 * observed   — directly witnessed (tool output, file read)
 * told       — user explicitly stated it
 * inferred   — LLM derived it from other signals
 * consolidated — merged/summarized from other memories
 */
type MemoryOrigin = 'observed' | 'told' | 'inferred' | 'consolidated'

interface Memory extends BaseEntry {
  kind: 'memory'

  /** Cognitive role — drives retrieval routing and decay policy */
  role: MemoryRole

  /** Content shape — drives serialization strategy */
  shape: MemoryShape

  /**
   * Provenance — where did this come from?
   * Kept minimal: just what you need to reason about trust.
   * 
   * Promote from ext:
   *   - derivedFrom: ID[]          (when you build consolidation)
   *   - confidence: number         (when you build conflict resolution)
   *   - sessionId: ID              (when you build session scoping)
   */
  provenance: {
    /** How this memory came to exist */
    origin: MemoryOrigin

    /**
     * Free-text source description.
     * Stage 1: "user message", "tool:file_read", "llm inference"
     * Stage 3+: promote to typed MemorySource union
     */
    source: string
  }

  /**
   * Temporal validity.
   * 
   * `recordedAt` — when we stored it (always set)
   * `observedAt` — when it actually happened (can differ from recordedAt)
   *                Set to recordedAt if unknown.
   * 
   * Promote from ext:
   *   - validUntil: Timestamp      (explicit expiry)
   *   - decay: DecayPolicy         (when you build salience scoring)
   */
  time: {
    recordedAt: Timestamp
    observedAt: Timestamp
  }
}


// ─────────────────────────────────────────────
// ARTIFACT
// ─────────────────────────────────────────────

/**
 * What kind of produced thing is this?
 * 
 * code       — executable output
 * document   — human-readable output  
 * data       — structured data payload
 * skill      — procedural memory in artifact form (SKILL.md pattern)
 * plan       — agent execution blueprint
 * prompt     — reusable prompt template
 */
type ArtifactType = 'code' | 'document' | 'data' | 'skill' | 'plan' | 'prompt'

/**
 * Where is this artifact in its lifecycle?
 * 
 * draft      — being built, not yet usable
 * active     — current, use this
 * deprecated — superseded but kept for reference
 * archived   — execution complete (plans), or retired
 */
type ArtifactStatus = 'draft' | 'active' | 'deprecated' | 'archived'

interface Artifact extends BaseEntry {
  kind: 'artifact'

  /** What type of produced thing this is */
  type: ArtifactType

  /**
   * Human-readable stable name.
   * Used to retrieve artifacts by name rather than ID.
   * 
   * @example 'auth-module', 'onboarding-skill', 'data-pipeline-plan'
   */
  slug: string

  /** Monotonic version counter — increments on every update */
  version: number

  /** Lifecycle status */
  status: ArtifactStatus

  /**
   * Provenance — the bridge between artifact and memory.
   * "What memories were consumed to produce this?"
   * 
   * Promote from ext:
   *   - buildTrace: BuildStep[]    (when you build orchestrator tracing)
   *   - constraints: string[]      (when you build constraint tracking)
   *   - quality: ArtifactQuality   (when you build validation)
   */
  provenance: {
    /**
     * IDs of memories that informed this artifact's creation.
     * This is what lets you answer: "why was this built this way?"
     */
    inputMemoryIds: ID[]

    /**
     * Human/agent-readable intent.
     * @example "implement JWT auth per user request in session X"
     */
    intent: string
  }

  /**
   * ID of the artifact this supersedes, if any.
   * Forms a linked list of versions over time.
   */
  supersedes?: ID
}


// ─────────────────────────────────────────────
// STORE INTERFACE
// ─────────────────────────────────────────────

/** Discriminated union — the store holds both */
type StoreEntry = Memory | Artifact

/**
 * Query interface — intentionally simple at Stage 1.
 * All fields optional — omitting them means "no constraint".
 * 
 * Promote from ext:
 *   - strategy: RetrievalStrategy  (when you add vector/graph backends)
 *   - filters: FilterExpression    (when you need complex filtering)
 *   - minConfidence: number        (when you add confidence scoring)
 */
interface MemoryQuery {
  /** Natural language query — works at every stage */
  text: string

  /** Restrict to one kind */
  kind?: 'memory' | 'artifact'

  /** Restrict memories to one role */
  role?: MemoryRole

  /** Restrict artifacts to one type */
  artifactType?: ArtifactType

  /** Must include all of these tags */
  tags?: string[]

  /** Maximum results to return */
  topK?: number
}

interface QueryResult {
  entry: StoreEntry

  /**
   * Relevance score — 0.0 to 1.0.
   * Stage 1: always 1.0 (no ranking yet).
   * Stage 3+: cosine similarity or fused rank score.
   * Always present so consumers can sort from Day 1.
   */
  score: number
}

/**
 * The one interface your agent code ever touches.
 * Implementations swap underneath without breaking callers.
 * 
 * Stage 1: FileMemoryStore    — reads/writes .md files
 * Stage 2: IndexedMemoryStore — adds frontmatter filtering  
 * Stage 3: VectorMemoryStore  — adds embedding-based recall
 * Stage 4: HybridMemoryStore  — adds BM25 + RRF fusion
 * Stage 5: CognitiveStore     — adds graph traversal + consolidation
 */
interface ContextStore {
  // ── Write ──────────────────────────────────

  /** Store a new memory. Returns the assigned ID. */
  remember(entry: Omit<Memory, 'id' | 'createdAt' | 'updatedAt'>): Promise<ID>

  /** Store a new artifact or create a new version of an existing slug. */
  store(entry: Omit<Artifact, 'id' | 'createdAt' | 'updatedAt' | 'version'>): Promise<ID>

  /** Update any fields on an existing entry. Bumps updatedAt (and version for artifacts). */
  update(id: ID, patch: Partial<Omit<StoreEntry, 'id' | 'kind'>>): Promise<void>

  // ── Read ───────────────────────────────────

  /** Semantic + structural search across the store. */
  recall(query: MemoryQuery): Promise<QueryResult[]>

  /** Direct fetch by stable ID. */
  get(id: ID): Promise<StoreEntry | null>

  /** Fetch artifact by slug — returns latest active version by default. */
  getArtifact(slug: string, version?: number): Promise<Artifact | null>

  // ── Maintain ───────────────────────────────

  /**
   * Mark a memory or artifact as no longer relevant.
   * Does NOT delete — sets status/role flags for filtering.
   * Hard deletes are a Stage 4+ concern.
   */
  forget(id: ID): Promise<void>

  /**
   * Background maintenance pass.
   * Stage 1: no-op.
   * Stage 3+: merge redundant memories, prune expired working memory,
   *           rebuild indexes, detect contradictions.
   */
  consolidate(): Promise<void>
}


// ─────────────────────────────────────────────
// SERIALIZER INTERFACE
// ─────────────────────────────────────────────

/**
 * Token budget allocation across memory types.
 * The serializer respects these limits when rendering context for the LLM.
 */
interface TokenBudget {
  total: number
  working: number       // highest priority — active task context
  episodic: number      // recent relevant episodes
  semantic: number      // background beliefs
  procedural: number    // skill instructions
  artifacts: number     // referenced produced outputs
}

/**
 * Converts QueryResults into the text string that goes into the LLM prompt.
 * Separate from storage — evolves independently.
 * 
 * Stage 1: NaiveSerializer     — concatenate text fields
 * Stage 2: SectionSerializer   — group by role with headers
 * Stage 3: BudgetSerializer    — token-aware ranking and truncation
 * Stage 4: SmartSerializer     — compression + conflict surfacing
 * Stage 5: LLMSerializer       — sub-agent summarizes before injecting
 */
interface ContextSerializer {
  render(results: QueryResult[], budget: TokenBudget): string
}


// ─────────────────────────────────────────────
// FACTORY HELPERS
// ─────────────────────────────────────────────

/** Minimal fields needed to create a Memory — everything else has defaults */
type CreateMemory = {
  content: EntryContent
  role: MemoryRole
  shape: MemoryShape
  provenance: Memory['provenance']
  tags?: string[]
  ext?: Ext
  /** Defaults to now() if omitted */
  observedAt?: Timestamp
}

/** Minimal fields needed to create an Artifact — everything else has defaults */
type CreateArtifact = {
  content: EntryContent
  type: ArtifactType
  slug: string
  provenance: Artifact['provenance']
  tags?: string[]
  ext?: Ext
  /** Defaults to 'active' if omitted */
  status?: ArtifactStatus
}


// ─────────────────────────────────────────────
// USAGE EXAMPLES (comments only — not runtime)
// ─────────────────────────────────────────────

/*

── Creating a memory ────────────────────────────────────────────────────────

await store.remember({
  content: { text: "User prefers TypeScript and rejects verbose APIs" },
  role: 'semantic',
  shape: 'atom',
  provenance: { origin: 'told', source: 'user message' },
  tags: ['user-preference', 'typescript'],
  ext: { confidence: 0.95 }    // ← lives in ext until confidence is promoted
})

── Creating an artifact ──────────────────────────────────────────────────────

await store.store({
  content: {
    text: skillMarkdownText,
    summary: "Skill: generate JWT auth middleware in TypeScript",
    format: 'structured/skill'
  },
  type: 'skill',
  slug: 'jwt-auth-skill',
  status: 'active',
  provenance: {
    inputMemoryIds: [mem_preference_id, mem_constraint_id],
    intent: "Encode JWT auth pattern per user preference"
  },
  tags: ['auth', 'jwt', 'typescript']
})

── Recalling context ────────────────────────────────────────────────────────

const results = await store.recall({
  text: "implement authentication",
  topK: 5
})

const prompt = serializer.render(results, {
  total: 4000,
  working: 1000,
  episodic: 1000,
  semantic: 500,
  procedural: 1000,
  artifacts: 500
})

── Extending without breaking ───────────────────────────────────────────────

// Stage 1 — confidence lives in ext
await store.remember({ ..., ext: { confidence: 0.8 } })

// Stage 3 — confidence promoted to first-class (old ext entries still readable)
await store.remember({ ..., provenance: { origin: 'inferred', source: '...', confidence: 0.8 } })

// Old entries with ext.confidence still work — reader checks both:
const confidence = memory.provenance.confidence ?? memory.ext?.confidence ?? 1.0

*/

export type {
  // Primitives
  ID, Timestamp, Ext,

  // Shared
  BaseEntry, EntryContent, StoreEntry,

  // Memory
  Memory, MemoryRole, MemoryShape, MemoryOrigin, CreateMemory,

  // Artifact
  Artifact, ArtifactType, ArtifactStatus, CreateArtifact,

  // Store & Serializer
  ContextStore, MemoryQuery, QueryResult,
  ContextSerializer, TokenBudget,
}
```