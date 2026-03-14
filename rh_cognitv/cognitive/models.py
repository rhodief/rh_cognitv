"""
Cognitive Layer Pydantic models.

All data types for skills, plans, context, results, prompts, and messages.
These models are the shared vocabulary of the Cognitive Layer — every other
cognitive module imports from here.

Phase 3.1.2 — Foundation models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from rh_cognitv.execution_platform.types import ID, generate_ulid
from rh_cognitv.execution_platform.models import (
    BudgetSnapshot,
    MemoryQuery,
    TokenBudget,
)


# ──────────────────────────────────────────────
# Message & LLM Result Types
# ──────────────────────────────────────────────


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in an LLM conversation.

    Maps to the universal chat format used by OpenAI, Anthropic, etc.
    """

    role: MessageRole
    content: str
    name: str | None = None  # tool name when role=tool


class CompletionResult(BaseModel):
    """Result from an LLM completion call.

    Aligned with L3's LLMResultData: text, thinking, token_usage, model.
    """

    text: str
    thinking: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    finish_reason: str = ""


class ToolCall(BaseModel):
    """A single tool call made by the LLM."""

    id: str = Field(default_factory=generate_ulid)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from an LLM call with tools.

    Aligned with L3's ToolResultData: LLM response + tool calls.
    """

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    finish_reason: str = ""


# ──────────────────────────────────────────────
# Built Prompt (DD-L1-03)
# ──────────────────────────────────────────────


class BuiltPrompt(BaseModel):
    """Output of PromptBuilder / TemplateRenderer.

    Maps 1:1 to L3's TextPayload fields: prompt + system_prompt.
    """

    prompt: str
    system_prompt: str | None = None


# ──────────────────────────────────────────────
# Context Reference (DD-L1-07)
# ──────────────────────────────────────────────


class ContextRef(BaseModel):
    """Declarative reference to context needed at execution time.

    Carried on SkillSteps. The orchestrator resolves these before
    executing each node (Phase 3.7 context resolution hook).

    Kinds:
        memory: Resolve via ContextStore.get(id)
        artifact: Resolve via ContextStore.get_artifact(slug, version)
        query: Resolve via ContextStore.recall(query)
        previous_result: Resolve from DAGTraversalState.node_results[from_step]
    """

    kind: Literal["memory", "artifact", "query", "previous_result"]
    id: str | None = None
    slug: str | None = None
    version: int | None = None
    query: MemoryQuery | None = None
    from_step: str | None = None
    key: str = "context"


# ──────────────────────────────────────────────
# Step Configuration (DD-L1-04)
# ──────────────────────────────────────────────


class TextStepConfig(BaseModel):
    """Configuration for a text-generation step.

    Aligns with L2's TextNode fields.
    """

    prompt: str
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class DataStepConfig(BaseModel):
    """Configuration for a structured data extraction step.

    Aligns with L2's DataNode fields.
    """

    prompt: str
    output_schema: dict[str, Any] | None = None
    model: str | None = None


class FunctionStepConfig(BaseModel):
    """Configuration for a function invocation step.

    Aligns with L2's FunctionNode fields.
    """

    function_name: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ToolStepConfig(BaseModel):
    """Configuration for an LLM-with-tools step.

    Aligns with L2's ToolNode fields.
    """

    prompt: str
    tools: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None


# Discriminated union of all step config types
StepConfig = TextStepConfig | DataStepConfig | FunctionStepConfig | ToolStepConfig


# ──────────────────────────────────────────────
# Skill Step & Plan (DD-L1-04)
# ──────────────────────────────────────────────


class SkillStep(BaseModel):
    """A single step in a SkillPlan.

    Each step maps to one L2 node via the adapter.
    context_refs declare execution-time context dependencies (DD-L1-07).
    """

    id: str
    kind: Literal["text", "data", "function", "tool"]
    config: StepConfig
    context_refs: list[ContextRef] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class SkillConstraints(BaseModel):
    """Execution constraints for a SkillPlan.

    Maps to BaseNode.timeout_seconds and BaseNode.max_retries in L2.
    """

    timeout_seconds: float | None = None
    max_retries: int | None = None
    max_tokens: int | None = None


class SkillPlan(BaseModel):
    """Intermediate representation produced by Skill.plan() (DD-L1-04 Option B).

    The adapter translates this into a PlanDAG for the orchestrator.
    Steps are typed with their config and optional context refs.
    """

    name: str
    steps: list[SkillStep]
    constraints: SkillConstraints | None = None


# ──────────────────────────────────────────────
# Skill Context (DD-L1-05 Option C)
# ──────────────────────────────────────────────


class SkillContext(BaseModel):
    """Pre-loaded context passed to Skill.plan().

    Built by the framework before plan() is called.
    Carries memories, artifacts, budget info, and pre-serialized context.

    The skill never touches the ContextStore directly — all context
    flows through this object.
    """

    memories: list[Any] = Field(default_factory=list)
    artifacts: list[Any] = Field(default_factory=list)
    budget: TokenBudget | None = None
    budget_snapshot: BudgetSnapshot | None = None
    serialized_context: str = ""
    remaining_prompt_tokens: int = 0
    ext: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


# ──────────────────────────────────────────────
# Skill Result & Provenance (DI-L1-02)
# ──────────────────────────────────────────────


class SkillProvenance(BaseModel):
    """Provenance metadata for a SkillResult.

    Enables full traceability: which skill, what input, which memories
    were used. Maps to L3 Provenance/ArtifactProvenance for storage.
    """

    skill_name: str
    input_hash: str = ""
    context_memory_ids: list[ID] = Field(default_factory=list)


class CreateMemory(BaseModel):
    """Request to create a memory from a skill result.

    The framework routes this to ContextStore.remember().
    """

    text: str
    role: str = "semantic"
    shape: str = "atom"
    origin: str = "inferred"
    source: str = ""
    tags: list[str] = Field(default_factory=list)


class CreateArtifact(BaseModel):
    """Request to create an artifact from a skill result.

    The framework routes this to ContextStore.store().
    """

    text: str
    type: str = "document"
    slug: str
    intent: str = ""
    tags: list[str] = Field(default_factory=list)


class ReplanRequest(BaseModel):
    """Signal from interpret() that the approach should change.

    The framework handles this: interrupt current DAG, build new
    PlanDAG from suggested_steps, re-execute.
    """

    reason: str
    suggested_steps: list[SkillStep] | None = None


class SkillResult(BaseModel):
    """Typed result from Skill.interpret() (DI-L1-02).

    Carries the output, provenance, optional memories/artifacts to store,
    and an optional replan signal.
    """

    output: Any = None
    success: bool = True
    error_message: str | None = None
    provenance: SkillProvenance | None = None
    suggested_memories: list[CreateMemory] = Field(default_factory=list)
    suggested_artifacts: list[CreateArtifact] = Field(default_factory=list)
    replan: ReplanRequest | None = None


# ──────────────────────────────────────────────
# Skill Config (DD-L1-01 Option C — ConfigSkill)
# ──────────────────────────────────────────────


class SkillConfig(BaseModel):
    """Declarative skill definition — zero code for simple skills.

    ConfigSkill auto-generates plan() and interpret() from this config.
    The prompt_template uses str.format() with {input.*} and {context}.
    memory_query defines what context to pre-load (DD-L1-05).
    """

    name: str
    description: str = ""
    prompt_template: str
    system_prompt: str | None = None
    input_schema: Any = None  # type[BaseModel] — Any to avoid forward ref issues
    output_schema: Any = None  # type[BaseModel] — for Pydantic validation
    memory_query: MemoryQuery | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tags: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
