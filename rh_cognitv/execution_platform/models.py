"""
Pydantic models for the Execution Platform.

BaseEntry, Memory, Artifact, and all supporting types.
ExecutionResult and kind-specific result payloads.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from .types import ID, Ext, Timestamp, generate_ulid, now_timestamp


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────


class MemoryRole(str, Enum):
    """Cognitive role a memory plays."""

    EPISODIC = "episodic"  # something that happened
    SEMANTIC = "semantic"  # something believed to be true
    PROCEDURAL = "procedural"  # how to do something
    WORKING = "working"  # active right now, ephemeral


class MemoryShape(str, Enum):
    """Shape of the memory content. Orthogonal to role."""

    ATOM = "atom"  # one indivisible claim or event
    SEQUENCE = "sequence"  # ordered list of atoms
    SUMMARY = "summary"  # lossy compression of multiple memories
    NARRATIVE = "narrative"  # atoms connected by causality


class MemoryOrigin(str, Enum):
    """Where a memory came from — drives trust reasoning."""

    OBSERVED = "observed"  # directly witnessed (tool output, file read)
    TOLD = "told"  # user explicitly stated
    INFERRED = "inferred"  # LLM derived from other signals
    CONSOLIDATED = "consolidated"  # merged/summarized from other memories


class ArtifactType(str, Enum):
    """What kind of produced thing an artifact is."""

    CODE = "code"
    DOCUMENT = "document"
    DATA = "data"
    SKILL = "skill"
    PLAN = "plan"
    PROMPT = "prompt"


class ArtifactStatus(str, Enum):
    """Lifecycle status of an artifact."""

    DRAFT = "draft"  # being built, not yet usable
    ACTIVE = "active"  # current, use this
    DEPRECATED = "deprecated"  # superseded but kept for reference
    ARCHIVED = "archived"  # retired


class EventKind(str, Enum):
    """Kind of execution event."""

    TEXT = "text"
    DATA = "data"
    FUNCTION = "function"
    TOOL = "tool"


class EventStatus(str, Enum):
    """Lifecycle status of an execution event."""

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ESCALATED = "escalated"  # human-in-the-loop: awaiting user decision
    WAITING = "waiting"  # paused, waiting for external input


# ──────────────────────────────────────────────
# Content & Base Entry
# ──────────────────────────────────────────────


class EntryContent(BaseModel):
    """Content payload shared by Memory and Artifact entries."""

    text: str
    summary: str | None = None
    format: str | None = None  # MIME-like: text/plain, text/markdown, code/python, data/json


class Provenance(BaseModel):
    """Memory provenance — where it came from."""

    origin: MemoryOrigin
    source: str  # free-text: "user message", "tool:file_read", "llm inference"


class ArtifactProvenance(BaseModel):
    """Artifact provenance — what memories informed its creation."""

    input_memory_ids: list[ID] = Field(default_factory=list)
    intent: str


class TimeInfo(BaseModel):
    """Temporal validity for a memory."""

    recorded_at: Timestamp
    observed_at: Timestamp


class BaseEntry(BaseModel):
    """Base for all entries in the context store."""

    id: ID = Field(default_factory=generate_ulid)
    kind: str  # "memory" | "artifact" — discriminator
    content: EntryContent
    created_at: Timestamp = Field(default_factory=now_timestamp)
    updated_at: Timestamp = Field(default_factory=now_timestamp)
    tags: list[str] = Field(default_factory=list)
    ext: Ext = Field(default_factory=dict)


class Memory(BaseEntry):
    """A memory entry in the context store."""

    kind: str = "memory"
    role: MemoryRole
    shape: MemoryShape
    provenance: Provenance
    time: TimeInfo


class Artifact(BaseEntry):
    """An artifact entry in the context store."""

    kind: str = "artifact"
    type: ArtifactType
    slug: str
    version: int = 1
    status: ArtifactStatus = ArtifactStatus.ACTIVE
    provenance: ArtifactProvenance
    supersedes: ID | None = None


# ──────────────────────────────────────────────
# Query Types
# ──────────────────────────────────────────────


class MemoryQuery(BaseModel):
    """Query interface for the context store."""

    text: str = ""
    kind: str | None = None  # "memory" | "artifact"
    role: MemoryRole | None = None
    artifact_type: ArtifactType | None = None
    tags: list[str] | None = None
    top_k: int | None = None


class QueryResult(BaseModel):
    """A single result from a context store query."""

    entry: Memory | Artifact
    score: float = 1.0  # Stage 1: always 1.0


# ──────────────────────────────────────────────
# Token Budget
# ──────────────────────────────────────────────


class TokenBudget(BaseModel):
    """Token budget allocation across memory types."""

    total: int
    working: int = 0
    episodic: int = 0
    semantic: int = 0
    procedural: int = 0
    artifacts: int = 0


# ──────────────────────────────────────────────
# Execution Result Types
# ──────────────────────────────────────────────


class TokenUsage(BaseModel):
    """Token usage from an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total: int = 0


class ResultMetadata(BaseModel):
    """Metadata about an execution result."""

    duration_ms: float = 0.0
    attempt: int = 1
    started_at: Timestamp | None = None
    completed_at: Timestamp | None = None


class LLMResultData(BaseModel):
    """Result payload from an LLM text/data call."""

    text: str
    thinking: str | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: str = ""


class FunctionResultData(BaseModel):
    """Result payload from a function call."""

    return_value: Any = None
    duration_ms: float = 0.0


class ToolResultData(BaseModel):
    """Result payload from a tool invocation (LLM + function)."""

    llm_result: LLMResultData
    function_result: FunctionResultData


T = TypeVar("T")


class ExecutionResult(BaseModel, Generic[T]):
    """Generic result of executing an event."""

    ok: bool
    value: T | None = None
    error_message: str | None = None
    error_category: str | None = None
    metadata: ResultMetadata = Field(default_factory=ResultMetadata)


# ──────────────────────────────────────────────
# Budget Snapshot
# ──────────────────────────────────────────────


class BudgetSnapshot(BaseModel):
    """Point-in-time snapshot of budget remaining."""

    tokens_remaining: int
    calls_remaining: int
    time_remaining_seconds: float
