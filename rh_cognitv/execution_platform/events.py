"""
Execution events — data-only event types with kind-specific payloads.

DD-L3-05: Events are data-only. External handlers are registered in a handler
registry (Strategy pattern). Each event carries a `kind` and a typed `payload`.

OQ-L3-05: EscalationRequested / EscalationResolved for human-in-the-loop.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .models import EventKind, EventStatus
from .types import ID, Ext, Timestamp, generate_ulid, now_timestamp


# ──────────────────────────────────────────────
# Kind-Specific Payloads
# ──────────────────────────────────────────────


class TextPayload(BaseModel):
    """Payload for TEXT events — LLM text generation."""

    prompt: str
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    ext: Ext = Field(default_factory=dict)


class DataPayload(BaseModel):
    """Payload for DATA events — structured data generation."""

    prompt: str
    output_schema: dict[str, Any] | None = None
    model: str | None = None
    ext: Ext = Field(default_factory=dict)


class FunctionPayload(BaseModel):
    """Payload for FUNCTION events — direct function invocation."""

    function_name: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    ext: Ext = Field(default_factory=dict)


class ToolPayload(BaseModel):
    """Payload for TOOL events — LLM-driven tool use (LLM call + function)."""

    prompt: str
    tools: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    ext: Ext = Field(default_factory=dict)


# Union of all payload types for type hints
EventPayload = TextPayload | DataPayload | FunctionPayload | ToolPayload


# ──────────────────────────────────────────────
# ExecutionEvent
# ──────────────────────────────────────────────


class ExecutionEvent(BaseModel):
    """Data-only execution event.

    Carries a `kind` (which handler to dispatch to) and a kind-specific `payload`.
    Status tracks the event lifecycle (CREATED → RUNNING → SUCCESS/FAILED/...).
    """

    id: ID = Field(default_factory=generate_ulid)
    kind: EventKind
    payload: EventPayload
    status: EventStatus = EventStatus.CREATED
    created_at: Timestamp = Field(default_factory=now_timestamp)
    parent_id: ID | None = None
    ext: Ext = Field(default_factory=dict)


# ──────────────────────────────────────────────
# Escalation Events (OQ-L3-05)
# ──────────────────────────────────────────────


class EscalationRequested(BaseModel):
    """Emitted when a handler needs a human decision.

    Payload includes enough context for cloud-safe recovery:
    question, options, originating event_id, and resume data.
    """

    event_id: ID
    question: str
    options: list[str] = Field(default_factory=list)
    node_id: str | None = None
    resume_data: dict[str, Any] = Field(default_factory=dict)
    created_at: Timestamp = Field(default_factory=now_timestamp)


class EscalationResolved(BaseModel):
    """Emitted when a human decision arrives for a prior escalation."""

    event_id: ID
    decision: str
    resolved_at: Timestamp = Field(default_factory=now_timestamp)
    ext: Ext = Field(default_factory=dict)
