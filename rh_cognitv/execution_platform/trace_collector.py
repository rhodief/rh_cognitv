"""
TraceCollector — execution trace recording as an opt-in EventBus subscriber.

DD-L3-06: Machine-readable spans, optimized for performance analysis and replay.
OQ-L3-04: Custom OTel-compatible schema. No OTel SDK dependency.
Optional by design — zero overhead if not registered.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .events import ExecutionEvent
from .protocols import TraceCollectorProtocol
from .types import ID, Timestamp, generate_ulid, now_timestamp


# ──────────────────────────────────────────────
# Span Models (OTel-compatible field names)
# ──────────────────────────────────────────────


class SpanStatus(str, Enum):
    """Span completion status, matching OTel conventions."""

    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class SpanEvent(BaseModel):
    """An event within a span (OTel-compatible)."""

    name: str
    timestamp: Timestamp = Field(default_factory=now_timestamp)
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


class Span(BaseModel):
    """A single trace span with OTel-compatible field names.

    Uses OTel attribute naming conventions (e.g. ``execution.node_id``,
    ``event.kind``) so exporting to Jaeger/Grafana is a trivial mapping.
    """

    trace_id: str
    span_id: str = Field(default_factory=generate_ulid)
    parent_span_id: str | None = None
    name: str
    status: SpanStatus = SpanStatus.UNSET
    start_time: Timestamp = Field(default_factory=now_timestamp)
    end_time: Timestamp | None = None
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Event-status → span-status mapping
# ──────────────────────────────────────────────

_TERMINAL_OK = frozenset({"success"})
_TERMINAL_ERROR = frozenset({"failed", "timed_out", "cancelled"})


# ──────────────────────────────────────────────
# TraceCollector
# ──────────────────────────────────────────────


class TraceCollector(TraceCollectorProtocol):
    """Opt-in execution trace recorder — EventBus async subscriber.

    Produces machine-readable spans with OTel-compatible schema.
    Registered only when explicitly configured — zero overhead if unused.

    Usage::

        collector = TraceCollector(trace_id="trace-abc")
        bus.on_async(ExecutionEvent, collector.on_event)
    """

    def __init__(
        self,
        *,
        trace_id: str | None = None,
        execution_id: str | None = None,
        node_id: str | None = None,
    ) -> None:
        self._trace_id = trace_id or generate_ulid()
        self._execution_id = execution_id
        self._node_id = node_id
        self._spans: dict[str, Span] = {}  # event_id → Span
        self._completed_spans: list[Span] = []

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def spans(self) -> list[Span]:
        """All completed spans."""
        return list(self._completed_spans)

    @property
    def active_spans(self) -> dict[str, Span]:
        """Currently open (in-progress) spans keyed by event_id."""
        return dict(self._spans)

    def clear(self) -> None:
        """Clear all spans."""
        self._spans.clear()
        self._completed_spans.clear()

    async def on_event(self, event: Any) -> None:
        """Handle an ExecutionEvent for tracing."""
        if not isinstance(event, ExecutionEvent):
            return

        status_val = event.status.value

        if status_val == "running":
            self._start_span(event)
        elif status_val in _TERMINAL_OK:
            self._end_span(event, SpanStatus.OK)
        elif status_val in _TERMINAL_ERROR:
            self._end_span(event, SpanStatus.ERROR)
        elif status_val in ("retrying", "escalated", "waiting"):
            self._add_span_event(event)
        # created/queued: no span action needed

    def _start_span(self, event: ExecutionEvent) -> None:
        """Open a new span when an event starts running.

        If a span already exists for this event (e.g. after RETRYING → RUNNING),
        preserve accumulated span events.
        """
        existing = self._spans.get(event.id)
        prior_events = existing.events if existing else []

        attrs: dict[str, str | int | float | bool] = {
            "event.kind": event.kind.value,
        }
        if self._execution_id:
            attrs["execution.id"] = self._execution_id
        if self._node_id:
            attrs["execution.node_id"] = self._node_id
        if event.parent_id:
            attrs["event.parent_id"] = event.parent_id

        parent_span_id = None
        if event.parent_id and event.parent_id in self._spans:
            parent_span_id = self._spans[event.parent_id].span_id

        span = Span(
            trace_id=self._trace_id,
            span_id=existing.span_id if existing else generate_ulid(),
            parent_span_id=parent_span_id or (existing.parent_span_id if existing else None),
            name=f"{event.kind.value}.execute",
            start_time=existing.start_time if existing else now_timestamp(),
            attributes=attrs,
            events=prior_events,
        )
        self._spans[event.id] = span

    def _end_span(self, event: ExecutionEvent, status: SpanStatus) -> None:
        """Close an existing span on terminal status."""
        span = self._spans.pop(event.id, None)
        if span is None:
            # Late event for an event we never saw running — create a minimal span
            span = Span(
                trace_id=self._trace_id,
                name=f"{event.kind.value}.execute",
                attributes={"event.kind": event.kind.value},
            )
        span.end_time = now_timestamp()
        span.status = status
        span.events.append(
            SpanEvent(
                name=event.status.value,
                attributes={"event.status": event.status.value},
            )
        )
        self._completed_spans.append(span)

    def _add_span_event(self, event: ExecutionEvent) -> None:
        """Add a span event (e.g. retrying, escalated) to an open span."""
        span = self._spans.get(event.id)
        if span is None:
            return
        span.events.append(
            SpanEvent(
                name=event.status.value,
                attributes={"event.status": event.status.value},
            )
        )
