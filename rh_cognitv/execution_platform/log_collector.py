"""
LogCollector — structured log collection as an EventBus subscriber.

DD-L3-06: Human-readable structured logs (JSON lines), optimized for debugging.
Carries execution context (execution_id, node_id).
Subscribes to ExecutionEvent and escalation events.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .events import EscalationRequested, EscalationResolved, ExecutionEvent
from .protocols import LogCollectorProtocol
from .types import now_timestamp


class LogEntry:
    """A single structured log line."""

    __slots__ = (
        "timestamp",
        "level",
        "event_id",
        "event_kind",
        "event_status",
        "execution_id",
        "node_id",
        "message",
        "extra",
    )

    def __init__(
        self,
        *,
        level: str,
        message: str,
        event_id: str | None = None,
        event_kind: str | None = None,
        event_status: str | None = None,
        execution_id: str | None = None,
        node_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.timestamp = now_timestamp()
        self.level = level
        self.event_id = event_id
        self.event_kind = event_kind
        self.event_status = event_status
        self.execution_id = execution_id
        self.node_id = node_id
        self.message = message
        self.extra = extra or {}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
        }
        if self.event_id is not None:
            d["event_id"] = self.event_id
        if self.event_kind is not None:
            d["event_kind"] = self.event_kind
        if self.event_status is not None:
            d["event_status"] = self.event_status
        if self.execution_id is not None:
            d["execution_id"] = self.execution_id
        if self.node_id is not None:
            d["node_id"] = self.node_id
        if self.extra:
            d["extra"] = self.extra
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class LogCollector(LogCollectorProtocol):
    """Structured log collector — EventBus async subscriber.

    Produces JSON-lines log entries for each execution event and escalation.
    Maintains an in-memory log buffer for retrieval and optional sink callback.

    Usage::

        collector = LogCollector(execution_id="exec-123")
        bus.on_async(ExecutionEvent, collector.on_event)
        bus.on_async(EscalationRequested, collector.on_event)
        bus.on_async(EscalationResolved, collector.on_event)
    """

    def __init__(
        self,
        *,
        execution_id: str | None = None,
        node_id: str | None = None,
        sink: Callable[[LogEntry], None] | None = None,
    ) -> None:
        self._execution_id = execution_id
        self._node_id = node_id
        self._sink = sink
        self._entries: list[LogEntry] = []

    @property
    def entries(self) -> list[LogEntry]:
        """All collected log entries."""
        return list(self._entries)

    @property
    def execution_id(self) -> str | None:
        return self._execution_id

    @property
    def node_id(self) -> str | None:
        return self._node_id

    def clear(self) -> None:
        """Clear all collected entries."""
        self._entries.clear()

    async def on_event(self, event: Any) -> None:
        """Handle an event for logging. Dispatches by event type."""
        if isinstance(event, ExecutionEvent):
            self._log_execution_event(event)
        elif isinstance(event, EscalationRequested):
            self._log_escalation_requested(event)
        elif isinstance(event, EscalationResolved):
            self._log_escalation_resolved(event)

    def _log_execution_event(self, event: ExecutionEvent) -> None:
        level = _status_to_level(event.status.value)
        entry = LogEntry(
            level=level,
            message=f"Event {event.kind.value} → {event.status.value}",
            event_id=event.id,
            event_kind=event.kind.value,
            event_status=event.status.value,
            execution_id=self._execution_id,
            node_id=self._node_id,
        )
        self._append(entry)

    def _log_escalation_requested(self, event: EscalationRequested) -> None:
        entry = LogEntry(
            level="WARN",
            message=f"Escalation requested: {event.question}",
            event_id=event.event_id,
            execution_id=self._execution_id,
            node_id=event.node_id or self._node_id,
            extra={"options": event.options},
        )
        self._append(entry)

    def _log_escalation_resolved(self, event: EscalationResolved) -> None:
        entry = LogEntry(
            level="INFO",
            message=f"Escalation resolved: {event.decision}",
            event_id=event.event_id,
            execution_id=self._execution_id,
            node_id=self._node_id,
            extra={"decision": event.decision},
        )
        self._append(entry)

    def _append(self, entry: LogEntry) -> None:
        self._entries.append(entry)
        if self._sink is not None:
            self._sink(entry)


def _status_to_level(status: str) -> str:
    """Map event status to log level."""
    _map = {
        "created": "DEBUG",
        "queued": "DEBUG",
        "running": "INFO",
        "success": "INFO",
        "failed": "ERROR",
        "retrying": "WARN",
        "cancelled": "WARN",
        "timed_out": "ERROR",
        "escalated": "WARN",
        "waiting": "INFO",
    }
    return _map.get(status, "INFO")
