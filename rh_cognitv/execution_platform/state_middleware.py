"""
State integration — middleware wiring ExecutionState into the EventBus lifecycle.

Phase 4.2: Snapshot on each status transition, emit lifecycle events.
"""

from __future__ import annotations

from typing import Any

from .events import ExecutionEvent
from .models import EventStatus
from .protocols import MiddlewareProtocol
from .state import ExecutionState


class StateSnapshotMiddleware(MiddlewareProtocol):
    """EventBus middleware that snapshots state on ExecutionEvent status transitions.

    When an ExecutionEvent passes through with a terminal or transition status
    (RUNNING, SUCCESS, FAILED, RETRYING, ESCALATED, TIMED_OUT, CANCELLED),
    the middleware takes a state snapshot capturing the event's current status.

    Usage:
        state = ExecutionState()
        bus.use(StateSnapshotMiddleware(state))
    """

    # Statuses that trigger a snapshot
    SNAPSHOT_STATUSES = frozenset({
        EventStatus.RUNNING,
        EventStatus.SUCCESS,
        EventStatus.FAILED,
        EventStatus.RETRYING,
        EventStatus.ESCALATED,
        EventStatus.TIMED_OUT,
        EventStatus.CANCELLED,
    })

    def __init__(self, state: ExecutionState) -> None:
        self._state = state

    def handle(self, event: Any, next_fn: Any) -> Any:
        """Snapshot state when an ExecutionEvent has a transition status."""
        if isinstance(event, ExecutionEvent) and event.status in self.SNAPSHOT_STATUSES:
            self._state.update("_last_event_id", event.id)
            self._state.update("_last_event_status", event.status.value)
            self._state.snapshot()

        return next_fn(event)
