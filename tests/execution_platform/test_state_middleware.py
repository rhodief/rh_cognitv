"""Tests for state_middleware.py — StateSnapshotMiddleware EventBus integration."""

import pytest

from rh_cognitv.execution_platform.event_bus import EventBus
from rh_cognitv.execution_platform.events import ExecutionEvent, TextPayload
from rh_cognitv.execution_platform.models import EventKind, EventStatus
from rh_cognitv.execution_platform.state import ExecutionState
from rh_cognitv.execution_platform.state_middleware import StateSnapshotMiddleware


class TestStateSnapshotMiddleware:
    @pytest.mark.asyncio
    async def test_snapshots_on_running(self):
        state = ExecutionState({"task": "pending"})
        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))

        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.RUNNING,
        )
        await bus.emit(event)

        assert state.version_count == 1
        assert state.get_current()["_last_event_status"] == "running"

    @pytest.mark.asyncio
    async def test_snapshots_on_success(self):
        state = ExecutionState()
        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))

        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.SUCCESS,
        )
        await bus.emit(event)

        assert state.version_count == 1
        assert state.get_current()["_last_event_status"] == "success"

    @pytest.mark.asyncio
    async def test_no_snapshot_on_created(self):
        """CREATED status should not trigger a snapshot."""
        state = ExecutionState()
        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))

        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.CREATED,
        )
        await bus.emit(event)

        assert state.version_count == 0

    @pytest.mark.asyncio
    async def test_no_snapshot_on_queued(self):
        """QUEUED status should not trigger a snapshot."""
        state = ExecutionState()
        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))

        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.QUEUED,
        )
        await bus.emit(event)

        assert state.version_count == 0

    @pytest.mark.asyncio
    async def test_snapshots_on_failed(self):
        state = ExecutionState()
        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))

        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.FAILED,
        )
        await bus.emit(event)

        assert state.version_count == 1
        assert state.get_current()["_last_event_status"] == "failed"

    @pytest.mark.asyncio
    async def test_snapshots_on_escalated(self):
        state = ExecutionState()
        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))

        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.ESCALATED,
        )
        await bus.emit(event)

        assert state.version_count == 1
        assert state.get_current()["_last_event_status"] == "escalated"

    @pytest.mark.asyncio
    async def test_captures_event_id(self):
        state = ExecutionState()
        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))

        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.RUNNING,
        )
        await bus.emit(event)

        assert state.get_current()["_last_event_id"] == event.id

    @pytest.mark.asyncio
    async def test_multiple_events_create_multiple_snapshots(self):
        state = ExecutionState()
        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))

        for status in [EventStatus.RUNNING, EventStatus.SUCCESS]:
            event = ExecutionEvent(
                kind=EventKind.TEXT,
                payload=TextPayload(prompt="x"),
                status=status,
            )
            await bus.emit(event)

        assert state.version_count == 2

    @pytest.mark.asyncio
    async def test_undo_after_events(self):
        """Time-travel: undo to before a state transition."""
        state = ExecutionState({"task": "initial"})
        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))

        # Emit RUNNING
        await bus.emit(ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.RUNNING,
        ))

        # Emit SUCCESS
        await bus.emit(ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.SUCCESS,
        ))

        assert state.get_current()["_last_event_status"] == "success"

        # Undo to RUNNING snapshot
        state.undo()
        assert state.get_current()["_last_event_status"] == "running"

    @pytest.mark.asyncio
    async def test_non_execution_events_pass_through(self):
        """Non-ExecutionEvent objects should pass through without snapshotting."""
        state = ExecutionState()
        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))

        await bus.emit({"type": "some_other_event"})
        assert state.version_count == 0

    @pytest.mark.asyncio
    async def test_middleware_calls_next(self):
        """Middleware should always call next_fn to continue the chain."""
        state = ExecutionState()
        order = []

        class TrackMiddleware:
            def handle(self, event, next_fn):
                order.append("track")
                return next_fn(event)

        bus = EventBus()
        bus.use(StateSnapshotMiddleware(state))
        bus.use(TrackMiddleware())

        await bus.emit(ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.RUNNING,
        ))

        assert "track" in order
