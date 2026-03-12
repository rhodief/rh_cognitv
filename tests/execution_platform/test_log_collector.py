"""Tests for LogCollector — structured log collection."""

import pytest

from rh_cognitv.execution_platform.event_bus import EventBus
from rh_cognitv.execution_platform.events import (
    EscalationRequested,
    EscalationResolved,
    ExecutionEvent,
    TextPayload,
)
from rh_cognitv.execution_platform.log_collector import LogCollector, LogEntry
from rh_cognitv.execution_platform.models import EventKind, EventStatus


# ── Helpers ──


def _event(status: EventStatus = EventStatus.CREATED, **kw) -> ExecutionEvent:
    defaults = dict(
        kind=EventKind.TEXT,
        payload=TextPayload(prompt="hello"),
        status=status,
    )
    defaults.update(kw)
    return ExecutionEvent(**defaults)


# ──────────────────────────────────────────────
# Basic LogEntry
# ──────────────────────────────────────────────


class TestLogEntry:
    def test_to_dict_minimal(self):
        entry = LogEntry(level="INFO", message="test")
        d = entry.to_dict()
        assert d["level"] == "INFO"
        assert d["message"] == "test"
        assert "timestamp" in d
        assert "event_id" not in d

    def test_to_dict_full(self):
        entry = LogEntry(
            level="ERROR",
            message="failed",
            event_id="ev1",
            event_kind="text",
            event_status="failed",
            execution_id="exec1",
            node_id="node1",
            extra={"key": "val"},
        )
        d = entry.to_dict()
        assert d["event_id"] == "ev1"
        assert d["execution_id"] == "exec1"
        assert d["node_id"] == "node1"
        assert d["extra"] == {"key": "val"}

    def test_to_json(self):
        entry = LogEntry(level="INFO", message="test")
        j = entry.to_json()
        import json

        parsed = json.loads(j)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "test"


# ──────────────────────────────────────────────
# LogCollector — direct on_event calls
# ──────────────────────────────────────────────


class TestLogCollectorDirect:
    @pytest.mark.asyncio
    async def test_logs_execution_event(self):
        collector = LogCollector(execution_id="exec-1")
        event = _event(status=EventStatus.RUNNING)
        await collector.on_event(event)
        assert len(collector.entries) == 1
        entry = collector.entries[0]
        assert entry.level == "INFO"
        assert "text" in entry.message
        assert "running" in entry.message
        assert entry.event_id == event.id
        assert entry.execution_id == "exec-1"

    @pytest.mark.asyncio
    async def test_logs_all_statuses(self):
        collector = LogCollector()
        for status in EventStatus:
            await collector.on_event(_event(status=status))
        assert len(collector.entries) == len(EventStatus)

    @pytest.mark.asyncio
    async def test_status_level_mapping(self):
        collector = LogCollector()
        cases = [
            (EventStatus.CREATED, "DEBUG"),
            (EventStatus.QUEUED, "DEBUG"),
            (EventStatus.RUNNING, "INFO"),
            (EventStatus.SUCCESS, "INFO"),
            (EventStatus.FAILED, "ERROR"),
            (EventStatus.RETRYING, "WARN"),
            (EventStatus.CANCELLED, "WARN"),
            (EventStatus.TIMED_OUT, "ERROR"),
            (EventStatus.ESCALATED, "WARN"),
            (EventStatus.WAITING, "INFO"),
        ]
        for status, expected_level in cases:
            collector.clear()
            await collector.on_event(_event(status=status))
            assert collector.entries[0].level == expected_level, (
                f"status={status.value} expected {expected_level}"
            )

    @pytest.mark.asyncio
    async def test_logs_escalation_requested(self):
        collector = LogCollector(execution_id="exec-2", node_id="n1")
        esc = EscalationRequested(
            event_id="ev-1",
            question="Approve?",
            options=["yes", "no"],
            node_id="n2",
        )
        await collector.on_event(esc)
        assert len(collector.entries) == 1
        entry = collector.entries[0]
        assert entry.level == "WARN"
        assert "Approve?" in entry.message
        assert entry.event_id == "ev-1"
        assert entry.node_id == "n2"  # uses event's node_id
        assert entry.extra["options"] == ["yes", "no"]

    @pytest.mark.asyncio
    async def test_escalation_requested_fallback_node_id(self):
        collector = LogCollector(node_id="default-node")
        esc = EscalationRequested(
            event_id="ev-1",
            question="Approve?",
        )
        await collector.on_event(esc)
        assert collector.entries[0].node_id == "default-node"

    @pytest.mark.asyncio
    async def test_logs_escalation_resolved(self):
        collector = LogCollector(execution_id="exec-3")
        res = EscalationResolved(event_id="ev-1", decision="approved")
        await collector.on_event(res)
        assert len(collector.entries) == 1
        entry = collector.entries[0]
        assert entry.level == "INFO"
        assert "approved" in entry.message
        assert entry.extra["decision"] == "approved"

    @pytest.mark.asyncio
    async def test_carries_execution_context(self):
        collector = LogCollector(execution_id="exec-ctx", node_id="node-ctx")
        await collector.on_event(_event(status=EventStatus.RUNNING))
        entry = collector.entries[0]
        assert entry.execution_id == "exec-ctx"
        assert entry.node_id == "node-ctx"

    @pytest.mark.asyncio
    async def test_clear_entries(self):
        collector = LogCollector()
        await collector.on_event(_event())
        assert len(collector.entries) == 1
        collector.clear()
        assert len(collector.entries) == 0

    @pytest.mark.asyncio
    async def test_sink_callback(self):
        received: list[LogEntry] = []
        collector = LogCollector(sink=received.append)
        await collector.on_event(_event(status=EventStatus.SUCCESS))
        assert len(received) == 1
        assert received[0].level == "INFO"

    @pytest.mark.asyncio
    async def test_ignores_unknown_events(self):
        collector = LogCollector()
        await collector.on_event("not an event")
        assert len(collector.entries) == 0

    @pytest.mark.asyncio
    async def test_entries_returns_copy(self):
        collector = LogCollector()
        await collector.on_event(_event())
        entries1 = collector.entries
        entries2 = collector.entries
        assert entries1 is not entries2
        assert len(entries1) == len(entries2)


# ──────────────────────────────────────────────
# LogCollector + EventBus integration
# ──────────────────────────────────────────────


class TestLogCollectorEventBus:
    @pytest.mark.asyncio
    async def test_eventbus_integration(self):
        bus = EventBus()
        collector = LogCollector(execution_id="bus-test")
        bus.on_async(ExecutionEvent, collector.on_event)
        bus.on_async(EscalationRequested, collector.on_event)

        # Emit lifecycle transitions
        event = _event(status=EventStatus.CREATED)
        await bus.emit(event)

        running = event.model_copy(update={"status": EventStatus.RUNNING})
        await bus.emit(running)

        success = event.model_copy(update={"status": EventStatus.SUCCESS})
        await bus.emit(success)

        assert len(collector.entries) == 3
        levels = [e.level for e in collector.entries]
        assert levels == ["DEBUG", "INFO", "INFO"]

    @pytest.mark.asyncio
    async def test_eventbus_escalation_flow(self):
        bus = EventBus()
        collector = LogCollector()
        bus.on_async(ExecutionEvent, collector.on_event)
        bus.on_async(EscalationRequested, collector.on_event)
        bus.on_async(EscalationResolved, collector.on_event)

        event = _event(status=EventStatus.ESCALATED)
        await bus.emit(event)

        esc = EscalationRequested(event_id=event.id, question="Continue?")
        await bus.emit(esc)

        res = EscalationResolved(event_id=event.id, decision="yes")
        await bus.emit(res)

        assert len(collector.entries) == 3
        assert collector.entries[0].level == "WARN"  # escalated
        assert collector.entries[1].level == "WARN"  # escalation requested
        assert collector.entries[2].level == "INFO"  # resolved

    @pytest.mark.asyncio
    async def test_multiple_events_ordered(self):
        bus = EventBus()
        collector = LogCollector()
        bus.on_async(ExecutionEvent, collector.on_event)

        for status in [EventStatus.CREATED, EventStatus.RUNNING, EventStatus.FAILED]:
            await bus.emit(_event(status=status))

        assert len(collector.entries) == 3
        statuses = [e.event_status for e in collector.entries]
        assert statuses == ["created", "running", "failed"]
