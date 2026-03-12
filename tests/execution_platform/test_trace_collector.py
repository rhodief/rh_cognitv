"""Tests for TraceCollector — execution trace recording."""

import pytest

from rh_cognitv.execution_platform.event_bus import EventBus
from rh_cognitv.execution_platform.events import ExecutionEvent, TextPayload
from rh_cognitv.execution_platform.models import EventKind, EventStatus
from rh_cognitv.execution_platform.trace_collector import (
    Span,
    SpanEvent,
    SpanStatus,
    TraceCollector,
)


# ── Helpers ──


def _event(
    status: EventStatus = EventStatus.CREATED,
    event_id: str | None = None,
    parent_id: str | None = None,
    **kw,
) -> ExecutionEvent:
    defaults = dict(
        kind=EventKind.TEXT,
        payload=TextPayload(prompt="hello"),
        status=status,
    )
    if event_id is not None:
        defaults["id"] = event_id
    if parent_id is not None:
        defaults["parent_id"] = parent_id
    defaults.update(kw)
    return ExecutionEvent(**defaults)


# ──────────────────────────────────────────────
# Span model
# ──────────────────────────────────────────────


class TestSpanModel:
    def test_span_defaults(self):
        span = Span(trace_id="t1", name="test")
        assert span.trace_id == "t1"
        assert span.status == SpanStatus.UNSET
        assert span.end_time is None
        assert span.span_id  # auto-generated

    def test_span_event(self):
        se = SpanEvent(name="retrying", attributes={"event.status": "retrying"})
        assert se.name == "retrying"
        assert se.attributes["event.status"] == "retrying"


# ──────────────────────────────────────────────
# TraceCollector — span lifecycle
# ──────────────────────────────────────────────


class TestTraceCollectorLifecycle:
    @pytest.mark.asyncio
    async def test_running_creates_span(self):
        tc = TraceCollector(trace_id="t1")
        event = _event(status=EventStatus.RUNNING, event_id="e1")
        await tc.on_event(event)
        assert "e1" in tc.active_spans
        span = tc.active_spans["e1"]
        assert span.trace_id == "t1"
        assert span.name == "text.execute"
        assert span.status == SpanStatus.UNSET

    @pytest.mark.asyncio
    async def test_success_completes_span(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.SUCCESS, event_id="e1"))
        assert "e1" not in tc.active_spans
        assert len(tc.spans) == 1
        span = tc.spans[0]
        assert span.status == SpanStatus.OK
        assert span.end_time is not None

    @pytest.mark.asyncio
    async def test_failed_completes_with_error(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.FAILED, event_id="e1"))
        assert len(tc.spans) == 1
        assert tc.spans[0].status == SpanStatus.ERROR

    @pytest.mark.asyncio
    async def test_timed_out_completes_with_error(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.TIMED_OUT, event_id="e1"))
        assert len(tc.spans) == 1
        assert tc.spans[0].status == SpanStatus.ERROR

    @pytest.mark.asyncio
    async def test_cancelled_completes_with_error(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.CANCELLED, event_id="e1"))
        assert len(tc.spans) == 1
        assert tc.spans[0].status == SpanStatus.ERROR

    @pytest.mark.asyncio
    async def test_retrying_adds_span_event(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.RETRYING, event_id="e1"))
        span = tc.active_spans["e1"]
        assert len(span.events) == 1
        assert span.events[0].name == "retrying"

    @pytest.mark.asyncio
    async def test_escalated_adds_span_event(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.ESCALATED, event_id="e1"))
        span = tc.active_spans["e1"]
        assert any(e.name == "escalated" for e in span.events)

    @pytest.mark.asyncio
    async def test_waiting_adds_span_event(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.WAITING, event_id="e1"))
        span = tc.active_spans["e1"]
        assert any(e.name == "waiting" for e in span.events)

    @pytest.mark.asyncio
    async def test_created_queued_no_span(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.CREATED, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.QUEUED, event_id="e1"))
        assert len(tc.active_spans) == 0
        assert len(tc.spans) == 0

    @pytest.mark.asyncio
    async def test_late_terminal_creates_minimal_span(self):
        """If we see SUCCESS for an event we never saw RUNNING, create a minimal span."""
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.SUCCESS, event_id="e1"))
        assert len(tc.spans) == 1
        span = tc.spans[0]
        assert span.status == SpanStatus.OK
        assert span.end_time is not None


# ──────────────────────────────────────────────
# Span attributes & execution context
# ──────────────────────────────────────────────


class TestTraceCollectorContext:
    @pytest.mark.asyncio
    async def test_execution_context_in_attributes(self):
        tc = TraceCollector(
            trace_id="t1", execution_id="exec-1", node_id="node-1"
        )
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        span = tc.active_spans["e1"]
        assert span.attributes["execution.id"] == "exec-1"
        assert span.attributes["execution.node_id"] == "node-1"
        assert span.attributes["event.kind"] == "text"

    @pytest.mark.asyncio
    async def test_trace_id_auto_generated(self):
        tc = TraceCollector()
        assert tc.trace_id  # non-empty

    @pytest.mark.asyncio
    async def test_parent_span_linking(self):
        tc = TraceCollector(trace_id="t1")
        # Parent event
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="parent"))
        parent_span_id = tc.active_spans["parent"].span_id
        # Child event with parent_id
        await tc.on_event(
            _event(status=EventStatus.RUNNING, event_id="child", parent_id="parent")
        )
        child_span = tc.active_spans["child"]
        assert child_span.parent_span_id == parent_span_id

    @pytest.mark.asyncio
    async def test_parent_span_no_link_if_parent_not_active(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(
            _event(status=EventStatus.RUNNING, event_id="child", parent_id="unknown")
        )
        child_span = tc.active_spans["child"]
        assert child_span.parent_span_id is None

    @pytest.mark.asyncio
    async def test_parent_id_in_attributes(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(
            _event(status=EventStatus.RUNNING, event_id="e1", parent_id="p1")
        )
        assert tc.active_spans["e1"].attributes["event.parent_id"] == "p1"


# ──────────────────────────────────────────────
# Full lifecycle + nesting
# ──────────────────────────────────────────────


class TestTraceCollectorFullFlow:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.CREATED, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.QUEUED, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.RETRYING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.SUCCESS, event_id="e1"))
        # One completed span with retrying event
        assert len(tc.spans) == 1
        span = tc.spans[0]
        assert span.status == SpanStatus.OK
        # retrying + success events
        assert len(span.events) == 2

    @pytest.mark.asyncio
    async def test_nested_spans(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="parent"))
        await tc.on_event(
            _event(status=EventStatus.RUNNING, event_id="child", parent_id="parent")
        )
        await tc.on_event(
            _event(status=EventStatus.SUCCESS, event_id="child", parent_id="parent")
        )
        await tc.on_event(_event(status=EventStatus.SUCCESS, event_id="parent"))

        assert len(tc.spans) == 2
        child_span = tc.spans[0]
        parent_span = tc.spans[1]
        assert child_span.parent_span_id == parent_span.span_id

    @pytest.mark.asyncio
    async def test_multiple_concurrent_events(self):
        tc = TraceCollector(trace_id="t1")
        for i in range(5):
            await tc.on_event(_event(status=EventStatus.RUNNING, event_id=f"e{i}"))
        assert len(tc.active_spans) == 5

        for i in range(5):
            await tc.on_event(_event(status=EventStatus.SUCCESS, event_id=f"e{i}"))
        assert len(tc.active_spans) == 0
        assert len(tc.spans) == 5

    @pytest.mark.asyncio
    async def test_clear(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event(_event(status=EventStatus.RUNNING, event_id="e1"))
        await tc.on_event(_event(status=EventStatus.SUCCESS, event_id="e1"))
        tc.clear()
        assert len(tc.spans) == 0
        assert len(tc.active_spans) == 0

    @pytest.mark.asyncio
    async def test_ignores_non_execution_events(self):
        tc = TraceCollector(trace_id="t1")
        await tc.on_event("not an event")
        assert len(tc.active_spans) == 0
        assert len(tc.spans) == 0


# ──────────────────────────────────────────────
# EventBus integration
# ──────────────────────────────────────────────


class TestTraceCollectorEventBus:
    @pytest.mark.asyncio
    async def test_eventbus_integration(self):
        bus = EventBus()
        tc = TraceCollector(trace_id="t1", execution_id="exec-1")
        bus.on_async(ExecutionEvent, tc.on_event)

        event = _event(status=EventStatus.RUNNING, event_id="e1")
        await bus.emit(event)

        success = event.model_copy(update={"status": EventStatus.SUCCESS})
        await bus.emit(success)

        assert len(tc.spans) == 1
        span = tc.spans[0]
        assert span.status == SpanStatus.OK
        assert span.attributes["execution.id"] == "exec-1"

    @pytest.mark.asyncio
    async def test_eventbus_lifecycle_transitions(self):
        bus = EventBus()
        tc = TraceCollector(trace_id="t1")
        bus.on_async(ExecutionEvent, tc.on_event)

        eid = "lifecycle-e1"
        for status in [
            EventStatus.CREATED,
            EventStatus.QUEUED,
            EventStatus.RUNNING,
            EventStatus.RETRYING,
            EventStatus.RUNNING,
            EventStatus.SUCCESS,
        ]:
            await bus.emit(_event(status=status, event_id=eid))

        assert len(tc.spans) == 1
        assert tc.spans[0].status == SpanStatus.OK
