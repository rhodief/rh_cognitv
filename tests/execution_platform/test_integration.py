"""
Phase 7 — Integration & Smoke Tests.

End-to-end validation that all L3 Execution Platform components work together.

7.1: Full lifecycle — event → PolicyChain → handler → result → state → log + trace
7.2: Time-travel — execute 5 events, undo 2, verify state at step 3
7.3: Budget exhaustion — run until budget exceeded, verify BudgetError
7.4: Escalation — handler escalates, verify lifecycle pauses + cloud recovery
7.5: Parallel events — concurrent asyncio events, verify no state corruption
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rh_cognitv.execution_platform.budget import BudgetTracker
from rh_cognitv.execution_platform.errors import BudgetError, TransientError
from rh_cognitv.execution_platform.event_bus import EventBus
from rh_cognitv.execution_platform.events import (
    EscalationRequested,
    EscalationResolved,
    ExecutionEvent,
    TextPayload,
)
from rh_cognitv.execution_platform.handlers import HandlerRegistry, TextHandler
from rh_cognitv.execution_platform.log_collector import LogCollector
from rh_cognitv.execution_platform.models import (
    EventKind,
    EventStatus,
    ExecutionResult,
    LLMResultData,
    ResultMetadata,
    TokenUsage,
)
from rh_cognitv.execution_platform.policies import (
    BudgetPolicy,
    PolicyChain,
    RetryPolicy,
    TimeoutPolicy,
)
from rh_cognitv.execution_platform.protocols import EventHandlerProtocol
from rh_cognitv.execution_platform.state import ExecutionState
from rh_cognitv.execution_platform.state_middleware import StateSnapshotMiddleware
from rh_cognitv.execution_platform.trace_collector import SpanStatus, TraceCollector


# ──────────────────────────────────────────────
# Helpers — custom handlers for integration tests
# ──────────────────────────────────────────────


class CountingTextHandler(EventHandlerProtocol[LLMResultData]):
    """Handler that counts calls and returns token usage for budget tracking."""

    def __init__(self, tokens_per_call: int = 10) -> None:
        self.call_count = 0
        self.tokens_per_call = tokens_per_call

    async def __call__(
        self, event: Any, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        self.call_count += 1
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=f"response-{self.call_count}",
                model="test-model",
                token_usage=TokenUsage(
                    prompt_tokens=self.tokens_per_call // 2,
                    completion_tokens=self.tokens_per_call // 2,
                    total=self.tokens_per_call,
                ),
            ),
            metadata=ResultMetadata(attempt=1),
        )


class FailThenSucceedHandler(EventHandlerProtocol[LLMResultData]):
    """Handler that fails N times with TransientError, then succeeds."""

    def __init__(self, fail_count: int = 2) -> None:
        self._fail_count = fail_count
        self._attempts = 0

    async def __call__(
        self, event: Any, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise TransientError(f"Transient failure #{self._attempts}")
        return ExecutionResult(
            ok=True,
            value=LLMResultData(text="recovered", model="test"),
            metadata=ResultMetadata(attempt=self._attempts),
        )


def _text_event(
    status: EventStatus = EventStatus.CREATED, event_id: str | None = None, **kw
) -> ExecutionEvent:
    defaults = dict(kind=EventKind.TEXT, payload=TextPayload(prompt="test"), status=status)
    if event_id is not None:
        defaults["id"] = event_id
    defaults.update(kw)
    return ExecutionEvent(**defaults)


# ══════════════════════════════════════════════
# 7.1 — Full Lifecycle Test
# ══════════════════════════════════════════════


class TestFullLifecycle:
    """Create event → PolicyChain (budget + retry) → handler → ExecutionResult
    → state snapshot → log + trace output."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_end_to_end(self):
        # --- Setup ---
        bus = EventBus()
        state = ExecutionState()
        tracker = BudgetTracker(token_budget=1000, call_budget=10)
        log_collector = LogCollector(execution_id="exec-lifecycle")
        trace_collector = TraceCollector(
            trace_id="trace-lifecycle", execution_id="exec-lifecycle"
        )

        # Wire middleware
        bus.use(StateSnapshotMiddleware(state))

        # Wire async subscribers
        bus.on_async(ExecutionEvent, log_collector.on_event)
        bus.on_async(ExecutionEvent, trace_collector.on_event)

        # Setup handler + policies
        handler = CountingTextHandler(tokens_per_call=50)
        registry = HandlerRegistry()
        registry.register(EventKind.TEXT, handler)

        chain = PolicyChain([
            BudgetPolicy(tracker=tracker),
            RetryPolicy(max_attempts=3, base_delay=0.01),
        ])

        # --- Event lifecycle ---
        event = _text_event(event_id="evt-1")

        # 1) CREATED
        await bus.emit(event)

        # 2) RUNNING
        running = event.model_copy(update={"status": EventStatus.RUNNING})
        await bus.emit(running)

        # 3) Execute through PolicyChain
        result = await chain(handler, event, None, None)

        # 4) SUCCESS
        success = event.model_copy(update={"status": EventStatus.SUCCESS})
        await bus.emit(success)

        # --- Assertions ---
        # Handler executed
        assert result.ok is True
        assert result.value.text == "response-1"
        assert handler.call_count == 1

        # Budget consumed (BudgetPolicy.after_execute auto-consumed from result)
        snap = tracker.remaining()
        assert snap.tokens_remaining == 950
        assert snap.calls_remaining == 9

        # State snapshots taken (RUNNING + SUCCESS = 2 snapshots)
        assert state.version_count == 2
        current = state.get_current()
        assert current["_last_event_status"] == "success"
        assert current["_last_event_id"] == "evt-1"

        # Logs captured
        assert len(log_collector.entries) >= 3  # CREATED + RUNNING + SUCCESS
        statuses = [e.event_status for e in log_collector.entries if e.event_status]
        assert "created" in statuses
        assert "running" in statuses
        assert "success" in statuses

        # Traces captured
        assert len(trace_collector.spans) == 1
        span = trace_collector.spans[0]
        assert span.status == SpanStatus.OK
        assert span.attributes["event.kind"] == "text"
        assert span.attributes["execution.id"] == "exec-lifecycle"

    @pytest.mark.asyncio
    async def test_lifecycle_with_retry(self):
        """Handler fails twice with TransientError, succeeds on 3rd attempt."""
        bus = EventBus()
        state = ExecutionState()
        log_collector = LogCollector(execution_id="exec-retry")
        trace_collector = TraceCollector(trace_id="trace-retry")

        bus.use(StateSnapshotMiddleware(state))
        bus.on_async(ExecutionEvent, log_collector.on_event)
        bus.on_async(ExecutionEvent, trace_collector.on_event)

        handler = FailThenSucceedHandler(fail_count=2)
        retry = RetryPolicy(max_attempts=3, base_delay=0.01)

        event = _text_event(event_id="evt-retry")

        # Emit RUNNING
        await bus.emit(event.model_copy(update={"status": EventStatus.RUNNING}))

        # Execute with retry
        result = await retry.execute_with_retry(handler, event, None, None)
        assert result.ok is True
        assert result.value.text == "recovered"

        # Emit RETRYING events for logging
        await bus.emit(event.model_copy(update={"status": EventStatus.RETRYING}))
        await bus.emit(event.model_copy(update={"status": EventStatus.RETRYING}))

        # Emit SUCCESS
        await bus.emit(event.model_copy(update={"status": EventStatus.SUCCESS}))

        # State had snapshots on RUNNING + 2 RETRYING + SUCCESS = 4
        assert state.version_count == 4

        # Log captured all transitions
        log_statuses = [e.event_status for e in log_collector.entries if e.event_status]
        assert log_statuses.count("retrying") == 2

        # Trace: span opened on RUNNING, retrying events, closed on SUCCESS
        assert len(trace_collector.spans) == 1
        span = trace_collector.spans[0]
        assert span.status == SpanStatus.OK
        # 2 retrying span events + 1 success terminal event
        assert len(span.events) == 3


# ══════════════════════════════════════════════
# 7.2 — Time-Travel Test
# ══════════════════════════════════════════════


class TestTimeTravel:
    """Execute 5 events, undo 2, verify state matches snapshot at step 3."""

    @pytest.mark.asyncio
    async def test_execute_5_undo_2_verify_step_3(self):
        bus = EventBus()
        state = ExecutionState()
        bus.use(StateSnapshotMiddleware(state))

        # Execute 5 events (each goes through RUNNING → SUCCESS)
        event_ids = []
        for i in range(1, 6):
            eid = f"evt-{i}"
            event_ids.append(eid)
            running = _text_event(status=EventStatus.RUNNING, event_id=eid)
            await bus.emit(running)
            success = _text_event(status=EventStatus.SUCCESS, event_id=eid)
            await bus.emit(success)

        # 5 events × 2 transitions = 10 snapshots
        assert state.version_count == 10

        # Current state should reflect the 5th event
        current = state.get_current()
        assert current["_last_event_id"] == "evt-5"
        assert current["_last_event_status"] == "success"

        # Undo 4 times to get back to evt-3 RUNNING (snapshot at step 6 → step 5 is evt-3 success)
        # Snapshots: 0=evt-1 running, 1=evt-1 success, 2=evt-2 running, 3=evt-2 success,
        #            4=evt-3 running, 5=evt-3 success, 6=evt-4 running, 7=evt-4 success,
        #            8=evt-5 running, 9=evt-5 success
        # Undo 4 → cursor at 5 (evt-3 success)
        for _ in range(4):
            assert state.undo()

        restored = state.get_current()
        assert restored["_last_event_id"] == "evt-3"
        assert restored["_last_event_status"] == "success"

        # Redo 2 to get to evt-4 success
        assert state.redo()
        assert state.redo()
        restored = state.get_current()
        assert restored["_last_event_id"] == "evt-4"
        assert restored["_last_event_status"] == "success"

    @pytest.mark.asyncio
    async def test_restore_specific_version(self):
        bus = EventBus()
        state = ExecutionState()
        bus.use(StateSnapshotMiddleware(state))

        versions = []
        for i in range(1, 6):
            running = _text_event(status=EventStatus.RUNNING, event_id=f"evt-{i}")
            await bus.emit(running)
            versions.append(state.current_version)

        # Restore to version of evt-3 running
        state.restore(versions[2])
        restored = state.get_current()
        assert restored["_last_event_id"] == "evt-3"
        assert restored["_last_event_status"] == "running"

    @pytest.mark.asyncio
    async def test_undo_redo_preserves_isolation(self):
        """Verify that state restored via undo is fully independent."""
        state = ExecutionState({"counter": 0})
        state.update("counter", 1)
        state.snapshot()  # v0: counter=1
        state.update("counter", 2)
        state.snapshot()  # v1: counter=2
        state.update("counter", 3)
        state.snapshot()  # v2: counter=3

        state.undo()  # back to v1
        assert state.get_current()["counter"] == 2
        # Mutating current state after undo shouldn't affect stored snapshots
        state.update("counter", 99)
        state.undo()  # back to v0
        assert state.get_current()["counter"] == 1


# ══════════════════════════════════════════════
# 7.3 — Budget Exhaustion Test
# ══════════════════════════════════════════════


class TestBudgetExhaustion:
    """Run events until budget exceeded, verify BudgetError and graceful stop."""

    @pytest.mark.asyncio
    async def test_budget_exhaustion_via_policy_chain(self):
        # token_budget=100, handler yields 40 tokens per call
        # Call 1: 40/100, Call 2: 80/100, Call 3: after_execute tries 120/100 → BudgetError
        tracker = BudgetTracker(token_budget=100, call_budget=10)
        handler = CountingTextHandler(tokens_per_call=40)
        chain = PolicyChain([BudgetPolicy(tracker=tracker)])

        event = _text_event()

        # Calls 1 and 2 succeed (BudgetPolicy.after_execute auto-consumes)
        await chain(handler, event, None, None)
        await chain(handler, event, None, None)
        assert tracker.tokens_used == 80
        assert tracker.calls_made == 2

        # Call 3: before_execute passes (80 < 100), but after_execute tries
        # to consume 40 more → 120 > 100 → BudgetError
        with pytest.raises(BudgetError, match="Token budget exceeded"):
            await chain(handler, event, None, None)

    @pytest.mark.asyncio
    async def test_budget_blocks_before_execute(self):
        """BudgetPolicy.before_execute raises BudgetError when exhausted."""
        tracker = BudgetTracker(call_budget=2)
        handler = CountingTextHandler()
        chain = PolicyChain([BudgetPolicy(tracker=tracker)])

        event = _text_event()

        # Calls 1 and 2 (after_execute auto-consumes 1 call each)
        await chain(handler, event, None, None)
        await chain(handler, event, None, None)
        assert tracker.calls_made == 2

        # Call 3 blocked by before_execute (calls_made=2 >= call_budget=2)
        assert not tracker.can_proceed()
        with pytest.raises(BudgetError, match="Budget exhausted"):
            await chain(handler, event, None, None)

    @pytest.mark.asyncio
    async def test_budget_with_logging(self):
        """Budget exhaustion is observable through logging."""
        bus = EventBus()
        log = LogCollector(execution_id="budget-test")
        bus.on_async(ExecutionEvent, log.on_event)

        tracker = BudgetTracker(call_budget=1)
        handler = CountingTextHandler()
        chain = PolicyChain([BudgetPolicy(tracker=tracker)])

        event = _text_event(event_id="budget-evt")

        # Emit RUNNING and execute (BudgetPolicy.after_execute auto-consumes)
        await bus.emit(event.model_copy(update={"status": EventStatus.RUNNING}))
        await chain(handler, event, None, None)
        await bus.emit(event.model_copy(update={"status": EventStatus.SUCCESS}))

        # Second call should fail
        event2 = _text_event(event_id="budget-evt-2")
        await bus.emit(event2.model_copy(update={"status": EventStatus.RUNNING}))
        with pytest.raises(BudgetError):
            await chain(handler, event2, None, None)
        await bus.emit(event2.model_copy(update={"status": EventStatus.FAILED}))

        # Logs captured both flows
        statuses = [e.event_status for e in log.entries if e.event_status]
        assert "running" in statuses
        assert "success" in statuses
        assert "failed" in statuses


# ══════════════════════════════════════════════
# 7.4 — Escalation Test
# ══════════════════════════════════════════════


class TestEscalation:
    """Handler returns ESCALATED status, verify event lifecycle pauses."""

    @pytest.mark.asyncio
    async def test_escalation_roundtrip_via_eventbus(self):
        """Full escalation flow: emit ESCALATED → EscalationRequested →
        wait_for EscalationResolved → resume."""
        bus = EventBus()
        state = ExecutionState()
        log = LogCollector(execution_id="esc-test")
        trace = TraceCollector(trace_id="esc-trace")

        bus.use(StateSnapshotMiddleware(state))
        bus.on_async(ExecutionEvent, log.on_event)
        bus.on_async(ExecutionEvent, trace.on_event)
        bus.on_async(EscalationRequested, log.on_event)
        bus.on_async(EscalationResolved, log.on_event)

        event = _text_event(event_id="esc-1")

        # 1. Event starts running
        await bus.emit(event.model_copy(update={"status": EventStatus.RUNNING}))

        # 2. Handler escalates
        await bus.emit(event.model_copy(update={"status": EventStatus.ESCALATED}))

        # 3. Escalation context saved in state
        state.set_escalated(
            event_id="esc-1",
            question="Approve deployment?",
            options=["approve", "reject"],
            node_id="deploy-node",
        )
        esc = state.get_escalation()
        assert esc is not None
        assert esc["question"] == "Approve deployment?"
        assert esc["status"] == "escalated"

        # 4. Emit EscalationRequested (would go to SSE/frontend)
        await bus.emit(
            EscalationRequested(
                event_id="esc-1",
                question="Approve deployment?",
                options=["approve", "reject"],
                node_id="deploy-node",
            )
        )

        # 5. Simulate human decision arriving asynchronously
        async def resolve_after_delay():
            await asyncio.sleep(0.05)
            await bus.emit(EscalationResolved(event_id="esc-1", decision="approve"))

        # Start wait_for and resolve concurrently
        resolve_task = asyncio.ensure_future(resolve_after_delay())
        decision = await bus.wait_for(
            EscalationResolved,
            filter=lambda e: e.event_id == "esc-1",
            timeout=2.0,
        )
        await resolve_task

        assert decision.decision == "approve"

        # 6. Clear escalation and resume
        state.clear_escalation()
        assert state.get_escalation() is None

        # 7. Event completes
        await bus.emit(event.model_copy(update={"status": EventStatus.SUCCESS}))

        # --- Verify state ---
        # Snapshots: RUNNING, ESCALATED, set_escalated snapshot, SUCCESS = 4 from bus + 1 from set_escalated
        assert state.version_count >= 3

        # --- Verify logs ---
        log_messages = [e.message for e in log.entries]
        # Should have: running, escalated, escalation requested, escalation resolved, success
        assert any("Approve deployment?" in m for m in log_messages)
        assert any("approve" in m for m in log_messages)

    @pytest.mark.asyncio
    async def test_escalation_cloud_recovery(self):
        """State persisted with ESCALATED can be recovered by new process."""
        # --- Process 1: escalate and persist ---
        state1 = ExecutionState()
        state1.update("current_node", "deploy-step")
        version = state1.set_escalated(
            event_id="cloud-evt-1",
            question="Approve?",
            options=["yes", "no"],
            node_id="deploy-step",
            resume_data={"attempt": 3},
        )

        # Serialize (simulating persist to durable storage)
        serialized = state1.serialize_current()

        # --- Process 2: recover and resume ---
        state2 = ExecutionState()
        state2.deserialize_into(serialized)

        esc = state2.get_escalation()
        assert esc is not None
        assert esc["status"] == "escalated"
        assert esc["event_id"] == "cloud-evt-1"
        assert esc["question"] == "Approve?"
        assert esc["options"] == ["yes", "no"]
        assert esc["resume_data"] == {"attempt": 3}

        # Inject decision and resume
        state2.clear_escalation()
        state2.update("decision", "yes")
        state2.update("resumed", True)

        current = state2.get_current()
        assert current["decision"] == "yes"
        assert current["resumed"] is True
        assert state2.get_escalation() is None

    @pytest.mark.asyncio
    async def test_escalation_trace_span_stays_open(self):
        """Trace span should remain active during escalation (not completed)."""
        trace = TraceCollector(trace_id="esc-trace")

        await trace.on_event(
            _text_event(status=EventStatus.RUNNING, event_id="esc-t1")
        )
        await trace.on_event(
            _text_event(status=EventStatus.ESCALATED, event_id="esc-t1")
        )

        # Span should still be active (escalated is not terminal)
        assert "esc-t1" in trace.active_spans
        assert len(trace.spans) == 0

        span = trace.active_spans["esc-t1"]
        assert any(e.name == "escalated" for e in span.events)

        # After resolution, event succeeds
        await trace.on_event(
            _text_event(status=EventStatus.SUCCESS, event_id="esc-t1")
        )
        assert len(trace.spans) == 1
        assert trace.spans[0].status == SpanStatus.OK


# ══════════════════════════════════════════════
# 7.5 — Parallel Event Test
# ══════════════════════════════════════════════


class TestParallelEvents:
    """Multiple concurrent events via asyncio, verify no state corruption."""

    @pytest.mark.asyncio
    async def test_parallel_events_no_state_corruption(self):
        """Run 10 concurrent events through EventBus + state middleware."""
        bus = EventBus()
        state = ExecutionState()
        log = LogCollector(execution_id="parallel-test")

        bus.use(StateSnapshotMiddleware(state))
        bus.on_async(ExecutionEvent, log.on_event)

        async def run_event(i: int):
            eid = f"par-{i}"
            running = _text_event(status=EventStatus.RUNNING, event_id=eid)
            await bus.emit(running)
            # Small delay to interleave
            await asyncio.sleep(0.01)
            success = _text_event(status=EventStatus.SUCCESS, event_id=eid)
            await bus.emit(success)

        # Launch all 10 concurrently
        await asyncio.gather(*[run_event(i) for i in range(10)])

        # All 10 events × 2 transitions = 20 snapshots
        assert state.version_count == 20

        # Logs captured all events (20 transitions)
        assert len(log.entries) == 20

        # All event IDs are represented
        logged_ids = {e.event_id for e in log.entries}
        expected_ids = {f"par-{i}" for i in range(10)}
        assert expected_ids == logged_ids

    @pytest.mark.asyncio
    async def test_parallel_handlers_independent_results(self):
        """Concurrent handler executions produce independent results."""
        handler = CountingTextHandler(tokens_per_call=10)
        registry = HandlerRegistry()
        registry.register(EventKind.TEXT, handler)

        async def execute_event(i: int) -> ExecutionResult:
            event = _text_event(event_id=f"h-{i}")
            return await registry.handle(event, None, None)

        results = await asyncio.gather(*[execute_event(i) for i in range(10)])

        assert all(r.ok for r in results)
        assert handler.call_count == 10
        # Each result has a response
        texts = {r.value.text for r in results}
        assert len(texts) == 10  # all unique (response-1 through response-10)

    @pytest.mark.asyncio
    async def test_parallel_with_trace_collector(self):
        """Trace collector handles concurrent events correctly."""
        trace = TraceCollector(trace_id="parallel-trace")

        async def trace_event(i: int):
            eid = f"t-{i}"
            await trace.on_event(_text_event(status=EventStatus.RUNNING, event_id=eid))
            await asyncio.sleep(0.01)
            await trace.on_event(_text_event(status=EventStatus.SUCCESS, event_id=eid))

        await asyncio.gather(*[trace_event(i) for i in range(10)])

        assert len(trace.spans) == 10
        assert len(trace.active_spans) == 0
        assert all(s.status == SpanStatus.OK for s in trace.spans)

    @pytest.mark.asyncio
    async def test_parallel_mixed_success_and_failure(self):
        """Concurrent events with mixed outcomes."""
        bus = EventBus()
        state = ExecutionState()
        trace = TraceCollector(trace_id="mixed-trace")

        bus.use(StateSnapshotMiddleware(state))
        bus.on_async(ExecutionEvent, trace.on_event)

        async def run_event(i: int):
            eid = f"mix-{i}"
            await bus.emit(_text_event(status=EventStatus.RUNNING, event_id=eid))
            await asyncio.sleep(0.005)
            final = EventStatus.SUCCESS if i % 2 == 0 else EventStatus.FAILED
            await bus.emit(_text_event(status=final, event_id=eid))

        await asyncio.gather(*[run_event(i) for i in range(10)])

        assert state.version_count == 20  # 10 RUNNING + 10 terminal
        assert len(trace.spans) == 10
        ok_spans = [s for s in trace.spans if s.status == SpanStatus.OK]
        err_spans = [s for s in trace.spans if s.status == SpanStatus.ERROR]
        assert len(ok_spans) == 5
        assert len(err_spans) == 5


# ══════════════════════════════════════════════
# 7.6 — Cross-Component Wiring (bonus)
# ══════════════════════════════════════════════


class TestCrossComponent:
    """Verify that all components integrate cleanly."""

    @pytest.mark.asyncio
    async def test_all_components_wired(self, tmp_path):
        """Full system: EventBus + State + ContextStore + Log + Trace + Budget."""
        from rh_cognitv.execution_platform.context_store import ContextStore
        from rh_cognitv.execution_platform.models import (
            EntryContent,
            Memory,
            MemoryOrigin,
            MemoryQuery,
            MemoryRole,
            MemoryShape,
            Provenance,
            TimeInfo,
        )
        from rh_cognitv.execution_platform.types import EntryRef

        bus = EventBus()
        state = ExecutionState()
        store = ContextStore(tmp_path / "store")
        tracker = BudgetTracker(token_budget=500, call_budget=10)
        log = LogCollector(execution_id="full-system")
        trace = TraceCollector(trace_id="full-trace", execution_id="full-system")

        bus.use(StateSnapshotMiddleware(state))
        bus.on_async(ExecutionEvent, log.on_event)
        bus.on_async(ExecutionEvent, trace.on_event)

        handler = CountingTextHandler(tokens_per_call=25)
        registry = HandlerRegistry()
        registry.register(EventKind.TEXT, handler)
        chain = PolicyChain([BudgetPolicy(tracker=tracker)])

        # Store a memory
        mem = Memory(
            content=EntryContent(text="User prefers Python"),
            role=MemoryRole.SEMANTIC,
            shape=MemoryShape.ATOM,
            provenance=Provenance(origin=MemoryOrigin.TOLD, source="user"),
            time=TimeInfo(
                recorded_at="2024-01-01T00:00:00Z",
                observed_at="2024-01-01T00:00:00Z",
            ),
        )
        mem_id = await store.remember(mem)

        # Put ref in state
        state.update("context_refs", [mem_id])
        state.snapshot()

        # Execute an event
        event = _text_event(event_id="full-1")
        await bus.emit(event.model_copy(update={"status": EventStatus.RUNNING}))
        result = await chain(handler, event, None, None)
        await bus.emit(event.model_copy(update={"status": EventStatus.SUCCESS}))

        # Verify everything (BudgetPolicy.after_execute auto-consumed)
        assert result.ok
        assert handler.call_count == 1
        assert tracker.calls_made == 1
        assert state.version_count >= 2  # initial snapshot + RUNNING + SUCCESS
        assert len(log.entries) >= 2
        assert len(trace.spans) == 1

        # Resolve EntryRef from state
        refs = state.get_current()["context_refs"]
        ref = EntryRef(id=refs[0], entry_type=Memory)
        resolved = await ref.resolve(store)
        assert isinstance(resolved, Memory)
        assert resolved.content.text == "User prefers Python"

        # Recall from store
        results = await store.recall(MemoryQuery(role=MemoryRole.SEMANTIC))
        assert len(results) == 1
        assert results[0].entry.content.text == "User prefers Python"
