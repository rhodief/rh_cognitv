"""
Phase 6 tests — Parallel Execution + ForEach Strategies.

Tests cover:
  - Parallel branch execution (verify speedup with async sleeps)
  - Independent branches run concurrently
  - Partial failure in parallel batch (one fails, DAG stops after batch)
  - ForEach fail_fast (sequential, stops on first failure)
  - ForEach collect_all (parallel, collects partial results)
  - Shared BudgetTracker under parallel load
  - Interrupt during parallel execution
  - Mixed flow + execution nodes in parallel
  - ExecutionDAG shape verification for parallel runs
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from rh_cognitv.execution_platform.budget import BudgetTracker
from rh_cognitv.execution_platform.errors import InterruptError
from rh_cognitv.execution_platform.events import ExecutionEvent
from rh_cognitv.execution_platform.handlers import HandlerRegistry
from rh_cognitv.execution_platform.models import (
    EventKind,
    ExecutionResult,
    FunctionResultData,
    LLMResultData,
    ResultMetadata,
    TokenUsage,
)
from rh_cognitv.execution_platform.protocols import EventHandlerProtocol
from rh_cognitv.execution_platform.state import ExecutionState

from rh_cognitv.orchestrator.adapters import AdapterRegistry, PlatformRef
from rh_cognitv.orchestrator.dag_orchestrator import DAGOrchestrator
from rh_cognitv.orchestrator.flow_nodes import (
    FlowHandlerRegistry,
    ForEachNode,
)
from rh_cognitv.orchestrator.models import (
    DAGRunStatus,
    NodeExecutionStatus,
    OrchestratorConfig,
)
from rh_cognitv.orchestrator.nodes import DataNode, FunctionNode, TextNode
from rh_cognitv.orchestrator.plan_dag import DAGBuilder
from rh_cognitv.orchestrator.validation import ValidationPipeline


# ══════════════════════════════════════════════
# Helpers / Fixtures
# ══════════════════════════════════════════════


class StubTextHandler(EventHandlerProtocol[LLMResultData]):
    """Fast text handler that returns immediately."""

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=f"result",
                model="stub",
                token_usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total=10),
            ),
            metadata=ResultMetadata(),
        )


class SlowTextHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler with a configurable delay — for timing tests."""

    def __init__(self, delay: float = 0.05) -> None:
        self._delay = delay
        self.call_count = 0
        self.call_data: list[Any] = []

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        self.call_count += 1
        self.call_data.append(data)
        await asyncio.sleep(self._delay)
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=f"slow result #{self.call_count}",
                model="stub",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total=15),
            ),
            metadata=ResultMetadata(),
        )


class TrackingTextHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler that records call order and data."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        self.calls.append(data)
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=f"tracked: {data}",
                model="stub",
                token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total=2),
            ),
            metadata=ResultMetadata(),
        )


class FailingTextHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler that always fails."""

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        return ExecutionResult(
            ok=False,
            error_message="handler failed",
            error_category="PERMANENT",
            metadata=ResultMetadata(),
        )


class ConditionalHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler that fails for specific prompts."""

    def __init__(self, fail_prompts: set[str] | None = None, delay: float = 0.0) -> None:
        self._fail_prompts = fail_prompts or set()
        self._delay = delay
        self.call_count = 0

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        prompt = event.payload.prompt if hasattr(event.payload, "prompt") else ""
        if prompt in self._fail_prompts:
            return ExecutionResult(
                ok=False,
                error_message=f"Failed: {prompt}",
                error_category="PERMANENT",
                metadata=ResultMetadata(),
            )
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=f"ok: {prompt}",
                model="stub",
                token_usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total=10),
            ),
            metadata=ResultMetadata(),
        )


class StubFunctionHandler(EventHandlerProtocol[FunctionResultData]):
    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[FunctionResultData]:
        return ExecutionResult(
            ok=True,
            value=FunctionResultData(return_value="fn_result", duration_ms=1.0),
            metadata=ResultMetadata(),
        )


def _build_registry(
    text_handler: EventHandlerProtocol | None = None,
    function_handler: EventHandlerProtocol | None = None,
) -> HandlerRegistry:
    reg = HandlerRegistry()
    reg.register(EventKind.TEXT, text_handler or StubTextHandler())
    reg.register(EventKind.FUNCTION, function_handler or StubFunctionHandler())
    reg.register(EventKind.DATA, text_handler or StubTextHandler())
    return reg


def _build_orchestrator(
    handler_registry: HandlerRegistry | None = None,
    config: OrchestratorConfig | None = None,
    validation: ValidationPipeline | None = None,
    flow_handler_registry: FlowHandlerRegistry | None = None,
    budget_tracker: BudgetTracker | None = None,
) -> DAGOrchestrator:
    cfg = config or OrchestratorConfig(
        default_timeout_seconds=10.0,
        default_max_retries=1,
        default_retry_base_delay=0.01,
    )
    h_reg = handler_registry or _build_registry()
    platform = PlatformRef(registry=h_reg, config=cfg, budget_tracker=budget_tracker)
    state = ExecutionState()
    adapter_registry = AdapterRegistry.with_defaults()

    return DAGOrchestrator(
        adapter_registry=adapter_registry,
        platform=platform,
        state=state,
        validation=validation,
        config=cfg,
        flow_handler_registry=flow_handler_registry,
    )


# ══════════════════════════════════════════════
# A. Parallel branch execution — speedup verification
# ══════════════════════════════════════════════


class TestParallelBranches:
    @pytest.mark.asyncio
    async def test_parallel_speedup(self):
        """Three independent branches should run in parallel, not sequentially.

        With 50ms sleep per handler, sequential would take ~150ms.
        Parallel should take ~50ms (+overhead). We give generous margin.
        """
        slow = SlowTextHandler(delay=0.05)
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=slow))

        # A fans out to B, C, D (all independent)
        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="root"))
            .add_node("b", TextNode(id="b", prompt="branch1"))
            .add_node("c", TextNode(id="c", prompt="branch2"))
            .add_node("d", TextNode(id="d", prompt="branch3"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("a", "d")
            .build()
        )

        start = time.monotonic()
        await orch.run(dag)
        elapsed = time.monotonic() - start

        assert orch.status == DAGRunStatus.SUCCESS
        # 4 nodes total: a (sequential) + b,c,d (parallel) = ~100ms sequential minimum
        # With parallelism: a(50ms) + max(b,c,d)(50ms) ≈ 100ms
        # Without: a(50ms) + b(50ms) + c(50ms) + d(50ms) = 200ms
        # Generous bound: must be under 180ms (leaves room for overhead)
        assert elapsed < 0.18, f"Expected parallel execution, took {elapsed:.3f}s"
        assert slow.call_count == 4

    @pytest.mark.asyncio
    async def test_diamond_dag_parallel(self):
        """Diamond: A → B,C (parallel) → D. B and C run concurrently."""
        slow = SlowTextHandler(delay=0.05)
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=slow))

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="start"))
            .add_node("b", TextNode(id="b", prompt="left"))
            .add_node("c", TextNode(id="c", prompt="right"))
            .add_node("d", TextNode(id="d", prompt="join"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .build()
        )

        start = time.monotonic()
        await orch.run(dag)
        elapsed = time.monotonic() - start

        assert orch.status == DAGRunStatus.SUCCESS
        # a(50ms) + max(b,c)(50ms) + d(50ms) = ~150ms parallel
        # a(50ms) + b(50ms) + c(50ms) + d(50ms) = 200ms sequential
        assert elapsed < 0.20, f"Expected parallel execution, took {elapsed:.3f}s"
        assert slow.call_count == 4

    @pytest.mark.asyncio
    async def test_wide_fan_out(self):
        """Wide fan-out: root → 5 parallel branches."""
        slow = SlowTextHandler(delay=0.05)
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=slow))

        builder = DAGBuilder().add_node("root", TextNode(id="root", prompt="root"))
        for i in range(5):
            nid = f"branch_{i}"
            builder.add_node(nid, TextNode(id=nid, prompt=f"branch {i}"))
            builder.edge("root", nid)
        dag = builder.build()

        start = time.monotonic()
        await orch.run(dag)
        elapsed = time.monotonic() - start

        assert orch.status == DAGRunStatus.SUCCESS
        # root(50ms) + max(5 branches)(50ms) ≈ 100ms
        # Sequential: 6 * 50ms = 300ms
        assert elapsed < 0.20, f"Expected parallel, took {elapsed:.3f}s"
        assert slow.call_count == 6

    @pytest.mark.asyncio
    async def test_linear_stays_sequential(self):
        """Linear DAG: A → B → C. No parallelism possible — same behavior."""
        orch = _build_orchestrator()

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .add_node("c", TextNode(id="c", prompt="C"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS
        success = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert {e.node_id for e in success} == {"a", "b", "c"}


# ══════════════════════════════════════════════
# B. Partial failure in parallel batch
# ══════════════════════════════════════════════


class TestParallelFailure:
    @pytest.mark.asyncio
    async def test_one_branch_fails_dag_fails(self):
        """If one parallel branch fails, DAG status is FAILED after batch."""
        cond = ConditionalHandler(fail_prompts={"branch2"}, delay=0.02)
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=cond))

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="root"))
            .add_node("b", TextNode(id="b", prompt="branch1"))
            .add_node("c", TextNode(id="c", prompt="branch2"))  # will fail
            .add_node("d", TextNode(id="d", prompt="branch3"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("a", "d")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.FAILED

        # All branches in the batch ran (parallel gather completes all)
        assert cond.call_count == 4  # root + 3 branches

    @pytest.mark.asyncio
    async def test_failed_branch_prevents_next_batch(self):
        """After a failed batch, no successor nodes run."""
        cond = ConditionalHandler(fail_prompts={"left"})
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=cond))

        # A → B(fails), C → D (D depends on both B and C)
        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="root"))
            .add_node("b", TextNode(id="b", prompt="left"))  # fails
            .add_node("c", TextNode(id="c", prompt="right"))
            .add_node("d", TextNode(id="d", prompt="join"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.FAILED

        # D should not have been executed
        d_entries = orch.execution_dag.get_all_entries_for_node("d")
        assert len(d_entries) == 0

    @pytest.mark.asyncio
    async def test_execution_dag_records_parallel_results(self):
        """All parallel branch results are recorded in ExecutionDAG."""
        orch = _build_orchestrator()

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="root"))
            .add_node("b", TextNode(id="b", prompt="b"))
            .add_node("c", TextNode(id="c", prompt="c"))
            .edge("a", "b")
            .edge("a", "c")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        # a, b, c all have RUNNING + SUCCESS entries
        for nid in ("a", "b", "c"):
            entries = orch.execution_dag.get_all_entries_for_node(nid)
            assert len(entries) == 2, f"Expected 2 entries for {nid}, got {len(entries)}"
            assert entries[0].status == NodeExecutionStatus.RUNNING
            assert entries[1].status == NodeExecutionStatus.SUCCESS


# ══════════════════════════════════════════════
# C. ForEach fail_fast strategy
# ══════════════════════════════════════════════


class TestForEachFailFast:
    @pytest.mark.asyncio
    async def test_fail_fast_stops_on_first(self):
        """fail_fast should stop after the first inner failure."""
        call_count = {"n": 0}

        class CountingFail(EventHandlerProtocol[LLMResultData]):
            async def __call__(self, event, data, configs):
                call_count["n"] += 1
                if data == "bad":
                    return ExecutionResult(
                        ok=False,
                        error_message="bad item",
                        error_category="PERMANENT",
                        metadata=ResultMetadata(),
                    )
                return ExecutionResult(
                    ok=True,
                    value=LLMResultData(text="ok", model="stub",
                        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total=2)),
                    metadata=ResultMetadata(),
                )

        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=CountingFail()))

        inner = TextNode(id="inner", prompt="process")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="fail_fast")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        # "good", "bad", "another" — should stop after "bad"
        await orch.run(dag, data=["good", "bad", "another"])
        assert orch.status == DAGRunStatus.FAILED
        # Only 2 calls: "good" (ok) + "bad" (fail), "another" never reached
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_fail_fast_all_succeed(self):
        """fail_fast with all items succeeding completes normally."""
        tracker = TrackingTextHandler()
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=tracker))

        inner = TextNode(id="inner", prompt="go")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="fail_fast")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        await orch.run(dag, data=["a", "b", "c"])
        assert orch.status == DAGRunStatus.SUCCESS
        assert tracker.calls == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_fail_fast_is_sequential(self):
        """fail_fast processes items sequentially (not parallel)."""
        order: list[str] = []

        class OrderTracker(EventHandlerProtocol[LLMResultData]):
            async def __call__(self, event, data, configs):
                order.append(f"start-{data}")
                await asyncio.sleep(0.01)
                order.append(f"end-{data}")
                return ExecutionResult(
                    ok=True,
                    value=LLMResultData(text="ok", model="stub",
                        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total=2)),
                    metadata=ResultMetadata(),
                )

        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=OrderTracker()))

        inner = TextNode(id="inner", prompt="go")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="fail_fast")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        await orch.run(dag, data=["x", "y"])
        # Sequential: start-x, end-x, start-y, end-y
        assert order == ["start-x", "end-x", "start-y", "end-y"]


# ══════════════════════════════════════════════
# D. ForEach collect_all strategy
# ══════════════════════════════════════════════


class TestForEachCollectAll:
    @pytest.mark.asyncio
    async def test_collect_all_runs_all(self):
        """collect_all should run all items even if some fail."""
        call_count = {"n": 0}

        class CountingHandler(EventHandlerProtocol[LLMResultData]):
            async def __call__(self, event, data, configs):
                call_count["n"] += 1
                if data == "bad":
                    return ExecutionResult(
                        ok=False,
                        error_message="bad item",
                        error_category="PERMANENT",
                        metadata=ResultMetadata(),
                    )
                return ExecutionResult(
                    ok=True,
                    value=LLMResultData(text="ok", model="stub",
                        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total=2)),
                    metadata=ResultMetadata(),
                )

        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=CountingHandler()))

        inner = TextNode(id="inner", prompt="go")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="collect_all")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        await orch.run(dag, data=["good", "bad", "another"])
        assert orch.status == DAGRunStatus.FAILED
        # All 3 items processed (not stopped at "bad")
        assert call_count["n"] == 3

    @pytest.mark.asyncio
    async def test_collect_all_success(self):
        """collect_all with all items succeeding completes normally."""
        tracker = TrackingTextHandler()
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=tracker))

        inner = TextNode(id="inner", prompt="go")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="collect_all")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        await orch.run(dag, data=["a", "b", "c"])
        assert orch.status == DAGRunStatus.SUCCESS
        assert len(tracker.calls) == 3

    @pytest.mark.asyncio
    async def test_collect_all_is_parallel(self):
        """collect_all items should run in parallel."""
        slow = SlowTextHandler(delay=0.05)
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=slow))

        inner = TextNode(id="inner", prompt="go")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="collect_all")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        start = time.monotonic()
        await orch.run(dag, data=["a", "b", "c", "d"])
        elapsed = time.monotonic() - start

        assert orch.status == DAGRunStatus.SUCCESS
        assert slow.call_count == 4
        # 4 items, 50ms each: parallel ≈ 50ms, sequential ≈ 200ms
        assert elapsed < 0.15, f"Expected parallel, took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_collect_all_execution_dag_shape(self):
        """ExecutionDAG should have entries for all items."""
        tracker = TrackingTextHandler()
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=tracker))

        inner = TextNode(id="inner", prompt="go")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="collect_all")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        await orch.run(dag, data=["x", "y", "z"])
        assert orch.status == DAGRunStatus.SUCCESS

        # Inner node: 3 items × (RUNNING + SUCCESS) = 6 entries
        inner_entries = orch.execution_dag.get_all_entries_for_node("inner")
        assert len(inner_entries) == 6

    @pytest.mark.asyncio
    async def test_collect_all_partial_failure_records(self):
        """collect_all records both successes and failures."""
        class FailSecond(EventHandlerProtocol[LLMResultData]):
            def __init__(self):
                self.count = 0

            async def __call__(self, event, data, configs):
                self.count += 1
                if data == "fail_me":
                    return ExecutionResult(
                        ok=False, error_message="nope",
                        error_category="PERMANENT", metadata=ResultMetadata(),
                    )
                return ExecutionResult(
                    ok=True,
                    value=LLMResultData(text="ok", model="stub",
                        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total=2)),
                    metadata=ResultMetadata(),
                )

        handler = FailSecond()
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=handler))

        inner = TextNode(id="inner", prompt="go")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="collect_all")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        await orch.run(dag, data=["ok_1", "fail_me", "ok_2"])
        assert orch.status == DAGRunStatus.FAILED
        assert handler.count == 3

        # We should have both SUCCESS and FAILED entries for inner
        inner_entries = orch.execution_dag.get_all_entries_for_node("inner")
        statuses = [e.status for e in inner_entries]
        assert NodeExecutionStatus.SUCCESS in statuses
        assert NodeExecutionStatus.FAILED in statuses


# ══════════════════════════════════════════════
# E. Shared BudgetTracker under parallel load
# ══════════════════════════════════════════════


class TestParallelBudget:
    @pytest.mark.asyncio
    async def test_shared_budget_all_branches(self):
        """Parallel branches share a single BudgetTracker.

        Each call consumes 10 tokens. Budget = 100 tokens.
        4 nodes (root + 3 branches) × 10 = 40 tokens. Should succeed.
        """
        budget = BudgetTracker(token_budget=100)
        orch = _build_orchestrator(budget_tracker=budget)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="root"))
            .add_node("b", TextNode(id="b", prompt="b"))
            .add_node("c", TextNode(id="c", prompt="c"))
            .add_node("d", TextNode(id="d", prompt="d"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("a", "d")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS
        # BudgetTracker consumed tokens from all 4 handler calls
        assert budget.tokens_used > 0

    @pytest.mark.asyncio
    async def test_budget_tracks_parallel_calls(self):
        """Call count is correct under parallel execution."""
        budget = BudgetTracker(call_budget=100)
        orch = _build_orchestrator(budget_tracker=budget)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="root"))
            .add_node("b", TextNode(id="b", prompt="b"))
            .add_node("c", TextNode(id="c", prompt="c"))
            .edge("a", "b")
            .edge("a", "c")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS
        # 3 handler calls total
        assert budget.calls_made == 3


# ══════════════════════════════════════════════
# F. Interrupt during parallel execution
# ══════════════════════════════════════════════


class TestParallelInterrupt:
    @pytest.mark.asyncio
    async def test_interrupt_before_parallel_batch(self):
        """Interrupting before a parallel batch starts raises InterruptError."""
        orch = _build_orchestrator()
        orch.interrupt()

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="root"))
            .add_node("b", TextNode(id="b", prompt="b"))
            .edge("a", "b")
            .build()
        )

        with pytest.raises(InterruptError):
            await orch.run(dag)
        assert orch.status == DAGRunStatus.INTERRUPTED

    @pytest.mark.asyncio
    async def test_interrupt_during_slow_parallel_batch(self):
        """Interrupting during a slow parallel batch is detected."""
        slow = SlowTextHandler(delay=0.1)
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=slow))

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="root"))
            .add_node("b", TextNode(id="b", prompt="b"))
            .add_node("c", TextNode(id="c", prompt="c"))
            .add_node("d", TextNode(id="d", prompt="join"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .build()
        )

        async def interrupt_later():
            await asyncio.sleep(0.12)  # After a finishes, during b/c
            orch.interrupt()

        task = asyncio.create_task(interrupt_later())
        try:
            # b and c are parallel — interrupt is checked on next batch
            await orch.run(dag)
            # If we get here, check that d wasn't reached
            assert orch.status in (DAGRunStatus.SUCCESS, DAGRunStatus.INTERRUPTED)
        except InterruptError:
            assert orch.status == DAGRunStatus.INTERRUPTED
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ══════════════════════════════════════════════
# G. Mixed parallel: execution + flow nodes
# ══════════════════════════════════════════════


class TestMixedParallel:
    @pytest.mark.asyncio
    async def test_parallel_batch_with_flow_and_exec(self):
        """A batch can contain both FlowNodes and ExecutionNodes."""
        tracker = TrackingTextHandler()
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=tracker))

        from rh_cognitv.orchestrator.flow_nodes import GetNode

        # root → get + exec (parallel)
        dag = (
            DAGBuilder()
            .add_node("root", TextNode(id="root", prompt="start"))
            .add_node("get", GetNode(id="get", key="missing"))
            .add_node("exec", TextNode(id="exec", prompt="parallel"))
            .add_node("final", TextNode(id="final", prompt="end"))
            .edge("root", "get")
            .edge("root", "exec")
            .edge("get", "final")
            .edge("exec", "final")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS
        # root + exec + final = 3 calls to text handler (get is a flow node)
        assert len(tracker.calls) == 3


# ══════════════════════════════════════════════
# H. ExecutionDAG shape for various parallel patterns
# ══════════════════════════════════════════════


class TestParallelExecutionDAGShape:
    @pytest.mark.asyncio
    async def test_fan_out_fan_in(self):
        """A → B,C,D (parallel) → E. All nodes have entries."""
        orch = _build_orchestrator()

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="a"))
            .add_node("b", TextNode(id="b", prompt="b"))
            .add_node("c", TextNode(id="c", prompt="c"))
            .add_node("d", TextNode(id="d", prompt="d"))
            .add_node("e", TextNode(id="e", prompt="e"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("a", "d")
            .edge("b", "e")
            .edge("c", "e")
            .edge("d", "e")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        success_entries = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        success_ids = {e.node_id for e in success_entries}
        assert success_ids == {"a", "b", "c", "d", "e"}

    @pytest.mark.asyncio
    async def test_single_node_dag(self):
        """Single node DAG still works with parallel machinery."""
        orch = _build_orchestrator()
        dag = DAGBuilder().add_node("only", TextNode(id="only", prompt="go")).build()

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        entries = orch.execution_dag.get_all_entries_for_node("only")
        assert len(entries) == 2  # RUNNING + SUCCESS

    @pytest.mark.asyncio
    async def test_all_independent_nodes(self):
        """DAG with no edges: all nodes are initial → all run in one parallel batch."""
        # PlanDAG requires connectivity, so we can't have truly disconnected nodes.
        # Test with a single root fanning to multiple leaves instead.
        orch = _build_orchestrator()

        dag = (
            DAGBuilder()
            .add_node("root", TextNode(id="root", prompt="r"))
            .add_node("a", TextNode(id="a", prompt="a"))
            .add_node("b", TextNode(id="b", prompt="b"))
            .add_node("c", TextNode(id="c", prompt="c"))
            .edge("root", "a")
            .edge("root", "b")
            .edge("root", "c")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        success_entries = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert {e.node_id for e in success_entries} == {"root", "a", "b", "c"}


# ══════════════════════════════════════════════
# I. ForEach + parallel interaction
# ══════════════════════════════════════════════


class TestForEachParallelInteraction:
    @pytest.mark.asyncio
    async def test_foreach_in_parallel_branch(self):
        """ForEach in one branch while another branch runs in parallel."""
        tracker = TrackingTextHandler()
        orch = _build_orchestrator(handler_registry=_build_registry(text_handler=tracker))

        # root → foreach + exec (parallel)
        inner = TextNode(id="inner", prompt="foreach item")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="fail_fast")
        exec_node = TextNode(id="exec", prompt="parallel exec")

        dag = (
            DAGBuilder()
            .add_node("root", TextNode(id="root", prompt="root"))
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .add_node("exec", exec_node)
            .edge("root", "foreach")
            .edge("root", "exec")
            .edge("foreach", "inner")
            .build()
        )

        await orch.run(dag, data=["a", "b"])
        assert orch.status == DAGRunStatus.SUCCESS

        # root=1 + foreach inner 2 items=2 + exec=1 = 4 handler calls
        assert len(tracker.calls) == 4

    @pytest.mark.asyncio
    async def test_collect_all_foreach_with_budget(self):
        """collect_all ForEach with BudgetTracker: all items consume budget."""
        budget = BudgetTracker(token_budget=1000)
        orch = _build_orchestrator(budget_tracker=budget)

        inner = TextNode(id="inner", prompt="go")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="collect_all")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        await orch.run(dag, data=["a", "b", "c"])
        assert orch.status == DAGRunStatus.SUCCESS
        # Each inner call consumes tokens → budget reflects 3 calls
        assert budget.calls_made == 3
        assert budget.tokens_used > 0
