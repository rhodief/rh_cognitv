"""
Phase 8 — Integration & Smoke Tests.

End-to-end validation with the full L3 stack.

  8.1  Full lifecycle: PlanDAG → DAGOrchestrator → adapters → L3 handlers → ExecutionDAG
  8.2  Time-travel: execute 5 nodes, restore to step 3, verify state + ExecutionDAG
  8.3  Budget exhaustion: run nodes until budget exceeded, verify graceful stop
  8.4  ForEach parallel: fan-out 3 branches, collect results (full L3 stack)
  8.5  Mixed DAG: linear + branch + ForEach + Switch in one DAG
  8.6  Escalation: handler error flow through orchestrator + L3 escalation primitives
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from rh_cognitv.execution_platform.budget import BudgetTracker
from rh_cognitv.execution_platform.errors import (
    BudgetError,
    EscalationError,
    InterruptError,
)
from rh_cognitv.execution_platform.event_bus import EventBus
from rh_cognitv.execution_platform.events import (
    EscalationRequested,
    EscalationResolved,
    ExecutionEvent,
)
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
from rh_cognitv.orchestrator.execution_dag import ExecutionDAG
from rh_cognitv.orchestrator.flow_nodes import (
    FilterNode,
    FlowHandlerRegistry,
    ForEachNode,
    GetNode,
    IfNotOkNode,
    SwitchNode,
)
from rh_cognitv.orchestrator.models import (
    DAGRunStatus,
    NodeExecutionStatus,
    NodeResult,
    OrchestratorConfig,
)
from rh_cognitv.orchestrator.nodes import DataNode, FunctionNode, TextNode, ToolNode
from rh_cognitv.orchestrator.plan_dag import DAGBuilder
from rh_cognitv.orchestrator.validation import (
    BudgetValidator,
    DependencyValidator,
    InputSchemaValidator,
    ValidationPipeline,
)


# ══════════════════════════════════════════════
# Realistic L3 handlers for integration tests
# ══════════════════════════════════════════════


class RealisticTextHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler that echoes prompt and tracks realistic token usage."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        prompt = event.payload.prompt
        self.calls.append(prompt)
        prompt_tokens = max(len(prompt.split()), 1)
        completion_tokens = 10
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=f"Response to: {prompt}",
                model="integration-test",
                token_usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total=prompt_tokens + completion_tokens,
                ),
            ),
            metadata=ResultMetadata(),
        )


class RealisticFunctionHandler(EventHandlerProtocol[FunctionResultData]):
    """Function handler that returns controllable results."""

    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self._results = results or {}
        self.calls: list[str] = []

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[FunctionResultData]:
        fn_name = event.payload.function_name
        self.calls.append(fn_name)
        return_value = self._results.get(fn_name, f"result:{fn_name}")
        return ExecutionResult(
            ok=True,
            value=FunctionResultData(return_value=return_value, duration_ms=1.0),
            metadata=ResultMetadata(),
        )


class FailOnPromptHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler that fails for specific prompts, succeeds otherwise."""

    def __init__(self, fail_prompts: set[str]) -> None:
        self._fail_prompts = fail_prompts
        self.calls: list[str] = []

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        prompt = event.payload.prompt
        self.calls.append(prompt)
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
                text=f"OK: {prompt}",
                model="integration-test",
                token_usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total=10),
            ),
            metadata=ResultMetadata(),
        )


class EscalatingHandler(EventHandlerProtocol[LLMResultData]):
    """Handler that raises EscalationError for specific prompts."""

    def __init__(self, escalate_prompts: set[str]) -> None:
        self._escalate_prompts = escalate_prompts
        self.calls: list[str] = []

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        prompt = event.payload.prompt
        self.calls.append(prompt)
        if prompt in self._escalate_prompts:
            raise EscalationError(f"Need human input for: {prompt}")
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=f"OK: {prompt}",
                model="integration-test",
                token_usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total=10),
            ),
            metadata=ResultMetadata(),
        )


class SlowRealisticHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler with a configurable delay for timing tests."""

    def __init__(self, delay: float = 0.05) -> None:
        self._delay = delay
        self.calls: list[str] = []

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        prompt = event.payload.prompt
        self.calls.append(prompt)
        await asyncio.sleep(self._delay)
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=f"Slow response: {prompt}",
                model="integration-test",
                token_usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total=10),
            ),
            metadata=ResultMetadata(),
        )


# ══════════════════════════════════════════════
# Integration helper — full L3 stack
# ══════════════════════════════════════════════


def _build_full_stack(
    text_handler: EventHandlerProtocol | None = None,
    function_handler: EventHandlerProtocol | None = None,
    budget_tracker: BudgetTracker | None = None,
    validators: list | None = None,
    config: OrchestratorConfig | None = None,
    state: ExecutionState | None = None,
) -> DAGOrchestrator:
    """Build a DAGOrchestrator with real L3 components.

    Uses real HandlerRegistry, real PolicyChain (BudgetPolicy +
    TimeoutPolicy + RetryPolicy), real ExecutionState.
    """
    text_h = text_handler or RealisticTextHandler()
    fn_h = function_handler or RealisticFunctionHandler()

    # Real L3 HandlerRegistry
    handler_registry = HandlerRegistry()
    handler_registry.register(EventKind.TEXT, text_h)
    handler_registry.register(EventKind.FUNCTION, fn_h)
    handler_registry.register(EventKind.DATA, text_h)  # DATA uses same handler shape
    handler_registry.register(EventKind.TOOL, text_h)  # simplified for integration

    cfg = config or OrchestratorConfig(
        default_timeout_seconds=10.0,
        default_max_retries=1,
        default_retry_base_delay=0.01,
    )

    # Real PlatformRef → builds real PolicyChain per node
    platform = PlatformRef(
        registry=handler_registry,
        config=cfg,
        budget_tracker=budget_tracker,
    )

    exec_state = state or ExecutionState()

    # ValidationPipeline with real validators
    vp = ValidationPipeline(validators) if validators else ValidationPipeline()

    adapter_registry = AdapterRegistry.with_defaults()

    return DAGOrchestrator(
        adapter_registry=adapter_registry,
        platform=platform,
        state=exec_state,
        validation=vp,
        config=cfg,
        flow_handler_registry=FlowHandlerRegistry.with_defaults(),
    )


# ══════════════════════════════════════════════
# 8.1 Full Lifecycle
# ══════════════════════════════════════════════


class TestFullLifecycle:
    """End-to-end: PlanDAG → DAGOrchestrator → adapters → L3 handlers → ExecutionDAG."""

    @pytest.mark.asyncio
    async def test_linear_dag_full_l3_stack(self):
        """A→B→C with real handlers, real PolicyChain, real ExecutionState."""
        text_h = RealisticTextHandler()
        state = ExecutionState()
        orch = _build_full_stack(text_handler=text_h, state=state)

        dag = (
            DAGBuilder("linear-lifecycle")
            .add_node("a", TextNode(id="a", prompt="Hello world"))
            .add_node("b", TextNode(id="b", prompt="Process data"))
            .add_node("c", TextNode(id="c", prompt="Summarize results"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )

        result_dag = await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        # Handler called in order
        assert text_h.calls == ["Hello world", "Process data", "Summarize results"]

        # ExecutionDAG has RUNNING + SUCCESS for each node (6 entries)
        assert result_dag.entry_count() == 6

        # All three nodes succeeded
        success = result_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert {e.node_id for e in success} == {"a", "b", "c"}

        # Every SUCCESS entry has a state_version
        for entry in success:
            assert entry.state_version is not None

        # State snapshots were taken (3 per node)
        assert state.version_count == 3

    @pytest.mark.asyncio
    async def test_mixed_node_types(self):
        """Mix TextNode and FunctionNode in one DAG with full L3 stack."""
        text_h = RealisticTextHandler()
        fn_h = RealisticFunctionHandler(results={"clean": "cleaned_data"})
        orch = _build_full_stack(text_handler=text_h, function_handler=fn_h)

        dag = (
            DAGBuilder("mixed-types")
            .add_node("extract", TextNode(id="extract", prompt="Extract entities"))
            .add_node("clean", FunctionNode(id="clean", function_name="clean"))
            .add_node("summarize", TextNode(id="summarize", prompt="Summarize"))
            .edge("extract", "clean")
            .edge("clean", "summarize")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        # Text handler called for extract + summarize
        assert text_h.calls == ["Extract entities", "Summarize"]
        # Function handler called for clean
        assert fn_h.calls == ["clean"]

        # ExecutionDAG shape
        success = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert {e.node_id for e in success} == {"extract", "clean", "summarize"}

    @pytest.mark.asyncio
    async def test_execution_dag_entries_complete(self):
        """Every entry has required fields populated."""
        orch = _build_full_stack()

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="go"))
            .add_node("b", TextNode(id="b", prompt="next"))
            .edge("a", "b")
            .build()
        )

        await orch.run(dag)

        for entry in orch.execution_dag.entries():
            assert entry.id is not None
            assert entry.node_id in ("a", "b")
            assert entry.plan_node_ref == entry.node_id
            assert entry.started_at is not None
            if entry.status == NodeExecutionStatus.SUCCESS:
                assert entry.result is not None
                assert entry.result.ok is True
                assert entry.completed_at is not None
                assert entry.state_version is not None

    @pytest.mark.asyncio
    async def test_node_result_has_token_usage(self):
        """NodeResult.from_execution_result preserves token usage."""
        text_h = RealisticTextHandler()
        orch = _build_full_stack(text_handler=text_h)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="one two three four"))
            .build()
        )

        await orch.run(dag)

        entry = orch.execution_dag.get_entry("a")
        assert entry is not None
        assert entry.result is not None
        assert entry.result.ok
        # "one two three four" = 4 prompt tokens + 10 completion = 14 total
        assert entry.result.token_usage is not None
        assert entry.result.token_usage.prompt_tokens == 4
        assert entry.result.token_usage.completion_tokens == 10
        assert entry.result.token_usage.total == 14

    @pytest.mark.asyncio
    async def test_node_result_value_is_text(self):
        """NodeResult.value is the LLM text response (normalized from LLMResultData)."""
        orch = _build_full_stack()

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="Hello"))
            .build()
        )

        await orch.run(dag)

        entry = orch.execution_dag.get_entry("a")
        assert entry.result.value == "Response to: Hello"

    @pytest.mark.asyncio
    async def test_function_node_result_value(self):
        """FunctionNode result is the return_value."""
        fn_h = RealisticFunctionHandler(results={"compute": 42})
        orch = _build_full_stack(function_handler=fn_h)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="start"))
            .add_node("fn", FunctionNode(id="fn", function_name="compute"))
            .edge("a", "fn")
            .build()
        )

        await orch.run(dag)
        entry = orch.execution_dag.get_entry("fn")
        assert entry.result.ok
        assert entry.result.value == 42

    @pytest.mark.asyncio
    async def test_state_nesting_levels(self):
        """ExecutionState.add_level/remove_level bracket the run correctly."""
        state = ExecutionState()
        assert state.level == 0

        orch = _build_full_stack(state=state)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="go"))
            .build()
        )

        await orch.run(dag)
        # After run completes, level should be back to 0
        assert state.level == 0

    @pytest.mark.asyncio
    async def test_failed_node_records_failure(self):
        """A node that returns ok=False is recorded as FAILED."""
        fail_h = FailOnPromptHandler(fail_prompts={"fail_me"})
        orch = _build_full_stack(text_handler=fail_h)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="ok_step"))
            .add_node("b", TextNode(id="b", prompt="fail_me"))
            .edge("a", "b")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.FAILED

        a_entry = orch.execution_dag.get_entry("a")
        assert a_entry.status == NodeExecutionStatus.SUCCESS

        b_entry = orch.execution_dag.get_entry("b")
        assert b_entry.status == NodeExecutionStatus.FAILED
        assert "fail_me" in b_entry.result.error_message


# ══════════════════════════════════════════════
# 8.2 Time-Travel
# ══════════════════════════════════════════════


class TestTimeTravel:
    """Execute nodes, verify state_version recording, restore to earlier state."""

    @pytest.mark.asyncio
    async def test_5_nodes_state_versions_sequential(self):
        """5-node linear DAG: each entry has a unique, increasing state_version."""
        state = ExecutionState()
        orch = _build_full_stack(state=state)

        builder = DAGBuilder("five-step")
        for i in range(5):
            nid = f"step_{i}"
            builder.add_node(nid, TextNode(id=nid, prompt=f"Step {i}"))
        for i in range(4):
            builder.edge(f"step_{i}", f"step_{i+1}")
        dag = builder.build()

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        # Collect state_versions from SUCCESS entries in order
        versions = []
        for i in range(5):
            entry = orch.execution_dag.get_entry(f"step_{i}")
            assert entry.status == NodeExecutionStatus.SUCCESS
            assert entry.state_version is not None
            versions.append(entry.state_version)

        # Versions should be strictly increasing
        assert versions == sorted(versions)
        assert len(set(versions)) == 5  # all unique

        # State has 5 snapshots
        assert state.version_count == 5

    @pytest.mark.asyncio
    async def test_restore_to_step_3(self):
        """Execute 5 nodes, restore state to step 3, verify current_version."""
        state = ExecutionState()
        orch = _build_full_stack(state=state)

        builder = DAGBuilder()
        for i in range(5):
            nid = f"n{i}"
            builder.add_node(nid, TextNode(id=nid, prompt=f"Node {i}"))
        for i in range(4):
            builder.edge(f"n{i}", f"n{i+1}")
        dag = builder.build()

        await orch.run(dag)

        # Get version at step 3 (index 2, 0-based)
        step3_entry = orch.execution_dag.get_entry("n2")
        v3 = step3_entry.state_version

        # Restore to step 3
        state.restore(v3)
        assert state.current_version == v3

        # ExecutionDAG entries are still all there (append-only)
        assert orch.execution_dag.entry_count() == 10  # 5 RUNNING + 5 SUCCESS

    @pytest.mark.asyncio
    async def test_restore_preserves_execution_dag(self):
        """Restoring state doesn't affect ExecutionDAG (it's separate)."""
        state = ExecutionState()
        orch = _build_full_stack(state=state)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="one"))
            .add_node("b", TextNode(id="b", prompt="two"))
            .add_node("c", TextNode(id="c", prompt="three"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )

        await orch.run(dag)

        v_a = orch.execution_dag.get_entry("a").state_version
        state.restore(v_a)

        # All entries still present
        success = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert {e.node_id for e in success} == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_mark_rolled_back_after_restore(self):
        """mark_rolled_back marks entries from a target point forward."""
        state = ExecutionState()
        orch = _build_full_stack(state=state)

        builder = DAGBuilder()
        for i in range(5):
            nid = f"s{i}"
            builder.add_node(nid, TextNode(id=nid, prompt=f"Step {i}"))
        for i in range(4):
            builder.edge(f"s{i}", f"s{i+1}")
        dag = builder.build()

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        # Find step 2's SUCCESS entry
        s2_entries = orch.execution_dag.get_all_entries_for_node("s2")
        s2_success = [e for e in s2_entries if e.status == NodeExecutionStatus.SUCCESS][0]

        # Mark rolled back from step 2 forward
        count = orch.execution_dag.mark_rolled_back(s2_success)
        assert count > 0

        # Steps 0 and 1 should still be SUCCESS
        for nid in ("s0", "s1"):
            entries = orch.execution_dag.get_all_entries_for_node(nid)
            # RUNNING entry was before s2's SUCCESS, so it stays
            # except s0 RUNNING and s0 SUCCESS should be before s2's SUCCESS
            # and remain unaffected (they're before the rollback point)
            success_entries = [e for e in entries if e.status == NodeExecutionStatus.SUCCESS]
            assert len(success_entries) == 1

        # Steps 3 and 4 should be ROLLED_BACK
        for nid in ("s3", "s4"):
            entries = orch.execution_dag.get_all_entries_for_node(nid)
            assert all(
                e.status == NodeExecutionStatus.ROLLED_BACK for e in entries
            )

    @pytest.mark.asyncio
    async def test_undo_redo_via_l3_state(self):
        """L3 ExecutionState.undo()/redo() works with orchestrator-created snapshots."""
        state = ExecutionState()
        orch = _build_full_stack(state=state)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="first"))
            .add_node("b", TextNode(id="b", prompt="second"))
            .add_node("c", TextNode(id="c", prompt="third"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )

        await orch.run(dag)

        v_c = orch.execution_dag.get_entry("c").state_version
        v_b = orch.execution_dag.get_entry("b").state_version
        v_a = orch.execution_dag.get_entry("a").state_version

        # Current should be at last snapshot (c)
        assert state.current_version == v_c

        # Undo once → b
        assert state.undo() is True
        assert state.current_version == v_b

        # Undo again → a
        assert state.undo() is True
        assert state.current_version == v_a

        # Can't undo further
        assert state.undo() is False

        # Redo → b
        assert state.redo() is True
        assert state.current_version == v_b

        # Redo → c
        assert state.redo() is True
        assert state.current_version == v_c

    @pytest.mark.asyncio
    async def test_gc_collect_long_running(self):
        """L3 gc_collect keeps recent snapshots for long-running DAGs."""
        state = ExecutionState()
        orch = _build_full_stack(state=state)

        builder = DAGBuilder()
        for i in range(10):
            nid = f"n{i}"
            builder.add_node(nid, TextNode(id=nid, prompt=f"Node {i}"))
        for i in range(9):
            builder.edge(f"n{i}", f"n{i+1}")
        dag = builder.build()

        await orch.run(dag)
        assert state.version_count == 10

        # Keep first 2 and last 3
        removed = state.gc_collect(keep_first=2, keep_last=3)
        assert removed == 5
        assert state.version_count == 5


# ══════════════════════════════════════════════
# 8.3 Budget Exhaustion
# ══════════════════════════════════════════════


class TestBudgetExhaustion:
    """Budget limits halt execution gracefully."""

    @pytest.mark.asyncio
    async def test_budget_exhaustion_at_validation(self):
        """BudgetValidator pre-flight check stops before adapter call."""
        # "first" = 1 word → 1 prompt + 10 completion = 11 tokens consumed.
        # With token_budget=11, after node a: tokens_used=11 >= 11 → can_proceed()=False.
        budget = BudgetTracker(token_budget=11)
        orch = _build_full_stack(budget_tracker=budget, validators=[BudgetValidator()])

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="first"))
            .add_node("b", TextNode(id="b", prompt="second"))
            .add_node("c", TextNode(id="c", prompt="third"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.FAILED

        # First node succeeded (consumed exactly 11 tokens)
        a_entry = orch.execution_dag.get_entry("a")
        assert a_entry.status == NodeExecutionStatus.SUCCESS

        # Second node failed at validation (BudgetValidator sees budget exhausted)
        b_entry = orch.execution_dag.get_entry("b")
        assert b_entry.status == NodeExecutionStatus.FAILED
        assert "Budget exhausted" in b_entry.result.error_message

        # Third node never reached
        c_entries = orch.execution_dag.get_all_entries_for_node("c")
        assert len(c_entries) == 0

    @pytest.mark.asyncio
    async def test_budget_exhaustion_at_policy_layer(self):
        """BudgetPolicy in PolicyChain catches budget exhaustion during execution."""
        # Set call_budget=2: after 2 calls, BudgetPolicy.before_execute raises BudgetError
        budget = BudgetTracker(call_budget=2)
        orch = _build_full_stack(budget_tracker=budget)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="first"))
            .add_node("b", TextNode(id="b", prompt="second"))
            .add_node("c", TextNode(id="c", prompt="third"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.FAILED

        # First two nodes succeed (calls 1 and 2)
        assert orch.execution_dag.get_entry("a").status == NodeExecutionStatus.SUCCESS
        assert orch.execution_dag.get_entry("b").status == NodeExecutionStatus.SUCCESS

        # Third node fails (BudgetError caught by orchestrator)
        c_entry = orch.execution_dag.get_entry("c")
        assert c_entry.status == NodeExecutionStatus.FAILED
        assert "Budget" in c_entry.result.error_message or "budget" in c_entry.result.error_message.lower()

    @pytest.mark.asyncio
    async def test_budget_consumed_accurately(self):
        """BudgetTracker tracks tokens and calls across the full pipeline."""
        budget = BudgetTracker(token_budget=1000, call_budget=100)
        orch = _build_full_stack(budget_tracker=budget)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="one two three"))
            .add_node("b", TextNode(id="b", prompt="four five"))
            .edge("a", "b")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        # 2 calls made
        assert budget.calls_made == 2

        # "one two three" = 3 prompt + 10 completion = 13
        # "four five"     = 2 prompt + 10 completion = 12
        # Total: 25
        assert budget.tokens_used == 25

    @pytest.mark.asyncio
    async def test_budget_with_parallel_branches(self):
        """Budget tracking is accurate under parallel execution."""
        budget = BudgetTracker(token_budget=1000, call_budget=100)
        orch = _build_full_stack(budget_tracker=budget)

        dag = (
            DAGBuilder()
            .add_node("root", TextNode(id="root", prompt="start"))
            .add_node("b1", TextNode(id="b1", prompt="branch one"))
            .add_node("b2", TextNode(id="b2", prompt="branch two"))
            .add_node("b3", TextNode(id="b3", prompt="branch three"))
            .edge("root", "b1")
            .edge("root", "b2")
            .edge("root", "b3")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS
        assert budget.calls_made == 4  # root + 3 branches
        assert budget.tokens_used > 0

    @pytest.mark.asyncio
    async def test_budget_validator_plus_policy_coordination(self):
        """BudgetValidator and BudgetPolicy coordinate on the same BudgetTracker.

        The validator pre-checks; the policy enforces and consumes.
        """
        budget = BudgetTracker(call_budget=3)
        orch = _build_full_stack(
            budget_tracker=budget,
            validators=[BudgetValidator()],
        )

        builder = DAGBuilder()
        for i in range(5):
            nid = f"n{i}"
            builder.add_node(nid, TextNode(id=nid, prompt=f"step {i}"))
        for i in range(4):
            builder.edge(f"n{i}", f"n{i+1}")
        dag = builder.build()

        await orch.run(dag)
        assert orch.status == DAGRunStatus.FAILED

        # First 3 nodes succeed (calls 1, 2, 3)
        for i in range(3):
            assert orch.execution_dag.get_entry(f"n{i}").status == NodeExecutionStatus.SUCCESS

        # 4th node fails at validation (BudgetValidator sees budget exhausted)
        n3_entry = orch.execution_dag.get_entry("n3")
        assert n3_entry.status == NodeExecutionStatus.FAILED

        # Exactly 3 calls consumed
        assert budget.calls_made == 3


# ══════════════════════════════════════════════
# 8.4 ForEach Parallel (Full L3 Stack)
# ══════════════════════════════════════════════


class TestForEachParallelFullStack:
    """ForEach with full L3 handlers and PolicyChain."""

    @pytest.mark.asyncio
    async def test_foreach_collect_all_3_items(self):
        """ForEach collect_all with 3 items through full L3 stack."""
        text_h = RealisticTextHandler()
        budget = BudgetTracker(token_budget=1000, call_budget=100)
        orch = _build_full_stack(text_handler=text_h, budget_tracker=budget)

        inner = TextNode(id="inner", prompt="process item")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="collect_all")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        await orch.run(dag, data=["item_a", "item_b", "item_c"])
        assert orch.status == DAGRunStatus.SUCCESS

        # 3 inner executions
        inner_entries = orch.execution_dag.get_all_entries_for_node("inner")
        # Each item: RUNNING + SUCCESS = 2 entries × 3 items = 6
        assert len(inner_entries) == 6
        assert budget.calls_made == 3

    @pytest.mark.asyncio
    async def test_foreach_fail_fast_with_budget(self):
        """ForEach fail_fast stops early, budget reflects only executed items."""
        fail_h = FailOnPromptHandler(fail_prompts={"process item"})
        budget = BudgetTracker(token_budget=1000, call_budget=100)
        orch = _build_full_stack(text_handler=fail_h, budget_tracker=budget)

        inner = TextNode(id="inner", prompt="process item")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="fail_fast")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        await orch.run(dag, data=["a", "b", "c"])
        assert orch.status == DAGRunStatus.FAILED

        # Handler fails on every item (prompt is "process item" for all).
        # fail_fast stops after first failure.
        assert budget.calls_made == 1

    @pytest.mark.asyncio
    async def test_foreach_collect_all_parallel_speedup(self):
        """collect_all items run in parallel — verify speedup."""
        slow_h = SlowRealisticHandler(delay=0.05)
        orch = _build_full_stack(text_handler=slow_h)

        inner = TextNode(id="inner", prompt="slow work")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="collect_all")

        dag = (
            DAGBuilder()
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .edge("foreach", "inner")
            .build()
        )

        start = time.monotonic()
        await orch.run(dag, data=[1, 2, 3, 4])
        elapsed = time.monotonic() - start

        assert orch.status == DAGRunStatus.SUCCESS
        # 4 items parallel, 50ms each → ~50ms total, not 200ms
        assert elapsed < 0.15, f"Expected parallel, took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_foreach_with_subsequent_node(self):
        """ForEach followed by a subsequent node (successor of forEach)."""
        text_h = RealisticTextHandler()
        orch = _build_full_stack(text_handler=text_h)

        inner = TextNode(id="inner", prompt="process")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="collect_all")
        final = TextNode(id="final", prompt="summarize")

        dag = (
            DAGBuilder()
            .add_node("start", TextNode(id="start", prompt="init"))
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .add_node("final", final)
            .edge("start", "foreach")
            .edge("foreach", "inner")
            .edge("foreach", "final")
            .build()
        )

        await orch.run(dag, data=["a", "b"])
        assert orch.status == DAGRunStatus.SUCCESS

        # start + 2 inner calls + final = 4 text handler calls
        assert len(text_h.calls) == 4


# ══════════════════════════════════════════════
# 8.5 Mixed DAG
# ══════════════════════════════════════════════


class TestMixedDAG:
    """Complex DAGs combining linear, branching, ForEach, and Switch."""

    @pytest.mark.asyncio
    async def test_linear_plus_parallel_branches(self):
        """A → B,C (parallel) → D. Mixed with function + text nodes."""
        text_h = RealisticTextHandler()
        fn_h = RealisticFunctionHandler(results={"transform": "transformed"})
        orch = _build_full_stack(text_handler=text_h, function_handler=fn_h)

        dag = (
            DAGBuilder("linear-parallel")
            .add_node("extract", TextNode(id="extract", prompt="Extract"))
            .add_node("transform", FunctionNode(id="transform", function_name="transform"))
            .add_node("analyze", TextNode(id="analyze", prompt="Analyze"))
            .add_node("merge", TextNode(id="merge", prompt="Merge results"))
            .edge("extract", "transform")
            .edge("extract", "analyze")
            .edge("transform", "merge")
            .edge("analyze", "merge")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        success = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert {e.node_id for e in success} == {"extract", "transform", "analyze", "merge"}

    @pytest.mark.asyncio
    async def test_switch_routing(self):
        """Switch node routes to correct branch based on data."""
        text_h = RealisticTextHandler()
        orch = _build_full_stack(text_handler=text_h)

        dag = (
            DAGBuilder("switch-dag")
            .add_node("start", TextNode(id="start", prompt="Begin"))
            .add_node(
                "route",
                SwitchNode(
                    id="route",
                    condition="mode",
                    branches={"fast": "quick_path", "thorough": "deep_path"},
                    default_branch="quick_path",
                ),
            )
            .add_node("quick_path", TextNode(id="quick_path", prompt="Quick"))
            .add_node("deep_path", TextNode(id="deep_path", prompt="Deep"))
            .add_node("done", TextNode(id="done", prompt="Done"))
            .edge("start", "route")
            .edge("route", "quick_path")
            .edge("route", "deep_path")
            .edge("quick_path", "done")
            .edge("deep_path", "done")
            .build()
        )

        await orch.run(dag, data={"mode": "thorough"})
        assert orch.status == DAGRunStatus.SUCCESS

        # deep_path was taken
        deep = orch.execution_dag.get_entry("deep_path")
        assert deep.status == NodeExecutionStatus.SUCCESS

        # quick_path was skipped
        quick_entries = orch.execution_dag.get_all_entries_for_node("quick_path")
        skipped = [e for e in quick_entries if e.status == NodeExecutionStatus.SKIPPED]
        assert len(skipped) == 1

        # done was reached
        assert orch.execution_dag.get_entry("done").status == NodeExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_if_not_ok_passes_when_ok(self):
        """IfNotOk passes through when checked node succeeded."""
        orch = _build_full_stack()

        dag = (
            DAGBuilder("error-handling")
            .add_node("step", TextNode(id="step", prompt="safe"))
            .add_node(
                "check",
                IfNotOkNode(id="check", check_node_id="step", redirect_to="fallback"),
            )
            .add_node("fallback", TextNode(id="fallback", prompt="recover"))
            .add_node("done", TextNode(id="done", prompt="finish"))
            .edge("step", "check")
            .edge("check", "fallback")
            .edge("check", "done")
            .edge("fallback", "done")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        # step succeeded → check passes through (no redirect)
        assert orch.execution_dag.get_entry("step").status == NodeExecutionStatus.SUCCESS
        assert orch.execution_dag.get_entry("check").status == NodeExecutionStatus.SUCCESS
        assert orch.execution_dag.get_entry("done").status == NodeExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_if_not_ok_dag_stops_on_upstream_failure(self):
        """When the checked node fails, the DAG stops before IfNotOk runs.

        IfNotOk error-redirect requires the orchestrator to continue
        past individual node failures (Phase 7+ enhancement).
        """
        fail_h = FailOnPromptHandler(fail_prompts={"risky"})
        orch = _build_full_stack(text_handler=fail_h)

        dag = (
            DAGBuilder("error-handling")
            .add_node("step", TextNode(id="step", prompt="risky"))
            .add_node(
                "check",
                IfNotOkNode(id="check", check_node_id="step", redirect_to="fallback"),
            )
            .add_node("fallback", TextNode(id="fallback", prompt="recover"))
            .add_node("done", TextNode(id="done", prompt="finish"))
            .edge("step", "check")
            .edge("check", "fallback")
            .edge("check", "done")
            .edge("fallback", "done")
            .build()
        )

        await orch.run(dag)
        # DAG fails because step fails (batch_failed) before check can redirect
        assert orch.status == DAGRunStatus.FAILED

        assert orch.execution_dag.get_entry("step").status == NodeExecutionStatus.FAILED
        # check never ran
        assert len(orch.execution_dag.get_all_entries_for_node("check")) == 0

    @pytest.mark.asyncio
    async def test_foreach_in_branch_of_diamond(self):
        """Diamond DAG with ForEach in one branch."""
        text_h = RealisticTextHandler()
        budget = BudgetTracker(token_budget=10000)
        orch = _build_full_stack(text_handler=text_h, budget_tracker=budget)

        inner = TextNode(id="inner", prompt="process item")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="collect_all")

        dag = (
            DAGBuilder("diamond-foreach")
            .add_node("start", TextNode(id="start", prompt="Begin"))
            .add_node("left", TextNode(id="left", prompt="Left branch"))
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .add_node("join", TextNode(id="join", prompt="Join"))
            .edge("start", "left")
            .edge("start", "foreach")
            .edge("left", "join")
            .edge("foreach", "inner")
            .edge("foreach", "join")
            .build()
        )

        await orch.run(dag, data=["x", "y"])
        assert orch.status == DAGRunStatus.SUCCESS

        # start=1 + left=1 + inner×2 + join=1 = 5 calls
        assert len(text_h.calls) == 5
        assert budget.calls_made == 5

    @pytest.mark.asyncio
    async def test_complex_mixed_dag(self):
        """Larger DAG: linear → parallel split → one branch has ForEach, other has function."""
        text_h = RealisticTextHandler()
        fn_h = RealisticFunctionHandler(results={"compute": 99})
        orch = _build_full_stack(text_handler=text_h, function_handler=fn_h)

        inner = TextNode(id="inner", prompt="iterate")
        foreach = ForEachNode(id="foreach", inner_node_id="inner", failure_strategy="fail_fast")

        dag = (
            DAGBuilder("complex")
            .add_node("start", TextNode(id="start", prompt="Init"))
            .add_node("prep", TextNode(id="prep", prompt="Prepare"))
            # Parallel split
            .add_node("foreach", foreach)
            .add_node("inner", inner)
            .add_node("compute", FunctionNode(id="compute", function_name="compute"))
            # Join
            .add_node("merge", TextNode(id="merge", prompt="Merge all"))
            .edge("start", "prep")
            .edge("prep", "foreach")
            .edge("prep", "compute")
            .edge("foreach", "inner")
            .edge("foreach", "merge")
            .edge("compute", "merge")
            .build()
        )

        await orch.run(dag, data=["a", "b", "c"])
        assert orch.status == DAGRunStatus.SUCCESS

        # start + prep + inner×3 + compute + merge = 7
        assert len(text_h.calls) == 6  # start + prep + inner×3 + merge
        assert fn_h.calls == ["compute"]

        success = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        success_ids = {e.node_id for e in success}
        assert {"start", "prep", "foreach", "inner", "compute", "merge"} <= success_ids

    @pytest.mark.asyncio
    async def test_get_node_injects_data(self):
        """GetNode retrieves data from a prior node result."""
        text_h = RealisticTextHandler()
        orch = _build_full_stack(text_handler=text_h)

        dag = (
            DAGBuilder("get-node")
            .add_node("source", TextNode(id="source", prompt="Generate"))
            .add_node("get", GetNode(id="get", key="source"))
            .add_node("use", TextNode(id="use", prompt="Use data"))
            .edge("source", "get")
            .edge("get", "use")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        # get node resolved data from source result
        get_entry = orch.execution_dag.get_entry("get")
        assert get_entry.status == NodeExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_filter_node_in_dag(self):
        """FilterNode filters data for successor nodes."""
        text_h = RealisticTextHandler()
        orch = _build_full_stack(text_handler=text_h)

        # Custom flow handler registry with filter function
        flow_registry = FlowHandlerRegistry.with_defaults()

        from rh_cognitv.orchestrator.flow_nodes import DAGTraversalState

        orch_with_filter = _build_full_stack(text_handler=text_h)

        dag = (
            DAGBuilder("filter-dag")
            .add_node("start", TextNode(id="start", prompt="Begin"))
            .add_node(
                "filter",
                FilterNode(id="filter", condition="is_valid"),
            )
            .add_node("process", TextNode(id="process", prompt="Process filtered"))
            .edge("start", "filter")
            .edge("filter", "process")
            .build()
        )

        # Filter condition needs to be in dag_state.ext — which we can't
        # inject directly. FilterHandler returns ok=False if condition not found.
        # This tests the error path through the full stack.
        await orch_with_filter.run(dag, data=["a", "b"])
        # Filter fails because condition function not in ext → flow failure
        assert orch_with_filter.status == DAGRunStatus.FAILED


# ══════════════════════════════════════════════
# 8.6 Escalation
# ══════════════════════════════════════════════


class TestEscalation:
    """Escalation error flow and L3 escalation primitives."""

    @pytest.mark.asyncio
    async def test_escalation_error_recorded_as_failure(self):
        """Handler raising EscalationError is caught and recorded as FAILED.

        Phase 7 would add pause/resume — here we verify the error flows correctly.
        """
        esc_h = EscalatingHandler(escalate_prompts={"need_approval"})
        orch = _build_full_stack(text_handler=esc_h)

        dag = (
            DAGBuilder("escalation")
            .add_node("auto", TextNode(id="auto", prompt="automatic"))
            .add_node("gate", TextNode(id="gate", prompt="need_approval"))
            .add_node("after", TextNode(id="after", prompt="after gate"))
            .edge("auto", "gate")
            .edge("gate", "after")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.FAILED

        # auto succeeded
        assert orch.execution_dag.get_entry("auto").status == NodeExecutionStatus.SUCCESS

        # gate failed with escalation error
        gate_entry = orch.execution_dag.get_entry("gate")
        assert gate_entry.status == NodeExecutionStatus.FAILED
        assert "need_approval" in gate_entry.result.error_message

        # after was never reached
        after_entries = orch.execution_dag.get_all_entries_for_node("after")
        assert len(after_entries) == 0

    @pytest.mark.asyncio
    async def test_l3_escalation_state_primitives(self):
        """L3 ExecutionState escalation: set_escalated → get_escalation → clear."""
        state = ExecutionState()

        version = state.set_escalated(
            event_id="evt_123",
            question="Approve action?",
            options=["yes", "no"],
            node_id="gate",
            resume_data={"context": "important"},
        )

        # Escalation is recorded
        esc = state.get_escalation()
        assert esc is not None
        assert esc["question"] == "Approve action?"
        assert esc["options"] == ["yes", "no"]
        assert esc["node_id"] == "gate"
        assert esc["resume_data"] == {"context": "important"}

        # Version snapshot was created
        assert version >= 0

        # Clear escalation
        state.clear_escalation()
        assert state.get_escalation() is None

    @pytest.mark.asyncio
    async def test_l3_event_bus_escalation_roundtrip(self):
        """EventBus wait_for + emit: simulate escalation resolution roundtrip."""
        bus = EventBus()

        async def resolve_later():
            await asyncio.sleep(0.02)
            await bus.emit(
                EscalationResolved(event_id="evt_42", decision="approved")
            )

        task = asyncio.create_task(resolve_later())
        try:
            resolution = await bus.wait_for(
                EscalationResolved,
                filter=lambda e: e.event_id == "evt_42",
                timeout=2.0,
            )
            assert resolution.decision == "approved"
            assert resolution.event_id == "evt_42"
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_escalation_in_parallel_branch(self):
        """Escalation in one parallel branch fails the DAG."""
        esc_h = EscalatingHandler(escalate_prompts={"needs_human"})
        orch = _build_full_stack(text_handler=esc_h)

        dag = (
            DAGBuilder()
            .add_node("root", TextNode(id="root", prompt="start"))
            .add_node("auto", TextNode(id="auto", prompt="auto_task"))
            .add_node("human", TextNode(id="human", prompt="needs_human"))
            .edge("root", "auto")
            .edge("root", "human")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.FAILED

        # auto still ran (parallel batch completes all)
        assert esc_h.calls == ["start", "auto_task", "needs_human"] or set(esc_h.calls) == {
            "start", "auto_task", "needs_human"
        }


# ══════════════════════════════════════════════
# Smoke: everything together
# ══════════════════════════════════════════════


class TestSmokeEndToEnd:
    """Full end-to-end smoke tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_budget_and_state(self):
        """Full run: budget tracking + state snapshots + ExecutionDAG recording."""
        text_h = RealisticTextHandler()
        budget = BudgetTracker(token_budget=10000, call_budget=100)
        state = ExecutionState()
        orch = _build_full_stack(
            text_handler=text_h,
            budget_tracker=budget,
            state=state,
            validators=[BudgetValidator(), DependencyValidator()],
        )

        dag = (
            DAGBuilder("smoke")
            .add_node("a", TextNode(id="a", prompt="step one"))
            .add_node("b", TextNode(id="b", prompt="step two"))
            .add_node("c", TextNode(id="c", prompt="step three"))
            .add_node("d", TextNode(id="d", prompt="step four"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .build()
        )

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

        # All 4 nodes succeeded
        success = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert {e.node_id for e in success} == {"a", "b", "c", "d"}

        # Budget tracked
        assert budget.calls_made == 4
        assert budget.tokens_used > 0

        # State has snapshots for all 4 nodes
        assert state.version_count == 4

        # Level back to 0
        assert state.level == 0

    @pytest.mark.asyncio
    async def test_interrupt_during_full_pipeline(self):
        """Interrupt halts the full pipeline cleanly."""
        slow_h = SlowRealisticHandler(delay=0.1)
        state = ExecutionState()
        orch = _build_full_stack(text_handler=slow_h, state=state)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="slow"))
            .add_node("b", TextNode(id="b", prompt="slow"))
            .add_node("c", TextNode(id="c", prompt="slow"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )

        async def interrupt_midway():
            await asyncio.sleep(0.15)  # After a finishes, during b
            orch.interrupt()

        task = asyncio.create_task(interrupt_midway())
        try:
            with pytest.raises(InterruptError):
                await orch.run(dag)
            assert orch.status == DAGRunStatus.INTERRUPTED
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Level cleaned up even on interrupt
        assert state.level == 0

    @pytest.mark.asyncio
    async def test_rerun_after_failure(self):
        """Orchestrator can be used again after a failed run."""
        text_h = RealisticTextHandler()
        orch = _build_full_stack(text_handler=text_h)

        # First run: good
        dag1 = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="run1"))
            .build()
        )
        await orch.run(dag1)
        assert orch.status == DAGRunStatus.SUCCESS

        # Second run: new DAG
        dag2 = (
            DAGBuilder()
            .add_node("x", TextNode(id="x", prompt="run2"))
            .add_node("y", TextNode(id="y", prompt="run2b"))
            .edge("x", "y")
            .build()
        )
        await orch.run(dag2)
        assert orch.status == DAGRunStatus.SUCCESS

        # Second run's ExecutionDAG is fresh
        success = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert {e.node_id for e in success} == {"x", "y"}

    @pytest.mark.asyncio
    async def test_single_node_smoke(self):
        """Simplest possible DAG: single node, full stack."""
        text_h = RealisticTextHandler()
        budget = BudgetTracker(token_budget=1000)
        state = ExecutionState()
        orch = _build_full_stack(
            text_handler=text_h,
            budget_tracker=budget,
            state=state,
        )

        dag = DAGBuilder().add_node("only", TextNode(id="only", prompt="Hello")).build()
        await orch.run(dag)

        assert orch.status == DAGRunStatus.SUCCESS
        assert text_h.calls == ["Hello"]
        assert budget.calls_made == 1
        assert state.version_count == 1
        assert state.level == 0

        entry = orch.execution_dag.get_entry("only")
        assert entry.result.value == "Response to: Hello"
        assert entry.state_version == 0

    @pytest.mark.asyncio
    async def test_wide_fan_out_fan_in_smoke(self):
        """Wide fan-out (1 → 8) into fan-in (8 → 1)."""
        text_h = RealisticTextHandler()
        budget = BudgetTracker(token_budget=100000, call_budget=100)
        orch = _build_full_stack(text_handler=text_h, budget_tracker=budget)

        builder = DAGBuilder("fan-out-in")
        builder.add_node("root", TextNode(id="root", prompt="root"))
        builder.add_node("sink", TextNode(id="sink", prompt="sink"))
        for i in range(8):
            nid = f"branch_{i}"
            builder.add_node(nid, TextNode(id=nid, prompt=f"branch {i}"))
            builder.edge("root", nid)
            builder.edge(nid, "sink")
        dag = builder.build()

        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS
        assert budget.calls_made == 10  # root + 8 branches + sink

        success = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert len(success) == 10
