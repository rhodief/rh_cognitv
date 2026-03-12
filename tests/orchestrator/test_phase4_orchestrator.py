"""
Phase 4 tests — Core Orchestrator (Linear Traversal).

Integration tests using real L3 components (ExecutionState, HandlerRegistry)
with in-memory handlers. Tests cover:
  - Linear DAG (A→B→C)
  - Branching DAG (A→B, A→C run sequentially)
  - Failing node (ExecutionDAG records failure, run stops)
  - Interrupt mid-execution
  - Validation failure at node level
  - State snapshot per-node
  - FlowNode skip (deferred to Phase 5)
  - DAGOrchestrator status lifecycle
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rh_cognitv.execution_platform.errors import InterruptError
from rh_cognitv.execution_platform.events import ExecutionEvent, TextPayload
from rh_cognitv.execution_platform.handlers import HandlerRegistry
from rh_cognitv.execution_platform.models import (
    EventKind,
    ExecutionResult,
    FunctionResultData,
    LLMResultData,
    ResultMetadata,
    TokenUsage,
)
from rh_cognitv.execution_platform.protocols import (
    EventHandlerProtocol,
    HandlerRegistryProtocol,
)
from rh_cognitv.execution_platform.state import ExecutionState

from rh_cognitv.orchestrator.adapters import AdapterRegistry, PlatformRef
from rh_cognitv.orchestrator.dag_orchestrator import DAGOrchestrator
from rh_cognitv.orchestrator.execution_dag import ExecutionDAG
from rh_cognitv.orchestrator.flow_nodes import ForEachNode
from rh_cognitv.orchestrator.protocols import OrchestratorProtocol as _OrchestratorProtocol
from rh_cognitv.orchestrator.models import (
    DAGRunStatus,
    NodeExecutionStatus,
    NodeResult,
    OrchestratorConfig,
    ValidationContext,
    ValidationResult,
)
from rh_cognitv.orchestrator.nodes import DataNode, FunctionNode, TextNode
from rh_cognitv.orchestrator.plan_dag import DAGBuilder
from rh_cognitv.orchestrator.protocols import NodeValidatorProtocol
from rh_cognitv.orchestrator.validation import (
    DependencyValidator,
    InputSchemaValidator,
    ValidationPipeline,
)


# ──────────────────────────────────────────────
# Fixtures / Helpers
# ──────────────────────────────────────────────


class StubTextHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler that returns a predictable result based on prompt."""

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        prompt = event.payload.prompt if hasattr(event.payload, "prompt") else ""
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=f"response to: {prompt}",
                model="stub-model",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total=15),
            ),
            metadata=ResultMetadata(),
        )


class StubFunctionHandler(EventHandlerProtocol[FunctionResultData]):
    """Function handler that returns a predictable result."""

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[FunctionResultData]:
        fn_name = event.payload.function_name if hasattr(event.payload, "function_name") else ""
        return ExecutionResult(
            ok=True,
            value=FunctionResultData(return_value=f"result of {fn_name}", duration_ms=1.0),
            metadata=ResultMetadata(),
        )


class FailingHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler that always returns a failed result."""

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        return ExecutionResult(
            ok=False,
            error_message="handler failed deliberately",
            error_category="PERMANENT",
            metadata=ResultMetadata(),
        )


class ExplodingHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler that raises an exception."""

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        raise RuntimeError("handler exploded")


class SlowHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler with a small delay — for interrupt testing."""

    def __init__(self, delay: float = 0.05) -> None:
        self._delay = delay
        self.call_count = 0

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        self.call_count += 1
        await asyncio.sleep(self._delay)
        return ExecutionResult(
            ok=True,
            value=LLMResultData(text="slow response", model="stub"),
            metadata=ResultMetadata(),
        )


def _build_registry(
    text_handler: EventHandlerProtocol | None = None,
    function_handler: EventHandlerProtocol | None = None,
) -> HandlerRegistry:
    """Build an L3 HandlerRegistry with stub handlers."""
    reg = HandlerRegistry()
    reg.register(EventKind.TEXT, text_handler or StubTextHandler())
    reg.register(EventKind.FUNCTION, function_handler or StubFunctionHandler())
    reg.register(EventKind.DATA, text_handler or StubTextHandler())  # reuse text handler
    return reg


def _build_orchestrator(
    handler_registry: HandlerRegistry | None = None,
    config: OrchestratorConfig | None = None,
    validation: ValidationPipeline | None = None,
) -> DAGOrchestrator:
    """Build a DAGOrchestrator with real L3 components."""
    cfg = config or OrchestratorConfig(
        default_timeout_seconds=10.0,
        default_max_retries=1,
        default_retry_base_delay=0.01,
    )
    h_reg = handler_registry or _build_registry()
    platform = PlatformRef(registry=h_reg, config=cfg)
    state = ExecutionState()
    adapter_registry = AdapterRegistry.with_defaults()

    return DAGOrchestrator(
        adapter_registry=adapter_registry,
        platform=platform,
        state=state,
        validation=validation,
        config=cfg,
    )


# ──────────────────────────────────────────────
# Linear DAG: A → B → C
# ──────────────────────────────────────────────


class TestLinearDAG:
    @pytest.mark.asyncio
    async def test_three_node_linear(self) -> None:
        """A→B→C should execute in order and produce 3 SUCCESS entries."""
        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="step A"))
            .add_node("b", TextNode(id="b", prompt="step B"))
            .add_node("c", TextNode(id="c", prompt="step C"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )
        orch = _build_orchestrator()
        exec_dag = await orch.run(dag)

        # 3 RUNNING + 3 SUCCESS entries = 6 total
        all_entries = exec_dag.entries()
        success = exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert len(success) == 3
        assert orch.status == DAGRunStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execution_order(self) -> None:
        """SUCCESS entries should appear in topological order: a, b, c."""
        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .add_node("c", TextNode(id="c", prompt="C"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )
        orch = _build_orchestrator()
        exec_dag = await orch.run(dag)

        success = exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        node_ids = [e.node_id for e in success]
        assert node_ids == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_node_results_populated(self) -> None:
        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="hello"))
            .build()
        )
        orch = _build_orchestrator()
        exec_dag = await orch.run(dag)

        entry = exec_dag.get_entry("a")
        assert entry is not None
        assert entry.status == NodeExecutionStatus.SUCCESS
        assert entry.result is not None
        assert entry.result.ok
        assert "hello" in entry.result.value  # "response to: hello"

    @pytest.mark.asyncio
    async def test_state_version_recorded(self) -> None:
        """Each successful entry should have a state_version."""
        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .edge("a", "b")
            .build()
        )
        orch = _build_orchestrator()
        exec_dag = await orch.run(dag)

        success = exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        for entry in success:
            assert entry.state_version is not None
        # Versions should be sequential
        versions = [e.state_version for e in success]
        assert versions == sorted(versions)
        assert len(set(versions)) == len(versions)  # unique

    @pytest.mark.asyncio
    async def test_single_node_dag(self) -> None:
        dag = DAGBuilder().add_node("x", TextNode(id="x", prompt="only")).build()
        orch = _build_orchestrator()
        exec_dag = await orch.run(dag)
        assert len(exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)) == 1
        assert orch.status == DAGRunStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_mixed_node_kinds(self) -> None:
        """Linear DAG with text + function nodes."""
        dag = (
            DAGBuilder()
            .add_node("t", TextNode(id="t", prompt="describe"))
            .add_node("f", FunctionNode(id="f", function_name="process"))
            .edge("t", "f")
            .build()
        )
        orch = _build_orchestrator()
        exec_dag = await orch.run(dag)

        success = exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert len(success) == 2
        # Function node result
        f_entry = exec_dag.get_entry("f")
        assert f_entry is not None
        assert f_entry.result is not None
        assert f_entry.result.ok
        assert "process" in str(f_entry.result.value)


# ──────────────────────────────────────────────
# Branching DAG: A → B, A → C (sequential in Stage 1)
# ──────────────────────────────────────────────


class TestBranchingDAG:
    @pytest.mark.asyncio
    async def test_diamond(self) -> None:
        """
        Diamond: A→B, A→C, B→D, C→D
        All 4 nodes should succeed.
        """
        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .add_node("c", TextNode(id="c", prompt="C"))
            .add_node("d", TextNode(id="d", prompt="D"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .build()
        )
        orch = _build_orchestrator()
        exec_dag = await orch.run(dag)

        success = exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert len(success) == 4
        node_ids = {e.node_id for e in success}
        assert node_ids == {"a", "b", "c", "d"}
        assert orch.status == DAGRunStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_fan_out(self) -> None:
        """
        Fan-out: A→B, A→C, A→D
        All 4 should succeed (executed sequentially in Stage 1).
        """
        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .add_node("c", TextNode(id="c", prompt="C"))
            .add_node("d", TextNode(id="d", prompt="D"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("a", "d")
            .build()
        )
        orch = _build_orchestrator()
        exec_dag = await orch.run(dag)

        success = exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert len(success) == 4

    @pytest.mark.asyncio
    async def test_a_executes_before_b_and_c(self) -> None:
        """In A→B, A→C: A must be the first SUCCESS entry."""
        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .add_node("c", TextNode(id="c", prompt="C"))
            .edge("a", "b")
            .edge("a", "c")
            .build()
        )
        orch = _build_orchestrator()
        exec_dag = await orch.run(dag)

        success = exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert success[0].node_id == "a"


# ──────────────────────────────────────────────
# Failing Node
# ──────────────────────────────────────────────


class TestFailingNode:
    @pytest.mark.asyncio
    async def test_failure_stops_execution(self) -> None:
        """When a node fails, the DAG stops and records the failure."""
        h_reg = _build_registry(text_handler=FailingHandler())
        orch = _build_orchestrator(handler_registry=h_reg)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="fail me"))
            .add_node("b", TextNode(id="b", prompt="never reached"))
            .edge("a", "b")
            .build()
        )
        exec_dag = await orch.run(dag)

        assert orch.status == DAGRunStatus.FAILED
        failed = exec_dag.get_by_status(NodeExecutionStatus.FAILED)
        assert len(failed) == 1
        assert failed[0].node_id == "a"
        # B should NOT have any SUCCESS entries
        success = exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert len(success) == 0

    @pytest.mark.asyncio
    async def test_failure_records_error_details(self) -> None:
        h_reg = _build_registry(text_handler=FailingHandler())
        orch = _build_orchestrator(handler_registry=h_reg)

        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="fail")).build()
        exec_dag = await orch.run(dag)

        entry = exec_dag.get_entry("a")
        assert entry is not None
        assert entry.status == NodeExecutionStatus.FAILED
        assert entry.result is not None
        assert not entry.result.ok
        assert "handler failed deliberately" in entry.result.error_message

    @pytest.mark.asyncio
    async def test_exception_in_handler_produces_failure(self) -> None:
        """If the handler raises an exception, it should be caught and recorded."""
        h_reg = _build_registry(text_handler=ExplodingHandler())
        orch = _build_orchestrator(handler_registry=h_reg)

        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="boom")).build()
        exec_dag = await orch.run(dag)

        assert orch.status == DAGRunStatus.FAILED
        entry = exec_dag.get_entry("a")
        assert entry is not None
        assert entry.status == NodeExecutionStatus.FAILED
        assert "exploded" in (entry.result.error_message or "")

    @pytest.mark.asyncio
    async def test_failure_mid_chain(self) -> None:
        """A→B→C where B fails: A succeeds, B fails, C not executed."""
        # A uses stub, B uses failing
        call_count = {"text": 0}

        class CountingHandler(EventHandlerProtocol[LLMResultData]):
            async def __call__(self, event, data, configs):
                call_count["text"] += 1
                if call_count["text"] == 2:  # second call = B
                    return ExecutionResult(
                        ok=False, error_message="B failed", metadata=ResultMetadata()
                    )
                return ExecutionResult(
                    ok=True,
                    value=LLMResultData(text="ok", model="stub"),
                    metadata=ResultMetadata(),
                )

        h_reg = _build_registry(text_handler=CountingHandler())
        orch = _build_orchestrator(handler_registry=h_reg)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .add_node("c", TextNode(id="c", prompt="C"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )
        exec_dag = await orch.run(dag)

        assert orch.status == DAGRunStatus.FAILED
        assert exec_dag.get_entry("a").status == NodeExecutionStatus.SUCCESS
        assert exec_dag.get_entry("b").status == NodeExecutionStatus.FAILED
        # C should have no completed entry (only a RUNNING if it was started, but it shouldn't be)
        c_entry = exec_dag.get_entry("c")
        if c_entry is not None:
            assert c_entry.status != NodeExecutionStatus.SUCCESS


# ──────────────────────────────────────────────
# Interrupt
# ──────────────────────────────────────────────


class TestInterrupt:
    @pytest.mark.asyncio
    async def test_interrupt_raises(self) -> None:
        """Calling interrupt() should cause run() to raise InterruptError."""
        slow = SlowHandler(delay=0.1)
        h_reg = _build_registry(text_handler=slow)
        orch = _build_orchestrator(handler_registry=h_reg)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .add_node("c", TextNode(id="c", prompt="C"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )

        async def interrupt_after_first():
            # Wait for first node to start, then interrupt
            await asyncio.sleep(0.15)
            orch.interrupt()

        with pytest.raises(InterruptError):
            await asyncio.gather(
                orch.run(dag),
                interrupt_after_first(),
            )

        assert orch.status == DAGRunStatus.INTERRUPTED

    @pytest.mark.asyncio
    async def test_interrupt_before_run(self) -> None:
        """If interrupted before run starts, first node should trigger interrupt."""
        orch = _build_orchestrator()
        orch.interrupt()

        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="A")).build()
        with pytest.raises(InterruptError):
            await orch.run(dag)

        assert orch.status == DAGRunStatus.INTERRUPTED

    @pytest.mark.asyncio
    async def test_interrupt_preserves_partial_results(self) -> None:
        """Nodes completed before interrupt should be in the ExecutionDAG."""
        slow = SlowHandler(delay=0.1)
        h_reg = _build_registry(text_handler=slow)
        orch = _build_orchestrator(handler_registry=h_reg)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .add_node("c", TextNode(id="c", prompt="C"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )

        async def interrupt_after_first():
            await asyncio.sleep(0.15)
            orch.interrupt()

        with pytest.raises(InterruptError):
            await asyncio.gather(
                orch.run(dag),
                interrupt_after_first(),
            )

        # At least the first node should have completed
        success = orch.execution_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert len(success) >= 1
        assert success[0].node_id == "a"


# ──────────────────────────────────────────────
# State snapshots
# ──────────────────────────────────────────────


class TestStateSnapshots:
    @pytest.mark.asyncio
    async def test_state_level_managed(self) -> None:
        """Run should add_level at start and remove_level at end."""
        state = ExecutionState()
        assert state.level == 0

        h_reg = _build_registry()
        cfg = OrchestratorConfig(default_timeout_seconds=10.0, default_max_retries=1)
        platform = PlatformRef(registry=h_reg, config=cfg)
        orch = DAGOrchestrator(
            adapter_registry=AdapterRegistry.with_defaults(),
            platform=platform,
            state=state,
        )

        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="A")).build()
        await orch.run(dag)

        # Level should be back to 0 after run
        assert state.level == 0

    @pytest.mark.asyncio
    async def test_state_level_restored_on_failure(self) -> None:
        """Even on failure, remove_level should be called (finally block)."""
        state = ExecutionState()
        h_reg = _build_registry(text_handler=FailingHandler())
        cfg = OrchestratorConfig(default_timeout_seconds=10.0, default_max_retries=1)
        platform = PlatformRef(registry=h_reg, config=cfg)
        orch = DAGOrchestrator(
            adapter_registry=AdapterRegistry.with_defaults(),
            platform=platform,
            state=state,
        )

        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="fail")).build()
        await orch.run(dag)

        assert state.level == 0

    @pytest.mark.asyncio
    async def test_state_level_restored_on_interrupt(self) -> None:
        state = ExecutionState()
        h_reg = _build_registry()
        cfg = OrchestratorConfig(default_timeout_seconds=10.0, default_max_retries=1)
        platform = PlatformRef(registry=h_reg, config=cfg)
        orch = DAGOrchestrator(
            adapter_registry=AdapterRegistry.with_defaults(),
            platform=platform,
            state=state,
        )
        orch.interrupt()

        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="A")).build()
        with pytest.raises(InterruptError):
            await orch.run(dag)

        assert state.level == 0

    @pytest.mark.asyncio
    async def test_snapshots_created_per_node(self) -> None:
        """Each executed node should produce one snapshot."""
        state = ExecutionState()
        h_reg = _build_registry()
        cfg = OrchestratorConfig(default_timeout_seconds=10.0, default_max_retries=1)
        platform = PlatformRef(registry=h_reg, config=cfg)
        orch = DAGOrchestrator(
            adapter_registry=AdapterRegistry.with_defaults(),
            platform=platform,
            state=state,
        )

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

        assert state.version_count == 3


# ──────────────────────────────────────────────
# Validation integration
# ──────────────────────────────────────────────


class TestValidationIntegration:
    @pytest.mark.asyncio
    async def test_validation_failure_records_failed_entry(self) -> None:
        """If validation fails, node should be recorded as FAILED."""

        class AlwaysFailValidator(NodeValidatorProtocol):
            async def validate(self, node, data, context):
                return ValidationResult.failed(
                    "always fails", validator_name="AlwaysFail"
                )

        pipe = ValidationPipeline([AlwaysFailValidator()])
        orch = _build_orchestrator(validation=pipe)

        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="A")).build()
        exec_dag = await orch.run(dag)

        assert orch.status == DAGRunStatus.FAILED
        entry = exec_dag.get_entry("a")
        assert entry is not None
        assert entry.status == NodeExecutionStatus.FAILED
        assert "always fails" in (entry.result.error_message or "")

    @pytest.mark.asyncio
    async def test_dependency_validator_integration(self) -> None:
        """DependencyValidator should pass for nodes whose predecessors completed."""
        pipe = ValidationPipeline([DependencyValidator()])
        orch = _build_orchestrator(validation=pipe)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .edge("a", "b")
            .build()
        )
        exec_dag = await orch.run(dag)

        # Both should succeed — a has no predecessors, b's predecessor (a) completed
        success = exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert len(success) == 2
        assert orch.status == DAGRunStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_input_schema_validator_integration(self) -> None:
        """InputSchemaValidator should fail if required input_key missing."""
        pipe = ValidationPipeline([InputSchemaValidator()])
        orch = _build_orchestrator(validation=pipe)

        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A", ext={"input_key": "user_text"}))
            .build()
        )
        # data is a dict missing 'user_text'
        exec_dag = await orch.run(dag, data={"other": "val"})

        assert orch.status == DAGRunStatus.FAILED
        entry = exec_dag.get_entry("a")
        assert entry.status == NodeExecutionStatus.FAILED
        assert "user_text" in (entry.result.error_message or "")


# ──────────────────────────────────────────────
# FlowNode handling (stub — skip in Phase 4)
# ──────────────────────────────────────────────


class TestFlowNodeSkip:
    @pytest.mark.asyncio
    async def test_flow_node_handled(self) -> None:
        """FlowNodes are handled via FlowHandlerRegistry (Phase 5)."""
        dag = (
            DAGBuilder()
            .add_node("a", TextNode(id="a", prompt="A"))
            .add_node("fe", ForEachNode(id="fe", inner_node_id="a"))
            .add_node("b", TextNode(id="b", prompt="B"))
            .edge("a", "fe")
            .edge("fe", "b")
            .build()
        )
        orch = _build_orchestrator()
        exec_dag = await orch.run(dag)

        # fe should be SUCCESS (handled by FlowHandlerRegistry)
        fe_entries = exec_dag.get_all_entries_for_node("fe")
        assert any(e.status == NodeExecutionStatus.SUCCESS for e in fe_entries)
        # a and b should succeed
        success = exec_dag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert {"a", "b"}.issubset({e.node_id for e in success})


# ──────────────────────────────────────────────
# DAGOrchestrator status lifecycle
# ──────────────────────────────────────────────


class TestStatusLifecycle:
    def test_initial_status(self) -> None:
        orch = _build_orchestrator()
        assert orch.status == DAGRunStatus.PENDING

    @pytest.mark.asyncio
    async def test_success_status(self) -> None:
        orch = _build_orchestrator()
        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="A")).build()
        await orch.run(dag)
        assert orch.status == DAGRunStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_failed_status(self) -> None:
        h_reg = _build_registry(text_handler=FailingHandler())
        orch = _build_orchestrator(handler_registry=h_reg)
        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="fail")).build()
        await orch.run(dag)
        assert orch.status == DAGRunStatus.FAILED

    @pytest.mark.asyncio
    async def test_interrupted_status(self) -> None:
        orch = _build_orchestrator()
        orch.interrupt()
        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="A")).build()
        with pytest.raises(InterruptError):
            await orch.run(dag)
        assert orch.status == DAGRunStatus.INTERRUPTED


# ──────────────────────────────────────────────
# Rerun
# ──────────────────────────────────────────────


class TestRerun:
    @pytest.mark.asyncio
    async def test_rerun_resets_execution_dag(self) -> None:
        """Running multiple times should produce fresh ExecutionDAGs."""
        orch = _build_orchestrator()
        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="A")).build()

        exec_dag1 = await orch.run(dag)
        assert len(exec_dag1.get_by_status(NodeExecutionStatus.SUCCESS)) == 1

        exec_dag2 = await orch.run(dag)
        assert len(exec_dag2.get_by_status(NodeExecutionStatus.SUCCESS)) == 1

        # They should be different objects
        assert exec_dag1 is not exec_dag2


# ──────────────────────────────────────────────
# RUNNING entries (record_start)
# ──────────────────────────────────────────────


class TestRunningEntries:
    @pytest.mark.asyncio
    async def test_running_entry_created_before_success(self) -> None:
        """Each node should have a RUNNING entry before the final entry."""
        orch = _build_orchestrator()
        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="A")).build()
        exec_dag = await orch.run(dag)

        all_for_a = exec_dag.get_all_entries_for_node("a")
        assert len(all_for_a) >= 2
        # First should be RUNNING, last should be SUCCESS
        assert all_for_a[0].status == NodeExecutionStatus.RUNNING
        assert all_for_a[-1].status == NodeExecutionStatus.SUCCESS


# ──────────────────────────────────────────────
# Data passthrough
# ──────────────────────────────────────────────


class TestDataPassthrough:
    @pytest.mark.asyncio
    async def test_data_passed_to_handler(self) -> None:
        """Data argument should reach the handler."""
        received_data = {}

        class CapturingHandler(EventHandlerProtocol[LLMResultData]):
            async def __call__(self, event, data, configs):
                received_data["value"] = data
                return ExecutionResult(
                    ok=True,
                    value=LLMResultData(text="ok", model="stub"),
                    metadata=ResultMetadata(),
                )

        h_reg = _build_registry(text_handler=CapturingHandler())
        orch = _build_orchestrator(handler_registry=h_reg)
        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="A")).build()
        await orch.run(dag, data={"context": "important"})

        assert received_data["value"] == {"context": "important"}

    @pytest.mark.asyncio
    async def test_none_data_default(self) -> None:
        """When no data is provided, None should be passed to the handler."""
        received_data = {"value": "NOT_SET"}

        class CapturingHandler(EventHandlerProtocol[LLMResultData]):
            async def __call__(self, event, data, configs):
                received_data["value"] = data
                return ExecutionResult(
                    ok=True,
                    value=LLMResultData(text="ok", model="stub"),
                    metadata=ResultMetadata(),
                )

        h_reg = _build_registry(text_handler=CapturingHandler())
        orch = _build_orchestrator(handler_registry=h_reg)
        dag = DAGBuilder().add_node("a", TextNode(id="a", prompt="A")).build()
        await orch.run(dag)

        assert received_data["value"] is None


# ──────────────────────────────────────────────
# OrchestratorProtocol conformance
# ──────────────────────────────────────────────


class TestProtocolConformance:
    def test_is_orchestrator_protocol(self) -> None:
        assert issubclass(DAGOrchestrator, _OrchestratorProtocol)
