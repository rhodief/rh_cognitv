"""
Phase 3 tests — Adapter Registry + Validation Pipeline.

Tests adapter dispatch for all 4 execution-node kinds (mock HandlerRegistry
+ PolicyChain), NodeResult normalisation from each ExecutionResult[T],
ValidationPipeline ordering + short-circuit, and BudgetValidator with a
mock BudgetTracker.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rh_cognitv.execution_platform.events import (
    DataPayload,
    ExecutionEvent,
    FunctionPayload,
    TextPayload,
    ToolPayload,
)
from rh_cognitv.execution_platform.models import (
    EventKind,
    ExecutionResult,
    FunctionResultData,
    LLMResultData,
    ResultMetadata,
    TokenUsage,
    ToolResultData,
)
from rh_cognitv.execution_platform.protocols import (
    BudgetTrackerProtocol,
    HandlerRegistryProtocol,
)

from rh_cognitv.orchestrator.adapters import (
    AdapterRegistry,
    DataNodeAdapter,
    FunctionNodeAdapter,
    PlatformRef,
    TextNodeAdapter,
    ToolNodeAdapter,
)
from rh_cognitv.orchestrator.models import (
    NodeResult,
    OrchestratorConfig,
    ValidationContext,
    ValidationResult,
)
from rh_cognitv.orchestrator.nodes import (
    BaseNode,
    DataNode,
    FunctionNode,
    TextNode,
    ToolNode,
)
from rh_cognitv.orchestrator.protocols import NodeAdapterProtocol, NodeValidatorProtocol
from rh_cognitv.orchestrator.validation import (
    BudgetValidator,
    DependencyValidator,
    InputSchemaValidator,
    ValidationPipeline,
)


# ──────────────────────────────────────────────
# Helpers / Fixtures
# ──────────────────────────────────────────────


def _mock_registry(result: ExecutionResult[Any]) -> HandlerRegistryProtocol:
    """Create a mock HandlerRegistryProtocol that returns ``result``."""
    mock = MagicMock(spec=HandlerRegistryProtocol)
    mock.handle = AsyncMock(return_value=result)
    return mock


def _mock_budget_tracker(*, can_proceed: bool = True) -> BudgetTrackerProtocol:
    """Create a mock BudgetTrackerProtocol."""
    mock = MagicMock(spec=BudgetTrackerProtocol)
    mock.can_proceed.return_value = can_proceed
    return mock


def _llm_result(text: str = "hello", model: str = "gpt-4") -> ExecutionResult[LLMResultData]:
    return ExecutionResult(
        ok=True,
        value=LLMResultData(
            text=text,
            model=model,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total=15),
        ),
        metadata=ResultMetadata(),
    )


def _function_result(return_value: Any = 42) -> ExecutionResult[FunctionResultData]:
    return ExecutionResult(
        ok=True,
        value=FunctionResultData(return_value=return_value, duration_ms=1.5),
        metadata=ResultMetadata(),
    )


def _tool_result() -> ExecutionResult[ToolResultData]:
    return ExecutionResult(
        ok=True,
        value=ToolResultData(
            llm_result=LLMResultData(text="tool answer", model="gpt-4"),
            function_result=FunctionResultData(return_value={"key": "val"}, duration_ms=2.0),
        ),
        metadata=ResultMetadata(),
    )


def _platform(result: ExecutionResult[Any], **kw: Any) -> PlatformRef:
    """Build a PlatformRef backed by a mock registry."""
    return PlatformRef(registry=_mock_registry(result), **kw)


# ──────────────────────────────────────────────
# PlatformRef
# ──────────────────────────────────────────────


class TestPlatformRef:
    def test_default_config(self) -> None:
        reg = MagicMock(spec=HandlerRegistryProtocol)
        pf = PlatformRef(registry=reg)
        assert pf.config.default_timeout_seconds == 30.0
        assert pf.config.default_max_retries == 3
        assert pf.budget_tracker is None

    def test_build_policy_chain_defaults(self) -> None:
        reg = MagicMock(spec=HandlerRegistryProtocol)
        pf = PlatformRef(registry=reg)
        node = TextNode(prompt="hi")
        chain = pf.build_policy_chain(node)
        # No budget tracker → only TimeoutPolicy + RetryPolicy (2 policies)
        assert len(chain._policies) == 2

    def test_build_policy_chain_with_budget(self) -> None:
        reg = MagicMock(spec=HandlerRegistryProtocol)
        tracker = _mock_budget_tracker()
        pf = PlatformRef(registry=reg, budget_tracker=tracker)
        node = TextNode(prompt="hi")
        chain = pf.build_policy_chain(node)
        # BudgetPolicy + TimeoutPolicy + RetryPolicy
        assert len(chain._policies) == 3

    def test_build_policy_chain_node_overrides(self) -> None:
        """Per-node overrides should take precedence over config defaults."""
        reg = MagicMock(spec=HandlerRegistryProtocol)
        pf = PlatformRef(registry=reg)
        node = TextNode(prompt="hi", timeout_seconds=5.0, max_retries=1)
        chain = pf.build_policy_chain(node)
        # TimeoutPolicy + RetryPolicy
        timeout_pol = chain._policies[0]
        retry_pol = chain._policies[1]
        assert timeout_pol.seconds == 5.0
        assert retry_pol.max_attempts == 1

    def test_build_policy_chain_uses_config_defaults(self) -> None:
        cfg = OrchestratorConfig(
            default_timeout_seconds=60.0,
            default_max_retries=5,
            default_retry_base_delay=0.5,
        )
        reg = MagicMock(spec=HandlerRegistryProtocol)
        pf = PlatformRef(registry=reg, config=cfg)
        node = TextNode(prompt="hi")
        chain = pf.build_policy_chain(node)
        timeout_pol = chain._policies[0]
        retry_pol = chain._policies[1]
        assert timeout_pol.seconds == 60.0
        assert retry_pol.max_attempts == 5
        assert retry_pol.base_delay == 0.5


# ──────────────────────────────────────────────
# AdapterRegistry
# ──────────────────────────────────────────────


class TestAdapterRegistry:
    def test_register_and_get(self) -> None:
        reg = AdapterRegistry()
        adapter = TextNodeAdapter()
        reg.register("text", adapter)
        assert reg.get("text") is adapter

    def test_get_unknown_returns_none(self) -> None:
        reg = AdapterRegistry()
        assert reg.get("unknown") is None

    def test_registered_kinds(self) -> None:
        reg = AdapterRegistry.with_defaults()
        assert sorted(reg.registered_kinds) == ["data", "function", "text", "tool"]

    @pytest.mark.asyncio
    async def test_execute_unknown_kind_returns_failure(self) -> None:
        reg = AdapterRegistry()
        node = BaseNode(kind="unknown")
        platform = _platform(_llm_result())
        result = await reg.execute(node, None, None, platform)
        assert not result.ok
        assert "No adapter registered" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_execute_dispatches_to_correct_adapter(self) -> None:
        reg = AdapterRegistry.with_defaults()
        platform = _platform(_llm_result())
        node = TextNode(prompt="test")
        result = await reg.execute(node, None, None, platform)
        assert result.ok
        assert result.value == "hello"

    def test_with_defaults(self) -> None:
        reg = AdapterRegistry.with_defaults()
        assert isinstance(reg.get("text"), TextNodeAdapter)
        assert isinstance(reg.get("data"), DataNodeAdapter)
        assert isinstance(reg.get("function"), FunctionNodeAdapter)
        assert isinstance(reg.get("tool"), ToolNodeAdapter)

    def test_register_overrides_existing(self) -> None:
        reg = AdapterRegistry.with_defaults()
        custom = MagicMock(spec=NodeAdapterProtocol)
        reg.register("text", custom)
        assert reg.get("text") is custom


# ──────────────────────────────────────────────
# TextNodeAdapter
# ──────────────────────────────────────────────


class TestTextNodeAdapter:
    @pytest.mark.asyncio
    async def test_creates_correct_event_kind(self) -> None:
        platform = _platform(_llm_result())
        node = TextNode(prompt="summarize this", model="gpt-4", temperature=0.7)
        adapter = TextNodeAdapter()
        await adapter.execute(node, None, None, platform)
        # Verify the handler was called with an ExecutionEvent
        call_args = platform.registry.handle.call_args
        event = call_args[0][0]
        assert isinstance(event, ExecutionEvent)
        assert event.kind == EventKind.TEXT
        assert isinstance(event.payload, TextPayload)
        assert event.payload.prompt == "summarize this"
        assert event.payload.model == "gpt-4"
        assert event.payload.temperature == 0.7

    @pytest.mark.asyncio
    async def test_normalises_llm_result(self) -> None:
        platform = _platform(_llm_result("generated text"))
        node = TextNode(prompt="go")
        result = await TextNodeAdapter().execute(node, None, None, platform)
        assert result.ok
        assert result.value == "generated text"
        assert result.token_usage is not None
        assert result.token_usage.total == 15

    @pytest.mark.asyncio
    async def test_passes_data_and_configs(self) -> None:
        platform = _platform(_llm_result())
        node = TextNode(prompt="go")
        await TextNodeAdapter().execute(node, {"key": "val"}, {"cfg": 1}, platform)
        call_args = platform.registry.handle.call_args
        assert call_args[0][1] == {"key": "val"}
        assert call_args[0][2] == {"cfg": 1}

    @pytest.mark.asyncio
    async def test_system_prompt_passthrough(self) -> None:
        platform = _platform(_llm_result())
        node = TextNode(prompt="go", system_prompt="you are helpful")
        await TextNodeAdapter().execute(node, None, None, platform)
        event = platform.registry.handle.call_args[0][0]
        assert event.payload.system_prompt == "you are helpful"

    @pytest.mark.asyncio
    async def test_max_tokens_passthrough(self) -> None:
        platform = _platform(_llm_result())
        node = TextNode(prompt="go", max_tokens=100)
        await TextNodeAdapter().execute(node, None, None, platform)
        event = platform.registry.handle.call_args[0][0]
        assert event.payload.max_tokens == 100


# ──────────────────────────────────────────────
# DataNodeAdapter
# ──────────────────────────────────────────────


class TestDataNodeAdapter:
    @pytest.mark.asyncio
    async def test_creates_correct_event(self) -> None:
        platform = _platform(_llm_result())
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        node = DataNode(prompt="extract", output_schema=schema, model="gpt-4")
        await DataNodeAdapter().execute(node, None, None, platform)
        event = platform.registry.handle.call_args[0][0]
        assert event.kind == EventKind.DATA
        assert isinstance(event.payload, DataPayload)
        assert event.payload.prompt == "extract"
        assert event.payload.output_schema == schema
        assert event.payload.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_normalises_result(self) -> None:
        platform = _platform(_llm_result("structured output"))
        node = DataNode(prompt="extract")
        result = await DataNodeAdapter().execute(node, None, None, platform)
        assert result.ok
        assert result.value == "structured output"


# ──────────────────────────────────────────────
# FunctionNodeAdapter
# ──────────────────────────────────────────────


class TestFunctionNodeAdapter:
    @pytest.mark.asyncio
    async def test_creates_correct_event(self) -> None:
        platform = _platform(_function_result())
        node = FunctionNode(function_name="clean_data", args=[1, 2], kwargs={"x": "y"})
        await FunctionNodeAdapter().execute(node, None, None, platform)
        event = platform.registry.handle.call_args[0][0]
        assert event.kind == EventKind.FUNCTION
        assert isinstance(event.payload, FunctionPayload)
        assert event.payload.function_name == "clean_data"
        assert event.payload.args == [1, 2]
        assert event.payload.kwargs == {"x": "y"}

    @pytest.mark.asyncio
    async def test_normalises_function_result(self) -> None:
        platform = _platform(_function_result({"processed": True}))
        node = FunctionNode(function_name="proc")
        result = await FunctionNodeAdapter().execute(node, None, None, platform)
        assert result.ok
        assert result.value == {"processed": True}
        assert result.token_usage is None  # no tokens for function calls


# ──────────────────────────────────────────────
# ToolNodeAdapter
# ──────────────────────────────────────────────


class TestToolNodeAdapter:
    @pytest.mark.asyncio
    async def test_creates_correct_event(self) -> None:
        platform = _platform(_tool_result())
        tools = [{"name": "search", "description": "search the web"}]
        node = ToolNode(prompt="use tools", tools=tools, model="gpt-4")
        await ToolNodeAdapter().execute(node, None, None, platform)
        event = platform.registry.handle.call_args[0][0]
        assert event.kind == EventKind.TOOL
        assert isinstance(event.payload, ToolPayload)
        assert event.payload.prompt == "use tools"
        assert event.payload.tools == tools
        assert event.payload.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_normalises_tool_result(self) -> None:
        platform = _platform(_tool_result())
        node = ToolNode(prompt="go")
        result = await ToolNodeAdapter().execute(node, None, None, platform)
        assert result.ok
        assert isinstance(result.value, dict)
        assert result.value["llm_text"] == "tool answer"
        assert result.value["function_return"] == {"key": "val"}


# ──────────────────────────────────────────────
# Adapter — Error Handling
# ──────────────────────────────────────────────


class TestAdapterErrorHandling:
    @pytest.mark.asyncio
    async def test_failed_execution_result(self) -> None:
        """A failed ExecutionResult should produce a failed NodeResult."""
        failed = ExecutionResult(
            ok=False,
            error_message="handler blew up",
            error_category="TRANSIENT",
        )
        platform = _platform(failed)
        node = TextNode(prompt="go")
        result = await TextNodeAdapter().execute(node, None, None, platform)
        assert not result.ok
        assert result.error_message == "handler blew up"
        assert result.error_category == "TRANSIENT"

    @pytest.mark.asyncio
    async def test_handler_exception_propagates(self) -> None:
        """If the handler raises, the adapter should let it propagate."""
        reg = MagicMock(spec=HandlerRegistryProtocol)
        reg.handle = AsyncMock(side_effect=RuntimeError("boom"))
        platform = PlatformRef(registry=reg)
        node = TextNode(prompt="go")
        with pytest.raises(RuntimeError, match="boom"):
            await TextNodeAdapter().execute(node, None, None, platform)


# ──────────────────────────────────────────────
# NodeResult.from_execution_result (normalisation)
# ──────────────────────────────────────────────


class TestNodeResultNormalisation:
    def test_from_llm_result(self) -> None:
        er = _llm_result("hi", "gpt-4")
        nr = NodeResult.from_execution_result(er)
        assert nr.ok
        assert nr.value == "hi"
        assert nr.token_usage is not None
        assert nr.token_usage.total == 15

    def test_from_function_result(self) -> None:
        er = _function_result({"x": 1})
        nr = NodeResult.from_execution_result(er)
        assert nr.ok
        assert nr.value == {"x": 1}
        assert nr.token_usage is None

    def test_from_tool_result(self) -> None:
        er = _tool_result()
        nr = NodeResult.from_execution_result(er)
        assert nr.ok
        assert nr.value["llm_text"] == "tool answer"
        assert nr.value["function_return"] == {"key": "val"}

    def test_from_none_value(self) -> None:
        er = ExecutionResult(ok=True, value=None)
        nr = NodeResult.from_execution_result(er)
        assert nr.ok
        assert nr.value is None

    def test_from_unknown_value_type(self) -> None:
        er = ExecutionResult(ok=True, value="raw string")
        nr = NodeResult.from_execution_result(er)
        assert nr.value == "raw string"

    def test_from_failed_result(self) -> None:
        er = ExecutionResult(ok=False, error_message="oops", error_category="PERMANENT")
        nr = NodeResult.from_execution_result(er)
        assert not nr.ok
        assert nr.error_message == "oops"
        assert nr.error_category == "PERMANENT"


# ──────────────────────────────────────────────
# ValidationPipeline
# ──────────────────────────────────────────────


class TestValidationPipeline:
    @pytest.mark.asyncio
    async def test_empty_pipeline_passes(self) -> None:
        pipe = ValidationPipeline()
        result = await pipe.validate(TextNode(prompt="x"), None, ValidationContext())
        assert result.ok

    @pytest.mark.asyncio
    async def test_single_passing_validator(self) -> None:
        v = MagicMock(spec=NodeValidatorProtocol)
        v.validate = AsyncMock(return_value=ValidationResult.passed())
        pipe = ValidationPipeline([v])
        result = await pipe.validate(TextNode(prompt="x"), None, ValidationContext())
        assert result.ok

    @pytest.mark.asyncio
    async def test_single_failing_validator(self) -> None:
        v = MagicMock(spec=NodeValidatorProtocol)
        v.validate = AsyncMock(return_value=ValidationResult.failed("bad input"))
        pipe = ValidationPipeline([v])
        result = await pipe.validate(TextNode(prompt="x"), None, ValidationContext())
        assert not result.ok
        assert result.error_message == "bad input"

    @pytest.mark.asyncio
    async def test_short_circuit_on_first_failure(self) -> None:
        """Second validator must NOT be called when first fails."""
        v1 = MagicMock(spec=NodeValidatorProtocol)
        v1.validate = AsyncMock(return_value=ValidationResult.failed("v1 failed"))
        v2 = MagicMock(spec=NodeValidatorProtocol)
        v2.validate = AsyncMock(return_value=ValidationResult.passed())
        pipe = ValidationPipeline([v1, v2])
        result = await pipe.validate(TextNode(prompt="x"), None, ValidationContext())
        assert not result.ok
        assert result.error_message == "v1 failed"
        v2.validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_validators_run_when_passing(self) -> None:
        v1 = MagicMock(spec=NodeValidatorProtocol)
        v1.validate = AsyncMock(return_value=ValidationResult.passed())
        v2 = MagicMock(spec=NodeValidatorProtocol)
        v2.validate = AsyncMock(return_value=ValidationResult.passed())
        pipe = ValidationPipeline([v1, v2])
        result = await pipe.validate(TextNode(prompt="x"), None, ValidationContext())
        assert result.ok
        v1.validate.assert_called_once()
        v2.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_ordering_preserved(self) -> None:
        """Validators must run in insertion order."""
        call_order: list[str] = []

        async def make_validator(name: str, ok: bool) -> NodeValidatorProtocol:
            v = MagicMock(spec=NodeValidatorProtocol)

            async def side_effect(*_: Any, **__: Any) -> ValidationResult:
                call_order.append(name)
                return ValidationResult.passed() if ok else ValidationResult.failed(name)

            v.validate = AsyncMock(side_effect=side_effect)
            return v

        v1 = await make_validator("first", True)
        v2 = await make_validator("second", False)
        v3 = await make_validator("third", True)

        pipe = ValidationPipeline([v1, v2, v3])
        result = await pipe.validate(TextNode(prompt="x"), None, ValidationContext())
        assert not result.ok
        assert call_order == ["first", "second"]

    def test_add_validator(self) -> None:
        pipe = ValidationPipeline()
        v = MagicMock(spec=NodeValidatorProtocol)
        pipe.add(v)
        assert len(pipe.validators) == 1

    def test_validators_property(self) -> None:
        v1 = MagicMock(spec=NodeValidatorProtocol)
        v2 = MagicMock(spec=NodeValidatorProtocol)
        pipe = ValidationPipeline([v1, v2])
        assert len(pipe.validators) == 2


# ──────────────────────────────────────────────
# InputSchemaValidator
# ──────────────────────────────────────────────


class TestInputSchemaValidator:
    @pytest.mark.asyncio
    async def test_passes_when_no_input_key(self) -> None:
        v = InputSchemaValidator()
        node = TextNode(prompt="x")
        result = await v.validate(node, {"anything": True}, ValidationContext())
        assert result.ok

    @pytest.mark.asyncio
    async def test_passes_when_input_key_present(self) -> None:
        v = InputSchemaValidator()
        node = TextNode(prompt="x", ext={"input_key": "user_text"})
        result = await v.validate(node, {"user_text": "hello"}, ValidationContext())
        assert result.ok

    @pytest.mark.asyncio
    async def test_fails_when_input_key_missing(self) -> None:
        v = InputSchemaValidator()
        node = TextNode(prompt="x", ext={"input_key": "user_text"})
        result = await v.validate(node, {"other": "val"}, ValidationContext())
        assert not result.ok
        assert "user_text" in (result.error_message or "")
        assert result.validator_name == "InputSchemaValidator"

    @pytest.mark.asyncio
    async def test_passes_when_data_is_not_dict(self) -> None:
        v = InputSchemaValidator()
        node = TextNode(prompt="x", ext={"input_key": "user_text"})
        result = await v.validate(node, "just a string", ValidationContext())
        assert result.ok  # only applies to dict data


# ──────────────────────────────────────────────
# DependencyValidator
# ──────────────────────────────────────────────


class TestDependencyValidator:
    @pytest.mark.asyncio
    async def test_passes_when_no_predecessors(self) -> None:
        v = DependencyValidator()
        result = await v.validate(TextNode(prompt="x"), None, ValidationContext())
        assert result.ok

    @pytest.mark.asyncio
    async def test_passes_when_all_predecessors_completed(self) -> None:
        v = DependencyValidator()
        ctx = ValidationContext(
            completed_node_ids={"a", "b"},
            ext={"predecessors": ["a", "b"]},
        )
        result = await v.validate(TextNode(prompt="x"), None, ctx)
        assert result.ok

    @pytest.mark.asyncio
    async def test_fails_when_predecessor_missing(self) -> None:
        v = DependencyValidator()
        ctx = ValidationContext(
            completed_node_ids={"a"},
            ext={"predecessors": ["a", "b"]},
        )
        result = await v.validate(TextNode(prompt="x"), None, ctx)
        assert not result.ok
        assert "b" in (result.error_message or "")
        assert result.validator_name == "DependencyValidator"

    @pytest.mark.asyncio
    async def test_fails_lists_all_missing(self) -> None:
        v = DependencyValidator()
        ctx = ValidationContext(
            completed_node_ids=set(),
            ext={"predecessors": ["a", "b", "c"]},
        )
        result = await v.validate(TextNode(prompt="x"), None, ctx)
        assert not result.ok
        for nid in ["a", "b", "c"]:
            assert nid in (result.error_message or "")


# ──────────────────────────────────────────────
# BudgetValidator
# ──────────────────────────────────────────────


class TestBudgetValidator:
    @pytest.mark.asyncio
    async def test_passes_when_no_tracker(self) -> None:
        v = BudgetValidator()
        result = await v.validate(TextNode(prompt="x"), None, ValidationContext())
        assert result.ok

    @pytest.mark.asyncio
    async def test_passes_when_budget_available(self) -> None:
        v = BudgetValidator()
        tracker = _mock_budget_tracker(can_proceed=True)
        ctx = ValidationContext(ext={"budget_tracker": tracker})
        result = await v.validate(TextNode(prompt="x"), None, ctx)
        assert result.ok
        tracker.can_proceed.assert_called_once()

    @pytest.mark.asyncio
    async def test_fails_when_budget_exhausted(self) -> None:
        v = BudgetValidator()
        tracker = _mock_budget_tracker(can_proceed=False)
        ctx = ValidationContext(ext={"budget_tracker": tracker})
        result = await v.validate(TextNode(prompt="x"), None, ctx)
        assert not result.ok
        assert "Budget exhausted" in (result.error_message or "")
        assert result.validator_name == "BudgetValidator"


# ──────────────────────────────────────────────
# Integration: Adapter + Validation pipeline
# ──────────────────────────────────────────────


class TestAdapterValidationIntegration:
    @pytest.mark.asyncio
    async def test_validate_then_execute(self) -> None:
        """Simulate the orchestrator flow: validate → execute."""
        pipe = ValidationPipeline([
            InputSchemaValidator(),
            DependencyValidator(),
        ])
        node = TextNode(prompt="go", ext={"input_key": "text"})
        data = {"text": "hello world"}
        ctx = ValidationContext(completed_node_ids={"a"}, ext={"predecessors": ["a"]})

        # Validate
        vr = await pipe.validate(node, data, ctx)
        assert vr.ok

        # Execute
        registry = AdapterRegistry.with_defaults()
        platform = _platform(_llm_result("result text"))
        nr = await registry.execute(node, data, None, platform)
        assert nr.ok
        assert nr.value == "result text"

    @pytest.mark.asyncio
    async def test_validate_fails_skips_execute(self) -> None:
        """When validation fails, execution should not proceed."""
        pipe = ValidationPipeline([
            DependencyValidator(),
        ])
        node = TextNode(prompt="go")
        ctx = ValidationContext(
            completed_node_ids=set(),
            ext={"predecessors": ["missing_node"]},
        )

        vr = await pipe.validate(node, None, ctx)
        assert not vr.ok
        # No adapter call — validation failed

    @pytest.mark.asyncio
    async def test_full_chain_with_budget(self) -> None:
        """Validate budget → execute with budget policy in chain."""
        tracker = _mock_budget_tracker(can_proceed=True)

        # Validation
        pipe = ValidationPipeline([BudgetValidator()])
        ctx = ValidationContext(ext={"budget_tracker": tracker})
        node = TextNode(prompt="go")
        vr = await pipe.validate(node, None, ctx)
        assert vr.ok

        # Execution (budget tracked in PolicyChain too)
        platform = PlatformRef(
            registry=_mock_registry(_llm_result()),
            budget_tracker=tracker,
        )
        registry = AdapterRegistry.with_defaults()
        nr = await registry.execute(node, None, None, platform)
        assert nr.ok

    @pytest.mark.asyncio
    async def test_budget_exhausted_blocks_at_validation(self) -> None:
        tracker = _mock_budget_tracker(can_proceed=False)
        pipe = ValidationPipeline([BudgetValidator()])
        ctx = ValidationContext(ext={"budget_tracker": tracker})
        node = TextNode(prompt="go")
        vr = await pipe.validate(node, None, ctx)
        assert not vr.ok
        assert "Budget exhausted" in (vr.error_message or "")


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_adapter_registry_execute_all_four_kinds(self) -> None:
        """Quick smoke test: all 4 kind adapters produce ok NodeResults."""
        reg = AdapterRegistry.with_defaults()
        nodes_and_results = [
            (TextNode(prompt="hi"), _llm_result()),
            (DataNode(prompt="extract"), _llm_result()),
            (FunctionNode(function_name="fn"), _function_result()),
            (ToolNode(prompt="use tool"), _tool_result()),
        ]
        for node, er in nodes_and_results:
            platform = _platform(er)
            nr = await reg.execute(node, None, None, platform)
            assert nr.ok, f"Failed for node kind={node.kind}"

    @pytest.mark.asyncio
    async def test_custom_adapter_in_registry(self) -> None:
        """User can register a custom adapter for a new node kind."""

        class CustomAdapter(NodeAdapterProtocol):
            async def execute(self, node: BaseNode, data: Any, configs: Any, platform: Any) -> NodeResult:
                return NodeResult.success(value="custom")

        reg = AdapterRegistry()
        reg.register("custom", CustomAdapter())
        node = BaseNode(kind="custom")
        platform = _platform(_llm_result())  # won't be used
        nr = await reg.execute(node, None, None, platform)
        assert nr.ok
        assert nr.value == "custom"

    def test_validation_pipeline_is_validation_pipeline_protocol(self) -> None:
        from rh_cognitv.orchestrator.protocols import ValidationPipelineProtocol

        assert issubclass(ValidationPipeline, ValidationPipelineProtocol)

    def test_adapters_are_node_adapter_protocol(self) -> None:
        for cls in (TextNodeAdapter, DataNodeAdapter, FunctionNodeAdapter, ToolNodeAdapter):
            assert issubclass(cls, NodeAdapterProtocol)

    def test_validators_are_node_validator_protocol(self) -> None:
        for cls in (InputSchemaValidator, DependencyValidator, BudgetValidator):
            assert issubclass(cls, NodeValidatorProtocol)
