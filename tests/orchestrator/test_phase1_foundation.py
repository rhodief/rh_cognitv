"""Tests for Phase 1 — Protocols, Models, Node Types."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from rh_cognitv.execution_platform.models import (
    ExecutionResult,
    FunctionResultData,
    LLMResultData,
    ResultMetadata,
    TokenUsage,
    ToolResultData,
)
from rh_cognitv.execution_platform.types import generate_ulid, now_timestamp

from rh_cognitv.orchestrator.models import (
    DAGRunStatus,
    ExecutionDAGEntry,
    FlowResult,
    NodeExecutionStatus,
    NodeResult,
    OrchestratorConfig,
    ValidationContext,
    ValidationResult,
)
from rh_cognitv.orchestrator.nodes import (
    BaseNode,
    DataNode,
    ExecutionNode,
    FlowNode,
    FunctionNode,
    TextNode,
    ToolNode,
)
from rh_cognitv.orchestrator.flow_nodes import (
    CompositeNode,
    FilterNode,
    ForEachNode,
    GetNode,
    IfNotOkNode,
    Node,
    SwitchNode,
)
from rh_cognitv.orchestrator.protocols import (
    DAGProtocol,
    FlowHandlerProtocol,
    NodeAdapterProtocol,
    NodeProtocol,
    NodeValidatorProtocol,
    OrchestratorProtocol,
    ValidationPipelineProtocol,
)


# ═══════════════════════════════════════════════
#  Protocol smoke tests — verify ABCs exist
# ═══════════════════════════════════════════════


class TestProtocolsExist:
    """Verify that all protocol ABCs can be referenced and are abstract."""

    def test_orchestrator_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            OrchestratorProtocol()  # type: ignore[abstract]

    def test_dag_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            DAGProtocol()  # type: ignore[abstract]

    def test_node_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            NodeProtocol()  # type: ignore[abstract]

    def test_node_adapter_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            NodeAdapterProtocol()  # type: ignore[abstract]

    def test_flow_handler_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            FlowHandlerProtocol()  # type: ignore[abstract]

    def test_node_validator_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            NodeValidatorProtocol()  # type: ignore[abstract]

    def test_validation_pipeline_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            ValidationPipelineProtocol()  # type: ignore[abstract]


# ═══════════════════════════════════════════════
#  Enum tests
# ═══════════════════════════════════════════════


class TestEnums:
    def test_node_execution_status_values(self):
        assert NodeExecutionStatus.PENDING == "pending"
        assert NodeExecutionStatus.RUNNING == "running"
        assert NodeExecutionStatus.SUCCESS == "success"
        assert NodeExecutionStatus.FAILED == "failed"
        assert NodeExecutionStatus.ROLLED_BACK == "rolled_back"
        assert NodeExecutionStatus.SKIPPED == "skipped"
        assert NodeExecutionStatus.WAITING == "waiting"

    def test_dag_run_status_values(self):
        assert DAGRunStatus.PENDING == "pending"
        assert DAGRunStatus.RUNNING == "running"
        assert DAGRunStatus.SUCCESS == "success"
        assert DAGRunStatus.FAILED == "failed"
        assert DAGRunStatus.INTERRUPTED == "interrupted"


# ═══════════════════════════════════════════════
#  NodeResult tests
# ═══════════════════════════════════════════════


class TestNodeResult:
    def test_success_factory(self):
        r = NodeResult.success("hello")
        assert r.ok is True
        assert r.value == "hello"
        assert r.error_message is None

    def test_failure_factory(self):
        r = NodeResult.failure("boom", error_category="transient")
        assert r.ok is False
        assert r.error_message == "boom"
        assert r.error_category == "transient"

    def test_from_execution_result_llm(self):
        llm_data = LLMResultData(
            text="generated text",
            thinking=None,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total=30),
            model="gpt-4",
            finish_reason="stop",
        )
        meta = ResultMetadata(duration_ms=100.0, attempt=1)
        er = ExecutionResult[LLMResultData](ok=True, value=llm_data, metadata=meta)

        nr = NodeResult.from_execution_result(er)
        assert nr.ok is True
        assert nr.value == "generated text"
        assert nr.token_usage is not None
        assert nr.token_usage.total == 30
        assert nr.metadata is not None
        assert nr.metadata.duration_ms == 100.0

    def test_from_execution_result_function(self):
        func_data = FunctionResultData(return_value={"key": "val"}, duration_ms=50.0)
        meta = ResultMetadata(duration_ms=50.0, attempt=1)
        er = ExecutionResult[FunctionResultData](ok=True, value=func_data, metadata=meta)

        nr = NodeResult.from_execution_result(er)
        assert nr.ok is True
        assert nr.value == {"key": "val"}
        assert nr.token_usage is None

    def test_from_execution_result_tool(self):
        llm = LLMResultData(
            text="tool output",
            thinking=None,
            token_usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total=15),
            model="gpt-4",
            finish_reason="tool_call",
        )
        func = FunctionResultData(return_value=42, duration_ms=20.0)
        tool_data = ToolResultData(llm_result=llm, function_result=func)
        meta = ResultMetadata(duration_ms=70.0, attempt=1)
        er = ExecutionResult[ToolResultData](ok=True, value=tool_data, metadata=meta)

        nr = NodeResult.from_execution_result(er)
        assert nr.ok is True
        assert nr.value == {"llm_text": "tool output", "function_return": 42}
        assert nr.token_usage is not None
        assert nr.token_usage.total == 15

    def test_from_execution_result_error(self):
        meta = ResultMetadata(duration_ms=10.0, attempt=2)
        er = ExecutionResult(
            ok=False, value=None, error_message="timeout", error_category="transient", metadata=meta
        )
        nr = NodeResult.from_execution_result(er)
        assert nr.ok is False
        assert nr.error_message == "timeout"
        assert nr.error_category == "transient"

    def test_serialization_roundtrip(self):
        r = NodeResult.success("data", token_usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total=3))
        data = r.model_dump()
        r2 = NodeResult.model_validate(data)
        assert r2 == r


# ═══════════════════════════════════════════════
#  ValidationResult tests
# ═══════════════════════════════════════════════


class TestValidationResult:
    def test_passed(self):
        vr = ValidationResult.passed()
        assert vr.ok is True
        assert vr.error_message is None

    def test_failed(self):
        vr = ValidationResult.failed("bad input", validator_name="InputSchemaValidator")
        assert vr.ok is False
        assert vr.error_message == "bad input"
        assert vr.validator_name == "InputSchemaValidator"


# ═══════════════════════════════════════════════
#  FlowResult tests
# ═══════════════════════════════════════════════


class TestFlowResult:
    def test_defaults(self):
        fr = FlowResult()
        assert fr.ok is True
        assert fr.expanded_node_ids == []
        assert fr.skipped_node_ids == []
        assert fr.redirect_to is None
        assert fr.data is None

    def test_expanded(self):
        fr = FlowResult(expanded_node_ids=["a", "b", "c"])
        assert len(fr.expanded_node_ids) == 3


# ═══════════════════════════════════════════════
#  ExecutionDAGEntry tests
# ═══════════════════════════════════════════════


class TestExecutionDAGEntry:
    def test_defaults(self):
        entry = ExecutionDAGEntry(node_id="n1", plan_node_ref="n1")
        assert entry.id  # auto-generated ULID
        assert entry.status == NodeExecutionStatus.PENDING
        assert entry.result is None
        assert entry.started_at  # auto-generated timestamp
        assert entry.completed_at is None
        assert entry.parent_entry_id is None
        assert entry.state_version is None

    def test_with_result(self):
        entry = ExecutionDAGEntry(
            node_id="n1",
            plan_node_ref="n1",
            status=NodeExecutionStatus.SUCCESS,
            result=NodeResult.success("done"),
            state_version=3,
        )
        assert entry.result is not None
        assert entry.result.ok is True
        assert entry.state_version == 3

    def test_serialization_roundtrip(self):
        entry = ExecutionDAGEntry(
            node_id="n1",
            plan_node_ref="n1",
            status=NodeExecutionStatus.FAILED,
            result=NodeResult.failure("oops"),
        )
        data = entry.model_dump()
        entry2 = ExecutionDAGEntry.model_validate(data)
        assert entry2.node_id == entry.node_id
        assert entry2.status == NodeExecutionStatus.FAILED
        assert entry2.result is not None
        assert entry2.result.error_message == "oops"


# ═══════════════════════════════════════════════
#  OrchestratorConfig tests
# ═══════════════════════════════════════════════


class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.default_timeout_seconds == 30.0
        assert cfg.default_max_retries == 3
        assert cfg.default_retry_base_delay == 0.1

    def test_override(self):
        cfg = OrchestratorConfig(default_timeout_seconds=60.0, default_max_retries=5)
        assert cfg.default_timeout_seconds == 60.0
        assert cfg.default_max_retries == 5


# ═══════════════════════════════════════════════
#  ExecutionNode construction & serialization
# ═══════════════════════════════════════════════


class TestTextNode:
    def test_construction(self):
        n = TextNode(prompt="Hello world")
        assert n.kind == "text"
        assert n.prompt == "Hello world"
        assert n.system_prompt is None
        assert n.model is None
        assert n.id  # auto-generated

    def test_full_construction(self):
        n = TextNode(
            id="node-1",
            prompt="Summarize",
            system_prompt="You are helpful.",
            model="gpt-4",
            temperature=0.7,
            max_tokens=512,
            label="summary",
            timeout_seconds=10.0,
            max_retries=2,
            ext={"custom": "value"},
        )
        assert n.id == "node-1"
        assert n.kind == "text"
        assert n.label == "summary"
        assert n.timeout_seconds == 10.0
        assert n.max_retries == 2
        assert n.ext == {"custom": "value"}
        assert n.temperature == 0.7
        assert n.max_tokens == 512

    def test_serialization_roundtrip(self):
        n = TextNode(prompt="test", model="gpt-4", temperature=0.5)
        data = n.model_dump()
        n2 = TextNode.model_validate(data)
        assert n2 == n
        assert n2.kind == "text"


class TestDataNode:
    def test_construction(self):
        n = DataNode(prompt="Extract entities")
        assert n.kind == "data"
        assert n.prompt == "Extract entities"
        assert n.output_schema is None

    def test_with_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        n = DataNode(prompt="Parse", output_schema=schema)
        assert n.output_schema == schema

    def test_serialization_roundtrip(self):
        n = DataNode(prompt="test", output_schema={"type": "string"}, model="gpt-4")
        data = n.model_dump()
        n2 = DataNode.model_validate(data)
        assert n2 == n


class TestFunctionNode:
    def test_construction(self):
        n = FunctionNode(function_name="clean_data")
        assert n.kind == "function"
        assert n.function_name == "clean_data"
        assert n.args == []
        assert n.kwargs == {}

    def test_with_args(self):
        n = FunctionNode(function_name="process", args=[1, "two"], kwargs={"flag": True})
        assert n.args == [1, "two"]
        assert n.kwargs == {"flag": True}

    def test_serialization_roundtrip(self):
        n = FunctionNode(function_name="fn", args=[1], kwargs={"k": "v"})
        data = n.model_dump()
        n2 = FunctionNode.model_validate(data)
        assert n2 == n


class TestToolNode:
    def test_construction(self):
        n = ToolNode(prompt="Use tools")
        assert n.kind == "tool"
        assert n.prompt == "Use tools"
        assert n.tools == []

    def test_with_tools(self):
        tools = [{"name": "search", "description": "Search the web"}]
        n = ToolNode(prompt="Find info", tools=tools, model="gpt-4")
        assert n.tools == tools
        assert n.model == "gpt-4"

    def test_serialization_roundtrip(self):
        n = ToolNode(prompt="test", tools=[{"name": "t"}], model="gpt-4")
        data = n.model_dump()
        n2 = ToolNode.model_validate(data)
        assert n2 == n


# ═══════════════════════════════════════════════
#  FlowNode construction & serialization
# ═══════════════════════════════════════════════


class TestForEachNode:
    def test_construction(self):
        n = ForEachNode(inner_node_id="process-item")
        assert n.kind == "foreach"
        assert n.inner_node_id == "process-item"
        assert n.failure_strategy == "fail_fast"

    def test_collect_all(self):
        n = ForEachNode(inner_node_id="x", failure_strategy="collect_all")
        assert n.failure_strategy == "collect_all"

    def test_invalid_strategy(self):
        with pytest.raises(ValidationError):
            ForEachNode(inner_node_id="x", failure_strategy="invalid")

    def test_serialization_roundtrip(self):
        n = ForEachNode(inner_node_id="x", failure_strategy="collect_all")
        data = n.model_dump()
        n2 = ForEachNode.model_validate(data)
        assert n2 == n


class TestFilterNode:
    def test_construction(self):
        n = FilterNode(condition="len(items) > 0")
        assert n.kind == "filter"
        assert n.condition == "len(items) > 0"


class TestSwitchNode:
    def test_construction(self):
        n = SwitchNode(
            condition="category",
            branches={"a": "node-a", "b": "node-b"},
            default_branch="node-default",
        )
        assert n.kind == "switch"
        assert n.branches == {"a": "node-a", "b": "node-b"}
        assert n.default_branch == "node-default"

    def test_empty_branches(self):
        n = SwitchNode(condition="x")
        assert n.branches == {}
        assert n.default_branch is None


class TestGetNode:
    def test_construction(self):
        n = GetNode(key="user_input")
        assert n.kind == "get"
        assert n.key == "user_input"


class TestIfNotOkNode:
    def test_construction(self):
        n = IfNotOkNode(check_node_id="prev")
        assert n.kind == "if_not_ok"
        assert n.check_node_id == "prev"
        assert n.redirect_to is None

    def test_with_redirect(self):
        n = IfNotOkNode(check_node_id="prev", redirect_to="fallback")
        assert n.redirect_to == "fallback"


class TestCompositeNode:
    def test_construction(self):
        n = CompositeNode()
        assert n.kind == "composite"
        assert n.sub_dag is None


# ═══════════════════════════════════════════════
#  Discriminated union parsing
# ═══════════════════════════════════════════════


class TestDiscriminatedUnion:
    """Verify Pydantic tagged-union deserialization via the `kind` discriminator."""

    adapter = TypeAdapter(Node)

    def test_parse_text_node(self):
        data = {"kind": "text", "prompt": "Hello", "id": "n1"}
        node = self.adapter.validate_python(data)
        assert isinstance(node, TextNode)
        assert node.prompt == "Hello"

    def test_parse_data_node(self):
        data = {"kind": "data", "prompt": "Extract", "id": "n1"}
        node = self.adapter.validate_python(data)
        assert isinstance(node, DataNode)

    def test_parse_function_node(self):
        data = {"kind": "function", "function_name": "fn", "id": "n1"}
        node = self.adapter.validate_python(data)
        assert isinstance(node, FunctionNode)

    def test_parse_tool_node(self):
        data = {"kind": "tool", "prompt": "Use tools", "id": "n1"}
        node = self.adapter.validate_python(data)
        assert isinstance(node, ToolNode)

    def test_parse_foreach_node(self):
        data = {"kind": "foreach", "inner_node_id": "x", "id": "n1"}
        node = self.adapter.validate_python(data)
        assert isinstance(node, ForEachNode)

    def test_parse_filter_node(self):
        data = {"kind": "filter", "condition": "true", "id": "n1"}
        node = self.adapter.validate_python(data)
        assert isinstance(node, FilterNode)

    def test_parse_switch_node(self):
        data = {"kind": "switch", "condition": "x", "id": "n1"}
        node = self.adapter.validate_python(data)
        assert isinstance(node, SwitchNode)

    def test_parse_get_node(self):
        data = {"kind": "get", "key": "k", "id": "n1"}
        node = self.adapter.validate_python(data)
        assert isinstance(node, GetNode)

    def test_parse_if_not_ok_node(self):
        data = {"kind": "if_not_ok", "check_node_id": "prev", "id": "n1"}
        node = self.adapter.validate_python(data)
        assert isinstance(node, IfNotOkNode)

    def test_parse_composite_node(self):
        data = {"kind": "composite", "id": "n1"}
        node = self.adapter.validate_python(data)
        assert isinstance(node, CompositeNode)

    def test_unknown_kind_fails(self):
        data = {"kind": "unknown", "id": "n1"}
        with pytest.raises(ValidationError):
            self.adapter.validate_python(data)

    def test_json_roundtrip_all_kinds(self):
        """Serialize each node to JSON and back via the union adapter."""
        nodes = [
            TextNode(id="1", prompt="hi"),
            DataNode(id="2", prompt="extract"),
            FunctionNode(id="3", function_name="fn"),
            ToolNode(id="4", prompt="use"),
            ForEachNode(id="5", inner_node_id="x"),
            FilterNode(id="6", condition="c"),
            SwitchNode(id="7", condition="c"),
            GetNode(id="8", key="k"),
            IfNotOkNode(id="9", check_node_id="prev"),
            CompositeNode(id="10"),
        ]
        for original in nodes:
            json_bytes = self.adapter.dump_json(original)
            restored = self.adapter.validate_json(json_bytes)
            assert type(restored) is type(original)
            assert restored.id == original.id
            assert restored.kind == original.kind


# ═══════════════════════════════════════════════
#  Node inheritance checks
# ═══════════════════════════════════════════════


class TestNodeHierarchy:
    def test_text_is_execution_node(self):
        assert issubclass(TextNode, ExecutionNode)
        assert issubclass(TextNode, BaseNode)

    def test_data_is_execution_node(self):
        assert issubclass(DataNode, ExecutionNode)

    def test_function_is_execution_node(self):
        assert issubclass(FunctionNode, ExecutionNode)

    def test_tool_is_execution_node(self):
        assert issubclass(ToolNode, ExecutionNode)

    def test_foreach_is_flow_node(self):
        assert issubclass(ForEachNode, FlowNode)
        assert issubclass(ForEachNode, BaseNode)

    def test_filter_is_flow_node(self):
        assert issubclass(FilterNode, FlowNode)

    def test_switch_is_flow_node(self):
        assert issubclass(SwitchNode, FlowNode)

    def test_get_is_flow_node(self):
        assert issubclass(GetNode, FlowNode)

    def test_if_not_ok_is_flow_node(self):
        assert issubclass(IfNotOkNode, FlowNode)

    def test_composite_is_base_node(self):
        assert issubclass(CompositeNode, BaseNode)
        # CompositeNode extends BaseNode directly, not FlowNode
        assert not issubclass(CompositeNode, FlowNode)
        assert not issubclass(CompositeNode, ExecutionNode)


# ═══════════════════════════════════════════════
#  Field validation
# ═══════════════════════════════════════════════


class TestFieldValidation:
    def test_text_node_requires_prompt(self):
        with pytest.raises(ValidationError):
            TextNode()  # type: ignore[call-arg]

    def test_data_node_requires_prompt(self):
        with pytest.raises(ValidationError):
            DataNode()  # type: ignore[call-arg]

    def test_function_node_requires_function_name(self):
        with pytest.raises(ValidationError):
            FunctionNode()  # type: ignore[call-arg]

    def test_tool_node_requires_prompt(self):
        with pytest.raises(ValidationError):
            ToolNode()  # type: ignore[call-arg]

    def test_foreach_node_requires_inner_node_id(self):
        with pytest.raises(ValidationError):
            ForEachNode()  # type: ignore[call-arg]

    def test_filter_node_requires_condition(self):
        with pytest.raises(ValidationError):
            FilterNode()  # type: ignore[call-arg]

    def test_switch_node_requires_condition(self):
        with pytest.raises(ValidationError):
            SwitchNode()  # type: ignore[call-arg]

    def test_get_node_requires_key(self):
        with pytest.raises(ValidationError):
            GetNode()  # type: ignore[call-arg]

    def test_if_not_ok_requires_check_node_id(self):
        with pytest.raises(ValidationError):
            IfNotOkNode()  # type: ignore[call-arg]


# ═══════════════════════════════════════════════
#  ValidationContext tests
# ═══════════════════════════════════════════════


class TestValidationContext:
    def test_defaults(self):
        ctx = ValidationContext()
        assert ctx.completed_node_ids == set()
        assert ctx.ext == {}

    def test_with_completed_nodes(self):
        ctx = ValidationContext(completed_node_ids={"a", "b"})
        assert "a" in ctx.completed_node_ids
        assert "b" in ctx.completed_node_ids


# ═══════════════════════════════════════════════
#  Package-level import test
# ═══════════════════════════════════════════════


class TestPackageImports:
    """Verify that all public names are accessible from the package."""

    def test_import_all_from_package(self):
        import rh_cognitv.orchestrator as orch

        # Protocols
        assert orch.OrchestratorProtocol is not None
        assert orch.DAGProtocol is not None
        assert orch.NodeProtocol is not None
        assert orch.NodeAdapterProtocol is not None
        assert orch.FlowHandlerProtocol is not None
        assert orch.NodeValidatorProtocol is not None
        assert orch.ValidationPipelineProtocol is not None

        # Models
        assert orch.NodeResult is not None
        assert orch.ValidationResult is not None
        assert orch.FlowResult is not None
        assert orch.OrchestratorConfig is not None
        assert orch.NodeExecutionStatus is not None
        assert orch.ExecutionDAGEntry is not None
        assert orch.DAGRunStatus is not None

        # Nodes
        assert orch.BaseNode is not None
        assert orch.ExecutionNode is not None
        assert orch.FlowNode is not None
        assert orch.TextNode is not None
        assert orch.DataNode is not None
        assert orch.FunctionNode is not None
        assert orch.ToolNode is not None
        assert orch.ForEachNode is not None
        assert orch.FilterNode is not None
        assert orch.SwitchNode is not None
        assert orch.GetNode is not None
        assert orch.IfNotOkNode is not None
        assert orch.CompositeNode is not None
        assert orch.Node is not None
