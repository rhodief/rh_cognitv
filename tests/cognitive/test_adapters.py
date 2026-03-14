"""
Tests for cognitive/adapters.py — Phase 3.6 L1→L2 Adapter.

Covers:
  - SkillToDAGAdapter.to_dag: kind→node mapping, edge wiring, context_refs, constraints
  - ResultAdapter.from_result: NodeResult extraction from ExecutionDAG
  - OrchestratorResult model
"""

import pytest

from rh_cognitv.cognitive.adapters import (
    OrchestratorResult,
    ResultAdapter,
    SkillToDAGAdapter,
)
from rh_cognitv.cognitive.models import (
    ContextRef,
    DataStepConfig,
    FunctionStepConfig,
    SkillConstraints,
    SkillPlan,
    SkillStep,
    TextStepConfig,
    ToolStepConfig,
)
from rh_cognitv.execution_platform.models import MemoryQuery
from rh_cognitv.orchestrator.execution_dag import ExecutionDAG
from rh_cognitv.orchestrator.models import NodeResult
from rh_cognitv.orchestrator.nodes import (
    DataNode,
    FunctionNode,
    TextNode,
    ToolNode,
)


# ──── Helpers ────


def _text_step(id: str = "s1", prompt: str = "Hello", **kw) -> SkillStep:
    return SkillStep(id=id, kind="text", config=TextStepConfig(prompt=prompt, **kw))


def _data_step(id: str = "s1", prompt: str = "Extract", **kw) -> SkillStep:
    return SkillStep(id=id, kind="data", config=DataStepConfig(prompt=prompt, **kw))


def _function_step(id: str = "s1", function_name: str = "fn", **kw) -> SkillStep:
    return SkillStep(
        id=id, kind="function", config=FunctionStepConfig(function_name=function_name, **kw)
    )


def _tool_step(id: str = "s1", prompt: str = "Use tool", **kw) -> SkillStep:
    return SkillStep(id=id, kind="tool", config=ToolStepConfig(prompt=prompt, **kw))


def _plan(
    steps: list[SkillStep],
    name: str = "test",
    constraints: SkillConstraints | None = None,
) -> SkillPlan:
    return SkillPlan(name=name, steps=steps, constraints=constraints)


# ══════════════════════════════════════════════
# SkillToDAGAdapter — single-step plans
# ══════════════════════════════════════════════


class TestSingleStepPlan:
    """Single-step plans produce single-node PlanDAGs."""

    def setup_method(self):
        self.adapter = SkillToDAGAdapter()

    def test_single_text_step(self):
        dag = self.adapter.to_dag(_plan([_text_step("s1", "Hello")]))
        assert dag.node_count() == 1
        node = dag.get_node("s1")
        assert isinstance(node, TextNode)
        assert node.prompt == "Hello"

    def test_single_data_step(self):
        dag = self.adapter.to_dag(_plan([_data_step("s1", "Extract data")]))
        node = dag.get_node("s1")
        assert isinstance(node, DataNode)
        assert node.prompt == "Extract data"

    def test_single_function_step(self):
        dag = self.adapter.to_dag(_plan([_function_step("s1", "process")]))
        node = dag.get_node("s1")
        assert isinstance(node, FunctionNode)
        assert node.function_name == "process"

    def test_single_tool_step(self):
        dag = self.adapter.to_dag(_plan([_tool_step("s1", "Use the tool")]))
        node = dag.get_node("s1")
        assert isinstance(node, ToolNode)
        assert node.prompt == "Use the tool"

    def test_dag_name_matches_plan(self):
        dag = self.adapter.to_dag(_plan([_text_step()], name="my-pipeline"))
        assert dag.name == "my-pipeline"

    def test_single_node_no_edges(self):
        dag = self.adapter.to_dag(_plan([_text_step()]))
        assert dag.to_edge_list() == []


# ══════════════════════════════════════════════
# SkillToDAGAdapter — multi-step edge wiring
# ══════════════════════════════════════════════


class TestMultiStepEdgeWiring:
    """Multi-step plans with sequential and explicit dependency wiring."""

    def setup_method(self):
        self.adapter = SkillToDAGAdapter()

    def test_two_sequential_steps(self):
        dag = self.adapter.to_dag(
            _plan([_text_step("a", "first"), _text_step("b", "second")])
        )
        assert dag.node_count() == 2
        edges = dag.to_edge_list()
        assert len(edges) == 1
        assert edges[0] == {"from": "a", "to": "b"}

    def test_three_sequential_steps(self):
        dag = self.adapter.to_dag(
            _plan([
                _text_step("a", "1"),
                _data_step("b", "2"),
                _function_step("c", "process"),
            ])
        )
        assert dag.node_count() == 3
        edges = dag.to_edge_list()
        assert len(edges) == 2
        assert {"from": "a", "to": "b"} in edges
        assert {"from": "b", "to": "c"} in edges

    def test_explicit_depends_on(self):
        steps = [
            _text_step("a", "first"),
            SkillStep(
                id="b", kind="text",
                config=TextStepConfig(prompt="second"),
                depends_on=["a"],
            ),
        ]
        dag = self.adapter.to_dag(_plan(steps))
        edges = dag.to_edge_list()
        assert len(edges) == 1
        assert edges[0] == {"from": "a", "to": "b"}

    def test_fan_out_depends_on(self):
        """Two steps depend on the same parent → fan-out."""
        steps = [
            _text_step("a", "root"),
            SkillStep(id="b", kind="text", config=TextStepConfig(prompt="b"), depends_on=["a"]),
            SkillStep(id="c", kind="data", config=DataStepConfig(prompt="c"), depends_on=["a"]),
        ]
        dag = self.adapter.to_dag(_plan(steps))
        assert dag.node_count() == 3
        edges = dag.to_edge_list()
        assert len(edges) == 2
        assert {"from": "a", "to": "b"} in edges
        assert {"from": "a", "to": "c"} in edges

    def test_diamond_pattern(self):
        """Diamond: A→B, A→C, B→D, C→D."""
        steps = [
            _text_step("a", "root"),
            SkillStep(id="b", kind="text", config=TextStepConfig(prompt="b"), depends_on=["a"]),
            SkillStep(id="c", kind="text", config=TextStepConfig(prompt="c"), depends_on=["a"]),
            SkillStep(
                id="d", kind="text",
                config=TextStepConfig(prompt="d"),
                depends_on=["b", "c"],
            ),
        ]
        dag = self.adapter.to_dag(_plan(steps))
        assert dag.node_count() == 4
        assert len(dag.to_edge_list()) == 4

    def test_topological_order_respects_edges(self):
        dag = self.adapter.to_dag(
            _plan([_text_step("a", "1"), _text_step("b", "2"), _text_step("c", "3")])
        )
        order = dag.topological_order()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_mixed_sequential_and_depends_on(self):
        """Step B has depends_on; step C falls back to sequential from B."""
        steps = [
            _text_step("a", "first"),
            SkillStep(id="b", kind="text", config=TextStepConfig(prompt="second"), depends_on=["a"]),
            _text_step("c", "third"),  # no depends_on → sequential from b
        ]
        dag = self.adapter.to_dag(_plan(steps))
        edges = dag.to_edge_list()
        assert {"from": "a", "to": "b"} in edges
        assert {"from": "b", "to": "c"} in edges


# ══════════════════════════════════════════════
# SkillToDAGAdapter — node field mapping
# ══════════════════════════════════════════════


class TestNodeFieldMapping:
    """Step config fields map correctly to L2 node fields."""

    def setup_method(self):
        self.adapter = SkillToDAGAdapter()

    def test_text_step_all_fields(self):
        step = SkillStep(
            id="s1",
            kind="text",
            config=TextStepConfig(
                prompt="Hello",
                system_prompt="You are helpful",
                model="gpt-4",
                temperature=0.7,
                max_tokens=100,
            ),
        )
        dag = self.adapter.to_dag(_plan([step]))
        node = dag.get_node("s1")
        assert isinstance(node, TextNode)
        assert node.prompt == "Hello"
        assert node.system_prompt == "You are helpful"
        assert node.model == "gpt-4"
        assert node.temperature == 0.7
        assert node.max_tokens == 100

    def test_data_step_all_fields(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        step = SkillStep(
            id="s1",
            kind="data",
            config=DataStepConfig(prompt="Extract", output_schema=schema, model="gpt-4"),
        )
        dag = self.adapter.to_dag(_plan([step]))
        node = dag.get_node("s1")
        assert isinstance(node, DataNode)
        assert node.prompt == "Extract"
        assert node.output_schema == schema
        assert node.model == "gpt-4"

    def test_function_step_all_fields(self):
        step = SkillStep(
            id="s1",
            kind="function",
            config=FunctionStepConfig(
                function_name="process", args=[1, 2], kwargs={"key": "value"}
            ),
        )
        dag = self.adapter.to_dag(_plan([step]))
        node = dag.get_node("s1")
        assert isinstance(node, FunctionNode)
        assert node.function_name == "process"
        assert node.args == [1, 2]
        assert node.kwargs == {"key": "value"}

    def test_tool_step_all_fields(self):
        tools = [{"name": "search", "params": {"q": "test"}}]
        step = SkillStep(
            id="s1",
            kind="tool",
            config=ToolStepConfig(prompt="Use tool", tools=tools, model="gpt-4"),
        )
        dag = self.adapter.to_dag(_plan([step]))
        node = dag.get_node("s1")
        assert isinstance(node, ToolNode)
        assert node.prompt == "Use tool"
        assert node.tools == tools
        assert node.model == "gpt-4"

    def test_node_id_matches_step_id(self):
        dag = self.adapter.to_dag(_plan([_text_step("my-step-1", "prompt")]))
        node = dag.get_node("my-step-1")
        assert node.id == "my-step-1"

    def test_kind_config_mismatch_raises(self):
        """Mismatched kind and config type raises ValueError."""
        step = SkillStep(
            id="s1",
            kind="text",
            config=DataStepConfig(prompt="oops"),
        )
        with pytest.raises(ValueError, match="Cannot map step"):
            self.adapter.to_dag(_plan([step]))


# ══════════════════════════════════════════════
# SkillToDAGAdapter — context_refs in ext
# ══════════════════════════════════════════════


class TestContextRefsInExt:
    """context_refs are serialized into BaseNode.ext."""

    def setup_method(self):
        self.adapter = SkillToDAGAdapter()

    def test_context_refs_in_ext(self):
        ref = ContextRef(kind="memory", id="mem-1", key="background")
        step = SkillStep(
            id="s1", kind="text",
            config=TextStepConfig(prompt="Hello"),
            context_refs=[ref],
        )
        dag = self.adapter.to_dag(_plan([step]))
        node = dag.get_node("s1")
        assert "context_refs" in node.ext
        refs = node.ext["context_refs"]
        assert len(refs) == 1
        assert refs[0]["kind"] == "memory"
        assert refs[0]["id"] == "mem-1"
        assert refs[0]["key"] == "background"

    def test_no_context_refs_no_ext_key(self):
        dag = self.adapter.to_dag(_plan([_text_step("s1", "Hello")]))
        node = dag.get_node("s1")
        assert "context_refs" not in node.ext

    def test_multiple_context_refs(self):
        refs = [
            ContextRef(kind="memory", id="m1"),
            ContextRef(kind="artifact", slug="draft", version=2),
            ContextRef(kind="previous_result", from_step="s0", key="prior"),
        ]
        step = SkillStep(
            id="s1", kind="text",
            config=TextStepConfig(prompt="Hello"),
            context_refs=refs,
        )
        dag = self.adapter.to_dag(_plan([step]))
        node = dag.get_node("s1")
        ext_refs = node.ext["context_refs"]
        assert len(ext_refs) == 3
        assert [r["kind"] for r in ext_refs] == ["memory", "artifact", "previous_result"]

    def test_query_context_ref_serialized(self):
        ref = ContextRef(kind="query", query=MemoryQuery(text="related", top_k=5))
        step = SkillStep(
            id="s1", kind="text",
            config=TextStepConfig(prompt="Hello"),
            context_refs=[ref],
        )
        dag = self.adapter.to_dag(_plan([step]))
        node = dag.get_node("s1")
        query_data = node.ext["context_refs"][0]["query"]
        assert query_data["text"] == "related"
        assert query_data["top_k"] == 5

    def test_artifact_ref_fields(self):
        ref = ContextRef(kind="artifact", slug="my-doc", version=3, key="doc")
        step = SkillStep(
            id="s1", kind="text",
            config=TextStepConfig(prompt="Hello"),
            context_refs=[ref],
        )
        dag = self.adapter.to_dag(_plan([step]))
        ext_ref = dag.get_node("s1").ext["context_refs"][0]
        assert ext_ref["slug"] == "my-doc"
        assert ext_ref["version"] == 3
        assert ext_ref["key"] == "doc"


# ══════════════════════════════════════════════
# SkillToDAGAdapter — constraints
# ══════════════════════════════════════════════


class TestConstraintsPropagation:
    """Plan constraints propagate to node fields."""

    def setup_method(self):
        self.adapter = SkillToDAGAdapter()

    def test_constraints_on_single_node(self):
        constraints = SkillConstraints(timeout_seconds=30.0, max_retries=5)
        dag = self.adapter.to_dag(_plan([_text_step("s1", "Hello")], constraints=constraints))
        node = dag.get_node("s1")
        assert node.timeout_seconds == 30.0
        assert node.max_retries == 5

    def test_constraints_on_all_nodes(self):
        constraints = SkillConstraints(timeout_seconds=10.0, max_retries=2)
        dag = self.adapter.to_dag(
            _plan([_text_step("a", "1"), _data_step("b", "2")], constraints=constraints)
        )
        for nid in ["a", "b"]:
            node = dag.get_node(nid)
            assert node.timeout_seconds == 10.0
            assert node.max_retries == 2

    def test_no_constraints_defaults_to_none(self):
        dag = self.adapter.to_dag(_plan([_text_step("s1", "Hello")]))
        node = dag.get_node("s1")
        assert node.timeout_seconds is None
        assert node.max_retries is None

    def test_partial_constraints(self):
        constraints = SkillConstraints(timeout_seconds=5.0)
        dag = self.adapter.to_dag(_plan([_text_step("s1", "Hello")], constraints=constraints))
        node = dag.get_node("s1")
        assert node.timeout_seconds == 5.0
        assert node.max_retries is None


# ══════════════════════════════════════════════
# ResultAdapter
# ══════════════════════════════════════════════


class TestResultAdapter:
    """ResultAdapter extracts NodeResult values from ExecutionDAG."""

    def setup_method(self):
        self.adapter = ResultAdapter()

    def test_empty_execution_dag(self):
        result = self.adapter.from_result(ExecutionDAG())
        assert result.step_results == {}
        assert result.success is True

    def test_single_success(self):
        edag = ExecutionDAG()
        node = TextNode(id="s1", prompt="Hello")
        edag.record(node, NodeResult.success("output text"))
        result = self.adapter.from_result(edag)
        assert "s1" in result.step_results
        assert result.step_results["s1"].ok is True
        assert result.step_results["s1"].value == "output text"
        assert result.success is True

    def test_single_failure(self):
        edag = ExecutionDAG()
        node = TextNode(id="s1", prompt="Hello")
        edag.record(node, NodeResult.failure("something went wrong"))
        result = self.adapter.from_result(edag)
        assert result.step_results["s1"].ok is False
        assert result.step_results["s1"].error_message == "something went wrong"
        assert result.success is False

    def test_multiple_nodes(self):
        edag = ExecutionDAG()
        edag.record(TextNode(id="a", prompt="first"), NodeResult.success("result_a"))
        edag.record(TextNode(id="b", prompt="second"), NodeResult.success("result_b"))
        result = self.adapter.from_result(edag)
        assert len(result.step_results) == 2
        assert result.step_results["a"].value == "result_a"
        assert result.step_results["b"].value == "result_b"
        assert result.success is True

    def test_mixed_success_and_failure(self):
        edag = ExecutionDAG()
        edag.record(TextNode(id="a", prompt="ok"), NodeResult.success("good"))
        edag.record(TextNode(id="b", prompt="bad"), NodeResult.failure("bad"))
        result = self.adapter.from_result(edag)
        assert result.success is False
        assert result.step_results["a"].ok is True
        assert result.step_results["b"].ok is False

    def test_running_then_success_uses_latest(self):
        """RUNNING entry (no result) followed by SUCCESS — from_result sees SUCCESS."""
        edag = ExecutionDAG()
        node = TextNode(id="s1", prompt="Hello")
        edag.record_start(node)  # RUNNING, result=None
        edag.record(node, NodeResult.success("done"))
        result = self.adapter.from_result(edag)
        assert result.step_results["s1"].ok is True
        assert result.step_results["s1"].value == "done"
        assert result.success is True

    def test_retry_uses_latest_result(self):
        """Failed then succeeded on retry — from_result sees the success."""
        edag = ExecutionDAG()
        node = TextNode(id="s1", prompt="Hello")
        edag.record(node, NodeResult.failure("first try failed"))
        edag.record(node, NodeResult.success("retry succeeded"))
        result = self.adapter.from_result(edag)
        assert result.step_results["s1"].ok is True
        assert result.step_results["s1"].value == "retry succeeded"
        assert result.success is True


# ══════════════════════════════════════════════
# OrchestratorResult model
# ══════════════════════════════════════════════


class TestOrchestratorResult:
    """OrchestratorResult model construction and serialization."""

    def test_default_construction(self):
        r = OrchestratorResult()
        assert r.step_results == {}
        assert r.success is True

    def test_construction_with_results(self):
        nr = NodeResult.success("hello")
        r = OrchestratorResult(step_results={"s1": nr}, success=True)
        assert r.step_results["s1"].value == "hello"

    def test_serialization_round_trip(self):
        nr = NodeResult.success("data")
        r = OrchestratorResult(step_results={"s1": nr}, success=True)
        data = r.model_dump()
        r2 = OrchestratorResult.model_validate(data)
        assert r2.step_results["s1"].value == "data"
        assert r2.success is True

    def test_failed_result(self):
        nr = NodeResult.failure("broke")
        r = OrchestratorResult(step_results={"s1": nr}, success=False)
        assert r.success is False
        assert r.step_results["s1"].error_message == "broke"
