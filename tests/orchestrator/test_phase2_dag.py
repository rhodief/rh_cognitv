"""Tests for Phase 2 — DAG Data Structures (PlanDAG + ExecutionDAG)."""

from __future__ import annotations

import pytest

from rh_cognitv.orchestrator.nodes import FunctionNode, TextNode
from rh_cognitv.orchestrator.models import NodeExecutionStatus, NodeResult
from rh_cognitv.orchestrator.plan_dag import (
    DAG,
    DAGBuilder,
    CycleError,
    DisconnectedError,
    DuplicateEdgeError,
    DuplicateNodeError,
    FrozenDAGError,
    MissingNodeError,
    PlanDAG,
)
from rh_cognitv.orchestrator.execution_dag import ExecutionDAG


# ═══════════════════════════════════════════════
#  Helper factories
# ═══════════════════════════════════════════════


def _text(nid: str) -> TextNode:
    return TextNode(id=nid, prompt=f"prompt-{nid}")


def _func(nid: str) -> FunctionNode:
    return FunctionNode(id=nid, function_name=f"fn_{nid}")


def _linear_dag() -> DAG:
    """A → B → C"""
    dag = DAG()
    dag.add_node("a", _text("a"))
    dag.add_node("b", _text("b"))
    dag.add_node("c", _text("c"))
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    return dag


def _diamond_dag() -> DAG:
    """
    A → B
    A → C
    B → D
    C → D
    """
    dag = DAG()
    for nid in "abcd":
        dag.add_node(nid, _text(nid))
    dag.add_edge("a", "b")
    dag.add_edge("a", "c")
    dag.add_edge("b", "d")
    dag.add_edge("c", "d")
    return dag


# ═══════════════════════════════════════════════
#  DAG — Construction
# ═══════════════════════════════════════════════


class TestDAGConstruction:
    def test_empty_dag(self):
        dag = DAG()
        assert dag.node_count() == 0
        assert dag.get_initial_nodes() == []

    def test_single_node(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        assert dag.node_count() == 1
        assert dag.get_initial_nodes() == ["a"]
        assert dag.successors("a") == []
        assert dag.predecessors("a") == []

    def test_add_edge(self):
        dag = _linear_dag()
        assert dag.successors("a") == ["b"]
        assert dag.successors("b") == ["c"]
        assert dag.predecessors("b") == ["a"]
        assert dag.predecessors("c") == ["b"]

    def test_duplicate_node_raises(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        with pytest.raises(DuplicateNodeError, match="'a'"):
            dag.add_node("a", _text("a"))

    def test_edge_missing_source_raises(self):
        dag = DAG()
        dag.add_node("b", _text("b"))
        with pytest.raises(MissingNodeError, match="Source.*'a'"):
            dag.add_edge("a", "b")

    def test_edge_missing_target_raises(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        with pytest.raises(MissingNodeError, match="Target.*'z'"):
            dag.add_edge("a", "z")

    def test_self_loop_raises(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        with pytest.raises(CycleError, match="Self-loop"):
            dag.add_edge("a", "a")

    def test_duplicate_edge_raises(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        dag.add_node("b", _text("b"))
        dag.add_edge("a", "b")
        with pytest.raises(DuplicateEdgeError):
            dag.add_edge("a", "b")

    def test_has_node(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        assert dag.has_node("a") is True
        assert dag.has_node("z") is False

    def test_node_ids(self):
        dag = _linear_dag()
        assert set(dag.node_ids()) == {"a", "b", "c"}

    def test_get_node_returns_correct_node(self):
        dag = DAG()
        node = _text("a")
        dag.add_node("a", node)
        assert dag.get_node("a") is node

    def test_get_node_missing_raises(self):
        dag = DAG()
        with pytest.raises(MissingNodeError):
            dag.get_node("x")


# ═══════════════════════════════════════════════
#  DAG — Topological Sort
# ═══════════════════════════════════════════════


class TestTopologicalSort:
    def test_linear_order(self):
        dag = _linear_dag()
        order = dag.topological_order()
        assert order == ["a", "b", "c"]

    def test_diamond_order(self):
        dag = _diamond_dag()
        order = dag.topological_order()
        assert order[0] == "a"
        assert order[-1] == "d"
        # b and c must both come before d
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_detected(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        dag.add_node("b", _text("b"))
        dag.add_node("c", _text("c"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")
        with pytest.raises(CycleError, match="cycle"):
            dag.topological_order()

    def test_single_node_topo(self):
        dag = DAG()
        dag.add_node("x", _text("x"))
        assert dag.topological_order() == ["x"]

    def test_wide_parallel(self):
        """Multiple roots with no edges — all are initial, order is valid."""
        dag = DAG()
        for nid in ["x", "y", "z"]:
            dag.add_node(nid, _text(nid))
        order = dag.topological_order()
        assert set(order) == {"x", "y", "z"}


# ═══════════════════════════════════════════════
#  DAG — Initial Nodes & Ready Queue
# ═══════════════════════════════════════════════


class TestInitialAndReady:
    def test_initial_nodes_linear(self):
        dag = _linear_dag()
        assert dag.get_initial_nodes() == ["a"]

    def test_initial_nodes_diamond(self):
        dag = _diamond_dag()
        assert dag.get_initial_nodes() == ["a"]

    def test_initial_nodes_multiple_roots(self):
        dag = DAG()
        dag.add_node("x", _text("x"))
        dag.add_node("y", _text("y"))
        dag.add_node("z", _text("z"))
        dag.add_edge("x", "z")
        dag.add_edge("y", "z")
        initials = dag.get_initial_nodes()
        assert set(initials) == {"x", "y"}

    def test_ready_queue_linear(self):
        dag = _linear_dag()
        # After "a" completes
        ready = dag.get_newly_ready_nodes({"a"})
        assert ready == ["b"]
        # After "a" and "b" complete
        ready = dag.get_newly_ready_nodes({"a", "b"})
        assert ready == ["c"]
        # After all complete — nothing new
        ready = dag.get_newly_ready_nodes({"a", "b", "c"})
        assert ready == []

    def test_ready_queue_diamond(self):
        dag = _diamond_dag()
        # After "a" — b and c are ready
        ready = dag.get_newly_ready_nodes({"a"})
        assert set(ready) == {"b", "c"}
        # After "a" and "b" — only c is ready (d needs both b and c)
        ready = dag.get_newly_ready_nodes({"a", "b"})
        assert ready == ["c"]
        # After "a", "b", "c" — d is ready
        ready = dag.get_newly_ready_nodes({"a", "b", "c"})
        assert ready == ["d"]

    def test_ready_queue_full_traversal(self):
        """Simulate full sequential traversal of a diamond DAG."""
        dag = _diamond_dag()
        completed: set[str] = set()
        execution_order: list[str] = []

        ready = dag.get_initial_nodes()
        while ready:
            for nid in ready:
                execution_order.append(nid)
                completed.add(nid)
            ready = dag.get_newly_ready_nodes(completed)

        assert execution_order[0] == "a"
        assert execution_order[-1] == "d"
        assert set(execution_order) == {"a", "b", "c", "d"}


# ═══════════════════════════════════════════════
#  DAG — Validation
# ═══════════════════════════════════════════════


class TestDAGValidation:
    def test_valid_linear(self):
        dag = _linear_dag()
        dag.validate()  # should not raise

    def test_valid_diamond(self):
        dag = _diamond_dag()
        dag.validate()

    def test_valid_empty(self):
        dag = DAG()
        dag.validate()  # empty is trivially valid

    def test_cycle_fails_validation(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        dag.add_node("b", _text("b"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(CycleError):
            dag.validate()

    def test_disconnected_fails_validation(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        dag.add_node("b", _text("b"))
        # No edges — two isolated components
        with pytest.raises(DisconnectedError):
            dag.validate()

    def test_single_node_is_connected(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        dag.validate()  # single node is trivially connected


# ═══════════════════════════════════════════════
#  DAG — Serialization
# ═══════════════════════════════════════════════


class TestDAGSerialization:
    def test_edge_list_linear(self):
        dag = _linear_dag()
        edges = dag.to_edge_list()
        assert {"from": "a", "to": "b"} in edges
        assert {"from": "b", "to": "c"} in edges
        assert len(edges) == 2

    def test_edge_list_empty(self):
        dag = DAG()
        dag.add_node("x", _text("x"))
        assert dag.to_edge_list() == []

    def test_edge_list_diamond(self):
        dag = _diamond_dag()
        edges = dag.to_edge_list()
        assert len(edges) == 4


# ═══════════════════════════════════════════════
#  PlanDAG — Frozen Wrapper
# ═══════════════════════════════════════════════


class TestPlanDAG:
    def test_creation_validates(self):
        """PlanDAG constructor runs validation — cycle raises."""
        dag = DAG()
        dag.add_node("a", _text("a"))
        dag.add_node("b", _text("b"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(CycleError):
            PlanDAG(dag)

    def test_disconnected_rejected(self):
        dag = DAG()
        dag.add_node("a", _text("a"))
        dag.add_node("b", _text("b"))
        with pytest.raises(DisconnectedError):
            PlanDAG(dag)

    def test_read_methods_work(self):
        plan = PlanDAG(_linear_dag(), name="test")
        assert plan.name == "test"
        assert plan.node_count() == 3
        assert plan.get_initial_nodes() == ["a"]
        assert plan.successors("a") == ["b"]
        assert plan.predecessors("c") == ["b"]
        assert plan.has_node("a") is True
        assert plan.has_node("z") is False
        assert set(plan.node_ids()) == {"a", "b", "c"}

    def test_topological_order(self):
        plan = PlanDAG(_linear_dag())
        assert plan.topological_order() == ["a", "b", "c"]

    def test_newly_ready(self):
        plan = PlanDAG(_diamond_dag())
        assert set(plan.get_newly_ready_nodes({"a"})) == {"b", "c"}

    def test_to_edge_list(self):
        plan = PlanDAG(_linear_dag())
        assert len(plan.to_edge_list()) == 2

    def test_validate(self):
        plan = PlanDAG(_linear_dag())
        plan.validate()  # should not raise

    def test_empty_dag_allowed(self):
        """Empty PlanDAG is valid (no nodes = nothing to do)."""
        plan = PlanDAG(DAG())
        assert plan.node_count() == 0


# ═══════════════════════════════════════════════
#  DAGBuilder — Fluent API
# ═══════════════════════════════════════════════


class TestDAGBuilder:
    def test_basic_build(self):
        plan = (
            DAGBuilder("pipeline")
            .add_node("a", _text("a"))
            .add_node("b", _func("b"))
            .edge("a", "b")
            .build()
        )
        assert isinstance(plan, PlanDAG)
        assert plan.name == "pipeline"
        assert plan.node_count() == 2
        assert plan.successors("a") == ["b"]

    def test_chaining(self):
        builder = DAGBuilder("test")
        r1 = builder.add_node("x", _text("x"))
        assert r1 is builder
        r2 = builder.add_node("y", _text("y"))
        assert r2 is builder
        r3 = builder.edge("x", "y")
        assert r3 is builder

    def test_chaining_returns_self(self):
        builder = DAGBuilder()
        r1 = builder.add_node("a", _text("a"))
        assert r1 is builder
        r2 = builder.add_node("b", _text("b"))
        assert r2 is builder
        r3 = builder.edge("a", "b")
        assert r3 is builder

    def test_build_validates(self):
        """Build catches cycles."""
        builder = DAGBuilder()
        builder.add_node("a", _text("a"))
        builder.add_node("b", _text("b"))
        builder.edge("a", "b")
        builder.edge("b", "a")
        with pytest.raises(CycleError):
            builder.build()

    def test_build_catches_disconnected(self):
        builder = DAGBuilder()
        builder.add_node("a", _text("a"))
        builder.add_node("b", _text("b"))
        with pytest.raises(DisconnectedError):
            builder.build()

    def test_diamond_builder(self):
        plan = (
            DAGBuilder("diamond")
            .add_node("a", _text("a"))
            .add_node("b", _text("b"))
            .add_node("c", _text("c"))
            .add_node("d", _text("d"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .build()
        )
        assert plan.node_count() == 4
        topo = plan.topological_order()
        assert topo[0] == "a"
        assert topo[-1] == "d"

    def test_spec_example(self):
        """Example from DI-L2-04 in the spec."""
        from rh_cognitv.orchestrator.nodes import DataNode, FunctionNode, TextNode

        dag = (
            DAGBuilder("my-pipeline")
            .add_node("extract", DataNode(id="extract", prompt="Extract entities"))
            .add_node("transform", FunctionNode(id="transform", function_name="clean_data"))
            .add_node("summarize", TextNode(id="summarize", prompt="Summarize results"))
            .edge("extract", "transform")
            .edge("transform", "summarize")
            .build()
        )
        assert dag.node_count() == 3
        assert dag.get_initial_nodes() == ["extract"]
        assert dag.topological_order() == ["extract", "transform", "summarize"]


# ═══════════════════════════════════════════════
#  ExecutionDAG — Recording
# ═══════════════════════════════════════════════


class TestExecutionDAGRecording:
    def test_empty(self):
        edag = ExecutionDAG()
        assert edag.entry_count() == 0
        assert edag.entries() == []

    def test_record_success(self):
        edag = ExecutionDAG()
        node = _text("a")
        result = NodeResult.success("output")
        entry = edag.record(node, result, state_version=1)

        assert entry.node_id == "a"
        assert entry.plan_node_ref == "a"
        assert entry.status == NodeExecutionStatus.SUCCESS
        assert entry.result is not None
        assert entry.result.ok is True
        assert entry.state_version == 1
        assert entry.completed_at is not None
        assert edag.entry_count() == 1

    def test_record_failure(self):
        edag = ExecutionDAG()
        node = _text("b")
        result = NodeResult.failure("boom")
        entry = edag.record(node, result)

        assert entry.status == NodeExecutionStatus.FAILED
        assert entry.result is not None
        assert entry.result.ok is False

    def test_record_start(self):
        edag = ExecutionDAG()
        node = _text("a")
        entry = edag.record_start(node)
        assert entry.status == NodeExecutionStatus.RUNNING
        assert entry.result is None

    def test_multiple_records_same_node(self):
        """Append-only: multiple entries for the same node are allowed."""
        edag = ExecutionDAG()
        node = _text("a")
        edag.record(node, NodeResult.failure("first attempt"))
        edag.record(node, NodeResult.success("second attempt"))
        assert edag.entry_count() == 2

        entries = edag.get_all_entries_for_node("a")
        assert len(entries) == 2
        assert entries[0].status == NodeExecutionStatus.FAILED
        assert entries[1].status == NodeExecutionStatus.SUCCESS

    def test_get_entry_returns_latest(self):
        edag = ExecutionDAG()
        node = _text("a")
        edag.record(node, NodeResult.failure("v1"))
        edag.record(node, NodeResult.success("v2"))
        entry = edag.get_entry("a")
        assert entry is not None
        assert entry.status == NodeExecutionStatus.SUCCESS

    def test_get_entry_missing(self):
        edag = ExecutionDAG()
        assert edag.get_entry("nonexistent") is None

    def test_entries_order(self):
        edag = ExecutionDAG()
        for nid in ["a", "b", "c"]:
            edag.record(_text(nid), NodeResult.success(nid))
        entries = edag.entries()
        assert [e.node_id for e in entries] == ["a", "b", "c"]

    def test_record_with_parent_entry_id(self):
        edag = ExecutionDAG()
        node = _text("a")
        e1 = edag.record(node, NodeResult.success("v1"), state_version=1)
        e2 = edag.record(node, NodeResult.success("v2"), parent_entry_id=e1.id)
        assert e2.parent_entry_id == e1.id


# ═══════════════════════════════════════════════
#  ExecutionDAG — Status Queries
# ═══════════════════════════════════════════════


class TestExecutionDAGStatusQueries:
    def test_get_by_status(self):
        edag = ExecutionDAG()
        edag.record(_text("a"), NodeResult.success("ok"))
        edag.record(_text("b"), NodeResult.failure("bad"))
        edag.record(_text("c"), NodeResult.success("ok"))

        successes = edag.get_by_status(NodeExecutionStatus.SUCCESS)
        assert len(successes) == 2
        failures = edag.get_by_status(NodeExecutionStatus.FAILED)
        assert len(failures) == 1
        assert failures[0].node_id == "b"

    def test_is_node_completed(self):
        edag = ExecutionDAG()
        edag.record(_text("a"), NodeResult.success("ok"))
        edag.record(_text("b"), NodeResult.failure("bad"))
        assert edag.is_node_completed("a") is True
        assert edag.is_node_completed("b") is False
        assert edag.is_node_completed("z") is False

    def test_completed_node_ids(self):
        edag = ExecutionDAG()
        edag.record(_text("a"), NodeResult.success("ok"))
        edag.record(_text("b"), NodeResult.failure("bad"))
        edag.record(_text("c"), NodeResult.success("ok"))
        assert edag.completed_node_ids() == {"a", "c"}


# ═══════════════════════════════════════════════
#  ExecutionDAG — Rollback
# ═══════════════════════════════════════════════


class TestExecutionDAGRollback:
    def test_mark_rolled_back(self):
        edag = ExecutionDAG()
        e1 = edag.record(_text("a"), NodeResult.success("v1"), state_version=1)
        e2 = edag.record(_text("b"), NodeResult.success("v2"), state_version=2)
        e3 = edag.record(_text("c"), NodeResult.success("v3"), state_version=3)

        # Rollback from e2 forward
        count = edag.mark_rolled_back(e2)
        assert count == 2  # e2 and e3
        assert e1.status == NodeExecutionStatus.SUCCESS
        assert e2.status == NodeExecutionStatus.ROLLED_BACK
        assert e3.status == NodeExecutionStatus.ROLLED_BACK

    def test_rollback_preserves_entries(self):
        """Append-only invariant: rolled-back entries are still in the list."""
        edag = ExecutionDAG()
        edag.record(_text("a"), NodeResult.success("v1"))
        e2 = edag.record(_text("b"), NodeResult.success("v2"))
        edag.record(_text("c"), NodeResult.success("v3"))

        edag.mark_rolled_back(e2)
        assert edag.entry_count() == 3  # still 3, nothing removed

    def test_rollback_idempotent(self):
        """Rolling back already-rolled-back entries doesn't re-count."""
        edag = ExecutionDAG()
        e1 = edag.record(_text("a"), NodeResult.success("v1"))
        edag.record(_text("b"), NodeResult.success("v2"))

        count1 = edag.mark_rolled_back(e1)
        count2 = edag.mark_rolled_back(e1)
        assert count2 == 0  # already rolled back

    def test_rollback_does_not_affect_skipped(self):
        edag = ExecutionDAG()
        e1 = edag.record(_text("a"), NodeResult.success("v1"))
        edag.mark_skipped(_text("b"))
        e3 = edag.record(_text("c"), NodeResult.success("v3"))

        count = edag.mark_rolled_back(e1)
        # e1 → rolled back, b is skipped (unchanged), e3 → rolled back
        assert count == 2
        entries = edag.entries()
        assert entries[1].status == NodeExecutionStatus.SKIPPED

    def test_completed_node_ids_after_rollback(self):
        edag = ExecutionDAG()
        edag.record(_text("a"), NodeResult.success("v1"))
        e2 = edag.record(_text("b"), NodeResult.success("v2"))
        edag.record(_text("c"), NodeResult.success("v3"))

        edag.mark_rolled_back(e2)
        # "a" is still success, "b" and "c" are rolled back
        assert edag.completed_node_ids() == {"a"}


# ═══════════════════════════════════════════════
#  ExecutionDAG — Waiting (Escalation)
# ═══════════════════════════════════════════════


class TestExecutionDAGWaiting:
    def test_mark_waiting(self):
        edag = ExecutionDAG()
        edag.record(_text("a"), NodeResult.success("ok"))
        entry = edag.mark_waiting("a")
        assert entry is not None
        assert entry.status == NodeExecutionStatus.WAITING

    def test_mark_waiting_missing(self):
        edag = ExecutionDAG()
        assert edag.mark_waiting("nonexistent") is None

    def test_mark_skipped(self):
        edag = ExecutionDAG()
        entry = edag.mark_skipped(_text("a"))
        assert entry.status == NodeExecutionStatus.SKIPPED
        assert edag.entry_count() == 1


# ═══════════════════════════════════════════════
#  ExecutionDAG — Append-Only Invariant
# ═══════════════════════════════════════════════


class TestAppendOnlyInvariant:
    """Verify the append-only contract: no entries are ever removed."""

    def test_record_only_appends(self):
        edag = ExecutionDAG()
        for i in range(5):
            edag.record(_text(f"n{i}"), NodeResult.success(f"v{i}"))
        assert edag.entry_count() == 5

    def test_rollback_then_record_grows(self):
        """After rollback, new records append — nothing is deleted."""
        edag = ExecutionDAG()
        e1 = edag.record(_text("a"), NodeResult.success("v1"), state_version=1)
        edag.record(_text("b"), NodeResult.success("v2"), state_version=2)
        edag.record(_text("c"), NodeResult.success("v3"), state_version=3)

        edag.mark_rolled_back(e1)
        assert edag.entry_count() == 3

        # New execution after rollback
        edag.record(_text("a"), NodeResult.success("v1-retry"), state_version=4)
        assert edag.entry_count() == 4

        # History shows both the original and the retry
        all_a = edag.get_all_entries_for_node("a")
        assert len(all_a) == 2
        assert all_a[0].status == NodeExecutionStatus.ROLLED_BACK
        assert all_a[1].status == NodeExecutionStatus.SUCCESS


# ═══════════════════════════════════════════════
#  Integration: DAGBuilder → PlanDAG → simulated traversal → ExecutionDAG
# ═══════════════════════════════════════════════


class TestPlanToExecution:
    def test_linear_traversal(self):
        """Build a PlanDAG, simulate traversal, record in ExecutionDAG."""
        plan = (
            DAGBuilder("linear")
            .add_node("a", _text("a"))
            .add_node("b", _text("b"))
            .add_node("c", _text("c"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )
        edag = ExecutionDAG()
        completed: set[str] = set()
        version = 0

        ready = plan.get_initial_nodes()
        while ready:
            for nid in ready:
                node = plan.get_node(nid)
                result = NodeResult.success(f"output-{nid}")
                version += 1
                edag.record(node, result, state_version=version)
                completed.add(nid)
            ready = plan.get_newly_ready_nodes(completed)

        assert edag.entry_count() == 3
        assert edag.completed_node_ids() == {"a", "b", "c"}
        entries = edag.entries()
        assert [e.node_id for e in entries] == ["a", "b", "c"]
        assert entries[-1].state_version == 3

    def test_diamond_traversal(self):
        plan = (
            DAGBuilder("diamond")
            .add_node("a", _text("a"))
            .add_node("b", _text("b"))
            .add_node("c", _text("c"))
            .add_node("d", _text("d"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .build()
        )
        edag = ExecutionDAG()
        completed: set[str] = set()

        ready = plan.get_initial_nodes()
        while ready:
            for nid in ready:
                node = plan.get_node(nid)
                edag.record(node, NodeResult.success(f"out-{nid}"))
                completed.add(nid)
            ready = plan.get_newly_ready_nodes(completed)

        assert edag.entry_count() == 4
        assert edag.completed_node_ids() == {"a", "b", "c", "d"}
        order = [e.node_id for e in edag.entries()]
        assert order[0] == "a"
        assert order[-1] == "d"

    def test_traversal_with_failure_and_rollback(self):
        """Simulate: run A→B→C, B fails, rollback B+C, retry B, continue C."""
        plan = (
            DAGBuilder("retry")
            .add_node("a", _text("a"))
            .add_node("b", _text("b"))
            .add_node("c", _text("c"))
            .edge("a", "b")
            .edge("b", "c")
            .build()
        )
        edag = ExecutionDAG()

        # Execute A (success)
        edag.record(plan.get_node("a"), NodeResult.success("ok-a"), state_version=1)

        # Execute B (failure)
        e_b = edag.record(plan.get_node("b"), NodeResult.failure("timeout"), state_version=2)

        # Rollback from B
        edag.mark_rolled_back(e_b)
        assert edag.entry_count() == 2

        # Retry B (success)
        edag.record(plan.get_node("b"), NodeResult.success("ok-b-retry"), state_version=3)

        # Execute C (success)
        edag.record(plan.get_node("c"), NodeResult.success("ok-c"), state_version=4)

        assert edag.entry_count() == 4
        assert edag.completed_node_ids() == {"a", "b", "c"}

        # Verify history for B
        b_entries = edag.get_all_entries_for_node("b")
        assert len(b_entries) == 2
        assert b_entries[0].status == NodeExecutionStatus.ROLLED_BACK
        assert b_entries[1].status == NodeExecutionStatus.SUCCESS


# ═══════════════════════════════════════════════
#  Package-level import test
# ═══════════════════════════════════════════════


class TestPhase2PackageImports:
    def test_all_phase2_exports(self):
        import rh_cognitv.orchestrator as orch

        assert orch.DAG is not None
        assert orch.PlanDAG is not None
        assert orch.DAGBuilder is not None
        assert orch.ExecutionDAG is not None
        assert orch.DAGError is not None
        assert orch.CycleError is not None
        assert orch.DisconnectedError is not None
        assert orch.DuplicateNodeError is not None
        assert orch.MissingNodeError is not None
        assert orch.DuplicateEdgeError is not None
        assert orch.FrozenDAGError is not None
