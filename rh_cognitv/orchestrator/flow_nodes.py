"""
Flow node types, handlers, and the full Node discriminated union.

FlowNodes control DAG traversal (expand, skip, branch, redirect).
They are a pure L2 concern — L3 is never involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from .models import FlowResult, NodeResult
from .nodes import (
    BaseNode,
    DataNode,
    FlowNode,
    FunctionNode,
    TextNode,
    ToolNode,
)
from .protocols import FlowHandlerProtocol

if TYPE_CHECKING:
    from .execution_dag import ExecutionDAG


# ──────────────────────────────────────────────
# Concrete FlowNode subtypes
# ──────────────────────────────────────────────


class ForEachNode(FlowNode):
    """Expand an inner node for each item in the data collection."""

    kind: Literal["foreach"] = "foreach"
    inner_node_id: str
    failure_strategy: Literal["fail_fast", "collect_all"] = "fail_fast"


class FilterNode(FlowNode):
    """Filter data, passing only matching items to successors."""

    kind: Literal["filter"] = "filter"
    condition: str  # expression or callable name to evaluate


class SwitchNode(FlowNode):
    """Pick a branch based on a condition value."""

    kind: Literal["switch"] = "switch"
    condition: str  # expression or key to evaluate
    branches: dict[str, str] = Field(default_factory=dict)  # value → target node_id
    default_branch: str | None = None  # fallback target node_id


class GetNode(FlowNode):
    """Retrieve data from context and inject into the pipeline."""

    kind: Literal["get"] = "get"
    key: str  # context key to retrieve


class IfNotOkNode(FlowNode):
    """Skip or redirect when the previous result was not ok."""

    kind: Literal["if_not_ok"] = "if_not_ok"
    check_node_id: str  # node whose result to inspect
    redirect_to: str | None = None  # node to jump to on failure (None = skip successors)


class CompositeNode(BaseNode):
    """
    A node that references a nested PlanDAG (sub-graph).

    The sub-DAG is inlined during traversal. L3's
    ExecutionState.add_level() / remove_level() tracks nesting.

    Priority P3 — stub for now; full implementation in Phase 5+.
    """

    kind: Literal["composite"] = "composite"
    sub_dag: Any = None  # Will be PlanDAG once plan_dag.py exists


# ──────────────────────────────────────────────
# DAG Traversal State
# ──────────────────────────────────────────────


@dataclass
class DAGTraversalState:
    """Snapshot of traversal state passed to FlowHandlers.

    Provides handlers with enough context to make branching/expansion
    decisions without coupling them to the full orchestrator.
    """

    completed_node_ids: set[str]
    execution_dag: ExecutionDAG
    node_results: dict[str, NodeResult]
    ext: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Flow Handler Registry
# ──────────────────────────────────────────────


class FlowHandlerRegistry:
    """Maps FlowNode kinds to their handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, FlowHandlerProtocol] = {}

    def register(self, kind: str, handler: FlowHandlerProtocol) -> None:
        self._handlers[kind] = handler

    def get(self, kind: str) -> FlowHandlerProtocol:
        handler = self._handlers.get(kind)
        if handler is None:
            raise KeyError(f"No flow handler registered for kind '{kind}'")
        return handler

    async def handle(
        self, node: FlowNode, data: Any, dag_state: DAGTraversalState
    ) -> FlowResult:
        handler = self.get(node.kind)
        return await handler.handle(node, data, dag_state)

    @classmethod
    def with_defaults(cls) -> FlowHandlerRegistry:
        """Create a registry pre-loaded with all built-in handlers."""
        registry = cls()
        registry.register("foreach", ForEachHandler())
        registry.register("filter", FilterHandler())
        registry.register("switch", SwitchHandler())
        registry.register("get", GetHandler())
        registry.register("if_not_ok", IfNotOkHandler())
        return registry


# ──────────────────────────────────────────────
# Concrete Flow Handlers
# ──────────────────────────────────────────────


class ForEachHandler(FlowHandlerProtocol):
    """Expand an inner node for each item in the data collection."""

    async def handle(
        self, node: Any, data: Any, dag_state: Any
    ) -> FlowResult:
        if data is None:
            return FlowResult(ok=True, expanded_node_ids=[node.inner_node_id], data=[])
        try:
            items = list(data)
        except TypeError:
            return FlowResult(
                ok=False, error_message="ForEach data is not iterable"
            )
        return FlowResult(
            ok=True,
            expanded_node_ids=[node.inner_node_id],
            data=items,
        )


class FilterHandler(FlowHandlerProtocol):
    """Filter data using a callable looked up from ``dag_state.ext``."""

    async def handle(
        self, node: Any, data: Any, dag_state: Any
    ) -> FlowResult:
        if data is None:
            return FlowResult(ok=True, data=[])
        filter_fn = dag_state.ext.get(node.condition)
        if filter_fn is None:
            return FlowResult(
                ok=False,
                error_message=f"Filter condition '{node.condition}' not found in ext",
            )
        try:
            filtered = [item for item in data if filter_fn(item)]
        except Exception as exc:
            return FlowResult(ok=False, error_message=f"Filter error: {exc}")
        return FlowResult(ok=True, data=filtered)


class SwitchHandler(FlowHandlerProtocol):
    """Pick a branch based on a condition value."""

    async def handle(
        self, node: Any, data: Any, dag_state: Any
    ) -> FlowResult:
        # Resolve condition value: ext first, then data
        condition_value = dag_state.ext.get(node.condition)
        if condition_value is None and data is not None:
            if isinstance(data, dict):
                condition_value = data.get(node.condition)
            else:
                condition_value = str(data)

        target = (
            node.branches.get(str(condition_value))
            if condition_value is not None
            else None
        )
        if target is None:
            target = node.default_branch

        if target is None:
            return FlowResult(
                ok=False,
                error_message=(
                    f"No branch matched for condition "
                    f"'{node.condition}' = '{condition_value}'"
                ),
            )

        skipped = [bid for bid in node.branches.values() if bid != target]
        if node.default_branch and node.default_branch != target:
            if node.default_branch not in skipped:
                skipped.append(node.default_branch)

        return FlowResult(ok=True, redirect_to=target, skipped_node_ids=skipped)


class GetHandler(FlowHandlerProtocol):
    """Retrieve a value from ext or a previous node result."""

    async def handle(
        self, node: Any, data: Any, dag_state: Any
    ) -> FlowResult:
        value = dag_state.ext.get(node.key)
        if value is None:
            result = dag_state.node_results.get(node.key)
            if result is not None:
                value = result.value
        return FlowResult(ok=True, data=value)


class IfNotOkHandler(FlowHandlerProtocol):
    """Redirect or fail when a checked node's result was not ok."""

    async def handle(
        self, node: Any, data: Any, dag_state: Any
    ) -> FlowResult:
        result = dag_state.node_results.get(node.check_node_id)
        if result is None:
            return FlowResult(
                ok=False,
                error_message=f"Node '{node.check_node_id}' has no result",
            )
        if result.ok:
            return FlowResult(ok=True)
        if node.redirect_to:
            return FlowResult(ok=True, redirect_to=node.redirect_to)
        return FlowResult(
            ok=False,
            error_message=(
                f"Node '{node.check_node_id}' failed: "
                f"{result.error_message}"
            ),
        )


# ──────────────────────────────────────────────
# Full discriminated union (execution + flow nodes)
# ──────────────────────────────────────────────

Node = Annotated[
    TextNode
    | DataNode
    | FunctionNode
    | ToolNode
    | ForEachNode
    | FilterNode
    | SwitchNode
    | GetNode
    | IfNotOkNode
    | CompositeNode,
    Field(discriminator="kind"),
]
