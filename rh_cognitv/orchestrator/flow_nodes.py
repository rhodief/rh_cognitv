"""
Flow node types and the full Node discriminated union.

FlowNodes control DAG traversal (expand, skip, branch, redirect).
They are a pure L2 concern — L3 is never involved.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from .nodes import (
    BaseNode,
    DataNode,
    FlowNode,
    FunctionNode,
    TextNode,
    ToolNode,
)


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
