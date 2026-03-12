"""
Node type hierarchy.

BaseNode → ExecutionNode → TextNode, DataNode, FunctionNode, ToolNode
BaseNode → FlowNode       (concrete flow subtypes live in flow_nodes.py)

The `kind` field is the Pydantic discriminator for the tagged union
AND the key used by the AdapterRegistry for dispatch.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from rh_cognitv.execution_platform.types import ID, Ext, generate_ulid


# ──────────────────────────────────────────────
# Base hierarchy
# ──────────────────────────────────────────────


class BaseNode(BaseModel):
    """Root of the node type hierarchy."""

    id: ID = Field(default_factory=generate_ulid)
    kind: str
    label: str | None = None
    timeout_seconds: float | None = None  # per-node override
    max_retries: int | None = None  # per-node override
    ext: Ext = Field(default_factory=dict)


class ExecutionNode(BaseNode):
    """
    Node that maps to an L3 ExecutionEvent.

    Subtypes correspond 1:1 to EventKind values.
    """

    pass


class FlowNode(BaseNode):
    """
    Node that controls DAG traversal — pure L2, never reaches L3.

    Concrete subtypes live in flow_nodes.py.
    """

    pass


# ──────────────────────────────────────────────
# Concrete ExecutionNode subtypes
# ──────────────────────────────────────────────


class TextNode(ExecutionNode):
    """LLM text-generation node (EventKind.TEXT)."""

    kind: Literal["text"] = "text"
    prompt: str
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class DataNode(ExecutionNode):
    """Structured data extraction node (EventKind.DATA)."""

    kind: Literal["data"] = "data"
    prompt: str
    output_schema: dict[str, Any] | None = None
    model: str | None = None


class FunctionNode(ExecutionNode):
    """Direct function invocation node (EventKind.FUNCTION)."""

    kind: Literal["function"] = "function"
    function_name: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ToolNode(ExecutionNode):
    """LLM-with-tools node (EventKind.TOOL)."""

    kind: Literal["tool"] = "tool"
    prompt: str
    tools: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None


# ──────────────────────────────────────────────
# Discriminated union  (flow subtypes added at bottom of flow_nodes.py)
# ──────────────────────────────────────────────

# NOTE: The full `Node` union including flow nodes is assembled in
# flow_nodes.py to avoid circular imports.  This module exports a
# partial union covering execution nodes only — useful for type-checks
# that don't involve flow nodes.

Node = Annotated[
    TextNode | DataNode | FunctionNode | ToolNode,
    Field(discriminator="kind"),
]
