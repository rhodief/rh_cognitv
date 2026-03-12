"""
Orchestrator protocols (ABCs).

These define the contracts for the orchestrator layer.
Upper layers (Cognitive) and this layer's own components
depend only on these abstractions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import (
        ExecutionDAGEntry,
        FlowResult,
        NodeResult,
        ValidationContext,
        ValidationResult,
    )
    from .nodes import BaseNode


# ──────────────────────────────────────────────
# Node Protocol
# ──────────────────────────────────────────────


class NodeProtocol(ABC):
    """Minimal contract every node must satisfy."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique node identifier."""
        ...

    @property
    @abstractmethod
    def kind(self) -> str:
        """Discriminator used for serialization and adapter dispatch."""
        ...


# ──────────────────────────────────────────────
# DAG Protocol
# ──────────────────────────────────────────────


class DAGProtocol(ABC):
    """Contract for a directed acyclic graph of nodes."""

    @abstractmethod
    def add_node(self, node_id: str, node: Any) -> None:
        """Add a node to the graph."""
        ...

    @abstractmethod
    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a directed edge between two nodes."""
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> Any:
        """Retrieve a node by its ID."""
        ...

    @abstractmethod
    def successors(self, node_id: str) -> list[str]:
        """Return the IDs of all immediate successors."""
        ...

    @abstractmethod
    def predecessors(self, node_id: str) -> list[str]:
        """Return the IDs of all immediate predecessors."""
        ...

    @abstractmethod
    def topological_order(self) -> list[str]:
        """Return node IDs in a valid topological order."""
        ...

    @abstractmethod
    def get_initial_nodes(self) -> list[str]:
        """Return the IDs of nodes with no predecessors."""
        ...

    @abstractmethod
    def get_newly_ready_nodes(self, completed: set[str]) -> list[str]:
        """Return node IDs whose dependencies are all in *completed*."""
        ...

    @abstractmethod
    def validate(self) -> None:
        """Validate the graph (acyclic, connected, edges valid). Raises on failure."""
        ...

    @abstractmethod
    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        ...


# ──────────────────────────────────────────────
# Orchestrator Protocol
# ──────────────────────────────────────────────


class OrchestratorProtocol(ABC):
    """Top-level contract for DAG orchestration."""

    @abstractmethod
    async def run(self, dag: DAGProtocol, data: Any) -> Any:
        """
        Execute a PlanDAG and return the resulting ExecutionDAG.
        """
        ...

    @abstractmethod
    def interrupt(self) -> None:
        """Request a graceful interrupt of the current run."""
        ...


# ──────────────────────────────────────────────
# Node Adapter Protocol
# ──────────────────────────────────────────────


class NodeAdapterProtocol(ABC):
    """Bridge between an L2 node and the L3 execution platform."""

    @abstractmethod
    async def execute(
        self,
        node: BaseNode,
        data: Any,
        configs: Any,
        platform: Any,
    ) -> NodeResult:
        """
        Convert the node into an L3 ExecutionEvent, execute via PolicyChain,
        and normalise the result into a NodeResult.
        """
        ...


# ──────────────────────────────────────────────
# Flow Handler Protocol
# ──────────────────────────────────────────────


class FlowHandlerProtocol(ABC):
    """Handler for a FlowNode — pure L2, never touches L3."""

    @abstractmethod
    async def handle(
        self,
        node: Any,
        data: Any,
        dag_state: Any,
    ) -> FlowResult:
        """
        Process a FlowNode and return a FlowResult describing
        how the DAG traversal should proceed.
        """
        ...


# ──────────────────────────────────────────────
# Validation Protocols
# ──────────────────────────────────────────────


class NodeValidatorProtocol(ABC):
    """A single pre-flight validation check."""

    @abstractmethod
    async def validate(
        self,
        node: BaseNode,
        data: Any,
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate a node before execution. Return ok or a failure reason."""
        ...


class ValidationPipelineProtocol(ABC):
    """Composable chain of NodeValidators, run sequentially with short-circuit."""

    @abstractmethod
    async def validate(
        self,
        node: BaseNode,
        data: Any,
        context: ValidationContext,
    ) -> ValidationResult:
        """Run all validators in order; return first failure or ok."""
        ...
