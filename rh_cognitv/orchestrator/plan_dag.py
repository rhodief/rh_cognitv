"""
PlanDAG — static intent graph + DAGBuilder fluent API.

The DAG class is a lightweight directed acyclic graph with adjacency-dict
internals and edge-list serialization.  PlanDAG is an immutable wrapper
that freezes the graph once built.  DAGBuilder provides an ergonomic
fluent interface for construction.
"""

from __future__ import annotations

from typing import Any

from .nodes import BaseNode


# ──────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────


class DAGError(Exception):
    """Base error for DAG operations."""


class CycleError(DAGError):
    """The graph contains a cycle."""


class DisconnectedError(DAGError):
    """The graph is not weakly connected."""


class DuplicateNodeError(DAGError):
    """A node with this ID already exists."""


class MissingNodeError(DAGError):
    """Referenced node ID does not exist in the graph."""


class DuplicateEdgeError(DAGError):
    """This edge already exists."""


class FrozenDAGError(DAGError):
    """Attempted mutation on a frozen PlanDAG."""


# ──────────────────────────────────────────────
# DAG — mutable graph
# ──────────────────────────────────────────────


class DAG:
    """
    Lightweight directed acyclic graph.

    Stores nodes and edges with forward/reverse adjacency dicts.
    No external dependencies (no networkx).
    """

    __slots__ = ("_nodes", "_forward", "_reverse")

    def __init__(self) -> None:
        self._nodes: dict[str, BaseNode] = {}
        self._forward: dict[str, list[str]] = {}  # node → successors
        self._reverse: dict[str, list[str]] = {}  # node → predecessors

    # ── Mutation ──

    def add_node(self, node_id: str, node: BaseNode) -> None:
        """Add a node. Raises DuplicateNodeError if ID already present."""
        if node_id in self._nodes:
            raise DuplicateNodeError(f"Node '{node_id}' already exists")
        self._nodes[node_id] = node
        self._forward.setdefault(node_id, [])
        self._reverse.setdefault(node_id, [])

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a directed edge. Both nodes must exist."""
        if from_id not in self._nodes:
            raise MissingNodeError(f"Source node '{from_id}' not in graph")
        if to_id not in self._nodes:
            raise MissingNodeError(f"Target node '{to_id}' not in graph")
        if from_id == to_id:
            raise CycleError(f"Self-loop on node '{from_id}'")
        if to_id in self._forward[from_id]:
            raise DuplicateEdgeError(f"Edge '{from_id}' → '{to_id}' already exists")
        self._forward[from_id].append(to_id)
        self._reverse[to_id].append(from_id)

    # ── Queries ──

    def get_node(self, node_id: str) -> BaseNode:
        """Retrieve a node by ID. Raises MissingNodeError if absent."""
        if node_id not in self._nodes:
            raise MissingNodeError(f"Node '{node_id}' not in graph")
        return self._nodes[node_id]

    def successors(self, node_id: str) -> list[str]:
        if node_id not in self._nodes:
            raise MissingNodeError(f"Node '{node_id}' not in graph")
        return list(self._forward[node_id])

    def predecessors(self, node_id: str) -> list[str]:
        if node_id not in self._nodes:
            raise MissingNodeError(f"Node '{node_id}' not in graph")
        return list(self._reverse[node_id])

    def get_initial_nodes(self) -> list[str]:
        """Return IDs of nodes with no predecessors (roots)."""
        return [nid for nid, preds in self._reverse.items() if not preds]

    def get_newly_ready_nodes(self, completed: set[str]) -> list[str]:
        """
        Return node IDs not yet in *completed* whose predecessors
        are all in *completed*.
        """
        ready: list[str] = []
        for nid in self._nodes:
            if nid in completed:
                continue
            if all(p in completed for p in self._reverse[nid]):
                ready.append(nid)
        return ready

    def topological_order(self) -> list[str]:
        """
        Kahn's algorithm.  Returns a valid topological ordering.
        Raises CycleError if the graph contains a cycle.
        """
        in_degree: dict[str, int] = {nid: len(preds) for nid, preds in self._reverse.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for succ in self._forward[nid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(order) != len(self._nodes):
            raise CycleError("Graph contains a cycle")
        return order

    def validate(self) -> None:
        """
        Validate the DAG:
        1. Acyclic (via topological sort)
        2. All edge refs point to existing nodes (guaranteed by add_edge)
        3. Weakly connected (single component when ignoring direction)
        """
        if not self._nodes:
            return  # empty graph is trivially valid

        # 1. Acyclic
        self.topological_order()

        # 2. Weakly connected — BFS over undirected view
        visited: set[str] = set()
        start = next(iter(self._nodes))
        stack = [start]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            for s in self._forward[nid]:
                if s not in visited:
                    stack.append(s)
            for p in self._reverse[nid]:
                if p not in visited:
                    stack.append(p)

        if len(visited) != len(self._nodes):
            raise DisconnectedError(
                f"Graph is not connected: {len(visited)} of {len(self._nodes)} nodes reachable"
            )

    # ── Serialization ──

    def to_edge_list(self) -> list[dict[str, str]]:
        """Return edges as a list of {from, to} dicts for serialization."""
        edges: list[dict[str, str]] = []
        for from_id, succs in self._forward.items():
            for to_id in succs:
                edges.append({"from": from_id, "to": to_id})
        return edges

    def node_count(self) -> int:
        return len(self._nodes)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())


# ──────────────────────────────────────────────
# PlanDAG — frozen wrapper
# ──────────────────────────────────────────────


class PlanDAG:
    """
    Immutable Plan DAG — frozen on creation.

    All read methods delegate to the inner DAG.
    Mutation methods raise FrozenDAGError.
    """

    __slots__ = ("_dag", "_name")

    def __init__(self, dag: DAG, name: str = "") -> None:
        # Validate before freezing
        dag.validate()
        object.__setattr__(self, "_dag", dag)
        object.__setattr__(self, "_name", name)

    @property
    def name(self) -> str:
        return self._name

    # ── Delegated read methods ──

    def get_node(self, node_id: str) -> BaseNode:
        return self._dag.get_node(node_id)

    def successors(self, node_id: str) -> list[str]:
        return self._dag.successors(node_id)

    def predecessors(self, node_id: str) -> list[str]:
        return self._dag.predecessors(node_id)

    def topological_order(self) -> list[str]:
        return self._dag.topological_order()

    def get_initial_nodes(self) -> list[str]:
        return self._dag.get_initial_nodes()

    def get_newly_ready_nodes(self, completed: set[str]) -> list[str]:
        return self._dag.get_newly_ready_nodes(completed)

    def to_edge_list(self) -> list[dict[str, str]]:
        return self._dag.to_edge_list()

    def node_count(self) -> int:
        return self._dag.node_count()

    def has_node(self, node_id: str) -> bool:
        return self._dag.has_node(node_id)

    def node_ids(self) -> list[str]:
        return self._dag.node_ids()

    def validate(self) -> None:
        self._dag.validate()


# ──────────────────────────────────────────────
# DAGBuilder — fluent API
# ──────────────────────────────────────────────


class DAGBuilder:
    """
    Ergonomic fluent builder for PlanDAGs.

    Usage::

        dag = (
            DAGBuilder("my-pipeline")
            .add_node("a", TextNode(prompt="Hello"))
            .add_node("b", FunctionNode(function_name="process"))
            .edge("a", "b")
            .build()
        )
    """

    def __init__(self, name: str = "") -> None:
        self._name = name
        self._dag = DAG()

    def add_node(self, node_id: str, node: BaseNode) -> DAGBuilder:
        """Add a node. Returns self for chaining."""
        self._dag.add_node(node_id, node)
        return self

    def edge(self, from_id: str, to_id: str) -> DAGBuilder:
        """Add an edge. Returns self for chaining."""
        self._dag.add_edge(from_id, to_id)
        return self

    def build(self) -> PlanDAG:
        """Validate and freeze the graph into a PlanDAG."""
        return PlanDAG(self._dag, name=self._name)
