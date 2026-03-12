"""
ExecutionDAG — append-only runtime recording of what actually happened.

Each entry records one node execution: the plan node it came from,
the result, timing, and the L3 ExecutionState snapshot version.
Entries are never removed — undo marks them as rolled_back.
"""

from __future__ import annotations

from rh_cognitv.execution_platform.types import now_timestamp

from .models import (
    ExecutionDAGEntry,
    NodeExecutionStatus,
    NodeResult,
)
from .nodes import BaseNode


class ExecutionDAG:
    """
    Append-only execution log.

    Records what actually happened during DAG traversal.
    Supports rollback marking and status-based queries.
    """

    __slots__ = ("_entries", "_by_node_id")

    def __init__(self) -> None:
        self._entries: list[ExecutionDAGEntry] = []
        self._by_node_id: dict[str, list[ExecutionDAGEntry]] = {}

    # ── Recording ──

    def record(
        self,
        node: BaseNode,
        result: NodeResult,
        *,
        state_version: int | None = None,
        parent_entry_id: str | None = None,
    ) -> ExecutionDAGEntry:
        """
        Append a completed execution entry.

        Returns the created entry.
        """
        status = NodeExecutionStatus.SUCCESS if result.ok else NodeExecutionStatus.FAILED
        entry = ExecutionDAGEntry(
            node_id=node.id,
            plan_node_ref=node.id,
            status=status,
            result=result,
            completed_at=now_timestamp(),
            state_version=state_version,
            parent_entry_id=parent_entry_id,
        )
        self._entries.append(entry)
        self._by_node_id.setdefault(node.id, []).append(entry)
        return entry

    def record_start(self, node: BaseNode) -> ExecutionDAGEntry:
        """Record that a node has started executing (RUNNING status)."""
        entry = ExecutionDAGEntry(
            node_id=node.id,
            plan_node_ref=node.id,
            status=NodeExecutionStatus.RUNNING,
        )
        self._entries.append(entry)
        self._by_node_id.setdefault(node.id, []).append(entry)
        return entry

    # ── Status mutations (append-only — never removes entries) ──

    def mark_rolled_back(self, from_entry: ExecutionDAGEntry) -> int:
        """
        Mark *from_entry* and all entries recorded after it as ROLLED_BACK.

        Returns the count of entries marked.
        """
        idx = self._find_entry_index(from_entry.id)
        count = 0
        for entry in self._entries[idx:]:
            if entry.status not in (
                NodeExecutionStatus.ROLLED_BACK,
                NodeExecutionStatus.SKIPPED,
            ):
                entry.status = NodeExecutionStatus.ROLLED_BACK
                count += 1
        return count

    def mark_waiting(self, node_id: str) -> ExecutionDAGEntry | None:
        """
        Mark the latest entry for *node_id* as WAITING (escalation in progress).

        Returns the updated entry, or None if no entry exists.
        """
        entries = self._by_node_id.get(node_id)
        if not entries:
            return None
        latest = entries[-1]
        latest.status = NodeExecutionStatus.WAITING
        return latest

    def mark_skipped(self, node: BaseNode) -> ExecutionDAGEntry:
        """Record a skipped node."""
        entry = ExecutionDAGEntry(
            node_id=node.id,
            plan_node_ref=node.id,
            status=NodeExecutionStatus.SKIPPED,
        )
        self._entries.append(entry)
        self._by_node_id.setdefault(node.id, []).append(entry)
        return entry

    # ── Queries ──

    def get_entry(self, node_id: str) -> ExecutionDAGEntry | None:
        """Return the latest entry for a given node_id, or None."""
        entries = self._by_node_id.get(node_id)
        if not entries:
            return None
        return entries[-1]

    def get_all_entries_for_node(self, node_id: str) -> list[ExecutionDAGEntry]:
        """Return all entries (including rolled-back) for a given node_id."""
        return list(self._by_node_id.get(node_id, []))

    def entries(self) -> list[ExecutionDAGEntry]:
        """Return all entries in insertion order."""
        return list(self._entries)

    def get_by_status(self, status: NodeExecutionStatus) -> list[ExecutionDAGEntry]:
        """Return all entries with the given status."""
        return [e for e in self._entries if e.status == status]

    def entry_count(self) -> int:
        return len(self._entries)

    def is_node_completed(self, node_id: str) -> bool:
        """Check if the latest entry for a node is SUCCESS."""
        entry = self.get_entry(node_id)
        return entry is not None and entry.status == NodeExecutionStatus.SUCCESS

    def completed_node_ids(self) -> set[str]:
        """Return the set of node_ids whose latest entry is SUCCESS."""
        result: set[str] = set()
        for node_id, entries in self._by_node_id.items():
            if entries and entries[-1].status == NodeExecutionStatus.SUCCESS:
                result.add(node_id)
        return result

    # ── Internal ──

    def _find_entry_index(self, entry_id: str) -> int:
        """Find the index of an entry by its unique ID."""
        for i, entry in enumerate(self._entries):
            if entry.id == entry_id:
                return i
        raise KeyError(f"Entry '{entry_id}' not found in ExecutionDAG")
