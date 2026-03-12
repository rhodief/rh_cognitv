"""
ExecutionState — Immutable snapshot chain for time-travel state management.

DD-L3-02: Full deep-copy snapshot on every state transition.
OQ-L3-02: Keep all snapshots + explicit gc_collect() for manual pruning.
OQ-L3-03: Accepts SnapshotSerializer (default: JsonSnapshotSerializer).
OQ-L3-05: Snapshots capture ESCALATED status + escalation context.
"""

from __future__ import annotations

import copy
from typing import Any

from .models import EventStatus
from .protocols import (
    ExecutionStateProtocol,
    JsonSnapshotSerializer,
    SnapshotSerializerProtocol,
)
from .types import now_timestamp


class _Snapshot:
    """Internal snapshot record."""

    __slots__ = ("version", "data", "timestamp", "level")

    def __init__(self, version: int, data: dict, timestamp: str, level: int) -> None:
        self.version = version
        self.data = data
        self.timestamp = timestamp
        self.level = level


class ExecutionState(ExecutionStateProtocol):
    """Full-snapshot time-travel state manager.

    Each call to `snapshot()` deep-copies the entire current state.
    Random access to any point in time via `restore(version)`.
    Undo/redo navigates the snapshot chain.

    Args:
        initial_state: Optional starting state dict. Defaults to empty.
        serializer: Snapshot serializer for persistence. Default: JsonSnapshotSerializer.
    """

    def __init__(
        self,
        initial_state: dict[str, Any] | None = None,
        *,
        serializer: SnapshotSerializerProtocol | None = None,
    ) -> None:
        self._state: dict[str, Any] = dict(initial_state) if initial_state else {}
        self._serializer = serializer or JsonSnapshotSerializer()

        self._snapshots: list[_Snapshot] = []
        self._next_version: int = 0
        self._cursor: int = -1  # index into _snapshots for undo/redo
        self._level: int = 0

    # ──────────────────────────────────────────
    # State access
    # ──────────────────────────────────────────

    def get_current(self) -> dict:
        """Get a deep copy of the current state."""
        return copy.deepcopy(self._state)

    def update(self, key: str, value: Any) -> None:
        """Update a single key in the current state.

        Does NOT automatically snapshot — call snapshot() explicitly.
        """
        self._state[key] = value

    def merge(self, data: dict[str, Any]) -> None:
        """Merge a dict into the current state.

        Does NOT automatically snapshot — call snapshot() explicitly.
        """
        self._state.update(data)

    def remove(self, key: str) -> bool:
        """Remove a key from the current state. Returns True if key existed."""
        if key in self._state:
            del self._state[key]
            return True
        return False

    # ──────────────────────────────────────────
    # Snapshot management
    # ──────────────────────────────────────────

    def snapshot(self) -> int:
        """Take a deep-copy snapshot of the current state.

        Returns the snapshot version number. After taking a snapshot,
        any forward redo history (snapshots after cursor) is discarded.
        """
        version = self._next_version
        self._next_version += 1

        snap = _Snapshot(
            version=version,
            data=copy.deepcopy(self._state),
            timestamp=now_timestamp(),
            level=self._level,
        )

        # Discard any redo history beyond current cursor
        if self._cursor < len(self._snapshots) - 1:
            self._snapshots = self._snapshots[: self._cursor + 1]

        self._snapshots.append(snap)
        self._cursor = len(self._snapshots) - 1

        return version

    def restore(self, version: int) -> None:
        """Restore state to a specific snapshot version.

        Raises:
            KeyError: If the version does not exist.
        """
        for i, snap in enumerate(self._snapshots):
            if snap.version == version:
                self._state = copy.deepcopy(snap.data)
                self._cursor = i
                return
        raise KeyError(f"Snapshot version {version} not found")

    @property
    def version_count(self) -> int:
        """Number of snapshots currently stored."""
        return len(self._snapshots)

    @property
    def current_version(self) -> int | None:
        """Version number at the current cursor position, or None if no snapshots."""
        if self._cursor < 0 or not self._snapshots:
            return None
        return self._snapshots[self._cursor].version

    @property
    def versions(self) -> list[int]:
        """List of all snapshot version numbers."""
        return [s.version for s in self._snapshots]

    # ──────────────────────────────────────────
    # Undo / Redo
    # ──────────────────────────────────────────

    def undo(self) -> bool:
        """Move to the previous snapshot. Returns True if successful."""
        if self._cursor <= 0:
            return False
        self._cursor -= 1
        self._state = copy.deepcopy(self._snapshots[self._cursor].data)
        return True

    def redo(self) -> bool:
        """Move to the next snapshot. Returns True if successful."""
        if self._cursor >= len(self._snapshots) - 1:
            return False
        self._cursor += 1
        self._state = copy.deepcopy(self._snapshots[self._cursor].data)
        return True

    # ──────────────────────────────────────────
    # Execution nesting levels
    # ──────────────────────────────────────────

    @property
    def level(self) -> int:
        """Current execution nesting level."""
        return self._level

    def add_level(self) -> None:
        """Increment execution nesting level."""
        self._level += 1

    def remove_level(self) -> None:
        """Decrement execution nesting level.

        Raises:
            ValueError: If level would go below 0.
        """
        if self._level <= 0:
            raise ValueError("Cannot remove level: already at level 0")
        self._level -= 1

    # ──────────────────────────────────────────
    # Escalation support (OQ-L3-05)
    # ──────────────────────────────────────────

    def set_escalated(
        self,
        event_id: str,
        question: str,
        options: list[str] | None = None,
        node_id: str | None = None,
        resume_data: dict[str, Any] | None = None,
    ) -> int:
        """Mark the state as ESCALATED and snapshot for cloud recovery.

        Stores escalation context in the state so a new process can
        recover and resume after human decision.

        Returns the snapshot version.
        """
        self._state["_escalation"] = {
            "status": EventStatus.ESCALATED.value,
            "event_id": event_id,
            "question": question,
            "options": options or [],
            "node_id": node_id,
            "resume_data": resume_data or {},
        }
        return self.snapshot()

    def get_escalation(self) -> dict[str, Any] | None:
        """Get escalation context if the state is ESCALATED, else None."""
        esc = self._state.get("_escalation")
        if esc and esc.get("status") == EventStatus.ESCALATED.value:
            return dict(esc)
        return None

    def clear_escalation(self) -> None:
        """Clear the escalation context (after human decision arrives)."""
        self._state.pop("_escalation", None)

    # ──────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────

    def serialize_current(self) -> bytes:
        """Serialize the current state using the configured serializer."""
        return self._serializer.serialize(self._state)

    def deserialize_into(self, data: bytes) -> None:
        """Deserialize bytes and replace the current state."""
        self._state = self._serializer.deserialize(data)

    # ──────────────────────────────────────────
    # Garbage Collection (OQ-L3-02)
    # ──────────────────────────────────────────

    def gc_collect(
        self, *, keep_first: int | None = None, keep_last: int | None = None
    ) -> int:
        """Manually collect old snapshots.

        Args:
            keep_first: Keep the first N snapshots (preserves early history).
            keep_last: Keep the last N snapshots (preserves recent undo ability).

        If both are specified, snapshots in either range are kept.
        The cursor is adjusted to remain valid.

        Returns:
            Count of removed snapshots.
        """
        total = len(self._snapshots)
        if total == 0:
            return 0

        first_n = keep_first if keep_first is not None else 0
        last_n = keep_last if keep_last is not None else 0

        # If neither specified, remove nothing
        if keep_first is None and keep_last is None:
            return 0

        keep_indices: set[int] = set()

        if keep_first is not None:
            for i in range(min(first_n, total)):
                keep_indices.add(i)

        if keep_last is not None:
            for i in range(max(0, total - last_n), total):
                keep_indices.add(i)

        if len(keep_indices) == total:
            return 0

        # Build new snapshot list preserving order
        old_cursor_version = (
            self._snapshots[self._cursor].version
            if 0 <= self._cursor < total
            else None
        )

        new_snapshots = [self._snapshots[i] for i in sorted(keep_indices)]
        removed = total - len(new_snapshots)

        self._snapshots = new_snapshots

        # Adjust cursor
        if old_cursor_version is not None:
            # Try to keep cursor at same version
            for i, snap in enumerate(self._snapshots):
                if snap.version == old_cursor_version:
                    self._cursor = i
                    break
            else:
                # Cursor version was removed — move to last snapshot
                self._cursor = len(self._snapshots) - 1 if self._snapshots else -1
        else:
            self._cursor = len(self._snapshots) - 1 if self._snapshots else -1

        # Restore state from current cursor position
        if self._cursor >= 0:
            self._state = copy.deepcopy(self._snapshots[self._cursor].data)

        return removed
