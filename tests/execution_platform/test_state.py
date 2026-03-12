"""Tests for state.py — ExecutionState snapshot chain, undo/redo, gc, escalation."""

import pytest

from rh_cognitv.execution_platform.models import EventStatus
from rh_cognitv.execution_platform.protocols import (
    JsonSnapshotSerializer,
    SnapshotSerializerProtocol,
)
from rh_cognitv.execution_platform.state import ExecutionState


# ──────────────────────────────────────────────
# Snapshot creation & restore
# ──────────────────────────────────────────────


class TestSnapshotBasics:
    def test_initial_state_is_empty(self):
        state = ExecutionState()
        assert state.get_current() == {}

    def test_initial_state_from_dict(self):
        state = ExecutionState({"key": "value"})
        assert state.get_current() == {"key": "value"}

    def test_snapshot_returns_version(self):
        state = ExecutionState()
        v = state.snapshot()
        assert v == 0

    def test_snapshot_versions_increment(self):
        state = ExecutionState()
        v0 = state.snapshot()
        v1 = state.snapshot()
        v2 = state.snapshot()
        assert v0 == 0
        assert v1 == 1
        assert v2 == 2

    def test_version_count(self):
        state = ExecutionState()
        assert state.version_count == 0
        state.snapshot()
        assert state.version_count == 1
        state.snapshot()
        assert state.version_count == 2

    def test_current_version_none_initially(self):
        state = ExecutionState()
        assert state.current_version is None

    def test_current_version_after_snapshot(self):
        state = ExecutionState()
        v = state.snapshot()
        assert state.current_version == v

    def test_versions_list(self):
        state = ExecutionState()
        state.snapshot()
        state.snapshot()
        state.snapshot()
        assert state.versions == [0, 1, 2]

    def test_snapshot_is_deep_copy(self):
        state = ExecutionState({"items": [1, 2, 3]})
        state.snapshot()

        # Mutate the list in current state
        state.get_current()  # this returns a copy
        state.update("items", [1, 2, 3, 4])
        state.snapshot()

        # Restore v0 — should have original list
        state.restore(0)
        assert state.get_current()["items"] == [1, 2, 3]

    def test_restore_to_specific_version(self):
        state = ExecutionState()
        state.update("step", 1)
        state.snapshot()

        state.update("step", 2)
        state.snapshot()

        state.update("step", 3)
        state.snapshot()

        state.restore(1)
        assert state.get_current()["step"] == 2

    def test_restore_nonexistent_version_raises(self):
        state = ExecutionState()
        state.snapshot()
        with pytest.raises(KeyError, match="Snapshot version 99"):
            state.restore(99)


# ──────────────────────────────────────────────
# State modification
# ──────────────────────────────────────────────


class TestStateModification:
    def test_update_key(self):
        state = ExecutionState()
        state.update("x", 42)
        assert state.get_current()["x"] == 42

    def test_merge_dict(self):
        state = ExecutionState({"a": 1})
        state.merge({"b": 2, "c": 3})
        current = state.get_current()
        assert current == {"a": 1, "b": 2, "c": 3}

    def test_merge_overwrites_existing(self):
        state = ExecutionState({"a": 1})
        state.merge({"a": 99})
        assert state.get_current()["a"] == 99

    def test_remove_key(self):
        state = ExecutionState({"a": 1, "b": 2})
        removed = state.remove("a")
        assert removed is True
        assert "a" not in state.get_current()

    def test_remove_nonexistent_key(self):
        state = ExecutionState()
        removed = state.remove("nope")
        assert removed is False

    def test_get_current_returns_deep_copy(self):
        state = ExecutionState({"items": [1, 2]})
        current = state.get_current()
        current["items"].append(3)
        # Original should be unchanged
        assert state.get_current()["items"] == [1, 2]


# ──────────────────────────────────────────────
# Undo / Redo
# ──────────────────────────────────────────────


class TestUndoRedo:
    def test_undo_no_snapshots(self):
        state = ExecutionState()
        assert state.undo() is False

    def test_undo_single_snapshot(self):
        state = ExecutionState()
        state.snapshot()
        assert state.undo() is False  # Can't go before first snapshot

    def test_undo_restores_previous(self):
        state = ExecutionState()
        state.update("v", "a")
        state.snapshot()

        state.update("v", "b")
        state.snapshot()

        result = state.undo()
        assert result is True
        assert state.get_current()["v"] == "a"

    def test_redo_after_undo(self):
        state = ExecutionState()
        state.update("v", "a")
        state.snapshot()

        state.update("v", "b")
        state.snapshot()

        state.undo()
        result = state.redo()
        assert result is True
        assert state.get_current()["v"] == "b"

    def test_redo_at_end(self):
        state = ExecutionState()
        state.snapshot()
        assert state.redo() is False

    def test_undo_redo_sequence(self):
        state = ExecutionState()

        for i in range(5):
            state.update("step", i)
            state.snapshot()

        # We're at step 4 (version 4)
        assert state.get_current()["step"] == 4

        # Undo twice → step 2
        state.undo()
        state.undo()
        assert state.get_current()["step"] == 2

        # Redo once → step 3
        state.redo()
        assert state.get_current()["step"] == 3

    def test_new_snapshot_after_undo_discards_redo_history(self):
        state = ExecutionState()

        state.update("v", "a")
        state.snapshot()

        state.update("v", "b")
        state.snapshot()

        state.update("v", "c")
        state.snapshot()

        # Undo to "a"
        state.undo()
        state.undo()
        assert state.get_current()["v"] == "a"

        # New snapshot from here discards b, c
        state.update("v", "x")
        state.snapshot()

        assert state.version_count == 2  # v0 ("a") + v3 ("x")
        assert state.redo() is False  # No redo available
        assert state.get_current()["v"] == "x"

    def test_state_isolation_between_snapshots(self):
        """Modifying state after snapshot doesn't affect the snapshot."""
        state = ExecutionState()
        state.update("data", {"nested": [1, 2]})
        v0 = state.snapshot()

        # Modify current state
        state.update("data", {"nested": [1, 2, 3]})
        v1 = state.snapshot()

        # Restore v0 — should have original nested data
        state.restore(v0)
        assert state.get_current()["data"]["nested"] == [1, 2]

        # Restore v1 — should have modified
        state.restore(v1)
        assert state.get_current()["data"]["nested"] == [1, 2, 3]


# ──────────────────────────────────────────────
# Level management
# ──────────────────────────────────────────────


class TestLevelManagement:
    def test_initial_level_is_zero(self):
        state = ExecutionState()
        assert state.level == 0

    def test_add_level(self):
        state = ExecutionState()
        state.add_level()
        assert state.level == 1
        state.add_level()
        assert state.level == 2

    def test_remove_level(self):
        state = ExecutionState()
        state.add_level()
        state.add_level()
        state.remove_level()
        assert state.level == 1

    def test_remove_level_at_zero_raises(self):
        state = ExecutionState()
        with pytest.raises(ValueError, match="already at level 0"):
            state.remove_level()

    def test_snapshot_captures_level(self):
        state = ExecutionState()
        state.add_level()
        state.snapshot()

        state.add_level()
        state.snapshot()

        # Verify levels are independent of snapshot
        # (level is metadata on _Snapshot, not in state dict)
        assert state.level == 2


# ──────────────────────────────────────────────
# GC (OQ-L3-02)
# ──────────────────────────────────────────────


class TestGCCollect:
    def test_gc_empty_state(self):
        state = ExecutionState()
        removed = state.gc_collect(keep_last=1)
        assert removed == 0

    def test_gc_no_args_removes_nothing(self):
        state = ExecutionState()
        for i in range(5):
            state.update("i", i)
            state.snapshot()
        removed = state.gc_collect()
        assert removed == 0
        assert state.version_count == 5

    def test_gc_keep_last(self):
        state = ExecutionState()
        for i in range(5):
            state.update("i", i)
            state.snapshot()

        removed = state.gc_collect(keep_last=2)
        assert removed == 3
        assert state.version_count == 2
        # Should have the last 2 snapshots (versions 3 and 4)
        assert state.versions == [3, 4]

    def test_gc_keep_first(self):
        state = ExecutionState()
        for i in range(5):
            state.update("i", i)
            state.snapshot()

        removed = state.gc_collect(keep_first=2)
        assert removed == 3
        assert state.version_count == 2
        assert state.versions == [0, 1]

    def test_gc_keep_first_and_last(self):
        state = ExecutionState()
        for i in range(10):
            state.update("i", i)
            state.snapshot()

        removed = state.gc_collect(keep_first=2, keep_last=2)
        assert removed == 6
        assert state.version_count == 4
        assert state.versions == [0, 1, 8, 9]

    def test_gc_keep_more_than_available(self):
        state = ExecutionState()
        state.snapshot()
        state.snapshot()
        removed = state.gc_collect(keep_last=10)
        assert removed == 0
        assert state.version_count == 2

    def test_gc_preserves_cursor_at_kept_version(self):
        state = ExecutionState()
        for i in range(5):
            state.update("i", i)
            state.snapshot()

        # Cursor is at version 4 (last)
        assert state.current_version == 4

        state.gc_collect(keep_last=2)
        # Cursor should still be at version 4
        assert state.current_version == 4
        assert state.get_current()["i"] == 4

    def test_gc_adjusts_cursor_when_removed(self):
        state = ExecutionState()
        for i in range(5):
            state.update("i", i)
            state.snapshot()

        # Move cursor to version 1 via undo
        state.undo()
        state.undo()
        state.undo()
        assert state.current_version == 1

        # Keep only last 2 (versions 3, 4) — cursor version 1 gets removed
        state.gc_collect(keep_last=2)
        # Cursor should move to last available
        assert state.current_version == 4

    def test_gc_restores_state_from_cursor(self):
        state = ExecutionState()
        for i in range(5):
            state.update("i", i)
            state.snapshot()

        state.gc_collect(keep_last=1)
        # State should match the last snapshot
        assert state.get_current()["i"] == 4

    def test_gc_keep_first_0(self):
        """keep_first=0 means keep none from front."""
        state = ExecutionState()
        for i in range(3):
            state.snapshot()
        removed = state.gc_collect(keep_first=0)
        assert removed == 3
        assert state.version_count == 0


# ──────────────────────────────────────────────
# Escalation (OQ-L3-05)
# ──────────────────────────────────────────────


class TestEscalation:
    def test_set_escalated(self):
        state = ExecutionState()
        state.update("task", "in-progress")
        version = state.set_escalated(
            event_id="evt-1",
            question="Should I proceed?",
            options=["yes", "no"],
            node_id="node-1",
            resume_data={"partial": "result"},
        )
        assert version >= 0

        esc = state.get_escalation()
        assert esc is not None
        assert esc["status"] == "escalated"
        assert esc["event_id"] == "evt-1"
        assert esc["question"] == "Should I proceed?"
        assert esc["options"] == ["yes", "no"]
        assert esc["node_id"] == "node-1"
        assert esc["resume_data"]["partial"] == "result"

    def test_get_escalation_when_not_escalated(self):
        state = ExecutionState()
        assert state.get_escalation() is None

    def test_clear_escalation(self):
        state = ExecutionState()
        state.set_escalated(event_id="evt-1", question="q")
        assert state.get_escalation() is not None

        state.clear_escalation()
        assert state.get_escalation() is None

    def test_escalation_persists_in_snapshot(self):
        state = ExecutionState()
        state.update("task", "running")
        state.set_escalated(event_id="evt-1", question="Approve?")

        # set_escalated already snapshots — restore to verify persistence
        version = state.current_version
        state.clear_escalation()
        assert state.get_escalation() is None

        state.restore(version)
        esc = state.get_escalation()
        assert esc is not None
        assert esc["event_id"] == "evt-1"

    def test_escalation_cloud_recovery_roundtrip(self):
        """Simulate cloud-safe recovery: serialize → deserialize → resume."""
        state = ExecutionState()
        state.update("progress", 50)
        state.set_escalated(
            event_id="evt-cloud",
            question="Deploy to prod?",
            options=["deploy", "cancel"],
            node_id="deploy-node",
        )

        # Serialize for persistence
        serialized = state.serialize_current()

        # New process recovers
        recovered = ExecutionState()
        recovered.deserialize_into(serialized)

        esc = recovered.get_escalation()
        assert esc is not None
        assert esc["question"] == "Deploy to prod?"
        assert esc["options"] == ["deploy", "cancel"]

        # Resume after human decision
        recovered.clear_escalation()
        recovered.update("progress", 100)
        assert recovered.get_escalation() is None
        assert recovered.get_current()["progress"] == 100


# ──────────────────────────────────────────────
# Serialization roundtrip
# ──────────────────────────────────────────────


class TestSerialization:
    def test_serialize_current(self):
        state = ExecutionState({"key": "value", "count": 42})
        data = state.serialize_current()
        assert isinstance(data, bytes)

    def test_deserialize_into(self):
        state = ExecutionState({"key": "value"})
        data = state.serialize_current()

        new_state = ExecutionState()
        new_state.deserialize_into(data)
        assert new_state.get_current() == {"key": "value"}

    def test_serialization_roundtrip_complex(self):
        state = ExecutionState({
            "events": [{"id": "e1", "status": "success"}],
            "nested": {"deep": {"value": [1, 2, 3]}},
            "count": 7,
        })
        data = state.serialize_current()

        recovered = ExecutionState()
        recovered.deserialize_into(data)
        assert recovered.get_current()["events"][0]["id"] == "e1"
        assert recovered.get_current()["nested"]["deep"]["value"] == [1, 2, 3]

    def test_custom_serializer(self):
        class ReverseSerializer(SnapshotSerializerProtocol):
            def __init__(self):
                self._inner = JsonSnapshotSerializer()

            def serialize(self, s: dict) -> bytes:
                return self._inner.serialize(s)[::-1]

            def deserialize(self, data: bytes) -> dict:
                return self._inner.deserialize(data[::-1])

        ser = ReverseSerializer()
        state = ExecutionState({"x": 1}, serializer=ser)
        data = state.serialize_current()
        # Reversed bytes should not be valid JSON
        assert data != b'{"x": 1}'

        recovered = ExecutionState(serializer=ser)
        recovered.deserialize_into(data)
        assert recovered.get_current()["x"] == 1

    def test_default_serializer_is_json(self):
        state = ExecutionState({"a": 1})
        data = state.serialize_current()
        import json
        parsed = json.loads(data)
        assert parsed["a"] == 1
