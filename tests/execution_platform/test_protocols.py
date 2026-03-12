"""Tests for protocols.py — ABC contracts and JsonSnapshotSerializer."""

import json

from rh_cognitv.execution_platform.protocols import (
    BudgetTrackerProtocol,
    ContextStoreProtocol,
    EventBusProtocol,
    EventHandlerProtocol,
    ExecutionStateProtocol,
    HandlerRegistryProtocol,
    JsonSnapshotSerializer,
    LogCollectorProtocol,
    MiddlewareProtocol,
    PolicyChainProtocol,
    PolicyProtocol,
    SnapshotSerializerProtocol,
    TraceCollectorProtocol,
)

import pytest


# ──────────────────────────────────────────────
# JsonSnapshotSerializer
# ──────────────────────────────────────────────


class TestJsonSnapshotSerializer:
    def test_serialize_returns_bytes(self):
        s = JsonSnapshotSerializer()
        result = s.serialize({"key": "value"})
        assert isinstance(result, bytes)

    def test_deserialize_returns_dict(self):
        s = JsonSnapshotSerializer()
        data = s.serialize({"key": "value", "num": 42})
        result = s.deserialize(data)
        assert result == {"key": "value", "num": 42}

    def test_roundtrip_simple(self):
        s = JsonSnapshotSerializer()
        state = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        assert s.deserialize(s.serialize(state)) == state

    def test_roundtrip_complex(self):
        s = JsonSnapshotSerializer()
        state = {
            "execution_dag": {"nodes": ["n1", "n2"]},
            "status": "running",
            "level": 2,
            "refs": ["ref_01HX", "ref_02HY"],
        }
        assert s.deserialize(s.serialize(state)) == state

    def test_handles_non_serializable_with_default_str(self):
        """Non-JSON types should be handled via default=str."""
        from datetime import datetime, timezone

        s = JsonSnapshotSerializer()
        dt = datetime.now(timezone.utc)
        data = s.serialize({"ts": dt})
        result = s.deserialize(data)
        assert result["ts"] == str(dt)

    def test_implements_protocol(self):
        s = JsonSnapshotSerializer()
        assert isinstance(s, SnapshotSerializerProtocol)


# ──────────────────────────────────────────────
# ABCs cannot be instantiated
# ──────────────────────────────────────────────


class TestABCsAreAbstract:
    """Verify that all protocol ABCs cannot be instantiated directly."""

    def test_event_bus_is_abstract(self):
        with pytest.raises(TypeError):
            EventBusProtocol()

    def test_middleware_is_abstract(self):
        with pytest.raises(TypeError):
            MiddlewareProtocol()

    def test_event_handler_is_abstract(self):
        with pytest.raises(TypeError):
            EventHandlerProtocol()

    def test_handler_registry_is_abstract(self):
        with pytest.raises(TypeError):
            HandlerRegistryProtocol()

    def test_execution_state_is_abstract(self):
        with pytest.raises(TypeError):
            ExecutionStateProtocol()

    def test_context_store_is_abstract(self):
        with pytest.raises(TypeError):
            ContextStoreProtocol()

    def test_policy_is_abstract(self):
        with pytest.raises(TypeError):
            PolicyProtocol()

    def test_policy_chain_is_abstract(self):
        with pytest.raises(TypeError):
            PolicyChainProtocol()

    def test_budget_tracker_is_abstract(self):
        with pytest.raises(TypeError):
            BudgetTrackerProtocol()

    def test_log_collector_is_abstract(self):
        with pytest.raises(TypeError):
            LogCollectorProtocol()

    def test_trace_collector_is_abstract(self):
        with pytest.raises(TypeError):
            TraceCollectorProtocol()

    def test_snapshot_serializer_is_abstract(self):
        with pytest.raises(TypeError):
            SnapshotSerializerProtocol()
