"""Tests for EntryRef[T] — typed lazy reference."""

import pytest

from rh_cognitv.execution_platform.models import (
    Artifact,
    ArtifactProvenance,
    ArtifactType,
    EntryContent,
    Memory,
    MemoryOrigin,
    MemoryRole,
    MemoryShape,
    Provenance,
    TimeInfo,
)
from rh_cognitv.execution_platform.types import EntryRef, now_timestamp


class FakeStore:
    """Minimal fake ContextStore for testing EntryRef.resolve()."""

    def __init__(self, entries: dict):
        self._entries = entries

    async def get(self, id: str):
        return self._entries.get(id)


def _make_memory(id: str = "mem-1") -> Memory:
    return Memory(
        id=id,
        content=EntryContent(text="test memory"),
        role=MemoryRole.EPISODIC,
        shape=MemoryShape.ATOM,
        provenance=Provenance(origin=MemoryOrigin.OBSERVED, source="test"),
        time=TimeInfo(recorded_at=now_timestamp(), observed_at=now_timestamp()),
    )


def _make_artifact(id: str = "art-1") -> Artifact:
    return Artifact(
        id=id,
        content=EntryContent(text="test artifact"),
        type=ArtifactType.CODE,
        slug="test-slug",
        provenance=ArtifactProvenance(intent="test"),
    )


class TestEntryRef:
    def test_creation(self):
        ref = EntryRef[Memory](id="mem-1", entry_type=Memory)
        assert ref.id == "mem-1"
        assert ref.entry_type is Memory

    def test_not_resolved_initially(self):
        ref = EntryRef[Memory](id="mem-1", entry_type=Memory)
        assert ref.is_resolved is False
        assert ref.value is None

    @pytest.mark.asyncio
    async def test_resolve_memory(self):
        mem = _make_memory("mem-1")
        store = FakeStore({"mem-1": mem})
        ref = EntryRef[Memory](id="mem-1", entry_type=Memory)

        result = await ref.resolve(store)
        assert result is mem
        assert ref.is_resolved is True
        assert ref.value is mem

    @pytest.mark.asyncio
    async def test_resolve_artifact(self):
        art = _make_artifact("art-1")
        store = FakeStore({"art-1": art})
        ref = EntryRef[Artifact](id="art-1", entry_type=Artifact)

        result = await ref.resolve(store)
        assert result is art
        assert ref.is_resolved is True

    @pytest.mark.asyncio
    async def test_resolve_caches_result(self):
        mem = _make_memory("mem-1")
        store = FakeStore({"mem-1": mem})
        ref = EntryRef[Memory](id="mem-1", entry_type=Memory)

        result1 = await ref.resolve(store)
        result2 = await ref.resolve(store)
        assert result1 is result2

    @pytest.mark.asyncio
    async def test_resolve_not_found_raises(self):
        store = FakeStore({})
        ref = EntryRef[Memory](id="missing", entry_type=Memory)

        with pytest.raises(LookupError, match="Entry not found"):
            await ref.resolve(store)

    @pytest.mark.asyncio
    async def test_resolve_wrong_type_raises(self):
        art = _make_artifact("art-1")
        store = FakeStore({"art-1": art})
        ref = EntryRef[Memory](id="art-1", entry_type=Memory)

        with pytest.raises(TypeError, match="Expected Memory, got Artifact"):
            await ref.resolve(store)

    @pytest.mark.asyncio
    async def test_resolve_uses_cached_even_if_store_changes(self):
        """Once resolved, the cached value is returned regardless of store changes."""
        mem = _make_memory("mem-1")
        store = FakeStore({"mem-1": mem})
        ref = EntryRef[Memory](id="mem-1", entry_type=Memory)

        await ref.resolve(store)

        # Now remove from store
        empty_store = FakeStore({})
        # Still returns cached value
        result = await ref.resolve(empty_store)
        assert result is mem
