"""Tests for MemoryStore — memory-specific storage logic."""

import pytest

from rh_cognitv.execution_platform.memory_store import MemoryStore
from rh_cognitv.execution_platform.models import (
    EntryContent,
    Memory,
    MemoryOrigin,
    MemoryQuery,
    MemoryRole,
    MemoryShape,
    Provenance,
    TimeInfo,
)


# ── Helpers ──


def _memory(**kw) -> Memory:
    defaults = dict(
        content=EntryContent(text="test memory"),
        role=MemoryRole.EPISODIC,
        shape=MemoryShape.ATOM,
        provenance=Provenance(origin=MemoryOrigin.OBSERVED, source="test"),
        time=TimeInfo(
            recorded_at="2024-01-01T00:00:00Z",
            observed_at="2024-01-01T00:00:00Z",
        ),
    )
    defaults.update(kw)
    return Memory(**defaults)


@pytest.fixture
def mem_store(tmp_path):
    return MemoryStore(tmp_path / "memories")


# ──────────────────────────────────────────────
# Save / Load roundtrip
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_and_get(mem_store):
    mem = _memory()
    mid = await mem_store.save(mem)
    loaded = await mem_store.get(mid)
    assert loaded is not None
    assert loaded.id == mem.id
    assert loaded.role == MemoryRole.EPISODIC
    assert loaded.content.text == "test memory"


@pytest.mark.asyncio
async def test_get_nonexistent(mem_store):
    assert await mem_store.get("NONEXISTENT0000000000000000") is None


@pytest.mark.asyncio
async def test_save_overwrites(mem_store):
    mem = _memory(content=EntryContent(text="v1"))
    mid = await mem_store.save(mem)
    updated = mem.model_copy(update={"content": EntryContent(text="v2")})
    await mem_store.save(updated)
    loaded = await mem_store.get(mid)
    assert loaded.content.text == "v2"


# ──────────────────────────────────────────────
# Serialization roundtrips per role
# ──────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("role", list(MemoryRole))
async def test_roundtrip_all_roles(mem_store, role):
    mem = _memory(role=role)
    mid = await mem_store.save(mem)
    loaded = await mem_store.get(mid)
    assert loaded.role == role


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", list(MemoryShape))
async def test_roundtrip_all_shapes(mem_store, shape):
    mem = _memory(shape=shape)
    mid = await mem_store.save(mem)
    loaded = await mem_store.get(mid)
    assert loaded.shape == shape


@pytest.mark.asyncio
async def test_roundtrip_with_tags_and_ext(mem_store):
    mem = _memory(tags=["a", "b"], ext={"custom": True})
    mid = await mem_store.save(mem)
    loaded = await mem_store.get(mid)
    assert loaded.tags == ["a", "b"]
    assert loaded.ext == {"custom": True}


@pytest.mark.asyncio
async def test_roundtrip_content_formats(mem_store):
    """Different content formats are preserved through serialization."""
    for fmt in ["text/plain", "text/markdown", "data/json"]:
        mem = _memory(content=EntryContent(text="hello", format=fmt, summary="short"))
        mid = await mem_store.save(mem)
        loaded = await mem_store.get(mid)
        assert loaded.content.format == fmt
        assert loaded.content.summary == "short"


# ──────────────────────────────────────────────
# Search / filtering
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_by_role(mem_store):
    await mem_store.save(_memory(role=MemoryRole.EPISODIC))
    await mem_store.save(_memory(role=MemoryRole.SEMANTIC))
    await mem_store.save(_memory(role=MemoryRole.WORKING))
    results = await mem_store.search(MemoryQuery(role=MemoryRole.SEMANTIC))
    assert len(results) == 1
    assert results[0].entry.role == MemoryRole.SEMANTIC


@pytest.mark.asyncio
async def test_search_by_tags(mem_store):
    await mem_store.save(_memory(tags=["debug", "important"]))
    await mem_store.save(_memory(tags=["trivial"]))
    results = await mem_store.search(MemoryQuery(tags=["debug"]))
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_by_tags_multiple(mem_store):
    """All specified tags must be present."""
    await mem_store.save(_memory(tags=["a", "b", "c"]))
    await mem_store.save(_memory(tags=["a", "c"]))
    results = await mem_store.search(MemoryQuery(tags=["a", "b"]))
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_by_text(mem_store):
    await mem_store.save(_memory(content=EntryContent(text="the quick brown fox")))
    await mem_store.save(_memory(content=EntryContent(text="lazy dog")))
    results = await mem_store.search(MemoryQuery(text="fox"))
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_text_case_insensitive(mem_store):
    await mem_store.save(_memory(content=EntryContent(text="Hello World")))
    results = await mem_store.search(MemoryQuery(text="hello"))
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_empty_query_returns_all(mem_store):
    await mem_store.save(_memory())
    await mem_store.save(_memory())
    results = await mem_store.search(MemoryQuery())
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_artifact_type_excludes_memories(mem_store):
    """Setting artifact_type should exclude all memories."""
    await mem_store.save(_memory())
    from rh_cognitv.execution_platform.models import ArtifactType

    results = await mem_store.search(MemoryQuery(artifact_type=ArtifactType.CODE))
    assert len(results) == 0


# ──────────────────────────────────────────────
# Soft delete / forget
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forget_hides_from_get(mem_store):
    mem = _memory()
    mid = await mem_store.save(mem)
    assert await mem_store.forget(mid)
    assert await mem_store.get(mid) is None


@pytest.mark.asyncio
async def test_forget_hides_from_search(mem_store):
    mem = _memory()
    mid = await mem_store.save(mem)
    await mem_store.forget(mid)
    results = await mem_store.search(MemoryQuery())
    assert len(results) == 0


@pytest.mark.asyncio
async def test_forget_nonexistent_returns_false(mem_store):
    assert not await mem_store.forget("NOTFOUND00000000000000000000")


@pytest.mark.asyncio
async def test_forget_already_forgotten_returns_false(mem_store):
    mem = _memory()
    mid = await mem_store.save(mem)
    assert await mem_store.forget(mid)
    assert not await mem_store.forget(mid)


# ──────────────────────────────────────────────
# Consolidation
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consolidate_removes_forgotten(mem_store):
    mem = _memory()
    mid = await mem_store.save(mem)
    await mem_store.forget(mid)
    await mem_store.consolidate()
    # File should be gone
    assert not (mem_store._dir / f"{mid}.json").exists()


@pytest.mark.asyncio
async def test_consolidate_preserves_active(mem_store):
    mem = _memory()
    mid = await mem_store.save(mem)
    await mem_store.consolidate()
    loaded = await mem_store.get(mid)
    assert loaded is not None


# ──────────────────────────────────────────────
# Path safety
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_id_rejected(mem_store):
    with pytest.raises(ValueError, match="Invalid entry ID"):
        await mem_store.get("../../etc/passwd")


@pytest.mark.asyncio
async def test_empty_id_rejected(mem_store):
    with pytest.raises(ValueError, match="Invalid entry ID"):
        await mem_store.get("")
