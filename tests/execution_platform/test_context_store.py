"""Tests for ContextStore — unified store facade."""

import pytest

from rh_cognitv.execution_platform.context_store import ContextStore
from rh_cognitv.execution_platform.models import (
    Artifact,
    ArtifactProvenance,
    ArtifactStatus,
    ArtifactType,
    EntryContent,
    Memory,
    MemoryOrigin,
    MemoryQuery,
    MemoryRole,
    MemoryShape,
    Provenance,
    TimeInfo,
)
from rh_cognitv.execution_platform.types import EntryRef


# ── Helpers ──


def _memory(**kw) -> Memory:
    defaults = dict(
        content=EntryContent(text="observed event"),
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


def _artifact(**kw) -> Artifact:
    defaults = dict(
        content=EntryContent(text="def hello(): pass"),
        type=ArtifactType.CODE,
        slug="hello-fn",
        provenance=ArtifactProvenance(intent="testing"),
    )
    defaults.update(kw)
    return Artifact(**defaults)


@pytest.fixture
def store(tmp_path):
    return ContextStore(tmp_path / "store")


# ──────────────────────────────────────────────
# CRUD basics
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remember_and_get(store):
    mem = _memory()
    mid = await store.remember(mem)
    loaded = await store.get(mid)
    assert loaded is not None
    assert loaded.id == mem.id
    assert loaded.kind == "memory"
    assert loaded.content.text == "observed event"


@pytest.mark.asyncio
async def test_store_and_get_artifact(store):
    art = _artifact()
    aid = await store.store(art)
    loaded = await store.get(aid)
    assert loaded is not None
    assert loaded.id == art.id
    assert loaded.kind == "artifact"
    assert loaded.content.text == "def hello(): pass"


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none(store):
    result = await store.get("NONEXISTENT0000000000000000")
    assert result is None


# ──────────────────────────────────────────────
# recall / query
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_all(store):
    await store.remember(_memory())
    await store.store(_artifact())
    results = await store.recall(MemoryQuery())
    assert len(results) == 2


@pytest.mark.asyncio
async def test_recall_kind_memory(store):
    await store.remember(_memory())
    await store.store(_artifact())
    results = await store.recall(MemoryQuery(kind="memory"))
    assert len(results) == 1
    assert results[0].entry.kind == "memory"


@pytest.mark.asyncio
async def test_recall_kind_artifact(store):
    await store.remember(_memory())
    await store.store(_artifact())
    results = await store.recall(MemoryQuery(kind="artifact"))
    assert len(results) == 1
    assert results[0].entry.kind == "artifact"


@pytest.mark.asyncio
async def test_recall_by_role(store):
    await store.remember(_memory(role=MemoryRole.EPISODIC))
    await store.remember(_memory(role=MemoryRole.SEMANTIC))
    results = await store.recall(MemoryQuery(role=MemoryRole.SEMANTIC))
    assert len(results) == 1
    assert results[0].entry.role == MemoryRole.SEMANTIC


@pytest.mark.asyncio
async def test_recall_by_tags(store):
    await store.remember(_memory(tags=["important", "debug"]))
    await store.remember(_memory(tags=["trivial"]))
    results = await store.recall(MemoryQuery(kind="memory", tags=["important"]))
    assert len(results) == 1
    assert "important" in results[0].entry.tags


@pytest.mark.asyncio
async def test_recall_by_text(store):
    await store.remember(_memory(content=EntryContent(text="the cat sat on the mat")))
    await store.remember(_memory(content=EntryContent(text="the dog ran")))
    results = await store.recall(MemoryQuery(kind="memory", text="cat"))
    assert len(results) == 1
    assert "cat" in results[0].entry.content.text


@pytest.mark.asyncio
async def test_recall_by_artifact_type(store):
    await store.store(_artifact(type=ArtifactType.CODE, slug="a"))
    await store.store(_artifact(type=ArtifactType.DOCUMENT, slug="b"))
    results = await store.recall(MemoryQuery(artifact_type=ArtifactType.DOCUMENT))
    assert len(results) == 1
    assert results[0].entry.type == ArtifactType.DOCUMENT


@pytest.mark.asyncio
async def test_recall_top_k(store):
    for i in range(5):
        await store.remember(
            _memory(content=EntryContent(text=f"memory {i}"))
        )
    results = await store.recall(MemoryQuery(top_k=3))
    assert len(results) == 3


@pytest.mark.asyncio
async def test_recall_role_filter_excludes_artifacts(store):
    """Setting role filter should not return artifacts."""
    await store.store(_artifact())
    await store.remember(_memory(role=MemoryRole.EPISODIC))
    results = await store.recall(MemoryQuery(role=MemoryRole.EPISODIC))
    assert len(results) == 1
    assert results[0].entry.kind == "memory"


@pytest.mark.asyncio
async def test_recall_artifact_type_filter_excludes_memories(store):
    """Setting artifact_type filter should not return memories."""
    await store.remember(_memory())
    await store.store(_artifact(type=ArtifactType.CODE, slug="x"))
    results = await store.recall(MemoryQuery(artifact_type=ArtifactType.CODE))
    assert len(results) == 1
    assert results[0].entry.kind == "artifact"


# ──────────────────────────────────────────────
# get_artifact by slug
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_artifact_by_slug(store):
    art = _artifact(slug="my-tool")
    await store.store(art)
    loaded = await store.get_artifact("my-tool")
    assert loaded is not None
    assert loaded.slug == "my-tool"


@pytest.mark.asyncio
async def test_get_artifact_by_slug_not_found(store):
    result = await store.get_artifact("no-such-slug")
    assert result is None


# ──────────────────────────────────────────────
# forget / soft delete
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forget_memory(store):
    mem = _memory()
    mid = await store.remember(mem)
    await store.forget(mid)
    assert await store.get(mid) is None


@pytest.mark.asyncio
async def test_forget_artifact(store):
    art = _artifact()
    aid = await store.store(art)
    await store.forget(aid)
    assert await store.get(aid) is None


@pytest.mark.asyncio
async def test_forgotten_entries_excluded_from_recall(store):
    mem = _memory()
    mid = await store.remember(mem)
    await store.forget(mid)
    results = await store.recall(MemoryQuery(kind="memory"))
    assert len(results) == 0


# ──────────────────────────────────────────────
# consolidate
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consolidate_removes_forgotten(store):
    mem = _memory()
    mid = await store.remember(mem)
    await store.forget(mid)
    assert (store.base_dir / "memories" / f"{mid}.json").exists()
    await store.consolidate()
    assert not (store.base_dir / "memories" / f"{mid}.json").exists()


@pytest.mark.asyncio
async def test_consolidate_preserves_active(store):
    mem = _memory()
    mid = await store.remember(mem)
    await store.consolidate()
    loaded = await store.get(mid)
    assert loaded is not None


# ──────────────────────────────────────────────
# EntryRef integration
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_entry_ref_resolve_memory(store):
    mem = _memory()
    mid = await store.remember(mem)
    ref = EntryRef(id=mid, entry_type=Memory)
    resolved = await ref.resolve(store)
    assert isinstance(resolved, Memory)
    assert resolved.id == mid


@pytest.mark.asyncio
async def test_entry_ref_resolve_artifact(store):
    art = _artifact()
    aid = await store.store(art)
    ref = EntryRef(id=aid, entry_type=Artifact)
    resolved = await ref.resolve(store)
    assert isinstance(resolved, Artifact)
    assert resolved.id == aid


@pytest.mark.asyncio
async def test_entry_ref_resolve_not_found(store):
    ref = EntryRef(id="NONEXISTENT0000000000000000", entry_type=Memory)
    with pytest.raises(LookupError):
        await ref.resolve(store)
