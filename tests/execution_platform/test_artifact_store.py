"""Tests for ArtifactStore — artifact-specific storage logic."""

import pytest

from rh_cognitv.execution_platform.artifact_store import ArtifactStore
from rh_cognitv.execution_platform.models import (
    Artifact,
    ArtifactProvenance,
    ArtifactStatus,
    ArtifactType,
    EntryContent,
    MemoryQuery,
    MemoryRole,
)


# ── Helpers ──


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
def art_store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


# ──────────────────────────────────────────────
# Save / Load roundtrip
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_and_get(art_store):
    art = _artifact()
    aid = await art_store.save(art)
    loaded = await art_store.get(aid)
    assert loaded is not None
    assert loaded.id == art.id
    assert loaded.slug == "hello-fn"
    assert loaded.type == ArtifactType.CODE


@pytest.mark.asyncio
async def test_get_nonexistent(art_store):
    assert await art_store.get("NONEXISTENT0000000000000000") is None


# ──────────────────────────────────────────────
# Format-diverse serialization roundtrips
# ──────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fmt,text",
    [
        ("text/plain", "Hello, world!"),
        ("text/markdown", "# Title\n\nParagraph."),
        ("code/python", "def greet(name):\n    return f'Hello {name}'"),
        ("data/json", '{"key": "value", "nums": [1, 2, 3]}'),
        (None, "no format specified"),
    ],
)
async def test_format_roundtrip(art_store, fmt, text):
    art = _artifact(
        content=EntryContent(text=text, format=fmt),
        slug=f"fmt-{fmt or 'none'}",
    )
    aid = await art_store.save(art)
    loaded = await art_store.get(aid)
    assert loaded.content.text == text
    assert loaded.content.format == fmt


@pytest.mark.asyncio
async def test_roundtrip_with_summary(art_store):
    art = _artifact(
        content=EntryContent(text="full content", summary="short", format="text/plain")
    )
    aid = await art_store.save(art)
    loaded = await art_store.get(aid)
    assert loaded.content.summary == "short"


@pytest.mark.asyncio
@pytest.mark.parametrize("atype", list(ArtifactType))
async def test_roundtrip_all_types(art_store, atype):
    art = _artifact(type=atype, slug=f"slug-{atype.value}")
    aid = await art_store.save(art)
    loaded = await art_store.get(aid)
    assert loaded.type == atype


@pytest.mark.asyncio
async def test_roundtrip_with_tags_and_ext(art_store):
    art = _artifact(tags=["v1", "release"], ext={"ci": True})
    aid = await art_store.save(art)
    loaded = await art_store.get(aid)
    assert loaded.tags == ["v1", "release"]
    assert loaded.ext == {"ci": True}


# ──────────────────────────────────────────────
# Slug-based retrieval
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_by_slug(art_store):
    art = _artifact(slug="my-tool")
    await art_store.save(art)
    loaded = await art_store.get_by_slug("my-tool")
    assert loaded is not None
    assert loaded.slug == "my-tool"


@pytest.mark.asyncio
async def test_get_by_slug_not_found(art_store):
    assert await art_store.get_by_slug("no-slug") is None


@pytest.mark.asyncio
async def test_get_by_slug_version(art_store):
    """Fetch a specific version by slug+version."""
    v1 = _artifact(slug="tool", version=1)
    v2 = _artifact(slug="tool", version=2, supersedes=v1.id)
    await art_store.save(v1)
    await art_store.save(v2)
    loaded = await art_store.get_by_slug("tool", version=1)
    assert loaded is not None
    assert loaded.version == 1


@pytest.mark.asyncio
async def test_get_by_slug_version_not_found(art_store):
    art = _artifact(slug="tool")
    await art_store.save(art)
    assert await art_store.get_by_slug("tool", version=99) is None


@pytest.mark.asyncio
async def test_get_by_slug_prefers_active(art_store):
    """When multiple versions exist, prefer the latest active one."""
    v1 = _artifact(slug="tool", version=1, status=ArtifactStatus.DEPRECATED)
    v2 = _artifact(slug="tool", version=2, status=ArtifactStatus.ACTIVE)
    v3 = _artifact(slug="tool", version=3, status=ArtifactStatus.DRAFT)
    await art_store.save(v1)
    await art_store.save(v2)
    await art_store.save(v3)
    loaded = await art_store.get_by_slug("tool")
    assert loaded.version == 2


@pytest.mark.asyncio
async def test_get_by_slug_fallback_latest(art_store):
    """When no active version, return latest regardless of status."""
    v1 = _artifact(slug="tool", version=1, status=ArtifactStatus.DEPRECATED)
    v2 = _artifact(slug="tool", version=2, status=ArtifactStatus.DRAFT)
    await art_store.save(v1)
    await art_store.save(v2)
    loaded = await art_store.get_by_slug("tool")
    assert loaded.version == 2


# ──────────────────────────────────────────────
# Auto-versioning (supersedes chain)
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_version_on_duplicate_slug(art_store):
    v1 = _artifact(slug="my-code")
    await art_store.save(v1)

    v2 = _artifact(slug="my-code", content=EntryContent(text="updated code"))
    v2_id = await art_store.save(v2)

    loaded_v2 = await art_store.get(v2_id)
    assert loaded_v2.version == 2
    assert loaded_v2.supersedes == v1.id

    # Old version should be deprecated
    loaded_v1 = await art_store.get(v1.id)
    assert loaded_v1.status == ArtifactStatus.DEPRECATED


@pytest.mark.asyncio
async def test_auto_version_three_versions(art_store):
    v1 = _artifact(slug="evolving")
    await art_store.save(v1)

    v2 = _artifact(slug="evolving", content=EntryContent(text="v2"))
    v2_id = await art_store.save(v2)

    v3 = _artifact(slug="evolving", content=EntryContent(text="v3"))
    v3_id = await art_store.save(v3)

    loaded_v3 = await art_store.get(v3_id)
    assert loaded_v3.version == 3

    # Only latest should be active
    latest = await art_store.get_by_slug("evolving")
    assert latest.version == 3
    assert latest.status == ArtifactStatus.ACTIVE


@pytest.mark.asyncio
async def test_explicit_version_not_auto_incremented(art_store):
    """When version > 1 or supersedes is set, skip auto-versioning."""
    v1 = _artifact(slug="manual", version=1)
    await art_store.save(v1)

    v5 = _artifact(slug="manual", version=5, supersedes=v1.id)
    v5_id = await art_store.save(v5)
    loaded = await art_store.get(v5_id)
    assert loaded.version == 5


# ──────────────────────────────────────────────
# Search / filtering
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_by_artifact_type(art_store):
    await art_store.save(_artifact(type=ArtifactType.CODE, slug="a"))
    await art_store.save(_artifact(type=ArtifactType.DOCUMENT, slug="b"))
    results = await art_store.search(MemoryQuery(artifact_type=ArtifactType.DOCUMENT))
    assert len(results) == 1
    assert results[0].entry.type == ArtifactType.DOCUMENT


@pytest.mark.asyncio
async def test_search_by_tags(art_store):
    await art_store.save(_artifact(tags=["prod", "v1"], slug="a"))
    await art_store.save(_artifact(tags=["dev"], slug="b"))
    results = await art_store.search(MemoryQuery(tags=["prod"]))
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_by_text(art_store):
    await art_store.save(
        _artifact(content=EntryContent(text="sorting algorithm"), slug="a")
    )
    await art_store.save(
        _artifact(content=EntryContent(text="web server"), slug="b")
    )
    results = await art_store.search(MemoryQuery(text="sorting"))
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_empty_query(art_store):
    await art_store.save(_artifact(slug="a"))
    await art_store.save(_artifact(slug="b"))
    results = await art_store.search(MemoryQuery())
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_role_filter_excludes_artifacts(art_store):
    """Setting role filter should exclude all artifacts."""
    await art_store.save(_artifact(slug="x"))
    results = await art_store.search(MemoryQuery(role=MemoryRole.EPISODIC))
    assert len(results) == 0


# ──────────────────────────────────────────────
# Soft delete
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forget_hides_from_get(art_store):
    art = _artifact()
    aid = await art_store.save(art)
    assert await art_store.forget(aid)
    assert await art_store.get(aid) is None


@pytest.mark.asyncio
async def test_forget_archives_artifact(art_store):
    """Forgotten artifacts are archived on disk."""
    import json

    art = _artifact(status=ArtifactStatus.ACTIVE)
    aid = await art_store.save(art)
    await art_store.forget(aid)
    # Read raw file to check status
    raw = json.loads((art_store._dir / f"{aid}.json").read_text())
    assert raw["entry"]["status"] == "archived"


@pytest.mark.asyncio
async def test_forget_hides_from_search(art_store):
    art = _artifact()
    await art_store.save(art)
    await art_store.forget(art.id)
    results = await art_store.search(MemoryQuery())
    assert len(results) == 0


@pytest.mark.asyncio
async def test_forget_hides_from_slug_lookup(art_store):
    art = _artifact(slug="doomed")
    await art_store.save(art)
    await art_store.forget(art.id)
    assert await art_store.get_by_slug("doomed") is None


@pytest.mark.asyncio
async def test_forget_nonexistent_returns_false(art_store):
    assert not await art_store.forget("NOTFOUND00000000000000000000")


@pytest.mark.asyncio
async def test_forget_already_forgotten_returns_false(art_store):
    art = _artifact()
    await art_store.save(art)
    assert await art_store.forget(art.id)
    assert not await art_store.forget(art.id)


# ──────────────────────────────────────────────
# Consolidation
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consolidate_removes_forgotten(art_store):
    art = _artifact()
    aid = await art_store.save(art)
    await art_store.forget(aid)
    await art_store.consolidate()
    assert not (art_store._dir / f"{aid}.json").exists()


@pytest.mark.asyncio
async def test_consolidate_preserves_active(art_store):
    art = _artifact()
    aid = await art_store.save(art)
    await art_store.consolidate()
    loaded = await art_store.get(aid)
    assert loaded is not None


# ──────────────────────────────────────────────
# Path safety
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_id_rejected(art_store):
    with pytest.raises(ValueError, match="Invalid entry ID"):
        await art_store.get("../../etc/passwd")


@pytest.mark.asyncio
async def test_null_byte_id_rejected(art_store):
    with pytest.raises(ValueError, match="Invalid entry ID"):
        await art_store.get("abc\0def")
