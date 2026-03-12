"""
ContextStore — file-based unified store for memories and artifacts.

Implements ContextStoreProtocol. One JSON file per entry. DD-L3-04.
DI-L3-04: Separate logic layers (MemoryStore, ArtifactStore), shared interface.
"""

from __future__ import annotations

from pathlib import Path

from .artifact_store import ArtifactStore
from .memory_store import MemoryStore
from .models import Artifact, Memory, MemoryQuery, QueryResult
from .protocols import ContextStoreProtocol
from .types import ID


class ContextStore(ContextStoreProtocol):
    """File-based unified context store.

    Delegates memory-specific logic to MemoryStore, artifact-specific
    logic to ArtifactStore. Both stores use subdirectories under base_dir.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._memory_store = MemoryStore(self._base_dir / "memories")
        self._artifact_store = ArtifactStore(self._base_dir / "artifacts")

    @property
    def base_dir(self) -> Path:
        """Root directory for all stored entries."""
        return self._base_dir

    async def remember(self, entry: Memory) -> ID:
        return await self._memory_store.save(entry)

    async def store(self, entry: Artifact) -> ID:
        return await self._artifact_store.save(entry)

    async def recall(self, query: MemoryQuery) -> list[QueryResult]:
        results: list[QueryResult] = []
        if query.kind is None or query.kind == "memory":
            results.extend(await self._memory_store.search(query))
        if query.kind is None or query.kind == "artifact":
            results.extend(await self._artifact_store.search(query))
        if query.top_k is not None:
            results = results[:query.top_k]
        return results

    async def get(self, id: ID) -> Memory | Artifact | None:
        entry = await self._memory_store.get(id)
        if entry is not None:
            return entry
        return await self._artifact_store.get(id)

    async def get_artifact(self, slug: str, version: int | None = None) -> Artifact | None:
        return await self._artifact_store.get_by_slug(slug, version)

    async def forget(self, id: ID) -> None:
        if await self._memory_store.forget(id):
            return
        await self._artifact_store.forget(id)

    async def consolidate(self) -> None:
        await self._memory_store.consolidate()
        await self._artifact_store.consolidate()
