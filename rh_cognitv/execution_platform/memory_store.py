"""
MemoryStore — memory-specific storage logic.

Text-optimized serialization, role/tag-based retrieval, working-memory lifecycle.
DI-L3-04: Separate logic layer for Memory entries.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import Memory, MemoryQuery, QueryResult
from .types import ID


def _safe_filename(entry_id: str) -> str:
    """Validate ID and return safe filename, preventing path traversal."""
    if not entry_id or "/" in entry_id or "\\" in entry_id or ".." in entry_id or "\0" in entry_id:
        raise ValueError(f"Invalid entry ID: {entry_id!r}")
    return f"{entry_id}.json"


class MemoryStore:
    """File-based memory store. One JSON file per memory entry.

    Storage format: JSON envelope ``{"entry": <model_dump>, "forgotten": bool}``.
    """

    def __init__(self, store_dir: Path) -> None:
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    async def save(self, entry: Memory) -> ID:
        """Persist a memory entry to disk."""
        filename = _safe_filename(entry.id)
        path = self._dir / filename
        envelope = {
            "entry": entry.model_dump(mode="json"),
            "forgotten": False,
        }
        path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")
        return entry.id

    async def get(self, entry_id: ID) -> Memory | None:
        """Load a memory entry by ID. Returns None if not found or forgotten."""
        envelope = self._read_envelope(entry_id)
        if envelope is None or envelope.get("forgotten"):
            return None
        return Memory.model_validate(envelope["entry"])

    async def search(self, query: MemoryQuery) -> list[QueryResult]:
        """Search memories matching query filters."""
        results: list[QueryResult] = []
        for path in sorted(self._dir.glob("*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("forgotten"):
                continue
            memory = Memory.model_validate(envelope["entry"])
            if self._matches(memory, query):
                results.append(QueryResult(entry=memory))
        return results

    async def forget(self, entry_id: ID) -> bool:
        """Soft-delete a memory. Returns True if the entry existed and was active."""
        envelope = self._read_envelope(entry_id)
        if envelope is None or envelope.get("forgotten"):
            return False
        envelope["forgotten"] = True
        self._write_envelope(entry_id, envelope)
        return True

    async def consolidate(self) -> None:
        """Maintenance pass: remove forgotten entries from disk."""
        for path in list(self._dir.glob("*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("forgotten"):
                path.unlink()

    # ── Internal helpers ──

    def _read_envelope(self, entry_id: ID) -> dict | None:
        filename = _safe_filename(entry_id)
        path = self._dir / filename
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_envelope(self, entry_id: ID, envelope: dict) -> None:
        filename = _safe_filename(entry_id)
        path = self._dir / filename
        path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _matches(memory: Memory, query: MemoryQuery) -> bool:
        """Check whether a memory matches the given query filters."""
        # If artifact_type filter is set, memories never match
        if query.artifact_type is not None:
            return False
        if query.role is not None and memory.role != query.role:
            return False
        if query.tags is not None and not all(t in memory.tags for t in query.tags):
            return False
        if query.text and query.text.lower() not in memory.content.text.lower():
            return False
        return True
