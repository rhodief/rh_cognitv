"""
ArtifactStore — artifact-specific storage logic.

Format-diverse serialization (driven by content.format), slug-based retrieval,
versioning with supersedes chain.
DI-L3-04: Separate logic layer for Artifact entries.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import Artifact, ArtifactStatus, MemoryQuery, QueryResult
from .types import ID


def _safe_filename(entry_id: str) -> str:
    """Validate ID and return safe filename, preventing path traversal."""
    if not entry_id or "/" in entry_id or "\\" in entry_id or ".." in entry_id or "\0" in entry_id:
        raise ValueError(f"Invalid entry ID: {entry_id!r}")
    return f"{entry_id}.json"


class ArtifactStore:
    """File-based artifact store with versioning support.

    Storage format: JSON envelope ``{"entry": <model_dump>, "forgotten": bool}``.
    Auto-versions when storing an artifact whose slug already exists.
    """

    def __init__(self, store_dir: Path) -> None:
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    async def save(self, entry: Artifact) -> ID:
        """Persist an artifact. Auto-versions if slug already exists.

        When an artifact with the same slug already exists and the new entry
        has ``version == 1`` and ``supersedes is None`` (defaults), the store
        automatically increments the version, sets supersedes, and deprecates
        the previous version.
        """
        # Auto-versioning: detect default (version=1, no supersedes) for existing slug
        if entry.version == 1 and entry.supersedes is None:
            existing = await self.get_by_slug(entry.slug)
            if existing is not None and existing.id != entry.id:
                entry = entry.model_copy(update={
                    "version": existing.version + 1,
                    "supersedes": existing.id,
                })
                # Deprecate the old version
                old_envelope = self._read_envelope(existing.id)
                if old_envelope is not None:
                    old_envelope["entry"]["status"] = ArtifactStatus.DEPRECATED.value
                    self._write_envelope(existing.id, old_envelope)

        self._write_envelope(entry.id, {
            "entry": entry.model_dump(mode="json"),
            "forgotten": False,
        })
        return entry.id

    async def get(self, entry_id: ID) -> Artifact | None:
        """Load an artifact by ID. Returns None if not found or forgotten."""
        envelope = self._read_envelope(entry_id)
        if envelope is None or envelope.get("forgotten"):
            return None
        return Artifact.model_validate(envelope["entry"])

    async def get_by_slug(self, slug: str, version: int | None = None) -> Artifact | None:
        """Fetch artifact by slug. Latest active version by default."""
        candidates: list[Artifact] = []
        for path in sorted(self._dir.glob("*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("forgotten"):
                continue
            artifact = Artifact.model_validate(envelope["entry"])
            if artifact.slug != slug:
                continue
            if version is not None and artifact.version == version:
                return artifact
            candidates.append(artifact)

        if version is not None:
            return None

        # Prefer latest active version
        active = [a for a in candidates if a.status == ArtifactStatus.ACTIVE]
        if active:
            return max(active, key=lambda a: a.version)
        # Fallback: latest regardless of status
        if candidates:
            return max(candidates, key=lambda a: a.version)
        return None

    async def search(self, query: MemoryQuery) -> list[QueryResult]:
        """Search artifacts matching query filters."""
        results: list[QueryResult] = []
        for path in sorted(self._dir.glob("*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("forgotten"):
                continue
            artifact = Artifact.model_validate(envelope["entry"])
            if self._matches(artifact, query):
                results.append(QueryResult(entry=artifact))
        return results

    async def forget(self, entry_id: ID) -> bool:
        """Soft-delete an artifact (mark forgotten and archived)."""
        envelope = self._read_envelope(entry_id)
        if envelope is None or envelope.get("forgotten"):
            return False
        envelope["forgotten"] = True
        envelope["entry"]["status"] = ArtifactStatus.ARCHIVED.value
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
    def _matches(artifact: Artifact, query: MemoryQuery) -> bool:
        """Check whether an artifact matches the given query filters."""
        # If role filter is set, artifacts never match
        if query.role is not None:
            return False
        if query.artifact_type is not None and artifact.type != query.artifact_type:
            return False
        if query.tags is not None and not all(t in artifact.tags for t in query.tags):
            return False
        if query.text and query.text.lower() not in artifact.content.text.lower():
            return False
        return True
