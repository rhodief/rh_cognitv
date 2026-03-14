"""
Context serializer — render memories and artifacts into prompt text.

Two implementations:
- NaiveSerializer (Stage 1): concatenate entries with headers
- SectionSerializer (Stage 2): group by MemoryRole with markdown headers

Both implement ContextSerializerProtocol. The output is a plain string
suitable for injection via PromptBuilder.context() or template {context}.

Phase 3.5.1 — Context Serializer.
"""

from __future__ import annotations

from typing import Any

from rh_cognitv.execution_platform.models import (
    Artifact,
    Memory,
    MemoryRole,
)

from .protocols import ContextSerializerProtocol


class NaiveSerializer(ContextSerializerProtocol):
    """Stage 1 serializer — simple concatenation with headers.

    Renders each memory and artifact as a labeled block:
      [Memory] <content.text>
      [Artifact: <type> "<slug>"] <content.text>

    Entries are joined with blank lines.
    """

    def serialize(
        self,
        memories: list[Any],
        artifacts: list[Any],
    ) -> str:
        parts: list[str] = []

        for mem in memories:
            text = _memory_text(mem)
            if text:
                parts.append(f"[Memory] {text}")

        for art in artifacts:
            text = _artifact_text(art)
            if text:
                label = _artifact_label(art)
                parts.append(f"[{label}] {text}")

        return "\n\n".join(parts)


class SectionSerializer(ContextSerializerProtocol):
    """Stage 2 serializer — group memories by role with markdown headers.

    Output format:
      ## Episodic
      - <memory text>
      - <memory text>

      ## Semantic
      - <memory text>

      ## Artifacts
      - [<type> "<slug>"] <text>

    Roles with no entries are omitted. Artifact section appears last.
    """

    _ROLE_ORDER = [
        MemoryRole.WORKING,
        MemoryRole.EPISODIC,
        MemoryRole.SEMANTIC,
        MemoryRole.PROCEDURAL,
    ]

    def serialize(
        self,
        memories: list[Any],
        artifacts: list[Any],
    ) -> str:
        # Group memories by role
        by_role: dict[MemoryRole, list[str]] = {}
        for mem in memories:
            role = _memory_role(mem)
            text = _memory_text(mem)
            if text:
                by_role.setdefault(role, []).append(text)

        sections: list[str] = []

        for role in self._ROLE_ORDER:
            items = by_role.get(role, [])
            if items:
                header = f"## {role.value.capitalize()}"
                bullets = "\n".join(f"- {item}" for item in items)
                sections.append(f"{header}\n{bullets}")

        # Artifacts section
        art_items: list[str] = []
        for art in artifacts:
            text = _artifact_text(art)
            if text:
                label = _artifact_label(art)
                art_items.append(f"- [{label}] {text}")

        if art_items:
            sections.append(f"## Artifacts\n" + "\n".join(art_items))

        return "\n\n".join(sections)


# ──────────────────────────────────────────────
# Helpers — extract fields safely from Any-typed entries
# ──────────────────────────────────────────────


def _memory_text(mem: Any) -> str:
    """Extract text from a Memory (or dict-like object)."""
    if isinstance(mem, Memory):
        return mem.content.text
    if isinstance(mem, dict):
        content = mem.get("content", {})
        if isinstance(content, dict):
            return content.get("text", "")
        return getattr(content, "text", "")
    return getattr(getattr(mem, "content", None), "text", "")


def _memory_role(mem: Any) -> MemoryRole:
    """Extract role from a Memory, defaulting to SEMANTIC."""
    if isinstance(mem, Memory):
        return mem.role
    if isinstance(mem, dict):
        raw = mem.get("role", "semantic")
        try:
            return MemoryRole(raw)
        except ValueError:
            return MemoryRole.SEMANTIC
    raw = getattr(mem, "role", "semantic")
    if isinstance(raw, MemoryRole):
        return raw
    try:
        return MemoryRole(raw)
    except ValueError:
        return MemoryRole.SEMANTIC


def _artifact_text(art: Any) -> str:
    """Extract text from an Artifact (or dict-like object)."""
    if isinstance(art, Artifact):
        return art.content.text
    if isinstance(art, dict):
        content = art.get("content", {})
        if isinstance(content, dict):
            return content.get("text", "")
        return getattr(content, "text", "")
    return getattr(getattr(art, "content", None), "text", "")


def _artifact_label(art: Any) -> str:
    """Build a label like 'Artifact: code "my-script"' from an artifact."""
    if isinstance(art, Artifact):
        return f"Artifact: {art.type.value} \"{art.slug}\""
    if isinstance(art, dict):
        atype = art.get("type", "document")
        slug = art.get("slug", "unknown")
    else:
        atype = getattr(art, "type", "document")
        slug = getattr(art, "slug", "unknown")
    # Handle enum values
    if hasattr(atype, "value"):
        atype = atype.value
    return f"Artifact: {atype} \"{slug}\""
