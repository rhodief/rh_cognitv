"""
Tests for cognitive/serializer.py — Phase 3.5 test gate.

Covers:
- NaiveSerializer: joins entries with headers, empty inputs
- SectionSerializer: groups by MemoryRole with markdown headers
- Artifact formatting includes type and slug
- Empty inputs produce empty string
- Round-trip: output injected into PromptBuilder.context() appears in BuiltPrompt
"""

from __future__ import annotations

import pytest

from rh_cognitv.execution_platform.models import (
    Artifact,
    ArtifactProvenance,
    ArtifactStatus,
    ArtifactType,
    EntryContent,
    Memory,
    MemoryOrigin,
    MemoryRole,
    MemoryShape,
    Provenance,
    TimeInfo,
)
from rh_cognitv.execution_platform.types import now_timestamp
from rh_cognitv.cognitive.protocols import ContextSerializerProtocol
from rh_cognitv.cognitive.serializer import NaiveSerializer, SectionSerializer
from rh_cognitv.cognitive.prompt import PromptBuilder


# ──────────────────────────────────────────────
# Helpers — build test fixtures
# ──────────────────────────────────────────────


def _make_memory(
    text: str,
    role: MemoryRole = MemoryRole.SEMANTIC,
    shape: MemoryShape = MemoryShape.ATOM,
) -> Memory:
    ts = now_timestamp()
    return Memory(
        content=EntryContent(text=text),
        role=role,
        shape=shape,
        provenance=Provenance(origin=MemoryOrigin.OBSERVED, source="test"),
        time=TimeInfo(recorded_at=ts, observed_at=ts),
    )


def _make_artifact(
    text: str,
    type: ArtifactType = ArtifactType.DOCUMENT,
    slug: str = "test-doc",
) -> Artifact:
    return Artifact(
        content=EntryContent(text=text),
        type=type,
        slug=slug,
        provenance=ArtifactProvenance(intent="testing"),
    )


# ──────────────────────────────────────────────
# Tests — NaiveSerializer Protocol Compliance
# ──────────────────────────────────────────────


class TestNaiveSerializerProtocol:
    def test_is_context_serializer_protocol(self):
        assert isinstance(NaiveSerializer(), ContextSerializerProtocol)

    def test_has_serialize_method(self):
        s = NaiveSerializer()
        assert hasattr(s, "serialize")
        assert callable(s.serialize)


# ──────────────────────────────────────────────
# Tests — NaiveSerializer Basic
# ──────────────────────────────────────────────


class TestNaiveSerializerBasic:
    def test_empty_inputs(self):
        result = NaiveSerializer().serialize([], [])
        assert result == ""

    def test_empty_memories_only(self):
        result = NaiveSerializer().serialize([], [])
        assert result == ""

    def test_single_memory(self):
        mem = _make_memory("The user prefers short answers")
        result = NaiveSerializer().serialize([mem], [])
        assert result == "[Memory] The user prefers short answers"

    def test_multiple_memories(self):
        m1 = _make_memory("Fact one")
        m2 = _make_memory("Fact two")
        result = NaiveSerializer().serialize([m1, m2], [])
        assert "[Memory] Fact one" in result
        assert "[Memory] Fact two" in result
        assert "\n\n" in result

    def test_single_artifact(self):
        art = _make_artifact("def hello(): pass", ArtifactType.CODE, "my-script")
        result = NaiveSerializer().serialize([], [art])
        assert '[Artifact: code "my-script"]' in result
        assert "def hello(): pass" in result

    def test_artifact_includes_type_and_slug(self):
        art = _make_artifact("Some doc", ArtifactType.DOCUMENT, "readme")
        result = NaiveSerializer().serialize([], [art])
        assert 'document' in result
        assert '"readme"' in result

    def test_memories_and_artifacts(self):
        mem = _make_memory("Background info")
        art = _make_artifact("Code content", ArtifactType.CODE, "util")
        result = NaiveSerializer().serialize([mem], [art])
        assert "[Memory] Background info" in result
        assert '[Artifact: code "util"]' in result

    def test_memories_before_artifacts(self):
        mem = _make_memory("Memory text")
        art = _make_artifact("Artifact text", slug="doc")
        result = NaiveSerializer().serialize([mem], [art])
        mem_pos = result.index("[Memory]")
        art_pos = result.index("[Artifact:")
        assert mem_pos < art_pos


# ──────────────────────────────────────────────
# Tests — NaiveSerializer Edge Cases
# ──────────────────────────────────────────────


class TestNaiveSerializerEdgeCases:
    def test_multiline_content(self):
        mem = _make_memory("Line one\nLine two\nLine three")
        result = NaiveSerializer().serialize([mem], [])
        assert "Line one\nLine two\nLine three" in result

    def test_all_artifact_types(self):
        for atype in ArtifactType:
            art = _make_artifact("content", atype, f"slug-{atype.value}")
            result = NaiveSerializer().serialize([], [art])
            assert atype.value in result
            assert f"slug-{atype.value}" in result


# ──────────────────────────────────────────────
# Tests — SectionSerializer Protocol Compliance
# ──────────────────────────────────────────────


class TestSectionSerializerProtocol:
    def test_is_context_serializer_protocol(self):
        assert isinstance(SectionSerializer(), ContextSerializerProtocol)

    def test_has_serialize_method(self):
        s = SectionSerializer()
        assert hasattr(s, "serialize")
        assert callable(s.serialize)


# ──────────────────────────────────────────────
# Tests — SectionSerializer Grouping
# ──────────────────────────────────────────────


class TestSectionSerializerGrouping:
    def test_empty_inputs(self):
        result = SectionSerializer().serialize([], [])
        assert result == ""

    def test_single_role_section(self):
        mem = _make_memory("A fact", MemoryRole.SEMANTIC)
        result = SectionSerializer().serialize([mem], [])
        assert "## Semantic" in result
        assert "- A fact" in result

    def test_multiple_roles(self):
        m1 = _make_memory("An event", MemoryRole.EPISODIC)
        m2 = _make_memory("A fact", MemoryRole.SEMANTIC)
        m3 = _make_memory("How to code", MemoryRole.PROCEDURAL)
        result = SectionSerializer().serialize([m1, m2, m3], [])
        assert "## Episodic" in result
        assert "## Semantic" in result
        assert "## Procedural" in result

    def test_working_role(self):
        mem = _make_memory("Current task", MemoryRole.WORKING)
        result = SectionSerializer().serialize([mem], [])
        assert "## Working" in result
        assert "- Current task" in result

    def test_multiple_memories_same_role(self):
        m1 = _make_memory("Fact one", MemoryRole.SEMANTIC)
        m2 = _make_memory("Fact two", MemoryRole.SEMANTIC)
        result = SectionSerializer().serialize([m1, m2], [])
        assert result.count("## Semantic") == 1
        assert "- Fact one" in result
        assert "- Fact two" in result

    def test_empty_roles_omitted(self):
        mem = _make_memory("A fact", MemoryRole.SEMANTIC)
        result = SectionSerializer().serialize([mem], [])
        assert "## Episodic" not in result
        assert "## Procedural" not in result
        assert "## Working" not in result

    def test_role_order(self):
        """Working > Episodic > Semantic > Procedural."""
        mw = _make_memory("working", MemoryRole.WORKING)
        me = _make_memory("episodic", MemoryRole.EPISODIC)
        ms = _make_memory("semantic", MemoryRole.SEMANTIC)
        mp = _make_memory("procedural", MemoryRole.PROCEDURAL)
        # Pass in reverse order
        result = SectionSerializer().serialize([mp, ms, me, mw], [])
        w_pos = result.index("## Working")
        e_pos = result.index("## Episodic")
        s_pos = result.index("## Semantic")
        p_pos = result.index("## Procedural")
        assert w_pos < e_pos < s_pos < p_pos


# ──────────────────────────────────────────────
# Tests — SectionSerializer Artifacts
# ──────────────────────────────────────────────


class TestSectionSerializerArtifacts:
    def test_artifact_section(self):
        art = _make_artifact("Code here", ArtifactType.CODE, "utils")
        result = SectionSerializer().serialize([], [art])
        assert "## Artifacts" in result
        assert 'code "utils"' in result
        assert "Code here" in result

    def test_artifacts_after_memories(self):
        mem = _make_memory("A fact", MemoryRole.SEMANTIC)
        art = _make_artifact("Code", ArtifactType.CODE, "script")
        result = SectionSerializer().serialize([mem], [art])
        sem_pos = result.index("## Semantic")
        art_pos = result.index("## Artifacts")
        assert sem_pos < art_pos

    def test_multiple_artifacts(self):
        a1 = _make_artifact("First", ArtifactType.CODE, "one")
        a2 = _make_artifact("Second", ArtifactType.DOCUMENT, "two")
        result = SectionSerializer().serialize([], [a1, a2])
        assert "## Artifacts" in result
        assert 'code "one"' in result
        assert 'document "two"' in result

    def test_artifact_formatting_includes_type_and_slug(self):
        art = _make_artifact("Content", ArtifactType.DATA, "my-data")
        result = SectionSerializer().serialize([], [art])
        assert 'data "my-data"' in result

    def test_no_artifact_section_when_empty(self):
        mem = _make_memory("A fact")
        result = SectionSerializer().serialize([mem], [])
        assert "## Artifacts" not in result


# ──────────────────────────────────────────────
# Tests — SectionSerializer Full
# ──────────────────────────────────────────────


class TestSectionSerializerFull:
    def test_full_context(self):
        memories = [
            _make_memory("Current task description", MemoryRole.WORKING),
            _make_memory("Meeting happened yesterday", MemoryRole.EPISODIC),
            _make_memory("Python is a programming language", MemoryRole.SEMANTIC),
            _make_memory("To deploy: run make deploy", MemoryRole.PROCEDURAL),
        ]
        artifacts = [
            _make_artifact("def main(): pass", ArtifactType.CODE, "main-py"),
        ]
        result = SectionSerializer().serialize(memories, artifacts)
        assert "## Working" in result
        assert "## Episodic" in result
        assert "## Semantic" in result
        assert "## Procedural" in result
        assert "## Artifacts" in result
        assert "Current task description" in result
        assert "def main(): pass" in result


# ──────────────────────────────────────────────
# Tests — Round-Trip with PromptBuilder
# ──────────────────────────────────────────────


class TestRoundTrip:
    """Serialized output injected into PromptBuilder.context() appears in BuiltPrompt."""

    def test_naive_into_prompt_builder(self):
        mem = _make_memory("User prefers concise answers")
        serialized = NaiveSerializer().serialize([mem], [])
        result = (
            PromptBuilder()
            .system("You are helpful.")
            .context(serialized)
            .user("Summarize this.")
            .build()
        )
        assert "User prefers concise answers" in result.prompt
        assert result.system_prompt == "You are helpful."

    def test_section_into_prompt_builder(self):
        mem = _make_memory("Important fact", MemoryRole.SEMANTIC)
        art = _make_artifact("Draft text", ArtifactType.DOCUMENT, "draft")
        serialized = SectionSerializer().serialize([mem], [art])
        result = (
            PromptBuilder()
            .context(serialized)
            .user("Review the draft.")
            .build()
        )
        assert "Important fact" in result.prompt
        assert "Draft text" in result.prompt
        assert "Review the draft." in result.prompt

    def test_empty_serialization_does_not_add_context(self):
        serialized = NaiveSerializer().serialize([], [])
        result = (
            PromptBuilder()
            .context(serialized)
            .user("Hello")
            .build()
        )
        assert result.prompt == "Hello"
