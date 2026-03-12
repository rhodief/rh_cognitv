"""Tests for models.py — Pydantic model validation and serialization roundtrips."""

import json

from pydantic import ValidationError as PydanticValidationError

from rh_cognitv.execution_platform.models import (
    Artifact,
    ArtifactProvenance,
    ArtifactStatus,
    ArtifactType,
    BaseEntry,
    BudgetSnapshot,
    EntryContent,
    EventKind,
    EventStatus,
    ExecutionResult,
    FunctionResultData,
    LLMResultData,
    Memory,
    MemoryOrigin,
    MemoryQuery,
    MemoryRole,
    MemoryShape,
    Provenance,
    QueryResult,
    ResultMetadata,
    TimeInfo,
    TokenBudget,
    TokenUsage,
    ToolResultData,
)
from rh_cognitv.execution_platform.types import generate_ulid, now_timestamp

import pytest


# ──────────────────────────────────────────────
# EntryContent
# ──────────────────────────────────────────────


class TestEntryContent:
    def test_minimal(self):
        c = EntryContent(text="hello")
        assert c.text == "hello"
        assert c.summary is None
        assert c.format is None

    def test_full(self):
        c = EntryContent(text="hello", summary="short", format="text/markdown")
        assert c.summary == "short"
        assert c.format == "text/markdown"

    def test_text_required(self):
        with pytest.raises(PydanticValidationError):
            EntryContent()

    def test_serialization_roundtrip(self):
        c = EntryContent(text="hello", format="code/python")
        data = c.model_dump()
        c2 = EntryContent.model_validate(data)
        assert c == c2

    def test_json_roundtrip(self):
        c = EntryContent(text="hello", summary="s")
        j = c.model_dump_json()
        c2 = EntryContent.model_validate_json(j)
        assert c == c2


# ──────────────────────────────────────────────
# Memory
# ──────────────────────────────────────────────


def _make_memory(**overrides) -> Memory:
    ts = now_timestamp()
    defaults = dict(
        content=EntryContent(text="User prefers TypeScript"),
        role=MemoryRole.SEMANTIC,
        shape=MemoryShape.ATOM,
        provenance=Provenance(origin=MemoryOrigin.TOLD, source="user message"),
        time=TimeInfo(recorded_at=ts, observed_at=ts),
        tags=["preference"],
    )
    defaults.update(overrides)
    return Memory(**defaults)


class TestMemory:
    def test_create_minimal(self):
        m = _make_memory()
        assert m.kind == "memory"
        assert m.role == MemoryRole.SEMANTIC
        assert m.shape == MemoryShape.ATOM
        assert len(m.id) == 26  # ULID

    def test_auto_id_generation(self):
        m1 = _make_memory()
        m2 = _make_memory()
        assert m1.id != m2.id

    def test_auto_timestamps(self):
        m = _make_memory()
        assert m.created_at is not None
        assert m.updated_at is not None

    def test_all_roles(self):
        for role in MemoryRole:
            m = _make_memory(role=role)
            assert m.role == role

    def test_all_shapes(self):
        for shape in MemoryShape:
            m = _make_memory(shape=shape)
            assert m.shape == shape

    def test_all_origins(self):
        ts = now_timestamp()
        for origin in MemoryOrigin:
            m = _make_memory(
                provenance=Provenance(origin=origin, source="test"),
            )
            assert m.provenance.origin == origin

    def test_ext_field(self):
        m = _make_memory(ext={"confidence": 0.95, "project": "auth"})
        assert m.ext["confidence"] == 0.95

    def test_tags(self):
        m = _make_memory(tags=["a", "b", "c"])
        assert m.tags == ["a", "b", "c"]

    def test_serialization_roundtrip(self):
        m = _make_memory(ext={"confidence": 0.9})
        data = m.model_dump()
        m2 = Memory.model_validate(data)
        assert m2.id == m.id
        assert m2.role == m.role
        assert m2.content.text == m.content.text
        assert m2.ext == m.ext

    def test_json_roundtrip(self):
        m = _make_memory()
        j = m.model_dump_json()
        m2 = Memory.model_validate_json(j)
        assert m2.id == m.id
        assert m2.kind == "memory"


# ──────────────────────────────────────────────
# Artifact
# ──────────────────────────────────────────────


def _make_artifact(**overrides) -> Artifact:
    defaults = dict(
        content=EntryContent(text="function auth() {}", format="code/typescript"),
        type=ArtifactType.CODE,
        slug="auth-module",
        provenance=ArtifactProvenance(
            input_memory_ids=["01HXYZ"],
            intent="implement JWT auth",
        ),
        tags=["auth", "jwt"],
    )
    defaults.update(overrides)
    return Artifact(**defaults)


class TestArtifact:
    def test_create_minimal(self):
        a = _make_artifact()
        assert a.kind == "artifact"
        assert a.type == ArtifactType.CODE
        assert a.slug == "auth-module"
        assert a.version == 1
        assert a.status == ArtifactStatus.ACTIVE

    def test_all_types(self):
        for t in ArtifactType:
            a = _make_artifact(type=t)
            assert a.type == t

    def test_all_statuses(self):
        for s in ArtifactStatus:
            a = _make_artifact(status=s)
            assert a.status == s

    def test_version_default(self):
        a = _make_artifact()
        assert a.version == 1

    def test_supersedes(self):
        id1 = generate_ulid()
        a = _make_artifact(supersedes=id1)
        assert a.supersedes == id1

    def test_provenance_memory_ids(self):
        a = _make_artifact()
        assert "01HXYZ" in a.provenance.input_memory_ids

    def test_serialization_roundtrip(self):
        a = _make_artifact(version=3, status=ArtifactStatus.DEPRECATED)
        data = a.model_dump()
        a2 = Artifact.model_validate(data)
        assert a2.version == 3
        assert a2.status == ArtifactStatus.DEPRECATED
        assert a2.slug == a.slug

    def test_json_roundtrip(self):
        a = _make_artifact()
        j = a.model_dump_json()
        a2 = Artifact.model_validate_json(j)
        assert a2.slug == "auth-module"
        assert a2.kind == "artifact"


# ──────────────────────────────────────────────
# Enums Coverage
# ──────────────────────────────────────────────


class TestEnums:
    def test_event_kind_values(self):
        assert set(EventKind) == {
            EventKind.TEXT,
            EventKind.DATA,
            EventKind.FUNCTION,
            EventKind.TOOL,
        }

    def test_event_status_values(self):
        expected = {
            "created", "queued", "running", "success", "failed",
            "retrying", "cancelled", "timed_out", "escalated", "waiting",
        }
        assert {s.value for s in EventStatus} == expected

    def test_memory_role_str(self):
        assert str(MemoryRole.EPISODIC) == "MemoryRole.EPISODIC"
        assert MemoryRole.EPISODIC.value == "episodic"

    def test_artifact_status_str(self):
        assert ArtifactStatus.DRAFT.value == "draft"


# ──────────────────────────────────────────────
# Execution Result Types
# ──────────────────────────────────────────────


class TestTokenUsage:
    def test_defaults(self):
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.completion_tokens == 0
        assert t.total == 0

    def test_values(self):
        t = TokenUsage(prompt_tokens=100, completion_tokens=50, total=150)
        assert t.total == 150


class TestResultMetadata:
    def test_defaults(self):
        m = ResultMetadata()
        assert m.duration_ms == 0.0
        assert m.attempt == 1

    def test_custom(self):
        ts = now_timestamp()
        m = ResultMetadata(duration_ms=123.4, attempt=3, started_at=ts, completed_at=ts)
        assert m.attempt == 3


class TestLLMResultData:
    def test_minimal(self):
        r = LLMResultData(text="Hello world")
        assert r.text == "Hello world"
        assert r.thinking is None
        assert r.model == ""

    def test_full(self):
        r = LLMResultData(
            text="response",
            thinking="chain of thought",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total=30),
            model="gpt-4",
            finish_reason="stop",
        )
        assert r.token_usage.total == 30
        assert r.model == "gpt-4"

    def test_json_roundtrip(self):
        r = LLMResultData(text="test", model="claude-3")
        j = r.model_dump_json()
        r2 = LLMResultData.model_validate_json(j)
        assert r2.text == "test"


class TestFunctionResultData:
    def test_minimal(self):
        r = FunctionResultData()
        assert r.return_value is None
        assert r.duration_ms == 0.0

    def test_with_value(self):
        r = FunctionResultData(return_value={"key": "value"}, duration_ms=42.5)
        assert r.return_value == {"key": "value"}


class TestToolResultData:
    def test_composition(self):
        llm = LLMResultData(text="call tool X", model="gpt-4")
        fn = FunctionResultData(return_value="result", duration_ms=10.0)
        t = ToolResultData(llm_result=llm, function_result=fn)
        assert t.llm_result.text == "call tool X"
        assert t.function_result.return_value == "result"

    def test_json_roundtrip(self):
        t = ToolResultData(
            llm_result=LLMResultData(text="call"),
            function_result=FunctionResultData(return_value=42),
        )
        j = t.model_dump_json()
        t2 = ToolResultData.model_validate_json(j)
        assert t2.function_result.return_value == 42


class TestExecutionResult:
    def test_success(self):
        r = ExecutionResult[LLMResultData](
            ok=True,
            value=LLMResultData(text="hello"),
        )
        assert r.ok is True
        assert r.value.text == "hello"
        assert r.error_message is None

    def test_failure(self):
        r = ExecutionResult[LLMResultData](
            ok=False,
            error_message="rate limited",
            error_category="transient",
        )
        assert r.ok is False
        assert r.value is None

    def test_with_metadata(self):
        ts = now_timestamp()
        r = ExecutionResult[FunctionResultData](
            ok=True,
            value=FunctionResultData(return_value="done"),
            metadata=ResultMetadata(duration_ms=100, attempt=2, started_at=ts),
        )
        assert r.metadata.attempt == 2


# ──────────────────────────────────────────────
# Query Types
# ──────────────────────────────────────────────


class TestMemoryQuery:
    def test_minimal(self):
        q = MemoryQuery()
        assert q.text == ""
        assert q.kind is None
        assert q.top_k is None

    def test_full(self):
        q = MemoryQuery(
            text="authentication",
            kind="memory",
            role=MemoryRole.SEMANTIC,
            tags=["auth"],
            top_k=5,
        )
        assert q.top_k == 5
        assert q.role == MemoryRole.SEMANTIC


class TestQueryResult:
    def test_with_memory(self):
        m = _make_memory()
        qr = QueryResult(entry=m)
        assert qr.score == 1.0
        assert qr.entry.kind == "memory"

    def test_with_artifact(self):
        a = _make_artifact()
        qr = QueryResult(entry=a, score=0.85)
        assert qr.score == 0.85


# ──────────────────────────────────────────────
# Budget
# ──────────────────────────────────────────────


class TestTokenBudget:
    def test_defaults(self):
        b = TokenBudget(total=4000)
        assert b.total == 4000
        assert b.working == 0

    def test_full(self):
        b = TokenBudget(
            total=4000, working=1000, episodic=1000,
            semantic=500, procedural=1000, artifacts=500,
        )
        assert b.working + b.episodic + b.semantic + b.procedural + b.artifacts == 4000


class TestBudgetSnapshot:
    def test_creation(self):
        s = BudgetSnapshot(tokens_remaining=1000, calls_remaining=5, time_remaining_seconds=30.0)
        assert s.tokens_remaining == 1000
