"""
Tests for Phase 3.7 — L2 Context Resolution Hook.

Verifies that DAGOrchestrator resolves context_refs from node.ext
before executing each node. Tests all four ref kinds:
  - memory: resolve by ID via ContextStore.get()
  - artifact: resolve by slug/version via ContextStore.get_artifact()
  - query: resolve via ContextStore.recall()
  - previous_result: resolve from node_results

Also verifies backward compatibility: nodes without context_refs
work unchanged, and orchestrators without a context_store skip resolution.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from rh_cognitv.execution_platform.events import ExecutionEvent
from rh_cognitv.execution_platform.handlers import HandlerRegistry
from rh_cognitv.execution_platform.models import (
    Artifact,
    ArtifactProvenance,
    ArtifactStatus,
    ArtifactType,
    EntryContent,
    EventKind,
    ExecutionResult,
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
    TokenUsage,
)
from rh_cognitv.execution_platform.protocols import EventHandlerProtocol
from rh_cognitv.execution_platform.state import ExecutionState
from rh_cognitv.execution_platform.types import now_timestamp

from rh_cognitv.orchestrator.adapters import AdapterRegistry, PlatformRef
from rh_cognitv.orchestrator.dag_orchestrator import DAGOrchestrator
from rh_cognitv.orchestrator.models import DAGRunStatus, NodeResult
from rh_cognitv.orchestrator.nodes import TextNode
from rh_cognitv.orchestrator.plan_dag import DAGBuilder


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_memory(id: str = "mem-1", text: str = "memory text") -> Memory:
    ts = now_timestamp()
    return Memory(
        id=id,
        content=EntryContent(text=text),
        role=MemoryRole.SEMANTIC,
        shape=MemoryShape.ATOM,
        provenance=Provenance(origin=MemoryOrigin.OBSERVED, source="test"),
        time=TimeInfo(recorded_at=ts, observed_at=ts),
    )


def _make_artifact(
    id: str = "art-1",
    text: str = "artifact text",
    slug: str = "doc",
    version: int = 1,
) -> Artifact:
    return Artifact(
        id=id,
        content=EntryContent(text=text),
        type=ArtifactType.DOCUMENT,
        slug=slug,
        version=version,
        status=ArtifactStatus.ACTIVE,
        provenance=ArtifactProvenance(intent="test"),
    )


class CapturingTextHandler(EventHandlerProtocol):
    """Handler that captures the data arg passed to it."""

    def __init__(self):
        self.captured_data: list[Any] = []

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult:
        self.captured_data.append(data)
        prompt = event.payload.prompt if hasattr(event.payload, "prompt") else ""
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=f"response to: {prompt}",
                model="test",
                token_usage=TokenUsage(),
            ),
            metadata=ResultMetadata(),
        )


class MockContextStore:
    """Minimal mock implementing the ContextStore methods we need."""

    def __init__(self):
        self._entries: dict[str, Any] = {}
        self._artifacts_by_slug: dict[tuple[str, int | None], Any] = {}
        self._recall_results: list[QueryResult] = []

    def add_memory(self, memory: Memory) -> None:
        self._entries[memory.id] = memory

    def add_artifact(self, artifact: Artifact) -> None:
        self._entries[artifact.id] = artifact
        self._artifacts_by_slug[(artifact.slug, artifact.version)] = artifact
        self._artifacts_by_slug[(artifact.slug, None)] = artifact  # latest

    def set_recall_results(self, results: list[QueryResult]) -> None:
        self._recall_results = results

    async def get(self, id: str) -> Any | None:
        return self._entries.get(id)

    async def get_artifact(self, slug: str, version: int | None = None) -> Any | None:
        return self._artifacts_by_slug.get((slug, version))

    async def recall(self, query: MemoryQuery) -> list[QueryResult]:
        return list(self._recall_results)

    # Unused but satisfy protocol
    async def remember(self, entry):
        pass

    async def store(self, entry):
        pass

    async def forget(self, id):
        pass

    async def consolidate(self):
        pass


class MockSerializer:
    """Minimal mock implementing ContextSerializerProtocol.serialize()."""

    def serialize(self, memories: list, artifacts: list) -> str:
        parts = []
        for m in memories:
            parts.append(f"[Memory] {m.content.text}")
        for a in artifacts:
            parts.append(f"[Artifact] {a.content.text}")
        return "\n".join(parts)


def _build_orchestrator(
    handler: CapturingTextHandler,
    context_store: MockContextStore | None = None,
    context_serializer: MockSerializer | None = None,
) -> DAGOrchestrator:
    registry = HandlerRegistry()
    registry.register(EventKind.TEXT, handler)
    adapter_reg = AdapterRegistry.with_defaults()
    platform = PlatformRef(registry=registry)
    state = ExecutionState()
    return DAGOrchestrator(
        adapter_registry=adapter_reg,
        platform=platform,
        state=state,
        context_store=context_store,
        context_serializer=context_serializer,
    )


# ══════════════════════════════════════════════
# Backward compatibility
# ══════════════════════════════════════════════


class TestBackwardCompatibility:
    """Nodes without context_refs and orchestrators without stores work unchanged."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_context_refs_passes_data_unchanged(self):
        handler = CapturingTextHandler()
        orch = _build_orchestrator(handler)
        dag = DAGBuilder("test").add_node("s1", TextNode(id="s1", prompt="Hello")).build()
        await orch.run(dag, {"input": "value"})
        assert handler.captured_data[0] == {"input": "value"}

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_context_store_skips_resolution(self):
        """Even with context_refs in ext, no store means no resolution."""
        handler = CapturingTextHandler()
        orch = _build_orchestrator(handler)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{"kind": "memory", "id": "m1", "key": "ctx"}]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag, {"input": "value"})
        assert handler.captured_data[0] == {"input": "value"}

    @pytest.mark.asyncio(loop_scope="function")
    async def test_none_data_works(self):
        handler = CapturingTextHandler()
        orch = _build_orchestrator(handler)
        dag = DAGBuilder("test").add_node("s1", TextNode(id="s1", prompt="Hello")).build()
        await orch.run(dag)
        assert handler.captured_data[0] is None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_empty_context_refs_passes_data_unchanged(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(id="s1", prompt="Hello", ext={"context_refs": []})
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag, {"input": "value"})
        assert handler.captured_data[0] == {"input": "value"}


# ══════════════════════════════════════════════
# Memory ref resolution
# ══════════════════════════════════════════════


class TestMemoryRefResolution:
    """kind=memory resolves via ContextStore.get(id)."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_memory_ref_resolved(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        mem = _make_memory("mem-1", "important context")
        store.add_memory(mem)

        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{"kind": "memory", "id": "mem-1", "key": "background"}]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        data = handler.captured_data[0]
        assert data["background"] == "important context"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_missing_memory_not_injected(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{"kind": "memory", "id": "nonexistent", "key": "ctx"}]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        data = handler.captured_data[0]
        assert "ctx" not in data

    @pytest.mark.asyncio(loop_scope="function")
    async def test_memory_ref_default_key(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        store.add_memory(_make_memory("m1", "default key text"))
        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{"kind": "memory", "id": "m1"}]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        data = handler.captured_data[0]
        assert data["context"] == "default key text"


# ══════════════════════════════════════════════
# Artifact ref resolution
# ══════════════════════════════════════════════


class TestArtifactRefResolution:
    """kind=artifact resolves via ContextStore.get_artifact(slug, version)."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_artifact_ref_by_slug(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        art = _make_artifact("a1", "draft content", slug="draft", version=1)
        store.add_artifact(art)

        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Review",
            ext={"context_refs": [{"kind": "artifact", "slug": "draft", "key": "doc"}]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        assert handler.captured_data[0]["doc"] == "draft content"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_artifact_ref_by_slug_and_version(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        art = _make_artifact("a1", "v2 content", slug="report", version=2)
        store.add_artifact(art)

        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Review",
            ext={"context_refs": [
                {"kind": "artifact", "slug": "report", "version": 2, "key": "report"},
            ]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        assert handler.captured_data[0]["report"] == "v2 content"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_missing_artifact_not_injected(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{"kind": "artifact", "slug": "missing", "key": "doc"}]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        assert "doc" not in handler.captured_data[0]


# ══════════════════════════════════════════════
# Query ref resolution
# ══════════════════════════════════════════════


class TestQueryRefResolution:
    """kind=query resolves via ContextStore.recall(query)."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_query_ref_without_serializer(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        mem = _make_memory("m1", "recalled text")
        store.set_recall_results([QueryResult(entry=mem, score=1.0)])

        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{
                "kind": "query",
                "query": {"text": "related", "top_k": 5},
                "key": "related",
            }]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        assert handler.captured_data[0]["related"] == "recalled text"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_query_ref_with_serializer(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        mem = _make_memory("m1", "semantic info")
        art = _make_artifact("a1", "code snippet", slug="code")
        store.set_recall_results([
            QueryResult(entry=mem, score=1.0),
            QueryResult(entry=art, score=0.9),
        ])

        serializer = MockSerializer()
        orch = _build_orchestrator(handler, context_store=store, context_serializer=serializer)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{
                "kind": "query",
                "query": {"text": "search", "top_k": 3},
                "key": "ctx",
            }]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        assert handler.captured_data[0]["ctx"] == "[Memory] semantic info\n[Artifact] code snippet"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_query_ref_empty_results(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        store.set_recall_results([])

        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{
                "kind": "query",
                "query": {"text": "nothing"},
                "key": "ctx",
            }]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        # Empty query results → empty string (falsy), not injected
        data = handler.captured_data[0]
        # Empty join produces "", which is falsy but still set
        assert data.get("ctx", "") == ""


# ══════════════════════════════════════════════
# Previous result ref resolution
# ══════════════════════════════════════════════


class TestPreviousResultRefResolution:
    """kind=previous_result resolves from node_results."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_previous_result_ref(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        orch = _build_orchestrator(handler, context_store=store)

        node_a = TextNode(id="a", prompt="First")
        node_b = TextNode(
            id="b", prompt="Second",
            ext={"context_refs": [{
                "kind": "previous_result",
                "from_step": "a",
                "key": "prior_output",
            }]},
        )
        dag = (
            DAGBuilder("test")
            .add_node("a", node_a)
            .add_node("b", node_b)
            .edge("a", "b")
            .build()
        )
        await orch.run(dag)

        # Node B should receive node A's result value
        data_b = handler.captured_data[1]
        assert data_b["prior_output"] == "response to: First"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_previous_result_missing_step(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{
                "kind": "previous_result",
                "from_step": "nonexistent",
                "key": "prior",
            }]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        assert "prior" not in handler.captured_data[0]


# ══════════════════════════════════════════════
# Multiple context_refs on one node
# ══════════════════════════════════════════════


class TestMultipleContextRefs:
    """Multiple context_refs on a single node all get resolved."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_multiple_refs_different_keys(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        store.add_memory(_make_memory("m1", "memory data"))
        store.add_artifact(_make_artifact("a1", "artifact data", slug="doc"))

        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [
                {"kind": "memory", "id": "m1", "key": "background"},
                {"kind": "artifact", "slug": "doc", "key": "document"},
            ]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        data = handler.captured_data[0]
        assert data["background"] == "memory data"
        assert data["document"] == "artifact data"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_same_key_last_wins(self):
        """If two refs have the same key, last one wins."""
        handler = CapturingTextHandler()
        store = MockContextStore()
        store.add_memory(_make_memory("m1", "first"))
        store.add_memory(_make_memory("m2", "second"))

        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [
                {"kind": "memory", "id": "m1", "key": "ctx"},
                {"kind": "memory", "id": "m2", "key": "ctx"},
            ]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag)

        assert handler.captured_data[0]["ctx"] == "second"


# ══════════════════════════════════════════════
# Data merging behavior
# ══════════════════════════════════════════════


class TestDataMerging:
    """Resolved context merges with existing data dict."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_existing_data_preserved(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        store.add_memory(_make_memory("m1", "extra context"))

        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{"kind": "memory", "id": "m1", "key": "ctx"}]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag, {"input": "original"})

        data = handler.captured_data[0]
        assert data["input"] == "original"
        assert data["ctx"] == "extra context"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_non_dict_data_becomes_dict_with_refs(self):
        """When data is not a dict, resolved refs create a new dict."""
        handler = CapturingTextHandler()
        store = MockContextStore()
        store.add_memory(_make_memory("m1", "injected"))

        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{"kind": "memory", "id": "m1", "key": "ctx"}]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        await orch.run(dag, "raw string data")

        data = handler.captured_data[0]
        assert data["ctx"] == "injected"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_dag_still_succeeds(self):
        handler = CapturingTextHandler()
        store = MockContextStore()
        store.add_memory(_make_memory("m1", "ctx text"))

        orch = _build_orchestrator(handler, context_store=store)
        node = TextNode(
            id="s1", prompt="Hello",
            ext={"context_refs": [{"kind": "memory", "id": "m1", "key": "ctx"}]},
        )
        dag = DAGBuilder("test").add_node("s1", node).build()
        edag = await orch.run(dag)

        assert orch.status == DAGRunStatus.SUCCESS
        assert edag.entry_count() > 0
