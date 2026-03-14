"""
Phase 3.10 — Integration Tests.

Full pipeline: Skill → Adapter → DAGOrchestrator → Execution Platform.

  3.10.1  Skill → Adapter → PlanDAG accepted by DAGOrchestrator
  3.10.2  Context resolution round-trip
  3.10.3  Full pipeline with MockLLM
  3.10.4  Replan flow
  3.10.5  Output validation retry
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from rh_cognitv.cognitive.adapters import (
    OrchestratorResult,
    ResultAdapter,
    SkillToDAGAdapter,
)
from rh_cognitv.cognitive.builtin_skills import (
    DataExtractionSkill,
    TextGenerationSkill,
)
from rh_cognitv.cognitive.models import (
    ContextRef,
    DataStepConfig,
    FunctionStepConfig,
    ReplanRequest,
    SkillConstraints,
    SkillContext,
    SkillPlan,
    SkillProvenance,
    SkillResult,
    SkillStep,
    TextStepConfig,
)
from rh_cognitv.cognitive.skill import RetryableValidationError, Skill

from rh_cognitv.execution_platform.events import ExecutionEvent, TextPayload
from rh_cognitv.execution_platform.handlers import HandlerRegistry
from rh_cognitv.execution_platform.models import (
    EventKind,
    ExecutionResult,
    LLMResultData,
    ResultMetadata,
    TokenUsage,
)
from rh_cognitv.execution_platform.protocols import EventHandlerProtocol
from rh_cognitv.execution_platform.state import ExecutionState

from rh_cognitv.orchestrator.adapters import AdapterRegistry, PlatformRef
from rh_cognitv.orchestrator.dag_orchestrator import DAGOrchestrator
from rh_cognitv.orchestrator.models import (
    DAGRunStatus,
    NodeExecutionStatus,
    OrchestratorConfig,
)


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════


class SimpleInput(BaseModel):
    """Minimal input model for skills."""

    text: str


class MockTextHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler that returns canned responses keyed by prompt substring."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[str] = []

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        prompt = event.payload.prompt
        self.calls.append(prompt)

        # Find a canned response by matching a key substring
        response_text = f"mock-response: {prompt[:40]}"
        for key, value in self._responses.items():
            if key in prompt:
                response_text = value
                break

        prompt_tokens = max(len(prompt.split()), 1)
        completion_tokens = max(len(response_text.split()), 1)
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=response_text,
                model="mock-llm",
                token_usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total=prompt_tokens + completion_tokens,
                ),
            ),
            metadata=ResultMetadata(),
        )


class MockDataHandler(EventHandlerProtocol[LLMResultData]):
    """Data handler that returns canned JSON responses."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[str] = []

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        prompt = event.payload.prompt
        self.calls.append(prompt)

        response_text = "{}"
        for key, value in self._responses.items():
            if key in prompt:
                response_text = value
                break

        return ExecutionResult(
            ok=True,
            value=LLMResultData(text=response_text, model="mock-llm"),
            metadata=ResultMetadata(),
        )


def _build_pipeline(
    *,
    text_handler: EventHandlerProtocol | None = None,
    data_handler: EventHandlerProtocol | None = None,
    config: OrchestratorConfig | None = None,
    context_store: Any = None,
    context_serializer: Any = None,
) -> tuple[DAGOrchestrator, ExecutionState]:
    """Wire up the full L1→L2→L3 pipeline and return (orchestrator, state)."""
    handler_registry = HandlerRegistry()
    handler_registry.register(EventKind.TEXT, text_handler or MockTextHandler())
    handler_registry.register(EventKind.DATA, data_handler or MockDataHandler())

    cfg = config or OrchestratorConfig()
    platform = PlatformRef(registry=handler_registry, config=cfg)
    adapter_registry = AdapterRegistry.with_defaults()
    state = ExecutionState()

    orchestrator = DAGOrchestrator(
        adapter_registry=adapter_registry,
        platform=platform,
        state=state,
        config=cfg,
        context_store=context_store,
        context_serializer=context_serializer,
    )
    return orchestrator, state


# ══════════════════════════════════════════════
# 3.10.1 — Skill → Adapter → PlanDAG
# ══════════════════════════════════════════════


class TestSkillToAdapterToPlanDAG:
    """Verify that a skill's plan() output translates into a valid PlanDAG
    that DAGOrchestrator accepts."""

    @pytest.mark.asyncio
    async def test_text_generation_skill_single_step(self):
        """TextGenerationSkill → single-step PlanDAG with TextNode."""
        skill = TextGenerationSkill(system_prompt="Be helpful.")
        input_data = SimpleInput(text="Tell me about Python.")
        ctx = SkillContext()

        plan = await skill.plan(input_data, ctx)
        adapter = SkillToDAGAdapter()
        dag = adapter.to_dag(plan)

        assert dag.node_count() == 1
        node_ids = dag.node_ids()
        node = dag.get_node(node_ids[0])
        assert node.kind == "text"
        assert "Python" in node.prompt

    @pytest.mark.asyncio
    async def test_text_generation_skill_runs_through_orchestrator(self):
        """TextGenerationSkill PlanDAG accepted and executed by DAGOrchestrator."""
        mock_handler = MockTextHandler({"Python": "Python is great!"})
        orchestrator, _ = _build_pipeline(text_handler=mock_handler)

        skill = TextGenerationSkill(system_prompt="Be helpful.")
        plan = await skill.plan(SimpleInput(text="Tell me about Python."), SkillContext())
        dag = SkillToDAGAdapter().to_dag(plan)

        exec_dag = await orchestrator.run(dag)
        assert orchestrator.status == DAGRunStatus.SUCCESS
        assert exec_dag.entry_count() >= 1

        # Verify handler was invoked with the skill's prompt
        assert len(mock_handler.calls) == 1
        assert "Python" in mock_handler.calls[0]

    @pytest.mark.asyncio
    async def test_multi_step_custom_skill(self):
        """Multi-step skill plan → multi-node PlanDAG with correct edges."""
        plan = SkillPlan(
            name="multi-step",
            steps=[
                SkillStep(
                    id="step-1",
                    kind="text",
                    config=TextStepConfig(prompt="Research topic"),
                ),
                SkillStep(
                    id="step-2",
                    kind="text",
                    config=TextStepConfig(prompt="Summarize findings"),
                ),
                SkillStep(
                    id="step-3",
                    kind="data",
                    config=DataStepConfig(
                        prompt="Extract key points",
                        output_schema={"type": "object"},
                    ),
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)

        assert dag.node_count() == 3
        # Sequential edges: step-1 → step-2 → step-3
        assert dag.successors("step-1") == ["step-2"]
        assert dag.successors("step-2") == ["step-3"]
        assert dag.successors("step-3") == []

    @pytest.mark.asyncio
    async def test_multi_step_runs_through_orchestrator(self):
        """Multi-step PlanDAG executes all nodes in order."""
        mock_text = MockTextHandler()
        mock_data = MockDataHandler()
        orchestrator, _ = _build_pipeline(
            text_handler=mock_text, data_handler=mock_data
        )

        plan = SkillPlan(
            name="multi-step",
            steps=[
                SkillStep(
                    id="step-1",
                    kind="text",
                    config=TextStepConfig(prompt="Research topic"),
                ),
                SkillStep(
                    id="step-2",
                    kind="text",
                    config=TextStepConfig(prompt="Summarize findings"),
                ),
                SkillStep(
                    id="step-3",
                    kind="data",
                    config=DataStepConfig(prompt="Extract key points"),
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)
        exec_dag = await orchestrator.run(dag)

        assert orchestrator.status == DAGRunStatus.SUCCESS
        assert len(mock_text.calls) == 2
        assert len(mock_data.calls) == 1

    @pytest.mark.asyncio
    async def test_parallel_steps_via_depends_on(self):
        """Steps with explicit depends_on create parallel branches."""
        plan = SkillPlan(
            name="parallel",
            steps=[
                SkillStep(
                    id="step-a",
                    kind="text",
                    config=TextStepConfig(prompt="Branch A"),
                ),
                SkillStep(
                    id="step-b",
                    kind="text",
                    config=TextStepConfig(prompt="Branch B"),
                    depends_on=["step-a"],
                ),
                SkillStep(
                    id="step-c",
                    kind="text",
                    config=TextStepConfig(prompt="Branch C"),
                    depends_on=["step-a"],
                ),
                SkillStep(
                    id="step-d",
                    kind="text",
                    config=TextStepConfig(prompt="Merge"),
                    depends_on=["step-b", "step-c"],
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)

        assert dag.node_count() == 4
        assert set(dag.successors("step-a")) == {"step-b", "step-c"}
        assert dag.successors("step-d") == []

        # Run through orchestrator
        orchestrator, _ = _build_pipeline()
        exec_dag = await orchestrator.run(dag)
        assert orchestrator.status == DAGRunStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_data_extraction_skill_single_step(self):
        """DataExtractionSkill → single data-step PlanDAG."""
        skill = DataExtractionSkill(output_schema={"type": "object"})
        plan = await skill.plan(SimpleInput(text="Extract entities"), SkillContext())
        dag = SkillToDAGAdapter().to_dag(plan)

        assert dag.node_count() == 1
        node = dag.get_node(dag.node_ids()[0])
        assert node.kind == "data"

    @pytest.mark.asyncio
    async def test_constraints_applied_to_nodes(self):
        """SkillConstraints map to node timeout_seconds and max_retries."""
        plan = SkillPlan(
            name="constrained",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="Go"),
                ),
            ],
            constraints=SkillConstraints(timeout_seconds=5.0, max_retries=2),
        )
        dag = SkillToDAGAdapter().to_dag(plan)
        node = dag.get_node("s1")
        assert node.timeout_seconds == 5.0
        assert node.max_retries == 2


# ══════════════════════════════════════════════
# 3.10.2 — Context Resolution Round-Trip
# ══════════════════════════════════════════════


class TestContextResolutionRoundTrip:
    """Skill declares context_refs → adapter → ext → orchestrator resolves
    from ContextStore → handler receives resolved data."""

    @pytest.mark.asyncio
    async def test_memory_ref_resolved(self):
        """context_ref(kind=memory) is resolved from ContextStore at execution time."""
        # Build a plan with a memory context_ref
        plan = SkillPlan(
            name="ctx-test",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="Use this context"),
                    context_refs=[
                        ContextRef(kind="memory", id="mem-001", key="background"),
                    ],
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)

        # Verify refs are serialized into ext
        node = dag.get_node("s1")
        assert "context_refs" in node.ext
        refs = node.ext["context_refs"]
        assert refs[0]["kind"] == "memory"
        assert refs[0]["id"] == "mem-001"

        # Wire up a mock context store
        mock_entry = AsyncMock()
        mock_entry.content.text = "Background info about the topic"

        mock_store = AsyncMock()
        mock_store.get = AsyncMock(return_value=mock_entry)

        text_handler = MockTextHandler()
        orchestrator, _ = _build_pipeline(
            text_handler=text_handler, context_store=mock_store
        )

        await orchestrator.run(dag)
        assert orchestrator.status == DAGRunStatus.SUCCESS
        mock_store.get.assert_called_once_with("mem-001")

    @pytest.mark.asyncio
    async def test_artifact_ref_resolved(self):
        """context_ref(kind=artifact) is resolved via get_artifact."""
        plan = SkillPlan(
            name="ctx-artifact",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="Use artifact"),
                    context_refs=[
                        ContextRef(
                            kind="artifact",
                            slug="my-doc",
                            version=2,
                            key="doc",
                        ),
                    ],
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)

        mock_entry = AsyncMock()
        mock_entry.content.text = "Artifact content v2"
        mock_store = AsyncMock()
        mock_store.get_artifact = AsyncMock(return_value=mock_entry)

        orchestrator, _ = _build_pipeline(context_store=mock_store)
        await orchestrator.run(dag)
        assert orchestrator.status == DAGRunStatus.SUCCESS
        mock_store.get_artifact.assert_called_once_with("my-doc", 2)

    @pytest.mark.asyncio
    async def test_previous_result_ref_resolved(self):
        """context_ref(kind=previous_result) resolves from earlier node results."""
        plan = SkillPlan(
            name="ctx-prev",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="Step one"),
                ),
                SkillStep(
                    id="s2",
                    kind="text",
                    config=TextStepConfig(prompt="Step two needs s1 result"),
                    context_refs=[
                        ContextRef(
                            kind="previous_result", from_step="s1", key="prev"
                        ),
                    ],
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)

        text_handler = MockTextHandler({"Step one": "First answer"})
        orchestrator, _ = _build_pipeline(text_handler=text_handler)

        exec_dag = await orchestrator.run(dag)
        assert orchestrator.status == DAGRunStatus.SUCCESS

        # s2 should have been called (the orchestrator resolved "prev" from s1's result)
        assert len(text_handler.calls) == 2

    @pytest.mark.asyncio
    async def test_query_ref_resolved(self):
        """context_ref(kind=query) resolves via ContextStore.recall()."""
        from rh_cognitv.execution_platform.models import MemoryQuery

        plan = SkillPlan(
            name="ctx-query",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="Use recalled memories"),
                    context_refs=[
                        ContextRef(
                            kind="query",
                            query=MemoryQuery(text="relevant info", top_k=3),
                            key="recalled",
                        ),
                    ],
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)

        mock_result = AsyncMock()
        mock_result.entry.content.text = "Recalled memory content"
        mock_store = AsyncMock()
        mock_store.recall = AsyncMock(return_value=[mock_result])

        orchestrator, _ = _build_pipeline(context_store=mock_store)
        await orchestrator.run(dag)
        assert orchestrator.status == DAGRunStatus.SUCCESS
        mock_store.recall.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_context_store_skips_resolution(self):
        """Without a context_store, context_refs are ignored (backward compat)."""
        plan = SkillPlan(
            name="no-store",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="Just text"),
                    context_refs=[
                        ContextRef(kind="memory", id="mem-001", key="bg"),
                    ],
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)

        # No context_store configured — should not error
        orchestrator, _ = _build_pipeline(context_store=None)
        exec_dag = await orchestrator.run(dag)
        assert orchestrator.status == DAGRunStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_multiple_refs_on_one_step(self):
        """Multiple context_refs on a single step are all resolved."""
        plan = SkillPlan(
            name="multi-ref",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="Multiple contexts"),
                    context_refs=[
                        ContextRef(kind="memory", id="mem-a", key="bg1"),
                        ContextRef(kind="memory", id="mem-b", key="bg2"),
                    ],
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)

        mock_entry = AsyncMock()
        mock_entry.content.text = "Some memory"
        mock_store = AsyncMock()
        mock_store.get = AsyncMock(return_value=mock_entry)

        orchestrator, _ = _build_pipeline(context_store=mock_store)
        await orchestrator.run(dag)
        assert mock_store.get.call_count == 2


# ══════════════════════════════════════════════
# 3.10.3 — Full Pipeline with MockLLM
# ══════════════════════════════════════════════


class TestFullPipelineWithMockLLM:
    """Skill → Adapter → DAGOrchestrator → L3 handlers (MockLLM) →
    ExecutionDAG → ResultAdapter.from_result() → Skill.interpret() → SkillResult."""

    @pytest.mark.asyncio
    async def test_text_generation_end_to_end(self):
        """TextGenerationSkill full round-trip through all layers."""
        mock_handler = MockTextHandler({"Python": "Python is a great language!"})
        orchestrator, _ = _build_pipeline(text_handler=mock_handler)

        skill = TextGenerationSkill()
        input_data = SimpleInput(text="Tell me about Python.")
        plan = await skill.plan(input_data, SkillContext())
        dag = SkillToDAGAdapter().to_dag(plan)

        # L2 executes the DAG via L3
        exec_dag = await orchestrator.run(dag)
        assert orchestrator.status == DAGRunStatus.SUCCESS

        # Adapt results back to L1
        orch_result = ResultAdapter().from_result(exec_dag)
        assert orch_result.success

        # Skill interprets the results
        skill_result = await skill.interpret(orch_result)
        assert skill_result.success
        assert "Python is a great language!" in skill_result.output

    @pytest.mark.asyncio
    async def test_data_extraction_end_to_end(self):
        """DataExtractionSkill full round-trip."""
        json_response = json.dumps({"entities": ["Python", "Java"]})
        mock_handler = MockDataHandler({"Extract": json_response})
        orchestrator, _ = _build_pipeline(data_handler=mock_handler)

        skill = DataExtractionSkill(output_schema={"type": "object"})
        plan = await skill.plan(SimpleInput(text="Extract languages"), SkillContext())
        dag = SkillToDAGAdapter().to_dag(plan)

        exec_dag = await orchestrator.run(dag)
        orch_result = ResultAdapter().from_result(exec_dag)
        skill_result = await skill.interpret(orch_result)

        assert skill_result.success
        assert skill_result.output == json_response

    @pytest.mark.asyncio
    async def test_multi_step_end_to_end(self):
        """Multi-step skill full round-trip: plan → execute → result → interpret."""
        mock_text = MockTextHandler({
            "Research": "Research findings here",
            "Summarize": "Summary of findings",
        })
        orchestrator, _ = _build_pipeline(text_handler=mock_text)

        # Custom multi-step plan
        plan = SkillPlan(
            name="research-then-summarize",
            steps=[
                SkillStep(
                    id="research",
                    kind="text",
                    config=TextStepConfig(prompt="Research the topic"),
                ),
                SkillStep(
                    id="summarize",
                    kind="text",
                    config=TextStepConfig(prompt="Summarize findings"),
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)
        exec_dag = await orchestrator.run(dag)

        assert orchestrator.status == DAGRunStatus.SUCCESS
        orch_result = ResultAdapter().from_result(exec_dag)
        assert orch_result.success

        # Both steps produced results
        assert "research" in orch_result.step_results
        assert "summarize" in orch_result.step_results
        assert orch_result.step_results["research"].ok
        assert orch_result.step_results["summarize"].ok

    @pytest.mark.asyncio
    async def test_token_usage_tracked(self):
        """Token usage from MockLLM flows through to NodeResult."""
        mock_handler = MockTextHandler({"test": "response"})
        orchestrator, _ = _build_pipeline(text_handler=mock_handler)

        plan = SkillPlan(
            name="token-test",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="test prompt"),
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)
        exec_dag = await orchestrator.run(dag)

        orch_result = ResultAdapter().from_result(exec_dag)
        assert orch_result.step_results["s1"].token_usage is not None
        assert orch_result.step_results["s1"].token_usage.total > 0

    @pytest.mark.asyncio
    async def test_failed_step_propagates(self):
        """A step that fails is reported through the full pipeline."""

        class FailingHandler(EventHandlerProtocol[LLMResultData]):
            async def __call__(self, event, data, configs):
                return ExecutionResult(
                    ok=False,
                    error_message="LLM call failed",
                    error_category="LLM_ERROR",
                    metadata=ResultMetadata(),
                )

        orchestrator, _ = _build_pipeline(text_handler=FailingHandler())

        skill = TextGenerationSkill()
        plan = await skill.plan(SimpleInput(text="Will fail"), SkillContext())
        dag = SkillToDAGAdapter().to_dag(plan)
        exec_dag = await orchestrator.run(dag)

        assert orchestrator.status == DAGRunStatus.FAILED
        orch_result = ResultAdapter().from_result(exec_dag)
        assert not orch_result.success

        skill_result = await skill.interpret(orch_result)
        assert not skill_result.success

    @pytest.mark.asyncio
    async def test_state_snapshots_created(self):
        """State snapshots are recorded during execution."""
        orchestrator, state = _build_pipeline()

        plan = SkillPlan(
            name="state-test",
            steps=[
                SkillStep(id="a", kind="text", config=TextStepConfig(prompt="Step A")),
                SkillStep(id="b", kind="text", config=TextStepConfig(prompt="Step B")),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)
        await orchestrator.run(dag)

        # Each node execution creates a snapshot
        assert state.version_count >= 2

    @pytest.mark.asyncio
    async def test_context_flows_to_handler(self):
        """Serialized context from SkillContext is included in the prompt."""
        text_handler = MockTextHandler()
        orchestrator, _ = _build_pipeline(text_handler=text_handler)

        skill = TextGenerationSkill()
        ctx = SkillContext(serialized_context="BACKGROUND: relevant info here")
        plan = await skill.plan(SimpleInput(text="Answer this"), ctx)
        dag = SkillToDAGAdapter().to_dag(plan)

        await orchestrator.run(dag)
        # The handler should receive a prompt containing the context
        assert len(text_handler.calls) == 1
        assert "BACKGROUND: relevant info here" in text_handler.calls[0]
        assert "Answer this" in text_handler.calls[0]


# ══════════════════════════════════════════════
# 3.10.4 — Replan Flow
# ══════════════════════════════════════════════


class TestReplanFlow:
    """Skill returns SkillResult with replan set → framework builds
    new PlanDAG → re-executes."""

    @pytest.mark.asyncio
    async def test_replan_signal_from_interpret(self):
        """Skill.interpret() can return a replan request with new steps."""

        class ReplanningSkill(Skill):
            """Skill that replans after first execution."""

            def __init__(self):
                self._attempt = 0

            @property
            def name(self):
                return "replanning"

            @property
            def description(self):
                return "A skill that replans."

            async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
                return SkillPlan(
                    name="replanning",
                    steps=[
                        SkillStep(
                            id="initial",
                            kind="text",
                            config=TextStepConfig(prompt="Initial attempt"),
                        ),
                    ],
                )

            async def interpret(self, result: Any) -> SkillResult:
                self._attempt += 1
                if self._attempt == 1:
                    return SkillResult(
                        output=None,
                        success=True,
                        replan=ReplanRequest(
                            reason="Need a different approach",
                            suggested_steps=[
                                SkillStep(
                                    id="revised",
                                    kind="text",
                                    config=TextStepConfig(
                                        prompt="Revised approach"
                                    ),
                                ),
                            ],
                        ),
                        provenance=SkillProvenance(skill_name=self.name),
                    )
                return SkillResult(
                    output="Final answer",
                    success=True,
                    provenance=SkillProvenance(skill_name=self.name),
                )

        mock_handler = MockTextHandler({
            "Initial": "First try result",
            "Revised": "Second try result",
        })

        skill = ReplanningSkill()

        # --- First execution ---
        orchestrator, _ = _build_pipeline(text_handler=mock_handler)
        plan = await skill.plan(SimpleInput(text="Do something"), SkillContext())
        dag = SkillToDAGAdapter().to_dag(plan)
        exec_dag = await orchestrator.run(dag)

        orch_result = ResultAdapter().from_result(exec_dag)
        skill_result = await skill.interpret(orch_result)

        assert skill_result.replan is not None
        assert skill_result.replan.reason == "Need a different approach"
        assert len(skill_result.replan.suggested_steps) == 1

        # --- Replan: build new DAG from suggested steps ---
        new_plan = SkillPlan(
            name="replanning-revised",
            steps=skill_result.replan.suggested_steps,
        )
        new_dag = SkillToDAGAdapter().to_dag(new_plan)

        # Fresh orchestrator for new run
        orchestrator2, _ = _build_pipeline(text_handler=mock_handler)
        exec_dag2 = await orchestrator2.run(new_dag)

        orch_result2 = ResultAdapter().from_result(exec_dag2)
        skill_result2 = await skill.interpret(orch_result2)

        assert skill_result2.success
        assert skill_result2.output == "Final answer"
        assert skill_result2.replan is None

    @pytest.mark.asyncio
    async def test_replan_with_no_suggested_steps(self):
        """Replan without suggested_steps signals the framework to re-plan."""
        skill_result = SkillResult(
            output=None,
            success=True,
            replan=ReplanRequest(reason="Need different approach"),
            provenance=SkillProvenance(skill_name="test"),
        )
        assert skill_result.replan is not None
        assert skill_result.replan.suggested_steps is None
        assert skill_result.replan.reason == "Need different approach"

    @pytest.mark.asyncio
    async def test_replan_loop_converges(self):
        """Simulated replan loop: first attempt replans, second succeeds."""
        mock_handler = MockTextHandler({"attempt": "result"})

        # First execution
        orchestrator, _ = _build_pipeline(text_handler=mock_handler)
        plan = SkillPlan(
            name="converge",
            steps=[
                SkillStep(
                    id="attempt-1",
                    kind="text",
                    config=TextStepConfig(prompt="attempt 1"),
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)
        await orchestrator.run(dag)

        # Simulate interpret producing replan
        replan_result = SkillResult(
            output=None,
            success=True,
            replan=ReplanRequest(
                reason="Refine",
                suggested_steps=[
                    SkillStep(
                        id="attempt-2",
                        kind="text",
                        config=TextStepConfig(prompt="attempt 2"),
                    ),
                ],
            ),
            provenance=SkillProvenance(skill_name="converge"),
        )
        assert replan_result.replan is not None

        # Second execution
        new_plan = SkillPlan(
            name="converge-v2",
            steps=replan_result.replan.suggested_steps,
        )
        new_dag = SkillToDAGAdapter().to_dag(new_plan)

        orchestrator2, _ = _build_pipeline(text_handler=mock_handler)
        exec_dag2 = await orchestrator2.run(new_dag)
        assert orchestrator2.status == DAGRunStatus.SUCCESS

        # Second interpret succeeds
        orch_result2 = ResultAdapter().from_result(exec_dag2)
        final = SkillResult(
            output=orch_result2.step_results.get("attempt-2"),
            success=True,
            provenance=SkillProvenance(skill_name="converge"),
        )
        assert final.success
        assert final.replan is None


# ══════════════════════════════════════════════
# 3.10.5 — Output Validation Retry
# ══════════════════════════════════════════════


class TestOutputValidationRetry:
    """Skill's validate_output() rejects output → RetryableValidationError →
    RetryPolicy retries → passes on second attempt."""

    @pytest.mark.asyncio
    async def test_validation_retry_succeeds_on_second_attempt(self):
        """RetryPolicy.execute_with_retry retries RetryableValidationError from handler."""
        from rh_cognitv.execution_platform.policies import RetryPolicy
        from rh_cognitv.execution_platform.events import ExecutionEvent, TextPayload
        from rh_cognitv.execution_platform.models import EventKind

        call_count = 0

        class ValidatingTextHandler(EventHandlerProtocol[LLMResultData]):
            """Handler that fails validation on first call, passes on second."""

            async def __call__(self, event, data, configs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RetryableValidationError("Output quality too low")
                return ExecutionResult(
                    ok=True,
                    value=LLMResultData(
                        text="High quality response",
                        model="mock-llm",
                        token_usage=TokenUsage(
                            prompt_tokens=5, completion_tokens=3, total=8
                        ),
                    ),
                    metadata=ResultMetadata(),
                )

        retry = RetryPolicy(max_attempts=3, base_delay=0.01)
        handler = ValidatingTextHandler()
        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="Generate something good"),
        )

        result = await retry.execute_with_retry(handler, event, None, None)
        assert result.ok
        assert call_count == 2
        assert result.value.text == "High quality response"

    @pytest.mark.asyncio
    async def test_validation_retries_exhausted(self):
        """All retry attempts fail → node fails."""

        class AlwaysFailingHandler(EventHandlerProtocol[LLMResultData]):
            async def __call__(self, event, data, configs):
                raise RetryableValidationError("Always bad output")

        config = OrchestratorConfig(default_max_retries=2, default_retry_base_delay=0.01)
        orchestrator, _ = _build_pipeline(
            text_handler=AlwaysFailingHandler(), config=config
        )

        plan = SkillPlan(
            name="exhaust-retry",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="Will always fail"),
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)
        exec_dag = await orchestrator.run(dag)

        assert orchestrator.status == DAGRunStatus.FAILED

    @pytest.mark.asyncio
    async def test_skill_validate_output_hook(self):
        """Skill's validate_output() raises RetryableValidationError for bad output."""

        class StrictSkill(Skill):
            """Skill with strict output validation."""

            @property
            def name(self):
                return "strict"

            @property
            def description(self):
                return "Strict output."

            async def plan(self, input, context):
                return SkillPlan(
                    name="strict",
                    steps=[
                        SkillStep(
                            id="gen",
                            kind="text",
                            config=TextStepConfig(prompt="Generate"),
                        ),
                    ],
                )

            async def interpret(self, result):
                if hasattr(result, "step_results"):
                    step = result.step_results.get("gen")
                    if step and step.ok:
                        # Run validation
                        self.validate_output(step.value)
                        return SkillResult(
                            output=step.value,
                            success=True,
                            provenance=SkillProvenance(skill_name=self.name),
                        )
                return SkillResult(
                    output=None,
                    success=False,
                    provenance=SkillProvenance(skill_name=self.name),
                )

            def validate_output(self, output):
                if not output or "quality" not in str(output).lower():
                    raise RetryableValidationError(
                        "Output lacks quality markers"
                    )

        skill = StrictSkill()
        plan = await skill.plan(SimpleInput(text="test"), SkillContext())

        # Test that validation raises for bad output
        with pytest.raises(RetryableValidationError, match="quality markers"):
            skill.validate_output("bad output without markers")

        # Test that validation passes for good output
        skill.validate_output("This is a high quality response")

    @pytest.mark.asyncio
    async def test_retryable_validation_error_is_transient(self):
        """RetryableValidationError is a TransientError subclass for RetryPolicy."""
        from rh_cognitv.execution_platform.errors import TransientError

        err = RetryableValidationError("bad output")
        assert isinstance(err, TransientError)
        assert err.retryable is True

    @pytest.mark.asyncio
    async def test_validation_retry_in_full_pipeline(self):
        """Full integration: RetryPolicy.execute_with_retry bridges L1 validation
        errors (RetryableValidationError) with L3 retry mechanism."""
        from rh_cognitv.execution_platform.policies import RetryPolicy
        from rh_cognitv.execution_platform.events import ExecutionEvent, TextPayload
        from rh_cognitv.execution_platform.models import EventKind

        attempts = []

        class RetryableHandler(EventHandlerProtocol[LLMResultData]):
            async def __call__(self, event, data, configs):
                attempts.append(1)
                if len(attempts) < 3:
                    raise RetryableValidationError(
                        f"Attempt {len(attempts)} failed validation"
                    )
                return ExecutionResult(
                    ok=True,
                    value=LLMResultData(
                        text="Finally good output",
                        model="mock-llm",
                    ),
                    metadata=ResultMetadata(),
                )

        retry = RetryPolicy(max_attempts=5, base_delay=0.01)
        handler = RetryableHandler()
        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="test"),
        )

        result = await retry.execute_with_retry(handler, event, None, None)
        assert result.ok
        assert len(attempts) == 3
        assert result.value.text == "Finally good output"

        # Verify the result flows into a SkillResult via NodeResult
        from rh_cognitv.orchestrator.models import NodeResult

        node_result = NodeResult.from_execution_result(result)
        assert node_result.ok
        assert node_result.value == "Finally good output"

        # And a skill can interpret it
        skill = TextGenerationSkill()
        orch_result = OrchestratorResult(
            step_results={"generate": node_result},
            success=True,
        )
        skill_result = await skill.interpret(orch_result)
        assert skill_result.success
        assert "Finally good output" in skill_result.output


# ══════════════════════════════════════════════
# Cross-cutting integration scenarios
# ══════════════════════════════════════════════


class TestCrossCuttingIntegration:
    """Additional integration scenarios testing layer interactions."""

    @pytest.mark.asyncio
    async def test_mixed_node_types_in_one_plan(self):
        """Skill plan with text + data steps runs correctly end-to-end."""
        text_handler = MockTextHandler({"Describe": "Text description"})
        data_handler = MockDataHandler(
            {"Extract": json.dumps({"summary": "extracted"})}
        )
        orchestrator, _ = _build_pipeline(
            text_handler=text_handler, data_handler=data_handler
        )

        plan = SkillPlan(
            name="mixed",
            steps=[
                SkillStep(
                    id="describe",
                    kind="text",
                    config=TextStepConfig(prompt="Describe the topic"),
                ),
                SkillStep(
                    id="extract",
                    kind="data",
                    config=DataStepConfig(prompt="Extract key info"),
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)
        exec_dag = await orchestrator.run(dag)

        assert orchestrator.status == DAGRunStatus.SUCCESS
        orch_result = ResultAdapter().from_result(exec_dag)
        assert orch_result.step_results["describe"].ok
        assert orch_result.step_results["extract"].ok

    @pytest.mark.asyncio
    async def test_skill_provenance_in_result(self):
        """SkillResult carries provenance metadata through full pipeline."""
        mock_handler = MockTextHandler({"test": "response"})
        orchestrator, _ = _build_pipeline(text_handler=mock_handler)

        skill = TextGenerationSkill()
        plan = await skill.plan(SimpleInput(text="test"), SkillContext())
        dag = SkillToDAGAdapter().to_dag(plan)
        exec_dag = await orchestrator.run(dag)

        orch_result = ResultAdapter().from_result(exec_dag)
        skill_result = await skill.interpret(orch_result)

        assert skill_result.provenance is not None
        assert skill_result.provenance.skill_name == "text_generation"

    @pytest.mark.asyncio
    async def test_execution_state_nesting(self):
        """DAGOrchestrator increments/decrements nesting level during run."""
        orchestrator, state = _build_pipeline()

        assert state.level == 0
        plan = SkillPlan(
            name="nesting-test",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="test"),
                ),
            ],
        )
        dag = SkillToDAGAdapter().to_dag(plan)
        await orchestrator.run(dag)

        # After run completes, level should be back to 0
        assert state.level == 0

    @pytest.mark.asyncio
    async def test_orchestrator_result_adapter_empty_dag(self):
        """ResultAdapter handles an empty ExecutionDAG gracefully."""
        from rh_cognitv.orchestrator.execution_dag import ExecutionDAG

        empty_dag = ExecutionDAG()
        result = ResultAdapter().from_result(empty_dag)
        assert result.success
        assert len(result.step_results) == 0

    @pytest.mark.asyncio
    async def test_skill_with_serialized_context_and_context_refs(self):
        """Both serialized_context (in prompt) and context_refs (resolved at execution)
        work together in a single pipeline run."""
        mock_entry = AsyncMock()
        mock_entry.content.text = "Resolved memory data"
        mock_store = AsyncMock()
        mock_store.get = AsyncMock(return_value=mock_entry)

        text_handler = MockTextHandler()
        orchestrator, _ = _build_pipeline(
            text_handler=text_handler, context_store=mock_store
        )

        # Skill produces a plan with both serialized_context in prompt and context_refs
        skill = TextGenerationSkill()
        ctx = SkillContext(serialized_context="PRE-LOADED: base context")
        plan = await skill.plan(SimpleInput(text="Answer with context"), ctx)

        # Manually add a context_ref to the first step
        plan.steps[0].context_refs = [
            ContextRef(kind="memory", id="mem-dynamic", key="dynamic_ctx"),
        ]

        dag = SkillToDAGAdapter().to_dag(plan)
        exec_dag = await orchestrator.run(dag)

        assert orchestrator.status == DAGRunStatus.SUCCESS
        # Serialized context is in the prompt
        assert "PRE-LOADED" in text_handler.calls[0]
        # Memory ref was resolved
        mock_store.get.assert_called_once_with("mem-dynamic")
