"""
Tests for cognitive/models.py — Phase 3.1

Model construction, Pydantic validation, serialization round-trips,
defaults, and L3 alignment checks.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from rh_cognitv.cognitive.models import (
    BuiltPrompt,
    CompletionResult,
    ContextRef,
    CreateArtifact,
    CreateMemory,
    DataStepConfig,
    FunctionStepConfig,
    Message,
    MessageRole,
    ReplanRequest,
    SkillConfig,
    SkillConstraints,
    SkillContext,
    SkillPlan,
    SkillProvenance,
    SkillResult,
    SkillStep,
    TextStepConfig,
    ToolCall,
    ToolResult,
    ToolStepConfig,
)
from rh_cognitv.execution_platform.models import (
    BudgetSnapshot,
    MemoryQuery,
    TokenBudget,
)


# ──────────────────────────────────────────────
# Message & LLM Types
# ──────────────────────────────────────────────


class TestMessage:
    def test_construction(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.name is None

    def test_tool_message_with_name(self):
        msg = Message(role=MessageRole.TOOL, content="result", name="calculator")
        assert msg.role == MessageRole.TOOL
        assert msg.name == "calculator"

    def test_all_roles(self):
        for role in MessageRole:
            msg = Message(role=role, content="test")
            assert msg.role == role

    def test_serialization_round_trip(self):
        msg = Message(role=MessageRole.ASSISTANT, content="answer")
        data = msg.model_dump()
        restored = Message.model_validate(data)
        assert restored == msg

    def test_json_round_trip(self):
        msg = Message(role=MessageRole.USER, content="hi")
        json_str = msg.model_dump_json()
        restored = Message.model_validate_json(json_str)
        assert restored == msg


class TestMessageRole:
    def test_values(self):
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL == "tool"


class TestCompletionResult:
    def test_defaults(self):
        r = CompletionResult(text="Hello")
        assert r.text == "Hello"
        assert r.thinking is None
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.total_tokens == 0
        assert r.model == ""
        assert r.finish_reason == ""

    def test_full_construction(self):
        r = CompletionResult(
            text="answer",
            thinking="let me think...",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4",
            finish_reason="stop",
        )
        assert r.prompt_tokens == 100
        assert r.model == "gpt-4"

    def test_serialization_round_trip(self):
        r = CompletionResult(text="test", prompt_tokens=10)
        data = r.model_dump()
        restored = CompletionResult.model_validate(data)
        assert restored == r


class TestToolCall:
    def test_construction(self):
        tc = ToolCall(name="search", arguments={"q": "hello"})
        assert tc.name == "search"
        assert tc.arguments == {"q": "hello"}
        assert tc.id  # auto-generated ULID

    def test_defaults(self):
        tc = ToolCall(name="noop")
        assert tc.arguments == {}


class TestToolResult:
    def test_defaults(self):
        r = ToolResult()
        assert r.text == ""
        assert r.tool_calls == []
        assert r.prompt_tokens == 0

    def test_with_tool_calls(self):
        r = ToolResult(
            text="I'll search for that",
            tool_calls=[
                ToolCall(name="search", arguments={"q": "info"}),
            ],
        )
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"

    def test_serialization_round_trip(self):
        r = ToolResult(text="done", tool_calls=[ToolCall(name="fn")])
        data = r.model_dump()
        restored = ToolResult.model_validate(data)
        assert restored.tool_calls[0].name == "fn"


# ──────────────────────────────────────────────
# BuiltPrompt
# ──────────────────────────────────────────────


class TestBuiltPrompt:
    def test_construction(self):
        bp = BuiltPrompt(prompt="Summarize this", system_prompt="You are helpful")
        assert bp.prompt == "Summarize this"
        assert bp.system_prompt == "You are helpful"

    def test_no_system_prompt(self):
        bp = BuiltPrompt(prompt="Hello")
        assert bp.system_prompt is None

    def test_serialization_round_trip(self):
        bp = BuiltPrompt(prompt="test", system_prompt="sys")
        restored = BuiltPrompt.model_validate(bp.model_dump())
        assert restored == bp


# ──────────────────────────────────────────────
# ContextRef (DD-L1-07)
# ──────────────────────────────────────────────


class TestContextRef:
    def test_memory_ref(self):
        ref = ContextRef(kind="memory", id="mem-123", key="background")
        assert ref.kind == "memory"
        assert ref.id == "mem-123"
        assert ref.key == "background"

    def test_artifact_ref(self):
        ref = ContextRef(kind="artifact", slug="draft-v1", version=2)
        assert ref.kind == "artifact"
        assert ref.slug == "draft-v1"
        assert ref.version == 2

    def test_query_ref(self):
        q = MemoryQuery(text="relevant info", top_k=5)
        ref = ContextRef(kind="query", query=q)
        assert ref.kind == "query"
        assert ref.query is not None
        assert ref.query.top_k == 5

    def test_previous_result_ref(self):
        ref = ContextRef(kind="previous_result", from_step="step-1", key="research")
        assert ref.kind == "previous_result"
        assert ref.from_step == "step-1"
        assert ref.key == "research"

    def test_default_key(self):
        ref = ContextRef(kind="memory", id="x")
        assert ref.key == "context"

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValidationError):
            ContextRef(kind="invalid", id="x")  # type: ignore[arg-type]

    def test_serialization_round_trip(self):
        ref = ContextRef(kind="artifact", slug="code", version=1, key="src")
        data = ref.model_dump()
        restored = ContextRef.model_validate(data)
        assert restored == ref

    def test_json_round_trip(self):
        q = MemoryQuery(text="find", role="semantic", top_k=3)
        ref = ContextRef(kind="query", query=q, key="ctx")
        json_str = ref.model_dump_json()
        restored = ContextRef.model_validate_json(json_str)
        assert restored.query.text == "find"


# ──────────────────────────────────────────────
# Step Configs
# ──────────────────────────────────────────────


class TestTextStepConfig:
    def test_construction(self):
        c = TextStepConfig(prompt="Hello", system_prompt="Be nice")
        assert c.prompt == "Hello"
        assert c.system_prompt == "Be nice"

    def test_defaults(self):
        c = TextStepConfig(prompt="Hi")
        assert c.system_prompt is None
        assert c.model is None
        assert c.temperature is None
        assert c.max_tokens is None

    def test_all_fields(self):
        c = TextStepConfig(
            prompt="p",
            system_prompt="s",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
        )
        assert c.model == "gpt-4"
        assert c.temperature == 0.7
        assert c.max_tokens == 1000


class TestDataStepConfig:
    def test_construction(self):
        c = DataStepConfig(prompt="Extract", output_schema={"type": "object"})
        assert c.prompt == "Extract"
        assert c.output_schema == {"type": "object"}

    def test_defaults(self):
        c = DataStepConfig(prompt="X")
        assert c.output_schema is None
        assert c.model is None


class TestFunctionStepConfig:
    def test_construction(self):
        c = FunctionStepConfig(
            function_name="process",
            args=[1, 2],
            kwargs={"flag": True},
        )
        assert c.function_name == "process"
        assert c.args == [1, 2]
        assert c.kwargs == {"flag": True}

    def test_defaults(self):
        c = FunctionStepConfig(function_name="noop")
        assert c.args == []
        assert c.kwargs == {}


class TestToolStepConfig:
    def test_construction(self):
        c = ToolStepConfig(
            prompt="Use tools",
            tools=[{"type": "function", "function": {"name": "search"}}],
            model="gpt-4",
        )
        assert c.prompt == "Use tools"
        assert len(c.tools) == 1
        assert c.model == "gpt-4"

    def test_defaults(self):
        c = ToolStepConfig(prompt="Hello")
        assert c.tools == []
        assert c.model is None


# ──────────────────────────────────────────────
# SkillStep
# ──────────────────────────────────────────────


class TestSkillStep:
    def test_text_step(self):
        step = SkillStep(
            id="s1",
            kind="text",
            config=TextStepConfig(prompt="Hello"),
        )
        assert step.id == "s1"
        assert step.kind == "text"
        assert isinstance(step.config, TextStepConfig)
        assert step.context_refs == []
        assert step.depends_on == []

    def test_data_step(self):
        step = SkillStep(
            id="s2",
            kind="data",
            config=DataStepConfig(prompt="Extract"),
        )
        assert step.kind == "data"

    def test_function_step(self):
        step = SkillStep(
            id="s3",
            kind="function",
            config=FunctionStepConfig(function_name="process"),
        )
        assert step.kind == "function"

    def test_tool_step(self):
        step = SkillStep(
            id="s4",
            kind="tool",
            config=ToolStepConfig(prompt="Use tool"),
        )
        assert step.kind == "tool"

    def test_with_context_refs(self):
        step = SkillStep(
            id="s1",
            kind="text",
            config=TextStepConfig(prompt="Hello"),
            context_refs=[
                ContextRef(kind="memory", id="m1"),
                ContextRef(kind="previous_result", from_step="s0", key="data"),
            ],
        )
        assert len(step.context_refs) == 2
        assert step.context_refs[0].kind == "memory"
        assert step.context_refs[1].from_step == "s0"

    def test_with_depends_on(self):
        step = SkillStep(
            id="s2",
            kind="text",
            config=TextStepConfig(prompt="Next"),
            depends_on=["s1"],
        )
        assert step.depends_on == ["s1"]

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValidationError):
            SkillStep(
                id="bad",
                kind="invalid",  # type: ignore[arg-type]
                config=TextStepConfig(prompt="x"),
            )

    def test_serialization_round_trip(self):
        step = SkillStep(
            id="s1",
            kind="text",
            config=TextStepConfig(prompt="Hello", model="gpt-4"),
            context_refs=[ContextRef(kind="memory", id="m1")],
        )
        data = step.model_dump()
        restored = SkillStep.model_validate(data)
        assert restored.id == "s1"
        assert restored.context_refs[0].id == "m1"


# ──────────────────────────────────────────────
# SkillConstraints
# ──────────────────────────────────────────────


class TestSkillConstraints:
    def test_defaults(self):
        c = SkillConstraints()
        assert c.timeout_seconds is None
        assert c.max_retries is None
        assert c.max_tokens is None

    def test_full_construction(self):
        c = SkillConstraints(timeout_seconds=30.0, max_retries=3, max_tokens=4000)
        assert c.timeout_seconds == 30.0
        assert c.max_retries == 3
        assert c.max_tokens == 4000


# ──────────────────────────────────────────────
# SkillPlan
# ──────────────────────────────────────────────


class TestSkillPlan:
    def test_minimal(self):
        plan = SkillPlan(name="test", steps=[])
        assert plan.name == "test"
        assert plan.steps == []
        assert plan.constraints is None

    def test_single_step(self):
        plan = SkillPlan(
            name="summarize",
            steps=[
                SkillStep(
                    id="gen",
                    kind="text",
                    config=TextStepConfig(prompt="Summarize {input.text}"),
                ),
            ],
        )
        assert len(plan.steps) == 1

    def test_multi_step_with_constraints(self):
        plan = SkillPlan(
            name="research",
            steps=[
                SkillStep(
                    id="research",
                    kind="text",
                    config=TextStepConfig(prompt="Research topic"),
                ),
                SkillStep(
                    id="extract",
                    kind="data",
                    config=DataStepConfig(prompt="Extract key points"),
                    depends_on=["research"],
                    context_refs=[
                        ContextRef(kind="previous_result", from_step="research"),
                    ],
                ),
                SkillStep(
                    id="synthesize",
                    kind="text",
                    config=TextStepConfig(prompt="Synthesize findings"),
                    depends_on=["extract"],
                ),
            ],
            constraints=SkillConstraints(timeout_seconds=60.0, max_retries=2),
        )
        assert len(plan.steps) == 3
        assert plan.steps[1].depends_on == ["research"]
        assert plan.constraints.timeout_seconds == 60.0

    def test_serialization_round_trip(self):
        plan = SkillPlan(
            name="test",
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="Hello"),
                ),
            ],
            constraints=SkillConstraints(max_retries=2),
        )
        data = plan.model_dump()
        restored = SkillPlan.model_validate(data)
        assert restored.name == plan.name
        assert len(restored.steps) == 1

    def test_json_round_trip(self):
        plan = SkillPlan(
            name="p",
            steps=[
                SkillStep(id="s", kind="function", config=FunctionStepConfig(function_name="fn")),
            ],
        )
        json_str = plan.model_dump_json()
        restored = SkillPlan.model_validate_json(json_str)
        assert restored.steps[0].kind == "function"


# ──────────────────────────────────────────────
# SkillContext
# ──────────────────────────────────────────────


class TestSkillContext:
    def test_defaults(self):
        ctx = SkillContext()
        assert ctx.memories == []
        assert ctx.artifacts == []
        assert ctx.budget is None
        assert ctx.budget_snapshot is None
        assert ctx.serialized_context == ""
        assert ctx.remaining_prompt_tokens == 0
        assert ctx.ext == {}

    def test_with_budget(self):
        budget = TokenBudget(total=8000, working=1000, episodic=2000, semantic=3000)
        snapshot = BudgetSnapshot(
            tokens_remaining=7000,
            calls_remaining=10,
            time_remaining_seconds=60.0,
        )
        ctx = SkillContext(
            budget=budget,
            budget_snapshot=snapshot,
            serialized_context="## Context\nSome memory text",
            remaining_prompt_tokens=5000,
        )
        assert ctx.budget.total == 8000
        assert ctx.budget_snapshot.tokens_remaining == 7000
        assert ctx.remaining_prompt_tokens == 5000

    def test_with_memories_and_artifacts(self):
        ctx = SkillContext(
            memories=["memory1", "memory2"],
            artifacts=["artifact1"],
        )
        assert len(ctx.memories) == 2
        assert len(ctx.artifacts) == 1

    def test_serialization_round_trip(self):
        ctx = SkillContext(
            serialized_context="test context",
            remaining_prompt_tokens=100,
            ext={"extra": "data"},
        )
        data = ctx.model_dump()
        restored = SkillContext.model_validate(data)
        assert restored.serialized_context == "test context"
        assert restored.ext["extra"] == "data"


# ──────────────────────────────────────────────
# SkillProvenance
# ──────────────────────────────────────────────


class TestSkillProvenance:
    def test_construction(self):
        p = SkillProvenance(
            skill_name="summarize",
            input_hash="abc123",
            context_memory_ids=["id1", "id2"],
        )
        assert p.skill_name == "summarize"
        assert p.input_hash == "abc123"
        assert len(p.context_memory_ids) == 2

    def test_defaults(self):
        p = SkillProvenance(skill_name="test")
        assert p.input_hash == ""
        assert p.context_memory_ids == []


# ──────────────────────────────────────────────
# CreateMemory / CreateArtifact
# ──────────────────────────────────────────────


class TestCreateMemory:
    def test_construction(self):
        m = CreateMemory(
            text="User prefers dark mode",
            role="semantic",
            origin="told",
            source="user-message",
            tags=["preference"],
        )
        assert m.text == "User prefers dark mode"
        assert m.role == "semantic"
        assert m.origin == "told"
        assert m.tags == ["preference"]

    def test_defaults(self):
        m = CreateMemory(text="fact")
        assert m.role == "semantic"
        assert m.shape == "atom"
        assert m.origin == "inferred"
        assert m.source == ""
        assert m.tags == []


class TestCreateArtifact:
    def test_construction(self):
        a = CreateArtifact(
            text="# Summary\nContent here",
            type="document",
            slug="summary-v1",
            intent="Summarize research",
            tags=["output"],
        )
        assert a.slug == "summary-v1"
        assert a.intent == "Summarize research"

    def test_defaults(self):
        a = CreateArtifact(text="code", slug="snippet")
        assert a.type == "document"
        assert a.intent == ""
        assert a.tags == []


# ──────────────────────────────────────────────
# ReplanRequest
# ──────────────────────────────────────────────


class TestReplanRequest:
    def test_construction(self):
        r = ReplanRequest(reason="Need different approach")
        assert r.reason == "Need different approach"
        assert r.suggested_steps is None

    def test_with_suggested_steps(self):
        r = ReplanRequest(
            reason="Try extraction instead",
            suggested_steps=[
                SkillStep(
                    id="new-s1",
                    kind="data",
                    config=DataStepConfig(prompt="Extract"),
                ),
            ],
        )
        assert len(r.suggested_steps) == 1
        assert r.suggested_steps[0].kind == "data"


# ──────────────────────────────────────────────
# SkillResult
# ──────────────────────────────────────────────


class TestSkillResult:
    def test_defaults(self):
        r = SkillResult()
        assert r.output is None
        assert r.success is True
        assert r.error_message is None
        assert r.provenance is None
        assert r.suggested_memories == []
        assert r.suggested_artifacts == []
        assert r.replan is None

    def test_success_with_output(self):
        r = SkillResult(
            output={"summary": "Short text"},
            success=True,
            provenance=SkillProvenance(
                skill_name="summarize",
                input_hash="abc",
                context_memory_ids=["m1"],
            ),
        )
        assert r.output["summary"] == "Short text"
        assert r.provenance.skill_name == "summarize"

    def test_failure(self):
        r = SkillResult(
            success=False,
            error_message="LLM returned empty",
        )
        assert r.success is False
        assert r.error_message == "LLM returned empty"

    def test_with_suggested_memories_and_artifacts(self):
        r = SkillResult(
            output="done",
            suggested_memories=[
                CreateMemory(text="new fact", source="summarize"),
            ],
            suggested_artifacts=[
                CreateArtifact(text="output", slug="result-v1"),
            ],
        )
        assert len(r.suggested_memories) == 1
        assert len(r.suggested_artifacts) == 1

    def test_with_replan(self):
        r = SkillResult(
            output="partial",
            success=True,
            replan=ReplanRequest(reason="revise approach"),
        )
        assert r.replan is not None
        assert r.replan.reason == "revise approach"

    def test_serialization_round_trip(self):
        r = SkillResult(
            output="test",
            success=True,
            provenance=SkillProvenance(skill_name="test-skill"),
            suggested_memories=[CreateMemory(text="memo")],
        )
        data = r.model_dump()
        restored = SkillResult.model_validate(data)
        assert restored.provenance.skill_name == "test-skill"
        assert restored.suggested_memories[0].text == "memo"


# ──────────────────────────────────────────────
# SkillConfig (DD-L1-01 Option C)
# ──────────────────────────────────────────────


class TestSkillConfig:
    def test_minimal(self):
        c = SkillConfig(
            name="summarize",
            prompt_template="Summarize: {input.text}",
        )
        assert c.name == "summarize"
        assert c.description == ""
        assert c.system_prompt is None
        assert c.input_schema is None
        assert c.output_schema is None
        assert c.memory_query is None

    def test_full_construction(self):
        class SumInput(BaseModel):
            text: str

        class SumOutput(BaseModel):
            summary: str

        c = SkillConfig(
            name="summarize",
            description="Summarize input text",
            prompt_template="Summarize: {text}",
            system_prompt="You are a summarization expert",
            input_schema=SumInput,
            output_schema=SumOutput,
            memory_query=MemoryQuery(role="semantic", top_k=5),
            model="gpt-4",
            temperature=0.3,
            max_tokens=500,
            tags=["text", "summarization"],
        )
        assert c.output_schema is SumOutput
        assert c.memory_query.top_k == 5
        assert c.tags == ["text", "summarization"]

    def test_serialization_round_trip(self):
        c = SkillConfig(
            name="test",
            prompt_template="{input}",
            model="gpt-4",
        )
        data = c.model_dump()
        restored = SkillConfig.model_validate(data)
        assert restored.name == "test"
        assert restored.model == "gpt-4"


# ──────────────────────────────────────────────
# L3 Alignment Checks
# ──────────────────────────────────────────────


class TestL3Alignment:
    """Verify that L1 models align with L3 types they'll interact with."""

    def test_context_ref_query_uses_l3_memory_query(self):
        """ContextRef.query is an L3 MemoryQuery — not a custom type."""
        ref = ContextRef(
            kind="query",
            query=MemoryQuery(text="search", role="semantic", top_k=3),
        )
        assert isinstance(ref.query, MemoryQuery)

    def test_skill_context_budget_uses_l3_token_budget(self):
        """SkillContext.budget is an L3 TokenBudget."""
        ctx = SkillContext(
            budget=TokenBudget(total=8000, episodic=2000),
        )
        assert isinstance(ctx.budget, TokenBudget)

    def test_skill_context_snapshot_uses_l3_budget_snapshot(self):
        """SkillContext.budget_snapshot is an L3 BudgetSnapshot."""
        ctx = SkillContext(
            budget_snapshot=BudgetSnapshot(
                tokens_remaining=5000,
                calls_remaining=10,
                time_remaining_seconds=30.0,
            ),
        )
        assert isinstance(ctx.budget_snapshot, BudgetSnapshot)

    def test_completion_result_aligns_with_llm_result_data(self):
        """CompletionResult fields align with L3's LLMResultData."""
        from rh_cognitv.execution_platform.models import LLMResultData

        # Both carry: text, thinking, token usage, model, finish_reason
        l3 = LLMResultData(
            text="hello",
            thinking="hmm",
            model="gpt-4",
            finish_reason="stop",
        )
        l1 = CompletionResult(
            text=l3.text,
            thinking=l3.thinking,
            prompt_tokens=l3.token_usage.prompt_tokens,
            completion_tokens=l3.token_usage.completion_tokens,
            total_tokens=l3.token_usage.total,
            model=l3.model,
            finish_reason=l3.finish_reason,
        )
        assert l1.text == l3.text
        assert l1.thinking == l3.thinking
        assert l1.model == l3.model

    def test_built_prompt_aligns_with_text_payload(self):
        """BuiltPrompt fields map 1:1 to TextPayload's prompt + system_prompt."""
        from rh_cognitv.execution_platform.events import TextPayload

        bp = BuiltPrompt(prompt="Hello", system_prompt="Be nice")
        payload = TextPayload(prompt=bp.prompt, system_prompt=bp.system_prompt)
        assert payload.prompt == bp.prompt
        assert payload.system_prompt == bp.system_prompt

    def test_step_config_aligns_with_l2_nodes(self):
        """Step config fields match corresponding L2 node constructor params."""
        from rh_cognitv.orchestrator.nodes import (
            DataNode,
            FunctionNode,
            TextNode,
            ToolNode,
        )

        # TextStepConfig → TextNode
        tc = TextStepConfig(prompt="p", system_prompt="s", model="m", temperature=0.5, max_tokens=100)
        tn = TextNode(prompt=tc.prompt, system_prompt=tc.system_prompt, model=tc.model, temperature=tc.temperature, max_tokens=tc.max_tokens)
        assert tn.prompt == tc.prompt

        # DataStepConfig → DataNode
        dc = DataStepConfig(prompt="p", output_schema={"type": "object"}, model="m")
        dn = DataNode(prompt=dc.prompt, output_schema=dc.output_schema, model=dc.model)
        assert dn.prompt == dc.prompt

        # FunctionStepConfig → FunctionNode
        fc = FunctionStepConfig(function_name="fn", args=[1], kwargs={"k": "v"})
        fn = FunctionNode(function_name=fc.function_name, args=fc.args, kwargs=fc.kwargs)
        assert fn.function_name == fc.function_name

        # ToolStepConfig → ToolNode
        tlc = ToolStepConfig(prompt="p", tools=[{"t": 1}], model="m")
        tln = ToolNode(prompt=tlc.prompt, tools=tlc.tools, model=tlc.model)
        assert tln.prompt == tlc.prompt
