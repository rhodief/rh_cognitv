"""
Tests for cognitive/protocols.py — Phase 3.1

Protocol structural checks:
- ABC enforcement (can't instantiate abstract classes)
- Concrete implementations satisfy protocol contracts
- MetaSkillProtocol extends SkillProtocol
- ISP: StreamingLLMProtocol extends LLMProtocol, EmbeddingProtocol is standalone
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from rh_cognitv.cognitive.protocols import (
    ContextSerializerProtocol,
    EmbeddingProtocol,
    LLMProtocol,
    MetaSkillProtocol,
    PromptProtocol,
    SkillProtocol,
    StreamingLLMProtocol,
)
from rh_cognitv.cognitive.models import (
    BuiltPrompt,
    CompletionResult,
    Message,
    MessageRole,
    SkillConfig,
    SkillContext,
    SkillPlan,
    SkillResult,
    SkillStep,
    TextStepConfig,
    ToolResult,
)


# ──────────────────────────────────────────────
# Helpers — minimal concrete implementations
# ──────────────────────────────────────────────


class ConcreteSkill(SkillProtocol):
    """Minimal concrete Skill for testing."""

    @property
    def name(self) -> str:
        return "test-skill"

    @property
    def description(self) -> str:
        return "A test skill"

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        return SkillPlan(
            name="test-plan",
            steps=[
                SkillStep(
                    id="step-1",
                    kind="text",
                    config=TextStepConfig(prompt="Hello"),
                ),
            ],
        )

    async def interpret(self, result: Any) -> SkillResult:
        return SkillResult(output="done", success=True)


class ConcreteLLM(LLMProtocol):
    """Minimal concrete LLM for testing."""

    async def complete(self, messages: list[Message]) -> CompletionResult:
        return CompletionResult(text="response")

    async def complete_structured(
        self, messages: list[Message], schema: type
    ) -> Any:
        return schema()

    async def complete_with_tools(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ToolResult:
        return ToolResult(text="tool response")


class ConcreteStreamingLLM(StreamingLLMProtocol):
    """Minimal concrete StreamingLLM for testing ISP."""

    async def complete(self, messages: list[Message]) -> CompletionResult:
        return CompletionResult(text="response")

    async def complete_structured(
        self, messages: list[Message], schema: type
    ) -> Any:
        return schema()

    async def complete_with_tools(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ToolResult:
        return ToolResult(text="tool response")

    async def stream(self, messages: list[Message]) -> Any:
        async def _gen():
            yield "chunk1"
            yield "chunk2"
        return _gen()


class ConcreteEmbedding(EmbeddingProtocol):
    """Minimal concrete EmbeddingProtocol for testing ISP."""

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class ConcretePrompt(PromptProtocol):
    """Minimal concrete PromptProtocol for testing."""

    def build(self) -> BuiltPrompt:
        return BuiltPrompt(prompt="Hello", system_prompt="You are helpful")


class ConcreteSerializer(ContextSerializerProtocol):
    """Minimal concrete ContextSerializerProtocol for testing."""

    def serialize(self, memories: list[Any], artifacts: list[Any]) -> str:
        return f"memories={len(memories)}, artifacts={len(artifacts)}"


class ConcreteMetaSkill(MetaSkillProtocol):
    """Minimal concrete MetaSkill for testing."""

    @property
    def name(self) -> str:
        return "meta-skill"

    @property
    def description(self) -> str:
        return "A meta skill"

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        return SkillPlan(name="meta-plan", steps=[])

    async def interpret(self, result: Any) -> SkillResult:
        return SkillResult(output="meta-done", success=True)

    async def generate_skill(
        self, description: str, context: SkillContext
    ) -> SkillConfig:
        raise NotImplementedError("V2")

    async def generate_dag(self, description: str, context: SkillContext) -> Any:
        raise NotImplementedError("V2")


# ──────────────────────────────────────────────
# ABC Enforcement Tests
# ──────────────────────────────────────────────


class TestABCEnforcement:
    """Verify ABCs can't be instantiated without implementing all methods."""

    def test_skill_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            SkillProtocol()  # type: ignore[abstract]

    def test_llm_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            LLMProtocol()  # type: ignore[abstract]

    def test_streaming_llm_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            StreamingLLMProtocol()  # type: ignore[abstract]

    def test_embedding_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            EmbeddingProtocol()  # type: ignore[abstract]

    def test_prompt_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            PromptProtocol()  # type: ignore[abstract]

    def test_context_serializer_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            ContextSerializerProtocol()  # type: ignore[abstract]

    def test_meta_skill_protocol_is_abstract(self):
        with pytest.raises(TypeError):
            MetaSkillProtocol()  # type: ignore[abstract]


# ──────────────────────────────────────────────
# Concrete Implementation Tests
# ──────────────────────────────────────────────


class TestSkillProtocol:
    """Test that a concrete Skill satisfies SkillProtocol."""

    def test_instantiation(self):
        skill = ConcreteSkill()
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"

    @pytest.mark.asyncio
    async def test_plan_returns_skill_plan(self):
        class TestInput(BaseModel):
            pass

        skill = ConcreteSkill()
        plan = await skill.plan(TestInput(), SkillContext())
        assert isinstance(plan, SkillPlan)
        assert plan.name == "test-plan"
        assert len(plan.steps) == 1
        assert plan.steps[0].kind == "text"

    @pytest.mark.asyncio
    async def test_interpret_returns_skill_result(self):
        skill = ConcreteSkill()
        result = await skill.interpret({"value": "test"})
        assert isinstance(result, SkillResult)
        assert result.success is True
        assert result.output == "done"

    @pytest.mark.asyncio
    async def test_validate_output_default_returns_true(self):
        skill = ConcreteSkill()
        assert await skill.validate_output("anything") is True

    def test_isinstance_check(self):
        skill = ConcreteSkill()
        assert isinstance(skill, SkillProtocol)


class TestLLMProtocol:
    """Test that a concrete LLM satisfies LLMProtocol."""

    def test_instantiation(self):
        llm = ConcreteLLM()
        assert isinstance(llm, LLMProtocol)

    @pytest.mark.asyncio
    async def test_complete(self):
        llm = ConcreteLLM()
        messages = [Message(role=MessageRole.USER, content="Hi")]
        result = await llm.complete(messages)
        assert isinstance(result, CompletionResult)
        assert result.text == "response"

    @pytest.mark.asyncio
    async def test_complete_structured(self):
        llm = ConcreteLLM()
        messages = [Message(role=MessageRole.USER, content="Extract")]

        class Output(BaseModel):
            pass

        result = await llm.complete_structured(messages, Output)
        assert isinstance(result, Output)

    @pytest.mark.asyncio
    async def test_complete_with_tools(self):
        llm = ConcreteLLM()
        messages = [Message(role=MessageRole.USER, content="Use tool")]
        result = await llm.complete_with_tools(messages, [{"type": "function"}])
        assert isinstance(result, ToolResult)
        assert result.text == "tool response"


class TestStreamingLLMProtocol:
    """Test ISP: StreamingLLMProtocol extends LLMProtocol."""

    def test_is_llm_protocol(self):
        llm = ConcreteStreamingLLM()
        assert isinstance(llm, LLMProtocol)
        assert isinstance(llm, StreamingLLMProtocol)

    @pytest.mark.asyncio
    async def test_stream_method(self):
        llm = ConcreteStreamingLLM()
        messages = [Message(role=MessageRole.USER, content="Stream")]
        gen = await llm.stream(messages)
        chunks = [c async for c in gen]
        assert chunks == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_still_has_complete(self):
        llm = ConcreteStreamingLLM()
        messages = [Message(role=MessageRole.USER, content="Hi")]
        result = await llm.complete(messages)
        assert isinstance(result, CompletionResult)


class TestEmbeddingProtocol:
    """Test ISP: EmbeddingProtocol is standalone (not LLMProtocol)."""

    def test_is_not_llm_protocol(self):
        emb = ConcreteEmbedding()
        assert isinstance(emb, EmbeddingProtocol)
        assert not isinstance(emb, LLMProtocol)

    @pytest.mark.asyncio
    async def test_embed(self):
        emb = ConcreteEmbedding()
        result = await emb.embed("hello")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


class TestPromptProtocol:
    """Test PromptProtocol contract."""

    def test_build_returns_built_prompt(self):
        prompt = ConcretePrompt()
        built = prompt.build()
        assert isinstance(built, BuiltPrompt)
        assert built.prompt == "Hello"
        assert built.system_prompt == "You are helpful"

    def test_isinstance(self):
        assert isinstance(ConcretePrompt(), PromptProtocol)


class TestContextSerializerProtocol:
    """Test ContextSerializerProtocol contract."""

    def test_serialize_returns_string(self):
        ser = ConcreteSerializer()
        result = ser.serialize(["m1", "m2"], ["a1"])
        assert isinstance(result, str)
        assert "memories=2" in result
        assert "artifacts=1" in result

    def test_isinstance(self):
        assert isinstance(ConcreteSerializer(), ContextSerializerProtocol)


class TestMetaSkillProtocol:
    """Test MetaSkillProtocol extends SkillProtocol."""

    def test_is_skill_protocol(self):
        meta = ConcreteMetaSkill()
        assert isinstance(meta, SkillProtocol)
        assert isinstance(meta, MetaSkillProtocol)

    @pytest.mark.asyncio
    async def test_plan_and_interpret(self):
        class TestInput(BaseModel):
            pass

        meta = ConcreteMetaSkill()
        plan = await meta.plan(TestInput(), SkillContext())
        assert isinstance(plan, SkillPlan)
        result = await meta.interpret({})
        assert isinstance(result, SkillResult)

    @pytest.mark.asyncio
    async def test_generate_skill_raises_not_implemented(self):
        meta = ConcreteMetaSkill()
        with pytest.raises(NotImplementedError, match="V2"):
            await meta.generate_skill("summarize", SkillContext())

    @pytest.mark.asyncio
    async def test_generate_dag_raises_not_implemented(self):
        meta = ConcreteMetaSkill()
        with pytest.raises(NotImplementedError, match="V2"):
            await meta.generate_dag("pipeline", SkillContext())


# ──────────────────────────────────────────────
# Partial Implementation Detection
# ──────────────────────────────────────────────


class TestPartialImplementation:
    """Verify that partial implementations still raise TypeError."""

    def test_skill_missing_interpret(self):
        class Partial(SkillProtocol):
            @property
            def name(self) -> str:
                return "p"

            @property
            def description(self) -> str:
                return "p"

            async def plan(self, input, context):
                return SkillPlan(name="p", steps=[])

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_skill_missing_plan(self):
        class Partial(SkillProtocol):
            @property
            def name(self) -> str:
                return "p"

            @property
            def description(self) -> str:
                return "p"

            async def interpret(self, result):
                return SkillResult()

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_llm_missing_complete_structured(self):
        class Partial(LLMProtocol):
            async def complete(self, messages):
                return CompletionResult(text="")

            async def complete_with_tools(self, messages, tools):
                return ToolResult()

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_meta_skill_missing_generate_methods(self):
        """MetaSkill that implements Skill methods but not generate_* should fail."""
        class Partial(MetaSkillProtocol):
            @property
            def name(self) -> str:
                return "p"

            @property
            def description(self) -> str:
                return "p"

            async def plan(self, input, context):
                return SkillPlan(name="p", steps=[])

            async def interpret(self, result):
                return SkillResult()

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]
