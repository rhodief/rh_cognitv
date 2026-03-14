"""
Tests for cognitive/skill.py — Phase 3.3 test gate.

Covers:
- ABC enforcement (can't instantiate Skill directly)
- Concrete test skill implementing plan/interpret
- plan() returns valid SkillPlan with SkillSteps and context_refs
- interpret() returns valid SkillResult with provenance
- validate_output() default (True) and custom override
- RetryableValidationError is TransientError, retryable
- Skill metadata: name, description, memory_query
- Statelessness: same input → same output
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio

from pydantic import BaseModel

from rh_cognitv.execution_platform.errors import TransientError
from rh_cognitv.execution_platform.models import MemoryQuery
from rh_cognitv.cognitive.models import (
    ContextRef,
    SkillConstraints,
    SkillContext,
    SkillPlan,
    SkillProvenance,
    SkillResult,
    SkillStep,
    TextStepConfig,
    DataStepConfig,
)
from rh_cognitv.cognitive.skill import RetryableValidationError, Skill


# ──────────────────────────────────────────────
# Test Input/Output Models
# ──────────────────────────────────────────────


class SummarizeInput(BaseModel):
    text: str
    max_length: int = 100


class SummarizeOutput(BaseModel):
    summary: str
    word_count: int = 0


# ──────────────────────────────────────────────
# Concrete Test Skills
# ──────────────────────────────────────────────


class SummarizeSkill(Skill):
    """A simple single-step text skill for testing."""

    @property
    def name(self) -> str:
        return "summarize"

    @property
    def description(self) -> str:
        return "Summarize text into a shorter version"

    @property
    def memory_query(self) -> MemoryQuery | None:
        return MemoryQuery(role="semantic", top_k=3)

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        assert isinstance(input, SummarizeInput)
        return SkillPlan(
            name="summarize",
            steps=[
                SkillStep(
                    id="gen",
                    kind="text",
                    config=TextStepConfig(
                        prompt=f"Summarize in {input.max_length} words: {input.text}",
                        system_prompt="You are a summarization assistant.",
                    ),
                    context_refs=[
                        ContextRef(kind="query", query=MemoryQuery(role="semantic", top_k=3)),
                    ],
                ),
            ],
            constraints=SkillConstraints(timeout_seconds=30.0, max_retries=2),
        )

    async def interpret(self, result: Any) -> SkillResult:
        text = result if isinstance(result, str) else str(result)
        output = SummarizeOutput(summary=text, word_count=len(text.split()))
        return SkillResult(
            output=output,
            success=True,
            provenance=SkillProvenance(skill_name=self.name),
        )


class MultiStepSkill(Skill):
    """A multi-step skill with data extraction for testing."""

    @property
    def name(self) -> str:
        return "research-and-extract"

    @property
    def description(self) -> str:
        return "Research a topic then extract structured data"

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        return SkillPlan(
            name="research-and-extract",
            steps=[
                SkillStep(
                    id="research",
                    kind="text",
                    config=TextStepConfig(prompt="Research the topic"),
                ),
                SkillStep(
                    id="extract",
                    kind="data",
                    config=DataStepConfig(prompt="Extract key facts"),
                    context_refs=[
                        ContextRef(kind="previous_result", from_step="research"),
                    ],
                    depends_on=["research"],
                ),
            ],
        )

    async def interpret(self, result: Any) -> SkillResult:
        return SkillResult(output=result, success=True)


class StrictSkill(Skill):
    """A skill with custom output validation."""

    @property
    def name(self) -> str:
        return "strict"

    @property
    def description(self) -> str:
        return "A skill that validates output strictly"

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        return SkillPlan(
            name="strict",
            steps=[
                SkillStep(
                    id="gen",
                    kind="text",
                    config=TextStepConfig(prompt="Generate output"),
                ),
            ],
        )

    async def interpret(self, result: Any) -> SkillResult:
        return SkillResult(output=result, success=True)

    async def validate_output(self, output: Any) -> bool:
        if not isinstance(output, str):
            raise RetryableValidationError("Output must be a string")
        if len(output) < 10:
            raise RetryableValidationError("Output too short")
        return True


# ──────────────────────────────────────────────
# Tests — ABC Enforcement
# ──────────────────────────────────────────────


class TestABCEnforcement:
    """Skill is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_skill_directly(self):
        with pytest.raises(TypeError):
            Skill()  # type: ignore[abstract]

    def test_missing_name_raises(self):
        class Incomplete(Skill):
            @property
            def description(self) -> str:
                return "x"

            async def plan(self, input, context):
                return SkillPlan(name="x", steps=[])

            async def interpret(self, result):
                return SkillResult()

        with pytest.raises(TypeError):
            Incomplete()

    def test_missing_description_raises(self):
        class Incomplete(Skill):
            @property
            def name(self) -> str:
                return "x"

            async def plan(self, input, context):
                return SkillPlan(name="x", steps=[])

            async def interpret(self, result):
                return SkillResult()

        with pytest.raises(TypeError):
            Incomplete()

    def test_missing_plan_raises(self):
        class Incomplete(Skill):
            @property
            def name(self) -> str:
                return "x"

            @property
            def description(self) -> str:
                return "x"

            async def interpret(self, result):
                return SkillResult()

        with pytest.raises(TypeError):
            Incomplete()

    def test_missing_interpret_raises(self):
        class Incomplete(Skill):
            @property
            def name(self) -> str:
                return "x"

            @property
            def description(self) -> str:
                return "x"

            async def plan(self, input, context):
                return SkillPlan(name="x", steps=[])

        with pytest.raises(TypeError):
            Incomplete()


# ──────────────────────────────────────────────
# Tests — Protocol Compliance
# ──────────────────────────────────────────────


class TestProtocolCompliance:
    """Skill satisfies SkillProtocol."""

    def test_skill_is_skill_protocol(self):
        from rh_cognitv.cognitive.protocols import SkillProtocol

        skill = SummarizeSkill()
        assert isinstance(skill, SkillProtocol)

    def test_concrete_skill_instantiable(self):
        skill = SummarizeSkill()
        assert skill is not None

    def test_has_name_property(self):
        skill = SummarizeSkill()
        assert skill.name == "summarize"

    def test_has_description_property(self):
        skill = SummarizeSkill()
        assert skill.description == "Summarize text into a shorter version"


# ──────────────────────────────────────────────
# Tests — Metadata
# ──────────────────────────────────────────────


class TestMetadata:
    """Skill metadata: name, description, memory_query."""

    def test_name(self):
        skill = SummarizeSkill()
        assert skill.name == "summarize"

    def test_description(self):
        skill = SummarizeSkill()
        assert isinstance(skill.description, str)
        assert len(skill.description) > 0

    def test_memory_query_defined(self):
        skill = SummarizeSkill()
        mq = skill.memory_query
        assert mq is not None
        assert mq.role == "semantic"
        assert mq.top_k == 3

    def test_memory_query_default_is_none(self):
        skill = MultiStepSkill()
        assert skill.memory_query is None


# ──────────────────────────────────────────────
# Tests — plan()
# ──────────────────────────────────────────────


class TestPlan:
    """plan() returns valid SkillPlan with SkillSteps and context_refs."""

    @pytest.fixture
    def skill(self):
        return SummarizeSkill()

    @pytest.fixture
    def context(self):
        return SkillContext()

    @pytest.fixture
    def input_data(self):
        return SummarizeInput(text="Hello world this is a test", max_length=50)

    @pytest.mark.asyncio
    async def test_returns_skill_plan(self, skill, context, input_data):
        plan = await skill.plan(input_data, context)
        assert isinstance(plan, SkillPlan)

    @pytest.mark.asyncio
    async def test_plan_has_name(self, skill, context, input_data):
        plan = await skill.plan(input_data, context)
        assert plan.name == "summarize"

    @pytest.mark.asyncio
    async def test_plan_has_steps(self, skill, context, input_data):
        plan = await skill.plan(input_data, context)
        assert len(plan.steps) == 1

    @pytest.mark.asyncio
    async def test_step_has_correct_kind(self, skill, context, input_data):
        plan = await skill.plan(input_data, context)
        assert plan.steps[0].kind == "text"

    @pytest.mark.asyncio
    async def test_step_has_id(self, skill, context, input_data):
        plan = await skill.plan(input_data, context)
        assert plan.steps[0].id == "gen"

    @pytest.mark.asyncio
    async def test_step_config_is_text_step_config(self, skill, context, input_data):
        plan = await skill.plan(input_data, context)
        assert isinstance(plan.steps[0].config, TextStepConfig)

    @pytest.mark.asyncio
    async def test_step_config_contains_input(self, skill, context, input_data):
        plan = await skill.plan(input_data, context)
        config = plan.steps[0].config
        assert isinstance(config, TextStepConfig)
        assert "50" in config.prompt
        assert input_data.text in config.prompt

    @pytest.mark.asyncio
    async def test_step_has_context_refs(self, skill, context, input_data):
        plan = await skill.plan(input_data, context)
        refs = plan.steps[0].context_refs
        assert len(refs) == 1
        assert refs[0].kind == "query"
        assert refs[0].query is not None

    @pytest.mark.asyncio
    async def test_plan_has_constraints(self, skill, context, input_data):
        plan = await skill.plan(input_data, context)
        assert plan.constraints is not None
        assert plan.constraints.timeout_seconds == 30.0
        assert plan.constraints.max_retries == 2

    @pytest.mark.asyncio
    async def test_multi_step_plan(self):
        skill = MultiStepSkill()
        context = SkillContext()
        input_data = SummarizeInput(text="topic")
        plan = await skill.plan(input_data, context)
        assert len(plan.steps) == 2
        assert plan.steps[0].id == "research"
        assert plan.steps[1].id == "extract"

    @pytest.mark.asyncio
    async def test_multi_step_dependencies(self):
        skill = MultiStepSkill()
        context = SkillContext()
        input_data = SummarizeInput(text="topic")
        plan = await skill.plan(input_data, context)
        assert plan.steps[1].depends_on == ["research"]

    @pytest.mark.asyncio
    async def test_multi_step_context_refs(self):
        skill = MultiStepSkill()
        context = SkillContext()
        input_data = SummarizeInput(text="topic")
        plan = await skill.plan(input_data, context)
        refs = plan.steps[1].context_refs
        assert len(refs) == 1
        assert refs[0].kind == "previous_result"
        assert refs[0].from_step == "research"


# ──────────────────────────────────────────────
# Tests — interpret()
# ──────────────────────────────────────────────


class TestInterpret:
    """interpret() returns valid SkillResult with provenance."""

    @pytest.mark.asyncio
    async def test_returns_skill_result(self):
        skill = SummarizeSkill()
        result = await skill.interpret("This is a summary of the text.")
        assert isinstance(result, SkillResult)

    @pytest.mark.asyncio
    async def test_result_has_output(self):
        skill = SummarizeSkill()
        result = await skill.interpret("This is a summary of the text.")
        assert result.output is not None
        assert isinstance(result.output, SummarizeOutput)
        assert result.output.summary == "This is a summary of the text."

    @pytest.mark.asyncio
    async def test_result_word_count(self):
        skill = SummarizeSkill()
        result = await skill.interpret("one two three")
        assert result.output.word_count == 3

    @pytest.mark.asyncio
    async def test_result_success(self):
        skill = SummarizeSkill()
        result = await skill.interpret("summary")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_result_has_provenance(self):
        skill = SummarizeSkill()
        result = await skill.interpret("summary")
        assert result.provenance is not None
        assert result.provenance.skill_name == "summarize"

    @pytest.mark.asyncio
    async def test_result_provenance_is_skill_provenance(self):
        skill = SummarizeSkill()
        result = await skill.interpret("summary")
        assert isinstance(result.provenance, SkillProvenance)


# ──────────────────────────────────────────────
# Tests — validate_output()
# ──────────────────────────────────────────────


class TestValidateOutput:
    """validate_output() default and custom override."""

    @pytest.mark.asyncio
    async def test_default_returns_true(self):
        skill = SummarizeSkill()
        assert await skill.validate_output("anything") is True

    @pytest.mark.asyncio
    async def test_default_accepts_none(self):
        skill = SummarizeSkill()
        assert await skill.validate_output(None) is True

    @pytest.mark.asyncio
    async def test_custom_accepts_valid_output(self):
        skill = StrictSkill()
        assert await skill.validate_output("This is a valid output string") is True

    @pytest.mark.asyncio
    async def test_custom_rejects_non_string(self):
        skill = StrictSkill()
        with pytest.raises(RetryableValidationError, match="Output must be a string"):
            await skill.validate_output(123)

    @pytest.mark.asyncio
    async def test_custom_rejects_short_output(self):
        skill = StrictSkill()
        with pytest.raises(RetryableValidationError, match="Output too short"):
            await skill.validate_output("short")

    @pytest.mark.asyncio
    async def test_custom_rejects_empty_string(self):
        skill = StrictSkill()
        with pytest.raises(RetryableValidationError, match="Output too short"):
            await skill.validate_output("")


# ──────────────────────────────────────────────
# Tests — RetryableValidationError
# ──────────────────────────────────────────────


class TestRetryableValidationError:
    """RetryableValidationError is a TransientError and retryable."""

    def test_is_transient_error(self):
        err = RetryableValidationError("bad output")
        assert isinstance(err, TransientError)

    def test_is_retryable(self):
        err = RetryableValidationError("bad output")
        assert err.retryable is True

    def test_category_is_transient(self):
        from rh_cognitv.execution_platform.errors import ErrorCategory

        err = RetryableValidationError("bad output")
        assert err.category == ErrorCategory.TRANSIENT

    def test_message(self):
        err = RetryableValidationError("bad output")
        assert str(err) == "bad output"

    def test_default_message(self):
        err = RetryableValidationError()
        assert str(err) == "Output validation failed"

    def test_attempt_tracking(self):
        err = RetryableValidationError("fail", attempt=3)
        assert err.attempt == 3

    def test_original_exception(self):
        cause = ValueError("root cause")
        err = RetryableValidationError("fail", original=cause)
        assert err.original is cause

    def test_is_exception(self):
        err = RetryableValidationError("fail")
        assert isinstance(err, Exception)

    def test_raises_and_catches_as_transient(self):
        with pytest.raises(TransientError):
            raise RetryableValidationError("bad")


# ──────────────────────────────────────────────
# Tests — Statelessness
# ──────────────────────────────────────────────


class TestStatelessness:
    """Skills produce consistent output with same input (DI-L1-01)."""

    @pytest.mark.asyncio
    async def test_plan_is_deterministic(self):
        skill = SummarizeSkill()
        context = SkillContext()
        input_data = SummarizeInput(text="Hello world", max_length=50)

        plan1 = await skill.plan(input_data, context)
        plan2 = await skill.plan(input_data, context)

        assert plan1.name == plan2.name
        assert len(plan1.steps) == len(plan2.steps)
        assert plan1.steps[0].id == plan2.steps[0].id
        assert plan1.steps[0].kind == plan2.steps[0].kind

    @pytest.mark.asyncio
    async def test_interpret_is_deterministic(self):
        skill = SummarizeSkill()

        r1 = await skill.interpret("the summary")
        r2 = await skill.interpret("the summary")

        assert r1.output.summary == r2.output.summary
        assert r1.output.word_count == r2.output.word_count
        assert r1.success == r2.success

    @pytest.mark.asyncio
    async def test_no_state_leaks_between_calls(self):
        skill = SummarizeSkill()
        context = SkillContext()

        input_a = SummarizeInput(text="First input", max_length=10)
        input_b = SummarizeInput(text="Second input", max_length=20)

        plan_a = await skill.plan(input_a, context)
        plan_b = await skill.plan(input_b, context)

        # Plans should reflect their respective inputs, not leak state
        config_a = plan_a.steps[0].config
        config_b = plan_b.steps[0].config
        assert isinstance(config_a, TextStepConfig)
        assert isinstance(config_b, TextStepConfig)
        assert "First input" in config_a.prompt
        assert "Second input" in config_b.prompt
        assert "10" in config_a.prompt
        assert "20" in config_b.prompt
