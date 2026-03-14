"""
Tests for built-in skill implementations.

Phase 3.8.1 — TextGenerationSkill, DataExtractionSkill,
CodeGenerationSkill, ReviewSkill.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from rh_cognitv.cognitive.adapters import OrchestratorResult
from rh_cognitv.cognitive.builtin_skills import (
    CodeGenerationSkill,
    DataExtractionSkill,
    ReviewSkill,
    TextGenerationSkill,
)
from rh_cognitv.cognitive.models import (
    DataStepConfig,
    SkillContext,
    SkillPlan,
    TextStepConfig,
)
from rh_cognitv.cognitive.skill import RetryableValidationError
from rh_cognitv.orchestrator.models import NodeResult


# ── helpers ──────────────────────────────────


class SimpleInput(BaseModel):
    text: str


class NoTextInput(BaseModel):
    value: int = 42


# ──────────────────────────────────────────────
# TextGenerationSkill
# ──────────────────────────────────────────────


class TestTextGenerationSkillProperties:
    def test_name(self):
        assert TextGenerationSkill().name == "text_generation"

    def test_description(self):
        desc = TextGenerationSkill().description.lower()
        assert "text" in desc or "generate" in desc

    def test_memory_query_default_none(self):
        assert TextGenerationSkill().memory_query is None


class TestTextGenerationSkillPlan:
    @pytest.mark.asyncio
    async def test_single_text_step(self):
        plan = await TextGenerationSkill().plan(SimpleInput(text="Hi"), SkillContext())
        assert isinstance(plan, SkillPlan)
        assert len(plan.steps) == 1
        assert plan.steps[0].kind == "text"
        assert plan.steps[0].id == "generate"

    @pytest.mark.asyncio
    async def test_prompt_contains_input(self):
        plan = await TextGenerationSkill().plan(
            SimpleInput(text="Summarize this"), SkillContext()
        )
        assert "Summarize this" in plan.steps[0].config.prompt

    @pytest.mark.asyncio
    async def test_system_prompt(self):
        skill = TextGenerationSkill(system_prompt="Be helpful")
        plan = await skill.plan(SimpleInput(text="Hi"), SkillContext())
        config = plan.steps[0].config
        assert isinstance(config, TextStepConfig)
        assert config.system_prompt == "Be helpful"

    @pytest.mark.asyncio
    async def test_context_injected_into_prompt(self):
        ctx = SkillContext(serialized_context="Background info")
        plan = await TextGenerationSkill().plan(SimpleInput(text="Question"), ctx)
        prompt = plan.steps[0].config.prompt
        assert "Background info" in prompt
        assert "Question" in prompt

    @pytest.mark.asyncio
    async def test_model_params_propagated(self):
        skill = TextGenerationSkill(model="gpt-4", temperature=0.5, max_tokens=100)
        plan = await skill.plan(SimpleInput(text="Go"), SkillContext())
        config = plan.steps[0].config
        assert isinstance(config, TextStepConfig)
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 100

    @pytest.mark.asyncio
    async def test_no_context_no_prefix(self):
        plan = await TextGenerationSkill().plan(SimpleInput(text="Just text"), SkillContext())
        assert plan.steps[0].config.prompt == "Just text"

    @pytest.mark.asyncio
    async def test_input_without_text_field(self):
        plan = await TextGenerationSkill().plan(NoTextInput(), SkillContext())
        # Falls back to JSON dump
        assert "42" in plan.steps[0].config.prompt


class TestTextGenerationSkillInterpret:
    @pytest.mark.asyncio
    async def test_success(self):
        result = OrchestratorResult(
            step_results={"generate": NodeResult.success("Generated text")},
            success=True,
        )
        sr = await TextGenerationSkill().interpret(result)
        assert sr.success is True
        assert sr.output == "Generated text"
        assert sr.provenance.skill_name == "text_generation"

    @pytest.mark.asyncio
    async def test_failure(self):
        result = OrchestratorResult(
            step_results={"generate": NodeResult.failure("LLM error")},
            success=False,
        )
        sr = await TextGenerationSkill().interpret(result)
        assert sr.success is False
        assert sr.error_message == "LLM error"

    @pytest.mark.asyncio
    async def test_missing_step_result(self):
        result = OrchestratorResult(step_results={}, success=False)
        sr = await TextGenerationSkill().interpret(result)
        assert sr.success is False
        assert "generate" in sr.error_message

    @pytest.mark.asyncio
    async def test_raw_value_fallback(self):
        sr = await TextGenerationSkill().interpret("raw text")
        assert sr.success is True
        assert sr.output == "raw text"


# ──────────────────────────────────────────────
# DataExtractionSkill
# ──────────────────────────────────────────────


class TestDataExtractionSkillProperties:
    def test_name(self):
        assert DataExtractionSkill().name == "data_extraction"

    def test_description(self):
        desc = DataExtractionSkill().description.lower()
        assert "data" in desc or "extract" in desc


class TestDataExtractionSkillPlan:
    @pytest.mark.asyncio
    async def test_single_data_step(self):
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        skill = DataExtractionSkill(output_schema=schema)
        plan = await skill.plan(SimpleInput(text="Extract data"), SkillContext())
        assert len(plan.steps) == 1
        assert plan.steps[0].kind == "data"
        assert plan.steps[0].id == "extract"

    @pytest.mark.asyncio
    async def test_output_schema_propagated(self):
        schema = {"type": "object"}
        skill = DataExtractionSkill(output_schema=schema)
        plan = await skill.plan(SimpleInput(text="Go"), SkillContext())
        config = plan.steps[0].config
        assert isinstance(config, DataStepConfig)
        assert config.output_schema == schema

    @pytest.mark.asyncio
    async def test_context_injected(self):
        skill = DataExtractionSkill()
        ctx = SkillContext(serialized_context="Some context")
        plan = await skill.plan(SimpleInput(text="Extract"), ctx)
        assert "Some context" in plan.steps[0].config.prompt
        assert "Extract" in plan.steps[0].config.prompt

    @pytest.mark.asyncio
    async def test_model_param(self):
        skill = DataExtractionSkill(model="gpt-4o")
        plan = await skill.plan(SimpleInput(text="Go"), SkillContext())
        assert plan.steps[0].config.model == "gpt-4o"


class TestDataExtractionSkillInterpret:
    @pytest.mark.asyncio
    async def test_success(self):
        result = OrchestratorResult(
            step_results={"extract": NodeResult.success({"key": "value"})},
            success=True,
        )
        sr = await DataExtractionSkill().interpret(result)
        assert sr.success is True
        assert sr.output == {"key": "value"}

    @pytest.mark.asyncio
    async def test_failure(self):
        result = OrchestratorResult(
            step_results={"extract": NodeResult.failure("parse error")},
            success=False,
        )
        sr = await DataExtractionSkill().interpret(result)
        assert sr.success is False
        assert "parse error" in sr.error_message


# ──────────────────────────────────────────────
# CodeGenerationSkill
# ──────────────────────────────────────────────


class TestCodeGenerationSkillProperties:
    def test_name(self):
        assert CodeGenerationSkill().name == "code_generation"

    def test_description_generic(self):
        assert "code" in CodeGenerationSkill().description.lower()

    def test_description_with_language(self):
        skill = CodeGenerationSkill(language="python")
        assert "python" in skill.description.lower()


class TestCodeGenerationSkillPlan:
    @pytest.mark.asyncio
    async def test_code_system_prompt_default(self):
        plan = await CodeGenerationSkill().plan(
            SimpleInput(text="Write a function"), SkillContext()
        )
        config = plan.steps[0].config
        assert isinstance(config, TextStepConfig)
        assert "programmer" in config.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_language_in_system_prompt(self):
        skill = CodeGenerationSkill(language="python")
        plan = await skill.plan(SimpleInput(text="Write code"), SkillContext())
        config = plan.steps[0].config
        assert isinstance(config, TextStepConfig)
        assert "python" in config.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self):
        skill = CodeGenerationSkill(system_prompt="Custom code prompt")
        plan = await skill.plan(SimpleInput(text="Go"), SkillContext())
        assert plan.steps[0].config.system_prompt == "Custom code prompt"

    @pytest.mark.asyncio
    async def test_step_id(self):
        plan = await CodeGenerationSkill().plan(SimpleInput(text="Go"), SkillContext())
        assert plan.steps[0].id == "generate_code"

    @pytest.mark.asyncio
    async def test_context_injected(self):
        ctx = SkillContext(serialized_context="API docs")
        plan = await CodeGenerationSkill().plan(SimpleInput(text="Write"), ctx)
        assert "API docs" in plan.steps[0].config.prompt


class TestCodeGenerationSkillInterpret:
    @pytest.mark.asyncio
    async def test_success(self):
        result = OrchestratorResult(
            step_results={"generate_code": NodeResult.success("def foo(): pass")},
            success=True,
        )
        sr = await CodeGenerationSkill().interpret(result)
        assert sr.success is True
        assert sr.output == "def foo(): pass"
        assert sr.provenance.skill_name == "code_generation"

    @pytest.mark.asyncio
    async def test_failure(self):
        result = OrchestratorResult(
            step_results={"generate_code": NodeResult.failure("timeout")},
            success=False,
        )
        sr = await CodeGenerationSkill().interpret(result)
        assert sr.success is False


# ──────────────────────────────────────────────
# ReviewSkill
# ──────────────────────────────────────────────


class TestReviewSkillProperties:
    def test_name(self):
        assert ReviewSkill().name == "review"

    def test_description(self):
        desc = ReviewSkill().description.lower()
        assert "review" in desc or "feedback" in desc


class TestReviewSkillPlan:
    @pytest.mark.asyncio
    async def test_single_text_step(self):
        plan = await ReviewSkill().plan(SimpleInput(text="Review this"), SkillContext())
        assert len(plan.steps) == 1
        assert plan.steps[0].kind == "text"
        assert plan.steps[0].id == "review"

    @pytest.mark.asyncio
    async def test_review_system_prompt(self):
        plan = await ReviewSkill().plan(SimpleInput(text="Review"), SkillContext())
        config = plan.steps[0].config
        assert isinstance(config, TextStepConfig)
        assert config.system_prompt is not None
        assert "review" in config.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_criteria_in_prompt(self):
        skill = ReviewSkill(criteria=["correct", "efficient"])
        plan = await skill.plan(SimpleInput(text="Review this"), SkillContext())
        prompt = plan.steps[0].config.prompt
        assert "correct" in prompt
        assert "efficient" in prompt

    @pytest.mark.asyncio
    async def test_context_injected(self):
        ctx = SkillContext(serialized_context="Reference material")
        plan = await ReviewSkill().plan(SimpleInput(text="Review this"), ctx)
        prompt = plan.steps[0].config.prompt
        assert "Reference material" in prompt


class TestReviewSkillValidateOutput:
    @pytest.mark.asyncio
    async def test_valid_output(self):
        valid = await ReviewSkill().validate_output(
            {"passed": True, "feedback": "Good", "issues": []}
        )
        assert valid is True

    @pytest.mark.asyncio
    async def test_missing_passed(self):
        valid = await ReviewSkill().validate_output({"feedback": "Missing passed"})
        assert valid is False

    @pytest.mark.asyncio
    async def test_missing_feedback(self):
        valid = await ReviewSkill().validate_output({"passed": True})
        assert valid is False

    @pytest.mark.asyncio
    async def test_non_dict(self):
        valid = await ReviewSkill().validate_output("just a string")
        assert valid is False

    @pytest.mark.asyncio
    async def test_empty_dict(self):
        valid = await ReviewSkill().validate_output({})
        assert valid is False

    @pytest.mark.asyncio
    async def test_failed_review_is_valid_structure(self):
        valid = await ReviewSkill().validate_output(
            {"passed": False, "feedback": "Bad code", "issues": ["bug"]}
        )
        assert valid is True


class TestReviewSkillInterpret:
    @pytest.mark.asyncio
    async def test_valid_review(self):
        result = OrchestratorResult(
            step_results={
                "review": NodeResult.success(
                    {"passed": True, "feedback": "Looks good", "issues": []}
                )
            },
            success=True,
        )
        sr = await ReviewSkill().interpret(result)
        assert sr.success is True
        assert sr.output["passed"] is True

    @pytest.mark.asyncio
    async def test_invalid_output_raises(self):
        result = OrchestratorResult(
            step_results={"review": NodeResult.success("invalid output")},
            success=True,
        )
        with pytest.raises(RetryableValidationError):
            await ReviewSkill().interpret(result)

    @pytest.mark.asyncio
    async def test_failed_step(self):
        result = OrchestratorResult(
            step_results={"review": NodeResult.failure("LLM error")},
            success=False,
        )
        sr = await ReviewSkill().interpret(result)
        assert sr.success is False

    @pytest.mark.asyncio
    async def test_provenance(self):
        result = OrchestratorResult(
            step_results={
                "review": NodeResult.success(
                    {"passed": True, "feedback": "OK", "issues": []}
                )
            },
            success=True,
        )
        sr = await ReviewSkill().interpret(result)
        assert sr.provenance.skill_name == "review"
