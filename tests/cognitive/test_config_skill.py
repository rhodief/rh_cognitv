"""
Tests for ConfigSkill — auto-generated plan/interpret from SkillConfig.

Phase 3.8.2 — ConfigSkill.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from rh_cognitv.cognitive.adapters import OrchestratorResult
from rh_cognitv.cognitive.models import (
    DataStepConfig,
    SkillConfig,
    SkillContext,
    TextStepConfig,
)
from rh_cognitv.cognitive.skill import ConfigSkill, RetryableValidationError
from rh_cognitv.execution_platform.models import MemoryQuery
from rh_cognitv.orchestrator.models import NodeResult


# ── test models ──────────────────────────────


class SummarizeInput(BaseModel):
    text: str
    max_length: int = 100


class SummaryOutput(BaseModel):
    summary: str
    word_count: int


# ──────────────────────────────────────────────
# Properties
# ──────────────────────────────────────────────


class TestConfigSkillProperties:
    def test_name_from_config(self):
        config = SkillConfig(name="summarize", prompt_template="{text}")
        assert ConfigSkill(config).name == "summarize"

    def test_description_from_config(self):
        config = SkillConfig(
            name="x", description="my desc", prompt_template="{text}"
        )
        assert ConfigSkill(config).description == "my desc"

    def test_memory_query_from_config(self):
        query = MemoryQuery(text="relevant", top_k=3)
        config = SkillConfig(
            name="x", prompt_template="{text}", memory_query=query
        )
        assert ConfigSkill(config).memory_query == query

    def test_memory_query_none_by_default(self):
        config = SkillConfig(name="x", prompt_template="{text}")
        assert ConfigSkill(config).memory_query is None


# ──────────────────────────────────────────────
# plan()
# ──────────────────────────────────────────────


class TestConfigSkillPlan:
    @pytest.mark.asyncio
    async def test_text_step_when_no_schema(self):
        config = SkillConfig(name="gen", prompt_template="Generate: {text}")
        plan = await ConfigSkill(config).plan(
            SummarizeInput(text="Hello"), SkillContext()
        )
        assert len(plan.steps) == 1
        assert plan.steps[0].kind == "text"
        assert plan.steps[0].id == "main"

    @pytest.mark.asyncio
    async def test_prompt_rendered(self):
        config = SkillConfig(name="gen", prompt_template="Summarize: {text}")
        plan = await ConfigSkill(config).plan(
            SummarizeInput(text="Hello world"), SkillContext()
        )
        assert plan.steps[0].config.prompt == "Summarize: Hello world"

    @pytest.mark.asyncio
    async def test_prompt_with_context(self):
        config = SkillConfig(
            name="gen", prompt_template="{context}\n\n{text}"
        )
        ctx = SkillContext(serialized_context="Background")
        plan = await ConfigSkill(config).plan(
            SummarizeInput(text="Question"), ctx
        )
        prompt = plan.steps[0].config.prompt
        assert "Background" in prompt
        assert "Question" in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_passed(self):
        config = SkillConfig(
            name="gen",
            prompt_template="{text}",
            system_prompt="Be concise",
        )
        plan = await ConfigSkill(config).plan(
            SummarizeInput(text="Hi"), SkillContext()
        )
        cfg = plan.steps[0].config
        assert isinstance(cfg, TextStepConfig)
        assert cfg.system_prompt == "Be concise"

    @pytest.mark.asyncio
    async def test_model_params(self):
        config = SkillConfig(
            name="gen",
            prompt_template="{text}",
            model="gpt-4",
            temperature=0.7,
            max_tokens=200,
        )
        plan = await ConfigSkill(config).plan(
            SummarizeInput(text="Go"), SkillContext()
        )
        cfg = plan.steps[0].config
        assert isinstance(cfg, TextStepConfig)
        assert cfg.model == "gpt-4"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 200

    @pytest.mark.asyncio
    async def test_data_step_with_dict_schema(self):
        schema = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        config = SkillConfig(
            name="extract", prompt_template="{text}", output_schema=schema
        )
        plan = await ConfigSkill(config).plan(
            SummarizeInput(text="Extract"), SkillContext()
        )
        assert plan.steps[0].kind == "data"
        cfg = plan.steps[0].config
        assert isinstance(cfg, DataStepConfig)
        assert cfg.output_schema == schema

    @pytest.mark.asyncio
    async def test_data_step_with_pydantic_schema(self):
        config = SkillConfig(
            name="extract",
            prompt_template="{text}",
            output_schema=SummaryOutput,
        )
        plan = await ConfigSkill(config).plan(
            SummarizeInput(text="Extract"), SkillContext()
        )
        assert plan.steps[0].kind == "data"
        cfg = plan.steps[0].config
        assert isinstance(cfg, DataStepConfig)
        assert cfg.output_schema is not None
        assert "properties" in cfg.output_schema

    @pytest.mark.asyncio
    async def test_multiple_input_fields(self):
        config = SkillConfig(
            name="gen",
            prompt_template="Text: {text}, Max: {max_length}",
        )
        plan = await ConfigSkill(config).plan(
            SummarizeInput(text="Hello", max_length=50), SkillContext()
        )
        prompt = plan.steps[0].config.prompt
        assert "Hello" in prompt
        assert "50" in prompt

    @pytest.mark.asyncio
    async def test_missing_template_vars_preserved(self):
        config = SkillConfig(
            name="gen", prompt_template="Hello {missing_var}"
        )
        plan = await ConfigSkill(config).plan(
            SummarizeInput(text="Hi"), SkillContext()
        )
        assert "{missing_var}" in plan.steps[0].config.prompt

    @pytest.mark.asyncio
    async def test_plan_name(self):
        config = SkillConfig(name="my_skill", prompt_template="{text}")
        plan = await ConfigSkill(config).plan(
            SummarizeInput(text="Go"), SkillContext()
        )
        assert plan.name == "my_skill"


# ──────────────────────────────────────────────
# interpret()
# ──────────────────────────────────────────────


class TestConfigSkillInterpret:
    @pytest.mark.asyncio
    async def test_success_no_schema(self):
        config = SkillConfig(name="gen", prompt_template="{text}")
        result = OrchestratorResult(
            step_results={"main": NodeResult.success("output text")},
            success=True,
        )
        sr = await ConfigSkill(config).interpret(result)
        assert sr.success is True
        assert sr.output == "output text"

    @pytest.mark.asyncio
    async def test_failure(self):
        config = SkillConfig(name="gen", prompt_template="{text}")
        result = OrchestratorResult(
            step_results={"main": NodeResult.failure("LLM error")},
            success=False,
        )
        sr = await ConfigSkill(config).interpret(result)
        assert sr.success is False
        assert "LLM error" in sr.error_message

    @pytest.mark.asyncio
    async def test_missing_step(self):
        config = SkillConfig(name="gen", prompt_template="{text}")
        result = OrchestratorResult(step_results={}, success=False)
        sr = await ConfigSkill(config).interpret(result)
        assert sr.success is False
        assert "main" in sr.error_message

    @pytest.mark.asyncio
    async def test_validates_pydantic_schema_from_dict(self):
        config = SkillConfig(
            name="gen",
            prompt_template="{text}",
            output_schema=SummaryOutput,
        )
        result = OrchestratorResult(
            step_results={
                "main": NodeResult.success(
                    {"summary": "Short", "word_count": 1}
                )
            },
            success=True,
        )
        sr = await ConfigSkill(config).interpret(result)
        assert sr.success is True
        assert isinstance(sr.output, SummaryOutput)
        assert sr.output.summary == "Short"

    @pytest.mark.asyncio
    async def test_pydantic_validation_error(self):
        config = SkillConfig(
            name="gen",
            prompt_template="{text}",
            output_schema=SummaryOutput,
        )
        result = OrchestratorResult(
            step_results={"main": NodeResult.success({"bad": "data"})},
            success=True,
        )
        with pytest.raises(RetryableValidationError):
            await ConfigSkill(config).interpret(result)

    @pytest.mark.asyncio
    async def test_wrong_type_raises(self):
        config = SkillConfig(
            name="gen",
            prompt_template="{text}",
            output_schema=SummaryOutput,
        )
        result = OrchestratorResult(
            step_results={"main": NodeResult.success("not a dict")},
            success=True,
        )
        with pytest.raises(RetryableValidationError):
            await ConfigSkill(config).interpret(result)

    @pytest.mark.asyncio
    async def test_already_validated_instance(self):
        config = SkillConfig(
            name="gen",
            prompt_template="{text}",
            output_schema=SummaryOutput,
        )
        result = OrchestratorResult(
            step_results={
                "main": NodeResult.success(
                    SummaryOutput(summary="Yes", word_count=1)
                )
            },
            success=True,
        )
        sr = await ConfigSkill(config).interpret(result)
        assert sr.success is True
        assert isinstance(sr.output, SummaryOutput)

    @pytest.mark.asyncio
    async def test_provenance(self):
        config = SkillConfig(name="my_skill", prompt_template="{text}")
        result = OrchestratorResult(
            step_results={"main": NodeResult.success("ok")},
            success=True,
        )
        sr = await ConfigSkill(config).interpret(result)
        assert sr.provenance.skill_name == "my_skill"

    @pytest.mark.asyncio
    async def test_raw_result_fallback(self):
        config = SkillConfig(name="gen", prompt_template="{text}")
        sr = await ConfigSkill(config).interpret("raw value")
        assert sr.success is True
        assert sr.output == "raw value"

    @pytest.mark.asyncio
    async def test_dict_schema_no_interpret_validation(self):
        """Dict output_schema (not a BaseModel class) skips interpret-time validation."""
        config = SkillConfig(
            name="gen",
            prompt_template="{text}",
            output_schema={"type": "object"},
        )
        result = OrchestratorResult(
            step_results={"main": NodeResult.success({"any": "data"})},
            success=True,
        )
        sr = await ConfigSkill(config).interpret(result)
        assert sr.success is True
        assert sr.output == {"any": "data"}
