"""
Tests for MetaSkill stub.

Phase 3.9 — V2 interface defined now, NotImplementedError for V1.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from rh_cognitv.cognitive.meta_skill import MetaSkill
from rh_cognitv.cognitive.models import (
    SkillConfig,
    SkillContext,
    SkillPlan,
    SkillProvenance,
    SkillResult,
    SkillStep,
    TextStepConfig,
)
from rh_cognitv.cognitive.protocols import MetaSkillProtocol, SkillProtocol


# ── concrete test subclass ───────────────────


class DummyInput(BaseModel):
    text: str = "hello"


class ConcreteMetaSkill(MetaSkill):
    """Minimal concrete subclass for testing."""

    @property
    def name(self) -> str:
        return "test_meta"

    @property
    def description(self) -> str:
        return "A test meta-skill."

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        return SkillPlan(
            name=self.name,
            steps=[
                SkillStep(
                    id="s1",
                    kind="text",
                    config=TextStepConfig(prompt="meta prompt"),
                )
            ],
        )

    async def interpret(self, result: Any) -> SkillResult:
        return SkillResult(
            output=result,
            success=True,
            provenance=SkillProvenance(skill_name=self.name),
        )


# ──────────────────────────────────────────────
# Importability & instantiation
# ──────────────────────────────────────────────


class TestMetaSkillImport:
    def test_importable(self):
        from rh_cognitv.cognitive.meta_skill import MetaSkill  # noqa: F811

        assert MetaSkill is not None

    def test_importable_from_package(self):
        from rh_cognitv.cognitive import MetaSkill  # noqa: F811

        assert MetaSkill is not None

    def test_concrete_instantiation(self):
        skill = ConcreteMetaSkill()
        assert skill.name == "test_meta"


# ──────────────────────────────────────────────
# Protocol compliance
# ──────────────────────────────────────────────


class TestMetaSkillProtocolCompliance:
    def test_is_skill_protocol(self):
        assert issubclass(MetaSkill, SkillProtocol)

    def test_is_meta_skill_protocol(self):
        assert issubclass(MetaSkill, MetaSkillProtocol)

    def test_concrete_is_instance_of_skill_protocol(self):
        skill = ConcreteMetaSkill()
        assert isinstance(skill, SkillProtocol)

    def test_concrete_is_instance_of_meta_skill_protocol(self):
        skill = ConcreteMetaSkill()
        assert isinstance(skill, MetaSkillProtocol)

    def test_has_generate_skill(self):
        assert hasattr(MetaSkill, "generate_skill")

    def test_has_generate_dag(self):
        assert hasattr(MetaSkill, "generate_dag")

    def test_has_plan(self):
        assert hasattr(MetaSkill, "plan")

    def test_has_interpret(self):
        assert hasattr(MetaSkill, "interpret")

    def test_has_validate_output(self):
        assert hasattr(MetaSkill, "validate_output")


# ──────────────────────────────────────────────
# V1 NotImplementedError
# ──────────────────────────────────────────────


class TestMetaSkillV1Stubs:
    @pytest.mark.asyncio
    async def test_generate_skill_raises(self):
        skill = ConcreteMetaSkill()
        with pytest.raises(NotImplementedError, match="V2"):
            await skill.generate_skill("make a summarizer", SkillContext())

    @pytest.mark.asyncio
    async def test_generate_dag_raises(self):
        skill = ConcreteMetaSkill()
        with pytest.raises(NotImplementedError, match="V2"):
            await skill.generate_dag("build a pipeline", SkillContext())


# ──────────────────────────────────────────────
# Standard Skill interface still works
# ──────────────────────────────────────────────


class TestMetaSkillAsSkill:
    @pytest.mark.asyncio
    async def test_plan_returns_skill_plan(self):
        skill = ConcreteMetaSkill()
        plan = await skill.plan(DummyInput(), SkillContext())
        assert isinstance(plan, SkillPlan)
        assert plan.name == "test_meta"
        assert len(plan.steps) == 1

    @pytest.mark.asyncio
    async def test_interpret_returns_skill_result(self):
        skill = ConcreteMetaSkill()
        result = await skill.interpret("some output")
        assert isinstance(result, SkillResult)
        assert result.success is True
        assert result.output == "some output"

    @pytest.mark.asyncio
    async def test_validate_output_default_true(self):
        skill = ConcreteMetaSkill()
        assert await skill.validate_output("anything") is True

    def test_name_property(self):
        assert ConcreteMetaSkill().name == "test_meta"

    def test_description_property(self):
        assert ConcreteMetaSkill().description == "A test meta-skill."

    def test_memory_query_default_none(self):
        assert ConcreteMetaSkill().memory_query is None
