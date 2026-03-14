"""
Built-in skill implementations for common LLM patterns.

Concrete skills:
- TextGenerationSkill: single text step with prompt + system prompt
- DataExtractionSkill: single data step with output schema
- CodeGenerationSkill: text step with code-oriented system prompt
- ReviewSkill: text step with pass/fail validation via validate_output()

Phase 3.8.1 — Built-in Skills.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .models import (
    DataStepConfig,
    SkillContext,
    SkillPlan,
    SkillProvenance,
    SkillResult,
    SkillStep,
    TextStepConfig,
)
from .skill import RetryableValidationError, Skill


def _extract_text(input: BaseModel) -> str:
    """Extract text content from an input model.

    Looks for a ``text`` attribute first, falls back to JSON dump.
    """
    if hasattr(input, "text"):
        return str(input.text)
    return input.model_dump_json()


# ──────────────────────────────────────────────
# TextGenerationSkill
# ──────────────────────────────────────────────


class TextGenerationSkill(Skill):
    """Single text-generation step.

    Takes an input with a ``text`` field and produces a text completion.
    Optionally accepts a system prompt and model parameters at construction.
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "text_generation"

    @property
    def description(self) -> str:
        return "Generate text from a prompt using an LLM."

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        text = _extract_text(input)
        prompt = text
        if context.serialized_context:
            prompt = f"{context.serialized_context}\n\n{text}"

        config = TextStepConfig(
            prompt=prompt,
            system_prompt=self._system_prompt,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return SkillPlan(
            name=self.name,
            steps=[SkillStep(id="generate", kind="text", config=config)],
        )

    async def interpret(self, result: Any) -> SkillResult:
        if hasattr(result, "step_results"):
            step = result.step_results.get("generate")
            if step and step.ok:
                return SkillResult(
                    output=step.value,
                    success=True,
                    provenance=SkillProvenance(skill_name=self.name),
                )
            error = step.error_message if step else "No result for step 'generate'"
            return SkillResult(
                output=None,
                success=False,
                error_message=error,
                provenance=SkillProvenance(skill_name=self.name),
            )
        return SkillResult(
            output=result,
            success=True,
            provenance=SkillProvenance(skill_name=self.name),
        )


# ──────────────────────────────────────────────
# DataExtractionSkill
# ──────────────────────────────────────────────


class DataExtractionSkill(Skill):
    """Single data-extraction step with output schema.

    Takes an input with a ``text`` field and extracts structured data
    according to the provided output schema.
    """

    def __init__(
        self,
        *,
        output_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> None:
        self._output_schema = output_schema
        self._system_prompt = system_prompt
        self._model = model

    @property
    def name(self) -> str:
        return "data_extraction"

    @property
    def description(self) -> str:
        return "Extract structured data from text using an LLM."

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        text = _extract_text(input)
        prompt = text
        if context.serialized_context:
            prompt = f"{context.serialized_context}\n\n{text}"

        config = DataStepConfig(
            prompt=prompt,
            output_schema=self._output_schema,
            model=self._model,
        )
        return SkillPlan(
            name=self.name,
            steps=[SkillStep(id="extract", kind="data", config=config)],
        )

    async def interpret(self, result: Any) -> SkillResult:
        if hasattr(result, "step_results"):
            step = result.step_results.get("extract")
            if step and step.ok:
                return SkillResult(
                    output=step.value,
                    success=True,
                    provenance=SkillProvenance(skill_name=self.name),
                )
            error = step.error_message if step else "No result for step 'extract'"
            return SkillResult(
                output=None,
                success=False,
                error_message=error,
                provenance=SkillProvenance(skill_name=self.name),
            )
        return SkillResult(
            output=result,
            success=True,
            provenance=SkillProvenance(skill_name=self.name),
        )


# ──────────────────────────────────────────────
# CodeGenerationSkill
# ──────────────────────────────────────────────


class CodeGenerationSkill(Skill):
    """Text step with code-oriented system prompt.

    Generates code from a description using a specialized system
    prompt. Optionally configured with a target language.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert programmer. Produce clean, well-documented, "
        "production-ready code. Return only the code without explanations "
        "unless specifically asked."
    )

    def __init__(
        self,
        *,
        language: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._language = language
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        if language and system_prompt is None:
            self._system_prompt = f"{self._system_prompt} Language: {language}."
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "code_generation"

    @property
    def description(self) -> str:
        lang = f" ({self._language})" if self._language else ""
        return f"Generate code{lang} from a description."

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        text = _extract_text(input)
        prompt = text
        if context.serialized_context:
            prompt = f"{context.serialized_context}\n\n{text}"

        config = TextStepConfig(
            prompt=prompt,
            system_prompt=self._system_prompt,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return SkillPlan(
            name=self.name,
            steps=[SkillStep(id="generate_code", kind="text", config=config)],
        )

    async def interpret(self, result: Any) -> SkillResult:
        if hasattr(result, "step_results"):
            step = result.step_results.get("generate_code")
            if step and step.ok:
                return SkillResult(
                    output=step.value,
                    success=True,
                    provenance=SkillProvenance(skill_name=self.name),
                )
            error = (
                step.error_message
                if step
                else "No result for step 'generate_code'"
            )
            return SkillResult(
                output=None,
                success=False,
                error_message=error,
                provenance=SkillProvenance(skill_name=self.name),
            )
        return SkillResult(
            output=result,
            success=True,
            provenance=SkillProvenance(skill_name=self.name),
        )


# ──────────────────────────────────────────────
# ReviewSkill
# ──────────────────────────────────────────────


class ReviewSkill(Skill):
    """Review step with pass/fail feedback and validation.

    Reviews content against criteria and produces a pass/fail
    assessment with detailed feedback. Uses ``validate_output()``
    to ensure the review output has the required structure
    (``passed``, ``feedback`` keys).
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a careful reviewer. Evaluate the provided content against "
        "the given criteria. Respond with a JSON object containing: "
        '"passed" (boolean), "feedback" (string with detailed assessment), '
        'and "issues" (list of specific issues found, empty if passed).'
    )

    def __init__(
        self,
        *,
        criteria: list[str] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> None:
        self._criteria = criteria or []
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._model = model

    @property
    def name(self) -> str:
        return "review"

    @property
    def description(self) -> str:
        return "Review content and produce pass/fail feedback."

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        text = _extract_text(input)
        prompt_parts: list[str] = []
        if context.serialized_context:
            prompt_parts.append(context.serialized_context)
        prompt_parts.append(text)
        if self._criteria:
            criteria_text = "\n".join(f"- {c}" for c in self._criteria)
            prompt_parts.append(f"Criteria:\n{criteria_text}")

        config = TextStepConfig(
            prompt="\n\n".join(prompt_parts),
            system_prompt=self._system_prompt,
            model=self._model,
        )
        return SkillPlan(
            name=self.name,
            steps=[SkillStep(id="review", kind="text", config=config)],
        )

    async def interpret(self, result: Any) -> SkillResult:
        if hasattr(result, "step_results"):
            step = result.step_results.get("review")
            if step and step.ok:
                output = step.value
                valid = await self.validate_output(output)
                if not valid:
                    raise RetryableValidationError(
                        "Review output failed validation"
                    )
                return SkillResult(
                    output=output,
                    success=True,
                    provenance=SkillProvenance(skill_name=self.name),
                )
            error = step.error_message if step else "No result for step 'review'"
            return SkillResult(
                output=None,
                success=False,
                error_message=error,
                provenance=SkillProvenance(skill_name=self.name),
            )
        return SkillResult(
            output=result,
            success=True,
            provenance=SkillProvenance(skill_name=self.name),
        )

    async def validate_output(self, output: Any) -> bool:
        """Validate that review output has required structure.

        Expects a dict with ``passed`` (bool) and ``feedback`` (str) keys.
        """
        if isinstance(output, dict):
            return "passed" in output and "feedback" in output
        return False
