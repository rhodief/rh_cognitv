"""
Skill base class — atomic cognitive unit.

Implements the plan/interpret split (DD-L1-01 Option A) with an optional
validate_output hook for output quality checks. Skills are stateless
(DI-L1-01): all state flows through input/output, configuration is
immutable at construction time.

Phase 3.3.1 — Skill Base Class.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from rh_cognitv.execution_platform.errors import TransientError
from rh_cognitv.execution_platform.models import MemoryQuery

from .models import (
    DataStepConfig,
    SkillConfig,
    SkillContext,
    SkillPlan,
    SkillProvenance,
    SkillResult,
    SkillStep,
    TextStepConfig,
)
from .prompt import TemplateRenderer
from .protocols import SkillProtocol


class RetryableValidationError(TransientError):
    """Output validation failed but the operation can be retried.

    Raised from validate_output() when the LLM output is structurally
    valid but doesn't meet quality criteria. L3's RetryPolicy picks
    this up for exponential backoff retries.
    """

    def __init__(
        self,
        message: str = "Output validation failed",
        *,
        attempt: int = 0,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message, attempt=attempt, original=original)


class Skill(SkillProtocol):
    """Abstract base class for all skills.

    Subclass this to create a concrete skill. You must implement:
      - name (property): unique skill identifier
      - description (property): human-readable description
      - plan(input, context) -> SkillPlan: produce the execution plan
      - interpret(result) -> SkillResult: interpret orchestrator results

    Optional overrides:
      - validate_output(output) -> bool: quality check on parsed output
      - memory_query: MemoryQuery for SkillContext building

    Skills are stateless. Configuration is set at construction via
    __init__ and must not change after that.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique skill identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this skill does."""
        ...

    @property
    def memory_query(self) -> MemoryQuery | None:
        """Optional memory query for SkillContext building.

        The framework uses this to pre-load relevant memories
        into SkillContext before calling plan(). Override to
        declare what context this skill needs.
        """
        return None

    @abstractmethod
    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        """Produce an execution plan from input and pre-loaded context."""
        ...

    @abstractmethod
    async def interpret(self, result: Any) -> SkillResult:
        """Interpret orchestrator results into a typed SkillResult."""
        ...

    async def validate_output(self, output: Any) -> bool:
        """Optional output quality check.

        Override to add custom validation logic. When validation fails,
        raise RetryableValidationError to trigger L3's RetryPolicy.

        The default implementation accepts all output.
        """
        return True


class ConfigSkill(Skill):
    """Auto-generates plan() and interpret() from SkillConfig.

    DD-L1-01 Option C — zero-code declarative skill.
    ``plan()`` builds the prompt via ``TemplateRenderer``, creates a
    single ``SkillStep``.  ``interpret()`` validates output against
    ``output_schema`` via Pydantic when the schema is a BaseModel class.

    Phase 3.8.2 — ConfigSkill.
    """

    def __init__(self, config: SkillConfig) -> None:
        self._config = config
        self._renderer = TemplateRenderer()

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def description(self) -> str:
        return self._config.description

    @property
    def memory_query(self) -> MemoryQuery | None:
        return self._config.memory_query

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        # Build template variables from input fields + serialized context
        variables: dict[str, str] = {}
        if hasattr(input, "model_dump"):
            for k, v in input.model_dump().items():
                variables[k] = str(v) if v is not None else ""
        variables["context"] = context.serialized_context

        built = self._renderer.render(
            template=self._config.prompt_template,
            variables=variables,
            system_prompt=self._config.system_prompt,
        )

        # Data step when output_schema is present, text step otherwise
        if self._config.output_schema is not None:
            schema_dict = self._config.output_schema
            if (
                isinstance(schema_dict, type)
                and issubclass(schema_dict, BaseModel)
            ):
                schema_dict = schema_dict.model_json_schema()
            step = SkillStep(
                id="main",
                kind="data",
                config=DataStepConfig(
                    prompt=built.prompt,
                    output_schema=schema_dict,
                    model=self._config.model,
                ),
            )
        else:
            step = SkillStep(
                id="main",
                kind="text",
                config=TextStepConfig(
                    prompt=built.prompt,
                    system_prompt=built.system_prompt,
                    model=self._config.model,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                ),
            )

        return SkillPlan(name=self.name, steps=[step])

    async def interpret(self, result: Any) -> SkillResult:
        value = result
        if hasattr(result, "step_results"):
            step = result.step_results.get("main")
            if step is None:
                return SkillResult(
                    output=None,
                    success=False,
                    error_message="No result for step 'main'",
                    provenance=SkillProvenance(skill_name=self.name),
                )
            if not step.ok:
                return SkillResult(
                    output=None,
                    success=False,
                    error_message=step.error_message,
                    provenance=SkillProvenance(skill_name=self.name),
                )
            value = step.value

        # Validate against output_schema (Pydantic BaseModel) if present
        output_schema = self._config.output_schema
        if (
            output_schema is not None
            and isinstance(output_schema, type)
            and issubclass(output_schema, BaseModel)
        ):
            if isinstance(value, dict):
                try:
                    value = output_schema.model_validate(value)
                except Exception as e:
                    raise RetryableValidationError(
                        f"Output validation failed: {e}"
                    ) from e
            elif not isinstance(value, output_schema):
                raise RetryableValidationError(
                    f"Expected {output_schema.__name__}, "
                    f"got {type(value).__name__}"
                )

        return SkillResult(
            output=value,
            success=True,
            provenance=SkillProvenance(skill_name=self.name),
        )
