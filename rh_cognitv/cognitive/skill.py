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

from .models import SkillContext, SkillPlan, SkillResult
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
