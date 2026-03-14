"""
Cognitive Layer protocols (ABCs).

Defines the contracts for skills, LLM providers, prompt building,
context serialization, and meta-skills. All concrete implementations
in the cognitive layer depend only on these abstractions.

Phase 3.1.1 — Foundation protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from .models import (
        BuiltPrompt,
        CompletionResult,
        Message,
        SkillConfig,
        SkillContext,
        SkillPlan,
        SkillResult,
        ToolResult,
    )

T = TypeVar("T")


# ──────────────────────────────────────────────
# Skill Protocol
# ──────────────────────────────────────────────


class SkillProtocol(ABC):
    """Atomic cognitive unit — plan/interpret split (DD-L1-01 Option A).

    Skills are stateless (DI-L1-01). All state flows through
    input (SkillContext + input data) and output (SkillPlan/SkillResult).
    Configuration is immutable and set at construction time.
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

    @abstractmethod
    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        """Produce an execution plan from input and pre-loaded context.

        The returned SkillPlan describes what work to do — the orchestrator
        decides how and when to execute it.

        Args:
            input: Typed input data for this skill invocation.
            context: Pre-loaded SkillContext with memories, artifacts, budget.

        Returns:
            A SkillPlan with typed steps and optional constraints.
        """
        ...

    @abstractmethod
    async def interpret(self, result: Any) -> SkillResult:
        """Interpret orchestrator results into a typed SkillResult.

        Called after the orchestrator executes the plan. Transforms
        raw node results into the skill's typed output with provenance.

        Args:
            result: Adapter-normalized result from the orchestrator.

        Returns:
            A SkillResult with typed output, provenance, and optional
            suggested memories/artifacts.
        """
        ...

    async def validate_output(self, output: Any) -> bool:
        """Optional output quality check (OQ resolved: YES).

        When validation fails, raise RetryableValidationError to trigger
        L3's RetryPolicy for automatic retry with exponential backoff.

        Args:
            output: The parsed output to validate.

        Returns:
            True if output is acceptable, False otherwise.
        """
        return True


# ──────────────────────────────────────────────
# LLM Protocol (DD-L1-02 Option B)
# ──────────────────────────────────────────────


class LLMProtocol(ABC):
    """Provider-agnostic LLM interface.

    Three methods mapping to L3 event kinds:
    - complete()            → TEXT events  → LLMResultData
    - complete_structured() → DATA events  → LLMResultData
    - complete_with_tools() → TOOL events  → ToolResultData

    Streaming and embeddings are separate protocols (ISP).
    """

    @abstractmethod
    async def complete(self, messages: list[Message]) -> CompletionResult:
        """Text completion.

        Args:
            messages: Conversation messages.

        Returns:
            CompletionResult with text, token usage, and metadata.
        """
        ...

    @abstractmethod
    async def complete_structured(
        self, messages: list[Message], schema: type[T]
    ) -> T:
        """Structured output — parse LLM response into a typed model.

        Args:
            messages: Conversation messages.
            schema: Pydantic model class for the expected output.

        Returns:
            An instance of the schema type, parsed from LLM output.
        """
        ...

    @abstractmethod
    async def complete_with_tools(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ToolResult:
        """LLM call with tool/function calling.

        Args:
            messages: Conversation messages.
            tools: Tool definitions (JSON Schema format).

        Returns:
            ToolResult with the LLM's tool call decision and any results.
        """
        ...


# ──────────────────────────────────────────────
# Streaming & Embedding Protocols (ISP stubs)
# ──────────────────────────────────────────────


class StreamingLLMProtocol(LLMProtocol):
    """LLM with streaming support — add when needed."""

    @abstractmethod
    async def stream(self, messages: list[Message]) -> Any:
        """Stream completion tokens.

        Returns:
            An async iterator yielding text chunks.
        """
        ...


class EmbeddingProtocol(ABC):
    """Embedding generation — separate from LLM (ISP)."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: Input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...


# ──────────────────────────────────────────────
# Prompt Protocol (DD-L1-03 Option D)
# ──────────────────────────────────────────────


class PromptProtocol(ABC):
    """Prompt construction contract.

    Implementations produce BuiltPrompt(system_prompt, prompt)
    matching L3's TextPayload fields.
    """

    @abstractmethod
    def build(self) -> BuiltPrompt:
        """Build the final prompt.

        Returns:
            BuiltPrompt with system_prompt and prompt strings.
        """
        ...


# ──────────────────────────────────────────────
# Context Serializer Protocol
# ──────────────────────────────────────────────


class ContextSerializerProtocol(ABC):
    """Render memories and artifacts into LLM-consumable text.

    Bridge between the ContextStore (L3) and prompt injection.
    Implementations may be budget-aware (DI-L1-04).
    """

    @abstractmethod
    def serialize(
        self,
        memories: list[Any],
        artifacts: list[Any],
    ) -> str:
        """Render memories and artifacts into a text string.

        Args:
            memories: List of Memory objects from the context store.
            artifacts: List of Artifact objects from the context store.

        Returns:
            A formatted string suitable for prompt injection.
        """
        ...


# ──────────────────────────────────────────────
# MetaSkill Protocol (DI-L1-03 — V2 stub)
# ──────────────────────────────────────────────


class MetaSkillProtocol(SkillProtocol):
    """MetaSkill — generates skills or DAGs from descriptions.

    V1: interface only. Both methods raise NotImplementedError.
    V2: implement to enable declarative skill/DAG generation.
    """

    @abstractmethod
    async def generate_skill(
        self, description: str, context: SkillContext
    ) -> SkillConfig:
        """Generate a SkillConfig from a natural language description.

        Args:
            description: What the skill should do.
            context: Pre-loaded context for generation decisions.

        Returns:
            A SkillConfig that can be used with ConfigSkill.
        """
        ...

    @abstractmethod
    async def generate_dag(
        self, description: str, context: SkillContext
    ) -> Any:
        """Generate a PlanDAG from a natural language description.

        Args:
            description: What the DAG should accomplish.
            context: Pre-loaded context for generation decisions.

        Returns:
            A PlanDAG (L2 type) describing the orchestration graph.
        """
        ...
