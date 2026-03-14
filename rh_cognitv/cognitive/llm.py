"""
LLM provider abstraction — MockLLM reference implementation.

Phase 3.2 — Provider-agnostic LLM interface with a mock for testing.

MockLLM:
- Returns configurable canned responses
- Records all calls for test inspection
- Tracks cumulative token usage
- Supports structured output via Pydantic schema parsing
- Supports tool call simulation
- Can simulate failures on demand
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from .models import (
    CompletionResult,
    Message,
    ToolCall,
    ToolResult,
)
from .protocols import LLMProtocol

T = TypeVar("T")


# ──────────────────────────────────────────────
# Call Records — for test inspection
# ──────────────────────────────────────────────


@dataclass
class CompletionCall:
    """Recorded call to complete()."""

    messages: list[Message]


@dataclass
class StructuredCall:
    """Recorded call to complete_structured()."""

    messages: list[Message]
    schema: type


@dataclass
class ToolsCall:
    """Recorded call to complete_with_tools()."""

    messages: list[Message]
    tools: list[dict[str, Any]]


@dataclass
class CallLog:
    """Accumulated call history for a MockLLM instance."""

    completions: list[CompletionCall] = field(default_factory=list)
    structured: list[StructuredCall] = field(default_factory=list)
    tools: list[ToolsCall] = field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return len(self.completions) + len(self.structured) + len(self.tools)

    def clear(self) -> None:
        self.completions.clear()
        self.structured.clear()
        self.tools.clear()


# ──────────────────────────────────────────────
# MockLLM — reference implementation of LLMProtocol
# ──────────────────────────────────────────────


class MockLLM(LLMProtocol):
    """Mock LLM for testing — returns canned responses, records calls.

    Usage::

        llm = MockLLM(responses=["Hello!", "World!"])
        result = await llm.complete([Message(role=MessageRole.USER, content="Hi")])
        assert result.text == "Hello!"
        assert llm.call_log.total_calls == 1

    Responses are consumed in order. When exhausted, the default response
    is returned. Use ``structured_responses`` for ``complete_structured()``
    and ``tool_responses`` for ``complete_with_tools()``.

    Args:
        responses: Canned text responses for ``complete()``.
        structured_responses: Canned responses for ``complete_structured()``.
            Each should be a dict or BaseModel instance matching the requested schema.
        tool_responses: Canned ToolResult responses for ``complete_with_tools()``.
        default_response: Fallback text when ``responses`` is exhausted.
        model: Model name to include in results.
        prompt_tokens_per_call: Simulated prompt tokens per call.
        completion_tokens_per_call: Simulated completion tokens per call.
        error_on_call: If set, raise this exception on the next call (then clear it).
    """

    def __init__(
        self,
        *,
        responses: list[str] | None = None,
        structured_responses: list[Any] | None = None,
        tool_responses: list[ToolResult] | None = None,
        default_response: str = "mock response",
        model: str = "mock-model",
        prompt_tokens_per_call: int = 10,
        completion_tokens_per_call: int = 20,
    ) -> None:
        self._responses: list[str] = list(responses or [])
        self._structured_responses: list[Any] = list(structured_responses or [])
        self._tool_responses: list[ToolResult] = list(tool_responses or [])
        self._default_response = default_response
        self._model = model
        self._prompt_tokens_per_call = prompt_tokens_per_call
        self._completion_tokens_per_call = completion_tokens_per_call

        self._response_index = 0
        self._structured_index = 0
        self._tool_index = 0

        self.call_log = CallLog()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        self._error_queue: list[Exception] = []

    # ── Configuration helpers ──

    def queue_error(self, error: Exception) -> None:
        """Queue an error to raise on the next call."""
        self._error_queue.append(error)

    def set_responses(self, responses: list[str]) -> None:
        """Replace the response queue."""
        self._responses = list(responses)
        self._response_index = 0

    def set_structured_responses(self, responses: list[Any]) -> None:
        """Replace the structured response queue."""
        self._structured_responses = list(responses)
        self._structured_index = 0

    def set_tool_responses(self, responses: list[ToolResult]) -> None:
        """Replace the tool response queue."""
        self._tool_responses = list(responses)
        self._tool_index = 0

    # ── LLMProtocol implementation ──

    async def complete(self, messages: list[Message]) -> CompletionResult:
        self._maybe_raise()
        self.call_log.completions.append(CompletionCall(messages=messages))

        text = self._next_response()
        prompt_tokens = self._prompt_tokens_per_call
        completion_tokens = self._completion_tokens_per_call
        total_tokens = prompt_tokens + completion_tokens

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return CompletionResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=self._model,
            finish_reason="stop",
        )

    async def complete_structured(
        self, messages: list[Message], schema: type[T]
    ) -> T:
        self._maybe_raise()
        self.call_log.structured.append(
            StructuredCall(messages=messages, schema=schema)
        )

        prompt_tokens = self._prompt_tokens_per_call
        completion_tokens = self._completion_tokens_per_call

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return self._next_structured_response(schema)

    async def complete_with_tools(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ToolResult:
        self._maybe_raise()
        self.call_log.tools.append(ToolsCall(messages=messages, tools=tools))

        prompt_tokens = self._prompt_tokens_per_call
        completion_tokens = self._completion_tokens_per_call

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return self._next_tool_response()

    # ── Internal helpers ──

    def _maybe_raise(self) -> None:
        if self._error_queue:
            raise self._error_queue.pop(0)

    def _next_response(self) -> str:
        if self._response_index < len(self._responses):
            text = self._responses[self._response_index]
            self._response_index += 1
            return text
        return self._default_response

    def _next_structured_response(self, schema: type[T]) -> T:
        if self._structured_index < len(self._structured_responses):
            raw = self._structured_responses[self._structured_index]
            self._structured_index += 1
            if isinstance(raw, schema):
                return raw
            if isinstance(raw, dict) and issubclass(schema, BaseModel):
                return schema.model_validate(raw)
            return raw  # type: ignore[return-value]
        # Default: construct schema with no args (works for models with all defaults)
        if issubclass(schema, BaseModel):
            return schema.model_validate({})
        raise ValueError(
            f"No structured response queued and cannot default-construct {schema.__name__}"
        )

    def _next_tool_response(self) -> ToolResult:
        if self._tool_index < len(self._tool_responses):
            result = self._tool_responses[self._tool_index]
            self._tool_index += 1
            return result
        return ToolResult(
            text=self._default_response,
            model=self._model,
            finish_reason="stop",
            prompt_tokens=self._prompt_tokens_per_call,
            completion_tokens=self._completion_tokens_per_call,
            total_tokens=self._prompt_tokens_per_call + self._completion_tokens_per_call,
        )
