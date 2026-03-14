"""
OpenAI-backed handlers for the Execution Platform.

Concrete TextHandler and DataHandler implementations that call
the OpenAI Chat Completions API. Requires the ``openai`` package.

Usage::

    from rh_cognitv.execution_platform.openai_handler import OpenAITextHandler

    handler = OpenAITextHandler(api_key="sk-...", model="gpt-4o-mini")
    registry.register(EventKind.TEXT, handler)
"""

from __future__ import annotations

from typing import Any

import openai

from .events import DataPayload, ExecutionEvent, TextPayload
from .models import (
    ExecutionResult,
    LLMResultData,
    ResultMetadata,
    TokenUsage,
)
from .protocols import EventHandlerProtocol


class OpenAITextHandler(EventHandlerProtocol[LLMResultData]):
    """TEXT handler backed by OpenAI Chat Completions API.

    Args:
        api_key: OpenAI API key.
        model: Default model name (overridden by TextPayload.model if set).
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        payload = event.payload
        if not isinstance(payload, TextPayload):
            return ExecutionResult(
                ok=False,
                error_message=(
                    f"OpenAITextHandler expects TextPayload, "
                    f"got {type(payload).__name__}"
                ),
            )

        messages: list[dict[str, str]] = []
        if payload.system_prompt:
            messages.append({"role": "system", "content": payload.system_prompt})
        messages.append({"role": "user", "content": payload.prompt})

        kwargs: dict[str, Any] = {
            "model": payload.model or self._model,
            "messages": messages,
        }
        if payload.temperature is not None:
            kwargs["temperature"] = payload.temperature
        if payload.max_tokens is not None:
            kwargs["max_tokens"] = payload.max_tokens

        response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=choice.message.content or "",
                model=response.model,
                token_usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total=prompt_tokens + completion_tokens,
                ),
                finish_reason=choice.finish_reason or "",
            ),
            metadata=ResultMetadata(),
        )


class OpenAIDataHandler(EventHandlerProtocol[LLMResultData]):
    """DATA handler backed by OpenAI Chat Completions with JSON mode.

    Args:
        api_key: OpenAI API key.
        model: Default model name (overridden by DataPayload.model if set).
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        payload = event.payload
        if not isinstance(payload, DataPayload):
            return ExecutionResult(
                ok=False,
                error_message=(
                    f"OpenAIDataHandler expects DataPayload, "
                    f"got {type(payload).__name__}"
                ),
            )

        messages: list[dict[str, str]] = []
        if payload.output_schema:
            schema_instruction = (
                "Respond with valid JSON matching this schema:\n"
                f"{payload.output_schema}"
            )
            messages.append({"role": "system", "content": schema_instruction})
        messages.append({"role": "user", "content": payload.prompt})

        response = await self._client.chat.completions.create(
            model=payload.model or self._model,
            messages=messages,
            response_format={"type": "json_object"},
        )

        choice = response.choices[0]
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=choice.message.content or "",
                model=response.model,
                token_usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total=prompt_tokens + completion_tokens,
                ),
                finish_reason=choice.finish_reason or "",
            ),
            metadata=ResultMetadata(),
        )
