"""
Handler registry and concrete handlers per event kind.

DD-L3-05: Strategy pattern — HandlerRegistry dispatches to the correct
EventHandler[T] based on the event's `kind`. Each handler returns
ExecutionResult[T] with kind-specific T.

Concrete handlers (TextHandler, DataHandler, FunctionHandler, ToolHandler)
are base implementations. Upper layers or users are expected to subclass
and provide real implementations (e.g., wiring to an LLM SDK).
"""

from __future__ import annotations

from typing import Any

from .errors import PermanentError, ValidationError
from .events import (
    DataPayload,
    ExecutionEvent,
    FunctionPayload,
    TextPayload,
    ToolPayload,
)
from .models import (
    EventKind,
    ExecutionResult,
    FunctionResultData,
    LLMResultData,
    ResultMetadata,
    ToolResultData,
)
from .protocols import EventHandlerProtocol, HandlerRegistryProtocol


# ──────────────────────────────────────────────
# Handler Registry
# ──────────────────────────────────────────────


class HandlerRegistry(HandlerRegistryProtocol):
    """Maps EventKind → EventHandler and dispatches execution.

    Usage:
        registry = HandlerRegistry()
        registry.register(EventKind.TEXT, TextHandler())
        result = await registry.handle(event, data, configs)
    """

    def __init__(self) -> None:
        self._handlers: dict[EventKind, EventHandlerProtocol[Any]] = {}

    def register(self, kind: EventKind, handler: EventHandlerProtocol[Any]) -> None:
        """Register a handler for a specific event kind."""
        self._handlers[kind] = handler

    async def handle(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[Any]:
        """Dispatch to the registered handler for the event's kind.

        Raises:
            ValidationError: If no handler is registered for the event kind.
        """
        handler = self._handlers.get(event.kind)
        if handler is None:
            raise ValidationError(f"No handler registered for kind: {event.kind.value}")
        return await handler.execute(event, data, configs)

    def has_handler(self, kind: EventKind) -> bool:
        """Check if a handler is registered for the given kind."""
        return kind in self._handlers


# ──────────────────────────────────────────────
# Concrete Handlers (Base Implementations)
# ──────────────────────────────────────────────


class TextHandler(EventHandlerProtocol[LLMResultData]):
    """Handler for TEXT events — LLM text generation.

    This is a base implementation that returns a placeholder result.
    Subclass and override `execute()` to wire to a real LLM SDK.
    """

    async def execute(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        if not isinstance(event.payload, TextPayload):
            raise ValidationError(
                f"TextHandler expects TextPayload, got {type(event.payload).__name__}"
            )
        return ExecutionResult(
            ok=True,
            value=LLMResultData(text="", model=event.payload.model or ""),
            metadata=ResultMetadata(),
        )


class DataHandler(EventHandlerProtocol[LLMResultData]):
    """Handler for DATA events — structured data generation.

    Base implementation returning a placeholder. Subclass for real use.
    """

    async def execute(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        if not isinstance(event.payload, DataPayload):
            raise ValidationError(
                f"DataHandler expects DataPayload, got {type(event.payload).__name__}"
            )
        return ExecutionResult(
            ok=True,
            value=LLMResultData(text="", model=event.payload.model or ""),
            metadata=ResultMetadata(),
        )


class FunctionHandler(EventHandlerProtocol[FunctionResultData]):
    """Handler for FUNCTION events — direct function invocation.

    Base implementation returning a placeholder. Subclass for real use.
    """

    async def execute(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[FunctionResultData]:
        if not isinstance(event.payload, FunctionPayload):
            raise ValidationError(
                f"FunctionHandler expects FunctionPayload, got {type(event.payload).__name__}"
            )
        return ExecutionResult(
            ok=True,
            value=FunctionResultData(return_value=None, duration_ms=0.0),
            metadata=ResultMetadata(),
        )


class ToolHandler(EventHandlerProtocol[ToolResultData]):
    """Handler for TOOL events — LLM-driven tool use (LLM call + function).

    Base implementation returning a placeholder. Subclass for real use.
    """

    async def execute(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[ToolResultData]:
        if not isinstance(event.payload, ToolPayload):
            raise ValidationError(
                f"ToolHandler expects ToolPayload, got {type(event.payload).__name__}"
            )
        return ExecutionResult(
            ok=True,
            value=ToolResultData(
                llm_result=LLMResultData(text="", model=event.payload.model or ""),
                function_result=FunctionResultData(return_value=None, duration_ms=0.0),
            ),
            metadata=ResultMetadata(),
        )
