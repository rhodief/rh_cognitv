"""Tests for handlers.py — HandlerRegistry dispatch, concrete handlers."""

import pytest

from rh_cognitv.execution_platform.errors import ValidationError
from rh_cognitv.execution_platform.events import (
    DataPayload,
    ExecutionEvent,
    FunctionPayload,
    TextPayload,
    ToolPayload,
)
from rh_cognitv.execution_platform.handlers import (
    DataHandler,
    FunctionHandler,
    HandlerRegistry,
    TextHandler,
    ToolHandler,
)
from rh_cognitv.execution_platform.models import (
    EventKind,
    ExecutionResult,
    FunctionResultData,
    LLMResultData,
    ToolResultData,
)


# ──────────────────────────────────────────────
# HandlerRegistry
# ──────────────────────────────────────────────


class TestHandlerRegistry:
    def test_register_and_has_handler(self):
        reg = HandlerRegistry()
        reg.register(EventKind.TEXT, TextHandler())
        assert reg.has_handler(EventKind.TEXT) is True
        assert reg.has_handler(EventKind.DATA) is False

    @pytest.mark.asyncio
    async def test_dispatch_to_correct_handler(self):
        reg = HandlerRegistry()
        reg.register(EventKind.TEXT, TextHandler())
        reg.register(EventKind.FUNCTION, FunctionHandler())

        text_event = ExecutionEvent(
            kind=EventKind.TEXT, payload=TextPayload(prompt="hello")
        )
        result = await reg.handle(text_event, None, None)
        assert result.ok is True
        assert isinstance(result.value, LLMResultData)

    @pytest.mark.asyncio
    async def test_dispatch_unregistered_kind_raises(self):
        reg = HandlerRegistry()
        event = ExecutionEvent(
            kind=EventKind.TEXT, payload=TextPayload(prompt="x")
        )
        with pytest.raises(ValidationError, match="No handler registered"):
            await reg.handle(event, None, None)

    @pytest.mark.asyncio
    async def test_register_overwrites_previous(self):
        reg = HandlerRegistry()
        handler1 = TextHandler()
        handler2 = TextHandler()
        reg.register(EventKind.TEXT, handler1)
        reg.register(EventKind.TEXT, handler2)
        # Should use the latest registered handler
        assert reg.has_handler(EventKind.TEXT)

    @pytest.mark.asyncio
    async def test_dispatch_all_four_kinds(self):
        reg = HandlerRegistry()
        reg.register(EventKind.TEXT, TextHandler())
        reg.register(EventKind.DATA, DataHandler())
        reg.register(EventKind.FUNCTION, FunctionHandler())
        reg.register(EventKind.TOOL, ToolHandler())

        events = [
            ExecutionEvent(kind=EventKind.TEXT, payload=TextPayload(prompt="t")),
            ExecutionEvent(kind=EventKind.DATA, payload=DataPayload(prompt="d")),
            ExecutionEvent(
                kind=EventKind.FUNCTION,
                payload=FunctionPayload(function_name="f"),
            ),
            ExecutionEvent(kind=EventKind.TOOL, payload=ToolPayload(prompt="u")),
        ]
        for event in events:
            result = await reg.handle(event, None, None)
            assert result.ok is True


# ──────────────────────────────────────────────
# TextHandler
# ──────────────────────────────────────────────


class TestTextHandler:
    @pytest.mark.asyncio
    async def test_returns_llm_result(self):
        handler = TextHandler()
        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="hello", model="gpt-4"),
        )
        result = await handler(event, None, None)
        assert result.ok is True
        assert isinstance(result.value, LLMResultData)
        assert result.value.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_model_defaults_to_empty(self):
        handler = TextHandler()
        event = ExecutionEvent(
            kind=EventKind.TEXT, payload=TextPayload(prompt="x")
        )
        result = await handler(event, None, None)
        assert result.value.model == ""

    @pytest.mark.asyncio
    async def test_wrong_payload_raises(self):
        handler = TextHandler()
        event = ExecutionEvent(
            kind=EventKind.FUNCTION,
            payload=FunctionPayload(function_name="f"),
        )
        with pytest.raises(ValidationError, match="TextHandler expects TextPayload"):
            await handler(event, None, None)


# ──────────────────────────────────────────────
# DataHandler
# ──────────────────────────────────────────────


class TestDataHandler:
    @pytest.mark.asyncio
    async def test_returns_llm_result(self):
        handler = DataHandler()
        event = ExecutionEvent(
            kind=EventKind.DATA,
            payload=DataPayload(prompt="extract", model="gpt-4"),
        )
        result = await handler(event, None, None)
        assert result.ok is True
        assert isinstance(result.value, LLMResultData)

    @pytest.mark.asyncio
    async def test_wrong_payload_raises(self):
        handler = DataHandler()
        event = ExecutionEvent(
            kind=EventKind.TEXT, payload=TextPayload(prompt="x")
        )
        with pytest.raises(ValidationError, match="DataHandler expects DataPayload"):
            await handler(event, None, None)


# ──────────────────────────────────────────────
# FunctionHandler
# ──────────────────────────────────────────────


class TestFunctionHandler:
    @pytest.mark.asyncio
    async def test_returns_function_result(self):
        handler = FunctionHandler()
        event = ExecutionEvent(
            kind=EventKind.FUNCTION,
            payload=FunctionPayload(function_name="my_func"),
        )
        result = await handler(event, None, None)
        assert result.ok is True
        assert isinstance(result.value, FunctionResultData)
        assert result.value.return_value is None
        assert result.value.duration_ms == 0.0

    @pytest.mark.asyncio
    async def test_wrong_payload_raises(self):
        handler = FunctionHandler()
        event = ExecutionEvent(
            kind=EventKind.TEXT, payload=TextPayload(prompt="x")
        )
        with pytest.raises(
            ValidationError, match="FunctionHandler expects FunctionPayload"
        ):
            await handler(event, None, None)


# ──────────────────────────────────────────────
# ToolHandler
# ──────────────────────────────────────────────


class TestToolHandler:
    @pytest.mark.asyncio
    async def test_returns_tool_result(self):
        handler = ToolHandler()
        event = ExecutionEvent(
            kind=EventKind.TOOL,
            payload=ToolPayload(prompt="use search", model="gpt-4"),
        )
        result = await handler(event, None, None)
        assert result.ok is True
        assert isinstance(result.value, ToolResultData)
        assert isinstance(result.value.llm_result, LLMResultData)
        assert isinstance(result.value.function_result, FunctionResultData)
        assert result.value.llm_result.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_wrong_payload_raises(self):
        handler = ToolHandler()
        event = ExecutionEvent(
            kind=EventKind.TEXT, payload=TextPayload(prompt="x")
        )
        with pytest.raises(ValidationError, match="ToolHandler expects ToolPayload"):
            await handler(event, None, None)

    @pytest.mark.asyncio
    async def test_model_defaults_to_empty(self):
        handler = ToolHandler()
        event = ExecutionEvent(
            kind=EventKind.TOOL, payload=ToolPayload(prompt="x")
        )
        result = await handler(event, None, None)
        assert result.value.llm_result.model == ""


# ──────────────────────────────────────────────
# ExecutionResult Generics
# ──────────────────────────────────────────────


class TestExecutionResultGenerics:
    def test_result_with_llm_data(self):
        result = ExecutionResult[LLMResultData](
            ok=True,
            value=LLMResultData(text="hello", model="gpt-4"),
        )
        assert result.value.text == "hello"

    def test_result_with_function_data(self):
        result = ExecutionResult[FunctionResultData](
            ok=True,
            value=FunctionResultData(return_value=42, duration_ms=1.5),
        )
        assert result.value.return_value == 42

    def test_result_with_tool_data(self):
        result = ExecutionResult[ToolResultData](
            ok=True,
            value=ToolResultData(
                llm_result=LLMResultData(text="call search"),
                function_result=FunctionResultData(return_value={"results": []}),
            ),
        )
        assert result.value.llm_result.text == "call search"
        assert result.value.function_result.return_value == {"results": []}

    def test_failed_result(self):
        result = ExecutionResult[LLMResultData](
            ok=False,
            error_message="rate limited",
            error_category="transient",
        )
        assert result.ok is False
        assert result.value is None
        assert result.error_message == "rate limited"
