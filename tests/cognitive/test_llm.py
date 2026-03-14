"""
Tests for cognitive/llm.py — Phase 3.2

MockLLM protocol compliance, call recording, token usage tracking,
structured output, tool calls, error simulation, and L3 alignment.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from rh_cognitv.cognitive.llm import (
    CallLog,
    CompletionCall,
    MockLLM,
    StructuredCall,
    ToolsCall,
)
from rh_cognitv.cognitive.models import (
    CompletionResult,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)
from rh_cognitv.cognitive.protocols import LLMProtocol


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def user_msg(text: str) -> Message:
    return Message(role=MessageRole.USER, content=text)


def system_msg(text: str) -> Message:
    return Message(role=MessageRole.SYSTEM, content=text)


class SummaryOutput(BaseModel):
    summary: str = ""
    word_count: int = 0


class ExtractOutput(BaseModel):
    name: str = ""
    age: int = 0


# ──────────────────────────────────────────────
# Protocol Compliance
# ──────────────────────────────────────────────


class TestProtocolCompliance:
    def test_mock_llm_is_llm_protocol(self):
        llm = MockLLM()
        assert isinstance(llm, LLMProtocol)

    def test_has_complete_method(self):
        assert hasattr(MockLLM, "complete")

    def test_has_complete_structured_method(self):
        assert hasattr(MockLLM, "complete_structured")

    def test_has_complete_with_tools_method(self):
        assert hasattr(MockLLM, "complete_with_tools")


# ──────────────────────────────────────────────
# complete()
# ──────────────────────────────────────────────


class TestComplete:
    @pytest.mark.asyncio
    async def test_returns_completion_result(self):
        llm = MockLLM()
        result = await llm.complete([user_msg("Hi")])
        assert isinstance(result, CompletionResult)

    @pytest.mark.asyncio
    async def test_default_response(self):
        llm = MockLLM()
        result = await llm.complete([user_msg("Hi")])
        assert result.text == "mock response"

    @pytest.mark.asyncio
    async def test_custom_default_response(self):
        llm = MockLLM(default_response="custom default")
        result = await llm.complete([user_msg("Hi")])
        assert result.text == "custom default"

    @pytest.mark.asyncio
    async def test_canned_responses_consumed_in_order(self):
        llm = MockLLM(responses=["first", "second", "third"])
        r1 = await llm.complete([user_msg("1")])
        r2 = await llm.complete([user_msg("2")])
        r3 = await llm.complete([user_msg("3")])
        assert r1.text == "first"
        assert r2.text == "second"
        assert r3.text == "third"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_when_exhausted(self):
        llm = MockLLM(responses=["only"])
        r1 = await llm.complete([user_msg("1")])
        r2 = await llm.complete([user_msg("2")])
        assert r1.text == "only"
        assert r2.text == "mock response"

    @pytest.mark.asyncio
    async def test_model_name_in_result(self):
        llm = MockLLM(model="gpt-4-turbo")
        result = await llm.complete([user_msg("Hi")])
        assert result.model == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_finish_reason(self):
        llm = MockLLM()
        result = await llm.complete([user_msg("Hi")])
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_token_usage_in_result(self):
        llm = MockLLM(prompt_tokens_per_call=15, completion_tokens_per_call=25)
        result = await llm.complete([user_msg("Hi")])
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 25
        assert result.total_tokens == 40

    @pytest.mark.asyncio
    async def test_set_responses_replaces_queue(self):
        llm = MockLLM(responses=["old"])
        llm.set_responses(["new1", "new2"])
        r1 = await llm.complete([user_msg("1")])
        r2 = await llm.complete([user_msg("2")])
        assert r1.text == "new1"
        assert r2.text == "new2"


# ──────────────────────────────────────────────
# complete_structured()
# ──────────────────────────────────────────────


class TestCompleteStructured:
    @pytest.mark.asyncio
    async def test_returns_schema_instance(self):
        llm = MockLLM(structured_responses=[
            SummaryOutput(summary="Short text", word_count=2),
        ])
        result = await llm.complete_structured([user_msg("Summarize")], SummaryOutput)
        assert isinstance(result, SummaryOutput)
        assert result.summary == "Short text"
        assert result.word_count == 2

    @pytest.mark.asyncio
    async def test_dict_response_validated_into_schema(self):
        llm = MockLLM(structured_responses=[
            {"name": "Alice", "age": 30},
        ])
        result = await llm.complete_structured([user_msg("Extract")], ExtractOutput)
        assert isinstance(result, ExtractOutput)
        assert result.name == "Alice"
        assert result.age == 30

    @pytest.mark.asyncio
    async def test_default_constructs_schema_with_defaults(self):
        llm = MockLLM()  # no structured_responses
        result = await llm.complete_structured([user_msg("X")], SummaryOutput)
        assert isinstance(result, SummaryOutput)
        assert result.summary == ""
        assert result.word_count == 0

    @pytest.mark.asyncio
    async def test_multiple_structured_responses_in_order(self):
        llm = MockLLM(structured_responses=[
            ExtractOutput(name="Alice", age=30),
            ExtractOutput(name="Bob", age=25),
        ])
        r1 = await llm.complete_structured([user_msg("1")], ExtractOutput)
        r2 = await llm.complete_structured([user_msg("2")], ExtractOutput)
        assert r1.name == "Alice"
        assert r2.name == "Bob"

    @pytest.mark.asyncio
    async def test_set_structured_responses_replaces_queue(self):
        llm = MockLLM(structured_responses=[ExtractOutput(name="old")])
        llm.set_structured_responses([ExtractOutput(name="new")])
        result = await llm.complete_structured([user_msg("1")], ExtractOutput)
        assert result.name == "new"

    @pytest.mark.asyncio
    async def test_tracks_token_usage(self):
        llm = MockLLM(
            structured_responses=[SummaryOutput()],
            prompt_tokens_per_call=5,
            completion_tokens_per_call=10,
        )
        await llm.complete_structured([user_msg("X")], SummaryOutput)
        assert llm.total_prompt_tokens == 5
        assert llm.total_completion_tokens == 10


# ──────────────────────────────────────────────
# complete_with_tools()
# ──────────────────────────────────────────────


class TestCompleteWithTools:
    @pytest.mark.asyncio
    async def test_returns_tool_result(self):
        llm = MockLLM()
        tools = [{"type": "function", "function": {"name": "search"}}]
        result = await llm.complete_with_tools([user_msg("Search")], tools)
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_default_tool_response(self):
        llm = MockLLM(default_response="tool default")
        result = await llm.complete_with_tools([user_msg("X")], [])
        assert result.text == "tool default"

    @pytest.mark.asyncio
    async def test_canned_tool_responses(self):
        custom = ToolResult(
            text="I'll search for that",
            tool_calls=[ToolCall(name="search", arguments={"q": "info"})],
        )
        llm = MockLLM(tool_responses=[custom])
        result = await llm.complete_with_tools([user_msg("Find")], [])
        assert result.text == "I'll search for that"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_falls_back_when_exhausted(self):
        custom = ToolResult(text="first")
        llm = MockLLM(tool_responses=[custom], default_response="fallback")
        r1 = await llm.complete_with_tools([user_msg("1")], [])
        r2 = await llm.complete_with_tools([user_msg("2")], [])
        assert r1.text == "first"
        assert r2.text == "fallback"

    @pytest.mark.asyncio
    async def test_set_tool_responses_replaces_queue(self):
        llm = MockLLM(tool_responses=[ToolResult(text="old")])
        llm.set_tool_responses([ToolResult(text="new")])
        result = await llm.complete_with_tools([user_msg("X")], [])
        assert result.text == "new"

    @pytest.mark.asyncio
    async def test_tracks_token_usage(self):
        llm = MockLLM(prompt_tokens_per_call=8, completion_tokens_per_call=12)
        await llm.complete_with_tools([user_msg("X")], [])
        assert llm.total_prompt_tokens == 8
        assert llm.total_completion_tokens == 12


# ──────────────────────────────────────────────
# Call Recording
# ──────────────────────────────────────────────


class TestCallRecording:
    @pytest.mark.asyncio
    async def test_records_complete_calls(self):
        llm = MockLLM()
        msgs = [system_msg("Be nice"), user_msg("Hi")]
        await llm.complete(msgs)
        assert len(llm.call_log.completions) == 1
        assert isinstance(llm.call_log.completions[0], CompletionCall)
        assert llm.call_log.completions[0].messages == msgs

    @pytest.mark.asyncio
    async def test_records_structured_calls(self):
        llm = MockLLM()
        msgs = [user_msg("Extract")]
        await llm.complete_structured(msgs, SummaryOutput)
        assert len(llm.call_log.structured) == 1
        assert isinstance(llm.call_log.structured[0], StructuredCall)
        assert llm.call_log.structured[0].schema is SummaryOutput

    @pytest.mark.asyncio
    async def test_records_tool_calls(self):
        llm = MockLLM()
        msgs = [user_msg("Search")]
        tools = [{"type": "function"}]
        await llm.complete_with_tools(msgs, tools)
        assert len(llm.call_log.tools) == 1
        assert isinstance(llm.call_log.tools[0], ToolsCall)
        assert llm.call_log.tools[0].tools == tools

    @pytest.mark.asyncio
    async def test_total_calls_counts_all_types(self):
        llm = MockLLM()
        await llm.complete([user_msg("1")])
        await llm.complete_structured([user_msg("2")], SummaryOutput)
        await llm.complete_with_tools([user_msg("3")], [])
        assert llm.call_log.total_calls == 3

    @pytest.mark.asyncio
    async def test_clear_resets_call_log(self):
        llm = MockLLM()
        await llm.complete([user_msg("1")])
        await llm.complete_structured([user_msg("2")], SummaryOutput)
        await llm.complete_with_tools([user_msg("3")], [])
        assert llm.call_log.total_calls == 3
        llm.call_log.clear()
        assert llm.call_log.total_calls == 0


# ──────────────────────────────────────────────
# Cumulative Token Usage
# ──────────────────────────────────────────────


class TestTokenUsageTracking:
    @pytest.mark.asyncio
    async def test_accumulates_across_complete_calls(self):
        llm = MockLLM(prompt_tokens_per_call=10, completion_tokens_per_call=20)
        await llm.complete([user_msg("1")])
        await llm.complete([user_msg("2")])
        await llm.complete([user_msg("3")])
        assert llm.total_prompt_tokens == 30
        assert llm.total_completion_tokens == 60

    @pytest.mark.asyncio
    async def test_accumulates_across_all_call_types(self):
        llm = MockLLM(prompt_tokens_per_call=5, completion_tokens_per_call=10)
        await llm.complete([user_msg("1")])
        await llm.complete_structured([user_msg("2")], SummaryOutput)
        await llm.complete_with_tools([user_msg("3")], [])
        assert llm.total_prompt_tokens == 15
        assert llm.total_completion_tokens == 30

    @pytest.mark.asyncio
    async def test_zero_token_config(self):
        llm = MockLLM(prompt_tokens_per_call=0, completion_tokens_per_call=0)
        await llm.complete([user_msg("1")])
        assert llm.total_prompt_tokens == 0
        assert llm.total_completion_tokens == 0


# ──────────────────────────────────────────────
# Error Simulation
# ──────────────────────────────────────────────


class TestErrorSimulation:
    @pytest.mark.asyncio
    async def test_queue_error_raises_on_complete(self):
        llm = MockLLM()
        llm.queue_error(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            await llm.complete([user_msg("Hi")])

    @pytest.mark.asyncio
    async def test_queue_error_raises_on_structured(self):
        llm = MockLLM()
        llm.queue_error(RuntimeError("fail"))
        with pytest.raises(RuntimeError, match="fail"):
            await llm.complete_structured([user_msg("X")], SummaryOutput)

    @pytest.mark.asyncio
    async def test_queue_error_raises_on_tools(self):
        llm = MockLLM()
        llm.queue_error(ConnectionError("timeout"))
        with pytest.raises(ConnectionError, match="timeout"):
            await llm.complete_with_tools([user_msg("X")], [])

    @pytest.mark.asyncio
    async def test_error_consumed_then_normal(self):
        llm = MockLLM(responses=["after error"])
        llm.queue_error(ValueError("once"))
        with pytest.raises(ValueError):
            await llm.complete([user_msg("1")])
        # Next call succeeds
        result = await llm.complete([user_msg("2")])
        assert result.text == "after error"

    @pytest.mark.asyncio
    async def test_multiple_queued_errors(self):
        llm = MockLLM()
        llm.queue_error(ValueError("first"))
        llm.queue_error(RuntimeError("second"))
        with pytest.raises(ValueError, match="first"):
            await llm.complete([user_msg("1")])
        with pytest.raises(RuntimeError, match="second"):
            await llm.complete([user_msg("2")])
        # Third call succeeds
        result = await llm.complete([user_msg("3")])
        assert result.text == "mock response"

    @pytest.mark.asyncio
    async def test_error_does_not_record_call(self):
        llm = MockLLM()
        llm.queue_error(ValueError("boom"))
        with pytest.raises(ValueError):
            await llm.complete([user_msg("Hi")])
        assert llm.call_log.total_calls == 0

    @pytest.mark.asyncio
    async def test_error_does_not_accumulate_tokens(self):
        llm = MockLLM(prompt_tokens_per_call=10)
        llm.queue_error(ValueError("boom"))
        with pytest.raises(ValueError):
            await llm.complete([user_msg("Hi")])
        assert llm.total_prompt_tokens == 0


# ──────────────────────────────────────────────
# L3 Alignment — CompletionResult ↔ LLMResultData
# ──────────────────────────────────────────────


class TestL3Alignment:
    @pytest.mark.asyncio
    async def test_completion_result_maps_to_llm_result_data(self):
        """Verify MockLLM output can be mechanically mapped to L3's LLMResultData."""
        from rh_cognitv.execution_platform.models import LLMResultData, TokenUsage

        llm = MockLLM(
            responses=["Generated text"],
            model="gpt-4",
            prompt_tokens_per_call=100,
            completion_tokens_per_call=50,
        )
        result = await llm.complete([user_msg("Do something")])

        # Map CompletionResult → LLMResultData (what the adapter would do)
        l3_result = LLMResultData(
            text=result.text,
            thinking=result.thinking,
            token_usage=TokenUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total=result.total_tokens,
            ),
            model=result.model,
            finish_reason=result.finish_reason,
        )
        assert l3_result.text == "Generated text"
        assert l3_result.model == "gpt-4"
        assert l3_result.token_usage.prompt_tokens == 100
        assert l3_result.token_usage.completion_tokens == 50
        assert l3_result.token_usage.total == 150

    @pytest.mark.asyncio
    async def test_tool_result_maps_to_tool_result_data(self):
        """Verify MockLLM tool output can be mapped to L3's ToolResultData."""
        from rh_cognitv.execution_platform.models import (
            FunctionResultData,
            LLMResultData,
            ToolResultData,
        )

        llm = MockLLM(tool_responses=[
            ToolResult(
                text="I'll call search",
                tool_calls=[ToolCall(name="search", arguments={"q": "test"})],
            ),
        ])
        result = await llm.complete_with_tools([user_msg("Find")], [])

        # Map ToolResult → ToolResultData (what the adapter would do)
        l3_result = ToolResultData(
            llm_result=LLMResultData(text=result.text),
            function_result=FunctionResultData(
                return_value=result.tool_calls[0].arguments if result.tool_calls else None,
            ),
        )
        assert l3_result.llm_result.text == "I'll call search"


# ──────────────────────────────────────────────
# CallLog Unit Tests
# ──────────────────────────────────────────────


class TestCallLog:
    def test_empty_log(self):
        log = CallLog()
        assert log.total_calls == 0
        assert log.completions == []
        assert log.structured == []
        assert log.tools == []

    def test_clear(self):
        log = CallLog()
        log.completions.append(CompletionCall(messages=[]))
        log.structured.append(StructuredCall(messages=[], schema=SummaryOutput))
        log.tools.append(ToolsCall(messages=[], tools=[]))
        assert log.total_calls == 3
        log.clear()
        assert log.total_calls == 0


# ──────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_messages_list(self):
        llm = MockLLM()
        result = await llm.complete([])
        assert isinstance(result, CompletionResult)

    @pytest.mark.asyncio
    async def test_multi_message_conversation(self):
        llm = MockLLM(responses=["reply"])
        msgs = [
            system_msg("Be helpful"),
            user_msg("Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
            user_msg("Tell me more"),
        ]
        result = await llm.complete(msgs)
        assert result.text == "reply"
        assert llm.call_log.completions[0].messages == msgs

    @pytest.mark.asyncio
    async def test_structured_with_non_default_schema_raises(self):
        """Schema without defaults and no queued response should raise."""

        class StrictOutput(BaseModel):
            required_field: str  # no default

        llm = MockLLM()
        with pytest.raises(Exception):  # Pydantic ValidationError
            await llm.complete_structured([user_msg("X")], StrictOutput)
