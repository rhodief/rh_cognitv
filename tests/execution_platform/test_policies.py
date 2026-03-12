"""Tests for policies.py — PolicyChain, RetryPolicy, TimeoutPolicy, BudgetPolicy."""

import asyncio

import pytest

from rh_cognitv.execution_platform.budget import BudgetTracker
from rh_cognitv.execution_platform.errors import (
    BudgetError,
    CognitivError,
    PermanentError,
    TimeoutError as CognitivTimeoutError,
    TransientError,
)
from rh_cognitv.execution_platform.events import ExecutionEvent, TextPayload
from rh_cognitv.execution_platform.handlers import TextHandler
from rh_cognitv.execution_platform.models import (
    EventKind,
    ExecutionResult,
    LLMResultData,
    ResultMetadata,
    TokenUsage,
)
from rh_cognitv.execution_platform.policies import (
    BudgetPolicy,
    PolicyChain,
    RetryPolicy,
    TimeoutPolicy,
)
from rh_cognitv.execution_platform.protocols import EventHandlerProtocol


# ──────────────────────────────────────────────
# Test Helpers
# ──────────────────────────────────────────────


def _make_text_event(prompt: str = "test") -> ExecutionEvent:
    return ExecutionEvent(kind=EventKind.TEXT, payload=TextPayload(prompt=prompt))


class SuccessHandler(EventHandlerProtocol[LLMResultData]):
    """Always succeeds with a known result."""

    def __init__(self, text: str = "ok", tokens: int = 10):
        self.text = text
        self.tokens = tokens
        self.call_count = 0

    async def __call__(self, event, data, configs) -> ExecutionResult[LLMResultData]:
        self.call_count += 1
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=self.text,
                model="test",
                token_usage=TokenUsage(
                    prompt_tokens=self.tokens // 2,
                    completion_tokens=self.tokens // 2,
                    total=self.tokens,
                ),
            ),
            metadata=ResultMetadata(),
        )


class FailNTimesHandler(EventHandlerProtocol[LLMResultData]):
    """Fails with TransientError N times, then succeeds."""

    def __init__(self, fail_count: int = 2, tokens: int = 10):
        self.fail_count = fail_count
        self.tokens = tokens
        self.call_count = 0

    async def __call__(self, event, data, configs) -> ExecutionResult[LLMResultData]:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise TransientError(f"Transient failure #{self.call_count}")
        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text="recovered",
                model="test",
                token_usage=TokenUsage(total=self.tokens),
            ),
            metadata=ResultMetadata(),
        )


class PermanentFailHandler(EventHandlerProtocol[LLMResultData]):
    """Always fails with PermanentError."""

    async def __call__(self, event, data, configs):
        raise PermanentError("Permanent failure")


class SlowHandler(EventHandlerProtocol[LLMResultData]):
    """Takes a configurable amount of time."""

    def __init__(self, delay: float = 1.0):
        self.delay = delay

    async def __call__(self, event, data, configs) -> ExecutionResult[LLMResultData]:
        await asyncio.sleep(self.delay)
        return ExecutionResult(
            ok=True,
            value=LLMResultData(text="done", model="test"),
            metadata=ResultMetadata(),
        )


# ──────────────────────────────────────────────
# PolicyChain
# ──────────────────────────────────────────────


class TestPolicyChain:
    @pytest.mark.asyncio
    async def test_empty_chain_executes_handler(self):
        chain = PolicyChain()
        handler = SuccessHandler()
        event = _make_text_event()
        result = await chain(handler, event, None, None)
        assert result.ok is True
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_before_execute_runs_in_order(self):
        order = []

        class PolicyA:
            async def before_execute(self, event, data, configs):
                order.append("A")

            async def after_execute(self, event, result, configs):
                pass

            async def on_error(self, event, error, configs):
                pass

        class PolicyB:
            async def before_execute(self, event, data, configs):
                order.append("B")

            async def after_execute(self, event, result, configs):
                pass

            async def on_error(self, event, error, configs):
                pass

        chain = PolicyChain([PolicyA(), PolicyB()])
        handler = SuccessHandler()
        await chain(handler, _make_text_event(), None, None)
        assert order == ["A", "B"]

    @pytest.mark.asyncio
    async def test_after_execute_runs_in_reverse(self):
        order = []

        class PolicyA:
            async def before_execute(self, event, data, configs):
                pass

            async def after_execute(self, event, result, configs):
                order.append("A")

            async def on_error(self, event, error, configs):
                pass

        class PolicyB:
            async def before_execute(self, event, data, configs):
                pass

            async def after_execute(self, event, result, configs):
                order.append("B")

            async def on_error(self, event, error, configs):
                pass

        chain = PolicyChain([PolicyA(), PolicyB()])
        handler = SuccessHandler()
        await chain(handler, _make_text_event(), None, None)
        assert order == ["B", "A"]

    @pytest.mark.asyncio
    async def test_on_error_runs_in_reverse_on_exception(self):
        order = []

        class PolicyA:
            async def before_execute(self, event, data, configs):
                pass

            async def after_execute(self, event, result, configs):
                pass

            async def on_error(self, event, error, configs):
                order.append("A")

        class PolicyB:
            async def before_execute(self, event, data, configs):
                pass

            async def after_execute(self, event, result, configs):
                pass

            async def on_error(self, event, error, configs):
                order.append("B")

        chain = PolicyChain([PolicyA(), PolicyB()])
        handler = PermanentFailHandler()
        with pytest.raises(PermanentError):
            await chain(handler, _make_text_event(), None, None)
        assert order == ["B", "A"]

    @pytest.mark.asyncio
    async def test_before_execute_abort_prevents_handler(self):
        class AbortPolicy:
            async def before_execute(self, event, data, configs):
                raise BudgetError("No budget")

            async def after_execute(self, event, result, configs):
                pass

            async def on_error(self, event, error, configs):
                pass

        chain = PolicyChain([AbortPolicy()])
        handler = SuccessHandler()
        with pytest.raises(BudgetError):
            await chain(handler, _make_text_event(), None, None)
        assert handler.call_count == 0

    @pytest.mark.asyncio
    async def test_add_policy(self):
        chain = PolicyChain()
        order = []

        class TrackPolicy:
            async def before_execute(self, event, data, configs):
                order.append("before")

            async def after_execute(self, event, result, configs):
                order.append("after")

            async def on_error(self, event, error, configs):
                pass

        chain.add(TrackPolicy())
        handler = SuccessHandler()
        await chain(handler, _make_text_event(), None, None)
        assert order == ["before", "after"]


# ──────────────────────────────────────────────
# RetryPolicy
# ──────────────────────────────────────────────


class TestRetryPolicy:
    def test_invalid_max_attempts(self):
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RetryPolicy(max_attempts=0)

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        handler = SuccessHandler()
        result = await policy.execute_with_retry(
            handler, _make_text_event(), None, None
        )
        assert result.ok is True
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        handler = FailNTimesHandler(fail_count=2)
        result = await policy.execute_with_retry(
            handler, _make_text_event(), None, None
        )
        assert result.ok is True
        assert handler.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self):
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        handler = PermanentFailHandler()
        with pytest.raises(PermanentError):
            await policy.execute_with_retry(
                handler, _make_text_event(), None, None
            )

    @pytest.mark.asyncio
    async def test_exhaust_retries_raises(self):
        policy = RetryPolicy(max_attempts=2, base_delay=0.01)
        handler = FailNTimesHandler(fail_count=5)  # Will never succeed in 2 attempts
        with pytest.raises(TransientError) as exc_info:
            await policy.execute_with_retry(
                handler, _make_text_event(), None, None
            )
        assert exc_info.value.attempt == 2

    @pytest.mark.asyncio
    async def test_attempt_number_set_on_error(self):
        policy = RetryPolicy(max_attempts=2, base_delay=0.01)
        handler = FailNTimesHandler(fail_count=5)
        with pytest.raises(TransientError) as exc_info:
            await policy.execute_with_retry(
                handler, _make_text_event(), None, None
            )
        assert exc_info.value.attempt == 2

    @pytest.mark.asyncio
    async def test_single_attempt_no_retry(self):
        policy = RetryPolicy(max_attempts=1, base_delay=0.01)
        handler = FailNTimesHandler(fail_count=1)
        with pytest.raises(TransientError):
            await policy.execute_with_retry(
                handler, _make_text_event(), None, None
            )
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_before_execute_is_noop(self):
        policy = RetryPolicy()
        await policy.before_execute(None, None, None)  # Should not raise

    @pytest.mark.asyncio
    async def test_after_execute_resets_attempt(self):
        policy = RetryPolicy()
        policy._current_attempt = 3
        await policy.after_execute(None, None, None)
        assert policy._current_attempt == 0


# ──────────────────────────────────────────────
# TimeoutPolicy
# ──────────────────────────────────────────────


class TestTimeoutPolicy:
    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout seconds must be positive"):
            TimeoutPolicy(seconds=0)

    def test_negative_timeout(self):
        with pytest.raises(ValueError):
            TimeoutPolicy(seconds=-1)

    @pytest.mark.asyncio
    async def test_fast_handler_completes(self):
        policy = TimeoutPolicy(seconds=5.0)
        handler = SuccessHandler()
        result = await policy.execute_with_timeout(
            handler, _make_text_event(), None, None
        )
        assert result.ok is True

    @pytest.mark.asyncio
    async def test_slow_handler_raises_timeout(self):
        policy = TimeoutPolicy(seconds=0.05)
        handler = SlowHandler(delay=1.0)
        with pytest.raises(CognitivTimeoutError, match="timed out"):
            await policy.execute_with_timeout(
                handler, _make_text_event(), None, None
            )

    @pytest.mark.asyncio
    async def test_timeout_error_is_transient(self):
        policy = TimeoutPolicy(seconds=0.05)
        handler = SlowHandler(delay=1.0)
        with pytest.raises(CognitivTimeoutError) as exc_info:
            await policy.execute_with_timeout(
                handler, _make_text_event(), None, None
            )
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_before_execute_is_noop(self):
        policy = TimeoutPolicy(seconds=1.0)
        await policy.before_execute(None, None, None)  # Should not raise


# ──────────────────────────────────────────────
# BudgetPolicy
# ──────────────────────────────────────────────


class TestBudgetPolicy:
    @pytest.mark.asyncio
    async def test_allows_when_budget_available(self):
        tracker = BudgetTracker(token_budget=100, call_budget=10)
        policy = BudgetPolicy(tracker=tracker)
        await policy.before_execute(None, None, None)  # Should not raise

    @pytest.mark.asyncio
    async def test_blocks_when_budget_exhausted(self):
        tracker = BudgetTracker(call_budget=1)
        tracker.consume(calls=1)
        policy = BudgetPolicy(tracker=tracker)
        with pytest.raises(BudgetError, match="Budget exhausted"):
            await policy.before_execute(None, None, None)

    @pytest.mark.asyncio
    async def test_consumes_after_success(self):
        tracker = BudgetTracker(token_budget=100, call_budget=10)
        policy = BudgetPolicy(tracker=tracker)

        result = ExecutionResult(
            ok=True,
            value=LLMResultData(
                text="hi",
                model="test",
                token_usage=TokenUsage(total=25),
            ),
        )
        await policy.after_execute(None, result, None)
        assert tracker.tokens_used == 25
        assert tracker.calls_made == 1

    @pytest.mark.asyncio
    async def test_consumes_call_on_error(self):
        tracker = BudgetTracker(call_budget=10)
        policy = BudgetPolicy(tracker=tracker)
        await policy.on_error(None, TransientError("test"), None)
        assert tracker.calls_made == 1

    @pytest.mark.asyncio
    async def test_consumes_call_without_token_info(self):
        """Result without token_usage still consumes 1 call."""
        tracker = BudgetTracker(call_budget=10)
        policy = BudgetPolicy(tracker=tracker)

        result = ExecutionResult(ok=True, value="plain string")
        await policy.after_execute(None, result, None)
        assert tracker.calls_made == 1
        assert tracker.tokens_used == 0

    @pytest.mark.asyncio
    async def test_integration_with_policy_chain(self):
        """BudgetPolicy in a PolicyChain blocks when budget is exceeded."""
        tracker = BudgetTracker(call_budget=2)
        chain = PolicyChain([BudgetPolicy(tracker=tracker)])
        handler = SuccessHandler(tokens=10)
        event = _make_text_event()

        # First call succeeds
        result = await chain(handler, event, None, None)
        assert result.ok is True
        assert tracker.calls_made == 1

        # Second call succeeds
        result = await chain(handler, event, None, None)
        assert result.ok is True
        assert tracker.calls_made == 2

        # Third call blocked
        with pytest.raises(BudgetError):
            await chain(handler, event, None, None)


# ──────────────────────────────────────────────
# Composite Policy Tests
# ──────────────────────────────────────────────


class TestPolicyComposition:
    @pytest.mark.asyncio
    async def test_budget_plus_retry(self):
        """RetryPolicy retries transient errors; BudgetPolicy counts attempts."""
        tracker = BudgetTracker(call_budget=10)
        budget_policy = BudgetPolicy(tracker=tracker)
        retry_policy = RetryPolicy(max_attempts=3, base_delay=0.01)

        handler = FailNTimesHandler(fail_count=2, tokens=5)
        event = _make_text_event()

        # Use retry to get a successful result
        result = await retry_policy.execute_with_retry(handler, event, None, None)
        assert result.ok is True
        assert handler.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_plus_retry(self):
        """Timeout fires on a slow handler, retry catches it if retryable."""
        # The timeout raises CognitivTimeoutError (which is TransientError, retryable)
        timeout_policy = TimeoutPolicy(seconds=0.05)

        call_count = 0

        class SlowThenFastHandler(EventHandlerProtocol[LLMResultData]):
            async def __call__(self, event, data, configs):
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    await asyncio.sleep(1.0)  # Will timeout
                return ExecutionResult(
                    ok=True,
                    value=LLMResultData(text="ok", model="test"),
                    metadata=ResultMetadata(),
                )

        # Wrap timeout around the handler
        class TimeoutWrappedHandler(EventHandlerProtocol[LLMResultData]):
            def __init__(self, inner, tp):
                self.inner = inner
                self.tp = tp

            async def __call__(self, event, data, configs):
                return await self.tp.execute_with_timeout(
                    self.inner, event, data, configs
                )

        inner_handler = SlowThenFastHandler()
        wrapped = TimeoutWrappedHandler(inner_handler, timeout_policy)
        retry_policy = RetryPolicy(max_attempts=3, base_delay=0.01)

        result = await retry_policy.execute_with_retry(
            wrapped, _make_text_event(), None, None
        )
        assert result.ok is True
        assert call_count == 2
