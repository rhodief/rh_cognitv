"""
PolicyChain and concrete policies for handler execution.

DD-L3-07: Policies are chainable middleware that wrap handler execution with
before_execute / after_execute / on_error hooks.

Policies:
  - RetryPolicy: retries transient errors with exponential backoff
  - TimeoutPolicy: enforces wall-clock time limits
  - BudgetPolicy: checks/consumes budget via BudgetTracker
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .errors import (
    BudgetError,
    CognitivError,
    ErrorCategory,
    PermanentError,
    TransientError,
)
from .models import ExecutionResult, ResultMetadata
from .protocols import (
    BudgetTrackerProtocol,
    EventHandlerProtocol,
    PolicyChainProtocol,
    PolicyProtocol,
)
from .types import now_timestamp


# ──────────────────────────────────────────────
# PolicyChain
# ──────────────────────────────────────────────


class PolicyChain(PolicyChainProtocol):
    """Composable chain of policies wrapping handler execution.

    Usage:
        chain = PolicyChain([
            BudgetPolicy(tracker=budget_tracker),
            TimeoutPolicy(seconds=30),
            RetryPolicy(max_attempts=3),
        ])
        result = await chain.execute(handler, event, data, configs)

    Policies run in order for before_execute (first → last),
    reverse order for after_execute (last → first),
    and reverse order for on_error (last → first).
    """

    def __init__(self, policies: list[PolicyProtocol] | None = None) -> None:
        self._policies: list[PolicyProtocol] = list(policies) if policies else []

    def add(self, policy: PolicyProtocol) -> None:
        """Append a policy to the chain."""
        self._policies.append(policy)

    async def execute(
        self,
        handler: EventHandlerProtocol[Any],
        event: Any,
        data: Any,
        configs: Any,
    ) -> ExecutionResult[Any]:
        """Run the handler wrapped by all policies in the chain."""
        # before_execute: first → last
        for policy in self._policies:
            await policy.before_execute(event, data, configs)

        try:
            result = await handler.execute(event, data, configs)
        except Exception as exc:
            # on_error: reverse order
            for policy in reversed(self._policies):
                await policy.on_error(event, exc, configs)
            raise

        # after_execute: reverse order
        for policy in reversed(self._policies):
            await policy.after_execute(event, result, configs)

        return result


# ──────────────────────────────────────────────
# RetryPolicy
# ──────────────────────────────────────────────


class RetryPolicy(PolicyProtocol):
    """Retries handler execution on transient errors with exponential backoff.

    Only retries errors where `retryable is True`. Non-retryable errors
    propagate immediately.

    Args:
        max_attempts: Total attempts (1 = no retry). Must be >= 1.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay between retries (caps exponential growth).
        multiplier: Backoff multiplier applied each retry.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 30.0,
        multiplier: float = 2.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self._current_attempt: int = 0

    async def before_execute(self, event: Any, data: Any, configs: Any) -> None:
        """No-op — retry logic is handled in execute_with_retry."""

    async def after_execute(
        self, event: Any, result: ExecutionResult[Any], configs: Any
    ) -> None:
        """Reset attempt counter on success."""
        self._current_attempt = 0

    async def on_error(self, event: Any, error: Exception, configs: Any) -> None:
        """No-op — retry decisions are made in execute_with_retry."""

    async def execute_with_retry(
        self,
        handler: EventHandlerProtocol[Any],
        event: Any,
        data: Any,
        configs: Any,
    ) -> ExecutionResult[Any]:
        """Execute the handler with retry logic.

        This is the primary method — used by PolicyChain when RetryPolicy
        needs to wrap execution directly.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            self._current_attempt = attempt
            try:
                result = await handler.execute(event, data, configs)
                self._current_attempt = 0
                return result
            except CognitivError as exc:
                exc.attempt = attempt
                last_error = exc
                if not exc.retryable or attempt >= self.max_attempts:
                    raise
                delay = min(
                    self.base_delay * (self.multiplier ** (attempt - 1)),
                    self.max_delay,
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                last_error = exc
                raise

        # Should not reach here, but safety net
        raise last_error  # type: ignore[misc]


# ──────────────────────────────────────────────
# TimeoutPolicy
# ──────────────────────────────────────────────


class TimeoutPolicy(PolicyProtocol):
    """Enforces a wall-clock time limit on handler execution.

    Args:
        seconds: Maximum seconds allowed for execution.
    """

    def __init__(self, *, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("timeout seconds must be positive")
        self.seconds = seconds

    async def before_execute(self, event: Any, data: Any, configs: Any) -> None:
        """No-op — timeout is enforced around the handler call."""

    async def after_execute(
        self, event: Any, result: ExecutionResult[Any], configs: Any
    ) -> None:
        """No-op."""

    async def on_error(self, event: Any, error: Exception, configs: Any) -> None:
        """No-op."""

    async def execute_with_timeout(
        self,
        handler: EventHandlerProtocol[Any],
        event: Any,
        data: Any,
        configs: Any,
    ) -> ExecutionResult[Any]:
        """Execute handler with a timeout.

        Raises:
            TimeoutError: If execution exceeds the time limit.
        """
        from .errors import TimeoutError as CognitivTimeoutError

        try:
            return await asyncio.wait_for(
                handler.execute(event, data, configs),
                timeout=self.seconds,
            )
        except asyncio.TimeoutError:
            raise CognitivTimeoutError(
                f"Execution timed out after {self.seconds}s"
            )


# ──────────────────────────────────────────────
# BudgetPolicy
# ──────────────────────────────────────────────


class BudgetPolicy(PolicyProtocol):
    """Checks budget before execution and consumes after.

    Queries BudgetTracker.can_proceed() in before_execute.
    Calls BudgetTracker.consume() in after_execute with token/call data
    from the result metadata.

    Args:
        tracker: The BudgetTracker instance to query/update.
    """

    def __init__(self, *, tracker: BudgetTrackerProtocol) -> None:
        self.tracker = tracker

    async def before_execute(self, event: Any, data: Any, configs: Any) -> None:
        """Check budget before handler runs. Raises BudgetError if exhausted."""
        if not self.tracker.can_proceed():
            raise BudgetError("Budget exhausted before execution")

    async def after_execute(
        self, event: Any, result: ExecutionResult[Any], configs: Any
    ) -> None:
        """Consume budget based on result. Consumes 1 call per execution.

        If the result value has token_usage info, also consume tokens.
        """
        tokens = 0
        if result.value is not None and hasattr(result.value, "token_usage"):
            tokens = result.value.token_usage.total
        self.tracker.consume(tokens=tokens, calls=1)

    async def on_error(self, event: Any, error: Exception, configs: Any) -> None:
        """On error, still consume 1 call (the attempt happened)."""
        self.tracker.consume(calls=1)
