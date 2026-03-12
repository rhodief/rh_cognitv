"""
BudgetTracker — First-class standalone resource for budget management.

DI-L3-03: Standalone component with self-contained logic.
Policies query can_proceed() before execution and call consume() after.
Handlers never touch the tracker directly.
"""

from __future__ import annotations

import time

from .errors import BudgetError
from .models import BudgetSnapshot
from .protocols import BudgetTrackerProtocol


class BudgetTracker(BudgetTrackerProtocol):
    """Tracks token, call, and time budgets.

    Args:
        token_budget: Maximum tokens allowed. 0 means unlimited.
        call_budget: Maximum calls allowed. 0 means unlimited.
        time_budget_seconds: Maximum wall-clock seconds. 0.0 means unlimited.
    """

    def __init__(
        self,
        *,
        token_budget: int = 0,
        call_budget: int = 0,
        time_budget_seconds: float = 0.0,
    ) -> None:
        self.token_budget = token_budget
        self.call_budget = call_budget
        self.time_budget_seconds = time_budget_seconds

        self.tokens_used: int = 0
        self.calls_made: int = 0
        self._start_time: float = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds since tracker creation."""
        return time.monotonic() - self._start_time

    def can_proceed(self) -> bool:
        """Check if there is remaining budget to continue.

        Returns True if all enabled budgets have remaining capacity.
        A budget of 0 means unlimited for that dimension.
        """
        if self.token_budget > 0 and self.tokens_used >= self.token_budget:
            return False
        if self.call_budget > 0 and self.calls_made >= self.call_budget:
            return False
        if self.time_budget_seconds > 0.0 and self.elapsed_seconds >= self.time_budget_seconds:
            return False
        return True

    def consume(self, *, tokens: int = 0, calls: int = 0) -> None:
        """Record consumption of budget resources.

        Args:
            tokens: Number of tokens consumed.
            calls: Number of calls made (typically 1).

        Raises:
            BudgetError: If consuming would exceed the budget.
        """
        if tokens < 0 or calls < 0:
            raise ValueError("Cannot consume negative resources")

        new_tokens = self.tokens_used + tokens
        new_calls = self.calls_made + calls

        if self.token_budget > 0 and new_tokens > self.token_budget:
            raise BudgetError(
                f"Token budget exceeded: {new_tokens}/{self.token_budget}"
            )
        if self.call_budget > 0 and new_calls > self.call_budget:
            raise BudgetError(
                f"Call budget exceeded: {new_calls}/{self.call_budget}"
            )
        if self.time_budget_seconds > 0.0 and self.elapsed_seconds >= self.time_budget_seconds:
            raise BudgetError(
                f"Time budget exceeded: {self.elapsed_seconds:.1f}s/{self.time_budget_seconds:.1f}s"
            )

        self.tokens_used = new_tokens
        self.calls_made = new_calls

    def remaining(self) -> BudgetSnapshot:
        """Get a snapshot of remaining budget."""
        tokens_remaining = (
            self.token_budget - self.tokens_used if self.token_budget > 0 else -1
        )
        calls_remaining = (
            self.call_budget - self.calls_made if self.call_budget > 0 else -1
        )
        time_remaining = (
            max(0.0, self.time_budget_seconds - self.elapsed_seconds)
            if self.time_budget_seconds > 0.0
            else -1.0
        )
        return BudgetSnapshot(
            tokens_remaining=tokens_remaining,
            calls_remaining=calls_remaining,
            time_remaining_seconds=time_remaining,
        )

    def is_exceeded(self) -> bool:
        """Check if any budget dimension is exceeded."""
        return not self.can_proceed()
