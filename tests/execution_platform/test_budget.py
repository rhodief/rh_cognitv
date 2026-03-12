"""Tests for budget.py — BudgetTracker consumption, limits, and snapshots."""

import time

import pytest

from rh_cognitv.execution_platform.budget import BudgetTracker
from rh_cognitv.execution_platform.errors import BudgetError
from rh_cognitv.execution_platform.protocols import BudgetTrackerProtocol


# ──────────────────────────────────────────────
# Protocol compliance
# ──────────────────────────────────────────────


class TestBudgetTrackerProtocol:
    def test_implements_protocol(self):
        tracker = BudgetTracker(token_budget=100)
        assert isinstance(tracker, BudgetTrackerProtocol)


# ──────────────────────────────────────────────
# Token budget
# ──────────────────────────────────────────────


class TestTokenBudget:
    def test_can_proceed_within_budget(self):
        t = BudgetTracker(token_budget=1000)
        assert t.can_proceed() is True

    def test_can_proceed_after_consumption(self):
        t = BudgetTracker(token_budget=1000)
        t.consume(tokens=500)
        assert t.can_proceed() is True

    def test_cannot_proceed_at_limit(self):
        t = BudgetTracker(token_budget=100)
        t.consume(tokens=100)
        assert t.can_proceed() is False

    def test_cannot_proceed_over_limit(self):
        t = BudgetTracker(token_budget=100)
        t.consume(tokens=50)
        with pytest.raises(BudgetError, match="Token budget exceeded"):
            t.consume(tokens=60)

    def test_consume_exact_limit(self):
        t = BudgetTracker(token_budget=100)
        t.consume(tokens=100)  # exact limit should succeed
        assert t.tokens_used == 100

    def test_remaining_tokens(self):
        t = BudgetTracker(token_budget=1000)
        t.consume(tokens=300)
        snap = t.remaining()
        assert snap.tokens_remaining == 700

    def test_unlimited_tokens(self):
        t = BudgetTracker(token_budget=0)
        t.consume(tokens=999999)
        assert t.can_proceed() is True
        snap = t.remaining()
        assert snap.tokens_remaining == -1  # unlimited sentinel


# ──────────────────────────────────────────────
# Call budget
# ──────────────────────────────────────────────


class TestCallBudget:
    def test_can_proceed_within_budget(self):
        t = BudgetTracker(call_budget=10)
        assert t.can_proceed() is True

    def test_cannot_proceed_at_limit(self):
        t = BudgetTracker(call_budget=3)
        t.consume(calls=3)
        assert t.can_proceed() is False

    def test_consume_over_limit_raises(self):
        t = BudgetTracker(call_budget=2)
        t.consume(calls=2)
        with pytest.raises(BudgetError, match="Call budget exceeded"):
            t.consume(calls=1)

    def test_remaining_calls(self):
        t = BudgetTracker(call_budget=10)
        t.consume(calls=4)
        snap = t.remaining()
        assert snap.calls_remaining == 6

    def test_unlimited_calls(self):
        t = BudgetTracker(call_budget=0)
        t.consume(calls=999)
        assert t.can_proceed() is True
        snap = t.remaining()
        assert snap.calls_remaining == -1


# ──────────────────────────────────────────────
# Time budget
# ──────────────────────────────────────────────


class TestTimeBudget:
    def test_can_proceed_within_time(self):
        t = BudgetTracker(time_budget_seconds=10.0)
        assert t.can_proceed() is True

    def test_time_budget_exceeded(self):
        t = BudgetTracker(time_budget_seconds=0.05)
        time.sleep(0.06)
        assert t.can_proceed() is False
        assert t.is_exceeded() is True

    def test_remaining_time(self):
        t = BudgetTracker(time_budget_seconds=10.0)
        snap = t.remaining()
        assert snap.time_remaining_seconds > 9.0
        assert snap.time_remaining_seconds <= 10.0

    def test_unlimited_time(self):
        t = BudgetTracker(time_budget_seconds=0.0)
        snap = t.remaining()
        assert snap.time_remaining_seconds == -1.0
        assert t.can_proceed() is True

    def test_consume_raises_when_time_exceeded(self):
        t = BudgetTracker(time_budget_seconds=0.02)
        time.sleep(0.03)
        with pytest.raises(BudgetError, match="Time budget exceeded"):
            t.consume(tokens=1)

    def test_elapsed_seconds(self):
        t = BudgetTracker()
        time.sleep(0.05)
        assert t.elapsed_seconds >= 0.04


# ──────────────────────────────────────────────
# Combined budgets
# ──────────────────────────────────────────────


class TestCombinedBudgets:
    def test_all_within_limits(self):
        t = BudgetTracker(token_budget=1000, call_budget=10, time_budget_seconds=60.0)
        t.consume(tokens=100, calls=2)
        assert t.can_proceed() is True

    def test_token_exceeded_blocks(self):
        t = BudgetTracker(token_budget=100, call_budget=10)
        t.consume(tokens=100)
        assert t.can_proceed() is False

    def test_call_exceeded_blocks(self):
        t = BudgetTracker(token_budget=1000, call_budget=2)
        t.consume(calls=2)
        assert t.can_proceed() is False

    def test_remaining_snapshot_combined(self):
        t = BudgetTracker(token_budget=1000, call_budget=10, time_budget_seconds=60.0)
        t.consume(tokens=250, calls=3)
        snap = t.remaining()
        assert snap.tokens_remaining == 750
        assert snap.calls_remaining == 7
        assert snap.time_remaining_seconds > 50.0

    def test_is_exceeded_mirrors_can_proceed(self):
        t = BudgetTracker(token_budget=100)
        assert t.is_exceeded() is False
        t.consume(tokens=100)
        assert t.is_exceeded() is True


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


class TestEdgeCases:
    def test_consume_zero(self):
        t = BudgetTracker(token_budget=100)
        t.consume(tokens=0, calls=0)
        assert t.tokens_used == 0
        assert t.calls_made == 0

    def test_consume_negative_raises(self):
        t = BudgetTracker(token_budget=100)
        with pytest.raises(ValueError, match="Cannot consume negative"):
            t.consume(tokens=-1)

    def test_consume_negative_calls_raises(self):
        t = BudgetTracker(call_budget=10)
        with pytest.raises(ValueError, match="Cannot consume negative"):
            t.consume(calls=-1)

    def test_no_budgets_set(self):
        """Tracker with no limits should always allow proceeding."""
        t = BudgetTracker()
        t.consume(tokens=1000000, calls=1000000)
        assert t.can_proceed() is True
        assert t.is_exceeded() is False

    def test_incremental_consumption(self):
        t = BudgetTracker(token_budget=100, call_budget=5)
        t.consume(tokens=30, calls=1)
        t.consume(tokens=30, calls=1)
        t.consume(tokens=30, calls=1)
        assert t.tokens_used == 90
        assert t.calls_made == 3
        assert t.can_proceed() is True

        t.consume(tokens=10, calls=2)
        assert t.tokens_used == 100
        assert t.calls_made == 5
        assert t.can_proceed() is False
