"""Tests for errors.py — Error hierarchy, isinstance checks, retryable flags."""

import pytest

from rh_cognitv.execution_platform.errors import (
    BudgetError,
    CognitivError,
    ErrorCategory,
    EscalationError,
    InterruptError,
    LLMTransientError,
    PermanentError,
    TimeoutError,
    TransientError,
    ValidationError,
)


# ──────────────────────────────────────────────
# Hierarchy (isinstance checks)
# ──────────────────────────────────────────────


class TestErrorHierarchy:
    def test_transient_is_cognitiv(self):
        assert issubclass(TransientError, CognitivError)

    def test_permanent_is_cognitiv(self):
        assert issubclass(PermanentError, CognitivError)

    def test_budget_is_permanent(self):
        assert issubclass(BudgetError, PermanentError)
        assert issubclass(BudgetError, CognitivError)

    def test_interrupt_is_permanent(self):
        assert issubclass(InterruptError, PermanentError)

    def test_escalation_is_cognitiv(self):
        assert issubclass(EscalationError, CognitivError)
        # EscalationError is NOT a PermanentError — it's its own branch
        assert not issubclass(EscalationError, PermanentError)

    def test_llm_transient_is_transient(self):
        assert issubclass(LLMTransientError, TransientError)
        assert issubclass(LLMTransientError, CognitivError)

    def test_timeout_is_transient(self):
        assert issubclass(TimeoutError, TransientError)

    def test_validation_is_permanent(self):
        assert issubclass(ValidationError, PermanentError)


# ──────────────────────────────────────────────
# Retryable flag
# ──────────────────────────────────────────────


class TestRetryable:
    def test_transient_is_retryable(self):
        e = TransientError("network blip")
        assert e.retryable is True

    def test_permanent_is_not_retryable(self):
        e = PermanentError("bad input")
        assert e.retryable is False

    def test_budget_is_not_retryable(self):
        e = BudgetError()
        assert e.retryable is False

    def test_interrupt_is_not_retryable(self):
        e = InterruptError()
        assert e.retryable is False

    def test_escalation_is_not_retryable(self):
        e = EscalationError()
        assert e.retryable is False

    def test_llm_transient_is_retryable(self):
        e = LLMTransientError("rate limited")
        assert e.retryable is True

    def test_timeout_is_retryable(self):
        e = TimeoutError()
        assert e.retryable is True

    def test_validation_is_not_retryable(self):
        e = ValidationError("invalid schema")
        assert e.retryable is False


# ──────────────────────────────────────────────
# Category
# ──────────────────────────────────────────────


class TestErrorCategory:
    def test_transient_category(self):
        e = TransientError()
        assert e.category == ErrorCategory.TRANSIENT

    def test_permanent_category(self):
        e = PermanentError()
        assert e.category == ErrorCategory.PERMANENT

    def test_budget_category(self):
        e = BudgetError()
        assert e.category == ErrorCategory.BUDGET

    def test_interrupt_category(self):
        e = InterruptError()
        assert e.category == ErrorCategory.INTERRUPT

    def test_escalation_category(self):
        e = EscalationError()
        assert e.category == ErrorCategory.ESCALATION

    def test_llm_transient_category(self):
        e = LLMTransientError()
        assert e.category == ErrorCategory.TRANSIENT

    def test_timeout_category(self):
        e = TimeoutError()
        assert e.category == ErrorCategory.TRANSIENT

    def test_validation_category(self):
        e = ValidationError()
        assert e.category == ErrorCategory.PERMANENT


# ──────────────────────────────────────────────
# Attributes
# ──────────────────────────────────────────────


class TestErrorAttributes:
    def test_message(self):
        e = CognitivError("something broke")
        assert str(e) == "something broke"

    def test_attempt_default(self):
        e = CognitivError("test")
        assert e.attempt == 0

    def test_attempt_custom(self):
        e = TransientError("retry", attempt=3)
        assert e.attempt == 3

    def test_original_none(self):
        e = PermanentError("fail")
        assert e.original is None

    def test_original_wraps(self):
        cause = ValueError("root cause")
        e = TransientError("wrapper", original=cause)
        assert e.original is cause

    def test_default_messages(self):
        assert str(BudgetError()) == "Budget exceeded"
        assert str(InterruptError()) == "Execution interrupted"
        assert str(EscalationError()) == "Escalation required"
        assert str(TimeoutError()) == "Operation timed out"
        assert str(ValidationError()) == "Validation failed"


# ──────────────────────────────────────────────
# Exception catching patterns
# ──────────────────────────────────────────────


class TestExceptionCatching:
    def test_catch_all_cognitiv(self):
        """All custom errors should be catchable via CognitivError."""
        errors = [
            TransientError(),
            PermanentError(),
            BudgetError(),
            InterruptError(),
            EscalationError(),
            LLMTransientError(),
            TimeoutError(),
            ValidationError(),
        ]
        for e in errors:
            with pytest.raises(CognitivError):
                raise e

    def test_catch_transient_only(self):
        with pytest.raises(TransientError):
            raise LLMTransientError("rate limit")

    def test_catch_permanent_only(self):
        with pytest.raises(PermanentError):
            raise BudgetError()

    def test_budget_not_caught_by_transient(self):
        with pytest.raises(BudgetError):
            try:
                raise BudgetError()
            except TransientError:
                pytest.fail("BudgetError should not be caught as TransientError")
