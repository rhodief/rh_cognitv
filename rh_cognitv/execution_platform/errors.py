"""
Error hierarchy for the Execution Platform.

Categorized exceptions with recoverability traits.
The `retryable` flag and `category` drive PolicyChain decisions.
"""

from __future__ import annotations

from enum import Enum


class ErrorCategory(str, Enum):
    """Classification of error types for policy decisions."""

    TRANSIENT = "transient"  # network blip, rate limit
    PERMANENT = "permanent"  # invalid input, auth failure
    BUDGET = "budget"  # token/call/time budget exceeded
    INTERRUPT = "interrupt"  # user cancellation, human-in-the-loop
    ESCALATION = "escalation"  # needs human decision


class CognitivError(Exception):
    """Base error for all execution platform exceptions.

    Attributes:
        retryable: Whether the operation can be retried.
        category: Classification for policy decisions.
        attempt: Which retry attempt produced this error.
        original: The wrapped root cause exception, if any.
    """

    def __init__(
        self,
        message: str = "",
        *,
        retryable: bool = False,
        category: ErrorCategory = ErrorCategory.PERMANENT,
        attempt: int = 0,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.category = category
        self.attempt = attempt
        self.original = original


class TransientError(CognitivError):
    """Transient error — retryable by default."""

    def __init__(
        self,
        message: str = "",
        *,
        attempt: int = 0,
        original: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            retryable=True,
            category=ErrorCategory.TRANSIENT,
            attempt=attempt,
            original=original,
        )


class PermanentError(CognitivError):
    """Permanent error — not retryable by default."""

    def __init__(
        self,
        message: str = "",
        *,
        attempt: int = 0,
        original: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            category=ErrorCategory.PERMANENT,
            attempt=attempt,
            original=original,
        )


class BudgetError(PermanentError):
    """Budget exceeded — token, call, or time limit hit."""

    def __init__(
        self,
        message: str = "Budget exceeded",
        *,
        attempt: int = 0,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message, attempt=attempt, original=original)
        self.category = ErrorCategory.BUDGET


class InterruptError(PermanentError):
    """User cancellation or human-in-the-loop interrupt."""

    def __init__(
        self,
        message: str = "Execution interrupted",
        *,
        attempt: int = 0,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message, attempt=attempt, original=original)
        self.category = ErrorCategory.INTERRUPT


class EscalationError(CognitivError):
    """Needs human decision — task is paused waiting for input."""

    def __init__(
        self,
        message: str = "Escalation required",
        *,
        attempt: int = 0,
        original: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            category=ErrorCategory.ESCALATION,
            attempt=attempt,
            original=original,
        )


class LLMTransientError(TransientError):
    """Transient LLM error — rate limit, network blip, server error."""

    pass


class TimeoutError(TransientError):
    """Operation timed out — retryable."""

    def __init__(
        self,
        message: str = "Operation timed out",
        *,
        attempt: int = 0,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message, attempt=attempt, original=original)


class ValidationError(PermanentError):
    """Input validation failed — not retryable."""

    def __init__(
        self,
        message: str = "Validation failed",
        *,
        attempt: int = 0,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message, attempt=attempt, original=original)
