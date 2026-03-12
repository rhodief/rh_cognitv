"""
Protocol definitions (ABCs) for the Execution Platform.

These are the contracts that upper layers depend on.
All concrete implementations live in their own modules.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from .models import (
    Artifact,
    BudgetSnapshot,
    EventKind,
    EventStatus,
    ExecutionResult,
    Memory,
    MemoryQuery,
    QueryResult,
)
from .types import ID, Timestamp

T = TypeVar("T")


# ──────────────────────────────────────────────
# Snapshot Serializer Protocol
# ──────────────────────────────────────────────


class SnapshotSerializerProtocol(ABC):
    """Source of truth for snapshot serialization. Implement to add formats."""

    @abstractmethod
    def serialize(self, state: dict) -> bytes:
        """Serialize a state dict to bytes."""
        ...

    @abstractmethod
    def deserialize(self, data: bytes) -> dict:
        """Deserialize bytes back to a state dict."""
        ...


class JsonSnapshotSerializer(SnapshotSerializerProtocol):
    """Default JSON implementation of snapshot serialization."""

    def serialize(self, state: dict) -> bytes:
        return json.dumps(state, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict:
        return json.loads(data)


# ──────────────────────────────────────────────
# EventBus Protocol
# ──────────────────────────────────────────────


class EventBusProtocol(ABC):
    """Hybrid sync middleware + async subscriber event bus."""

    @abstractmethod
    def use(self, middleware: MiddlewareProtocol) -> None:
        """Register a synchronous middleware in the pipeline."""
        ...

    @abstractmethod
    def on(self, event_type: type, handler: Any) -> None:
        """Register a synchronous handler for an event type."""
        ...

    @abstractmethod
    def on_async(self, event_type: type, handler: Any) -> None:
        """Register an async subscriber for an event type."""
        ...

    @abstractmethod
    async def emit(self, event: Any) -> None:
        """Emit an event: run sync middleware, then fan out to async subscribers."""
        ...

    @abstractmethod
    async def wait_for(
        self, event_type: type, *, filter: Any | None = None, timeout: float | None = None
    ) -> Any:
        """Block until an event of the given type (matching optional filter) is emitted."""
        ...


class MiddlewareProtocol(ABC):
    """Synchronous middleware that runs in the EventBus pipeline."""

    @abstractmethod
    def handle(self, event: Any, next_fn: Any) -> Any:
        """Process the event, optionally calling next_fn to continue the chain."""
        ...


# ──────────────────────────────────────────────
# Event Handler Protocol
# ──────────────────────────────────────────────


class EventHandlerProtocol(ABC, Generic[T]):
    """Handler for a specific kind of execution event."""

    @abstractmethod
    async def __call__(self, event: Any, data: Any, configs: Any) -> ExecutionResult[T]:
        """Execute the event and return a typed result."""
        ...


# ──────────────────────────────────────────────
# Handler Registry Protocol
# ──────────────────────────────────────────────


class HandlerRegistryProtocol(ABC):
    """Registry that maps event kinds to handlers."""

    @abstractmethod
    def register(self, kind: EventKind, handler: EventHandlerProtocol[Any]) -> None:
        """Register a handler for a specific event kind."""
        ...

    @abstractmethod
    async def handle(self, event: Any, data: Any, configs: Any) -> ExecutionResult[Any]:
        """Dispatch to the registered handler for the event's kind."""
        ...


# ──────────────────────────────────────────────
# Execution State Protocol
# ──────────────────────────────────────────────


class ExecutionStateProtocol(ABC):
    """Immutable snapshot chain for time-travel state management."""

    @abstractmethod
    def snapshot(self) -> int:
        """Take a snapshot of the current state. Returns the snapshot version."""
        ...

    @abstractmethod
    def restore(self, version: int) -> None:
        """Restore state to a specific snapshot version."""
        ...

    @abstractmethod
    def get_current(self) -> dict:
        """Get the current state as a dict."""
        ...

    @abstractmethod
    def undo(self) -> bool:
        """Undo to the previous snapshot. Returns True if successful."""
        ...

    @abstractmethod
    def redo(self) -> bool:
        """Redo to the next snapshot. Returns True if successful."""
        ...

    @abstractmethod
    def add_level(self) -> None:
        """Add an execution nesting level."""
        ...

    @abstractmethod
    def remove_level(self) -> None:
        """Remove an execution nesting level."""
        ...

    @abstractmethod
    def gc_collect(
        self, *, keep_first: int | None = None, keep_last: int | None = None
    ) -> int:
        """Manually collect old snapshots. Returns count of removed snapshots."""
        ...


# ──────────────────────────────────────────────
# Context Store Protocol
# ──────────────────────────────────────────────


class ContextStoreProtocol(ABC):
    """Unified interface for memory and artifact storage."""

    # ── Write ──

    @abstractmethod
    async def remember(self, entry: Memory) -> ID:
        """Store a new memory. Returns the assigned ID."""
        ...

    @abstractmethod
    async def store(self, entry: Artifact) -> ID:
        """Store a new artifact or a new version. Returns the assigned ID."""
        ...

    # ── Read ──

    @abstractmethod
    async def recall(self, query: MemoryQuery) -> list[QueryResult]:
        """Search across the store."""
        ...

    @abstractmethod
    async def get(self, id: ID) -> Memory | Artifact | None:
        """Direct fetch by stable ID."""
        ...

    @abstractmethod
    async def get_artifact(self, slug: str, version: int | None = None) -> Artifact | None:
        """Fetch artifact by slug — latest active version by default."""
        ...

    # ── Maintain ──

    @abstractmethod
    async def forget(self, id: ID) -> None:
        """Mark an entry as no longer relevant (soft delete)."""
        ...

    @abstractmethod
    async def consolidate(self) -> None:
        """Background maintenance pass."""
        ...


# ──────────────────────────────────────────────
# Policy Protocols
# ──────────────────────────────────────────────


class PolicyProtocol(ABC):
    """A single policy in the middleware chain."""

    @abstractmethod
    async def before_execute(self, event: Any, data: Any, configs: Any) -> None:
        """Hook before handler execution. May raise to abort."""
        ...

    @abstractmethod
    async def after_execute(
        self, event: Any, result: ExecutionResult[Any], configs: Any
    ) -> None:
        """Hook after handler execution."""
        ...

    @abstractmethod
    async def on_error(self, event: Any, error: Exception, configs: Any) -> None:
        """Hook when handler raises an exception."""
        ...


class PolicyChainProtocol(ABC):
    """Composable chain of policies wrapping handler execution."""

    @abstractmethod
    async def __call__(
        self,
        handler: EventHandlerProtocol[Any],
        event: Any,
        data: Any,
        configs: Any,
    ) -> ExecutionResult[Any]:
        """Run the handler wrapped by all policies in the chain."""
        ...


# ──────────────────────────────────────────────
# Budget Tracker Protocol
# ──────────────────────────────────────────────


class BudgetTrackerProtocol(ABC):
    """First-class standalone resource for budget management."""

    @abstractmethod
    def can_proceed(self) -> bool:
        """Check if there is remaining budget to continue."""
        ...

    @abstractmethod
    def consume(self, *, tokens: int = 0, calls: int = 0) -> None:
        """Record consumption of budget resources."""
        ...

    @abstractmethod
    def remaining(self) -> BudgetSnapshot:
        """Get a snapshot of remaining budget."""
        ...

    @abstractmethod
    def is_exceeded(self) -> bool:
        """Check if any budget dimension is exceeded."""
        ...


# ──────────────────────────────────────────────
# Observability Protocols
# ──────────────────────────────────────────────


class LogCollectorProtocol(ABC):
    """Structured log collection — EventBus subscriber."""

    @abstractmethod
    async def on_event(self, event: Any) -> None:
        """Handle an event for logging."""
        ...


class TraceCollectorProtocol(ABC):
    """Execution trace recording — opt-in EventBus subscriber."""

    @abstractmethod
    async def on_event(self, event: Any) -> None:
        """Handle an event for tracing."""
        ...
