"""
EventBus — Hybrid sync middleware pipeline + async subscriber fan-out.

DD-L3-01: Sync middleware runs in order (deterministic, required for replay).
Async subscribers receive events in real-time (fire-and-forget).

OQ-L3-01: Type-based dispatch only (V1).
OQ-L3-05: wait_for() for escalation round-trip.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from .protocols import EventBusProtocol, MiddlewareProtocol


class EventBus(EventBusProtocol):
    """Concrete EventBus implementation.

    - Synchronous middleware pipeline: runs in registration order, deterministic.
    - Async subscribers: fire-and-forget fan-out after middleware completes.
    - Type-based routing: subscribers register for event classes (V1).
    - wait_for(): blocks until a matching event is emitted (for escalation).
    """

    def __init__(self) -> None:
        self._middlewares: list[MiddlewareProtocol] = []
        self._sync_handlers: dict[type, list[Callable]] = {}
        self._async_handlers: dict[type, list[Callable]] = {}
        self._waiters: list[_Waiter] = []

    # ── Registration ──────────────────────────

    def use(self, middleware: MiddlewareProtocol) -> None:
        """Register a synchronous middleware in the pipeline."""
        self._middlewares.append(middleware)

    def on(self, event_type: type, handler: Callable) -> None:
        """Register a synchronous handler for an event type."""
        self._sync_handlers.setdefault(event_type, []).append(handler)

    def on_async(self, event_type: type, handler: Callable) -> None:
        """Register an async subscriber for an event type."""
        self._async_handlers.setdefault(event_type, []).append(handler)

    # ── Emit ──────────────────────────────────

    async def emit(self, event: Any) -> None:
        """Emit an event through the middleware pipeline, then fan out.

        1. Run sync middleware chain (in order).
        2. Call sync handlers (in order).
        3. Fire async subscribers (concurrent, non-blocking).
        4. Notify any wait_for() waiters.
        """
        # 1. Run sync middleware pipeline
        self._run_middleware_chain(event, index=0)

        # 2. Call sync handlers matching this event type
        for event_type, handlers in self._sync_handlers.items():
            if isinstance(event, event_type):
                for handler in handlers:
                    handler(event)

        # 3. Fire async subscribers (non-blocking)
        tasks = []
        for event_type, handlers in self._async_handlers.items():
            if isinstance(event, event_type):
                for handler in handlers:
                    tasks.append(asyncio.ensure_future(handler(event)))

        if tasks:
            # Gather but don't let subscriber errors propagate to emit()
            await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Notify waiters
        self._notify_waiters(event)

    def _run_middleware_chain(self, event: Any, index: int) -> Any:
        """Run middleware at `index`, passing a next_fn that calls index+1."""
        if index >= len(self._middlewares):
            return event

        middleware = self._middlewares[index]

        def next_fn(evt: Any) -> Any:
            return self._run_middleware_chain(evt, index + 1)

        return middleware.handle(event, next_fn)

    # ── Wait For ──────────────────────────────

    async def wait_for(
        self,
        event_type: type,
        *,
        filter: Callable[[Any], bool] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Block until an event of the given type (matching optional filter) is emitted.

        Used for escalation round-trips (OQ-L3-05):
            decision = await bus.wait_for(EscalationResolved, filter=lambda e: e.event_id == id)

        Args:
            event_type: The event class to wait for.
            filter: Optional predicate — only matching events resolve the wait.
            timeout: Optional timeout in seconds. Raises asyncio.TimeoutError if exceeded.

        Returns:
            The matching event.
        """
        waiter = _Waiter(event_type=event_type, filter_fn=filter)
        self._waiters.append(waiter)

        try:
            if timeout is not None:
                return await asyncio.wait_for(waiter.future, timeout=timeout)
            return await waiter.future
        finally:
            # Clean up the waiter whether resolved or timed-out
            if waiter in self._waiters:
                self._waiters.remove(waiter)

    def _notify_waiters(self, event: Any) -> None:
        """Check all waiters and resolve any that match."""
        resolved = []
        for waiter in self._waiters:
            if waiter.future.done():
                resolved.append(waiter)
                continue
            if not isinstance(event, waiter.event_type):
                continue
            if waiter.filter_fn is not None and not waiter.filter_fn(event):
                continue
            waiter.future.set_result(event)
            resolved.append(waiter)

        for w in resolved:
            if w in self._waiters:
                self._waiters.remove(w)


class _Waiter:
    """Internal: represents a pending wait_for() call."""

    __slots__ = ("event_type", "filter_fn", "future")

    def __init__(
        self,
        event_type: type,
        filter_fn: Callable[[Any], bool] | None = None,
    ) -> None:
        self.event_type = event_type
        self.filter_fn = filter_fn
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
