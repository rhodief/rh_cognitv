"""Tests for event_bus.py — EventBus middleware, subscribers, emit, wait_for."""

import asyncio

import pytest

from rh_cognitv.execution_platform.event_bus import EventBus
from rh_cognitv.execution_platform.protocols import MiddlewareProtocol


# ──────────────────────────────────────────────
# Test event classes
# ──────────────────────────────────────────────


class SampleEvent:
    def __init__(self, value: str = ""):
        self.value = value


class OtherEvent:
    def __init__(self, data: int = 0):
        self.data = data


class ChildEvent(SampleEvent):
    pass


# ──────────────────────────────────────────────
# Test middleware
# ──────────────────────────────────────────────


class RecordingMiddleware(MiddlewareProtocol):
    """Middleware that records events passing through and calls next."""

    def __init__(self, name: str):
        self.name = name
        self.events: list = []

    def handle(self, event, next_fn):
        self.events.append((self.name, event))
        return next_fn(event)


class MutatingMiddleware(MiddlewareProtocol):
    """Middleware that mutates the event before passing it along."""

    def __init__(self, suffix: str):
        self.suffix = suffix

    def handle(self, event, next_fn):
        if hasattr(event, "value"):
            event.value += self.suffix
        return next_fn(event)


class BlockingMiddleware(MiddlewareProtocol):
    """Middleware that does NOT call next_fn — stops the chain."""

    def __init__(self):
        self.events: list = []

    def handle(self, event, next_fn):
        self.events.append(event)
        # Intentionally not calling next_fn


# ──────────────────────────────────────────────
# Middleware ordering
# ──────────────────────────────────────────────


class TestMiddlewareOrdering:
    @pytest.mark.asyncio
    async def test_middleware_runs_in_registration_order(self):
        bus = EventBus()
        m1 = RecordingMiddleware("first")
        m2 = RecordingMiddleware("second")
        m3 = RecordingMiddleware("third")

        bus.use(m1)
        bus.use(m2)
        bus.use(m3)

        event = SampleEvent("test")
        await bus.emit(event)

        assert len(m1.events) == 1
        assert len(m2.events) == 1
        assert len(m3.events) == 1
        assert m1.events[0][0] == "first"
        assert m2.events[0][0] == "second"
        assert m3.events[0][0] == "third"

    @pytest.mark.asyncio
    async def test_middleware_can_mutate_event(self):
        bus = EventBus()
        m1 = MutatingMiddleware("_A")
        m2 = MutatingMiddleware("_B")
        m3 = RecordingMiddleware("check")

        bus.use(m1)
        bus.use(m2)
        bus.use(m3)

        event = SampleEvent("start")
        await bus.emit(event)

        # m3 should see the mutated value
        assert m3.events[0][1].value == "start_A_B"

    @pytest.mark.asyncio
    async def test_middleware_can_stop_chain(self):
        bus = EventBus()
        m1 = RecordingMiddleware("before")
        blocker = BlockingMiddleware()
        m3 = RecordingMiddleware("after")

        bus.use(m1)
        bus.use(blocker)
        bus.use(m3)

        await bus.emit(SampleEvent("test"))

        assert len(m1.events) == 1
        assert len(blocker.events) == 1
        assert len(m3.events) == 0  # never reached

    @pytest.mark.asyncio
    async def test_no_middleware(self):
        """emit() works even with no middleware registered."""
        bus = EventBus()
        await bus.emit(SampleEvent("test"))  # Should not raise


# ──────────────────────────────────────────────
# Sync handlers
# ──────────────────────────────────────────────


class TestSyncHandlers:
    @pytest.mark.asyncio
    async def test_sync_handler_receives_event(self):
        bus = EventBus()
        received = []
        bus.on(SampleEvent, lambda e: received.append(e))

        event = SampleEvent("hello")
        await bus.emit(event)

        assert len(received) == 1
        assert received[0] is event

    @pytest.mark.asyncio
    async def test_multiple_sync_handlers(self):
        bus = EventBus()
        r1, r2 = [], []
        bus.on(SampleEvent, lambda e: r1.append(e.value))
        bus.on(SampleEvent, lambda e: r2.append(e.value))

        await bus.emit(SampleEvent("hi"))

        assert r1 == ["hi"]
        assert r2 == ["hi"]

    @pytest.mark.asyncio
    async def test_sync_handler_type_matching(self):
        bus = EventBus()
        sample_received, other_received = [], []
        bus.on(SampleEvent, lambda e: sample_received.append(e))
        bus.on(OtherEvent, lambda e: other_received.append(e))

        await bus.emit(SampleEvent("s"))
        await bus.emit(OtherEvent(42))

        assert len(sample_received) == 1
        assert len(other_received) == 1

    @pytest.mark.asyncio
    async def test_sync_handler_subclass_matching(self):
        """Handlers for a parent class also receive child events (isinstance)."""
        bus = EventBus()
        received = []
        bus.on(SampleEvent, lambda e: received.append(e))

        await bus.emit(ChildEvent("child"))

        assert len(received) == 1
        assert isinstance(received[0], ChildEvent)


# ──────────────────────────────────────────────
# Async subscribers
# ──────────────────────────────────────────────


class TestAsyncSubscribers:
    @pytest.mark.asyncio
    async def test_async_subscriber_receives_event(self):
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.on_async(SampleEvent, handler)
        await bus.emit(SampleEvent("async"))

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_multiple_async_subscribers(self):
        bus = EventBus()
        r1, r2 = [], []

        async def h1(e):
            r1.append(e.value)

        async def h2(e):
            r2.append(e.value)

        bus.on_async(SampleEvent, h1)
        bus.on_async(SampleEvent, h2)

        await bus.emit(SampleEvent("multi"))

        assert r1 == ["multi"]
        assert r2 == ["multi"]

    @pytest.mark.asyncio
    async def test_async_subscriber_error_does_not_crash_emit(self):
        """Async subscriber errors should not propagate to the emitter."""
        bus = EventBus()
        received = []

        async def bad_handler(event):
            raise RuntimeError("subscriber crash")

        async def good_handler(event):
            received.append(event)

        bus.on_async(SampleEvent, bad_handler)
        bus.on_async(SampleEvent, good_handler)

        # Should not raise despite bad_handler crashing
        await bus.emit(SampleEvent("safe"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_async_subscriber_type_matching(self):
        bus = EventBus()
        sample_r, other_r = [], []

        async def sh(e):
            sample_r.append(e)

        async def oh(e):
            other_r.append(e)

        bus.on_async(SampleEvent, sh)
        bus.on_async(OtherEvent, oh)

        await bus.emit(SampleEvent("s"))
        await bus.emit(OtherEvent(1))

        assert len(sample_r) == 1
        assert len(other_r) == 1

    @pytest.mark.asyncio
    async def test_async_subscriber_subclass_matching(self):
        bus = EventBus()
        received = []

        async def handler(e):
            received.append(e)

        bus.on_async(SampleEvent, handler)
        await bus.emit(ChildEvent("child"))

        assert len(received) == 1


# ──────────────────────────────────────────────
# Middleware + handlers ordering
# ──────────────────────────────────────────────


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_middleware_runs_before_handlers(self):
        bus = EventBus()
        order = []

        class OrderMiddleware(MiddlewareProtocol):
            def handle(self, event, next_fn):
                order.append("middleware")
                return next_fn(event)

        bus.use(OrderMiddleware())
        bus.on(SampleEvent, lambda e: order.append("sync"))

        async def ah(e):
            order.append("async")

        bus.on_async(SampleEvent, ah)

        await bus.emit(SampleEvent("order"))

        assert order == ["middleware", "sync", "async"]

    @pytest.mark.asyncio
    async def test_many_events(self):
        bus = EventBus()
        count = []

        async def handler(e):
            count.append(1)

        bus.on_async(SampleEvent, handler)

        for i in range(50):
            await bus.emit(SampleEvent(str(i)))

        assert len(count) == 50


# ──────────────────────────────────────────────
# wait_for()
# ──────────────────────────────────────────────


class TestWaitFor:
    @pytest.mark.asyncio
    async def test_wait_for_resolves_on_matching_event(self):
        bus = EventBus()

        async def emit_later():
            await asyncio.sleep(0.01)
            await bus.emit(SampleEvent("resolved"))

        asyncio.ensure_future(emit_later())
        result = await bus.wait_for(SampleEvent, timeout=2.0)

        assert isinstance(result, SampleEvent)
        assert result.value == "resolved"

    @pytest.mark.asyncio
    async def test_wait_for_with_filter(self):
        bus = EventBus()

        async def emit_events():
            await asyncio.sleep(0.01)
            await bus.emit(SampleEvent("wrong"))
            await asyncio.sleep(0.01)
            await bus.emit(SampleEvent("right"))

        asyncio.ensure_future(emit_events())
        result = await bus.wait_for(
            SampleEvent,
            filter=lambda e: e.value == "right",
            timeout=2.0,
        )

        assert result.value == "right"

    @pytest.mark.asyncio
    async def test_wait_for_timeout(self):
        bus = EventBus()

        with pytest.raises(asyncio.TimeoutError):
            await bus.wait_for(SampleEvent, timeout=0.05)

    @pytest.mark.asyncio
    async def test_wait_for_ignores_wrong_type(self):
        bus = EventBus()

        async def emit_wrong_then_right():
            await asyncio.sleep(0.01)
            await bus.emit(OtherEvent(99))  # wrong type
            await asyncio.sleep(0.01)
            await bus.emit(SampleEvent("correct"))

        asyncio.ensure_future(emit_wrong_then_right())
        result = await bus.wait_for(SampleEvent, timeout=2.0)

        assert isinstance(result, SampleEvent)
        assert result.value == "correct"

    @pytest.mark.asyncio
    async def test_multiple_waiters(self):
        bus = EventBus()

        async def emit_two():
            await asyncio.sleep(0.01)
            await bus.emit(SampleEvent("first"))
            await asyncio.sleep(0.01)
            await bus.emit(OtherEvent(42))

        asyncio.ensure_future(emit_two())

        result1, result2 = await asyncio.gather(
            bus.wait_for(SampleEvent, timeout=2.0),
            bus.wait_for(OtherEvent, timeout=2.0),
        )

        assert result1.value == "first"
        assert result2.data == 42

    @pytest.mark.asyncio
    async def test_wait_for_cleans_up_on_timeout(self):
        bus = EventBus()

        with pytest.raises(asyncio.TimeoutError):
            await bus.wait_for(SampleEvent, timeout=0.02)

        # Waiter should be cleaned up
        assert len(bus._waiters) == 0

    @pytest.mark.asyncio
    async def test_wait_for_cleans_up_on_resolve(self):
        bus = EventBus()

        async def emit_later():
            await asyncio.sleep(0.01)
            await bus.emit(SampleEvent("done"))

        asyncio.ensure_future(emit_later())
        await bus.wait_for(SampleEvent, timeout=2.0)

        assert len(bus._waiters) == 0
