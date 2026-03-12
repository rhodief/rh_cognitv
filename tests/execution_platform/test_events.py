"""Tests for events.py — ExecutionEvent, payloads, escalation events."""

from rh_cognitv.execution_platform.events import (
    DataPayload,
    EscalationRequested,
    EscalationResolved,
    ExecutionEvent,
    FunctionPayload,
    TextPayload,
    ToolPayload,
)
from rh_cognitv.execution_platform.models import EventKind, EventStatus


class TestTextPayload:
    def test_required_prompt(self):
        p = TextPayload(prompt="hello")
        assert p.prompt == "hello"

    def test_optional_fields_default_none(self):
        p = TextPayload(prompt="x")
        assert p.system_prompt is None
        assert p.model is None
        assert p.temperature is None
        assert p.max_tokens is None

    def test_ext_defaults_to_empty(self):
        p = TextPayload(prompt="x")
        assert p.ext == {}

    def test_all_fields(self):
        p = TextPayload(
            prompt="hi",
            system_prompt="sys",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            ext={"key": "val"},
        )
        assert p.model == "gpt-4"
        assert p.temperature == 0.7
        assert p.ext["key"] == "val"


class TestDataPayload:
    def test_required_prompt(self):
        p = DataPayload(prompt="extract data")
        assert p.prompt == "extract data"

    def test_output_schema_default_none(self):
        p = DataPayload(prompt="x")
        assert p.output_schema is None

    def test_with_output_schema(self):
        s = {"type": "object", "properties": {"name": {"type": "string"}}}
        p = DataPayload(prompt="x", output_schema=s)
        assert p.output_schema == s


class TestFunctionPayload:
    def test_required_function_name(self):
        p = FunctionPayload(function_name="my_func")
        assert p.function_name == "my_func"

    def test_args_default_empty(self):
        p = FunctionPayload(function_name="f")
        assert p.args == []
        assert p.kwargs == {}

    def test_with_args_kwargs(self):
        p = FunctionPayload(function_name="f", args=[1, 2], kwargs={"a": 3})
        assert p.args == [1, 2]
        assert p.kwargs["a"] == 3


class TestToolPayload:
    def test_required_prompt(self):
        p = ToolPayload(prompt="use tools")
        assert p.prompt == "use tools"

    def test_tools_default_empty(self):
        p = ToolPayload(prompt="x")
        assert p.tools == []

    def test_with_tools(self):
        tools = [{"name": "search", "description": "search the web"}]
        p = ToolPayload(prompt="x", tools=tools)
        assert len(p.tools) == 1
        assert p.tools[0]["name"] == "search"


class TestExecutionEvent:
    def test_minimal_creation(self):
        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="hello"),
        )
        assert event.kind == EventKind.TEXT
        assert event.status == EventStatus.CREATED
        assert isinstance(event.payload, TextPayload)

    def test_auto_id_generated(self):
        e1 = ExecutionEvent(kind=EventKind.TEXT, payload=TextPayload(prompt="a"))
        e2 = ExecutionEvent(kind=EventKind.TEXT, payload=TextPayload(prompt="b"))
        assert e1.id != e2.id
        assert len(e1.id) == 26

    def test_auto_created_at(self):
        event = ExecutionEvent(
            kind=EventKind.TEXT, payload=TextPayload(prompt="x")
        )
        assert event.created_at is not None
        assert "T" in event.created_at  # ISO-8601

    def test_parent_id_default_none(self):
        event = ExecutionEvent(
            kind=EventKind.DATA, payload=DataPayload(prompt="x")
        )
        assert event.parent_id is None

    def test_ext_default_empty(self):
        event = ExecutionEvent(
            kind=EventKind.FUNCTION,
            payload=FunctionPayload(function_name="f"),
        )
        assert event.ext == {}

    def test_all_event_kinds(self):
        payloads = {
            EventKind.TEXT: TextPayload(prompt="t"),
            EventKind.DATA: DataPayload(prompt="d"),
            EventKind.FUNCTION: FunctionPayload(function_name="f"),
            EventKind.TOOL: ToolPayload(prompt="u"),
        }
        for kind, payload in payloads.items():
            event = ExecutionEvent(kind=kind, payload=payload)
            assert event.kind == kind

    def test_status_can_be_overridden(self):
        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="x"),
            status=EventStatus.RUNNING,
        )
        assert event.status == EventStatus.RUNNING

    def test_serialization_roundtrip(self):
        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(prompt="hello", model="gpt-4"),
        )
        data = event.model_dump()
        restored = ExecutionEvent.model_validate(data)
        assert restored.kind == event.kind
        assert restored.id == event.id
        assert restored.payload.prompt == "hello"


class TestEscalationRequested:
    def test_creation(self):
        req = EscalationRequested(
            event_id="01ABC",
            question="Should I proceed?",
            options=["yes", "no"],
        )
        assert req.event_id == "01ABC"
        assert req.question == "Should I proceed?"
        assert req.options == ["yes", "no"]

    def test_defaults(self):
        req = EscalationRequested(event_id="x", question="q")
        assert req.options == []
        assert req.node_id is None
        assert req.resume_data == {}
        assert req.created_at is not None

    def test_with_resume_data(self):
        req = EscalationRequested(
            event_id="x",
            question="q",
            node_id="node-1",
            resume_data={"step": 3, "partial": "result"},
        )
        assert req.node_id == "node-1"
        assert req.resume_data["step"] == 3

    def test_serialization_roundtrip(self):
        req = EscalationRequested(
            event_id="ev1",
            question="Approve?",
            options=["approve", "reject"],
            node_id="n1",
        )
        data = req.model_dump()
        restored = EscalationRequested.model_validate(data)
        assert restored.event_id == "ev1"
        assert restored.options == ["approve", "reject"]


class TestEscalationResolved:
    def test_creation(self):
        res = EscalationResolved(event_id="01ABC", decision="approved")
        assert res.event_id == "01ABC"
        assert res.decision == "approved"

    def test_defaults(self):
        res = EscalationResolved(event_id="x", decision="d")
        assert res.resolved_at is not None
        assert res.ext == {}

    def test_serialization_roundtrip(self):
        res = EscalationResolved(
            event_id="ev1", decision="rejected", ext={"reason": "too expensive"}
        )
        data = res.model_dump()
        restored = EscalationResolved.model_validate(data)
        assert restored.decision == "rejected"
        assert restored.ext["reason"] == "too expensive"
