"""
Tests for cognitive/prompt.py — Phase 3.4 test gate.

Covers:
- PromptBuilder: chainable API, system/context/user, build() -> BuiltPrompt
- TemplateRenderer: str.format variable substitution, system templates
- Edge cases: empty context, no system prompt, multiple .user() calls,
  missing variables, empty builder
- BuiltPrompt shape matches L3's TextPayload fields
"""

from __future__ import annotations

import pytest

from rh_cognitv.cognitive.models import BuiltPrompt
from rh_cognitv.cognitive.prompt import PromptBuilder, TemplateRenderer
from rh_cognitv.cognitive.protocols import PromptProtocol


# ──────────────────────────────────────────────
# Tests — PromptBuilder Protocol Compliance
# ──────────────────────────────────────────────


class TestPromptBuilderProtocol:
    """PromptBuilder satisfies PromptProtocol."""

    def test_is_prompt_protocol(self):
        builder = PromptBuilder()
        assert isinstance(builder, PromptProtocol)

    def test_has_build_method(self):
        builder = PromptBuilder()
        assert hasattr(builder, "build")
        assert callable(builder.build)

    def test_build_returns_built_prompt(self):
        result = PromptBuilder().build()
        assert isinstance(result, BuiltPrompt)


# ──────────────────────────────────────────────
# Tests — PromptBuilder System Prompt
# ──────────────────────────────────────────────


class TestPromptBuilderSystem:
    """PromptBuilder .system() handling."""

    def test_no_system_prompt_by_default(self):
        result = PromptBuilder().user("Hello").build()
        assert result.system_prompt is None

    def test_system_prompt_set(self):
        result = PromptBuilder().system("You are helpful.").user("Hi").build()
        assert result.system_prompt == "You are helpful."

    def test_system_overwrites_on_multiple_calls(self):
        result = (
            PromptBuilder()
            .system("First")
            .system("Second")
            .user("Hi")
            .build()
        )
        assert result.system_prompt == "Second"

    def test_system_only_no_user(self):
        result = PromptBuilder().system("System only").build()
        assert result.system_prompt == "System only"
        assert result.prompt == ""


# ──────────────────────────────────────────────
# Tests — PromptBuilder Context Injection
# ──────────────────────────────────────────────


class TestPromptBuilderContext:
    """PromptBuilder .context() places serialized context correctly."""

    def test_context_appears_in_prompt(self):
        result = (
            PromptBuilder()
            .context("Background info here")
            .user("Do something")
            .build()
        )
        assert "Background info here" in result.prompt

    def test_context_before_user(self):
        result = (
            PromptBuilder()
            .context("CONTEXT")
            .user("USER")
            .build()
        )
        ctx_pos = result.prompt.index("CONTEXT")
        usr_pos = result.prompt.index("USER")
        assert ctx_pos < usr_pos

    def test_context_separated_from_user(self):
        result = (
            PromptBuilder()
            .context("CTX")
            .user("USR")
            .build()
        )
        assert result.prompt == "CTX\n\nUSR"

    def test_context_overwrites_on_multiple_calls(self):
        result = (
            PromptBuilder()
            .context("First")
            .context("Second")
            .user("Go")
            .build()
        )
        assert "First" not in result.prompt
        assert "Second" in result.prompt

    def test_context_only_no_user(self):
        result = PromptBuilder().context("Just context").build()
        assert result.prompt == "Just context"

    def test_empty_context_omitted(self):
        result = (
            PromptBuilder()
            .context("")
            .user("Hello")
            .build()
        )
        assert result.prompt == "Hello"


# ──────────────────────────────────────────────
# Tests — PromptBuilder User Messages
# ──────────────────────────────────────────────


class TestPromptBuilderUser:
    """PromptBuilder .user() accumulation and joining."""

    def test_single_user_message(self):
        result = PromptBuilder().user("Hello world").build()
        assert result.prompt == "Hello world"

    def test_multiple_user_messages_joined(self):
        result = (
            PromptBuilder()
            .user("Line one")
            .user("Line two")
            .user("Line three")
            .build()
        )
        assert result.prompt == "Line one\nLine two\nLine three"

    def test_no_user_messages(self):
        result = PromptBuilder().build()
        assert result.prompt == ""

    def test_user_with_system(self):
        result = (
            PromptBuilder()
            .system("System")
            .user("User")
            .build()
        )
        assert result.system_prompt == "System"
        assert result.prompt == "User"


# ──────────────────────────────────────────────
# Tests — PromptBuilder Full Composition
# ──────────────────────────────────────────────


class TestPromptBuilderComposition:
    """Full builder composition: system + context + user."""

    def test_full_prompt(self):
        result = (
            PromptBuilder()
            .system("You are a summarizer.")
            .context("Memory: The user likes concise answers.")
            .user("Summarize this document.")
            .build()
        )
        assert result.system_prompt == "You are a summarizer."
        assert "Memory: The user likes concise answers." in result.prompt
        assert "Summarize this document." in result.prompt

    def test_order_independence_of_chaining(self):
        """Builder calls can be in any order — output structure is the same."""
        result = (
            PromptBuilder()
            .user("Do the thing")
            .system("Be helpful")
            .context("Background")
            .build()
        )
        assert result.system_prompt == "Be helpful"
        # Context still comes before user in prompt
        ctx_pos = result.prompt.index("Background")
        usr_pos = result.prompt.index("Do the thing")
        assert ctx_pos < usr_pos

    def test_chaining_returns_self(self):
        builder = PromptBuilder()
        assert builder.system("x") is builder
        assert builder.context("x") is builder
        assert builder.user("x") is builder

    def test_maps_to_text_payload_fields(self):
        """BuiltPrompt fields match L3's TextPayload(prompt, system_prompt)."""
        result = (
            PromptBuilder()
            .system("sys")
            .user("usr")
            .build()
        )
        assert hasattr(result, "prompt")
        assert hasattr(result, "system_prompt")


# ──────────────────────────────────────────────
# Tests — PromptBuilder Edge Cases
# ──────────────────────────────────────────────


class TestPromptBuilderEdgeCases:
    """Edge cases for PromptBuilder."""

    def test_empty_builder(self):
        result = PromptBuilder().build()
        assert result.prompt == ""
        assert result.system_prompt is None

    def test_empty_string_system(self):
        result = PromptBuilder().system("").user("Hi").build()
        assert result.system_prompt == ""

    def test_multiline_user(self):
        result = PromptBuilder().user("Line 1\nLine 2\nLine 3").build()
        assert result.prompt == "Line 1\nLine 2\nLine 3"

    def test_context_and_multiple_users(self):
        result = (
            PromptBuilder()
            .context("CTX")
            .user("U1")
            .user("U2")
            .build()
        )
        assert result.prompt == "CTX\n\nU1\nU2"


# ──────────────────────────────────────────────
# Tests — TemplateRenderer Basic
# ──────────────────────────────────────────────


class TestTemplateRendererBasic:
    """TemplateRenderer str.format variable substitution."""

    def test_returns_built_prompt(self):
        renderer = TemplateRenderer()
        result = renderer.render("Hello")
        assert isinstance(result, BuiltPrompt)

    def test_no_variables(self):
        renderer = TemplateRenderer()
        result = renderer.render("Plain text prompt")
        assert result.prompt == "Plain text prompt"

    def test_single_variable(self):
        renderer = TemplateRenderer()
        result = renderer.render("Summarize: {text}", {"text": "Hello world"})
        assert result.prompt == "Summarize: Hello world"

    def test_multiple_variables(self):
        renderer = TemplateRenderer()
        result = renderer.render(
            "Task: {task}. Input: {input_text}",
            {"task": "summarize", "input_text": "data"},
        )
        assert result.prompt == "Task: summarize. Input: data"

    def test_context_variable(self):
        renderer = TemplateRenderer()
        result = renderer.render(
            "Context: {context}\nTask: {task}",
            {"context": "background info", "task": "analyze"},
        )
        assert result.prompt == "Context: background info\nTask: analyze"


# ──────────────────────────────────────────────
# Tests — TemplateRenderer System Prompt
# ──────────────────────────────────────────────


class TestTemplateRendererSystem:
    """TemplateRenderer system prompt handling."""

    def test_no_system_by_default(self):
        renderer = TemplateRenderer()
        result = renderer.render("Hello")
        assert result.system_prompt is None

    def test_static_system_prompt(self):
        renderer = TemplateRenderer()
        result = renderer.render("Hello", system_prompt="Be helpful")
        assert result.system_prompt == "Be helpful"

    def test_system_template(self):
        renderer = TemplateRenderer()
        result = renderer.render(
            "Do {task}",
            {"task": "summarize", "role": "assistant"},
            system_template="You are a {role}.",
        )
        assert result.system_prompt == "You are a assistant."

    def test_system_template_overrides_system_prompt(self):
        renderer = TemplateRenderer()
        result = renderer.render(
            "Hello",
            {"role": "helper"},
            system_prompt="Static system",
            system_template="Dynamic: {role}",
        )
        assert result.system_prompt == "Dynamic: helper"


# ──────────────────────────────────────────────
# Tests — TemplateRenderer Missing Variables
# ──────────────────────────────────────────────


class TestTemplateRendererMissing:
    """Missing variables are left as-is (safe format)."""

    def test_missing_variable_preserved(self):
        renderer = TemplateRenderer()
        result = renderer.render("Hello {name}, your task is {task}", {"name": "Alice"})
        assert result.prompt == "Hello Alice, your task is {task}"

    def test_all_missing(self):
        renderer = TemplateRenderer()
        result = renderer.render("{a} and {b}")
        assert result.prompt == "{a} and {b}"

    def test_empty_variables_dict(self):
        renderer = TemplateRenderer()
        result = renderer.render("No vars: {x}", {})
        assert result.prompt == "No vars: {x}"

    def test_none_variables(self):
        renderer = TemplateRenderer()
        result = renderer.render("Template: {x}")
        assert result.prompt == "Template: {x}"

    def test_missing_in_system_template(self):
        renderer = TemplateRenderer()
        result = renderer.render(
            "Hello",
            {},
            system_template="You are {role}",
        )
        assert result.system_prompt == "You are {role}"


# ──────────────────────────────────────────────
# Tests — TemplateRenderer Edge Cases
# ──────────────────────────────────────────────


class TestTemplateRendererEdgeCases:
    """Edge cases for TemplateRenderer."""

    def test_empty_template(self):
        renderer = TemplateRenderer()
        result = renderer.render("")
        assert result.prompt == ""

    def test_template_with_braces(self):
        """Double braces are literal in str.format."""
        renderer = TemplateRenderer()
        result = renderer.render("JSON: {{\"key\": \"{value}\"}}", {"value": "hello"})
        assert result.prompt == 'JSON: {"key": "hello"}'

    def test_multiline_template(self):
        renderer = TemplateRenderer()
        template = """You are given:
{context}

Please {task} the above."""
        result = renderer.render(template, {"context": "Some data", "task": "summarize"})
        assert "Some data" in result.prompt
        assert "summarize" in result.prompt

    def test_repeated_variable(self):
        renderer = TemplateRenderer()
        result = renderer.render("{x} and again {x}", {"x": "hello"})
        assert result.prompt == "hello and again hello"

    def test_variable_with_special_chars(self):
        renderer = TemplateRenderer()
        result = renderer.render("Data: {text}", {"text": "line1\nline2\ttab"})
        assert result.prompt == "Data: line1\nline2\ttab"


# ──────────────────────────────────────────────
# Tests — Integration: Builder + Renderer
# ──────────────────────────────────────────────


class TestBuilderRendererIntegration:
    """TemplateRenderer output can feed into PromptBuilder.context()."""

    def test_renderer_output_into_builder_context(self):
        renderer = TemplateRenderer()
        rendered = renderer.render(
            "Memory: {memory}",
            {"memory": "User prefers short answers"},
        )

        result = (
            PromptBuilder()
            .system("You are helpful.")
            .context(rendered.prompt)
            .user("Summarize this.")
            .build()
        )

        assert result.system_prompt == "You are helpful."
        assert "User prefers short answers" in result.prompt
        assert "Summarize this." in result.prompt

    def test_both_produce_built_prompt(self):
        builder_result = PromptBuilder().user("test").build()
        renderer_result = TemplateRenderer().render("test")
        assert type(builder_result) is type(renderer_result)
        assert isinstance(builder_result, BuiltPrompt)
